"""Supervised out-of-process inference worker (``--worker-process``).

Runs inference in a child Python process speaking newline-delimited JSON
(:mod:`remi.worker.protocol`) over stdin/stdout pipes. If the worker crashes
(CUDA illegal memory access, MPS hang, OOM kill, segfault), the supervisor
captures the failure cleanly, returns a structured :class:`WorkerResult`, and
automatically respawns a fresh worker process so the FastAPI control plane,
React SPA, and SQLite queue survive with zero data loss.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from remi.worker.protocol import (
    JobEnvelope,
    PreviewCallback,
    PreviewChunk,
    ProgressCallback,
    WorkerCapabilities,
    WorkerHeartbeat,
    WorkerMessage,
    WorkerProgress,
    WorkerResult,
    WorkerState,
    decode_message,
    encode_message,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_RESTARTS = 5
DEFAULT_STARTUP_TIMEOUT = 15.0
DEFAULT_JOB_TIMEOUT = 3600.0
STDERR_TAIL_BYTES = 64 * 1024
"""how much of a worker's stderr is kept for the crash message"""


class _Pipes:
    """A worker process's output, read by two threads from the start: its
    stdout lines into :attr:`lines` (``None`` at the end), the end of its
    stderr into a bounded buffer. A pipe nobody reads fills up and blocks
    the child (a chatty library at startup never answered the ping), and a
    queue lets the supervisor wait for a line with a timeout."""

    def __init__(self, proc: Any) -> None:
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        self._stderr: deque[bytes] = deque()
        self._stderr_size = 0
        self._threads = [
            threading.Thread(target=self._read_lines, args=(proc.stdout,), name=f"worker-stdout-{proc.pid}", daemon=True),
            threading.Thread(target=self._read_stderr, args=(proc.stderr,), name=f"worker-stderr-{proc.pid}", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _read_lines(self, stream: Any) -> None:
        try:
            if stream is not None:
                for line in iter(stream.readline, b""):
                    self.lines.put(line)
        except (OSError, ValueError):  # closed
            pass
        finally:
            self.lines.put(None)

    def _read_stderr(self, stream: Any) -> None:
        with contextlib.suppress(OSError, ValueError):
            while stream is not None and (chunk := stream.read1(8192)):
                self._stderr.append(chunk)
                self._stderr_size += len(chunk)
                while self._stderr_size - len(self._stderr[0]) >= STDERR_TAIL_BYTES:
                    self._stderr_size -= len(self._stderr.popleft())

    def stderr_tail(self, limit: int) -> str:
        """The last ``limit`` characters the process wrote to stderr (after
        waiting a moment for what a process that just ended wrote last)."""

        self._threads[1].join(timeout=1.0)
        return b"".join(tuple(self._stderr)).decode("utf-8", errors="replace")[-limit:]

    def join(self, timeout: float) -> bool:
        """Wait for both readers to reach the end; whether they did."""

        for thread in self._threads:
            thread.join(timeout)
        return not any(thread.is_alive() for thread in self._threads)


_PIPES: weakref.WeakKeyDictionary[Any, _Pipes] = weakref.WeakKeyDictionary()
_PIPES_LOCK = threading.Lock()


def _pipes(proc: Any) -> _Pipes:
    """The output readers of ``proc``, started on first use."""

    with _PIPES_LOCK:
        pipes = _PIPES.get(proc)
        if pipes is None:
            pipes = _PIPES[proc] = _Pipes(proc)
        return pipes


class SubprocessWorker:
    """Supervised out-of-process worker implementing :class:`WorkerProtocol`."""

    def __init__(
        self,
        worker_id: str = "worker-0",
        *,
        gpu_id: str | None = None,
        command: Sequence[str] | None = None,
        handler_import: str = "remi.worker.subprocess_worker:default_forge_handler",
        env: dict[str, str] | None = None,
        auto_restart: bool = True,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        restart_backoff_s: float = 0.25,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    ) -> None:
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.handler_import = handler_import
        self._custom_command = list(command) if command is not None else None
        self._extra_env = dict(env or {})
        self.auto_restart = auto_restart
        self.max_restarts = max(0, int(max_restarts))
        self.restart_backoff_s = max(0.0, float(restart_backoff_s))
        self.startup_timeout = float(startup_timeout)

        self._lock = threading.RLock()
        self._exec_lock = threading.RLock()
        """one job at a time; :meth:`restart` takes it to start the new process (re-entered by the crash path)"""
        self._proc: subprocess.Popen[bytes] | None = None
        self._state: WorkerState = "stopped"
        self._started_at: float = 0.0
        self._last_heartbeat: float | None = None
        self._current_job_id: str | None = None
        self._jobs_completed: int = 0
        self._jobs_failed: int = 0
        self._restarts: int = 0
        self._capabilities = WorkerCapabilities(
            backend="forge-pytorch",
            device=f"cuda:{gpu_id}" if gpu_id is not None else "auto",
        )

    def _build_command(self) -> list[str]:
        if self._custom_command is not None:
            return list(self._custom_command)
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "remi.worker.subprocess_worker",
            "--worker-id",
            self.worker_id,
            "--handler",
            self.handler_import,
        ]
        if self.gpu_id is not None:
            cmd.extend(["--gpu", str(self.gpu_id)])
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env.update(self._extra_env)
        return env

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._state = "starting"
            proc = subprocess.Popen(
                self._build_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._build_env(),
            )
            self._proc = proc
            self._started_at = time.time()
            _pipes(proc)  # read from the start: a child writing to an unread pipe blocks
        # Wait for initial capabilities/heartbeat handshake
        try:
            self._handshake(proc)
        except BaseException:
            self._terminate_proc(proc, timeout=0.2)  # it did not answer: no grace period
            with self._lock:
                if self._proc is proc:
                    self._proc = None
                self._state = "error"
            raise

    def _handshake(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(encode_message(WorkerMessage("ping", {"worker_id": self.worker_id})))
        proc.stdin.flush()
        lines = _pipes(proc).lines
        deadline = time.monotonic() + self.startup_timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise queue.Empty
                line = lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"worker {self.worker_id} did not answer startup ping within {self.startup_timeout}s") from None
            if line is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=1.0)
                rc = proc.poll()
                stderr_text = _pipes(proc).stderr_tail(500)
                raise RuntimeError(f"worker {self.worker_id} exited during startup (code={rc}): {stderr_text}")
            msg = decode_message(line)
            if msg.type == "capabilities":
                with self._lock:
                    self._capabilities = WorkerCapabilities.from_dict(msg.payload)
                    self._last_heartbeat = time.time()
            elif msg.type in ("pong", "heartbeat"):
                caps_data = msg.payload.get("capabilities")
                with self._lock:
                    if isinstance(caps_data, dict):
                        self._capabilities = WorkerCapabilities.from_dict(caps_data)
                    self._state = "idle"
                    self._last_heartbeat = time.time()
                return

    @staticmethod
    def _terminate_proc(proc: subprocess.Popen[bytes] | None, timeout: float = 3.0) -> None:
        if proc is None:
            return
        with contextlib.suppress(Exception):
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write(encode_message(WorkerMessage("shutdown")))
                proc.stdin.flush()
                proc.stdin.close()
        if proc.poll() is None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=2.0)
        with _PIPES_LOCK:
            pipes = _PIPES.pop(proc, None)
        # the readers reach the end once the process is gone; a pipe is not closed under a reader still
        # blocked on it (a grandchild holding it open), which would block the close as well
        readers_done = pipes is None or pipes.join(timeout=2.0)
        for pipe in (proc.stdin, proc.stdout, proc.stderr) if readers_done else (proc.stdin,):
            if pipe is not None and not pipe.closed:
                with contextlib.suppress(Exception):
                    pipe.close()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
            self._state = "stopped"
            self._current_job_id = None
        self._terminate_proc(proc, timeout=timeout)

    def restart(self) -> WorkerHeartbeat:
        """Replace the process. A job running in the old one ends as a
        failure (its process is stopped first, so a hung job cannot hold up
        the restart); the new process starts once that job's crash handling
        is done, so the two never race over which process is the worker's."""

        with self._lock:
            self._state = "restarting"
            proc = self._proc
            self._proc = None
            self._current_job_id = None
            self._restarts += 1
        self._terminate_proc(proc, timeout=2.0)
        with self._exec_lock:
            if self.restart_backoff_s > 0:
                time.sleep(self.restart_backoff_s)
            self.start()
        return self.status()

    def is_alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def capabilities(self) -> WorkerCapabilities:
        with self._lock:
            return self._capabilities

    def status(self) -> WorkerHeartbeat:
        with self._lock:
            proc = self._proc
            alive = proc is not None and proc.poll() is None
            state: WorkerState = self._state
            if proc is not None and not alive and state not in ("stopped", "restarting"):
                state = "error"
            uptime = (time.time() - self._started_at) if (alive and self._started_at > 0) else 0.0
            return WorkerHeartbeat(
                worker_id=self.worker_id,
                pid=proc.pid if alive and proc is not None else None,
                state=state,
                uptime_s=uptime,
                current_job_id=self._current_job_id,
                jobs_completed=self._jobs_completed,
                jobs_failed=self._jobs_failed,
                restarts=self._restarts,
                last_heartbeat=self._last_heartbeat,
                capabilities=self._capabilities,
            )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if self._current_job_id != job_id or self._proc is None or self._proc.poll() is not None:
                return False
            stdin = self._proc.stdin
        if stdin is None or stdin.closed:
            return False
        try:
            stdin.write(encode_message(WorkerMessage("cancel", {"job_id": job_id})))
            stdin.flush()
            return True
        except OSError:
            return False

    def execute(
        self,
        envelope: JobEnvelope,
        *,
        on_progress: ProgressCallback | None = None,
        on_preview: PreviewCallback | None = None,
    ) -> WorkerResult:
        with self._exec_lock:
            if not self.is_alive():
                self.start()
            with self._lock:
                proc = self._proc
                if proc is None or proc.stdin is None:  # stopped meanwhile (a restart, stop())
                    return self._handle_crash(envelope.job_id, None, proc, detail="the worker process was stopped")
                self._state = "busy"
                self._current_job_id = envelope.job_id
            lines = _pipes(proc).lines
            try:
                proc.stdin.write(encode_message(WorkerMessage("run_job", envelope.to_dict())))
                proc.stdin.flush()
                while True:
                    line = lines.get()
                    if line is None:
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            proc.wait(timeout=1.0)
                        return self._handle_crash(envelope.job_id, proc.poll(), proc)
                    msg = decode_message(line)
                    now = time.time()
                    with self._lock:
                        self._last_heartbeat = now
                    if msg.type == "progress":
                        if on_progress is not None:
                            on_progress(WorkerProgress.from_dict(msg.payload))
                    elif msg.type == "preview":
                        if on_preview is not None:
                            on_preview(PreviewChunk.from_dict(msg.payload))
                    elif msg.type == "capabilities":
                        with self._lock:
                            self._capabilities = WorkerCapabilities.from_dict(msg.payload)
                    elif msg.type == "result":
                        result = WorkerResult.from_dict(msg.payload)
                        with self._lock:
                            self._current_job_id = None
                            self._state = "idle"
                            if result.status == "completed":
                                self._jobs_completed += 1
                            elif result.status == "failed":
                                self._jobs_failed += 1
                        return result
                    elif msg.type == "error":
                        detail = str(msg.payload.get("error") or "worker error")
                        with self._lock:
                            self._current_job_id = None
                            self._state = "idle"
                            self._jobs_failed += 1
                        return WorkerResult(job_id=envelope.job_id, status="failed", error=detail)
            except OSError as error:
                rc = proc.poll() if proc is not None else -1
                return self._handle_crash(envelope.job_id, rc, proc, detail=str(error))

    def _handle_crash(self, job_id: str, returncode: int | None, proc: subprocess.Popen[bytes] | None, *, detail: str | None = None) -> WorkerResult:
        stderr_hint = ""
        if proc is not None:
            stderr_hint = _pipes(proc).stderr_tail(2000).strip()[-300:]
        self._terminate_proc(proc, timeout=1.0)
        with self._lock:
            if self._proc is proc:
                self._proc = None
            self._current_job_id = None
            self._jobs_failed += 1
            self._state = "error"
            can_restart = self.auto_restart and self._restarts < self.max_restarts

        restarted = False
        if can_restart:
            try:
                self.restart()
                restarted = True
            except Exception as err:
                log.warning("worker %s auto-restart failed: %s", self.worker_id, err)

        reason = detail or f"exit code {returncode}"
        if stderr_hint:
            reason = f"{reason} ({stderr_hint})"
        suffix = "; worker automatically restarted" if restarted else ""
        return WorkerResult(
            job_id=job_id,
            status="failed",
            error=f"Worker process crashed: {reason}{suffix}",
        )


# region Child-process stdio loop


JobHandler = Callable[[JobEnvelope, Callable[[WorkerProgress], None], Callable[[PreviewChunk], None], threading.Event], WorkerResult]


def default_forge_handler(
    envelope: JobEnvelope,
    emit_progress: Callable[[WorkerProgress], None],
    emit_preview: Callable[[PreviewChunk], None],
    cancel_event: threading.Event,
) -> WorkerResult:
    """Default production handler executing a :class:`JobEnvelope` via ``remi.jobs.runner``."""

    from remi.core import JobContext
    from remi.files import FileStore
    from remi.jobs.runner import run_job
    from remi.jobs.store import Job

    files = FileStore(envelope.file_store_root)
    try:
        job = Job(id=envelope.job_id, seq=envelope.seq, tab=envelope.tab, core=envelope.core, scripts=envelope.scripts, meta=envelope.meta)
        ctx = JobContext(id_task=job.id_task, resolve_upload=files.resolve_upload)
        result = run_job(job, ctx)
        outputs: list[dict[str, Any]] = []
        file_ids: list[str] = []
        for idx, img in enumerate(result.images):
            infotext = result.infotexts[idx] if idx < len(result.infotexts) else result.info
            saved = getattr(img, "already_saved_as", None)
            if saved and os.path.isfile(saved):
                rec = files.register_output(saved, job_id=job.id, tab=job.tab, infotext=infotext, width=img.width, height=img.height)
            else:
                rec = files.save_output_image(img, infotext, job_id=job.id, tab=job.tab)
            file_ids.append(rec.id)
            outputs.append({"id": rec.id, "kind": "image", "width": rec.width, "height": rec.height, "infotext": infotext})
        status = "cancelled" if cancel_event.is_set() else "completed"
        return WorkerResult(job_id=envelope.job_id, status=status, output_file_ids=file_ids, outputs=outputs, infotext=result.info, warnings=result.warnings)
    finally:
        files.close()


def _resolve_handler(dotted: str) -> JobHandler:
    module_name, _, attr = dotted.rpartition(":")
    if not module_name or not attr:
        raise ValueError(f"handler must be 'module:callable' or 'path.py:callable', got {dotted!r}")
    if module_name.endswith(".py") and os.path.isfile(module_name):
        spec = importlib.util.spec_from_file_location("_remi_worker_handler", module_name)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load handler file {module_name!r}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"handler {dotted!r} is not callable")
    return fn


def run_worker_stdio_loop(
    handler: JobHandler,
    *,
    worker_id: str = "worker-0",
    capabilities: WorkerCapabilities | None = None,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    """Main JSON-lines event loop inside the worker child process."""

    in_stream = stdin or sys.stdin.buffer
    out_stream = stdout or sys.stdout.buffer
    out_lock = threading.Lock()
    caps = capabilities or WorkerCapabilities(backend="forge-pytorch", device=os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"))
    cancel_events: dict[str, threading.Event] = {}

    def send(msg: WorkerMessage) -> None:
        raw = encode_message(msg)
        with out_lock:
            out_stream.write(raw)
            out_stream.flush()

    while True:
        line = in_stream.readline()
        if not line:
            return 0
        try:
            msg = decode_message(line)
        except Exception as error:
            send(WorkerMessage("error", {"error": f"protocol decode error: {error}"}))
            continue

        if msg.type == "shutdown":
            return 0
        if msg.type == "ping":
            send(WorkerMessage("pong", {"worker_id": worker_id, "pid": os.getpid(), "capabilities": caps.to_dict()}))
        elif msg.type == "heartbeat":
            send(WorkerMessage("heartbeat", {"worker_id": worker_id, "pid": os.getpid(), "capabilities": caps.to_dict()}))
        elif msg.type == "cancel":
            jid = str(msg.payload.get("job_id") or "")
            ev = cancel_events.get(jid)
            if ev is not None:
                ev.set()
        elif msg.type == "run_job":
            envelope = JobEnvelope.from_dict(msg.payload)
            cancel_ev = threading.Event()
            cancel_events[envelope.job_id] = cancel_ev
            try:
                result = handler(
                    envelope,
                    lambda prog: send(WorkerMessage("progress", prog.to_dict())),
                    lambda chunk: send(WorkerMessage("preview", chunk.to_dict())),
                    cancel_ev,
                )
                send(WorkerMessage("result", result.to_dict()))
            except Exception as error:
                send(WorkerMessage("result", WorkerResult(job_id=envelope.job_id, status="failed", error=f"{type(error).__name__}: {error}").to_dict()))
            finally:
                cancel_events.pop(envelope.job_id, None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remi out-of-process inference worker")
    parser.add_argument("--worker-id", default="worker-0")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--handler", default="remi.worker.subprocess_worker:default_forge_handler")
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = _resolve_handler(args.handler)
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    caps = WorkerCapabilities(backend="forge-pytorch", device=device)
    return run_worker_stdio_loop(handler, worker_id=args.worker_id, capabilities=caps)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_RESTARTS",
    "JobHandler",
    "SubprocessWorker",
    "default_forge_handler",
    "run_worker_stdio_loop",
]
