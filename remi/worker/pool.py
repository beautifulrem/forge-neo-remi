"""Local Multi-GPU worker pool scheduler (``--worker-gpus 0,1``) and default
in-process worker adapter.

Dispatches independent queued jobs across GPU workers using model-checkpoint
affinity (preferring an idle worker that already holds the requested checkpoint)
and load balancing (fewest completed jobs, highest free VRAM).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from remi.worker.bridge import RuntimeBridge
from remi.worker.mlx_sidecar import MlxSidecarWorker, detect_mlx_environment
from remi.worker.protocol import (
    JobEnvelope,
    PreviewCallback,
    ProgressCallback,
    WorkerCapabilities,
    WorkerHeartbeat,
    WorkerProtocol,
    WorkerResult,
    WorkerState,
)
from remi.worker.subprocess_worker import SubprocessWorker


def parse_worker_gpus(spec: str | Sequence[str | int] | None) -> list[str]:
    """Parse ``"0,1"`` or ``[0, 1]`` into deduplicated GPU id strings."""

    if spec is None:
        return []
    if isinstance(spec, str):
        raw = [part.strip() for part in spec.split(",") if part.strip()]
    else:
        raw = [str(item).strip() for item in spec if str(item).strip()]
    return list(dict.fromkeys(raw))


class InProcessWorker:
    """Adapter exposing the default in-process Forge worker through :class:`WorkerProtocol`."""

    def __init__(
        self,
        worker_id: str = "worker-inprocess",
        *,
        bridge: RuntimeBridge | None = None,
        executor: Callable[[JobEnvelope], WorkerResult] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.bridge = bridge or RuntimeBridge()
        self._executor = executor
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._state: WorkerState = "idle"
        self._current_job_id: str | None = None
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._restarts = 0

    def start(self) -> None:
        with self._lock:
            self._state = "idle"
            self._started_at = time.time()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._state = "stopped"
            self._current_job_id = None

    def restart(self) -> WorkerHeartbeat:
        with self._lock:
            self._restarts += 1
            self._started_at = time.time()
            self._state = "idle"
            self._current_job_id = None
        return self.status()

    def is_alive(self) -> bool:
        with self._lock:
            return self._state != "stopped"

    def capabilities(self) -> WorkerCapabilities:
        return self.bridge.capabilities_snapshot()

    def status(self) -> WorkerHeartbeat:
        with self._lock:
            busy = self._current_job_id is not None or self.bridge.is_busy()
            state: WorkerState = "busy" if (busy and self._state != "stopped") else self._state
            return WorkerHeartbeat(
                worker_id=self.worker_id,
                pid=os.getpid() if self._state != "stopped" else None,
                state=state,
                uptime_s=time.time() - self._started_at if self._state != "stopped" else 0.0,
                current_job_id=self._current_job_id,
                jobs_completed=self._jobs_completed,
                jobs_failed=self._jobs_failed,
                restarts=self._restarts,
                last_heartbeat=time.time(),
                capabilities=self.capabilities(),
            )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if self._current_job_id == job_id:
                self.bridge.interrupt()
                return True
            return False

    def execute(
        self,
        envelope: JobEnvelope,
        *,
        on_progress: ProgressCallback | None = None,
        on_preview: PreviewCallback | None = None,
    ) -> WorkerResult:
        with self._lock:
            self._state = "busy"
            self._current_job_id = envelope.job_id
        try:
            if self._executor is not None:
                res = self._executor(envelope)
            else:
                res = WorkerResult(job_id=envelope.job_id, status="failed", error="in-process executor runs through JobQueue directly")
            with self._lock:
                if res.status == "completed":
                    self._jobs_completed += 1
                elif res.status == "failed":
                    self._jobs_failed += 1
            return res
        except Exception as error:
            with self._lock:
                self._jobs_failed += 1
            return WorkerResult(job_id=envelope.job_id, status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            with self._lock:
                self._current_job_id = None
                if self._state != "stopped":
                    self._state = "idle"


class WorkerPool:
    """Multi-GPU / heterogeneous worker pool with affinity and load-balancing."""

    def __init__(self, workers: Sequence[WorkerProtocol] = (), *, mode: str = "inprocess") -> None:
        self.mode = mode
        self._cv = threading.Condition()
        self._workers: dict[str, WorkerProtocol] = {}
        self._busy: set[str] = set()
        for worker in workers:
            self._workers[worker.worker_id] = worker

    def add_worker(self, worker: WorkerProtocol) -> None:
        with self._cv:
            self._workers[worker.worker_id] = worker
            self._cv.notify_all()

    def remove_worker(self, worker_id: str) -> bool:
        with self._cv:
            worker = self._workers.pop(worker_id, None)
            self._busy.discard(worker_id)
            self._cv.notify_all()
        if worker is not None:
            worker.stop()
            return True
        return False

    def get_worker(self, worker_id: str) -> WorkerProtocol | None:
        with self._cv:
            return self._workers.get(worker_id)

    def workers(self) -> list[WorkerProtocol]:
        with self._cv:
            return list(self._workers.values())

    def start_all(self) -> None:
        for worker in self.workers():
            worker.start()

    def shutdown(self) -> None:
        for worker in self.workers():
            worker.stop()

    def list_statuses(self, iso: Callable[[float | None], str | None] | None = None) -> list[dict[str, Any]]:
        return [worker.status().to_status_dict(iso=iso) for worker in self.workers()]

    def restart_worker(self, worker_id: str) -> WorkerHeartbeat:
        worker = self.get_worker(worker_id)
        if worker is None:
            raise KeyError(f"no worker {worker_id!r}")
        return worker.restart()

    def _score_candidate(self, worker: WorkerProtocol, *, preferred_model: str | None) -> tuple[int, int, int]:
        hb = worker.status()
        caps = hb.capabilities
        # Lower tuple sorts first:
        # 0) model affinity match (0 = exact loaded_model match, 1 = mismatch)
        affinity = 0 if (preferred_model and caps.loaded_model == preferred_model) else 1
        # 1) load-balancing: fewer completed jobs first
        load = hb.jobs_completed
        # 2) prefer higher free VRAM
        neg_vram = -(caps.vram_free_mb or 0)
        return (affinity, load, neg_vram)

    def _eligible(self, worker: WorkerProtocol, *, preferred_backend: str | None, model_family: str | None) -> bool:
        if not worker.is_alive():
            return False
        hb = worker.status()
        if hb.state in ("stopped", "unavailable", "error"):
            return False
        caps = hb.capabilities
        if preferred_backend and caps.backend != preferred_backend:
            return False
        return not (model_family and caps.supported_families and model_family not in caps.supported_families)

    def select_worker(
        self,
        *,
        preferred_model: str | None = None,
        preferred_backend: str | None = None,
        model_family: str | None = None,
    ) -> WorkerProtocol | None:
        """Pick the best idle worker using model affinity + load balancing."""

        with self._cv:
            candidates = [
                w
                for wid, w in self._workers.items()
                if wid not in self._busy and self._eligible(w, preferred_backend=preferred_backend, model_family=model_family)
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda w: self._score_candidate(w, preferred_model=preferred_model))
        return candidates[0]

    def acquire_worker(
        self,
        *,
        preferred_model: str | None = None,
        preferred_backend: str | None = None,
        model_family: str | None = None,
        timeout: float = 30.0,
    ) -> WorkerProtocol | None:
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                candidates = [
                    w
                    for wid, w in self._workers.items()
                    if wid not in self._busy and self._eligible(w, preferred_backend=preferred_backend, model_family=model_family)
                ]
                if candidates:
                    candidates.sort(key=lambda w: self._score_candidate(w, preferred_model=preferred_model))
                    chosen = candidates[0]
                    self._busy.add(chosen.worker_id)
                    return chosen
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=min(remaining, 0.2))

    def release_worker(self, worker_id: str) -> None:
        with self._cv:
            self._busy.discard(worker_id)
            self._cv.notify_all()

    def execute(
        self,
        envelope: JobEnvelope,
        *,
        on_progress: ProgressCallback | None = None,
        on_preview: PreviewCallback | None = None,
        preferred_backend: str | None = None,
        model_family: str | None = None,
        timeout: float = 30.0,
    ) -> WorkerResult:
        worker = self.acquire_worker(
            preferred_model=envelope.model_checkpoint,
            preferred_backend=preferred_backend,
            model_family=model_family,
            timeout=timeout,
        )
        if worker is None:
            return WorkerResult(job_id=envelope.job_id, status="failed", error="no eligible worker available in pool")
        try:
            return worker.execute(envelope, on_progress=on_progress, on_preview=on_preview)
        finally:
            self.release_worker(worker.worker_id)

    def dispatch_concurrent(
        self,
        envelopes: Sequence[JobEnvelope],
        *,
        preferred_backend: str | None = None,
        model_family: str | None = None,
    ) -> list[WorkerResult]:
        """Dispatch independent jobs concurrently across all workers in the pool."""

        if not envelopes:
            return []
        max_workers = max(1, len(self.workers()))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="remi-pool") as pool:
            futures = [
                pool.submit(
                    self.execute,
                    env,
                    preferred_backend=preferred_backend,
                    model_family=model_family,
                )
                for env in envelopes
            ]
            return [fut.result() for fut in futures]


def create_default_pool(
    *,
    worker_process: bool = False,
    worker_gpus: str | Sequence[str | int] | None = None,
    enable_mlx: bool = False,
    bridge: RuntimeBridge | None = None,
) -> WorkerPool:
    """Create the server's worker pool according to launch configuration."""

    gpus = parse_worker_gpus(worker_gpus)
    workers: list[WorkerProtocol] = []
    if gpus:
        mode = "pool"
        for gpu in gpus:
            workers.append(SubprocessWorker(worker_id=f"worker-gpu-{gpu}", gpu_id=gpu))
    elif worker_process:
        mode = "subprocess"
        workers.append(SubprocessWorker(worker_id="worker-0"))
    else:
        mode = "inprocess"
        workers.append(InProcessWorker(worker_id="worker-inprocess", bridge=bridge))

    if enable_mlx:
        env = detect_mlx_environment()
        if env.platform_supported:
            workers.append(MlxSidecarWorker(worker_id="worker-mlx-0"))
    return WorkerPool(workers, mode=mode)


__all__ = [
    "InProcessWorker",
    "WorkerPool",
    "create_default_pool",
    "parse_worker_gpus",
]
