"""Optional Apple Silicon MLX / ``mflux`` sidecar adapter implementing :class:`WorkerProtocol`.

Keeps zero PyTorch or Forge core imports: communicates with ``mflux`` (when
installed on Apple Silicon macOS) via an isolated command runner or injectable
test runner, writing outputs into the shared :class:`~remi.files.FileStore`
and returning standard :class:`WorkerResult` envelopes.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from remi.files import FileStore
from remi.worker.protocol import (
    JobEnvelope,
    PreviewCallback,
    ProgressCallback,
    WorkerCapabilities,
    WorkerHeartbeat,
    WorkerProgress,
    WorkerResult,
    WorkerState,
)

MLX_SUPPORTED_FAMILIES = ("flux3", "flux", "qwen21", "zimage")
MLX_QUANT_FORMATS = ("4bit", "6bit", "8bit", "bf16")


@dataclass(frozen=True)
class MlxEnvironment:
    available: bool
    platform_supported: bool
    mlx_version: str | None = None
    mflux_version: str | None = None
    unified_memory_mb: int | None = None
    supported_families: tuple[str, ...] = MLX_SUPPORTED_FAMILIES
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "platform_supported": self.platform_supported,
            "mlx_version": self.mlx_version,
            "mflux_version": self.mflux_version,
            "unified_memory_mb": self.unified_memory_mb,
            "supported_families": list(self.supported_families),
            "reason": self.reason,
        }


def _unified_memory_mb() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int((pages * page_size) // (1024 * 1024))
    except (AttributeError, ValueError, OSError):
        return None
    return None


def _pkg_version(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return "installed"


def detect_mlx_environment() -> MlxEnvironment:
    """Probe Apple Silicon + ``mlx`` / ``mflux`` availability safely without importing PyTorch."""

    is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")
    mem_mb = _unified_memory_mb() if is_apple_silicon else None
    if not is_apple_silicon:
        return MlxEnvironment(
            available=False,
            platform_supported=False,
            unified_memory_mb=mem_mb,
            reason="MLX sidecar requires Apple Silicon (macOS arm64)",
        )
    mlx_ver = _pkg_version("mlx")
    mflux_ver = _pkg_version("mflux")
    has_cli = shutil.which("mflux-generate") is not None
    if mlx_ver is None:
        return MlxEnvironment(
            available=False,
            platform_supported=True,
            mlx_version=None,
            mflux_version=mflux_ver,
            unified_memory_mb=mem_mb,
            reason="Python package 'mlx' is not installed (optional: pip install mlx mflux)",
        )
    if mflux_ver is None and not has_cli:
        return MlxEnvironment(
            available=False,
            platform_supported=True,
            mlx_version=mlx_ver,
            mflux_version=None,
            unified_memory_mb=mem_mb,
            reason="Package 'mflux' / CLI 'mflux-generate' is not installed",
        )
    return MlxEnvironment(
        available=True,
        platform_supported=True,
        mlx_version=mlx_ver,
        mflux_version=mflux_ver or "cli",
        unified_memory_mb=mem_mb,
        reason=None,
    )


MlxRunner = Callable[[JobEnvelope, Path, ProgressCallback | None, threading.Event], dict[str, Any]]
"""``(envelope, output_path, on_progress, cancel_event) -> {"infotext"?: str, "model"?: str}``."""


def build_mflux_command(envelope: JobEnvelope, output_path: Path, *, quantize: int = 8) -> list[str]:
    """Translate a :class:`JobEnvelope` into an ``mflux-generate`` CLI invocation."""

    core = envelope.core or {}
    scripts = envelope.scripts or {}
    sampler = scripts.get("sampler") if isinstance(scripts.get("sampler"), dict) else {}
    seed_cfg = scripts.get("seed") if isinstance(scripts.get("seed"), dict) else {}

    prompt = str(core.get("prompt") or "")
    width = int(core.get("width") or 1024)
    height = int(core.get("height") or 1024)
    steps = int(sampler.get("steps") or core.get("steps") or 20)
    seed = int(seed_cfg.get("seed", -1))
    model_name = str(envelope.model_checkpoint or envelope.meta.get("mlx_model") or "schnell")

    cmd = [
        shutil.which("mflux-generate") or "mflux-generate",
        "--model",
        model_name,
        f"--prompt={prompt}",  # one item: a prompt starting with "-" is not read as an option
        "--width",
        str(width),
        "--height",
        str(height),
        "--steps",
        str(steps),
        "--quantize",
        str(int(envelope.meta.get("mlx_quantize", quantize))),
        "--output",
        str(output_path),
    ]
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    return cmd


def _default_mflux_runner(
    envelope: JobEnvelope,
    output_path: Path,
    on_progress: ProgressCallback | None,
    cancel_event: threading.Event,
) -> dict[str, Any]:
    cmd = build_mflux_command(envelope, output_path)
    if on_progress is not None:
        on_progress(WorkerProgress(job_id=envelope.job_id, seq=envelope.seq, step=0, steps=1, progress=0.05, text="MLX sidecar generating"))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # communicate() reads both pipes while it waits (mflux's progress output would otherwise fill
        # one and block it); a timed-out call loses nothing and is simply called again
        while True:
            if cancel_event.is_set():
                proc.terminate()
                try:
                    proc.communicate(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                raise InterruptedError("cancelled")
            try:
                stdout_text, stderr_text = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()
    if proc.returncode != 0:
        raise RuntimeError(f"mflux-generate failed (code={proc.returncode}): {(stderr_text or stdout_text).strip()[-300:]}")
    if on_progress is not None:
        on_progress(WorkerProgress(job_id=envelope.job_id, seq=envelope.seq, step=1, steps=1, progress=1.0, text="MLX sidecar done"))
    prompt = str(envelope.core.get("prompt") or "")
    return {"infotext": f"{prompt}\nBackend: MLX Sidecar, Model: {envelope.model_checkpoint or 'mflux'}"}


class MlxSidecarWorker:
    """Apple Silicon MLX / ``mflux`` sidecar implementing :class:`WorkerProtocol`."""

    def __init__(
        self,
        worker_id: str = "worker-mlx-0",
        *,
        env_probe: Callable[[], MlxEnvironment] = detect_mlx_environment,
        runner: MlxRunner | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._env_probe = env_probe
        self._runner = runner or _default_mflux_runner
        self._lock = threading.RLock()
        self._env = env_probe()
        self._state: WorkerState = "idle" if self._env.available else "unavailable"
        self._started_at: float = time.time() if self._env.available else 0.0
        self._last_heartbeat: float | None = time.time() if self._env.available else None
        self._current_job_id: str | None = None
        self._cancel_event: threading.Event | None = None
        self._jobs_completed: int = 0
        self._jobs_failed: int = 0
        self._restarts: int = 0
        self._loaded_model: str | None = None

    def start(self) -> None:
        with self._lock:
            self._env = self._env_probe()
            if self._env.available:
                self._state = "idle"
                self._started_at = time.time()
                self._last_heartbeat = self._started_at
            else:
                self._state = "unavailable"

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._state = "stopped"
            self._current_job_id = None

    def restart(self) -> WorkerHeartbeat:
        with self._lock:
            self._restarts += 1
        self.stop()
        self.start()
        return self.status()

    def is_alive(self) -> bool:
        with self._lock:
            return self._state in ("idle", "busy")

    def capabilities(self) -> WorkerCapabilities:
        with self._lock:
            return WorkerCapabilities(
                backend="mlx-mflux",
                device="mlx",
                gpu_name="Apple Silicon MLX" if self._env.platform_supported else None,
                vram_total_mb=self._env.unified_memory_mb,
                vram_free_mb=self._env.unified_memory_mb,
                supported_families=self._env.supported_families,
                loaded_model=self._loaded_model,
                quant_formats=MLX_QUANT_FORMATS,
            )

    def status(self) -> WorkerHeartbeat:
        with self._lock:
            uptime = (time.time() - self._started_at) if (self._state in ("idle", "busy") and self._started_at > 0) else 0.0
            return WorkerHeartbeat(
                worker_id=self.worker_id,
                pid=os.getpid() if self._state in ("idle", "busy") else None,
                state=self._state,
                uptime_s=uptime,
                current_job_id=self._current_job_id,
                jobs_completed=self._jobs_completed,
                jobs_failed=self._jobs_failed,
                restarts=self._restarts,
                last_heartbeat=self._last_heartbeat,
                capabilities=self.capabilities(),
            )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if self._current_job_id == job_id and self._cancel_event is not None:
                self._cancel_event.set()
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
            if self._state == "unavailable":
                return WorkerResult(
                    job_id=envelope.job_id,
                    status="failed",
                    error=self._env.reason or "MLX sidecar is unavailable on this machine",
                )
            if self._state == "stopped":
                return WorkerResult(
                    job_id=envelope.job_id,
                    status="failed",
                    error="MLX sidecar is stopped",
                )
            cancel_ev = threading.Event()
            self._cancel_event = cancel_ev
            self._current_job_id = envelope.job_id
            self._state = "busy"

        try:
            with tempfile.TemporaryDirectory(prefix="remi-mlx-") as tmpdir:
                out_path = Path(tmpdir) / f"{envelope.job_id}.png"
                info = self._runner(envelope, out_path, on_progress, cancel_ev) or {}
                if cancel_ev.is_set():
                    return WorkerResult(job_id=envelope.job_id, status="cancelled", error="Cancelled")
                if not out_path.is_file():
                    raise RuntimeError("MLX sidecar completed without writing output image")
                infotext = info.get("infotext") if isinstance(info, dict) else None
                outputs: list[dict[str, Any]] = []
                file_ids: list[str] = []
                if envelope.file_store_root:
                    from PIL import Image

                    files = FileStore(envelope.file_store_root)
                    try:
                        with Image.open(out_path) as image:
                            image.load()
                            rec = files.save_output_image(image, infotext, job_id=envelope.job_id, tab=envelope.tab)
                        file_ids.append(rec.id)
                        outputs.append({"id": rec.id, "kind": "image", "width": rec.width, "height": rec.height, "infotext": infotext})
                    finally:
                        files.close()
                with self._lock:
                    self._jobs_completed += 1
                    self._loaded_model = envelope.model_checkpoint or info.get("model") or self._loaded_model
                    self._last_heartbeat = time.time()
                return WorkerResult(
                    job_id=envelope.job_id,
                    status="completed",
                    output_file_ids=file_ids,
                    outputs=outputs,
                    infotext=infotext,
                )
        except InterruptedError:
            return WorkerResult(job_id=envelope.job_id, status="cancelled", error="Cancelled")
        except Exception as error:
            with self._lock:
                self._jobs_failed += 1
            return WorkerResult(job_id=envelope.job_id, status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            with self._lock:
                if self._current_job_id == envelope.job_id:
                    self._current_job_id = None
                    if self._state == "busy":
                        self._state = "idle"
                if self._cancel_event is cancel_ev:
                    self._cancel_event = None


__all__ = [
    "MLX_QUANT_FORMATS",
    "MLX_SUPPORTED_FAMILIES",
    "MlxEnvironment",
    "MlxRunner",
    "MlxSidecarWorker",
    "build_mflux_command",
    "detect_mlx_environment",
]
