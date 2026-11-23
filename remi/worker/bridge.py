"""``RuntimeBridge`` facade decoupling FastAPI server routes from direct
in-process reads of ``modules.shared.state``, ``sd_model`` and memory state.

In default in-process mode the bridge reads Forge's runtime lazily; when an
external worker or pool is active, routes read snapshots through the exact same
interface without importing or locking PyTorch/Forge state directly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from remi.realtime import progress_snapshot as state_progress_snapshot
from remi.worker.protocol import WorkerCapabilities, WorkerProgress


class RuntimeBridge:
    """Facade between ``remi.server`` routes and the inference runtime."""

    def __init__(
        self,
        *,
        state_getter: Callable[[], Any] | None = None,
        remote_progress: Callable[[], WorkerProgress | None] | None = None,
        remote_capabilities: Callable[[], WorkerCapabilities | None] | None = None,
        remote_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._state_getter = state_getter
        self._remote_progress = remote_progress
        self._remote_capabilities = remote_capabilities
        self._remote_cancel = remote_cancel

    def _state(self) -> Any | None:
        if self._state_getter is not None:
            return self._state_getter()
        shared = sys.modules.get("modules.shared")
        return getattr(shared, "state", None) if shared is not None else None

    def _loaded_checkpoint_info(self) -> Any | None:
        sd_models = sys.modules.get("modules.sd_models")
        model = getattr(getattr(sd_models, "model_data", None), "sd_model", None) if sd_models is not None else None
        return getattr(model, "sd_checkpoint_info", None)

    def progress_snapshot(self) -> dict[str, Any]:
        """Step, ETA and job progress counters (remote worker or local state)."""

        if self._remote_progress is not None:
            prog = self._remote_progress()
            if prog is not None:
                return {
                    "step": prog.step,
                    "steps": prog.steps,
                    "job_no": prog.job_no,
                    "job_count": prog.job_count,
                    "progress": round(prog.progress, 4),
                    "eta": round(prog.eta, 2) if prog.eta is not None else None,
                    "text": prog.text,
                }
        state = self._state()
        if state is None:
            return {"step": 0, "steps": 0, "job_no": 0, "job_count": 0, "progress": 0.0, "eta": None, "text": None}
        return state_progress_snapshot(state)

    def loaded_model(self) -> str | None:
        """Title of the currently loaded checkpoint (or None)."""

        if self._remote_capabilities is not None:
            caps = self._remote_capabilities()
            if caps is not None and caps.loaded_model is not None:
                return caps.loaded_model
        info = self._loaded_checkpoint_info()
        return getattr(info, "title", None) or getattr(info, "name", None)

    def loaded_model_hash(self) -> str | None:
        """SHA-256 of the loaded checkpoint when available."""

        sha256 = getattr(self._loaded_checkpoint_info(), "sha256", None)
        return sha256 if isinstance(sha256, str) and sha256 else None

    def loaded_weights(self) -> list[dict[str, Any]]:
        """Loaded weights held by the memory manager."""

        memory_management = sys.modules.get("backend.memory_management")
        items: list[dict[str, Any]] = []
        for entry in list(getattr(memory_management, "current_loaded_models", None) or []):
            try:
                patcher = entry.model
                inner = getattr(patcher, "model", None) if patcher is not None else None
                memory = int(entry.model_loaded_memory())
                items.append(
                    {
                        "name": type(inner).__name__ if inner is not None else "?",
                        "memory_mb": memory // 2**20 if memory > 0 else None,
                        "device": str(entry.device) if entry.device is not None else None,
                    }
                )
            except Exception as error:
                items.append({"name": f"? ({type(error).__name__})", "memory_mb": None, "device": None})
        return items

    def is_busy(self) -> bool:
        state = self._state()
        if state is None:
            return False
        return int(getattr(state, "job_count", 0) or 0) > 0

    def interrupt(self) -> None:
        if self._remote_cancel is not None:
            self._remote_cancel()
            return
        state = self._state()
        if state is not None and hasattr(state, "interrupt"):
            state.interrupt()

    def skip(self) -> None:
        state = self._state()
        if state is not None and hasattr(state, "skip"):
            state.skip()

    def stop_generating(self) -> None:
        state = self._state()
        if state is not None and hasattr(state, "stop_generating"):
            state.stop_generating()

    def capabilities_snapshot(self) -> WorkerCapabilities:
        """Snapshot of device, VRAM and loaded model for the active runtime."""

        if self._remote_capabilities is not None:
            caps = self._remote_capabilities()
            if caps is not None:
                return caps

        device = "cpu"
        gpu_name: str | None = None
        vram_total_mb: int | None = None
        vram_free_mb: int | None = None
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            try:
                if torch_mod.cuda.is_available():
                    device = "cuda:0"
                    gpu_name = torch_mod.cuda.get_device_name(0)
                    free_b, total_b = torch_mod.cuda.mem_get_info(0)
                    vram_total_mb = int(total_b // (1024 * 1024))
                    vram_free_mb = int(free_b // (1024 * 1024))
                elif getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
                    device = "mps"
                    gpu_name = "Apple Silicon MPS"
            except Exception:
                pass

        return WorkerCapabilities(
            backend="inprocess",
            device=device,
            gpu_name=gpu_name,
            vram_total_mb=vram_total_mb,
            vram_free_mb=vram_free_mb,
            supported_families=("anima", "flux3", "qwen21", "moss", "krea2", "zimage", "ernie", "fibo", "sdxl", "sd"),
            loaded_model=self.loaded_model(),
            quant_formats=("bf16", "fp8", "int8_convrot", "gguf", "nvfp4", "svdquant"),
        )


__all__ = ["RuntimeBridge"]
