"""IPC / JSON-lines protocol between the FastAPI control plane and inference workers.

All job payloads, progress ticks, preview frames, capabilities snapshots,
cancellation requests and results are pure JSON-serialisable structures using
shared :class:`~remi.files.FileStore` IDs (no cross-process PIL/tensor objects).
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from remi.realtime import FORMAT_JPEG, FORMAT_WEBP, PreviewFrame

PROTOCOL_VERSION = 1

WorkerState = Literal["starting", "idle", "busy", "restarting", "stopped", "error", "unavailable"]
WorkerBackend = Literal["inprocess", "forge-pytorch", "mlx-mflux"]
MessageType = Literal["run_job", "cancel", "heartbeat", "ping", "pong", "progress", "preview", "result", "capabilities", "shutdown", "error"]


@dataclass(frozen=True)
class JobEnvelope:
    job_id: str
    seq: int
    tab: str
    core: dict[str, Any] = field(default_factory=dict)
    scripts: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    file_store_root: str = ""
    gpu_id: str | None = None
    model_checkpoint: str | None = None
    cancel_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "seq": self.seq,
            "tab": self.tab,
            "core": dict(self.core),
            "scripts": dict(self.scripts),
            "meta": dict(self.meta),
            "file_store_root": self.file_store_root,
            "gpu_id": self.gpu_id,
            "model_checkpoint": self.model_checkpoint,
            "cancel_token": self.cancel_token,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobEnvelope:
        return cls(
            job_id=str(data["job_id"]),
            seq=int(data.get("seq", 0)),
            tab=str(data.get("tab", "txt2img")),
            core=dict(data.get("core") or {}),
            scripts=dict(data.get("scripts") or {}),
            meta=dict(data.get("meta") or {}),
            file_store_root=str(data.get("file_store_root") or ""),
            gpu_id=data.get("gpu_id"),
            model_checkpoint=data.get("model_checkpoint"),
            cancel_token=data.get("cancel_token"),
        )


@dataclass(frozen=True)
class WorkerProgress:
    job_id: str
    seq: int
    step: int
    steps: int
    job_no: int = 0
    job_count: int = 1
    progress: float = 0.0
    eta: float | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "seq": self.seq,
            "step": self.step,
            "steps": self.steps,
            "job_no": self.job_no,
            "job_count": self.job_count,
            "progress": self.progress,
            "eta": self.eta,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkerProgress:
        return cls(
            job_id=str(data["job_id"]),
            seq=int(data.get("seq", 0)),
            step=int(data.get("step", 0)),
            steps=int(data.get("steps", 0)),
            job_no=int(data.get("job_no", 0)),
            job_count=int(data.get("job_count", 1)),
            progress=float(data.get("progress", 0.0)),
            eta=float(data["eta"]) if data.get("eta") is not None else None,
            text=data.get("text"),
        )


@dataclass(frozen=True)
class PreviewChunk:
    job_id: str
    seq: int
    step: int
    fmt: Literal["jpeg", "webp"] = "jpeg"
    data_b64: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "seq": self.seq,
            "step": self.step,
            "fmt": self.fmt,
            "data_b64": self.data_b64,
        }

    def to_preview_frame(self) -> PreviewFrame:
        fmt_code = FORMAT_JPEG if self.fmt == "jpeg" else FORMAT_WEBP
        raw = base64.b64decode(self.data_b64.encode("ascii")) if self.data_b64 else b""
        return PreviewFrame(fmt=fmt_code, job_seq=self.seq, step=self.step, data=raw)

    @classmethod
    def from_bytes(cls, job_id: str, seq: int, step: int, data: bytes, fmt: Literal["jpeg", "webp"] = "jpeg") -> PreviewChunk:
        return cls(job_id=job_id, seq=seq, step=step, fmt=fmt, data_b64=base64.b64encode(data).decode("ascii"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreviewChunk:
        fmt_raw = str(data.get("fmt", "jpeg"))
        fmt: Literal["jpeg", "webp"] = "webp" if fmt_raw == "webp" else "jpeg"
        return cls(
            job_id=str(data["job_id"]),
            seq=int(data.get("seq", 0)),
            step=int(data.get("step", 0)),
            fmt=fmt,
            data_b64=str(data.get("data_b64") or ""),
        )


@dataclass(frozen=True)
class WorkerResult:
    job_id: str
    status: Literal["completed", "failed", "interrupted", "cancelled"] = "completed"
    output_file_ids: list[str] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    infotext: str | None = None
    error: str | None = None
    errors: list[dict[str, Any]] | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "output_file_ids": list(self.output_file_ids),
            "outputs": list(self.outputs),
            "infotext": self.infotext,
            "error": self.error,
            "errors": list(self.errors) if self.errors is not None else None,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkerResult:
        status_raw = str(data.get("status", "completed"))
        status: Literal["completed", "failed", "interrupted", "cancelled"] = (
            status_raw if status_raw in ("completed", "failed", "interrupted", "cancelled") else "failed"  # type: ignore[assignment]
        )
        return cls(
            job_id=str(data["job_id"]),
            status=status,
            output_file_ids=list(data.get("output_file_ids") or []),
            outputs=list(data.get("outputs") or []),
            infotext=data.get("infotext"),
            error=data.get("error"),
            errors=list(data["errors"]) if data.get("errors") is not None else None,
            warnings=list(data.get("warnings") or []),
        )


@dataclass(frozen=True)
class WorkerCapabilities:
    backend: WorkerBackend = "inprocess"
    device: str = "cpu"
    gpu_name: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    supported_families: tuple[str, ...] = ()
    loaded_model: str | None = None
    quant_formats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": self.vram_free_mb,
            "supported_families": list(self.supported_families),
            "loaded_model": self.loaded_model,
            "quant_formats": list(self.quant_formats),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkerCapabilities:
        backend_raw = str(data.get("backend", "inprocess"))
        backend: WorkerBackend = backend_raw if backend_raw in ("inprocess", "forge-pytorch", "mlx-mflux") else "inprocess"  # type: ignore[assignment]
        return cls(
            backend=backend,
            device=str(data.get("device") or "cpu"),
            gpu_name=data.get("gpu_name"),
            vram_total_mb=int(data["vram_total_mb"]) if data.get("vram_total_mb") is not None else None,
            vram_free_mb=int(data["vram_free_mb"]) if data.get("vram_free_mb") is not None else None,
            supported_families=tuple(str(x) for x in (data.get("supported_families") or ())),
            loaded_model=data.get("loaded_model"),
            quant_formats=tuple(str(x) for x in (data.get("quant_formats") or ())),
        )


@dataclass(frozen=True)
class WorkerHeartbeat:
    worker_id: str
    pid: int | None
    state: WorkerState
    uptime_s: float = 0.0
    current_job_id: str | None = None
    jobs_completed: int = 0
    jobs_failed: int = 0
    restarts: int = 0
    last_heartbeat: float | None = None
    capabilities: WorkerCapabilities = field(default_factory=WorkerCapabilities)

    def to_status_dict(self, iso: Callable[[float | None], str | None] | None = None) -> dict[str, Any]:
        caps = self.capabilities
        hb_val = iso(self.last_heartbeat) if (iso is not None and self.last_heartbeat is not None) else self.last_heartbeat
        return {
            "worker_id": self.worker_id,
            "pid": self.pid,
            "state": self.state,
            "backend": caps.backend,
            "device": caps.device,
            "gpu_name": caps.gpu_name,
            "vram_total_mb": caps.vram_total_mb,
            "vram_free_mb": caps.vram_free_mb,
            "loaded_model": caps.loaded_model,
            "supported_families": list(caps.supported_families),
            "quant_formats": list(caps.quant_formats),
            "current_job_id": self.current_job_id,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "restarts": self.restarts,
            "uptime_s": round(max(0.0, self.uptime_s), 2),
            "last_heartbeat": hb_val,
        }


@dataclass(frozen=True)
class WorkerMessage:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


def encode_message(message: WorkerMessage) -> bytes:
    """Serialize a :class:`WorkerMessage` as a single UTF-8 JSON line."""

    return (json.dumps(message.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode_message(raw: str | bytes) -> WorkerMessage:
    """Parse a JSON line into a :class:`WorkerMessage`."""

    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    data = json.loads(text)
    if not isinstance(data, Mapping) or not isinstance(data.get("type"), str):
        raise ValueError("invalid worker message: missing 'type'")
    version = int(data.get("protocol_version", PROTOCOL_VERSION))
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported worker protocol version {version} (expected {PROTOCOL_VERSION})")
    payload = data.get("payload")
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("worker message payload must be an object")
    return WorkerMessage(
        type=data["type"],
        payload=dict(payload),
        protocol_version=version,
        timestamp=float(data.get("timestamp", time.time())),
    )


ProgressCallback = Callable[[WorkerProgress], None]
PreviewCallback = Callable[[PreviewChunk], None]


@runtime_checkable
class WorkerProtocol(Protocol):
    """Unified interface implemented by in-process, subprocess, and MLX sidecar workers."""

    worker_id: str

    def start(self) -> None: ...

    def stop(self, timeout: float = 5.0) -> None: ...

    def restart(self) -> WorkerHeartbeat: ...

    def is_alive(self) -> bool: ...

    def status(self) -> WorkerHeartbeat: ...

    def capabilities(self) -> WorkerCapabilities: ...

    def execute(
        self,
        envelope: JobEnvelope,
        *,
        on_progress: ProgressCallback | None = None,
        on_preview: PreviewCallback | None = None,
    ) -> WorkerResult: ...

    def cancel(self, job_id: str) -> bool: ...


__all__ = [
    "PROTOCOL_VERSION",
    "JobEnvelope",
    "MessageType",
    "PreviewCallback",
    "PreviewChunk",
    "ProgressCallback",
    "WorkerBackend",
    "WorkerCapabilities",
    "WorkerHeartbeat",
    "WorkerMessage",
    "WorkerProgress",
    "WorkerProtocol",
    "WorkerResult",
    "WorkerState",
    "decode_message",
    "encode_message",
]
