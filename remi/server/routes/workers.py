"""Worker process status & management: ``GET /api/v2/workers`` and restart."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException

from remi.server import models as m
from remi.server.deps import ServicesDep, iso
from remi.worker.mlx_sidecar import detect_mlx_environment
from remi.worker.pool import InProcessWorker
from remi.worker.subprocess_worker import SubprocessWorker

router = APIRouter(prefix="/workers")

WorkerMode = Literal["inprocess", "subprocess", "pool"]
WorkerState = Literal["starting", "idle", "busy", "restarting", "stopped", "error", "unavailable"]
WorkerBackend = Literal["inprocess", "forge-pytorch", "mlx-mflux"]


class WorkerStatusModel(m.Model):
    worker_id: str
    pid: int | None = None
    state: WorkerState
    backend: WorkerBackend
    device: str
    gpu_name: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    loaded_model: str | None = None
    supported_families: list[str] = []
    quant_formats: list[str] = []
    current_job_id: str | None = None
    jobs_completed: int = 0
    jobs_failed: int = 0
    restarts: int = 0
    uptime_s: float = 0.0
    last_heartbeat: str | None = None


class MlxEnvironmentModel(m.Model):
    available: bool
    platform_supported: bool
    mlx_version: str | None = None
    mflux_version: str | None = None
    unified_memory_mb: int | None = None
    supported_families: list[str] = []
    reason: str | None = None


class WorkersResponse(m.Model):
    mode: WorkerMode | str
    dispatch_active: bool = False
    preview: bool = False
    note: str | None = None
    workers: list[WorkerStatusModel]
    mlx: MlxEnvironmentModel


class WorkerRestartResponse(m.Model):
    ok: bool = True
    dispatch_active: bool = False
    note: str | None = None
    worker: WorkerStatusModel


@router.get("", response_model=WorkersResponse, summary="Inference worker pool & MLX sidecar status")
def list_workers(services: ServicesDep) -> dict[str, Any]:
    pool = services.workers
    dispatch_active = bool(getattr(pool, "dispatch_active", False))
    workers = pool.workers()
    preview = (not dispatch_active) and (pool.mode != "inprocess" or any(not isinstance(w, InProcessWorker) for w in workers))
    if dispatch_active:
        note = None
    elif preview:
        note = "Preview only: queued jobs still run in-process (worker pool dispatch is inactive)."
    else:
        note = "Queued jobs run in-process; restarting resets worker counters without restarting the server."
    return {
        "mode": pool.mode,
        "dispatch_active": dispatch_active,
        "preview": preview,
        "note": note,
        "workers": pool.list_statuses(iso=iso),
        "mlx": detect_mlx_environment().to_dict(),
    }


@router.get("/{worker_id}", response_model=WorkerStatusModel, responses={404: {"model": m.Detail}}, summary="Status of one inference worker")
def get_worker(worker_id: str, services: ServicesDep) -> dict[str, Any]:
    worker = services.workers.get_worker(worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"no worker {worker_id!r}")
    return worker.status().to_status_dict(iso=iso)


@router.post("/{worker_id}/restart", response_model=WorkerRestartResponse, responses={404: {"model": m.Detail}, 409: {"model": m.Detail}}, summary="Restart a worker cleanly")
def restart_worker(worker_id: str, services: ServicesDep) -> dict[str, Any]:
    pool = services.workers
    dispatch_active = bool(getattr(pool, "dispatch_active", False))
    worker = pool.get_worker(worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"no worker {worker_id!r}")
    if not dispatch_active and isinstance(worker, SubprocessWorker):
        raise HTTPException(
            status_code=409,
            detail="worker pool dispatch is inactive (preview mode); subprocess workers are not started",
        )
    hb = worker.restart()
    note = "In-process worker: counters reset without restarting the server process." if isinstance(worker, InProcessWorker) else None
    return {
        "ok": True,
        "dispatch_active": dispatch_active,
        "note": note,
        "worker": hb.to_status_dict(iso=iso),
    }
