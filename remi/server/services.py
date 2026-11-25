"""Process-wide services of the Remi server: the event bus, the file index,
the job queue and its worker, the network index and the realtime threads.

They outlive one app instance ("Reload UI" builds a new app, the queue keeps
running). Data lives under ``<data path>/remi-data``.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from remi.files import DEFAULT_MAX_UPLOAD, FileStore
from remi.jobs import ChainRunner, ChainStore, JobQueue, JobStore, run_job, run_on_main_thread
from remi.networks import NetworkIndex
from remi.realtime import EventBus, PreviewEncoder, ProgressMonitor
from remi.worker import RuntimeBridge, WorkerPool, create_default_pool

log = logging.getLogger(__name__)

OUTPUT_DIR_OPTIONS = ("outdir_samples", "outdir_txt2img_samples", "outdir_img2img_samples", "outdir_extras_samples", "outdir_hires_samples", "outdir_grids", "outdir_txt2img_grids", "outdir_img2img_grids", "outdir_save")


@dataclass
class Services:
    root: Path
    bus: EventBus
    files: FileStore
    store: JobStore
    queue: JobQueue
    networks: NetworkIndex
    state: Callable[[], Any] | None = None
    """``shared.state`` getter for the realtime threads (tests pass a fake)"""
    realtime: bool = True
    housekeeping_enabled: bool = True
    worker_process: bool | None = None
    worker_gpus: str | None = None
    worker_mlx: bool | None = None
    chains: ChainRunner = field(init=False)
    bridge: RuntimeBridge = field(init=False)
    workers: WorkerPool = field(init=False)
    _threads: list[Any] = field(default_factory=list)
    _started: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)

    HOUSEKEEPING_INTERVAL: ClassVar[float] = 3600

    def _upsample_stage(self, stage: Any, upstream_prompt: str | None, upstream_outputs: Any) -> str:
        """A pipeline's ``prompt_upsample`` stage, with the provider the prompt tools are set to."""

        from remi import prompt_tools as pt
        from remi.jobs.chains import _default_upsampler

        return _default_upsampler(stage, upstream_prompt, upstream_outputs, config=pt.tools_for(self).translation_config())

    def __post_init__(self) -> None:
        self.chains = ChainRunner(ChainStore(self.store), self.queue, self.files, self.bus, upsampler=self._upsample_stage)
        self.bridge = RuntimeBridge(state_getter=self.state)
        use_process = (os.environ.get("REMI_WORKER_PROCESS") == "1") if self.worker_process is None else bool(self.worker_process)
        use_gpus = (os.environ.get("REMI_WORKER_GPUS") or None) if self.worker_gpus is None else (self.worker_gpus.strip() or None)
        use_mlx = (os.environ.get("REMI_WORKER_MLX") == "1") if self.worker_mlx is None else bool(self.worker_mlx)
        self.workers = create_default_pool(
            worker_process=use_process,
            worker_gpus=use_gpus,
            enable_mlx=use_mlx,
            bridge=self.bridge,
        )
        self.queue.add_completion_hook(self._record_job_lineage)

    def _record_job_lineage(self, job: Any) -> None:
        outputs = getattr(job, "outputs", None) or []
        image_ids = [out["id"] for out in outputs if isinstance(out, dict) and out.get("kind", "image") == "image" and isinstance(out.get("id"), str)]
        if not image_ids:
            return
        try:
            from remi.library import gallery_for

            gallery_for(self.files).index_job_lineage(job, image_ids, job_store=self.store)
        except Exception:
            log.debug("lineage indexing skipped for job %s", getattr(job, "id", "?"), exc_info=True)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        max_upload_bytes: int | Callable[[], int] = DEFAULT_MAX_UPLOAD,
        realtime: bool = True,
        housekeeping: bool = True,
        worker_process: bool | None = None,
        worker_gpus: str | None = None,
        worker_mlx: bool | None = None,
        **queue_options: Any,
    ) -> Services:
        """``queue_options`` go to :class:`JobQueue` (``executor``,
        ``dispatch``, ``lock``, ``validate``, ``state``, ``hash_layout``);
        ``realtime``/``housekeeping``: start those threads with the queue;
        ``worker_process``/``worker_gpus``/``worker_mlx``: worker isolation,
        multi-GPU pool, and optional Apple Silicon MLX sidecar selection."""

        root = Path(root)
        bus = EventBus()
        files = FileStore(root, max_upload_bytes=max_upload_bytes)
        store = JobStore(root / "jobs.db")
        queue_options.setdefault("executor", run_job)
        queue_options.setdefault("dispatch", run_on_main_thread)
        queue = JobQueue(store, files, bus, **queue_options)
        return cls(
            root,
            bus,
            files,
            store,
            queue,
            NetworkIndex(files, on_change=lambda kind: bus.publish("networks.changed", **({"kind": kind} if kind else {}))),
            state=queue_options.get("state"),
            realtime=realtime,
            housekeeping_enabled=housekeeping,
            worker_process=worker_process,
            worker_gpus=worker_gpus,
            worker_mlx=worker_mlx,
        )

    def start(self) -> None:
        """Start the worker (and the progress/preview threads); idempotent."""

        if self._started:
            return
        self._started = True
        self._stop.clear()
        self.queue.start()
        try:
            moved = self.chains.reconcile()  # after start(): jobs the last process left running are interrupted now
        except Exception:
            log.exception("settling the pipelines left by the last run")
        else:
            if moved:
                log.info("Settled %d pipeline(s) left running by the last shutdown", moved)
        if self.realtime:
            state = {"state": self.state} if self.state is not None else {}
            monitor = ProgressMonitor(self.bus, self.queue.current_ref, on_tick=self.queue.poll, **state)
            # a step's progress goes out with its preview: one wake-up of the hub per step
            self._threads = [monitor, PreviewEncoder(self.bus, self.queue.current_ref, before_frame=monitor.publish_now, **state)]
            for thread in self._threads:
                thread.start()
        if self.housekeeping_enabled:
            threading.Thread(target=self._housekeeping_loop, name="remi-housekeeping", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.stop()
        self.queue.stop()
        self._started = False

    def _housekeeping_loop(self) -> None:
        stop = self._stop
        while True:
            self.housekeeping()
            if stop.wait(self.HOUSEKEEPING_INTERVAL):
                return

    def close(self) -> None:
        """Stop everything and close the databases."""

        from remi import prompt_tools
        from remi.library import close_gallery
        from remi.recipes import close_recipes

        self.stop()
        self.workers.shutdown()
        self.chains.close()
        close_recipes(self.files)
        close_gallery(self.files)
        prompt_tools.forget(self)
        self.files.close()
        self.store.close()

    def housekeeping(self) -> None:
        """Prune old uploads, unsaved outputs and gallery zips, and index
        outputs saved outside the queue (at start, then hourly)."""

        from remi.server.settings import option

        try:
            days = _positive(option("remi_upload_retention_days", 7), 7)
            referenced = self.store.referenced_strings()
            removed = self.files.prune_uploads(days * 86400, keep=referenced)
            if removed:
                log.info("Removed %d old upload(s)", removed)
            from remi.library import gallery_for

            # a starred image or one on a board was chosen to be kept, saved or not
            removed = self.files.prune_unsaved(days * 86400, keep=referenced | gallery_for(self.files).kept_ids())
            if removed:
                log.info("Removed %d old unsaved output(s)", removed)
        except Exception:
            log.exception("pruning uploads")
        try:
            from remi.library import gallery_for

            gallery_for(self.files).prune_exports()
        except Exception:
            log.exception("pruning gallery exports")
        try:
            added = self.files.scan_outputs(output_dirs())
            if added:
                log.info("Indexed %d output image(s)", added)
        except Exception:
            log.exception("indexing outputs")


def output_dirs() -> list[str]:
    """The output folders that are set (a value of another kind reads as unset)."""

    from remi.setting_values import setting

    return [path for name in OUTPUT_DIR_OPTIONS if (path := setting(name, ""))]


def worker_options_from_cmd_opts(cmd_opts: Any = None) -> tuple[bool, str | None, bool]:
    """Resolve ``(worker_process, worker_gpus, worker_mlx)`` from CLI flags
    (``--worker-process``, ``--worker-gpus``, ``--worker-mlx``) with
    ``REMI_WORKER_*`` environment variable fallbacks."""

    opts = cmd_opts
    if opts is None:
        try:
            from modules.shared_cmd_options import cmd_opts as active_opts

            opts = active_opts
        except Exception:
            opts = None

    cli_process = bool(getattr(opts, "worker_process", False)) if opts is not None else False
    raw_gpus = getattr(opts, "worker_gpus", None) if opts is not None else None
    cli_gpus = str(raw_gpus).strip() or None if raw_gpus is not None else None
    cli_mlx = bool(getattr(opts, "worker_mlx", False)) if opts is not None else False

    use_process = cli_process or os.environ.get("REMI_WORKER_PROCESS") == "1"
    use_gpus = cli_gpus if cli_gpus is not None else (os.environ.get("REMI_WORKER_GPUS") or None)
    use_mlx = cli_mlx or os.environ.get("REMI_WORKER_MLX") == "1"
    return use_process, use_gpus, use_mlx


_default: Services | None = None
_default_lock = threading.Lock()


def _positive(value: Any, default: float) -> float:
    """A size or a span: a finite number above zero, else ``default`` (an
    upload limit of 0 refused every upload). A registered setting is already
    within its slider here; this covers a reader without the registration."""

    return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else default


def _within(value: float, limits: tuple[float, float]) -> float:
    return float(min(max(value, limits[0]), limits[1]))


def configured_upload_limit() -> int:
    """The setting "Largest upload" in bytes. Read on every request (the body
    limit): a value of the wrong kind reads as the default rather than failing
    them all, and a number as the nearest within the slider's range, 1 to 2048
    MB (``0`` or ``1e-9`` made a limit of 0 bytes that refused every upload,
    ``1e303`` overflowed into bytes)."""

    from remi.server.settings import UPLOAD_MB, option

    return int(_within(_positive(option("remi_upload_max_mb", 64), 64), UPLOAD_MB) * 1024 * 1024)


def default_services(factory: Callable[[Path], Services] | None = None) -> Services:
    """The services of this process (created on first use)."""

    global _default
    with _default_lock:
        if _default is None:
            from modules.paths_internal import data_path

            # Not "remi": the data path defaults to the repository root, where remi/ is the package
            root = Path(data_path) / "remi-data"
            if factory:
                _default = factory(root)
            else:
                use_process, use_gpus, use_mlx = worker_options_from_cmd_opts()
                _default = Services.create(
                    root,
                    max_upload_bytes=configured_upload_limit,
                    worker_process=use_process,
                    worker_gpus=use_gpus,
                    worker_mlx=use_mlx,
                )
        return _default
