"""The generation queue: submission, ordering, and the single worker.

One worker thread takes the next queued job, holds the generation lock
(``modules.call_queue.queue_lock``, shared with ``/sdapi`` and GPU actions)
and runs the job on the WebUI main thread (``modules_forge.main_thread``)
through the named-value entry points. Events go to the :class:`EventBus`.

A claimed job only becomes *current* (``job.started``, progress, previews,
``shared.state`` interrupts) once the worker holds the lock: until then
someone else may be generating. Interrupts and skips that arrive before the
job's ``shared.state.begin()`` are recorded and applied right after it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

from remi.core import JobContext
from remi.files import FileStore
from remi.jobs import completion
from remi.jobs.runner import Executor, JobResult, run_job, run_on_main_thread
from remi.jobs.store import CANCELLED, COMPLETED, FAILED, INTERRUPTED, QUEUED, RUNNING, Job, JobStore
from remi.jobs.texts import DELETED, END_NOT_SAVED, MODEL_NOT_LOADED, OUT_OF_MEMORY
from remi.realtime import EventBus, JobRef
from remi.ui.sentence import Sentence

log = logging.getLogger(__name__)

TABS = ("txt2img", "img2img", "extras")
MAX_REPEAT = 100

Dispatch = Callable[..., Any]
Validator = Callable[[str, Any, Any], tuple[dict[str, Any], dict[str, Any]] | None]
"""Checks ``(tab, core, scripts)``, raising ``remi.ui.ValidationError``; returns
the ``(core, scripts)`` a job keeps, or None to keep them as sent."""

MAX_WARNINGS = 5

RETENTION_DAYS = {"1 day": 1, "3 days": 3, "7 days": 7, "14 days": 14, "30 days": 30, "90 days": 90, "Keep forever": 0}
"""``queue_history_retention_days`` (the Agent Scheduler's choices)"""
RETENTION_INTERVAL = 3600.0
"""seconds between two history clean-ups"""
MAX_ATTEMPTS = 3
"""runs of a failed job, in all, with ``queue_automatic_requeue_failed_task``"""
_SCRIPT_ERROR = re.compile(r"^Error running (\w+): (.+)$")


def _extension_in(error: BaseException | None) -> str | None:
    """The extension of the innermost traceback frame inside one, if any."""

    from remi.ui.manifest import extension_of

    found = None
    trace = error.__traceback__ if error is not None else None
    while trace is not None:
        found = extension_of(trace.tb_frame.f_code.co_filename) or found
        trace = trace.tb_next
    return found


@contextlib.contextmanager
def script_errors(sink: list[dict[str, Any]]) -> Iterator[None]:
    """Collects what ``modules.errors.report`` reports meanwhile (a script
    hook that raised; the run goes on without it) as job warnings: notes
    ``{"message", "values"}`` with an English catalog key. Extensions →
    Diagnostics counts a hook that raised as the extension's hook error, and
    another report from the extension's code as its hook warning."""

    errors = sys.modules.get("modules.errors")
    if errors is None:
        yield
        return
    from remi.ui.manifest import diagnostics, extension_of

    original = errors.report

    def report(message: str, *, exc_info: bool = False) -> None:
        original(message, exc_info=exc_info)
        error = sys.exc_info()[1] if exc_info else None
        detail = f"{type(error).__name__}: {error}" if error is not None else ""
        match = _SCRIPT_ERROR.match(message.splitlines()[0] if message else "")
        if match:
            note = {"message": "{{script}} failed ({{hook}}): {{error}}", "values": {"script": os.path.basename(match[2]), "hook": match[1], "error": detail or "?"}}
            extension = extension_of(match[2])
            if extension is not None:  # Extensions → Diagnostics counts it against the script's extension
                diagnostics.record_hook_error(extension, f"{match[1]}: {detail or '?'}")
        else:
            note = {"message": "{{error}}", "values": {"error": f"{message.splitlines()[0] if message else ''} {detail}".strip()}}
            extension = _extension_in(error)
            if extension is not None:  # an extension's code that reported an error and went on: a warning
                diagnostics.record_hook_warning(extension, note["values"]["error"])
        if note not in sink and len(sink) < MAX_WARNINGS:
            sink.append(note)

    errors.report = report
    try:
        yield
    finally:
        errors.report = original


class RecentSubmissions:
    """The jobs each recent keyed submission queued (``Idempotency-Key``):
    ``(user, key)`` → its jobs, for ``ttl`` seconds and at most ``limit``
    entries (the oldest are forgotten first). In memory: a restart forgets them."""

    def __init__(self, *, ttl: float = 600.0, limit: int = 1000, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl, self.limit, self.clock = ttl, limit, clock
        self._answers: OrderedDict[tuple[str | None, str], tuple[float, list[Job]]] = OrderedDict()

    def _expire(self) -> None:
        now = self.clock()
        while self._answers:
            key, (recorded, _jobs) = next(iter(self._answers.items()))
            if now - recorded < self.ttl:
                break
            del self._answers[key]

    def get(self, user: str | None, key: str) -> list[Job] | None:
        self._expire()
        entry = self._answers.get((user, key))
        return entry[1] if entry is not None else None

    def put(self, user: str | None, key: str, jobs: list[Job]) -> None:
        self._expire()
        self._answers[(user, key)] = (self.clock(), jobs)
        while len(self._answers) > self.limit:
            self._answers.popitem(last=False)


class JobError(ValueError):
    """A queue operation that cannot be done (``status``: HTTP status). Its
    ``message`` is a :class:`Sentence`: the error body carries its template,
    which the UI translates."""

    def __init__(self, message: Sentence, status: int = 409):
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass
class Submission:
    tab: str
    core: dict[str, Any]
    scripts: dict[str, Any]
    priority: int = 0
    repeat: int = 1
    forever: bool = False
    upscale_from: str | None = None
    """txt2img only: Hires. fix this image (a file id) instead of generating
    the first pass (A1111's ✨ button; ``modules.txt2img.run_txt2img_upscale``)"""
    meta: dict[str, Any] | None = None
    """Optional extra job metadata (e.g. ``chain_id``, ``stage_index``, ``upstream_file_ids``)."""


def _state() -> Any:
    from modules import shared

    return shared.state


def _interrupt_after_current() -> bool:
    """The setting "Don't Interrupt in the middle" (``interrupt_after_current``)."""

    return bool(_option("interrupt_after_current", False))


def _option(name: str, default: Any) -> Any:
    """A setting as the kind of ``default`` (``default`` without the WebUI, when
    unset or of another kind)."""

    from remi.setting_values import setting

    opts = getattr(sys.modules.get("modules.shared"), "opts", None)
    return default if opts is None else setting(name, default, opts=opts)


def _set_option(name: str, value: Any) -> None:
    """Changes and saves a setting (left as it is when settings are frozen)."""

    shared = sys.modules.get("modules.shared")
    opts = getattr(shared, "opts", None)
    if opts is None:
        return
    try:
        opts.set(name, value)
        opts.save(shared.config_filename)
    except Exception:
        log.warning("could not reset the setting %s", name, exc_info=True)


def _lock() -> Any:
    from modules.call_queue import queue_lock

    return queue_lock


def layout_hash(tab: str) -> str:
    """A fingerprint of the tab's parameter layout (owners and argument
    names), recorded with each job."""

    from remi.core import forms
    from remi.core.tabs import runner_for

    runner = runner_for(tab)
    parts: list[Any] = [forms.core_layout(tab, runner).names]
    if runner is not None:
        parts += [(script.script_id, runner.layout_for(script).names) for script in runner.scripts]
    else:
        from modules import scripts
        from remi.core.extras import assign_args

        parts += [(layout.owner, layout.names) for _script, layout in assign_args(scripts.scripts_postproc, assign=False)]
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def validate_values(files: FileStore) -> Validator:
    """Checks a submission the way the entry point will (every bad value at
    once, as ``remi.ui.ValidationError``), uploads only by id."""

    def validate(tab: str, core: Any, scripts: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from remi.core.tabs import carried_values, processing_args
        from remi.ui import Context

        processing_args(tab, core, scripts, Context(resolve_upload=files.check_upload), assign_slots=False)
        return carried_values(tab, core, scripts)

    return validate


def _summary(core: dict[str, Any], tab: str) -> str:
    prompt = core.get("prompt") if isinstance(core, dict) else None
    text = prompt if isinstance(prompt, str) and prompt.strip() else tab
    return " ".join(text.split())[:200]


class JobQueue:
    RETRY_DELAY = 2.0
    """Seconds the worker waits after its store failed before it asks again."""

    def __init__(
        self,
        store: JobStore,
        files: FileStore,
        bus: EventBus,
        *,
        executor: Executor = run_job,
        dispatch: Dispatch = run_on_main_thread,
        lock: Any = None,
        validate: Validator | None = None,
        state: Callable[[], Any] = _state,
        hash_layout: Callable[[str], str] = layout_hash,
        option: Callable[[str, Any], Any] = _option,
        set_option: Callable[[str, Any], None] = _set_option,
        when_done: Callable[[str], None] | None = None,
    ) -> None:
        self.store, self.files, self.bus = store, files, bus
        self.executor, self.dispatch = executor, dispatch
        self._lock = lock
        self.validate = validate if validate is not None else validate_values(files)
        self.state = state
        self.hash_layout = hash_layout
        self.option, self.set_option = option, set_option
        self.when_done = when_done if when_done is not None else self._perform_later
        self._spent_action: str | None = None
        """A completion action that ran although the setting could not be reset (frozen settings)"""
        self._retention_checked = 0.0
        self._cv = threading.Condition()
        self._claimed: Job | None = None
        """taken from the queue (maybe still waiting for the lock)"""
        self._current: Job | None = None
        self._done_lock = threading.Lock()
        """holding the lock, generating"""
        self._begun = False
        """``shared.state.begin()`` ran for the current job"""
        self._interrupted: set[str] = set()
        self._after_current: set[str] = set()
        """interrupts that stop after the current image (``interrupt_after_current``)"""
        self._cancelled: set[str] = set()
        self._skips: set[str] = set()
        self._holds = 0
        self._paused = store.get_meta("paused") == "1"
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._loading_model: str | None = None
        self._completion_hooks: list[Callable[[Job], None]] = []
        self._start_hooks: list[Callable[[Job], None]] = []
        self._busy_checks: list[Callable[[], bool]] = []
        self.recent = RecentSubmissions()
        """keyed submissions and their jobs (``submit(key=...)``)"""
        self._keyed = threading.Lock()
        """held across a keyed submission: a second one with the same key waits, then finds the answer"""

    def add_start_hook(self, callback: Callable[[Job], None]) -> Callable[[], None]:
        """Register a callback invoked when a job starts, on the worker thread
        under the generation lock, once the model selection it runs with is applied."""

        with self._cv:
            self._start_hooks.append(callback)

        def remove() -> None:
            with self._cv:
                if callback in self._start_hooks:
                    self._start_hooks.remove(callback)

        return remove

    def add_busy_check(self, check: Callable[[], bool]) -> Callable[[], None]:
        """Register ``check``: work outside the queued jobs (a pipeline's
        inline stage) that keeps the queue from counting as done
        (``queue_completion_action`` waits for it)."""

        with self._cv:
            self._busy_checks.append(check)

        def remove() -> None:
            with self._cv:
                if check in self._busy_checks:
                    self._busy_checks.remove(check)

        return remove

    def _busy_elsewhere(self) -> bool:
        with self._cv:
            checks = list(self._busy_checks)
        for check in checks:
            try:
                if check():
                    return True
            except Exception:
                log.exception("queue busy check")
        return False

    def _notify_started(self, job: Job) -> None:
        with self._cv:
            hooks = list(self._start_hooks)
        for hook in hooks:
            try:
                hook(job)
            except Exception:
                log.exception("job start hook failed for %s", job.id)

    def add_completion_hook(self, callback: Callable[[Job], None]) -> Callable[[], None]:
        """Register a callback invoked whenever a job reaches a final status."""

        with self._cv:
            self._completion_hooks.append(callback)

        def remove() -> None:
            with self._cv:
                if callback in self._completion_hooks:
                    self._completion_hooks.remove(callback)

        return remove

    def _notify_finished(self, job: Job | None) -> None:
        if job is None:
            return
        with self._cv:
            hooks = list(self._completion_hooks)
        for hook in hooks:
            try:
                hook(job)
            except Exception:
                log.exception("job completion hook failed for %s", job.id)

    # region Lifecycle

    @property
    def lock(self) -> Any:
        return self._lock if self._lock is not None else _lock()

    def start(self) -> None:
        self.apply_retention(force=True)
        with self._cv:
            if self._thread is not None and self._thread.is_alive():
                return
            recovered = self.store.recover()
            if recovered:
                log.info("Marked %d job(s) interrupted by the last shutdown", recovered)
            self._stopping = False
            self._register_pending()
            self._thread = threading.Thread(target=self._loop, name="remi-jobs", daemon=True)
            self._thread.start()

    def stop(self, timeout: float | None = 5, *, interrupt: bool = False) -> bool:
        """Stop the worker after the running job (``interrupt``: interrupt
        that job first); returns whether the worker ended in time. Queued
        jobs stay queued (persisted) for the next start."""

        with self._cv:
            self._stopping = True
            self._cv.notify_all()
            job = self._claimed
        if interrupt and job is not None:
            self._signal(job.id, self._interrupted)
        stopped = True
        if self._thread is not None:
            self._thread.join(timeout)
            stopped = not self._thread.is_alive()
        # The process-wide progress registry mirrors a running queue: start() registers them again
        for queued in self.store.queued():
            self._pending_done(queued)
        return stopped

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        """No job starts inside this block (UI reload); waits for the
        running one to finish first."""

        with self._cv:
            self._holds += 1
            while self._claimed is not None:
                self._cv.wait()
        try:
            yield
        finally:
            with self._cv:
                self._holds -= 1
                self._cv.notify_all()

    def wait_idle(self, timeout: float = 10) -> bool:
        """Until nothing runs and nothing runnable is queued (tests)."""

        with self._cv:
            return self._cv.wait_for(lambda: self._claimed is None and (self._paused or self._holds or self.store.count(QUEUED) == 0), timeout)

    # endregion

    # region Queries

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def current(self) -> Job | None:
        return self._current

    @property
    def busy(self) -> bool:
        """A job is running or waiting for the generation lock."""

        return self._claimed is not None

    def current_ref(self) -> JobRef | None:
        job = self._current
        return JobRef(job.id, job.seq, job.tab, self._begun) if job is not None else None

    def status(self) -> dict[str, Any]:
        job = self._current
        return {"size": self.store.count(QUEUED), "paused": self._paused, "running": job.id if job else None}

    # endregion

    # region Submission and control

    def submit(self, submission: Submission, *, user: str | None = None, key: str | None = None) -> list[Job]:
        """Validate and queue (``repeat`` copies); raises
        ``remi.ui.ValidationError`` or :class:`JobError`. A submission with a
        ``key`` (``Idempotency-Key``) that ``user`` sent within the last minutes
        queues nothing and answers the jobs the first one queued, whatever its
        values (a double click, a retry after a lost answer). A refused
        submission is not remembered."""

        if key is None:
            return self._submit(submission, user=user)
        # its own lock, not ``_cv``: the worker claims jobs under ``_cv`` while this one validates
        with self._keyed:
            jobs = self.recent.get(user, key)
            if jobs is None:
                jobs = self._submit(submission, user=user)
                self.recent.put(user, key, jobs)
            return jobs

    def _submit(self, submission: Submission, *, user: str | None) -> list[Job]:
        from remi.ui import ParamError, ValidationError
        from remi.ui.schema import SCHEMA_VERSION

        if submission.tab not in TABS:
            raise ValidationError([ParamError(Sentence("tab is one of {{tabs}}", tabs=", ".join(TABS)))])
        if not 1 <= submission.repeat <= MAX_REPEAT:
            raise ValidationError([ParamError(Sentence("repeat is between 1 and {{maximum}}", maximum=MAX_REPEAT))])
        core, scripts = self._checked(submission.tab, submission.core, submission.scripts)
        meta = dict(submission.meta) if submission.meta else None
        if submission.upscale_from is not None:
            self._check_upscale_source(submission.tab, submission.upscale_from)
            meta = {**(meta or {}), "upscale_from": submission.upscale_from}
        layout = self.hash_layout(submission.tab)
        jobs = [self.store.add(submission.tab, core, scripts, priority=submission.priority, layout_hash=layout, schema_version=SCHEMA_VERSION, summary=_summary(core, submission.tab), forever=submission.forever, user=user, meta=meta) for _ in range(submission.repeat)]
        for job in jobs:
            self._enqueued(job)
        self._changed()
        return jobs

    def _checked(self, tab: str, core: Any, scripts: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validated values as the job keeps them (:data:`Validator`)."""

        kept = self.validate(tab, core, scripts)
        return kept if kept is not None else (core or {}, scripts or {})

    def _check_upscale_source(self, tab: str, file_id: str) -> None:
        from remi.ui import ParamError, ValidationError

        if tab != "txt2img":
            raise ValidationError([ParamError("only txt2img jobs Hires. fix an existing image", param="upscale_from")])
        record = self.files.get(file_id)
        if record is None or record.kind not in ("upload", "output") or not record.is_image:
            raise ValidationError([ParamError(Sentence("unknown image {{id}}", id=repr(file_id)), param="upscale_from")])

    def retry(self, job_id: str, *, user: str | None = None) -> Job:
        job = self.get(job_id)
        if job.status in (QUEUED, RUNNING):
            raise JobError(Sentence("the job has not finished"))
        return self._copy(job, user=user)

    def duplicate(self, job_id: str, *, user: str | None = None) -> Job:
        """Queue a copy of any job: right after it while it is queued, else at the end."""

        job = self.get(job_id)
        return self._copy(job, user=user, after=job.id if job.status == QUEUED else None)

    def _copy(self, job: Job, *, user: str | None, after: str | None = None) -> Job:
        if not job.replayable:
            raise JobError(Sentence("this job has arguments that do not map to the current parameters; open it in its form to queue it again"))
        self.validate(job.tab, job.core, job.scripts)  # the layout or its uploads may have changed
        meta = {key: job.meta[key] for key in ("name", "upscale_from") if job.meta.get(key)}
        new = self.store.add(job.tab, job.core, job.scripts, priority=job.priority, layout_hash=self.hash_layout(job.tab), schema_version=job.schema_version, summary=job.summary, user=user, retry_of=job.id, after=after, meta=meta or None)
        self._enqueued(new)
        self._changed()
        return new

    def update_params(self, job_id: str, core: dict[str, Any], scripts: dict[str, Any]) -> Job:
        """Replace a queued job's parameters (validated like a submission)."""

        from remi.ui.schema import SCHEMA_VERSION

        job = self.get(job_id)
        if job.status != QUEUED:
            raise JobError(Sentence("only queued jobs can be edited"))
        if job.meta.get("chain_id"):
            # its result is cached under the stage's parameters: edit the pipeline instead
            raise JobError(Sentence("this job is a pipeline stage; edit the pipeline and re-run it"))
        core, scripts = self._checked(job.tab, core, scripts)
        updated = self.store.update_params(job_id, core, scripts, layout_hash=self.hash_layout(job.tab), schema_version=SCHEMA_VERSION, summary=_summary(core, job.tab))
        if updated is None:
            raise JobError(Sentence("the job started meanwhile"))
        self._changed()
        return updated

    def set_starred(self, ids: list[str], starred: bool) -> int:
        count = self.store.set_starred(ids, starred)
        self._changed()
        return count

    def clear_history(self, *, keep_starred: bool = True) -> int:
        count = self.store.clear_finished(keep_starred=keep_starred)
        self._changed()
        return count

    def add_imported(self, job: Job, *, source: str, source_id: str) -> Job:
        """Store an imported job (queued ones are announced like submissions)."""

        added = self.store.add_record(job, imported=(source, source_id))
        if added.status == QUEUED:
            self._enqueued(added)
        return added

    def changed(self) -> None:
        """Announce the queue state (after a batch of changes)."""

        self._changed()

    def interrupt(self, job_id: str) -> Job:
        """Stop the running job (keeping what it made) or cancel a queued one."""

        for _attempt in range(2):
            job = self.get(job_id)
            if job.status == RUNNING:
                self._user_interrupt(job_id)
            elif job.status == QUEUED and not self._cancel_queued(job):
                continue  # the worker claimed it between the read and the cancel: interrupt it
            break
        return self.get(job_id)

    def skip(self, job_id: str) -> None:
        if not self._signal(job_id, self._skips):
            raise JobError(Sentence("the job is not running"))

    def delete(self, job_id: str) -> None:
        """Cancel a running job, or remove a queued or finished one."""

        for _attempt in range(3):
            job = self.get(job_id)
            if job.status == RUNNING:
                if self._signal(job_id, self._cancelled):
                    return
                # not on the worker, yet still running in the store: the worker's end of it was
                # never saved (the store failed). It is over; record it so it can go.
                if self.get(job_id).status == RUNNING:
                    log.warning("job %s was left running by a store failure; it is marked interrupted", job_id)
                    message = END_NOT_SAVED
                    ended = self.store.finish(job_id, INTERRUPTED, error=message)
                    self._pending_done(job)  # the worker never got as far as dropping it from /internal/pending-tasks
                    # whoever waits for it (a pipeline stage) learns that it ended, as from the worker
                    self.bus.publish("job.failed", job_id=job.id, seq=job.seq, tab=job.tab, status=INTERRUPTED, error=message, errors=None)
                    self._notify_finished(ended)
            if self.store.delete(job_id):
                if job.status == QUEUED:
                    self._pending_done(job)
                    self._changed()
                    # it never ran: tell whoever waits for it (a pipeline stage) that it ended
                    self._notify_finished(replace(job, status=CANCELLED, error=DELETED))
                return
            # the worker claimed it between the read and the delete (the store keeps running
            # jobs): cancel it as the running job it now is
        raise JobError(Sentence("the job changed while it was being removed; try again"))

    def update(self, job_id: str, *, priority: int | None = None, forever: bool | None = None) -> Job:
        job = self.get(job_id)
        if forever and job.meta.get("chain_id"):
            # its copies would carry no chain and requeue each other on their own
            raise JobError(Sentence("a pipeline stage cannot generate forever"))
        job = self.store.update(job_id, priority=priority, forever=forever)
        self._changed()
        return job

    def reorder(self, ids: list[str]) -> None:
        self.store.reorder(ids)
        self._changed()

    def move(self, job_id: str, *, after: str | None = None, before: str | None = None, priority: int | None = None) -> None:
        """Move a queued job next to a neighbour (see :meth:`JobStore.move`)."""

        self.get(job_id)
        if not self.store.move(job_id, after=after, before=before, priority=priority):
            raise JobError(Sentence("the job is not queued any more"))
        self._changed()

    def interrupt_all(self, *, clear_queue: bool = False, stop_forever: bool = True) -> None:
        """Interrupt whatever generates (also ``/sdapi`` requests)."""

        if stop_forever:
            self.store.clear_forever()
        if clear_queue:
            for job in self.store.queued():
                self._cancel_queued(job, notify=False)
        job = self._claimed
        if job is not None:
            self._user_interrupt(job.id)
        if job is None or self._current is None:
            self.state().interrupt()  # an /sdapi request or a GPU action
        self._changed()

    @contextlib.contextmanager
    def interruptible(self) -> Iterator[Callable[[], bool]]:
        """Around GPU work that is no job but that Interrupt stops (a local prompt upsampler):
        the caller holds :attr:`lock`, so :meth:`interrupt_all` flips the state for it. As a
        job's ``shared.state.begin()`` and end do, an interrupt left from before is cleared first
        (one pressed while nothing ran must not stop it at once) and the flag again afterwards
        (it must not reach whatever runs next). Yields whether Interrupt was pressed since."""

        def interrupted() -> bool:
            return bool(getattr(self.state(), "interrupted", False))

        def clear() -> None:
            state = self.state()
            if state is not None:
                state.interrupted = False

        clear()
        try:
            yield interrupted
        finally:
            clear()

    def set_paused(self, paused: bool) -> None:
        with self._cv:
            self._paused = paused
            self.store.set_meta("paused", "1" if paused else "0")
            self._cv.notify_all()
        self._changed()

    def stop_forever(self) -> int:
        count = self.store.clear_forever()
        self._changed()
        return count

    # endregion

    # region Internals

    def get(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise JobError(Sentence("no such job"), 404)
        return job

    def _signal(self, job_id: str, bucket: set[str]) -> bool:
        """Record an interrupt/cancel (or skip, ``bucket`` ``_skips``) for
        the claimed job; flip ``shared.state`` now if the job has begun,
        else :meth:`_on_begin` does. False when ``job_id`` is not claimed."""

        with self._cv:
            job = self._claimed
            if job is None or job.id != job_id:
                return False
            bucket.add(job_id)
            if self._begun and self._current is job:
                self._apply_signals(job_id)
            return True

    def _user_interrupt(self, job_id: str) -> None:
        """Interrupt from the user. With ``interrupt_after_current`` the first
        one lets the image being made finish when the job makes more than
        one (A1111's "Don't Interrupt in the middle"); a second one stops at
        once."""

        with self._cv:
            if job_id not in self._interrupted and _interrupt_after_current():
                self._after_current.add(job_id)
        self._signal(job_id, self._interrupted)

    def _apply_signals(self, job_id: str) -> None:
        state = self.state()
        after_current = job_id in self._after_current
        self._after_current.discard(job_id)
        if job_id in self._interrupted and job_id not in self._cancelled and after_current and getattr(state, "job_count", 0) > 1 and not getattr(state, "stopping_generation", False):
            state.stop_generating()
        elif job_id in self._interrupted or job_id in self._cancelled:
            state.interrupt()
        elif job_id in self._skips:
            self._skips.discard(job_id)
            state.skip()

    def _on_begin(self, job_id: str) -> None:
        """``JobContext.on_begin``: after ``shared.state.begin()`` reset the
        flags, apply what arrived before."""

        with self._cv:
            self._begun = True
            self._apply_signals(job_id)

    def _enqueued(self, job: Job) -> None:
        self._progress_pending(job)
        self.bus.publish("job.queued", job_id=job.id, seq=job.seq, tab=job.tab, position=self.store.count(QUEUED))
        with self._cv:
            self._cv.notify_all()

    def _cancel_queued(self, job: Job, *, notify: bool = True) -> bool:
        """False when ``job`` is not queued any more (the worker claimed it)."""

        if not self.store.cancel_queued(job.id):
            return False
        self._pending_done(job)
        self.bus.publish("job.failed", job_id=job.id, seq=job.seq, tab=job.tab, status=CANCELLED, error="Cancelled")
        self._notify_finished(self.store.get(job.id))
        if notify:
            self._changed()
        return True

    def _changed(self) -> None:
        self.bus.publish("queue.changed", **self.status())

    def _register_pending(self) -> None:
        for job in self.store.queued():
            self._progress_pending(job)

    @staticmethod
    def _progress_pending(job: Job) -> None:
        progress = sys.modules.get("modules.progress")
        if progress is not None:
            progress.add_task_to_queue(job.id_task)

    @staticmethod
    def _pending_done(job: Job) -> None:
        progress = sys.modules.get("modules.progress")
        if progress is not None:
            progress.pending_tasks.pop(job.id_task, None)

    def _runnable(self) -> bool:
        return self._stopping or (not self._paused and not self._holds and self.store.count(QUEUED) > 0)

    def _loop(self) -> None:
        while True:
            try:
                if not self._step():
                    return
            except Exception:  # the store failed (a full disk, a locked database): ask again later
                log.exception("job queue")
                with self._cv:
                    if self._stopping:
                        return
                    self._cv.wait(self.RETRY_DELAY)

    def _step(self) -> bool:
        """Claim and run the next job; ``False`` once the queue stops."""

        with self._cv:
            self._cv.wait_for(self._runnable)
            if self._stopping:
                return False
            job = self.store.claim_next()
            if job is not None:
                self._claimed = job
        if job is None:  # it cancelled queued jobs it cannot run: nothing is queued now
            self._changed()
            return True
        try:
            with self.lock:  # behind /sdapi requests and GPU actions
                self._run(job)
        except BaseException:  # the worker outlives anything a job raises (SystemExit included)
            log.exception("job %s", job.id)
        finally:
            with self._cv:
                self._claimed = self._current = None
                self._begun = False
                self._interrupted.discard(job.id)
                self._after_current.discard(job.id)
                self._cancelled.discard(job.id)
                self._skips.discard(job.id)
                self._cv.notify_all()
        self._changed()
        self._after_job(job.id)
        return True

    def _run(self, job: Job) -> None:
        """Run ``job``; the caller holds the generation lock."""

        from remi.ui import ValidationError

        ref = JobRef(job.id, job.seq, job.tab)
        with self._cv:
            stopped = job.id in self._interrupted or job.id in self._cancelled
            if not stopped:
                self._current = job
        if stopped:  # interrupted while waiting for the lock: nothing made
            status = CANCELLED if job.id in self._cancelled else INTERRUPTED
            finished = self.store.finish(job.id, status)
            self._pending_done(job)
            self.bus.publish("job.completed", **ref.fields(), status=status, outputs=[], infotext=None)
            self._notify_finished(finished)
            return
        from remi import model_selection

        model_selection.apply_held()  # picked while the previous job ran: this job loads it
        self._notify_started(job)
        self.bus.publish("job.started", **ref.fields())
        self._changed()
        ctx = JobContext(id_task=job.id_task, user=job.user, resolve_upload=self.files.resolve_upload, on_begin=lambda: self._on_begin(job.id))
        self._note_model_loading()
        warnings: list[dict[str, Any]] = []
        try:
            with script_errors(warnings):
                result = self.dispatch(self.executor, job, ctx)
        except ValidationError as error:
            self._fail(job, str(error), errors=error.to_json()["errors"])
            return
        except BaseException as error:  # an extension's sys.exit() fails its job, not the queue
            self._fail(job, self._error_message(error))
            return
        finally:
            # The executor starts the progress task; one that failed (or never got there) must not stay "pending".
            self._pending_done(job)
            self.poll()
        # an interrupt past the queue (/sdapi/v1/interrupt, an extension) flips the state only: the
        # job's context kept it (the state's flag is cleared when the job ends)
        interrupted = job.id in self._interrupted or ctx.interrupted
        status = CANCELLED if job.id in self._cancelled else INTERRUPTED if interrupted else COMPLETED
        outputs = self._register(job, result or JobResult())
        for note in getattr(result, "warnings", None) or []:
            if note not in warnings and len(warnings) < MAX_WARNINGS:
                warnings.append(note)
        info = (result.info if result else None) or next((output.get("infotext") for output in outputs if output.get("infotext")), None)
        finished = self.store.finish(job.id, status, outputs=outputs, infotext=info, warnings=warnings)
        self.bus.publish("job.completed", **ref.fields(), status=status, outputs=outputs, infotext=info, **({"warnings": warnings} if warnings else {}))
        self._notify_finished(finished)
        if status == COMPLETED:  # reads the flag again: it may have been switched on while the job ran
            again = self.store.requeue_forever(job.id)
            if again is not None:
                self._enqueued(again)

    def _fail(self, job: Job, message: str, *, errors: list[dict] | None = None) -> None:
        from remi.image_meta import utf8_text

        message = utf8_text(message)  # an exception naming a file that is not UTF-8: SQLite cannot store it
        status = CANCELLED if job.id in self._cancelled else FAILED
        finished = self.store.finish(job.id, status, error=message, errors=errors)
        self.bus.publish("job.failed", job_id=job.id, seq=job.seq, tab=job.tab, status=status, error=message, errors=errors)
        self._notify_finished(finished)
        if status == FAILED and not errors and self.option("queue_automatic_requeue_failed_task", False):
            self._requeue_failed(job)

    def _requeue_failed(self, job: Job) -> None:
        """Queue a failed job again at the end (``queue_automatic_requeue_failed_task``),
        until it ran :data:`MAX_ATTEMPTS` times. Parameters the form rejects
        (``errors``) fail the same way every time: not retried."""

        attempt = job.meta.get("attempt", 1)
        attempt = attempt if isinstance(attempt, int) and attempt > 0 else 1
        if attempt >= MAX_ATTEMPTS or not job.replayable or job.meta.get("chain_id"):
            return  # a failed pipeline stage failed its chain: a retry would belong to nothing
        try:
            self.validate(job.tab, job.core, job.scripts)
        except Exception:
            return
        meta = {key: job.meta[key] for key in ("name", "upscale_from") if job.meta.get(key)}
        # generate-forever as it is now: `job` is the run as it was claimed, before any stop
        again = self.store.requeue_failed(job, {**meta, "attempt": attempt + 1})
        log.info("Queued failed job #%d again as #%d (attempt %d of %d)", job.seq, again.seq, attempt + 1, MAX_ATTEMPTS)
        self._enqueued(again)

    def apply_retention(self, *, force: bool = False, now: float | None = None) -> int:
        """Deletes finished jobs older than ``queue_history_retention_days``
        (starred ones are kept; at most once an hour unless ``force``);
        returns how many went."""

        now = time.time() if now is None else now
        if not force and now - self._retention_checked < RETENTION_INTERVAL:
            return 0
        self._retention_checked = now
        days = RETENTION_DAYS.get(str(self.option("queue_history_retention_days", "Keep forever")), 0)
        if days <= 0:
            return 0
        count = self.store.clear_finished(keep_starred=True, before=now - days * 86400)
        if count:
            log.info("Deleted %d finished job(s) older than %d days from the queue history", count, days)
            self._changed()
        return count

    def _after_job(self, job_id: str) -> None:
        self.apply_retention()
        with self._cv:
            done = not self._stopping and self.store.count(QUEUED) == 0
        last = self.store.get(job_id)
        # someone stopping the run by hand is at the computer
        if done and (last is None or last.status not in (INTERRUPTED, CANCELLED)) and not self._busy_elsewhere():
            self._queue_done()

    def recheck_done(self) -> None:
        """Work outside the queued jobs ended (a pipeline's last stage or its
        failure, off the worker): run ``queue_completion_action`` if that left
        the queue done. A job on the worker is left to ``_after_job``."""

        if self._idle():
            self._queue_done()

    def _queue_done(self) -> None:
        """The last queued job finished: ``queue_completion_action``, once
        (the setting goes back to "Do nothing", see remi/jobs/completion.py)."""

        with self._done_lock:  # the worker and a pipeline's upsampling thread may both get here
            action = str(self.option("queue_completion_action", completion.NOTHING))
            if action == completion.NOTHING or action not in completion.ACTIONS or action == self._spent_action:
                return
            self.set_option("queue_completion_action", completion.NOTHING)
            # frozen settings keep the action: run it once per process, not after every run
            self._spent_action = action if self.option("queue_completion_action", completion.NOTHING) == action else None
        self.when_done(action)

    def _idle(self) -> bool:
        with self._cv:
            idle = not self._stopping and self._claimed is None and self._current is None and self.store.count(QUEUED) == 0
        return idle and not self._busy_elsewhere()

    def _perform_later(self, action: str) -> None:
        """The action after a short delay, unless a job started meanwhile (it then waits again)."""

        def rearm(action: str) -> None:
            with self._done_lock:
                current = str(self.option("queue_completion_action", completion.NOTHING))
                # another action is the user's pick meanwhile: theirs stands. The same one is
                # frozen settings keeping it (``_spent_action``): it has not run, so it is armed again.
                if current not in (completion.NOTHING, self._spent_action):
                    return
                self._spent_action = None
                if current == completion.NOTHING:
                    self.set_option("queue_completion_action", action)
            # the job that kept it waiting may have ended before the action was back: nothing
            # would run it then, so look again
            try:
                idle = self._idle()
            except Exception:
                log.exception("checking whether the queue is done")
                return
            if idle:
                self._queue_done()

        completion.perform_later(action, still_done=self._idle, rearm=rearm)

    @staticmethod
    def _error_message(error: BaseException) -> str:
        try:
            from backend.memory_management import is_oom
        except Exception:  # no backend (tests)
            is_oom = None
        if is_oom is not None and is_oom(error):
            try:
                from modules import sd_models

                sd_models.unload_model_weights()
            except Exception:
                log.exception("unloading the model after running out of memory")
            return OUT_OF_MEMORY
        log.error("job failed", exc_info=error)
        return f"{type(error).__name__}: {error}"

    def _register(self, job: Job, result: JobResult) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for index, image in enumerate(result.images):
            infotext = result.infotexts[index] if index < len(result.infotexts) else result.info
            try:
                saved = getattr(image, "already_saved_as", None)
                if saved and os.path.isfile(saved):
                    record = self.files.register_output(saved, job_id=job.id, tab=job.tab, infotext=infotext, width=image.width, height=image.height)
                else:
                    record = self.files.save_output_image(image, infotext, job_id=job.id, tab=job.tab)
            except Exception:
                log.exception("registering an output of job %s", job.id)
                continue
            role = result.roles[index] if index < len(result.roles) else None
            outputs.append({"id": record.id, "kind": "image", "width": record.width, "height": record.height, "infotext": infotext, **({"role": role} if role else {})})
        for path in result.files:
            try:
                record = self.files.register_output(path, job_id=job.id, tab=job.tab, infotext=result.info)
            except OSError:
                continue
            outputs.append({"id": record.id, "kind": "video" if record.mime.startswith("video/") else "file", "infotext": result.info})
        return outputs

    # endregion

    # region Model loading events

    def _note_model_loading(self) -> None:
        model = pending_model()
        if model is not None:
            self._loading_model = model
            self.bus.publish("model.loading", stage="loading", model=model)

    def poll(self) -> None:
        """Announce the end of a model load started by a job (called by the
        progress monitor and after each job)."""

        model = self._loading_model
        if model is None:
            return
        if pending_model() is None:
            self._loading_model = None
            self.bus.publish("model.loaded", model=model)
        elif self._current is None:
            self._loading_model = None
            self.bus.publish("model.loaded", model=model, error=MODEL_NOT_LOADED)

    # endregion


def pending_model() -> str | None:
    """The checkpoint the next generation will load, if it differs from the
    loaded one."""

    from remi import model_selection

    if (held := model_selection.held().get("sd_model_checkpoint")) is not None:
        return str(held)  # picked during the running job, loaded by the next one
    sd_models = sys.modules.get("modules.sd_models")
    data = getattr(sd_models, "model_data", None)
    params = getattr(data, "forge_loading_parameters", None)
    if not params or str(params) == getattr(data, "forge_hash", None):
        return None
    info = params.get("checkpoint_info") if isinstance(params, dict) else None
    return str(getattr(info, "title", None) or getattr(info, "name", None) or "model")
