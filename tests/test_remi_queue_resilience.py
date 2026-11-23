"""The queue worker survives failures of its own store (a full disk, a locked
database) and keeps running jobs once the store answers again."""

from __future__ import annotations

import sqlite3
import threading
import time

from PIL import Image

from remi.files import FileStore
from remi.jobs import JobQueue, JobResult, JobStore, run_inline
from remi.jobs.queue import Submission
from remi.realtime import EventBus
from remi.server.testing import FakeState


def wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_a_store_error_while_claiming_does_not_end_the_worker(tmp_path, monkeypatch):
    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    state = FakeState()
    ran: list[str] = []

    def execute(job, ctx):
        ran.append(job.id)
        return JobResult(images=[Image.new("RGB", (8, 8))], infotexts=["x"], info="x")

    queue = JobQueue(jobs, files, EventBus(), executor=execute, dispatch=run_inline, lock=threading.Lock(), validate=lambda *a: None, state=lambda: state, hash_layout=lambda tab: "h")
    monkeypatch.setattr(JobQueue, "RETRY_DELAY", 0.05)
    claim = jobs.claim_next
    failures = {"left": 1}

    def flaky_claim():
        if failures["left"]:
            failures["left"] -= 1
            raise sqlite3.OperationalError("database or disk is full")
        return claim()

    monkeypatch.setattr(jobs, "claim_next", flaky_claim)
    queue.start()
    try:
        [job] = queue.submit(Submission(tab="txt2img", core={"prompt": "fox"}, scripts={}))
        assert wait_for(lambda: job.id in ran), "the worker stopped after the store failed once"
        assert queue._thread is not None and queue._thread.is_alive()
    finally:
        queue.stop()
        jobs.close()
        files.close()
