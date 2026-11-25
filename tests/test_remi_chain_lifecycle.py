"""Pipeline (chain) regressions: stage-cache keys, the chain state machine
across every way a stage job can end, restart recovery and stage inputs."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from PIL import Image

from remi.files import FileStore
from remi.jobs import ChainRunner, ChainStore, JobError, JobQueue, JobResult, JobStore, StageSpec, run_inline
from remi.jobs.chains import ChainError
from remi.realtime import EventBus
from remi.server.testing import FakeState


@dataclass
class Rig:
    files: FileStore
    jobs: JobStore
    queue: JobQueue
    runner: ChainRunner
    bus: EventBus
    model: dict[str, str]
    executed: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def wait(self, chain_id: str, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chain = self.runner.store.get_chain(chain_id)
            if chain is not None and chain.status in ("completed", "failed", "cancelled"):
                return chain
            time.sleep(0.02)
        return self.runner.store.get_chain(chain_id)

    def statuses(self, chain_id: str) -> list[str]:
        chain = self.runner.store.get_chain(chain_id)
        assert chain is not None
        return [stage.status for stage in chain.stages]

    def new_runner(self) -> ChainRunner:
        """Another runner on the same databases (what a restart starts with)."""

        self.runner.close()
        self.runner = ChainRunner(ChainStore(self.jobs), self.queue, self.files, self.bus, upsampler=lambda s, p, o: f"{p or s.core.get('prompt')}+", model_fingerprint=lambda: self.model["fp"])
        return self.runner


@contextmanager
def rig(tmp_path, *, upsampler=None, option=None, start: bool = True) -> Iterator[Rig]:
    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    bus = EventBus()
    state = FakeState()
    model = {"fp": "model-a"}
    holder: dict[str, Rig] = {}

    def execute(job, ctx):
        r = holder["rig"]
        r.executed.append({"kind": job.meta.get("stage_kind"), "core": dict(job.core), "model": model["fp"]})
        if r.result.get("during") is not None:
            r.result["during"]()
        if r.result.get("images") is not None:
            return JobResult(images=r.result["images"], infotexts=["x"] * len(r.result["images"]), info="x", roles=r.result.get("roles", []))
        color = (len(r.executed) * 40 % 255, 10, 20)
        return JobResult(images=[Image.new("RGB", (8, 8), color)], infotexts=["x"], info="x")

    options = {"option": option} if option is not None else {}
    queue = JobQueue(jobs, files, bus, executor=execute, dispatch=run_inline, lock=threading.Lock(), validate=lambda *a: None, state=lambda: state, hash_layout=lambda tab: "h", **options)
    runner = ChainRunner(ChainStore(jobs), queue, files, bus, upsampler=upsampler or (lambda s, p, o: f"{p or s.core.get('prompt')}+"), model_fingerprint=lambda: model["fp"])
    r = Rig(files, jobs, queue, runner, bus, model)
    holder["rig"] = r
    bus.subscribe(type("Sub", (), {"send_event": lambda self, e: r.events.append(e), "send_frame": lambda self, f: None})())
    # a queue that never started would leave its jobs in the process-wide progress registry;
    # "not started" here is a paused worker
    queue.start()
    if not start:
        queue.set_paused(True)
    try:
        yield r
    finally:
        queue.stop()
        r.runner.close()
        jobs.close()
        files.close()


def seeded(seed: int | None, prompt: str = "fox") -> list[StageSpec]:
    scripts = {"seed": {"seed": seed}} if seed is not None else {}
    return [StageSpec(kind="generate", tab="txt2img", core={"prompt": prompt}, scripts=scripts), StageSpec(kind="refine", tab="img2img", core={}, scripts=scripts)]


# region Stage-cache keys


def test_a_random_seed_reruns_in_a_new_chain_but_not_in_its_own_rerun(tmp_path):
    with rig(tmp_path) as r:
        first = r.wait(r.runner.create_chain(seeded(-1)).id)
        assert first.status == "completed"
        second = r.wait(r.runner.create_chain(seeded(-1)).id)
        assert [s.status for s in second.stages] == ["completed", "completed"]  # new images, not the old ones
        again = r.wait(r.runner.rerun_chain(first.id).id)
        assert [s.status for s in again.stages] == ["cached", "cached"]
        assert again.stages[1].output_file_ids == first.stages[1].output_file_ids
        assert len(r.executed) == 4


def test_no_seed_counts_as_random_and_a_fixed_seed_is_shared(tmp_path):
    with rig(tmp_path) as r:
        r.wait(r.runner.create_chain(seeded(None)).id)
        assert [s.status for s in r.wait(r.runner.create_chain(seeded(None)).id).stages] == ["completed", "completed"]
        r.wait(r.runner.create_chain(seeded(7)).id)
        assert [s.status for s in r.wait(r.runner.create_chain(seeded(7)).id).stages] == ["cached", "cached"]


def test_a_random_variation_seed_only_counts_with_a_strength(tmp_path):
    from remi.jobs.chains import uses_random_seed

    assert not uses_random_seed(StageSpec(kind="generate", scripts={"seed": {"seed": 3, "subseed": -1, "subseed_strength": 0}}))
    assert uses_random_seed(StageSpec(kind="generate", scripts={"seed": {"seed": 3, "subseed": -1, "subseed_strength": 0.4}}))
    assert not uses_random_seed(StageSpec(kind="finish", tab="extras"))  # upscaling draws no seed
    assert uses_random_seed(StageSpec(kind="generate", core={"seed": "-1"}))


def test_a_rerun_after_a_model_switch_runs_again(tmp_path):
    with rig(tmp_path) as r:
        chain = r.wait(r.runner.create_chain(seeded(5)).id)
        r.model["fp"] = "model-b"
        again = r.wait(r.runner.rerun_chain(chain.id).id)
        assert [s.status for s in again.stages] == ["completed", "completed"]
        assert r.executed[-1]["model"] == "model-b"


def test_outputs_are_cached_under_the_model_the_job_ran_on(tmp_path):
    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5)[:1])
        r.model["fp"] = "model-b"  # picked while the stage waited in the queue
        r.queue.set_paused(False)
        assert r.wait(chain.id).status == "completed"
        assert r.executed[0]["model"] == "model-b"
        assert [p["cache_hit"] for p in r.runner.plan_chain(seeded(5)[:1])] == [True]
        r.model["fp"] = "model-a"
        assert [p["cache_hit"] for p in r.runner.plan_chain(seeded(5)[:1])] == [False]


def test_a_setting_changed_while_the_job_runs_does_not_rekey_its_output(tmp_path):
    with rig(tmp_path) as r:
        r.result["during"] = lambda: r.model.update(fp="model-b")  # e.g. clip skip changed mid-run
        assert r.wait(r.runner.create_chain(seeded(5)[:1]).id).status == "completed"
        assert [p["cache_hit"] for p in r.runner.plan_chain(seeded(5)[:1])] == [False]
        r.model["fp"] = "model-a"
        assert [p["cache_hit"] for p in r.runner.plan_chain(seeded(5)[:1])] == [True]


def test_a_client_fingerprint_is_salt_not_a_substitute(tmp_path):
    with rig(tmp_path) as r:
        r.wait(r.runner.create_chain(seeded(5), model_fingerprint="client").id)
        assert [p["cache_hit"] for p in r.runner.plan_chain(seeded(5), model_fingerprint="client")] == [True, True]
        r.model["fp"] = "model-b"  # the client still claims the same fingerprint
        assert [p["cache_hit"] for p in r.runner.plan_chain(seeded(5), model_fingerprint="client")] == [False, False]


def test_the_plan_matches_the_run_around_a_disabled_stage(tmp_path):
    stages = [StageSpec(kind="generate", tab="txt2img", core={"prompt": "p"}, scripts={"seed": {"seed": 1}}), StageSpec(kind="refine", enabled=False), StageSpec(kind="finish", tab="extras", core={})]
    with rig(tmp_path) as r:
        r.wait(r.runner.create_chain(stages).id)
        planned = [p["cache_hit"] for p in r.runner.plan_chain(stages)]
        actual = r.statuses(r.wait(r.runner.create_chain(stages).id).id)
        assert planned == [True, False, True]
        assert actual == ["cached", "skipped", "cached"]


def test_editing_a_queued_stage_job_is_refused(tmp_path):
    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5)[:1])
        job_id = r.runner.store.get_chain(chain.id).stages[0].job_id
        with pytest.raises(JobError, match="pipeline stage"):
            r.queue.update_params(job_id, {"prompt": "other"}, {})


# endregion

# region Chain state machine


def test_resume_while_the_stage_job_waits_finishes_the_chain(tmp_path):
    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5))
        assert r.runner.pause_chain(chain.id).status == "paused"
        assert r.runner.resume_chain(chain.id).status == "running"
        r.queue.set_paused(False)
        done = r.wait(chain.id)
        assert done.status == "completed", r.statuses(chain.id)


def test_a_paused_chain_waits_for_its_resume(tmp_path):
    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5))
        r.runner.pause_chain(chain.id)
        r.queue.set_paused(False)
        assert r.queue.wait_idle(timeout=5)
        assert r.runner.store.get_chain(chain.id).status == "paused"
        assert r.statuses(chain.id) == ["completed", "pending"]
        r.runner.resume_chain(chain.id)
        assert r.wait(chain.id).status == "completed"


def test_deleting_a_queued_stage_job_cancels_the_chain(tmp_path):
    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5))
        r.queue.delete(r.runner.store.get_chain(chain.id).stages[0].job_id)
        done = r.runner.store.get_chain(chain.id)
        assert done.status == "cancelled" and "deleted" in (done.error or "")
        assert r.statuses(chain.id) == ["cancelled", "pending"]
        assert r.runner.rerun_chain(chain.id).status == "running"  # not stuck: it can run again


def test_cancelling_a_finished_chain_is_refused(tmp_path):
    with rig(tmp_path) as r:
        chain = r.wait(r.runner.create_chain(seeded(5)[:1]).id)
        finished = chain.finished
        with pytest.raises(ChainError) as error:
            r.runner.cancel_chain(chain.id)
        assert error.value.status == 409
        again = r.runner.store.get_chain(chain.id)
        assert again.status == "completed" and again.finished == finished


def test_a_failed_stage_job_is_not_requeued_on_its_own(tmp_path):
    def option(name, default):
        return True if name == "queue_automatic_requeue_failed_task" else default

    with rig(tmp_path, option=option) as r:
        r.queue.executor = lambda job, ctx: (_ for _ in ()).throw(RuntimeError("gpu oom"))
        chain = r.wait(r.runner.create_chain(seeded(5)[:1]).id)
        assert chain.status == "failed"
        assert r.queue.wait_idle(timeout=5)
        assert r.jobs.count("failed") == 1  # no orphan retries ran after the chain had failed


def test_chain_updated_carries_the_whole_chain(tmp_path):
    with rig(tmp_path) as r:
        chain = r.wait(r.runner.create_chain(seeded(5)[:1]).id)
        last = [e for e in r.events if e["type"] == "chain.updated" and e["chain_id"] == chain.id][-1]
        assert last["id"] == chain.id and last["status"] == "completed"
        assert last["updated_at"] and last["created_at"] and last["stages"][0]["output_file_ids"]


# endregion

# region Restart recovery


def test_reconcile_cancels_a_chain_whose_job_the_restart_interrupted(tmp_path):
    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5))
        job_id = r.runner.store.get_chain(chain.id).stages[0].job_id
        r.jobs.claim_next()  # the previous process was running it
        r.queue._pending_done(r.jobs.get(job_id))  # (and took it off the process-wide progress registry)
        r.jobs.recover()
        assert r.new_runner().reconcile() == 1
        done = r.runner.store.get_chain(chain.id)
        assert done.status == "cancelled" and r.statuses(chain.id) == ["cancelled", "pending"]
        assert job_id


def test_reconcile_records_a_job_that_finished_unseen_and_goes_on(tmp_path):
    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5))
        stage_job = r.runner.store.get_chain(chain.id).stages[0].job_id
        output = r.files.save_output_image(Image.new("RGB", (8, 8)), "x", job_id=stage_job, tab="txt2img")
        r.jobs.claim_next()
        r.queue._pending_done(r.jobs.get(stage_job))  # the previous process ran it to the end
        r.jobs.finish(stage_job, "completed", outputs=[{"id": output.id, "kind": "image"}], infotext="x")
        r.new_runner().reconcile()
        assert r.statuses(chain.id) == ["completed", "running"]
        r.queue.set_paused(False)
        assert r.wait(chain.id).status == "completed"


def test_reconcile_cancels_an_upsample_nobody_runs_and_a_vanished_job(tmp_path):
    with rig(tmp_path, start=False) as r:
        pup = r.runner.store.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "x"})], status="running")
        r.runner.store.update_stage(pup.id, 0, status="running")
        gone = r.runner.store.create_chain(seeded(5), status="running")
        r.runner.store.update_stage(gone.id, 0, status="running", job_id="no-such-job")
        queued = r.runner.store.create_chain(seeded(5), status="queued")
        assert r.new_runner().reconcile() == 3
        assert r.runner.store.get_chain(pup.id).status == "cancelled"
        assert r.runner.store.get_chain(gone.id).status == "cancelled"
        assert r.statuses(queued.id)[0] == "running"  # submitted: it goes on


def test_reconcile_cancels_a_paused_chain_whose_job_the_restart_interrupted(tmp_path):
    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5))
        job_id = r.runner.store.get_chain(chain.id).stages[0].job_id
        r.runner.pause_chain(chain.id)
        r.jobs.claim_next()
        r.queue._pending_done(r.jobs.get(job_id))
        r.jobs.recover()
        resting = r.runner.store.create_chain(seeded(5), status="paused")
        assert r.new_runner().reconcile() == 1
        assert r.runner.store.get_chain(chain.id).status == "cancelled"
        assert r.runner.store.get_chain(resting.id).status == "paused"  # nothing ran: it waits


def test_forced_stages_survive_a_restart(tmp_path):
    with rig(tmp_path) as r:
        chain = r.wait(r.runner.create_chain(seeded(5)).id)
        r.queue.set_paused(True)
        r.runner.rerun_chain(chain.id, from_stage=1)
        assert r.runner.store.get_forced(chain.id) == {1}
        r.new_runner()
        r.queue.set_paused(False)
        done = r.wait(chain.id)
        assert [s.status for s in done.stages] == ["cached", "completed"]
        assert r.runner.store.get_forced(chain.id) == set()


# endregion

# region Stage inputs and prompt upsampling


def test_the_next_stage_gets_the_image_not_the_grid_or_extras(tmp_path):
    with rig(tmp_path) as r:
        grid, image, extra = (Image.new("RGB", (8, 8), c) for c in ((1, 1, 1), (2, 2, 2), (3, 3, 3)))
        r.result.update(images=[grid, image, extra], roles=["grid", None, "extra"])
        chain = r.wait(r.runner.create_chain(seeded(5)).id)
        assert chain.status == "completed"
        assert len(chain.stages[0].output_file_ids) == 1
        assert r.executed[1]["core"]["init_img"] == chain.stages[0].output_file_ids[0]


def test_a_stage_without_an_image_fails_the_chain(tmp_path):
    with rig(tmp_path) as r:
        r.result.update(images=[])
        chain = r.wait(r.runner.create_chain(seeded(5)).id)
        assert chain.status == "failed" and "no image" in (chain.error or "")
        assert [s.status for s in chain.stages] == ["failed", "pending"]
        assert len(r.executed) == 1


def test_upsampling_with_a_local_model_runs_behind_the_generation_lock_and_off_the_request(tmp_path):
    seen: dict[str, Any] = {}
    gate = threading.Event()

    def upsampler(stage, prompt, outputs):
        seen["locked"] = r.queue.lock.locked()
        seen["thread"] = threading.current_thread().name
        gate.wait(5)
        return "big prompt"

    with rig(tmp_path, upsampler=upsampler) as r:
        chain = r.runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "x"}, config={"local_model_path": "/models/llm"}), *seeded(5)[:1]])
        assert chain.status == "running"  # the request did not wait for the upsampler
        assert r.runner.busy()  # no stage job is queued yet: the queue is not done
        gate.set()
        done = r.wait(chain.id)
        assert done.status == "completed"
        assert not r.runner.busy()
        assert seen["locked"] is True and seen["thread"].startswith("remi-chain-upsample")
        assert r.executed[0]["core"]["prompt"] == "fox"  # the stage's own prompt wins


def test_upsampling_through_an_endpoint_leaves_the_generation_lock_free(tmp_path):
    """A stage that asks the prompt tools' endpoint only waits on the network: it must not hold
    the queue's lock (no job could start) or the main thread for the whole request."""

    seen: dict[str, Any] = {}

    def upsampler(stage, prompt, outputs):
        seen["locked"] = r.queue.lock.locked()
        seen["thread"] = threading.current_thread().name
        return "big prompt"

    with rig(tmp_path, upsampler=upsampler) as r:
        chain = r.runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "x"}), *seeded(5)[:1]])
        assert r.wait(chain.id).status == "completed"
        assert seen["locked"] is False and seen["thread"].startswith("remi-chain-upsample")


def _wait_for(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.mark.parametrize("stale", ["fails", "answers"])
def test_a_cancelled_upsampling_request_that_answers_late_leaves_the_rerun_alone(tmp_path, stale):
    """Cancel while the endpoint is slow, then re-run: the cancelled request's late failure must not
    fail the re-run, and its late answer must not stand in for the re-run's own."""

    gates: list[threading.Event] = []

    def upsampler(stage, prompt, outputs):
        gate = threading.Event()
        gates.append(gate)
        first = len(gates) == 1
        assert gate.wait(10)
        if first and stale == "fails":
            raise TimeoutError("endpoint timed out")
        return "stale prompt" if first else "fresh prompt"

    with rig(tmp_path, upsampler=upsampler) as r:
        chain = r.runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "x"}), *seeded(5)[:1]])
        _wait_for(lambda: len(gates) == 1)
        r.runner.cancel_chain(chain.id)
        r.runner.rerun_chain(chain.id, force_stages=[0])
        _wait_for(lambda: len(gates) == 2)
        gates[0].set()  # the cancelled request answers while the re-run's is still going
        _wait_for(lambda: len(r.runner._threads) == 1)  # the cancelled run's thread is done
        middle = r.runner.store.get_chain(chain.id)
        assert middle.status == "running" and middle.stages[0].status == "running"
        gates[1].set()
        done = r.wait(chain.id)
        assert done.status == "completed"
        assert done.stages[0].output_prompt == "fresh prompt"


def completion_action(queue, action: str) -> list[str]:
    """Arm ``queue_completion_action``; returns the actions the queue ran."""

    values, ran = {"queue_completion_action": action}, []
    queue.option = lambda name, default: values.get(name, default)
    queue.set_option = values.__setitem__
    queue.when_done = ran.append
    return ran


def test_a_pipeline_that_ends_in_upsampling_runs_the_completion_action(tmp_path):
    with rig(tmp_path) as r:
        ran = completion_action(r.queue, "Shut down")
        chain = r.runner.create_chain([*seeded(5)[:1], StageSpec(kind="prompt_upsample", core={"prompt": "x"})])
        assert r.wait(chain.id).status == "completed"
        assert r.runner.wait_idle()
        assert ran == ["Shut down"]  # the last stage ran off the worker: no job's end saw the queue done


def test_a_pipeline_whose_upsampling_fails_runs_the_completion_action(tmp_path):
    def upsampler(stage, prompt, outputs):
        raise RuntimeError("model missing")

    with rig(tmp_path, upsampler=upsampler) as r:
        ran = completion_action(r.queue, "Shut down")
        chain = r.runner.create_chain([*seeded(5)[:1], StageSpec(kind="prompt_upsample", core={"prompt": "x"}), *seeded(5)[1:]])
        assert r.wait(chain.id).status == "failed"
        assert r.runner.wait_idle()
        assert ran == ["Shut down"]


def test_a_pipeline_that_ends_during_the_request_does_not_run_the_completion_action(tmp_path):
    with rig(tmp_path) as r:
        first = r.wait(r.runner.create_chain(seeded(5)).id)
        assert first.status == "completed" and r.queue.wait_idle()
        ran = completion_action(r.queue, "Shut down")
        again = r.runner.rerun_chain(first.id)
        assert again.status == "completed"  # every stage cached: done before the request answers
        assert r.runner.wait_idle() and r.queue.wait_idle()
        assert ran == []  # the person who clicked is at the computer


def test_the_completion_action_runs_once_when_two_threads_find_the_queue_done(tmp_path):
    with rig(tmp_path) as r:
        ran = completion_action(r.queue, "Shut down")
        read = r.queue.option

        def slow_read(name, default):  # widens the window between reading the action and switching it off
            value = read(name, default)
            time.sleep(0.02)
            return value

        r.queue.option = slow_read
        threads = [threading.Thread(target=r.queue.recheck_done) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert ran == ["Shut down"]


def test_a_pipeline_cancelled_by_hand_does_not_run_the_completion_action(tmp_path):
    gate = threading.Event()

    def upsampler(stage, prompt, outputs):
        gate.wait(5)
        return "late"

    with rig(tmp_path, upsampler=upsampler) as r:
        ran = completion_action(r.queue, "Shut down")
        chain = r.runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "x"}), *seeded(5)[:1]])
        r.runner.cancel_chain(chain.id)
        gate.set()
        assert r.runner.wait_idle()
        assert ran == []


def test_cancel_during_upsampling_drops_its_result(tmp_path):
    gate = threading.Event()

    def upsampler(stage, prompt, outputs):
        gate.wait(5)
        return "late"

    with rig(tmp_path, upsampler=upsampler) as r:
        chain = r.runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "x"}), *seeded(5)[:1]])
        r.runner.cancel_chain(chain.id)
        gate.set()
        assert r.runner.wait_idle()
        done = r.runner.store.get_chain(chain.id)
        assert done.status == "cancelled" and [s.status for s in done.stages] == ["cancelled", "cancelled"]
        assert r.executed == []


# endregion


def test_a_stale_upsampling_thread_does_not_write_into_a_rerun(tmp_path):
    gate = threading.Event()

    def upsampler(stage, prompt, outputs):
        if stage.core.get("prompt") == "old":
            gate.wait(5)
        return f"{stage.core.get('prompt')}!"

    def stages(prompt: str) -> list[StageSpec]:
        return [StageSpec(kind="prompt_upsample", core={"prompt": prompt}), StageSpec(kind="generate", tab="txt2img", core={}, scripts={"seed": {"seed": 5}})]

    with rig(tmp_path, upsampler=upsampler) as r:
        chain = r.runner.create_chain(stages("old"))
        r.runner.cancel_chain(chain.id)
        r.runner.rerun_chain(chain.id, stages=stages("new"))  # its thread waits behind the old one
        gate.set()
        done = r.wait(chain.id)
        assert done.status == "completed"
        assert done.stages[0].output_prompt == "new!"
        assert [e["core"]["prompt"] for e in r.executed] == ["new!"]


def test_deleting_a_chain_waits_for_the_runner(tmp_path):
    """A delete between a resume's status change and its next stage job would
    leave that job without its chain: delete takes the runner's lock too."""

    with rig(tmp_path, start=False) as r:
        chain = r.runner.create_chain(seeded(5), run_now=False)
        deleted = threading.Event()
        thread = threading.Thread(target=lambda: r.runner.delete_chain(chain.id) and deleted.set())
        with r.runner._lock:
            thread.start()
            assert not deleted.wait(0.3)
        thread.join(5)
        assert deleted.is_set() and r.runner.store.get_chain(chain.id) is None


def test_a_stage_that_cannot_start_fails_the_chain(tmp_path, monkeypatch):
    """An error while the next stage is looked up or queued must not leave the
    chain running with no stage running (busy() would hold the queue's done state)."""

    with rig(tmp_path) as r:

        def broken(*args, **kwargs):
            raise RuntimeError("the hash broke")

        monkeypatch.setattr(r.runner, "_stage_hash", broken)
        chain = r.runner.create_chain(seeded(5))
        assert chain.status == "failed" and "the hash broke" in (chain.error or "")
        assert r.statuses(chain.id) == ["failed", "pending"] and not r.runner.busy()


def test_an_upsample_thread_that_cannot_start_is_not_waited_for(tmp_path, monkeypatch):
    with rig(tmp_path) as r:

        def refuse(self):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Thread, "start", refuse)
        chain = r.runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "fox"}), *seeded(5)[:1]])
        assert chain.status == "failed" and not r.runner._threads
        monkeypatch.undo()
        assert r.runner.wait_idle(1)
