"""Unit and route tests for linear multi-stage JobChains with stage caching."""

from __future__ import annotations

import threading
import time

from PIL import Image

from remi.files import FileStore
from remi.jobs import ChainRunner, ChainStore, JobQueue, JobResult, JobStore, StageSpec, compute_stage_hash, run_inline
from remi.realtime import EventBus
from remi.server.testing import FakeState, StubExecutor, server_app
from remi.testing import webui_state


def test_compute_stage_hash_is_deterministic_and_sensitive():
    s1 = StageSpec(kind="generate", tab="txt2img", core={"prompt": "cat", "width": 512}, scripts={"sampler": {"steps": 20}})
    s2 = StageSpec(kind="generate", tab="txt2img", core={"width": 512, "prompt": "cat"}, scripts={"sampler": {"steps": 20}})
    s3 = StageSpec(kind="generate", tab="txt2img", core={"prompt": "cat", "width": 768}, scripts={"sampler": {"steps": 20}})

    assert compute_stage_hash(s1, model_fingerprint="m1") == compute_stage_hash(s2, model_fingerprint="m1")
    assert compute_stage_hash(s1, model_fingerprint="m1") != compute_stage_hash(s3, model_fingerprint="m1")
    assert compute_stage_hash(s1, model_fingerprint="m1") != compute_stage_hash(s1, model_fingerprint="m2")
    assert compute_stage_hash(s1, upstream_outputs=["fileA"]) != compute_stage_hash(s1, upstream_outputs=["fileB"])


def _wait_chain_done(store: ChainStore, chain_id: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ch = store.get_chain(chain_id)
        if ch is not None and ch.status in ("completed", "failed", "cancelled"):
            return ch
        time.sleep(0.02)
    return store.get_chain(chain_id)


def test_chain_four_stage_execution_and_downstream_rerun_cache_reuse(tmp_path):
    files = FileStore(tmp_path / "files")
    job_store = JobStore(tmp_path / "jobs.db")
    bus = EventBus()
    events: list[dict] = []
    bus.subscribe(type("Sub", (), {"send_event": lambda self, e: events.append(e), "send_frame": lambda self, f: None})())

    executed_jobs: list[tuple[str, str, dict]] = []
    state = FakeState()

    def fake_exec(job, ctx):
        executed_jobs.append((job.meta.get("stage_kind", job.tab), job.tab, dict(job.core)))
        color = (len(executed_jobs) * 30 % 255, 100, 150)
        img = Image.new("RGB", (16, 16), color)
        prompt = job.core.get("prompt", "")
        return JobResult(images=[img], infotexts=[f"{prompt}\nSteps: 10, Seed: {len(executed_jobs)}"], info=f"{prompt}\nSteps: 10")

    queue = JobQueue(
        job_store,
        files,
        bus,
        executor=fake_exec,
        dispatch=run_inline,
        lock=threading.Lock(),
        validate=lambda tab, core, scripts: None,
        state=lambda: state,
        hash_layout=lambda tab: "test-hash",
    )
    chain_store = ChainStore(job_store)
    upsample_calls: list[str] = []

    def fake_upsampler(stage, upstream_prompt, upstream_outputs):
        base = stage.core.get("prompt") or upstream_prompt or ""
        upsample_calls.append(base)
        return f"{base}, masterwork detailed lighting"

    runner = ChainRunner(chain_store, queue, files, bus, upsampler=fake_upsampler, model_fingerprint=lambda: "anima-v1")
    queue.start()
    try:
        stages = [
            StageSpec(kind="prompt_upsample", core={"prompt": "a fox in snow"}),
            StageSpec(kind="generate", tab="txt2img", core={"width": 512, "height": 512}),
            StageSpec(kind="refine", tab="img2img", core={"denoising_strength": 0.4}),
            StageSpec(kind="finish", tab="extras", core={"upscaling_resize": 2.0}),
        ]
        chain = runner.create_chain(stages, name="Fox 4-Stage")
        done = _wait_chain_done(chain_store, chain.id)
        assert done is not None and done.status == "completed"
        assert len(upsample_calls) == 1
        assert [kind for kind, _tab, _core in executed_jobs] == ["generate", "refine", "finish"]

        # Verify automatic upstream -> downstream propagation
        gen_core = executed_jobs[0][2]
        ref_core = executed_jobs[1][2]
        fin_core = executed_jobs[2][2]
        assert gen_core["prompt"] == "a fox in snow, masterwork detailed lighting"
        assert ref_core["init_img"] == done.stages[1].output_file_ids[0]
        assert fin_core["image"] == done.stages[2].output_file_ids[0]

        # Dry-run preview after tweaking ONLY stage 3 (finish): stages 0,1,2 hit cache, stage 3 misses!
        tweaked_stages = [
            stages[0],
            stages[1],
            stages[2],
            StageSpec(kind="finish", tab="extras", core={"upscaling_resize": 4.0}),
        ]
        plan = runner.plan_chain(tweaked_stages, chain_id=chain.id)
        assert [p["cache_hit"] for p in plan] == [True, True, True, False]
        # the generate stage has no seed (random): a new chain draws a new one
        assert [p["cache_hit"] for p in runner.plan_chain(tweaked_stages)] == [True, False, False, False]
        assert plan[2]["cached_outputs"] == done.stages[2].output_file_ids

        # Re-run chain with tweaked finish stage: stages 0, 1, 2 MUST be reused from cache!
        rerun = runner.rerun_chain(chain.id, stages=tweaked_stages)
        done2 = _wait_chain_done(chain_store, rerun.id)
        assert done2 is not None and done2.status == "completed"
        assert len(upsample_calls) == 1  # NOT called again!
        assert len(executed_jobs) == 4  # Only finish ran again (3 + 1 = 4)!
        assert [st.status for st in done2.stages] == ["cached", "cached", "cached", "completed"]
        assert [st.cached for st in done2.stages] == [True, True, True, False]
        assert executed_jobs[3][0] == "finish"
        assert executed_jobs[3][2]["image"] == done.stages[2].output_file_ids[0]

        # Verify WS events published
        stage_events = [e for e in events if e["type"] == "chain.stage_completed"]
        chain_events = [e for e in events if e["type"] == "chain.updated"]
        assert any(e["cached"] is True and e["stage_index"] == 1 for e in stage_events)
        assert any(e["status"] == "completed" for e in chain_events)

        # If cached output file is deleted from FileStore, cache must miss cleanly
        files.delete(done.stages[1].output_file_ids[0], delete_file=True)
        plan_after_delete = runner.plan_chain(tweaked_stages, chain_id=chain.id)
        assert plan_after_delete[0]["cache_hit"] is True
        assert plan_after_delete[1]["cache_hit"] is False
        assert plan_after_delete[2]["cache_hit"] is False
    finally:
        queue.stop()
        runner.close()
        job_store.close()
        files.close()


def test_chains_http_routes_templates_and_pause_cancel(tmp_path):
    state = FakeState()
    executor = StubExecutor(state, count=1, steps=0)
    with webui_state(), server_app(tmp_path, executor=executor, state=state, validate=lambda tab, core, scripts: None) as server:
        # Save a reusable template
        tmpl_res = server.client.post(
            "/api/v2/chains/templates",
            json={
                "name": "Standard Generate + Refine",
                "description": "Two-stage workflow",
                "model_family": "anima",
                "stages": [
                    {"kind": "generate", "tab": "txt2img", "core": {"prompt": "forest"}},
                    {"kind": "refine", "tab": "img2img", "core": {"denoising_strength": 0.35}},
                ],
                "tags": ["default"],
            },
        )
        assert tmpl_res.status_code == 201
        tid = tmpl_res.json()["id"]
        assert len(server.client.get("/api/v2/chains/templates").json()["items"]) == 1
        assert server.client.get(f"/api/v2/chains/templates/{tid}").status_code == 200

        # Create paused chain, then resume and wait completion
        created = server.client.post(
            "/api/v2/chains",
            json={
                "name": "Forest Run",
                "run_now": False,
                "stages": tmpl_res.json()["stages"],
                "model_fingerprint": "test-model",
            },
        )
        assert created.status_code == 201
        cid = created.json()["id"]
        assert created.json()["status"] == "paused"

        resumed = server.client.post(f"/api/v2/chains/{cid}/resume")
        assert resumed.status_code == 200
        assert server.services.queue.wait_idle(timeout=5)
        done = _wait_chain_done(server.services.chains.store, cid)
        assert done is not None and done.status == "completed"

        # Preview cache hit on second run
        preview = server.client.post(
            "/api/v2/chains/preview",
            json={"stages": tmpl_res.json()["stages"], "model_fingerprint": "test-model", "chain_id": cid},
        )
        assert preview.status_code == 200
        assert [st["cache_hit"] for st in preview.json()["stages"]] == [True, True]

        # List chains & delete
        listed = server.client.get("/api/v2/chains").json()
        assert listed["total"] == 1 and listed["items"][0]["id"] == cid

        # Cancel a paused chain
        paused_chain = server.client.post(
            "/api/v2/chains",
            json={"name": "Cancel Me", "run_now": False, "stages": [{"kind": "generate", "tab": "txt2img", "core": {"prompt": "x"}}]},
        ).json()
        cancelled = server.client.post(f"/api/v2/chains/{paused_chain['id']}/cancel")
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"

        assert server.client.delete(f"/api/v2/chains/{paused_chain['id']}").status_code == 204
        assert server.client.delete(f"/api/v2/chains/templates/{tid}").status_code == 204


def test_chains_edge_cases_errors_disabled_stages_and_routes(tmp_path, monkeypatch):
    import sys
    import types
    from types import SimpleNamespace

    import pytest

    from remi.jobs import chains as chains_mod
    from remi.jobs.chains import ChainError, StageRun

    # 1. Standalone ChainStore context manager + validation errors + pagination/filtering
    with ChainStore(tmp_path / "standalone.db") as store:
        with pytest.raises(ChainError):
            store.create_chain([])
        with pytest.raises(ChainError):
            store.create_chain([StageSpec(kind="generate")] * (chains_mod.MAX_STAGES + 1))

        c1 = store.create_chain([StageSpec(kind="generate", core={"prompt": "a"})], name="One")
        c2 = store.create_chain([StageSpec(kind="generate", core={"prompt": "b"})], name="Two", status="paused")
        with pytest.raises(ChainError):
            store.replace_stages(c1.id, [])
        store.update_stage(c1.id, 0)  # no-op sets
        store.update_stage(c1.id, 0, error="minor")

        page1, cursor1, total = store.list_chains(status="queued,paused", limit=1)
        assert total == 2 and len(page1) == 1 and cursor1 is not None
        page2, cursor2, _ = store.list_chains(status="queued,paused", cursor=cursor1, limit=1)
        assert len(page2) == 1 and cursor2 is None
        assert store.list_chains(status="not_a_real_status")[2] == 0
        # Bad cursor is ignored gracefully
        assert len(store.list_chains(cursor="bad:cursor")[0]) == 2

        with pytest.raises(ChainError):
            store.create_template("   ", [StageSpec(kind="generate")])
        with pytest.raises(ChainError):
            store.create_template("Empty", [])
        store.create_template("Dup", [StageSpec(kind="generate")])
        with pytest.raises(ChainError) as dup_exc:
            store.create_template("Dup", [StageSpec(kind="generate")])
        assert dup_exc.value.status == 409
        assert c2.status == "paused"

    # 2. _default_upsampler (empty prompt, stubbed pup, exception fallback) + _default_model_fingerprint
    assert chains_mod._default_upsampler(StageRun(chain_id="c", index=0, kind="prompt_upsample", name="PUP", enabled=True, tab="txt2img", core={"prompt": "  "}), None, []) == "  "
    fake_pup = types.ModuleType("remi.prompt_tools.pup")
    fake_pup.upsample_prompt = lambda prompt, **kw: SimpleNamespace(prompt=f"{prompt} [pup]")
    monkeypatch.setitem(sys.modules, "remi.prompt_tools.pup", fake_pup)
    from remi import prompt_tools

    monkeypatch.setattr(prompt_tools.registry, "group_dirs", ["group_tags"])  # the prompt tools are on
    assert chains_mod._default_upsampler(StageRun(chain_id="c", index=0, kind="prompt_upsample", name="PUP", enabled=True, tab="txt2img", core={"prompt": "sky"}), None, []) == "sky [pup]"
    fake_pup.upsample_prompt = lambda prompt, **kw: (_ for _ in ()).throw(RuntimeError("pup offline"))
    with pytest.raises(RuntimeError, match="pup offline"):  # fails the stage, never passes the prompt on as upsampled
        chains_mod._default_upsampler(StageRun(chain_id="c", index=0, kind="prompt_upsample", name="PUP", enabled=True, tab="txt2img", core={"prompt": "sky"}), None, [])
    assert isinstance(chains_mod._default_model_fingerprint(), str)

    # 3. ChainRunner failure paths, disabled stages, pause/resume/cancel/delete guards, and finish tabs
    files = FileStore(tmp_path / "files2")
    job_store = JobStore(tmp_path / "jobs2.db")
    bus = EventBus()
    state = FakeState()
    mode = {"behavior": "ok"}

    def flakey_exec(job, ctx):
        if mode["behavior"] == "fail":
            raise RuntimeError("gpu oom")
        if mode["behavior"] == "interrupt":
            state.interrupted = True
        return JobResult(images=[Image.new("RGB", (8, 8), (1, 2, 3))], infotexts=["p"], info="p")

    queue = JobQueue(
        job_store,
        files,
        bus,
        executor=flakey_exec,
        dispatch=run_inline,
        lock=threading.Lock(),
        validate=lambda tab, core, scripts: None,
        state=lambda: state,
        hash_layout=lambda tab: "h",
    )
    cstore = ChainStore(job_store)
    runner = ChainRunner(cstore, queue, files, bus, upsampler=lambda s, p, o: (_ for _ in ()).throw(RuntimeError("pup crash")), model_fingerprint=lambda: "fp1")
    queue.start()
    try:
        # Disabled stage in plan_chain
        plan = runner.plan_chain([StageSpec(kind="prompt_upsample", enabled=False), StageSpec(kind="generate", core={"prompt": "a"})])
        assert plan[0]["enabled"] is False and plan[0]["cache_hit"] is False

        # prompt_upsample failure marks chain failed
        ch_pup_fail = runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "boom"})])
        ch_pup_fail = _wait_chain_done(cstore, ch_pup_fail.id)
        assert ch_pup_fail is not None and ch_pup_fail.status == "failed" and "pup crash" in (ch_pup_fail.error or "")
        assert cstore.get_cached(ch_pup_fail.stages[0].stage_hash or "") is None

        # _build_stage_submission finish tab variants (txt2img -> upscale_from, img2img -> init_img)
        sub_t2i = runner._build_stage_submission(ch_pup_fail, StageRun(chain_id=ch_pup_fail.id, index=1, kind="finish", name="Finish", enabled=True, tab="txt2img"), ["img-1"], None)
        assert sub_t2i.upscale_from == "img-1"
        sub_i2i = runner._build_stage_submission(ch_pup_fail, StageRun(chain_id=ch_pup_fail.id, index=1, kind="finish", name="Finish", enabled=True, tab="img2img"), ["img-1"], None)
        assert sub_i2i.core["init_img"] == "img-1"

        # queue.submit error marks chain failed
        monkeypatch.setattr(queue, "submit", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("queue full")))
        ch_sub_fail = runner.create_chain([StageSpec(kind="generate", tab="txt2img", core={"prompt": "x"})])
        assert ch_sub_fail.status == "failed" and "queue full" in (ch_sub_fail.error or "")
        monkeypatch.undo()

        # Job failure & interruption propagate to chain status
        mode["behavior"] = "fail"
        ch_jfail = runner.create_chain([StageSpec(kind="generate", tab="txt2img", core={"prompt": "f"})])
        done_fail = _wait_chain_done(cstore, ch_jfail.id)
        assert done_fail is not None and done_fail.status == "failed"

        mode["behavior"] = "interrupt"
        ch_jint = runner.create_chain([StageSpec(kind="generate", tab="txt2img", core={"prompt": "i"})], run_now=False)
        j_can = job_store.add("txt2img", {"prompt": "i"}, {})
        j_can = job_store.finish(j_can.id, "cancelled", error="cancelled by user")
        cstore.update_chain(ch_jint.id, status="running")
        cstore.update_stage(ch_jint.id, 0, status="running", job_id=j_can.id)
        runner.on_job_finished(j_can)
        done_int = cstore.get_chain(ch_jint.id)
        assert done_int is not None and done_int.status == "cancelled"

        # Non-chain job completion hook returns cleanly
        non_chain_job = job_store.add("txt2img", {"prompt": "solo"}, {})
        runner.on_job_finished(non_chain_job)

        # Pause/resume/cancel/rerun/delete guards
        mode["behavior"] = "ok"
        ch_ok = runner.create_chain([StageSpec(kind="generate", tab="txt2img", core={"prompt": "ok"})])
        done_ok = _wait_chain_done(cstore, ch_ok.id)
        assert done_ok is not None and done_ok.status == "completed"

        # Rerun with stages=None and from_stage=0
        rerun_same = runner.rerun_chain(ch_ok.id, stages=None, from_stage=0)
        done_rerun = _wait_chain_done(cstore, rerun_same.id)
        assert done_rerun is not None and done_rerun.status == "completed"

        with pytest.raises(ChainError):
            runner.rerun_chain("no-such-chain")
        with pytest.raises(ChainError):
            runner.pause_chain("no-such-chain")
        with pytest.raises(ChainError):
            runner.pause_chain(ch_ok.id)  # already completed -> 409
        with pytest.raises(ChainError):
            runner.resume_chain("no-such-chain")
        with pytest.raises(ChainError):
            runner.resume_chain(ch_ok.id)  # not paused -> 409
        with pytest.raises(ChainError):
            runner.cancel_chain("no-such-chain")
        assert runner.delete_chain("no-such-chain") is False

        # Pause a queued/running chain and test running guards
        ch_paused = runner.create_chain([StageSpec(kind="generate", tab="txt2img", core={"prompt": "p"})], run_now=False)
        cstore.update_chain(ch_paused.id, status="running")
        with pytest.raises(ChainError):
            runner.rerun_chain(ch_paused.id)
        with pytest.raises(ChainError):
            runner.delete_chain(ch_paused.id)
        paused_back = runner.pause_chain(ch_paused.id)
        assert paused_back.status == "paused"
        assert runner._advance(ch_paused.id).status == "paused"
    finally:
        queue.stop()
        runner.close()
        job_store.close()
        files.close()

    # 4. Route 404 & rerun/pause coverage in remi/server/routes/chains.py
    state2 = FakeState()
    with webui_state(), server_app(tmp_path / "srv", executor=StubExecutor(state2, count=1, steps=0), state=state2, validate=lambda *a: None) as server:
        ch = server.client.post("/api/v2/chains", json={"name": "R", "run_now": False, "stages": [{"kind": "generate", "tab": "txt2img", "core": {"prompt": "a"}}]}).json()
        assert server.client.get(f"/api/v2/chains/{ch['id']}").status_code == 200
        assert server.client.get("/api/v2/chains/missing").status_code == 404
        assert server.client.get("/api/v2/chains/templates/missing").status_code == 404
        assert server.client.delete("/api/v2/chains/templates/missing").status_code == 404
        assert server.client.delete("/api/v2/chains/missing").status_code == 404
        assert server.client.post(f"/api/v2/chains/{ch['id']}/pause").status_code == 200
        assert server.client.post(f"/api/v2/chains/{ch['id']}/rerun", json={"from_stage": 0}).status_code == 200
        assert server.services.queue.wait_idle(timeout=5)


def test_the_chain_tables_have_one_schema(tmp_path):
    """The chain tables live in the jobs database (the queue's store reads
    them) or in a database of their own: both are made from one schema."""

    from remi.jobs.store import CHAIN_SCHEMA

    assert "chain_force" in CHAIN_SCHEMA

    def tables(db):
        return dict(db.execute("SELECT name, sql FROM sqlite_master WHERE name IN ('chains', 'stage_runs', 'stage_cache', 'chain_templates', 'chain_force') ORDER BY name").fetchall())

    jobs = JobStore(tmp_path / "jobs.db")
    shared, standalone = ChainStore(jobs), ChainStore(tmp_path / "chains.db")
    try:
        assert tables(shared._db) == tables(standalone._db) and len(tables(standalone._db)) == 5
    finally:
        standalone.close()
        jobs.close()


def test_an_upload_a_saved_template_uses_is_not_pruned(tmp_path):
    """Templates saved through the API live in the jobs database: the uploads
    their stages name (a refine stage's mask, say) stay with them."""

    import io

    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    chains = ChainStore(jobs)
    try:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(buffer, format="PNG")
        upload = files.add_upload_bytes(buffer.getvalue())
        chains.create_template("masked", [StageSpec(kind="refine", core={"mask": upload.id})])
        files._db.execute("UPDATE files SET created = ?", (time.time() - 30 * 86400,))
        files.prune_uploads(86400, keep=jobs.referenced_strings())
        assert files.get(upload.id) is not None
    finally:
        files.close()
        jobs.close()


def test_housekeeping_prunes_old_stage_cache_entries(tmp_path):
    from remi.jobs import run_inline
    from remi.server.services import Services

    with webui_state():
        services = Services.create(tmp_path, dispatch=run_inline, realtime=False, housekeeping=False)
        try:
            store = services.chains.store
            store.put_cached("old", "prompt_upsample", output_prompt="a")
            store.put_cached("new", "prompt_upsample", output_prompt="b")
            store._db.execute("UPDATE stage_cache SET created = ? WHERE stage_hash = 'old'", (time.time() - 30 * 86400,))
            services.housekeeping()
            assert store.get_cached("old") is None and store.get_cached("new") is not None
            assert store.prune_cache(time.time() + 1) == 1 and store.get_cached("new") is None
        finally:
            services.close()


def test_reusing_a_cached_stage_keeps_it_from_the_prune(tmp_path):
    """Housekeeping forgets the entries no pipeline used lately, not the
    ones stored long ago and reused since; a dry-run plan is no use."""

    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    bus = EventBus()
    state = FakeState()
    queue = JobQueue(jobs, files, bus, executor=lambda job, ctx: None, dispatch=run_inline, lock=threading.Lock(), validate=lambda *a: None, state=lambda: state, hash_layout=lambda tab: "h")
    calls = []
    runner = ChainRunner(ChainStore(jobs), queue, files, bus, upsampler=lambda s, p, o: calls.append(p) or f"{s.core['prompt']}, detailed", model_fingerprint=lambda: "m")
    store = runner.store
    try:
        stages = [StageSpec(kind="prompt_upsample", core={"prompt": "fox"})]
        first = runner.create_chain(stages)
        assert runner.wait_idle(5) and store.get_chain(first.id).status == "completed"
        stage_hash = store.get_chain(first.id).stages[0].stage_hash
        month_ago = time.time() - 30 * 86400
        store._db.execute("UPDATE stage_cache SET created = ?", (month_ago,))
        assert runner.plan_chain(stages)[0]["cache_hit"] and store.get_cached(stage_hash).created == month_ago
        again = runner.create_chain(stages)
        assert runner.wait_idle(5) and store.get_chain(again.id).stages[0].status == "cached" and len(calls) == 1
        assert store.get_cached(stage_hash).created > month_ago
        assert store.prune_cache(time.time() - 7 * 86400) == 0 and store.get_cached(stage_hash) is not None
    finally:
        runner.close()
        files.close()
        jobs.close()


def test_a_page_of_chains_loads_its_stages_in_one_query(tmp_path):
    with ChainStore(tmp_path / "chains.db") as store:
        made = [store.create_chain([StageSpec(kind="generate", core={"prompt": f"p{i}"}), StageSpec(kind="refine")]) for i in range(5)]
        singly = {chain.id: store.get_chain(chain.id) for chain in made}
        statements: list[str] = []
        store._db.set_trace_callback(statements.append)
        try:
            items, _cursor, total = store.list_chains(limit=50)
        finally:
            store._db.set_trace_callback(None)
        assert total == 5 and {chain.id: chain for chain in items} == singly
        assert len([sql for sql in statements if "FROM stage_runs" in sql]) == 1


def test_stages_with_non_finite_numbers_are_refused(tmp_path):
    """JSON would store them as null (or not at all): 1e999 is refused like a recipe's."""

    body = '{"name": "inf", "stages": [{"kind": "generate", "core": {"cfg_scale": 1e999}}]}'
    with webui_state(), server_app(tmp_path, validate=lambda tab, core, scripts: None) as server:
        headers = {"Content-Type": "application/json"}
        template = server.client.post("/api/v2/chains/templates", content=body, headers=headers)
        assert template.status_code == 422 and template.json() == {
            "detail": "stages[0].core.cfg_scale must not contain NaN or Infinity",
            "template": "{{path}} must not contain NaN or Infinity",
            "params": {"path": "stages[0].core.cfg_scale"},
        }
        chain = server.client.post("/api/v2/chains", content=body.replace("cfg_scale", "steps").replace('"stages"', '"run_now": false, "stages"'), headers=headers)
        assert chain.status_code == 422
        store = server.services.chains.store
        assert store.list_templates() == [] and store.list_chains()[2] == 0
        # one stored before this check still reads (json writes it as Infinity)
        store.create_template("old", [StageSpec(kind="generate")])
        store._db.execute("UPDATE chain_templates SET stages = ?", ('[{"kind": "generate", "core": {"cfg_scale": Infinity}}]',))
        assert server.client.get("/api/v2/chains/templates").status_code == 200


def test_an_upsampled_prompt_is_cached_per_provider(tmp_path):
    """Another endpoint or model writes another prompt: the prompt tools'
    provider is part of an upsample stage's cache key (never its API key)."""

    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    bus = EventBus()
    state = FakeState()
    queue = JobQueue(jobs, files, bus, executor=lambda job, ctx: None, dispatch=run_inline, lock=threading.Lock(), validate=lambda *a: None, state=lambda: state, hash_layout=lambda tab: "h")
    provider = {"id": "openai:http://a/v1:qwen"}
    runner = ChainRunner(ChainStore(jobs), queue, files, bus, upsampler=lambda s, p, o: f"{s.core['prompt']}, detailed", model_fingerprint=lambda: "m", upsample_identity=lambda: provider["id"])
    try:
        stages = [StageSpec(kind="prompt_upsample", core={"prompt": "fox"})]
        chain = runner.create_chain(stages)
        assert runner.wait_idle(5) and runner.store.get_chain(chain.id).status == "completed"
        first = runner.plan_chain(stages)[0]
        assert first["cache_hit"]
        provider["id"] = "openai:http://b/v1:qwen"
        second = runner.plan_chain(stages)[0]
        assert second["stage_hash"] != first["stage_hash"] and not second["cache_hit"]
        # a generate stage does not depend on the provider
        generate = [StageSpec(kind="generate", core={"seed": 1})]
        before = runner.plan_chain(generate)[0]["stage_hash"]
        provider["id"] = "deepl::"
        assert runner.plan_chain(generate)[0]["stage_hash"] == before
    finally:
        runner.close()
        files.close()
        jobs.close()


def test_the_server_keys_upsampling_by_the_provider_settings_without_the_key(tmp_path, monkeypatch):
    from remi import prompt_tools as pt
    from remi.prompt_tools import options as prompt_options

    settings = {"prompt_tools_translation_provider": "openai", "prompt_tools_translation_url": "http://a/v1", "prompt_tools_translation_model": "qwen"}
    monkeypatch.setattr(prompt_options, "option", lambda name, default=None: settings.get(name, default))
    with webui_state(), server_app(tmp_path) as server:
        pt.tools_for(server.services).set_translation_key("sk-secret")
        identity = server.services.chains.upsample_identity
        assert identity() == "openai:http://a/v1:qwen"
        settings["prompt_tools_translation_url"] = "http://b/v1"
        assert identity() == "openai:http://b/v1:qwen"


def test_an_interrupted_local_upsample_stage_ends_the_pipeline_cancelled(tmp_path):
    """A local model's upsample stage runs behind generation, where Interrupt flips the queue's
    state: the flag is cleared before it starts and after it ends, and a stage Interrupt stopped
    ends the pipeline ``cancelled`` (as an interrupted generation stage does), with nothing cached."""

    from remi.prompt_tools.translate import GenerationInterrupted

    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    bus = EventBus()
    state = FakeState()
    queue = JobQueue(jobs, files, bus, executor=lambda job, ctx: None, dispatch=run_inline, lock=threading.Lock(), validate=lambda *a: None, state=lambda: state, hash_layout=lambda tab: "h")
    seen = []

    def upsampler(stage, upstream_prompt, upstream_outputs):
        seen.append(state.interrupted)
        queue.interrupt_all()  # the user presses Interrupt while the local model writes
        raise GenerationInterrupted()

    runner = ChainRunner(ChainStore(jobs), queue, files, bus, upsampler=upsampler, model_fingerprint=lambda: "m")
    try:
        state.interrupt()  # left over from an Interrupt while nothing ran
        chain = runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "fox"}, config={"local_model_path": "/srv/qwen.gguf"}), StageSpec(kind="generate", core={"prompt": "fox"})])
        assert runner.wait_idle(5)
        done = runner.store.get_chain(chain.id)
        assert seen == [False]
        assert done.status == "cancelled" and done.stages[0].status == "cancelled" and done.stages[1].job_id is None  # the next stage never ran
        assert done.error == "the prompt upsampling was interrupted"
        assert runner.store.get_cached(done.stages[0].stage_hash or "") is None
        assert state.interrupted is False
    finally:
        runner.close()
        files.close()
        jobs.close()


def test_cancelling_a_pipeline_stops_its_local_upsample_stage(tmp_path):
    """Cancel stops a local model's upsample stage as Interrupt does: it holds the generation lock,
    so the queue's state is flipped for it (and cleared after, for whatever runs next). A stage that
    only asks the prompt tools' endpoint runs beside jobs: cancelling it leaves the state alone."""

    from remi.prompt_tools.translate import GenerationInterrupted

    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    bus = EventBus()
    state = FakeState()
    queue = JobQueue(jobs, files, bus, executor=lambda job, ctx: None, dispatch=run_inline, lock=threading.Lock(), validate=lambda *a: None, state=lambda: state, hash_layout=lambda tab: "h")
    started = threading.Event()
    seen: list[bool] = []

    def upsampler(stage, upstream_prompt, upstream_outputs):
        started.set()
        deadline = time.monotonic() + 1
        while not state.interrupted and time.monotonic() < deadline:
            time.sleep(0.01)
        seen.append(state.interrupted)
        if state.interrupted:
            raise GenerationInterrupted()
        return "fox, detailed"

    runner = ChainRunner(ChainStore(jobs), queue, files, bus, upsampler=upsampler, model_fingerprint=lambda: "m")
    try:
        for config, stopped in (({"local_model_path": "/srv/qwen.gguf"}, True), ({}, False)):
            started.clear()
            chain = runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "fox"}, config=config), StageSpec(kind="generate", core={"prompt": "fox"})])
            assert started.wait(5)
            assert runner.cancel_chain(chain.id).status == "cancelled"
            assert runner.wait_idle(5)
            assert seen.pop() is stopped
            done = runner.store.get_chain(chain.id)
            assert done.status == "cancelled" and done.stages[1].job_id is None
            assert runner.store.get_cached(done.stages[0].stage_hash or "") is None
            assert state.interrupted is False
    finally:
        runner.close()
        files.close()
        jobs.close()


def test_a_failed_upsample_stage_says_why_in_its_own_sentence(tmp_path, monkeypatch):
    """A failed upsample stage stored "TranslationError: the model answered with an empty prompt",
    and "PermissionError: ..." for a local model under --hide-ui-dir-config: with the class name
    in front, no catalog had the text. The pipeline and the stage keep the sentence itself, as the
    pipeline's other errors are plain sentences: one without values is a key of the server
    catalogs (one with values, a service's own words, is its English)."""

    import json
    from pathlib import Path
    from types import SimpleNamespace

    from modules import shared
    from remi.jobs import chains as chains_mod
    from remi.prompt_tools.translate import TranslationError
    from remi.ui import Sentence

    catalog = json.loads((Path(__file__).resolve().parents[1] / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(shared, "cmd_opts", SimpleNamespace(hide_ui_dir_config=True), raising=False)
    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    bus = EventBus()
    state = FakeState()
    queue = JobQueue(jobs, files, bus, executor=lambda job, ctx: None, dispatch=run_inline, lock=threading.Lock(), validate=lambda *a: None, state=lambda: state, hash_layout=lambda tab: "h")

    def upsampler(stage, upstream_prompt, upstream_outputs):
        if stage.config.get("local_model_path"):
            return chains_mod._default_upsampler(stage, upstream_prompt, upstream_outputs)
        raise TranslationError(Sentence("the model answered with an empty prompt"))

    runner = ChainRunner(ChainStore(jobs), queue, files, bus, upsampler=upsampler, model_fingerprint=lambda: "m")
    try:
        for config, sentence in (({}, "the model answered with an empty prompt"), ({"local_model_path": "/srv/qwen.gguf"}, "local_model_path is disabled under --hide-ui-dir-config")):
            chain = runner.create_chain([StageSpec(kind="prompt_upsample", core={"prompt": "fox"}, config=config), StageSpec(kind="generate", core={"prompt": "fox"})])
            assert runner.wait_idle(5)
            done = runner.store.get_chain(chain.id)
            assert done.status == "failed" and done.stages[1].job_id is None
            assert done.error == done.stages[0].error == sentence
            assert catalog.get(sentence)
    finally:
        runner.close()
        files.close()
        jobs.close()


def test_housekeeping_preserves_reused_stage_cache_outputs_and_prunes_dead_or_excess(tmp_path):
    """active stage_cache outputs survive FileStore.prune_unsaved();
    stage_cache enforces max_entries eviction and prune_cache(..., files=files) drops
    entries whose outputs were removed."""

    from remi.jobs import run_inline
    from remi.server.services import Services

    with webui_state():
        services = Services.create(tmp_path, dispatch=run_inline, realtime=False, housekeeping=False)
        try:
            out = services.files.save_output_image(Image.new("RGB", (8, 8)), "cached")
            stale_out = services.files.save_output_image(Image.new("RGB", (8, 8)), "stale")
            # Backdate file creation > 7 days ago, but touch stage_cache entry to now
            services.files._db.execute("UPDATE files SET created = ?", (time.time() - 30 * 86400,))
            services.chains.store.put_cached("hash-active", "generate", output_file_ids=[out.id])
            services.chains.store.touch_cached("hash-active", time.time())
            services.chains.store.put_cached("hash-dead", "generate", output_file_ids=[stale_out.id])
            services.files.delete(stale_out.id, delete_file=True)

            services.housekeeping()
            # Actively reused cached output MUST survive prune_unsaved()!
            assert services.files.get(out.id) is not None
            assert services.chains.store.get_cached("hash-active") is not None
            # Dead entry whose output file was deleted must be cleaned up
            assert services.chains.store.get_cached("hash-dead") is None

            # Bounded max_entries eviction on put_cached
            for idx in range(5):
                services.chains.store.put_cached(f"cap-{idx}", "prompt_upsample", output_prompt=f"p{idx}", max_entries=3)
                time.sleep(0.002)
            assert services.chains.store._db.execute("SELECT COUNT(*) FROM stage_cache").fetchone()[0] == 3
        finally:
            services.close()


def test_model_fingerprint_resolves_basenames_sharded_dirs_vae_and_loras(tmp_path, monkeypatch):
    """_default_model_fingerprint resolves short basenames, hashes sharded HF
    directory contents, and includes VAE + LoRA file identities."""

    import sys
    import types

    from remi.jobs import chains as chains_mod

    assert chains_mod.STAGE_HASH_VERSION >= 3

    vae_dir = tmp_path / "models" / "VAE"
    te_dir = tmp_path / "models" / "text_encoder"
    shard_dir = te_dir / "qwen3_sharded"
    lora_dir = tmp_path / "models" / "Lora"
    vae_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)
    lora_dir.mkdir(parents=True)

    vae_file = vae_dir / "custom.vae.safetensors"
    vae_file.write_bytes(b"vae-v1")
    mod_file = te_dir / "ae.safetensors"
    mod_file.write_bytes(b"ae-v1")
    shard_file = shard_dir / "model-00001-of-00002.safetensors"
    shard_file.write_bytes(b"shard-v1")
    (shard_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")
    lora_file = lora_dir / "style_lora.safetensors"
    lora_file.write_bytes(b"lora-v1")

    opts_values = {
        "sd_model_checkpoint": "",
        "forge_additional_modules": ["ae.safetensors", str(shard_dir)],
        "forge_preset": "anima",
        "forge_unet_storage_dtype": "Automatic",
        "sd_vae": "custom.vae.safetensors",
        "sd_lora": "None",
        "CLIP_stop_at_last_layers": 1,
        "emphasis": "Original",
        "eta_noise_seed_delta": 0,
        "randn_source": "GPU",
        "s_min_uncond": 0.0,
        "upcast_sampling": False,
        "extra_networks_default_multiplier": 1.0,
    }
    monkeypatch.setattr("remi.setting_values.setting", lambda name, default=None, **kw: opts_values.get(name, default))

    fake_paths = types.SimpleNamespace(models_path=str(tmp_path / "models"))
    fake_shared = types.SimpleNamespace(cmd_opts=types.SimpleNamespace(vae_dirs=[], text_encoder_dirs=[], lora_dir=None, lyco_dir_backcompat=None))
    monkeypatch.setitem(sys.modules, "modules.paths", fake_paths)
    monkeypatch.setitem(sys.modules, "modules.shared", fake_shared)
    monkeypatch.setattr("modules.paths", fake_paths, raising=False)
    monkeypatch.setattr("modules.shared", fake_shared, raising=False)

    fp1 = chains_mod._default_model_fingerprint()
    # Replacing short-basename module file changes fingerprint
    mod_file.write_bytes(b"ae-v2-modified!")
    fp2 = chains_mod._default_model_fingerprint()
    assert fp2 != fp1

    # Replacing shard inside sharded HF dir changes fingerprint
    shard_file.write_bytes(b"shard-v2-longer-weights")
    fp3 = chains_mod._default_model_fingerprint()
    assert fp3 != fp2

    # Replacing VAE file changes fingerprint
    vae_file.write_bytes(b"vae-v2-longer")
    fp4 = chains_mod._default_model_fingerprint()
    assert fp4 != fp3

    # Stage hash detects replaced LoRA file referenced in prompt
    files = FileStore(tmp_path / "files")
    jobs = JobStore(tmp_path / "jobs.db")
    bus = EventBus()
    queue = JobQueue(jobs, files, bus, executor=lambda j, c: None, dispatch=run_inline, lock=threading.Lock(), validate=lambda *a: None, state=FakeState, hash_layout=lambda t: "h")
    runner = ChainRunner(ChainStore(jobs), queue, files, bus, model_fingerprint=lambda: fp4)
    try:
        spec = StageSpec(kind="generate", tab="txt2img", core={"prompt": "fox <lora:style_lora:0.8>"}, scripts={"seed": {"seed": 42}})
        h1 = runner._stage_hash(spec, ("", [], None), fingerprint=fp4, chain_id="c1")
        lora_file.write_bytes(b"lora-v2-updated-weights!")
        h2 = runner._stage_hash(spec, ("", [], None), fingerprint=fp4, chain_id="c1")
        assert h1 != h2
    finally:
        runner.close()
        jobs.close()
        files.close()


def test_chain_store_nan_cursors_and_begin_immediate_update_stage(tmp_path):
    """ChainStore ignores NaN/Infinity cursors and update_stage is atomic."""

    import sqlite3

    import pytest

    from remi.files import encode_cursor

    with ChainStore(tmp_path / "chains.db") as store:
        chain = store.create_chain([StageSpec(kind="generate", core={"prompt": "p"})], name="One")
        assert len(store.list_chains(cursor=encode_cursor("nan", "z"))[0]) == 1
        assert len(store.list_chains(cursor=encode_cursor("inf", "z"))[0]) == 1

        store._db.execute("CREATE TEMP TRIGGER refuse_chain_update BEFORE UPDATE ON chains BEGIN SELECT RAISE(ABORT, 'refused'); END")
        with pytest.raises(sqlite3.IntegrityError):
            store.update_stage(chain.id, 0, status="running", error="partial")
        store._db.execute("DROP TRIGGER refuse_chain_update")
        assert store.get_chain(chain.id).stages[0].status == "pending"


def test_create_chain_idempotency_key(tmp_path):
    state = FakeState()
    with webui_state(), server_app(tmp_path, executor=StubExecutor(state, count=1, steps=0), state=state, validate=lambda *a: None) as server:
        payload = {"name": "Idemp", "run_now": False, "stages": [{"kind": "generate", "tab": "txt2img", "core": {"prompt": "fox"}}]}
        first = server.client.post("/api/v2/chains", json=payload, headers={"Idempotency-Key": "chain-k1"})
        assert first.status_code == 201
        again = server.client.post("/api/v2/chains", json=payload, headers={"Idempotency-Key": "chain-k1"})
        assert again.status_code == 201 and again.json()["id"] == first.json()["id"]
        assert server.client.get("/api/v2/chains").json()["total"] == 1

        # Reusing the key with a different payload returns 409 Conflict
        different = {**payload, "name": "Changed"}
        conflict = server.client.post("/api/v2/chains", json=different, headers={"Idempotency-Key": "chain-k1"})
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == "Idempotency-Key was already used with different parameters"
        assert conflict.json()["template"] == "Idempotency-Key was already used with different parameters"

        # A rejected 422 submission is not remembered under its key
        bad = server.client.post(
            "/api/v2/chains",
            content='{"name": "bad", "run_now": false, "stages": [{"kind": "generate", "core": {"cfg_scale": 1e999}}]}',
            headers={"Content-Type": "application/json", "Idempotency-Key": "chain-k2"},
        )
        assert bad.status_code == 422
        recovered = server.client.post("/api/v2/chains", json=payload, headers={"Idempotency-Key": "chain-k2"})
        assert recovered.status_code == 201 and recovered.json()["id"] != first.json()["id"]

        # Keys longer than 255 characters are rejected
        assert server.client.post("/api/v2/chains", json=payload, headers={"Idempotency-Key": "x" * 256}).status_code == 422


def test_preview_chain_refuses_non_finite_numbers(tmp_path):
    with webui_state(), server_app(tmp_path, validate=lambda tab, core, scripts: None) as server:
        headers = {"Content-Type": "application/json"}
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            body = f'{{"stages": [{{"kind": "generate", "core": {{"cfg_scale": {token}}}}}]}}'
            resp = server.client.post("/api/v2/chains/preview", content=body, headers=headers)
            assert resp.status_code == 422, (token, resp.status_code, resp.text)


def test_deep_nested_stages_never_raise_recursion_error(tmp_path):
    import pytest
    from fastapi.testclient import TestClient

    from remi.jobs.chains import ChainError

    deep_bad: object = float("nan")
    deep_ok: object = 1.0
    for _ in range(900):
        deep_bad = {"k": [deep_bad]}
        deep_ok = {"k": [deep_ok]}

    def nest_json(leaf: str, depth: int) -> str:
        raw = leaf
        for _ in range(depth):
            raw = f'{{"a": {raw}}}'
        return raw

    bad_1100 = nest_json("1e999", 1100)
    ok_1100 = nest_json("1.0", 1100)

    with webui_state(), server_app(tmp_path, validate=lambda tab, core, scripts: None) as server:
        with pytest.raises(ChainError) as exc:
            server.services.chains.plan_chain([StageSpec(kind="generate", core={"deep": deep_bad})])
        assert exc.value.status == 422
        assert len(server.services.chains.plan_chain([StageSpec(kind="generate", core={"deep": deep_ok})])) == 1

        client = TestClient(server.app, raise_server_exceptions=False, base_url="http://127.0.0.1:7860")
        headers = {"Content-Type": "application/json"}

        for path, bad_body, ok_body, ok_status in (
            (
                "/api/v2/chains/preview",
                f'{{"stages": [{{"kind": "generate", "core": {bad_1100}}}]}}',
                f'{{"stages": [{{"kind": "generate", "core": {ok_1100}}}]}}',
                200,
            ),
            (
                "/api/v2/chains",
                f'{{"name": "bad", "run_now": false, "stages": [{{"kind": "generate", "core": {bad_1100}}}]}}',
                f'{{"name": "ok", "run_now": false, "stages": [{{"kind": "generate", "core": {ok_1100}}}]}}',
                201,
            ),
            (
                "/api/v2/chains/templates",
                f'{{"name": "bad", "stages": [{{"kind": "generate", "core": {bad_1100}}}]}}',
                f'{{"name": "ok", "stages": [{{"kind": "generate", "core": {ok_1100}}}]}}',
                201,
            ),
        ):
            bad_resp = client.post(path, content=bad_body, headers=headers)
            assert bad_resp.status_code == 422, (path, bad_resp.status_code, bad_resp.text)
            ok_resp = client.post(path, content=ok_body, headers=headers)
            assert ok_resp.status_code == ok_status, (path, ok_resp.status_code, ok_resp.text)

        assert client.get("/api/v2/chains").status_code == 200
        assert client.get("/api/v2/chains/templates").status_code == 200


def test_create_chain_idempotency_key_deep_nesting(tmp_path):
    from fastapi.testclient import TestClient

    def nest_json(leaf: str, depth: int) -> str:
        raw = leaf
        for _ in range(depth):
            raw = f'{{"a": {raw}}}'
        return raw

    state = FakeState()
    with webui_state(), server_app(tmp_path, executor=StubExecutor(state, count=1, steps=0), state=state, validate=lambda *a: None) as server:
        client = TestClient(server.app, raise_server_exceptions=False, base_url="http://127.0.0.1:7860")

        for expected_total, depth in enumerate((260, 1100), start=1):
            for token in ("1e999", "NaN", "Infinity"):
                bad_body = f'{{"name": "bad-{depth}", "run_now": false, "stages": [{{"kind": "generate", "core": {nest_json(token, depth)}}}]}}'
                bad_resp = client.post(
                    "/api/v2/chains",
                    content=bad_body,
                    headers={"Content-Type": "application/json", "Idempotency-Key": f"deep-key-{depth}"},
                )
                assert bad_resp.status_code == 422, (depth, token, bad_resp.status_code, bad_resp.text)

            ok_body = f'{{"name": "ok-{depth}", "run_now": false, "stages": [{{"kind": "generate", "core": {nest_json("1.0", depth)}}}]}}'
            headers = {"Content-Type": "application/json", "Idempotency-Key": f"deep-key-{depth}"}
            created = client.post("/api/v2/chains", content=ok_body, headers=headers)
            assert created.status_code == 201, (depth, created.status_code, created.text)
            chain_id = created.json()["id"]

            replay = client.post("/api/v2/chains", content=ok_body, headers=headers)
            assert replay.status_code == 201, (depth, replay.status_code, replay.text)
            assert replay.json()["id"] == chain_id
            assert client.get("/api/v2/chains").json()["total"] == expected_total

            changed_body = f'{{"name": "ok-{depth}", "run_now": false, "stages": [{{"kind": "generate", "core": {nest_json("2.0", depth)}}}]}}'
            conflict = client.post("/api/v2/chains", content=changed_body, headers=headers)
            assert conflict.status_code == 409, (depth, conflict.status_code, conflict.text)
            assert conflict.json()["detail"] == "Idempotency-Key was already used with different parameters"
