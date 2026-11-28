"""End-to-end API tests: cross-feature interactions and multi-step workflows.

Every test exercises only live HTTP and WebSocket endpoints over TCP via the
``conftest.py`` fixtures and ``helpers`` namespace, with zero internal imports.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from collections.abc import Callable
from typing import Any

import pytest
import websockets.exceptions


def _set_cookie_attrs(response: Any) -> list[str]:
    header = response.headers.get("set-cookie", "")
    return [part.strip().lower() for part in header.split(";") if part.strip()]


def _poll_until(
    predicate: Callable[[], Any],
    *,
    timeout: float = 10.0,
    interval: float = 0.03,
    description: str = "condition",
) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout:.1f}s waiting for {description}")
        time.sleep(interval)


def _find_conflict_recipe_setup(client: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Discover a real conflict-bearing script/form setup from GET /api/v2/schema/txt2img."""

    schema_resp = client.get("/api/v2/schema/txt2img")
    assert schema_resp.status_code == 200
    schema = schema_resp.json()
    scripts_by_id = {entry["id"]: entry for entry in schema.get("scripts", [])}

    if "dynamic_shift" in scripts_by_id and "sampler" in scripts_by_id:
        draft_core = {"prompt": "1girl, test", "width": 64, "height": 64}
        draft_scripts = {"sampler": {"sampler_name": "Euler", "scheduler": "Karras", "steps": 2}}
        recipe_patch = {"dynamic_shift": {"enabled": True}}
        return draft_core, draft_scripts, recipe_patch, "sampler.scheduler"

    draft_core = {"prompt": "1girl, test", "width": 64, "height": 64, "enable_hr": False}
    draft_scripts = {"sampler": {"sampler_name": "Euler", "scheduler": "Normal", "steps": 2}}
    recipe_patch = {"sampler": {"steps": 2}}
    return draft_core, draft_scripts, recipe_patch, "__advice__"


# =============================================================================
#
# =============================================================================


def test_cross_forwarded_proto_secure_cookie_and_wildcard_cors(
    auth_server: Any,
    make_live_server: Callable[..., Any],
    helpers: Any,
) -> None:
    """X-Forwarded-Proto: https sets Secure cookie; wildcard CORS allows GET without credentials and blocks POST."""

    plain_client = auth_server.new_client()
    plain_resp = helpers.login(auth_server, "alice", "s3cret", client=plain_client)
    assert plain_resp.status_code == 204
    plain_attrs = _set_cookie_attrs(plain_resp)
    assert any(attr.startswith("remi_session=") for attr in plain_attrs)
    assert "httponly" in plain_attrs
    assert "secure" not in plain_attrs

    https_client = auth_server.new_client()
    https_resp = https_client.post(
        "/api/v2/auth/login",
        json={"username": "alice", "password": "s3cret"},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert https_resp.status_code == 204
    https_attrs = _set_cookie_attrs(https_resp)
    assert any(attr.startswith("remi_session=") for attr in https_attrs)
    assert "httponly" in https_attrs
    assert "secure" in https_attrs

    cors_srv = make_live_server(cmd_opts_overrides={"cors_allow_origins": "*"}, steps=1, step_time=0.01)
    foreign_origin = "https://third-party.example"

    get_resp = cors_srv.http.get(
        "/api/v2/app",
        headers={
            "Origin": foreign_origin,
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "cors",
        },
    )
    assert get_resp.status_code == 200
    assert get_resp.headers.get("access-control-allow-origin") == "*"
    assert get_resp.headers.get("access-control-allow-credentials") != "true"

    post_resp = cors_srv.http.post(
        "/api/v2/jobs",
        json={"tab": "txt2img", "core": {"prompt": "blocked"}, "scripts": {}},
        headers={
            "Origin": foreign_origin,
            "Sec-Fetch-Site": "cross-site",
        },
    )
    assert post_resp.status_code == 403


def test_cross_logout_revokes_session_and_disconnects_only_that_users_ws(
    auth_server: Any,
    helpers: Any,
) -> None:
    """POST /api/v2/auth/logout closes that session's WS with 1008 while keeping another user's WS open."""

    alice_client = auth_server.new_client()
    bob_client = auth_server.new_client()
    assert helpers.login(auth_server, "alice", "s3cret", client=alice_client).status_code == 204
    assert helpers.login(auth_server, "bob", "pw2", client=bob_client).status_code == 204

    ws_alice = auth_server.ws_connect(client=alice_client)
    ws_bob = auth_server.ws_connect(client=bob_client)

    assert helpers.recv_ws_until(ws_alice, lambda m: isinstance(m, dict) and m.get("type") == "hello")["type"] == "hello"
    assert helpers.recv_ws_until(ws_bob, lambda m: isinstance(m, dict) and m.get("type") == "hello")["type"] == "hello"
    helpers.drain_ws(ws_alice, idle_timeout=0.05)
    helpers.drain_ws(ws_bob, idle_timeout=0.05)

    saved_alice_cookie = dict(alice_client.cookies)
    logout_resp = alice_client.post("/api/v2/auth/logout")
    assert logout_resp.status_code == 204

    with pytest.raises(websockets.exceptions.ConnectionClosedError) as closed_exc:
        for _ in range(10):
            ws_alice.recv(timeout=2.0)
    assert closed_exc.value.rcvd is not None
    assert closed_exc.value.rcvd.code == 1008

    assert alice_client.get("/api/v2/options").status_code == 401
    replay_client = auth_server.new_client(cookies=saved_alice_cookie)
    assert replay_client.get("/api/v2/options").status_code == 401

    ws_bob.send(json.dumps({"type": "ping"}))
    pong = helpers.recv_ws_until(ws_bob, lambda m: isinstance(m, dict) and m.get("type") == "pong", timeout=3.0)
    assert pong["type"] == "pong"
    assert bob_client.get("/api/v2/options").status_code == 200


def test_cross_mid_job_ws_reconnect_receives_immediate_snapshot(
    gated_server: Any,
    helpers: Any,
) -> None:
    """Reconnecting WS mid-job immediately receives job.started and job.progress without waiting for the next step."""

    assert gated_server.gate is not None
    created = helpers.submit_txt2img(gated_server, prompt="mid-job reconnect", steps=2, wait=False)
    job_id = created["job_id"]
    job_seq = created["seq"]

    assert gated_server.executor.started.wait(timeout=5.0)
    running_detail = _poll_until(
        lambda: (
            d
            if (d := gated_server.http.get(f"/api/v2/jobs/{job_id}").json()).get("status") == "running"
            and d.get("progress") is not None
            else None
        ),
        timeout=5.0,
        description="running job with non-null progress",
    )
    assert running_detail["progress"]["progress"] > 0

    ws = gated_server.ws_connect()
    try:
        hello = helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "hello", timeout=3.0)
        assert "jobs.v1" in hello["features"]

        queue_evt = helpers.recv_ws_until(
            ws,
            lambda m: isinstance(m, dict) and m.get("type") == "queue.changed" and m.get("running") == job_id,
            timeout=3.0,
        )
        assert queue_evt["running"] == job_id

        started_evt = helpers.recv_ws_until(
            ws,
            lambda m: isinstance(m, dict) and m.get("type") == "job.started" and m.get("job_id") == job_id,
            timeout=3.0,
        )
        assert started_evt["seq"] == job_seq

        progress_evt = helpers.recv_ws_until(
            ws,
            lambda m: isinstance(m, dict) and m.get("type") == "job.progress" and m.get("job_id") == job_id,
            timeout=3.0,
        )
        assert progress_evt["progress"] > 0
    finally:
        gated_server.gate.set()

    done_evt = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == job_id,
        timeout=5.0,
    )
    assert done_evt["status"] == "completed"
    assert gated_server.wait_job(job_id)["status"] == "completed"


def test_cross_delete_output_keep_file_stays_excluded_after_refresh(
    live_server: Any,
    helpers: Any,
) -> None:
    """DELETE /api/v2/outputs/{id}?delete_file=false keeps file on disk while refresh skips re-indexing it."""

    sample_dir = live_server.outputs_dir / "txt2img-images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "excluded_on_refresh.png"
    sample_path.write_bytes(
        helpers.make_png_bytes(
            size=(40, 40),
            color=(90, 140, 210),
            text="excluded keep file test\nSteps: 4, Seed: 77",
        )
    )

    first_scan = live_server.http.post("/api/v2/outputs/refresh")
    assert first_scan.status_code == 200
    assert first_scan.json()["added"] == 1

    listed = live_server.http.get("/api/v2/outputs").json()
    match = next(item for item in listed["items"] if item["name"] == "excluded_on_refresh.png")
    file_id = match["id"]

    delete_resp = live_server.http.delete(f"/api/v2/outputs/{file_id}", params={"delete_file": "false"})
    assert delete_resp.status_code == 204
    assert sample_path.is_file()

    after_delete = live_server.http.get("/api/v2/outputs").json()
    assert all(item["id"] != file_id for item in after_delete["items"])
    assert live_server.http.get(f"/api/v2/files/{file_id}").status_code == 404

    rescan = live_server.http.post("/api/v2/outputs/refresh")
    assert rescan.status_code == 200
    assert rescan.json()["added"] == 0
    assert sample_path.is_file()
    assert live_server.http.get("/api/v2/outputs").json()["total"] == 0


def test_cross_housekeeping_retention_spares_referenced_and_curated(
    make_live_server: Callable[..., Any],
    helpers: Any,
) -> None:
    """Housekeeping preserves queued-job uploads and board/starred outputs past cutoff while pruning unreferenced ones."""

    srv = make_live_server(
        housekeeping=False,
        steps=1,
        step_time=0.01,
        options={"remi_upload_retention_days": 1},
    )

    board_job = helpers.submit_txt2img(srv, prompt="board kept")
    star_job = helpers.submit_txt2img(srv, prompt="starred kept")
    orphan_out_job = helpers.submit_txt2img(srv, prompt="expired unsaved")

    board_out_id = board_job["outputs"][0]["id"]
    star_out_id = star_job["outputs"][0]["id"]
    orphan_out_id = orphan_out_job["outputs"][0]["id"]

    board_resp = srv.http.post("/api/v2/boards", json={"name": "Retention Safe Board"})
    assert board_resp.status_code == 201
    board_id = board_resp.json()["id"]
    assert srv.http.post(f"/api/v2/boards/{board_id}/items", json={"file_ids": [board_out_id]}).status_code == 200
    assert srv.http.post("/api/v2/gallery/star", json={"ids": [star_out_id], "starred": True}).status_code == 200

    assert srv.http.post("/api/v2/queue/pause").status_code == 200
    queued_upload = helpers.upload_png(srv, color=(15, 30, 45), filename="queued_ref.png")
    expired_upload = helpers.upload_png(srv, color=(200, 100, 50), filename="expired_orphan.png")

    queued_job = helpers.submit_img2img(srv, queued_upload["id"], prompt="queued reference", wait=False)

    all_ids = [board_out_id, star_out_id, orphan_out_id, queued_upload["id"], expired_upload["id"]]
    srv.expire_files_and_run_housekeeping(all_ids)

    assert srv.http.get(f"/api/v2/files/{queued_upload['id']}").status_code == 200
    assert srv.http.get(f"/api/v2/files/{board_out_id}").status_code == 200
    assert srv.http.get(f"/api/v2/files/{star_out_id}").status_code == 200
    assert srv.http.get(f"/api/v2/files/{expired_upload['id']}").status_code == 404
    assert srv.http.get(f"/api/v2/files/{orphan_out_id}").status_code == 404

    assert srv.http.post("/api/v2/queue/resume").status_code == 200
    finished_queued = srv.wait_job(queued_job["job_id"])
    assert finished_queued["status"] == "completed"


def test_cross_forever_job_requeues_and_stop_forever_halts(
    live_server: Any,
    helpers: Any,
) -> None:
    """forever=true auto-requeues with retry_of until POST /api/v2/queue/stop-forever halts further requeuing."""

    created = helpers.submit_txt2img(live_server, prompt="forever loop", steps=1, forever=True, wait=False)
    first_job_id = created["job_id"]

    requeued = _poll_until(
        lambda: [
            item
            for item in live_server.http.get("/api/v2/jobs", params={"limit": 50}).json()["items"]
            if item.get("retry_of") == first_job_id or item.get("id") == first_job_id
        ],
        timeout=8.0,
        description="initial forever job",
    )
    assert len(requeued) >= 1

    def _has_requeued_job() -> list[dict[str, Any]] | None:
        items = live_server.http.get("/api/v2/jobs", params={"limit": 50}).json()["items"]
        requeued_items = [j for j in items if j.get("retry_of") is not None]
        return requeued_items or None

    _poll_until(
        _has_requeued_job,
        timeout=8.0,
        description="at least one auto-requeued retry_of job",
    )

    stop_resp = live_server.http.post("/api/v2/queue/stop-forever")
    assert stop_resp.status_code == 200

    _poll_until(
        lambda: (
            q
            if (q := live_server.http.get("/api/v2/queue").json()).get("size") == 0 and q.get("running") is None
            else None
        ),
        timeout=8.0,
        description="queue idle after stop-forever",
    )

    snapshot = live_server.http.get("/api/v2/jobs", params={"limit": 100}).json()
    count_at_idle = snapshot["total"]
    assert any(item.get("retry_of") == first_job_id for item in snapshot["items"])

    time.sleep(0.15)
    after_wait = live_server.http.get("/api/v2/jobs", params={"limit": 100}).json()
    assert after_wait["total"] == count_at_idle
    assert all(not item["forever"] for item in after_wait["items"])


def test_cross_star_job_and_clear_history_keeps_starred_and_gallery(
    live_server: Any,
    helpers: Any,
) -> None:
    """POST /api/v2/jobs/star + clear-history(keep_starred=true) keeps starred job and preserves all gallery outputs."""

    job_keep = helpers.submit_txt2img(live_server, prompt="star keep job")
    job_drop = helpers.submit_txt2img(live_server, prompt="unstarred drop job")

    keep_out_id = job_keep["outputs"][0]["id"]
    drop_out_id = job_drop["outputs"][0]["id"]

    star_resp = live_server.http.post("/api/v2/jobs/star", json={"ids": [job_keep["id"]], "starred": True})
    assert star_resp.status_code == 200
    assert star_resp.json()["count"] == 1

    clear_resp = live_server.http.post("/api/v2/jobs/clear-history", json={"keep_starred": True})
    assert clear_resp.status_code == 200
    assert clear_resp.json()["count"] == 1

    kept_resp = live_server.http.get(f"/api/v2/jobs/{job_keep['id']}")
    assert kept_resp.status_code == 200
    assert kept_resp.json()["starred"] is True
    assert live_server.http.get(f"/api/v2/jobs/{job_drop['id']}").status_code == 404

    assert live_server.http.get(f"/api/v2/gallery/{keep_out_id}").status_code == 200
    assert live_server.http.get(f"/api/v2/gallery/{drop_out_id}").status_code == 200


def test_cross_recipe_apply_resolves_conflicts_and_submits_job(
    tiny_anima_server: Any,
) -> None:
    """POST /api/v2/recipes/{id}/apply (check_conflicts=true) resolves conflicts and feeds directly into POST /api/v2/jobs."""

    draft_core, draft_scripts, recipe_scripts_patch, expected_key = _find_conflict_recipe_setup(tiny_anima_server.http)
    recipe_name = f"Conflict Recipe {int(time.monotonic() * 1000)}"

    created = tiny_anima_server.http.post(
        "/api/v2/recipes",
        json={
            "name": recipe_name,
            "description": "Triggers conflict lock resolution on apply",
            "model_family": "anima",
            "tab": "txt2img",
            "core_patch": {"cfg_scale": 4.5},
            "scripts_patch": recipe_scripts_patch,
            "tags": ["cross-feature", "conflict"],
        },
    )
    assert created.status_code == 201
    recipe_id = created.json()["id"]

    try:
        applied_resp = tiny_anima_server.http.post(
            f"/api/v2/recipes/{recipe_id}/apply",
            json={
                "tab": "txt2img",
                "core": draft_core,
                "scripts": draft_scripts,
                "check_conflicts": True,
            },
        )
        assert applied_resp.status_code == 200
        applied = applied_resp.json()
        assert applied["tab"] == "txt2img"
        assert expected_key in applied["conflicts"]
        if expected_key == "sampler.scheduler":
            assert applied["conflicts"]["sampler.scheduler"]["value"] == "Automatic"
            assert applied["scripts"]["sampler"]["scheduler"] == "Automatic"
            assert "sampler.scheduler" in applied["patch"]
            assert "__toast__" in applied["conflicts"]

        job_resp = tiny_anima_server.http.post(
            "/api/v2/jobs",
            json={
                "tab": applied["tab"],
                "core": applied["core"],
                "scripts": applied["scripts"],
            },
        )
        assert job_resp.status_code == 201
        job_id = job_resp.json()["job_id"]
        finished = tiny_anima_server.wait_job(job_id, timeout=60.0)
        assert finished["status"] == "completed"
        assert len(finished["outputs"]) >= 1
    finally:
        tiny_anima_server.http.delete(f"/api/v2/recipes/{recipe_id}")


def test_cross_chain_delete_output_invalidates_cache_and_cancel_emits_ws(
    live_server: Any,
    gated_server: Any,
    helpers: Any,
) -> None:
    """Deleting stage 0 output invalidates chain cache preview; cancelling a running chain interrupts job and emits WS chain.updated."""

    stages = [
        {
            "kind": "generate",
            "name": "Stage 0 Gen",
            "tab": "txt2img",
            "core": {"prompt": "cache invalidation root", "width": 64, "height": 64},
            "scripts": {"seed": {"seed": 1001}},
        },
        {
            "kind": "refine",
            "name": "Stage 1 Refine",
            "tab": "img2img",
            "core": {"prompt": "cache invalidation refine", "denoising_strength": 0.3, "width": 64, "height": 64},
            "scripts": {"seed": {"seed": 1002}},
        },
    ]

    chain_resp = live_server.http.post("/api/v2/chains", json={"name": "Cache Invalidation Chain", "stages": stages})
    assert chain_resp.status_code == 201
    chain_id = chain_resp.json()["id"]
    done_chain = live_server.wait_chain(chain_id)
    assert done_chain["status"] == "completed"
    stage0_out_id = done_chain["stages"][0]["output_file_ids"][0]

    hit_preview = live_server.http.post("/api/v2/chains/preview", json={"stages": stages, "chain_id": chain_id})
    assert hit_preview.status_code == 200
    assert [st["cache_hit"] for st in hit_preview.json()["stages"]] == [True, True]

    del_resp = live_server.http.delete(f"/api/v2/outputs/{stage0_out_id}", params={"delete_file": "true"})
    assert del_resp.status_code == 204

    miss_preview = live_server.http.post("/api/v2/chains/preview", json={"stages": stages, "chain_id": chain_id})
    assert miss_preview.status_code == 200
    assert [st["cache_hit"] for st in miss_preview.json()["stages"]] == [False, False]

    ws = gated_server.ws_connect()
    helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "hello", timeout=3.0)
    helpers.drain_ws(ws, idle_timeout=0.05)

    gated_resp = gated_server.http.post(
        "/api/v2/chains",
        json={"name": "Cancel Running Chain", "stages": stages, "run_now": True},
    )
    assert gated_resp.status_code == 201
    gated_chain_id = gated_resp.json()["id"]

    assert gated_server.executor.started.wait(timeout=5.0)
    running_chain = _poll_until(
        lambda: (
            c
            if (c := gated_server.http.get(f"/api/v2/chains/{gated_chain_id}").json()).get("status") == "running"
            and c["stages"][0].get("job_id")
            else None
        ),
        timeout=5.0,
        description="running chain stage 0 job",
    )
    stage0_job_id = running_chain["stages"][0]["job_id"]

    cancel_resp = gated_server.http.post(f"/api/v2/chains/{gated_chain_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    cancelled_ws = helpers.recv_ws_until(
        ws,
        lambda m: (
            isinstance(m, dict)
            and m.get("type") == "chain.updated"
            and m.get("chain_id") == gated_chain_id
            and m.get("status") == "cancelled"
        ),
        timeout=5.0,
    )
    assert cancelled_ws["stages"][0]["status"] == "cancelled"

    stage0_job = gated_server.wait_job(stage0_job_id)
    assert stage0_job["status"] in ("interrupted", "cancelled")


def test_cross_delete_ancestor_updates_lineage_and_removes_from_board(
    live_server: Any,
    helpers: Any,
) -> None:
    """Deleting an ancestor output updates descendant lineage gracefully and removes the deleted output from boards."""

    root_job = helpers.submit_txt2img(live_server, prompt="lineage root")
    root_id = root_job["outputs"][0]["id"]

    child_job = helpers.submit_img2img(live_server, root_id, prompt="lineage child")
    child_id = child_job["outputs"][0]["id"]

    board_resp = live_server.http.post("/api/v2/boards", json={"name": "Lineage Board"})
    assert board_resp.status_code == 201
    board_id = board_resp.json()["id"]

    add_resp = live_server.http.post(f"/api/v2/boards/{board_id}/items", json={"file_ids": [root_id, child_id]})
    assert add_resp.status_code == 200
    assert add_resp.json()["count"] == 2

    lineage_before = live_server.http.get(f"/api/v2/gallery/{child_id}/lineage")
    assert lineage_before.status_code == 200
    ancestors_before = lineage_before.json()["ancestors"]
    assert any(node["file_id"] == root_id and node["exists"] is True for node in ancestors_before)

    del_resp = live_server.http.post("/api/v2/gallery/delete", json={"ids": [root_id], "delete_files": True})
    assert del_resp.status_code == 200
    assert del_resp.json()["count"] == 1

    board_after = live_server.http.get(f"/api/v2/boards/{board_id}")
    assert board_after.status_code == 200
    assert board_after.json()["item_count"] == 1
    assert board_after.json()["cover_file_id"] == child_id

    board_gallery = live_server.http.get("/api/v2/gallery", params={"board_id": board_id}).json()
    assert [item["id"] for item in board_gallery["items"]] == [child_id]

    lineage_after = live_server.http.get(f"/api/v2/gallery/{child_id}/lineage")
    assert lineage_after.status_code == 200
    lineage_data = lineage_after.json()
    assert lineage_data["item"]["file_id"] == child_id
    assert lineage_data["item"]["exists"] is True
    for node in lineage_data["ancestors"]:
        if node["file_id"] == root_id:
            assert node["exists"] is False


def test_cross_forge_preset_action_updates_schema_defaults_and_capabilities(
    live_server: Any,
) -> None:
    """POST /api/v2/actions/core/__forge_preset__ updates preset and reflects new defaults and capabilities in schema."""

    presets = live_server.http.get("/api/v2/presets").json()
    assert "anima" in presets["forge"]
    assert "flux" in presets["forge"]

    anima_action = live_server.http.post(
        "/api/v2/actions/core/__forge_preset__",
        json={"tab": "txt2img", "values": {}, "preset": "anima"},
    )
    assert anima_action.status_code == 200
    anima_patch = anima_action.json()
    assert anima_patch["__settings__"]["forge_preset"] == "anima"
    assert anima_patch["__capabilities__"]["preset"] == "anima"
    assert isinstance(anima_patch["__advice__"], list)

    anima_schema = live_server.http.get("/api/v2/schema/txt2img").json()
    assert anima_schema["capabilities"]["preset"] == "anima"
    anima_caps = set(anima_schema["capabilities"]["capabilities"])
    anima_defaults = anima_schema["core"]["defaults"]
    anima_sampler_defaults = next(s["defaults"] for s in anima_schema["scripts"] if s["id"] == "sampler")

    flux_action = live_server.http.post(
        "/api/v2/actions/core/__forge_preset__",
        json={"tab": "txt2img", "values": {}, "preset": "flux"},
    )
    assert flux_action.status_code == 200
    assert flux_action.json()["__capabilities__"]["preset"] == "flux"

    flux_schema = live_server.http.get("/api/v2/schema/txt2img").json()
    assert flux_schema["capabilities"]["preset"] == "flux"
    flux_caps = set(flux_schema["capabilities"]["capabilities"])
    flux_defaults = flux_schema["core"]["defaults"]
    flux_sampler_defaults = next(s["defaults"] for s in flux_schema["scripts"] if s["id"] == "sampler")

    assert anima_caps != flux_caps
    assert anima_defaults["cfg_scale"] != flux_defaults["cfg_scale"] or anima_sampler_defaults != flux_sampler_defaults


def test_cross_sdapi_options_sync_and_interrupt_stops_v2_job(
    live_server: Any,
    tiny_anima_server: Any,
    helpers: Any,
) -> None:
    """POST /sdapi/v1/options syncs to GET /api/v2/options + WS options.changed; /sdapi/v1/interrupt stops /api/v2/jobs."""

    ws = live_server.ws_connect()
    helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "hello", timeout=3.0)
    helpers.drain_ws(ws, idle_timeout=0.05)

    set_opt = live_server.http.post("/sdapi/v1/options", json={"CLIP_stop_at_last_layers": 3})
    assert set_opt.status_code == 200

    v2_opts = live_server.http.get("/api/v2/options").json()
    assert v2_opts["values"]["CLIP_stop_at_last_layers"] == 3

    opt_evt = helpers.recv_ws_until(
        ws,
        lambda m: (
            isinstance(m, dict)
            and m.get("type") == "options.changed"
            and "CLIP_stop_at_last_layers" in m.get("keys", [])
        ),
        timeout=3.0,
    )
    assert "CLIP_stop_at_last_layers" in opt_evt["keys"]

    created = helpers.submit_txt2img(
        tiny_anima_server,
        prompt="sdapi interrupt target",
        width=256,
        height=256,
        steps=20,
        seed=9001,
        core={"batch_count": 3},
        wait=False,
    )
    job_id = created["job_id"]
    _poll_until(
        lambda: (
            (job_data := tiny_anima_server.http.get(f"/api/v2/jobs/{job_id}").json()).get("status") == "running"
            and (prog := job_data.get("progress")) is not None
            and prog.get("progress", 0) > 0
        ),
        timeout=20.0,
        interval=0.02,
        description="tiny_anima job sampling progress > 0",
    )

    interrupt_resp = tiny_anima_server.http.post("/sdapi/v1/interrupt")
    assert interrupt_resp.status_code == 200

    stopped_job = tiny_anima_server.wait_job(job_id, timeout=30.0)
    assert stopped_job["status"] == "interrupted"


# =============================================================================
#
# =============================================================================


def test_workflow_authenticated_workbench_session(
    auth_server: Any,
    helpers: Any,
) -> None:
    """Full authenticated session: status -> login -> WS hello/preview.jpeg -> options -> job + binary preview -> visibility -> logout."""

    client = auth_server.new_client()

    unauth_status = client.get("/api/v2/auth/status")
    assert unauth_status.status_code == 200
    assert unauth_status.json()["required"] is True
    assert unauth_status.json()["user"] is None

    login_resp = helpers.login(auth_server, "alice", "s3cret", client=client)
    assert login_resp.status_code == 204

    auth_status = client.get("/api/v2/auth/status")
    assert auth_status.status_code == 200
    assert auth_status.json()["user"] == "alice"

    ws = auth_server.ws_connect(client=client)
    hello = helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "hello", timeout=3.0)
    assert "preview.jpeg" in hello["features"]
    helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "queue.changed", timeout=3.0)

    ws.send(json.dumps({"type": "hello", "features": ["events.v1", "preview.binary", "preview.jpeg"], "visible": True}))

    before_prog = bool(client.get("/api/v2/options").json()["values"]["show_progressbar"])
    patch_opts = client.patch("/api/v2/options", json={"show_progressbar": not before_prog})
    assert patch_opts.status_code == 200
    assert patch_opts.json()["values"]["show_progressbar"] is (not before_prog)

    opt_evt = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "options.changed" and "show_progressbar" in m.get("keys", []),
        timeout=3.0,
    )
    assert "show_progressbar" in opt_evt["keys"]

    created = helpers.submit_txt2img(auth_server, prompt="workbench preview session", steps=2, wait=False, client=client)
    job_id = created["job_id"]
    job_seq = created["seq"]

    done_msg, stream = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == job_id,
        timeout=8.0,
        collect_all=True,
    )
    assert done_msg["status"] == "completed"
    json_types = [m["type"] for m in stream if isinstance(m, dict)]
    assert "job.queued" in json_types
    assert "job.started" in json_types

    binary_frames = [helpers.parse_preview_frame(m) for m in stream if isinstance(m, bytes)]
    assert len(binary_frames) >= 1
    assert all(frame["type"] == 1 and frame["format"] == "jpeg" and frame["job_seq"] == job_seq for frame in binary_frames)
    assert all(frame["payload"].startswith(b"\xff\xd8") for frame in binary_frames)

    ws.send(json.dumps({"type": "visibility", "visible": False}))
    ws.send(json.dumps({"type": "ping"}))
    assert helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "pong", timeout=3.0)["type"] == "pong"

    hidden_created = helpers.submit_txt2img(
        auth_server,
        prompt="workbench hidden preview session",
        steps=2,
        wait=False,
        client=client,
    )
    hidden_job_id = hidden_created["job_id"]
    hidden_done, hidden_stream = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == hidden_job_id,
        timeout=8.0,
        collect_all=True,
    )
    assert hidden_done["status"] == "completed"
    hidden_stream.extend(helpers.drain_ws(ws, idle_timeout=0.05))
    assert any(
        isinstance(m, dict) and m.get("type") == "job.started" and m.get("job_id") == hidden_job_id
        for m in hidden_stream
    )
    assert not any(isinstance(m, bytes) for m in hidden_stream)

    ws.send(json.dumps({"type": "visibility", "visible": True}))
    ws.send(json.dumps({"type": "ping"}))
    assert helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "pong", timeout=3.0)["type"] == "pong"

    logout_resp = client.post("/api/v2/auth/logout")
    assert logout_resp.status_code == 204

    with pytest.raises(websockets.exceptions.ConnectionClosedError) as closed_exc:
        for _ in range(10):
            ws.recv(timeout=2.0)
    assert closed_exc.value.rcvd is not None
    assert closed_exc.value.rcvd.code == 1008

    assert client.get("/api/v2/options").status_code == 401


def test_workflow_recipe_to_generation_to_compare_workbench(
    tiny_anima_server: Any,
) -> None:
    """Recipe create -> export -> delete -> import -> apply with conflicts -> A/B jobs -> compare -> adopt B -> job C."""

    draft_core, draft_scripts, recipe_scripts_patch, _ = _find_conflict_recipe_setup(tiny_anima_server.http)
    recipe_name = f"Workbench Recipe {int(time.monotonic() * 1000)}"
    created = tiny_anima_server.http.post(
        "/api/v2/recipes",
        json={
            "name": recipe_name,
            "description": "Portrait baseline with conflict check",
            "model_family": "anima",
            "tab": "txt2img",
            "core_patch": {"prompt": "1girl, studio portrait, soft light", "cfg_scale": 4.0},
            "scripts_patch": recipe_scripts_patch,
            "tags": ["workflow", "portrait"],
        },
    )
    assert created.status_code == 201
    original_id = created.json()["id"]

    imported_id: str | None = None
    try:
        exported = tiny_anima_server.http.get("/api/v2/recipes/export")
        assert exported.status_code == 200
        export_payload = exported.json()
        assert export_payload["version"] == 1
        exported_entry = next(r for r in export_payload["recipes"] if r["name"] == recipe_name)

        del_resp = tiny_anima_server.http.delete(f"/api/v2/recipes/{original_id}")
        assert del_resp.status_code == 204
        assert tiny_anima_server.http.get(f"/api/v2/recipes/{original_id}").status_code == 404

        reimported = tiny_anima_server.http.post(
            "/api/v2/recipes/import",
            json={"version": 1, "recipes": [exported_entry], "overwrite": False},
        )
        assert reimported.status_code == 200
        assert reimported.json()["added"] == 1
        imported_id = reimported.json()["items"][0]["id"]

        applied = tiny_anima_server.http.post(
            f"/api/v2/recipes/{imported_id}/apply",
            json={
                "tab": "txt2img",
                "core": draft_core,
                "scripts": draft_scripts,
                "check_conflicts": True,
            },
        )
        assert applied.status_code == 200
        applied_data = applied.json()

        scripts_a = {k: dict(v) for k, v in applied_data["scripts"].items()}
        scripts_a["seed"] = {"seed": 111}
        scripts_a.setdefault("sampler", {})["steps"] = 2
        job_a_resp = tiny_anima_server.http.post(
            "/api/v2/jobs",
            json={"tab": "txt2img", "core": applied_data["core"], "scripts": scripts_a},
        )
        assert job_a_resp.status_code == 201
        job_a = tiny_anima_server.wait_job(job_a_resp.json()["job_id"], timeout=60.0)
        assert job_a["status"] == "completed"
        out_a = job_a["outputs"][0]["id"]

        core_b = {**applied_data["core"], "prompt": "1girl, neon cyberpunk portrait, dramatic light", "cfg_scale": 6.5}
        scripts_b = {k: dict(v) for k, v in scripts_a.items()}
        scripts_b["seed"] = {"seed": 222}
        scripts_b["sampler"] = {**scripts_b.get("sampler", {}), "steps": 3}
        job_b_resp = tiny_anima_server.http.post(
            "/api/v2/jobs",
            json={"tab": "txt2img", "core": core_b, "scripts": scripts_b},
        )
        assert job_b_resp.status_code == 201
        job_b = tiny_anima_server.wait_job(job_b_resp.json()["job_id"], timeout=60.0)
        assert job_b["status"] == "completed"
        out_b = job_b["outputs"][0]["id"]

        cmp_resp = tiny_anima_server.http.post("/api/v2/gallery/compare", json={"id_a": out_a, "id_b": out_b})
        assert cmp_resp.status_code == 200
        cmp_data = cmp_resp.json()
        assert cmp_data["identical"] is False
        assert len(cmp_data["core_diffs"]) >= 1
        adopt_b = cmp_data["adopt_b_patch"]
        assert adopt_b["core"]["prompt"] == core_b["prompt"]
        assert adopt_b["core"]["cfg_scale"] == 6.5
        assert adopt_b["scripts"]["sampler"]["steps"] == 3
        assert adopt_b["scripts"]["seed"]["seed"] == 222

        core_c = {**applied_data["core"], **adopt_b["core"]}
        scripts_c = {k: dict(v) for k, v in scripts_a.items()}
        for sid, spatch in adopt_b["scripts"].items():
            scripts_c[sid] = {**scripts_c.get(sid, {}), **spatch}
        job_c_resp = tiny_anima_server.http.post(
            "/api/v2/jobs",
            json={"tab": "txt2img", "core": core_c, "scripts": scripts_c},
        )
        assert job_c_resp.status_code == 201
        job_c = tiny_anima_server.wait_job(job_c_resp.json()["job_id"], timeout=60.0)
        assert job_c["status"] == "completed"

        job_c_detail = tiny_anima_server.http.get(f"/api/v2/jobs/{job_c['id']}").json()
        assert job_c_detail["core"]["prompt"] == core_b["prompt"]
        assert job_c_detail["core"]["cfg_scale"] == 6.5
        assert job_c_detail["scripts"]["sampler"]["steps"] == 3
        assert job_c_detail["scripts"]["seed"]["seed"] == 222

        cmp_bc = tiny_anima_server.http.post(
            "/api/v2/gallery/compare",
            json={"id_a": out_b, "id_b": job_c["outputs"][0]["id"]},
        )
        assert cmp_bc.status_code == 200
        assert cmp_bc.json()["identical"] is True
    finally:
        if imported_id is not None:
            tiny_anima_server.http.delete(f"/api/v2/recipes/{imported_id}")
        tiny_anima_server.http.delete(f"/api/v2/recipes/{original_id}")


def test_workflow_multistage_pipeline_partial_rerun_and_stage_cache(
    live_server: Any,
) -> None:
    """Save chain template -> run 3-stage pipeline -> inspect lineage -> tweak stage 2 & rerun (stages 0 & 1 cached) -> delete."""

    stages = [
        {
            "kind": "generate",
            "name": "Stage 0 Generate",
            "tab": "txt2img",
            "core": {"prompt": "1girl, alpine lake", "width": 64, "height": 64},
            "scripts": {"seed": {"seed": 5001}},
        },
        {
            "kind": "refine",
            "name": "Stage 1 Refine",
            "tab": "img2img",
            "core": {"prompt": "1girl, alpine lake, crisp detail", "denoising_strength": 0.35, "width": 64, "height": 64},
            "scripts": {"seed": {"seed": 5002}},
        },
        {
            "kind": "finish",
            "name": "Stage 2 Finish",
            "tab": "extras",
            "core": {"mode": "Single Image"},
            "scripts": {},
        },
    ]

    tmpl_resp = live_server.http.post(
        "/api/v2/chains/templates",
        json={
            "name": "3-Stage Portrait Finish",
            "description": "Generate, refine, finish",
            "model_family": "anima",
            "stages": stages,
            "tags": ["workflow", "pipeline"],
        },
    )
    assert tmpl_resp.status_code == 201
    template = tmpl_resp.json()
    tmpl_id = template["id"]

    listed_tmpls = live_server.http.get("/api/v2/chains/templates").json()
    assert any(item["id"] == tmpl_id for item in listed_tmpls["items"])

    chain_resp = live_server.http.post(
        "/api/v2/chains",
        json={"name": "Three-Stage Run", "stages": template["stages"], "run_now": True},
    )
    assert chain_resp.status_code == 201
    chain_id = chain_resp.json()["id"]
    done_chain = live_server.wait_chain(chain_id)
    assert done_chain["status"] == "completed"
    assert [st["status"] for st in done_chain["stages"]] == ["completed", "completed", "completed"]

    s0_out = done_chain["stages"][0]["output_file_ids"][0]
    s1_out = done_chain["stages"][1]["output_file_ids"][0]
    s2_out = done_chain["stages"][2]["output_file_ids"][0]
    s2_orig_job = done_chain["stages"][2]["job_id"]

    s1_job_detail = live_server.http.get(f"/api/v2/jobs/{done_chain['stages'][1]['job_id']}").json()
    s2_job_detail = live_server.http.get(f"/api/v2/jobs/{s2_orig_job}").json()
    assert s1_job_detail["core"]["init_img"] == s0_out
    assert s2_job_detail["core"]["image"] == s1_out

    lineage = live_server.http.get(f"/api/v2/gallery/{s2_out}/lineage")
    assert lineage.status_code == 200
    ancestors = lineage.json()["ancestors"]
    assert any(
        a["file_id"] == s1_out and a["chain_id"] == chain_id and a["relation"] in ("chain_stage", "init_img") and a["depth"] == 1
        for a in ancestors
    )
    assert any(
        a["file_id"] == s0_out and a["chain_id"] == chain_id and a["relation"] in ("chain_stage", "init_img") and a["depth"] == 2
        for a in ancestors
    )

    tweaked_stages = [dict(st) for st in template["stages"]]
    tweaked_stages[2] = {**tweaked_stages[2], "name": "Stage 2 Finish Tweaked", "config": {"revision": 2}}

    rerun_resp = live_server.http.post(f"/api/v2/chains/{chain_id}/rerun", json={"stages": tweaked_stages})
    assert rerun_resp.status_code == 200
    rerun_done = live_server.wait_chain(chain_id)
    assert rerun_done["status"] == "completed"

    st0, st1, st2 = rerun_done["stages"]
    assert st0["status"] == "cached" and st0["cached"] is True
    assert st1["status"] == "cached" and st1["cached"] is True
    assert st0["output_file_ids"] == [s0_out]
    assert st1["output_file_ids"] == [s1_out]
    assert st2["status"] == "completed" and st2["cached"] is False
    assert st2["job_id"] != s2_orig_job

    assert live_server.http.delete(f"/api/v2/chains/{chain_id}").status_code == 204
    assert live_server.http.get(f"/api/v2/chains/{chain_id}").status_code == 404
    assert live_server.http.delete(f"/api/v2/chains/templates/{tmpl_id}").status_code == 204


def test_workflow_img2img_and_gallery_curation_flow(
    live_server: Any,
    helpers: Any,
) -> None:
    """Upload PNG -> img2img -> upscale_from -> thumbs & pnginfo -> board -> star & save zip -> download zip -> delete unstarred."""

    upload = helpers.upload_png(live_server, size=(48, 48), color=(30, 110, 190), filename="source_sketch.png")
    upload_id = upload["id"]

    i2i_job = helpers.submit_img2img(live_server, upload_id, prompt="curated img2img pass")
    assert i2i_job["status"] == "completed"
    i2i_out_id = i2i_job["outputs"][0]["id"]

    up_job = helpers.submit_txt2img(
        live_server,
        prompt="curated hires upscale",
        core={"enable_hr": True, "hr_scale": 2.0},
        upscale_from=i2i_out_id,
    )
    assert up_job["status"] == "completed"
    up_out_id = up_job["outputs"][0]["id"]

    thumb_resp = live_server.http.get(f"/api/v2/thumbs/{up_out_id}", params={"w": 200})
    assert thumb_resp.status_code == 200
    assert thumb_resp.headers.get("content-type", "").startswith("image/webp")
    etag = thumb_resp.headers.get("etag")
    assert etag
    assert live_server.http.get(f"/api/v2/thumbs/{up_out_id}", params={"w": 200}, headers={"If-None-Match": etag}).status_code == 304

    meta_resp = live_server.http.post("/api/v2/pnginfo", json={"file_id": up_out_id})
    assert meta_resp.status_code == 200
    meta = meta_resp.json()
    assert "curated hires upscale" in (meta.get("infotext") or "")
    assert "curated hires upscale" in meta.get("params", {}).get("Prompt", "")
    assert len(meta.get("entries", [])) >= 1
    assert len(meta.get("chunks", [])) >= 1

    board_resp = live_server.http.post("/api/v2/boards", json={"name": "Gallery Curation", "color": "#336699"})
    assert board_resp.status_code == 201
    board_id = board_resp.json()["id"]
    assert live_server.http.post(f"/api/v2/boards/{board_id}/items", json={"file_ids": [i2i_out_id, up_out_id]}).json()["count"] == 2

    assert live_server.http.post("/api/v2/gallery/star", json={"ids": [up_out_id], "starred": True}).status_code == 200

    save_resp = live_server.http.post("/api/v2/outputs/save", json={"ids": [up_out_id], "zip": True})
    assert save_resp.status_code == 200
    save_data = save_resp.json()
    assert len(save_data["saved"]) == 1
    zip_id = save_data["zip"]
    assert zip_id

    dl_resp = live_server.http.get(f"/api/v2/files/{zip_id}", params={"download": "1"})
    assert dl_resp.status_code == 200
    assert "attachment" in dl_resp.headers.get("content-disposition", "").lower()
    with io.BytesIO(dl_resp.content) as buf, zipfile.ZipFile(buf, "r") as zf:
        names = zf.namelist()
        assert len(names) == 1
        assert names[0].lower().endswith(".png")

    del_unstarred = live_server.http.post("/api/v2/gallery/delete", json={"ids": [i2i_out_id], "delete_files": True})
    assert del_unstarred.status_code == 200
    assert del_unstarred.json()["count"] == 1

    assert live_server.http.get(f"/api/v2/gallery/{i2i_out_id}").status_code == 404
    kept_item = live_server.http.get(f"/api/v2/gallery/{up_out_id}")
    assert kept_item.status_code == 200
    assert kept_item.json()["starred"] is True


def test_workflow_ops_extensions_and_governance_flow(
    tiny_anima_server: Any,
) -> None:
    """Inspect /workers & /extensions/diagnostics -> save config-state -> inspect detail -> restore -> verify options & sysinfo -> delete."""

    workers_resp = tiny_anima_server.http.get("/api/v2/workers")
    assert workers_resp.status_code == 200
    workers_data = workers_resp.json()
    assert "mode" in workers_data
    assert isinstance(workers_data["dispatch_active"], bool)
    assert len(workers_data["workers"]) >= 1
    assert "available" in workers_data["mlx"]

    diag_resp = tiny_anima_server.http.get("/api/v2/extensions/diagnostics")
    assert diag_resp.status_code == 200
    diag = diag_resp.json()
    assert isinstance(diag["api_version"], str) and diag["api_version"]
    assert isinstance(diag["schema_version"], int) and diag["schema_version"] >= 1
    assert diag["summary"]["total"] >= 1
    assert len(diag["items"]) == diag["summary"]["total"]

    save_cs = tiny_anima_server.http.post("/api/v2/extensions/config-states", json={"name": "governance-backup"})
    assert save_cs.status_code == 201
    state_id = save_cs.json()["id"]

    try:
        detail_cs = tiny_anima_server.http.get(f"/api/v2/extensions/config-states/{state_id}")
        assert detail_cs.status_code == 200
        detail_data = detail_cs.json()
        assert detail_data["id"] == state_id
        assert "saved" in detail_data["webui"] and "current" in detail_data["webui"]
        assert len(detail_data["items"]) >= 1

        restore_resp = tiny_anima_server.http.post(
            f"/api/v2/extensions/config-states/{state_id}/restore",
            json={"what": "extensions"},
        )
        assert restore_resp.status_code == 200
        assert restore_resp.json()["reload"] is True

        options_data = tiny_anima_server.http.get("/api/v2/options").json()
        assert state_id in options_data["values"]["restore_config_state_file"]

        sysinfo_json = tiny_anima_server.http.get("/api/v2/sysinfo")
        assert sysinfo_json.status_code == 200
        raw_sysinfo = tiny_anima_server.http.get("/internal/sysinfo")
        assert raw_sysinfo.status_code == 200

        valid_check = tiny_anima_server.http.post("/api/v2/sysinfo/check", json={"text": raw_sysinfo.text})
        assert valid_check.status_code == 200
        assert valid_check.json()["valid"] is True

        tampered_check = tiny_anima_server.http.post("/api/v2/sysinfo/check", json={"text": raw_sysinfo.text + "\n"})
        assert tampered_check.status_code == 200
        assert tampered_check.json()["valid"] is False
    finally:
        tiny_anima_server.http.patch("/api/v2/options", json={"restore_config_state_file": ""})
        del_cs = tiny_anima_server.http.delete(f"/api/v2/extensions/config-states/{state_id}")
        assert del_cs.status_code == 204
        assert tiny_anima_server.http.get(f"/api/v2/extensions/config-states/{state_id}").status_code == 404


def test_workflow_real_model_tiny_anima_cross_api_flow(
    tiny_anima_server: Any,
) -> None:
    """Real TinyAnima cross-API flow: /sdapi/v1/{samplers,script-info,txt2img,png-info,img2img} -> /api/v2/{outputs/refresh,gallery,options}."""

    samplers_resp = tiny_anima_server.http.get("/sdapi/v1/samplers")
    assert samplers_resp.status_code == 200
    sampler_names = {item["name"] for item in samplers_resp.json()}
    assert "Euler" in sampler_names

    script_info_resp = tiny_anima_server.http.get("/sdapi/v1/script-info")
    assert script_info_resp.status_code == 200
    assert isinstance(script_info_resp.json(), list)

    marker = f"tiny_anima_cross_api_{int(time.monotonic() * 1000)}"
    t2i_resp = tiny_anima_server.http.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": f"1girl, {marker}",
            "steps": 2,
            "width": 64,
            "height": 64,
            "seed": 4242,
            "sampler_name": "Euler",
            "save_images": True,
            "send_images": True,
        },
    )
    assert t2i_resp.status_code == 200
    t2i_data = t2i_resp.json()
    assert len(t2i_data["images"]) >= 1
    assert marker in t2i_data["info"]

    png_info_resp = tiny_anima_server.http.post("/sdapi/v1/png-info", json={"image": t2i_data["images"][0]})
    assert png_info_resp.status_code == 200
    png_info = png_info_resp.json()
    assert marker in png_info["info"]
    assert png_info["parameters"]["Steps"] == "2"
    assert png_info["parameters"]["Seed"] == "4242"

    i2i_resp = tiny_anima_server.http.post(
        "/sdapi/v1/img2img",
        json={
            "init_images": [t2i_data["images"][0]],
            "prompt": f"1girl, {marker}_refined",
            "steps": 2,
            "width": 64,
            "height": 64,
            "denoising_strength": 0.4,
            "sampler_name": "Euler",
            "save_images": True,
            "send_images": True,
        },
    )
    assert i2i_resp.status_code == 200
    assert len(i2i_resp.json()["images"]) >= 1

    refresh_resp = tiny_anima_server.http.post("/api/v2/outputs/refresh")
    assert refresh_resp.status_code == 200
    assert refresh_resp.json()["added"] >= 2

    gallery_resp = tiny_anima_server.http.get("/api/v2/gallery", params={"q": marker})
    assert gallery_resp.status_code == 200
    gallery_data = gallery_resp.json()
    assert gallery_data["total"] >= 2
    assert any(f"{marker}_refined" in (item.get("infotext") or "") for item in gallery_data["items"])

    sdapi_opts = tiny_anima_server.http.get("/sdapi/v1/options").json()
    v2_opts = tiny_anima_server.http.get("/api/v2/options").json()["values"]
    assert sdapi_opts["sd_model_checkpoint"] == v2_opts["sd_model_checkpoint"]
    assert sdapi_opts["forge_preset"] == v2_opts["forge_preset"]
