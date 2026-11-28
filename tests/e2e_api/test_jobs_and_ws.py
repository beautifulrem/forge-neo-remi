"""Opaque-box E2E tests for F3 (Generation, Jobs, Queue & Idempotency) and F4 (WebSocket /api/v2/ws Events & Binary Previews)."""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from typing import Any

import pytest
import websockets.exceptions
from PIL import Image

# =============================================================================
# Generation, Jobs, Queue & Idempotency
# =============================================================================


def test_submit_txt2img_img2img_extras_complete_and_files_retrievable(
    live_server: Any,
    helpers: Any,
) -> None:
    """Submitting txt2img, img2img (with uploaded init_img), and extras completes and produces retrievable files."""

    txt_job = helpers.submit_txt2img(live_server, prompt="1girl, mountain sunrise", width=64, height=64)
    assert txt_job["status"] == "completed"
    assert txt_job["tab"] == "txt2img"
    assert len(txt_job["outputs"]) >= 1
    txt_out_id = txt_job["outputs"][0]["id"]
    assert txt_job["outputs"][0]["width"] > 0
    assert txt_job["outputs"][0]["height"] > 0

    txt_file = live_server.http.get(f"/api/v2/files/{txt_out_id}")
    assert txt_file.status_code == 200
    assert txt_file.content.startswith(b"\x89PNG\r\n\x1a\n")

    uploaded = helpers.upload_png(live_server, size=(48, 48), filename="init_src.png")
    img_job = helpers.submit_img2img(live_server, uploaded["id"], prompt="1girl, watercolor", width=64, height=64)
    assert img_job["status"] == "completed"
    assert img_job["tab"] == "img2img"
    assert len(img_job["outputs"]) >= 1
    img_file = live_server.http.get(f"/api/v2/files/{img_job['outputs'][0]['id']}")
    assert img_file.status_code == 200
    assert img_file.content.startswith(b"\x89PNG\r\n\x1a\n")

    extras_created = helpers.create_job(
        live_server,
        tab="extras",
        core={"mode": "Single Image", "image": uploaded["id"]},
    )
    extras_job = live_server.wait_job(extras_created["job_id"])
    assert extras_job["status"] == "completed"
    assert extras_job["tab"] == "extras"
    assert len(extras_job["outputs"]) >= 1
    extras_file = live_server.http.get(f"/api/v2/files/{extras_job['outputs'][0]['id']}")
    assert extras_file.status_code == 200
    assert extras_file.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_repeat_and_jobs_pagination_and_filters(
    live_server: Any,
    helpers: Any,
) -> None:
    """repeat=3 queues 3 sequential jobs; GET /api/v2/jobs paginates via next_cursor and filters by status and q."""

    created = helpers.create_job(
        live_server,
        tab="txt2img",
        core={"prompt": "cat portrait alpine", "width": 64, "height": 64},
        repeat=3,
    )
    assert len(created["job_ids"]) == 3
    assert len(set(created["job_ids"])) == 3

    other = helpers.create_job(
        live_server,
        tab="txt2img",
        core={"prompt": "dog landscape desert", "width": 64, "height": 64},
    )

    for jid in [*created["job_ids"], other["job_id"]]:
        finished = live_server.wait_job(jid)
        assert finished["status"] == "completed"

    page1_resp = live_server.http.get("/api/v2/jobs", params={"limit": 2, "status": "completed"})
    assert page1_resp.status_code == 200
    page1 = page1_resp.json()
    assert len(page1["items"]) == 2
    assert page1["total"] == 4
    assert page1["next_cursor"] is not None

    page2_resp = live_server.http.get(
        "/api/v2/jobs",
        params={"limit": 2, "status": "completed", "cursor": page1["next_cursor"]},
    )
    assert page2_resp.status_code == 200
    page2 = page2_resp.json()
    assert len(page2["items"]) == 2
    assert page2["next_cursor"] is None

    seen_ids = [item["id"] for item in page1["items"]] + [item["id"] for item in page2["items"]]
    assert set(seen_ids) == set(created["job_ids"]) | {other["job_id"]}

    filtered_alpine = live_server.http.get("/api/v2/jobs", params={"q": "alpine", "status": "completed"}).json()
    assert filtered_alpine["total"] == 3
    assert {item["id"] for item in filtered_alpine["items"]} == set(created["job_ids"])

    filtered_desert = live_server.http.get("/api/v2/jobs", params={"q": "desert", "tab": "txt2img"}).json()
    assert filtered_desert["total"] == 1
    assert filtered_desert["items"][0]["id"] == other["job_id"]


def test_queue_priority_ordering_and_pause_resume(
    live_server: Any,
    helpers: Any,
) -> None:
    """Paused queue orders items by priority DESC (10, 0, -10) and executes highest priority first upon resume."""

    pause_resp = live_server.http.post("/api/v2/queue/pause")
    assert pause_resp.status_code == 200
    assert pause_resp.json()["paused"] is True

    j_zero = helpers.submit_txt2img(live_server, prompt="prio zero", priority=0, wait=False)["job_id"]
    j_high = helpers.submit_txt2img(live_server, prompt="prio high", priority=10, wait=False)["job_id"]
    j_low = helpers.submit_txt2img(live_server, prompt="prio low", priority=-10, wait=False)["job_id"]

    items_resp = live_server.http.get("/api/v2/queue/items")
    assert items_resp.status_code == 200
    items = items_resp.json()
    assert [item["id"] for item in items] == [j_high, j_zero, j_low]
    assert [item["priority"] for item in items] == [10, 0, -10]

    ws = live_server.ws_connect()
    helpers.recv_ws_json(ws)  # hello
    helpers.recv_ws_json(ws)  # queue.changed snapshot

    resume_resp = live_server.http.post("/api/v2/queue/resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json()["paused"] is False

    started_order: list[str] = []
    completed_ids: set[str] = set()
    deadline = time.monotonic() + 10.0
    while len(completed_ids) < 3:
        remaining = max(0.05, deadline - time.monotonic())
        msg = helpers.recv_ws_json(ws, timeout=remaining)
        if msg["type"] == "job.started":
            started_order.append(msg["job_id"])
        elif msg["type"] == "job.completed":
            completed_ids.add(msg["job_id"])

    assert started_order == [j_high, j_zero, j_low]
    assert completed_ids == {j_high, j_zero, j_low}


def test_idempotency_key_replay_returns_same_job(
    live_server: Any,
    helpers: Any,
) -> None:
    """Retrying POST /api/v2/jobs with identical Idempotency-Key and payload returns same job_id/seq and queues once."""

    assert live_server.http.post("/api/v2/queue/pause").status_code == 200

    payload = {
        "tab": "txt2img",
        "core": {"prompt": "idempotent galaxy", "width": 64, "height": 64},
        "scripts": {},
        "priority": 0,
        "repeat": 1,
        "forever": False,
    }
    headers = {"Idempotency-Key": "idem-same-key-001"}

    first = live_server.http.post("/api/v2/jobs", json=payload, headers=headers)
    assert first.status_code == 201
    first_data = first.json()

    status_after_first = live_server.http.get("/api/v2/queue").json()
    assert status_after_first["size"] == 1

    second = live_server.http.post("/api/v2/jobs", json=payload, headers=headers)
    assert second.status_code == 201
    second_data = second.json()

    assert second_data["job_id"] == first_data["job_id"]
    assert second_data["job_ids"] == first_data["job_ids"]
    assert second_data["seq"] == first_data["seq"]

    status_after_second = live_server.http.get("/api/v2/queue").json()
    assert status_after_second["size"] == 1

    assert live_server.http.post("/api/v2/queue/resume").status_code == 200
    assert live_server.wait_job(first_data["job_id"])["status"] == "completed"


def test_job_duplicate_params_move_reorder_and_retry(
    live_server: Any,
    helpers: Any,
) -> None:
    """duplicate, PUT /params, queue/move, queue/reorder, and retry on completed job all behave as specified."""

    assert live_server.http.post("/api/v2/queue/pause").status_code == 200

    job_a = helpers.submit_txt2img(live_server, prompt="alpha", wait=False)["job_id"]

    dup_resp = live_server.http.post(f"/api/v2/jobs/{job_a}/duplicate")
    assert dup_resp.status_code == 201
    job_b = dup_resp.json()["job_id"]
    assert job_b != job_a

    params_resp = live_server.http.put(
        f"/api/v2/jobs/{job_b}/params",
        json={"core": {"prompt": "beta edited", "width": 64, "height": 64}, "scripts": {}},
    )
    assert params_resp.status_code == 200
    assert params_resp.json()["core"]["prompt"] == "beta edited"
    assert params_resp.json()["summary"] == "beta edited"

    job_c = helpers.submit_txt2img(live_server, prompt="gamma", wait=False)["job_id"]

    # Initially [job_a, job_b, job_c] -> move job_c before job_a -> [job_c, job_a, job_b]
    move_resp = live_server.http.post("/api/v2/queue/move", json={"id": job_c, "before": job_a})
    assert move_resp.status_code == 200
    queued_ids = [item["id"] for item in live_server.http.get("/api/v2/queue/items").json()]
    assert queued_ids == [job_c, job_a, job_b]

    # Reorder explicitly to [job_b, job_a, job_c]
    reorder_resp = live_server.http.post("/api/v2/queue/reorder", json={"order": [job_b, job_a, job_c]})
    assert reorder_resp.status_code == 200
    reordered_ids = [item["id"] for item in live_server.http.get("/api/v2/queue/items").json()]
    assert reordered_ids == [job_b, job_a, job_c]

    assert live_server.http.post("/api/v2/queue/resume").status_code == 200
    for jid in (job_b, job_a, job_c):
        assert live_server.wait_job(jid)["status"] == "completed"

    retry_resp = live_server.http.post(f"/api/v2/jobs/{job_b}/retry")
    assert retry_resp.status_code == 201
    retried_id = retry_resp.json()["job_id"]
    assert retried_id != job_b
    retried_done = live_server.wait_job(retried_id)
    assert retried_done["status"] == "completed"
    assert retried_done["retry_of"] == job_b
    assert retried_done["core"]["prompt"] == "beta edited"


def test_interrupt_running_gated_job(
    gated_server: Any,
    helpers: Any,
) -> None:
    """POST /api/v2/jobs/{id}/interrupt stops an actively running job and transitions it to interrupted."""

    job_id = helpers.submit_txt2img(gated_server, prompt="interruptible run", wait=False)["job_id"]
    assert gated_server.executor.started.wait(timeout=5.0)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        detail = gated_server.http.get(f"/api/v2/jobs/{job_id}").json()
        if detail["status"] == "running":
            break
        time.sleep(0.01)
    assert gated_server.http.get(f"/api/v2/jobs/{job_id}").json()["status"] == "running"

    intr_resp = gated_server.http.post(f"/api/v2/jobs/{job_id}/interrupt")
    assert intr_resp.status_code == 200

    final_job = gated_server.wait_job(job_id)
    assert final_job["status"] == "interrupted"


# =============================================================================
# Generation, Jobs, Queue & Idempotency
# =============================================================================


def test_invalid_job_request_bounds_and_upscale_from_rejected(
    live_server: Any,
    helpers: Any,
) -> None:
    """Invalid tab, out-of-range repeat/priority, and upscale_from on img2img return 422."""

    uploaded = helpers.upload_png(live_server, size=(32, 32))

    bad_payloads: list[dict[str, Any]] = [
        {"tab": "unknown", "core": {"prompt": "x"}, "scripts": {}},
        {"tab": "txt2img", "core": {"prompt": "x"}, "scripts": {}, "repeat": 0},
        {"tab": "txt2img", "core": {"prompt": "x"}, "scripts": {}, "repeat": 101},
        {"tab": "txt2img", "core": {"prompt": "x"}, "scripts": {}, "priority": 1001},
        {
            "tab": "img2img",
            "core": {"prompt": "x", "init_img": uploaded["id"]},
            "scripts": {},
            "upscale_from": uploaded["id"],
        },
    ]

    for payload in bad_payloads:
        resp = live_server.http.post("/api/v2/jobs", json=payload)
        assert resp.status_code == 422, f"expected 422 for {payload}, got {resp.status_code}: {resp.text}"


def test_idempotency_key_conflict_and_recovery_after_422(
    live_server: Any,
) -> None:
    """A 422 submission does not consume the Idempotency-Key; reusing a committed key with modified prompt returns 409."""

    key = "idem-recovery-key"
    bad_resp = live_server.http.post(
        "/api/v2/jobs",
        json={"tab": "unknown", "core": {"prompt": "bad"}, "scripts": {}},
        headers={"Idempotency-Key": key},
    )
    assert bad_resp.status_code == 422

    good_payload = {
        "tab": "txt2img",
        "core": {"prompt": "first prompt", "width": 64, "height": 64},
        "scripts": {},
    }
    ok_resp = live_server.http.post("/api/v2/jobs", json=good_payload, headers={"Idempotency-Key": key})
    assert ok_resp.status_code == 201
    job_id = ok_resp.json()["job_id"]

    modified_payload = {
        "tab": "txt2img",
        "core": {"prompt": "modified prompt", "width": 64, "height": 64},
        "scripts": {},
    }
    conflict_resp = live_server.http.post("/api/v2/jobs", json=modified_payload, headers={"Idempotency-Key": key})
    assert conflict_resp.status_code == 409

    assert live_server.wait_job(job_id)["status"] == "completed"


def test_invalid_state_transitions_return_409(
    gated_server: Any,
    helpers: Any,
) -> None:
    """retry while queued/running, skip while queued/completed, and PUT /params while running return 409."""

    running_id = helpers.submit_txt2img(gated_server, prompt="running job", wait=False)["job_id"]
    assert gated_server.executor.started.wait(timeout=5.0)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if gated_server.http.get(f"/api/v2/jobs/{running_id}").json()["status"] == "running":
            break
        time.sleep(0.01)

    queued_id = helpers.submit_txt2img(gated_server, prompt="queued behind", wait=False)["job_id"]
    assert gated_server.http.get(f"/api/v2/jobs/{queued_id}").json()["status"] == "queued"

    assert gated_server.http.post(f"/api/v2/jobs/{running_id}/retry").status_code == 409
    assert gated_server.http.post(f"/api/v2/jobs/{queued_id}/retry").status_code == 409
    assert gated_server.http.post(f"/api/v2/jobs/{queued_id}/skip").status_code == 409
    assert (
        gated_server.http.put(
            f"/api/v2/jobs/{running_id}/params",
            json={"core": {"prompt": "cannot edit running", "width": 64, "height": 64}, "scripts": {}},
        ).status_code
        == 409
    )

    assert gated_server.gate is not None
    gated_server.gate.set()
    assert gated_server.wait_job(running_id)["status"] == "completed"
    assert gated_server.wait_job(queued_id)["status"] == "completed"

    assert gated_server.http.post(f"/api/v2/jobs/{running_id}/skip").status_code == 409


def test_missing_required_inputs_and_bad_video_input_return_422(
    live_server: Any,
    helpers: Any,
) -> None:
    """img2img without init_img, Inpaint without mask, and extras Single Video with URL or missing path return 422."""

    uploaded = helpers.upload_png(live_server, size=(32, 32))

    no_init = live_server.http.post(
        "/api/v2/jobs",
        json={"tab": "img2img", "core": {"prompt": "missing init"}, "scripts": {}},
    )
    assert no_init.status_code == 422
    assert any(err.get("param") == "init_img" for err in no_init.json().get("errors", []))

    no_mask = live_server.http.post(
        "/api/v2/jobs",
        json={
            "tab": "img2img",
            "core": {"mode": "Inpaint", "init_img": uploaded["id"], "prompt": "missing mask"},
            "scripts": {},
        },
    )
    assert no_mask.status_code == 422
    assert any(err.get("param") == "mask" for err in no_mask.json().get("errors", []))

    evil_video = live_server.http.post(
        "/api/v2/jobs",
        json={
            "tab": "extras",
            "core": {"mode": "Single Video", "video_input": "http://evil/video.mp4"},
            "scripts": {},
        },
    )
    assert evil_video.status_code == 422
    assert any(err.get("param") == "video_input" for err in evil_video.json().get("errors", []))

    missing_video = live_server.http.post(
        "/api/v2/jobs",
        json={
            "tab": "extras",
            "core": {"mode": "Single Video", "video_input": str(live_server.root / "does-not-exist.mp4")},
            "scripts": {},
        },
    )
    assert missing_video.status_code == 422
    assert any(err.get("param") == "video_input" for err in missing_video.json().get("errors", []))


def test_nonexistent_job_404_and_failing_server_marks_failed(
    live_server: Any,
    failing_server: Any,
    helpers: Any,
) -> None:
    """GET/DELETE on unknown job returns 404; executor failure marks job failed with error message."""

    assert live_server.http.get("/api/v2/jobs/nonexistent-job-id").status_code == 404
    assert live_server.http.delete("/api/v2/jobs/nonexistent-job-id").status_code == 404

    failed = helpers.submit_txt2img(failing_server, prompt="boom test")
    assert failed["status"] == "failed"
    assert "synthetic boom" in (failed["error"] or "")


# =============================================================================
# Real-Time WebSocket (/api/v2/ws) Events & Binary Previews
# =============================================================================


def test_ws_connect_receives_hello_and_queue_snapshot(
    live_server: Any,
    helpers: Any,
) -> None:
    """Connecting to /api/v2/ws yields hello with advertised features followed by queue.changed snapshot."""

    ws = live_server.ws_connect()
    hello = helpers.recv_ws_json(ws)
    assert hello["type"] == "hello"
    for feature in ("events.v1", "preview.binary", "preview.jpeg", "preview.webp", "jobs.v1"):
        assert feature in hello["features"]

    snapshot = helpers.recv_ws_json(ws)
    assert snapshot["type"] == "queue.changed"
    assert snapshot["size"] == 0
    assert snapshot["paused"] is False
    assert snapshot["running"] is None


def test_ws_ping_returns_pong(
    live_server: Any,
    helpers: Any,
) -> None:
    """Sending {"type": "ping"} over /api/v2/ws receives {"type": "pong"}."""

    ws = live_server.ws_connect()
    helpers.recv_ws_json(ws)  # hello
    helpers.recv_ws_json(ws)  # queue.changed

    ws.send(json.dumps({"type": "ping"}))
    pong = helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "pong")
    assert pong == {"type": "pong"}


def test_ws_job_lifecycle_events_and_progress(
    make_live_server: Callable[..., Any],
    helpers: Any,
) -> None:
    """Submitting a job while connected emits job.queued, job.started, job.progress, job.completed, and queue.changed."""

    server = make_live_server(steps=4, step_time=0.05, realtime=True)
    ws = server.ws_connect()
    helpers.recv_ws_json(ws)  # hello
    helpers.recv_ws_json(ws)  # initial queue.changed

    created = helpers.submit_txt2img(server, prompt="ws lifecycle", wait=False)
    job_id = created["job_id"]

    _completed, seen = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == job_id,
        timeout=10.0,
        collect_all=True,
    )
    seen.extend(helpers.drain_ws(ws, idle_timeout=0.15))

    json_events = [m for m in seen if isinstance(m, dict)]
    event_types = [m["type"] for m in json_events]
    assert "job.queued" in event_types
    assert "job.started" in event_types
    assert "job.progress" in event_types
    assert "job.completed" in event_types
    assert "queue.changed" in event_types

    progress_events = [m for m in json_events if m["type"] == "job.progress" and m.get("job_id") == job_id]
    assert len(progress_events) >= 1
    for evt in progress_events:
        assert 0.0 <= evt["progress"] <= 1.0
        assert evt["seq"] == created["seq"]

    completed_evt = next(m for m in json_events if m["type"] == "job.completed" and m.get("job_id") == job_id)
    assert completed_evt["status"] == "completed"
    assert len(completed_evt["outputs"]) >= 1


def test_ws_binary_preview_frames_valid_header_and_image(
    make_live_server: Callable[..., Any],
    helpers: Any,
) -> None:
    """Multi-step generation emits <BBII binary preview frames with valid type, fmt, job_seq, and JPEG/WebP payload."""

    server = make_live_server(steps=4, step_time=0.05, realtime=True)
    ws = server.ws_connect()
    helpers.recv_ws_json(ws)  # hello
    helpers.recv_ws_json(ws)  # queue.changed

    created = helpers.submit_txt2img(server, prompt="binary preview test", wait=False)
    job_id = created["job_id"]

    _completed, seen = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == job_id,
        timeout=10.0,
        collect_all=True,
    )

    binary_frames = [m for m in seen if isinstance(m, bytes)]
    assert len(binary_frames) >= 1

    for raw_frame in binary_frames:
        parsed = helpers.parse_preview_frame(raw_frame)
        assert parsed["type"] == 1
        assert parsed["fmt"] in (1, 2)
        assert parsed["job_seq"] == created["seq"]
        assert parsed["step"] >= 1
        payload = parsed["payload"]
        if parsed["fmt"] == 1:
            assert payload.startswith(b"\xff\xd8\xff")
        else:
            assert payload[:4] == b"RIFF" and payload[8:12] == b"WEBP"
        with Image.open(io.BytesIO(payload)) as img:
            assert img.size == (32, 32)


def test_ws_cross_feature_options_outputs_and_chain_events(
    live_server: Any,
    helpers: Any,
) -> None:
    """PATCH /options, POST /gallery/star, and POST /chains emit options.changed, outputs.changed, chain.updated, and chain.stage_completed."""

    done_job = helpers.submit_txt2img(live_server, prompt="star target")
    output_id = done_job["outputs"][0]["id"]

    ws = live_server.ws_connect()
    helpers.recv_ws_json(ws)  # hello
    helpers.recv_ws_json(ws)  # queue.changed

    current_fmt = live_server.http.get("/api/v2/options").json()["values"].get("samples_format")
    new_fmt = "jpg" if current_fmt != "jpg" else "png"
    opt_resp = live_server.http.patch("/api/v2/options", json={"samples_format": new_fmt})
    assert opt_resp.status_code == 200
    opt_evt = helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "options.changed")
    assert "samples_format" in opt_evt["keys"]

    star_resp = live_server.http.post("/api/v2/gallery/star", json={"ids": [output_id], "starred": True})
    assert star_resp.status_code == 200
    out_evt = helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "outputs.changed")
    assert output_id in out_evt["ids"]

    chain_resp = live_server.http.post(
        "/api/v2/chains",
        json={
            "name": "ws-cross-chain",
            "stages": [{"kind": "generate", "core": {"prompt": "chain ws", "width": 64, "height": 64}}],
            "run_now": True,
        },
    )
    assert chain_resp.status_code == 201
    chain_id = chain_resp.json()["id"]

    finished_chain = live_server.wait_chain(chain_id)
    assert finished_chain["status"] == "completed"

    collected = helpers.drain_ws(ws, idle_timeout=0.2)
    json_events = [m for m in collected if isinstance(m, dict)]
    assert any(m.get("type") == "chain.updated" and m.get("chain_id") == chain_id for m in json_events)
    assert any(m.get("type") == "chain.stage_completed" and m.get("chain_id") == chain_id and m.get("stage_index") == 0 for m in json_events)


# =============================================================================
# Real-Time WebSocket (/api/v2/ws) Events & Binary Previews
# =============================================================================


def test_ws_foreign_origin_rejected_with_403(
    live_server: Any,
) -> None:
    """Connecting to /api/v2/ws with a foreign Origin header is rejected at handshake with 403."""

    with pytest.raises(websockets.exceptions.InvalidStatus) as exc_info:
        live_server.ws_connect(origin="https://evil.example")
    assert exc_info.value.response.status_code == 403


def test_ws_auth_required_rejects_unauthenticated_with_403(
    auth_server: Any,
    helpers: Any,
) -> None:
    """Unauthenticated WS handshake on auth_server is rejected (403); logged-in session connects and closes 1008 on logout."""

    with pytest.raises(websockets.exceptions.InvalidStatus) as exc_info:
        auth_server.ws_connect()
    assert exc_info.value.response.status_code == 403

    client = auth_server.new_client()
    login_resp = helpers.login(auth_server, "alice", "s3cret", client=client)
    assert login_resp.status_code == 204

    ws = auth_server.ws_connect(client=client)
    hello = helpers.recv_ws_json(ws)
    assert hello["type"] == "hello"

    logout_resp = client.post("/api/v2/auth/logout")
    assert logout_resp.status_code == 204

    with pytest.raises(websockets.exceptions.ConnectionClosedError) as close_exc:
        helpers.recv_ws_until(ws, lambda _m: False, timeout=3.0)
    assert close_exc.value.rcvd is not None
    assert close_exc.value.rcvd.code == 1008


def test_ws_malformed_json_and_unknown_type_keep_connection_alive(
    live_server: Any,
    helpers: Any,
) -> None:
    """Malformed JSON, non-dict JSON, and unknown message types are ignored without dropping the WS connection."""

    ws = live_server.ws_connect()
    helpers.recv_ws_json(ws)  # hello
    helpers.recv_ws_json(ws)  # queue.changed

    ws.send("{not valid json")
    ws.send("[1, 2, 3]")
    ws.send(json.dumps({"type": "bogus", "extra": 123}))
    ws.send(json.dumps({"type": "ping"}))

    pong = helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "pong")
    assert pong == {"type": "pong"}


def test_ws_visibility_toggle_suppresses_and_resumes_binary_previews(
    make_live_server: Callable[..., Any],
    helpers: Any,
) -> None:
    """visibility=false suppresses binary frames while keeping JSON events; visibility=true resumes binary frames."""

    server = make_live_server(steps=4, step_time=0.05, realtime=True)
    ws = server.ws_connect()
    helpers.recv_ws_json(ws)  # hello
    helpers.recv_ws_json(ws)  # queue.changed

    # Synchronize via ping/pong so _receive() has processed visibility=false before job1 starts.
    ws.send(json.dumps({"type": "visibility", "visible": False}))
    ws.send(json.dumps({"type": "ping"}))
    helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "pong")

    job1_id = helpers.submit_txt2img(server, prompt="hidden preview job", wait=False)["job_id"]
    _done1, seen1 = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == job1_id,
        timeout=10.0,
        collect_all=True,
    )
    seen1.extend(helpers.drain_ws(ws, idle_timeout=0.1))
    assert not any(isinstance(m, bytes) for m in seen1)

    ws.send(json.dumps({"type": "visibility", "visible": True}))
    ws.send(json.dumps({"type": "ping"}))
    helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "pong")

    job2_created = helpers.submit_txt2img(server, prompt="visible preview job", wait=False)
    job2_id = job2_created["job_id"]
    _done2, seen2 = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == job2_id,
        timeout=10.0,
        collect_all=True,
    )
    binary2 = [m for m in seen2 if isinstance(m, bytes)]
    assert len(binary2) >= 1
    parsed = helpers.parse_preview_frame(binary2[0])
    assert parsed["job_seq"] == job2_created["seq"]


def test_ws_hello_without_preview_binary_suppresses_binary_frames(
    make_live_server: Callable[..., Any],
    helpers: Any,
) -> None:
    """Client hello declaring only features=["events.v1"] receives JSON events and zero binary preview frames."""

    server = make_live_server(steps=4, step_time=0.05, realtime=True)
    ws = server.ws_connect()
    helpers.recv_ws_json(ws)  # server hello
    helpers.recv_ws_json(ws)  # queue.changed

    ws.send(json.dumps({"type": "hello", "features": ["events.v1"]}))
    ws.send(json.dumps({"type": "ping"}))
    helpers.recv_ws_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "pong")

    job_id = helpers.submit_txt2img(server, prompt="events only job", wait=False)["job_id"]
    _done, seen = helpers.recv_ws_until(
        ws,
        lambda m: isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == job_id,
        timeout=10.0,
        collect_all=True,
    )
    seen.extend(helpers.drain_ws(ws, idle_timeout=0.1))

    assert any(isinstance(m, dict) and m.get("type") == "job.started" and m.get("job_id") == job_id for m in seen)
    assert any(isinstance(m, dict) and m.get("type") == "job.completed" and m.get("job_id") == job_id for m in seen)
    assert not any(isinstance(m, bytes) for m in seen)
