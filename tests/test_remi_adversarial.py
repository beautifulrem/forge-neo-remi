"""Adversarial regression, concurrency, and input-hardening tests."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from remi import extensions_admin as admin
from remi import networks as networks_mod
from remi.files import FileError, FileStore
from remi.jobs import JobQueue, JobStore
from remi.jobs.chains import ChainRunner, ChainStore, StageSpec
from remi.realtime import FORMAT_JPEG, EventBus, PreviewFrame, unpack_frame
from remi.server.limits import BodyLimitMiddleware, size
from remi.server.testing import WS_URL, StubExecutor, server_app
from remi.testing import webui_state
from remi.ui import Context, ParamError, nodes
from remi.worker import JobEnvelope, MlxEnvironment, MlxSidecarWorker, SubprocessWorker, WorkerMessage, WorkerProgress, encode_message


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


def _png_bytes(color: tuple[int, int, int] = (10, 20, 30), size_px: tuple[int, int] = (32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size_px, color).save(buf, format="PNG")
    return buf.getvalue()


def _nested(depth: int) -> dict:
    node: dict = {"leaf": 1}
    for _ in range(depth - 1):
        node = {"child": node}
    return node


def _safetensors(path: Path, metadata: dict[str, str]) -> Path:
    header = json.dumps({"__metadata__": metadata}).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(header).to_bytes(8, "little") + header)
    return path


# ---------------------------------------------------------------------------
# BUG-01: Network sidecar depth persistence mismatch (MAX_NESTING = 64)
# ---------------------------------------------------------------------------


def test_bug01_network_metadata_rejects_excessive_nesting_before_saving(webui, tmp_path, monkeypatch):
    lora_dir = tmp_path / "Lora"
    model_path = _safetensors(lora_dir / "m64.safetensors", {"ss_output_name": "m64"})

    class _FakeNetworks:
        def __init__(self):
            self.available_networks = {
                "m64": SimpleNamespace(
                    name="m64",
                    filename=str(model_path),
                    metadata={"ss_output_name": "m64"},
                    alias="m64",
                    shorthash="m64",
                    get_alias=lambda: "m64",
                )
            }

        def list_available_networks(self):
            return None

    monkeypatch.setitem(sys.modules, "networks", _FakeNetworks())
    with server_app(tmp_path / "data") as server:
        monkeypatch.setattr(server.services.networks, "_providers", lambda: {"lora": networks_mod.LoraProvider()})

        # Depth 64 inside {"custom": ...} means outer data depth is 65 > MAX_NESTING (64)
        bad = server.client.patch("/api/v2/networks/lora/m64/meta", json={"user_metadata": {"custom": _nested(64)}})
        assert bad.status_code == 422
        assert bad.json()["errors"][0]["param"] == "user_metadata"

        # Subsequent legitimate edit must still succeed (sidecar was not bricked)
        ok = server.client.patch("/api/v2/networks/lora/m64/meta", json={"user_metadata": {"notes": "ok"}})
        assert ok.status_code == 200
        assert ok.json()["user_metadata"]["notes"] == "ok"

        # Depth 62 inside {"custom": ...} (outer depth 63 <= 64) succeeds and remains readable
        deep_ok = server.client.patch("/api/v2/networks/lora/m64/meta", json={"user_metadata": {"custom": _nested(62)}})
        assert deep_ok.status_code == 200
        read_back = server.client.get("/api/v2/networks/lora/m64/meta")
        assert read_back.status_code == 200


# ---------------------------------------------------------------------------
# BUG-02: Options non-finite float rejection & Text.coerce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw_body", ['{"outdir_samples": NaN}', '{"outdir_samples": 1e999}', '{"sd_checkpoint_hash": -Infinity}', '{"outdir_samples": [Infinity]}'])
def test_bug02_options_reject_non_finite_floats_with_422(webui, tmp_path, raw_body):
    from modules import shared

    with server_app(tmp_path / "data") as server:
        before_outdir = shared.opts.outdir_samples
        patch_resp = server.client.patch("/api/v2/options", content=raw_body, headers={"Content-Type": "application/json"})
        assert patch_resp.status_code == 422
        assert shared.opts.outdir_samples == before_outdir

        val_resp = server.client.post("/api/v2/options/validate", content=raw_body, headers={"Content-Type": "application/json"})
        assert val_resp.status_code == 422


@pytest.mark.parametrize("bad_num", [float("nan"), float("inf"), float("-inf")])
def test_bug02_text_node_coerce_rejects_non_finite_floats(bad_num):
    node = nodes.Text("outdir_samples", "")
    with pytest.raises(ParamError):
        node.coerce(bad_num, Context())
    assert node.coerce(42, Context()) == "42"


# ---------------------------------------------------------------------------
# BUG-03: GET /api/v2/extensions/config-states skips bad created_at safely
# ---------------------------------------------------------------------------


def test_bug03_extensions_config_states_skips_nan_and_overflow_timestamps(webui, tmp_path, monkeypatch):
    from modules import config_states

    states_dir = tmp_path / "config_states"
    states_dir.mkdir()
    (states_dir / "bad_nan.json").write_text('{"created_at": NaN, "name": "BadNaN"}', encoding="utf-8")
    (states_dir / "bad_huge.json").write_text('{"created_at": 1e30, "name": "BadHuge"}', encoding="utf-8")
    (states_dir / "good.json").write_text(json.dumps({"created_at": 1_700_000_000.0, "name": "Good", "webui": {}, "extensions": {}}), encoding="utf-8")

    monkeypatch.setattr(config_states, "config_states_dir", str(states_dir))
    with server_app(tmp_path / "data") as server:
        resp = server.client.get("/api/v2/extensions/config-states")
        assert resp.status_code == 200
        items = resp.json()
        assert [item["name"] for item in items] == ["Good"]

    # save_config_state rejects non-finite created_at from get_config
    monkeypatch.setattr(config_states, "get_config", lambda: {"created_at": float("nan"), "webui": {}, "extensions": {}})
    with pytest.raises(admin.AdminError):
        admin.save_config_state("bad")


# ---------------------------------------------------------------------------
# BUG-10: SubprocessWorker.execute tolerates malformed telemetry payloads
# ---------------------------------------------------------------------------


def test_bug10_subprocess_worker_handles_malformed_telemetry_without_crashing():
    class _FakeStdin:
        def write(self, _data: bytes) -> int:
            return len(_data)

        def flush(self) -> None:
            return None

    class _FakeProc:
        def __init__(self, stdout_bytes: bytes):
            self.stdin = _FakeStdin()
            self.stdout = io.BytesIO(stdout_bytes)
            self.stderr = io.BytesIO(b"")
            self.pid = 99999

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    stdout_1 = b"".join(
        [
            encode_message(WorkerMessage("progress", {"job_id": "j1", "progress": "not-a-float"})),
            encode_message(WorkerMessage("preview", {"job_id": "j1", "step": "not-an-int"})),
            encode_message(WorkerMessage("capabilities", {"supported_families": 12345})),
            encode_message(WorkerMessage("result", {"job_id": "j1", "status": "completed", "output_file_ids": ["f1"]})),
        ]
    )
    worker = SubprocessWorker(worker_id="sub-bad-telemetry")
    worker._proc = _FakeProc(stdout_1)  # type: ignore[assignment]
    worker._state = "idle"

    progress_seen: list[WorkerProgress] = []
    previews_seen: list[object] = []
    res = worker.execute(
        JobEnvelope(job_id="j1", seq=1, tab="txt2img"),
        on_progress=progress_seen.append,
        on_preview=previews_seen.append,
        timeout=5.0,
    )
    assert res.status == "completed"
    assert res.output_file_ids == ["f1"]
    assert progress_seen == []
    assert previews_seen == []

    # Malformed result payload (missing required job_id -> KeyError) yields failed WorkerResult
    stdout_2 = encode_message(WorkerMessage("result", {"status": "completed"}))
    worker._proc = _FakeProc(stdout_2)  # type: ignore[assignment]
    res2 = worker.execute(JobEnvelope(job_id="j2", seq=2, tab="txt2img"), timeout=5.0)
    assert res2.status == "failed" and "malformed" in (res2.error or "")


# ---------------------------------------------------------------------------
# BUG-11: FileStore symlink escape prevention & /api/v2/files containment
# ---------------------------------------------------------------------------


def test_bug11_filestore_and_files_route_block_symlink_escapes(webui, tmp_path):
    external = tmp_path / "outside"
    external.mkdir()
    secret_png = external / "secret.png"
    secret_png.write_bytes(_png_bytes((200, 10, 10)))

    with server_app(tmp_path / "data") as server:
        files = server.services.files
        out_dir = tmp_path / "outputs" / "txt2img-images"
        out_dir.mkdir(parents=True)

        legit = out_dir / "legit.png"
        legit.write_bytes(_png_bytes((10, 200, 10)))

        file_link = out_dir / "escape.png"
        file_link.symlink_to(secret_png)

        dir_link = out_dir / "escape_dir"
        dir_link.symlink_to(external, target_is_directory=True)

        # scan_outputs skips both file symlinks and symlinked directories, indexing only legit.png
        added = files.scan_outputs([out_dir])
        assert added == 1

        # register_output rejects direct symlink and symlink-directory escape under outputs_dir/root
        with pytest.raises(FileError):
            files.register_output(file_link)
        inside_link = files.outputs_dir / "sub_escape"
        files.outputs_dir.mkdir(parents=True, exist_ok=True)
        inside_link.symlink_to(external, target_is_directory=True)
        with pytest.raises(ValueError):
            files.register_output(inside_link / "secret.png")

        # Post-registration symlink swap on disk is blocked by /api/v2/files/{id} and /thumbs/{id}
        rec = files.save_output_image(Image.new("RGB", (16, 16), (1, 2, 3)), None)
        assert server.client.get(f"/api/v2/files/{rec.id}").status_code == 200
        os.remove(rec.path)
        os.symlink(secret_png, rec.path)
        assert server.client.get(f"/api/v2/files/{rec.id}").status_code == 404
        assert server.client.get(f"/api/v2/thumbs/{rec.id}").status_code == 404


# ---------------------------------------------------------------------------
# Concurrent Idempotency-Key race on /api/v2/chains & /api/v2/jobs
# ---------------------------------------------------------------------------


def test_concurrent_idempotency_key_race_chains_and_jobs(webui, tmp_path):
    with server_app(tmp_path / "data") as server:
        assert server.client.post("/api/v2/queue/pause").status_code == 200
        chain_payload = {
            "name": "race-chain",
            "run_now": False,
            "stages": [{"kind": "generate", "tab": "txt2img", "core": {"prompt": "same"}}],
        }
        headers = {"Idempotency-Key": "race-chain-key-1"}

        def _post_chain(_idx: int):
            return server.client.post("/api/v2/chains", json=chain_payload, headers=headers)

        with ThreadPoolExecutor(max_workers=8) as pool:
            chain_resps = list(pool.map(_post_chain, range(8)))
        assert all(r.status_code == 201 for r in chain_resps)
        chain_ids = {r.json()["id"] for r in chain_resps}
        assert len(chain_ids) == 1

        # Conflicting params with same Idempotency-Key return 409
        conflict = server.client.post(
            "/api/v2/chains",
            json={"name": "race-chain", "run_now": False, "stages": [{"kind": "generate", "tab": "txt2img", "core": {"prompt": "different"}}]},
            headers=headers,
        )
        assert conflict.status_code == 409

        # Concurrent POST /api/v2/jobs with identical Idempotency-Key
        job_payload = {"tab": "txt2img", "core": {"prompt": "concurrent job"}}
        job_headers = {"Idempotency-Key": "race-job-key-1"}

        def _post_job(_idx: int):
            return server.client.post("/api/v2/jobs", json=job_payload, headers=job_headers)

        with ThreadPoolExecutor(max_workers=8) as pool:
            job_resps = list(pool.map(_post_job, range(8)))
        assert all(r.status_code == 201 for r in job_resps)
        job_ids = {r.json()["job_id"] for r in job_resps}
        assert len(job_ids) == 1
        assert server.services.store.count("queued") == 1


# ---------------------------------------------------------------------------
# ChainRunner cancel_chain mid-upsample & stage_cache invalidation/prune
# ---------------------------------------------------------------------------


def test_chain_runner_cancel_mid_upsample_and_stage_cache_retention(tmp_path):
    files = FileStore(tmp_path)
    jobs = JobStore(tmp_path / "jobs.db")
    chains = ChainStore(jobs)
    bus = EventBus()
    entered = threading.Event()
    release = threading.Event()

    def slow_upsampler(_stage, _prompt, _outputs):
        entered.set()
        release.wait(5.0)
        return "upsampled prompt"

    queue_inst = JobQueue(jobs, files, bus, executor=StubExecutor(files))
    runner = ChainRunner(chains, queue_inst, files, bus, upsampler=slow_upsampler)
    queue_inst.add_completion_hook(runner.on_job_finished)
    queue_inst.start()
    try:
        chain = runner.create_chain(
            [
                StageSpec(kind="prompt_upsample", core={"prompt": "cat"}),
                StageSpec(kind="generate", tab="txt2img", core={}),
            ],
            run_now=True,
        )
        assert entered.wait(5.0)
        cancelled = runner.cancel_chain(chain.id)
        assert cancelled.status == "cancelled"
        assert cancelled.stages[0].status == "cancelled"
        assert cancelled.stages[1].status == "cancelled"
        release.set()

        # Wait for upsample background thread to exit cleanly
        for th in list(runner._threads):
            th.join(timeout=5.0)

        # Cancelled upsample must NOT populate stage_cache
        refetched = chains.get_chain(chain.id)
        assert refetched is not None and refetched.status == "cancelled"
        assert chains._db.execute("SELECT COUNT(*) FROM stage_cache").fetchone()[0] == 0

        # Error stage invalidation & prune_cache retention
        rec = files.save_output_image(Image.new("RGB", (8, 8), (1, 2, 3)), None)
        chains.put_cached("hash-valid", "generate", output_file_ids=[rec.id], job_id="j1")
        assert chains.get_cached("hash-valid") is not None
        os.remove(rec.path)
        removed = chains.prune_cache(older_than=0.0, files=files)
        assert removed == 1
        assert chains.get_cached("hash-valid") is None
    finally:
        release.set()
        queue_inst.stop(timeout=5.0, interrupt=True)
        chains.close()
        jobs.close()
        files.close()


# ---------------------------------------------------------------------------
# WebSocket /api/v2/ws binary PreviewFrame, wants_frames, resync, disconnect
# ---------------------------------------------------------------------------


def test_ws_binary_preview_wants_frames_and_disconnect(webui, tmp_path):
    with server_app(tmp_path / "data") as server:
        hub = server.app.state.hub
        assert hub.wants_frames() is False

        with server.client.websocket_connect(WS_URL) as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            queue_msg = ws.receive_json()
            assert queue_msg["type"] == "queue.changed"

            # Wait until hub registers the client
            deadline = time.monotonic() + 2.0
            while not hub.clients and time.monotonic() < deadline:
                time.sleep(0.01)
            assert hub.wants_frames() is True

            ws.send_json({"type": "hello", "features": ["events.v1", "preview.binary", "preview.jpeg"], "visible": True})
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

            frame = PreviewFrame(FORMAT_JPEG, job_seq=7, step=3, data=b"\xff\xd8\xff\xe0test")
            hub.send_frame(frame)
            raw_frame = ws.receive_bytes()
            assert unpack_frame(raw_frame) == frame

            # Visibility toggle disables wants_frames
            ws.send_json({"type": "visibility", "visible": False})
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
            assert hub.wants_frames() is False

            # Queue overflow triggers resync (scheduled atomically on hub.loop)
            [client] = list(hub.clients)
            total_events = (client.events.maxlen or 256) + 5

            def _overflow() -> None:
                for idx in range(total_events):
                    client.push_event({"type": "tick", "n": idx})

            hub._call(_overflow)
            resync = ws.receive_json()
            assert resync["type"] == "resync" and resync["dropped"] >= 5
            snap = ws.receive_json()
            assert snap["type"] == "queue.changed"

        deadline = time.monotonic() + 2.0
        while hub.clients and time.monotonic() < deadline:
            time.sleep(0.01)
        assert hub.wants_frames() is False


# ---------------------------------------------------------------------------
# MlxSidecarWorker concurrent execute / cancel / stop / restart
# ---------------------------------------------------------------------------


def test_mlx_sidecar_concurrent_execute_cancel_stop_restart(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def slow_runner(envelope: JobEnvelope, out_path: Path, _on_progress, cancel_ev: threading.Event):
        entered.set()
        while not cancel_ev.is_set() and not release.is_set():
            time.sleep(0.01)
        if cancel_ev.is_set():
            raise InterruptedError("cancelled")
        Image.new("RGB", (8, 8), (5, 6, 7)).save(out_path)
        return {"infotext": "ok"}

    env_ok = MlxEnvironment(available=True, platform_supported=True, mlx_version="0.1", mflux_version="0.1")
    worker = MlxSidecarWorker(env_probe=lambda: env_ok, runner=slow_runner)

    results: dict[str, object] = {}
    t1 = threading.Thread(target=lambda: results.setdefault("first", worker.execute(JobEnvelope(job_id="mlx-1", seq=1, tab="txt2img"))))
    t1.start()
    assert entered.wait(5.0)
    assert worker.cancel("mlx-1") is True
    t1.join(timeout=5.0)
    assert results["first"].status == "cancelled"  # type: ignore[attr-defined]
    assert worker.status().state == "idle"

    # Restart while an old runner is still winding down must not corrupt new job's busy state
    entered.clear()
    release.clear()
    slow_done = threading.Event()

    def overlapping_runner(envelope: JobEnvelope, out_path: Path, _on_progress, cancel_ev: threading.Event):
        if envelope.job_id == "mlx-old":
            entered.set()
            slow_done.wait(5.0)
            return {}
        Image.new("RGB", (8, 8), (9, 9, 9)).save(out_path)
        return {"infotext": "new"}

    worker2 = MlxSidecarWorker(env_probe=lambda: env_ok, runner=overlapping_runner)
    t_old = threading.Thread(target=lambda: results.setdefault("old", worker2.execute(JobEnvelope(job_id="mlx-old", seq=1, tab="txt2img"))))
    t_old.start()
    assert entered.wait(5.0)
    worker2.restart()
    assert worker2.status().state == "idle"
    res_new = worker2.execute(JobEnvelope(job_id="mlx-new", seq=2, tab="txt2img", file_store_root=str(tmp_path)))
    slow_done.set()
    t_old.join(timeout=5.0)
    assert res_new.status == "completed"
    assert worker2.status().state == "idle"


# ---------------------------------------------------------------------------
# FileStore SQLite WAL concurrent register_output, SVG/XSS uploads, BodyLimit < 1024
# ---------------------------------------------------------------------------


def test_filestore_wal_concurrency_svg_xss_and_small_body_limit(webui, tmp_path):
    store = FileStore(tmp_path / "wal")
    store2 = FileStore(tmp_path / "wal")
    try:
        out_dir = tmp_path / "imgs"
        out_dir.mkdir()
        paths = []
        for idx in range(12):
            p = out_dir / f"img_{idx}.png"
            p.write_bytes(_png_bytes((idx, 20, 30)))
            paths.append(p)

        def _reg(idx: int):
            target = store if idx % 2 == 0 else store2
            return target.register_output(paths[idx % len(paths)], job_id=f"job-{idx % len(paths)}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            recs = list(pool.map(_reg, range(24)))
        assert len({r.id for r in recs}) == 12
        assert store._db.execute("SELECT COUNT(*) FROM files WHERE kind = 'output'").fetchone()[0] == 12
    finally:
        store.close()
        store2.close()

    # Upload SVG/XSS/polyglot rejections
    with server_app(tmp_path / "data") as server:
        svg_payload = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        assert server.client.post("/api/v2/uploads", files={"file": ("evil.svg", svg_payload, "image/svg+xml")}).status_code == 415
        assert server.client.post("/api/v2/uploads", files={"file": ("evil.png", svg_payload, "image/png")}).status_code == 415
        assert server.client.post("/api/v2/uploads", files={"file": ("poly.png", b"\x89PNG\r\n\x1a\n<script>alert(1)</script>", "image/png")}).status_code == 415

    # BodyLimitMiddleware small limit formatting (< 1024 bytes)
    assert size(32) == "32 B"
    assert size(1024) == "1 KB"
    assert size(1024 * 1024) == "1 MB"

    app = FastAPI()

    @app.post("/echo")
    def _echo():
        return {"ok": True}

    app.add_middleware(BodyLimitMiddleware, limit=lambda _scope: 32)
    client = TestClient(app)
    resp = client.post("/echo", content=b"x" * 100)
    assert resp.status_code == 413
    assert resp.json()["detail"] == "the request body is larger than 32 B"


# ---------------------------------------------------------------------------
# FileStore.get() / get_many() reject post-registration symlinks & escaped uploads
# ---------------------------------------------------------------------------


def test_filestore_blocks_symlink_swap_across_all_consuming_routes(webui, tmp_path, monkeypatch):
    from PIL import PngImagePlugin

    external = tmp_path / "outside"
    external.mkdir()
    secret_png = external / "secret.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "SECRET_PROMPT\nSteps: 20, Seed: 7")
    Image.new("RGB", (24, 24), (210, 10, 10)).save(secret_png, format="PNG", pnginfo=info)

    outside_upload = external / "escaped_upload.png"
    outside_upload.write_bytes(_png_bytes((5, 5, 5)))

    lora_dir = tmp_path / "Lora"
    model_path = _safetensors(lora_dir / "m_prev.safetensors", {"ss_output_name": "m_prev"})

    class _FakeNetworks:
        def __init__(self):
            self.available_networks = {
                "m_prev": SimpleNamespace(
                    name="m_prev",
                    filename=str(model_path),
                    metadata={"ss_output_name": "m_prev"},
                    alias="m_prev",
                    shorthash="m_prev",
                    get_alias=lambda: "m_prev",
                )
            }

        def list_available_networks(self):
            return None

    monkeypatch.setitem(sys.modules, "networks", _FakeNetworks())
    with server_app(tmp_path / "data") as server:
        monkeypatch.setattr(server.services.networks, "_providers", lambda: {"lora": networks_mod.LoraProvider()})
        files = server.services.files
        rec = files.save_output_image(Image.new("RGB", (16, 16), (1, 2, 3)), "normal prompt\nSteps: 10")
        assert files.get(rec.id) is not None
        assert list(files.get_many([rec.id])) == [rec.id]

        # Swap the registered output on disk to a symlink targeting external secret.png
        os.remove(rec.path)
        os.symlink(secret_png, rec.path)

        assert files.get(rec.id) is None
        assert files.get_many([rec.id]) == {}

        for method, url, kwargs in (
            ("GET", f"/api/v2/files/{rec.id}", {}),
            ("GET", f"/api/v2/thumbs/{rec.id}", {}),
            ("POST", "/api/v2/gallery/zip", {"json": {"ids": [rec.id]}}),
            ("POST", "/api/v2/pnginfo", {"json": {"file_id": rec.id}}),
            ("POST", "/api/v2/infotext/image", {"json": {"file_id": rec.id}}),
            ("POST", "/api/v2/outputs/save", {"json": {"ids": [rec.id], "zip": True}}),
            ("PUT", "/api/v2/networks/lora/m_prev/preview", {"json": {"file_id": rec.id}}),
            ("POST", "/api/v2/gallery/compare", {"json": {"id_a": rec.id, "infotext_b": "other"}}),
        ):
            resp = server.client.request(method, url, **kwargs)
            assert resp.status_code == 404, f"{method} {url} -> {resp.status_code}: {resp.text}"
            assert b"SECRET_PROMPT" not in resp.content

        # Upload record pointing outside uploads_dir must also be rejected by get() and get_many()
        up = files.add_upload_bytes(_png_bytes((2, 3, 4)), "ok.png")
        with files._lock:
            files._db.execute("UPDATE files SET path = ? WHERE id = ?", (os.path.realpath(outside_upload), up.id))
        assert files.get(up.id) is None
        assert files.get_many([up.id]) == {}


# ---------------------------------------------------------------------------
# diff_parameters / gallery compare preserve non-finite scalars & LoRA weights as strings
# ---------------------------------------------------------------------------


def test_compare_preserves_non_finite_infotext_and_lora_weights_as_strings(webui, tmp_path):
    from remi.library import diff_parameters

    infotext_a = "cat <lora:style:inf> <lora:x:nan> <lora:neg_inf:-inf>\nSteps: 20, CFG scale: inf, Model: Infinity, Seed: 1e999, CustomVal: nan"
    infotext_b = "cat\nSteps: 20"

    diff = diff_parameters(infotext_a, infotext_b)
    core_by_key = {item["key"]: item for item in diff["core_diffs"]}
    assert core_by_key["CFG scale"]["a"] == "inf" and isinstance(core_by_key["CFG scale"]["a"], str)
    assert core_by_key["Model"]["a"] == "Infinity" and isinstance(core_by_key["Model"]["a"], str)
    assert core_by_key["Seed"]["a"] == "1e999" and isinstance(core_by_key["Seed"]["a"], str)

    ext_by_key = {item["key"]: item for item in diff["extension_diffs"]}
    assert ext_by_key["CustomVal"]["a"] == "nan" and isinstance(ext_by_key["CustomVal"]["a"], str)

    lora_by_name = {item["name"]: item for item in diff["lora_diffs"]}
    assert lora_by_name["style"]["a"] == "inf" and isinstance(lora_by_name["style"]["a"], str)
    assert lora_by_name["x"]["a"] == "nan" and isinstance(lora_by_name["x"]["a"], str)
    assert lora_by_name["neg_inf"]["a"] == "-inf" and isinstance(lora_by_name["neg_inf"]["a"], str)

    assert diff["adopt_a_patch"]["raw_params"]["Model"] == "Infinity"
    assert diff["adopt_a_patch"]["core"]["cfg_scale"] == "inf"

    with server_app(tmp_path / "data") as server:
        resp = server.client.post(
            "/api/v2/gallery/compare",
            json={"infotext_a": infotext_a, "infotext_b": infotext_b},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        http_core = {item["key"]: item for item in payload["core_diffs"]}
        http_lora = {item["name"]: item for item in payload["lora_diffs"]}
        assert http_core["CFG scale"]["a"] == "inf"
        assert http_core["Model"]["a"] == "Infinity"
        assert http_core["Seed"]["a"] == "1e999"
        assert http_lora["style"]["a"] == "inf"
        assert http_lora["x"]["a"] == "nan"
        assert payload["adopt_a_patch"]["raw_params"]["Model"] == "Infinity"


# ---------------------------------------------------------------------------
# Auth.revoke lock hygiene under concurrent revoke & verify
# ---------------------------------------------------------------------------


def test_auth_revoke_concurrent_revoke_and_verify_with_persistence(tmp_path):
    from remi.server.auth import Auth

    revoked_file = tmp_path / "revoked.json"
    auth = Auth(credentials={"alice": "secret"}, secret=b"k" * 32, revoked_path=revoked_file)
    tokens = [auth.issue("alice") for _ in range(16)]
    assert all(auth.verify(t) == "alice" for t in tokens)

    def _revoke_and_check(tok: str) -> str | None:
        auth.revoke(tok)
        return auth.verify(tok)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_revoke_and_check, tokens))
    assert all(r is None for r in results)

    # Reload from disk into a fresh Auth instance -> all tokens remain revoked
    reloaded = Auth(credentials={"alice": "secret"}, secret=b"k" * 32, revoked_path=revoked_file)
    assert all(reloaded.verify(t) is None for t in tokens)


# ---------------------------------------------------------------------------
# POST /api/v2/infotext/image returns 415 (never 500) on truncated/corrupt/bomb PNGs
# ---------------------------------------------------------------------------


def test_infotext_image_and_pnginfo_return_415_on_unreadable_or_bomb_images(webui, tmp_path, monkeypatch):
    with server_app(tmp_path / "data") as server:
        files = server.services.files
        files.outputs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Truncated PNG registered via register_output
        trunc_path = files.outputs_dir / "truncated.png"
        trunc_path.write_bytes(b"\x89PNG\r\n\x1a\ntruncated_garbage")
        rec_trunc = files.register_output(trunc_path)
        assert rec_trunc.is_image is True

        for route in ("/api/v2/infotext/image", "/api/v2/pnginfo"):
            resp = server.client.post(route, json={"file_id": rec_trunc.id})
            assert resp.status_code == 415, f"{route} -> {resp.status_code}: {resp.text}"
            assert resp.json()["detail"] == "cannot read the image"
            assert str(tmp_path) not in resp.text

        # 2. Valid PNG uploaded, then corrupted on disk before inspection
        up_resp = server.client.post("/api/v2/uploads", files={"file": ("ok.png", _png_bytes((12, 34, 56)), "image/png")})
        assert up_resp.status_code == 201
        upload_id = up_resp.json()["id"]
        up_record = files.get(upload_id)
        assert up_record is not None
        Path(up_record.path).write_bytes(b"corrupted-not-an-image")

        for route in ("/api/v2/infotext/image", "/api/v2/pnginfo"):
            resp = server.client.post(route, json={"file_id": upload_id})
            assert resp.status_code == 415, f"{route} -> {resp.status_code}: {resp.text}"
            assert resp.json()["detail"] == "cannot read the image"
            assert str(tmp_path) not in resp.text

        # 3. Genuine Image.DecompressionBombError (32x32 = 1024 px > 2 * MAX_IMAGE_PIXELS)
        rec_bomb = files.save_output_image(Image.new("RGB", (32, 32), (7, 8, 9)), "a cat\nSteps: 10")
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
        for route in ("/api/v2/infotext/image", "/api/v2/pnginfo"):
            resp = server.client.post(route, json={"file_id": rec_bomb.id})
            assert resp.status_code == 415, f"{route} -> {resp.status_code}: {resp.text}"
            assert resp.json()["detail"] == "the image has too many pixels to read"
            assert str(tmp_path) not in resp.text


# ---------------------------------------------------------------------------
# POST /sdapi/v1/png-info handles WebP non-UTF-8 XMP bytes
# ---------------------------------------------------------------------------


def test_sdapi_png_info_handles_webp_non_utf8_xmp_bytes(webui, tmp_path):
    import base64

    from remi import prompt_tools
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    webp_buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(webp_buf, format="WEBP", xmp=b"\xff\xfe\x00\x01binary_xmp")
    b64_webp = base64.b64encode(webp_buf.getvalue()).decode("ascii")

    services = Services.create(tmp_path / "data", dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:
            resp = client.post("/sdapi/v1/png-info", json={"image": b64_webp})
            assert resp.status_code == 200, resp.text
            payload = resp.json()
            assert payload["info"] == ""
            assert payload["items"]["xmp"] == "<14 bytes>"
    finally:
        prompt_tools.forget(services)
        services.close()


# ---------------------------------------------------------------------------
# Iterative check_finite_json & ControlNet render_pose / jobs deep nesting
# ---------------------------------------------------------------------------


def test_deep_json_and_controlnet_render_pose_and_jobs(webui, tmp_path, monkeypatch):
    import importlib.util

    from modules import scripts

    deep_dict = _nested(1200)
    assert nodes.check_finite_json(deep_dict) is deep_dict

    deep_list: list = [1]
    for _ in range(1199):
        deep_list = [deep_list]
    assert nodes.check_finite_json(deep_list) is deep_list

    deep_bad: dict = {"leaf": float("nan")}
    for _ in range(1199):
        deep_bad = {"child": deep_bad}
    with pytest.raises(ParamError):
        nodes.check_finite_json(deep_bad)

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "extensions-builtin" / "forge_legacy_preprocessors"))
    harness_spec = importlib.util.spec_from_file_location("remi_port_harness", repo_root / "tests" / "remi_port_harness.py")
    assert harness_spec is not None and harness_spec.loader is not None
    harness = importlib.util.module_from_spec(harness_spec)
    harness_spec.loader.exec_module(harness)

    cn_ext = repo_root / "extensions-builtin" / "sd_forge_controlnet"
    cn_script_path = cn_ext / "scripts" / "controlnet.py"
    cn_mod = harness.load_script(cn_script_path, "cn_deep_pose_under_test")
    cn_data = scripts.ScriptClassData(cn_mod.ControlNetForForgeOfficial, str(cn_script_path), str(cn_ext), cn_mod)

    with webui_state(script_classes=[cn_data]), server_app(tmp_path / "data") as server:
        # 1. render_pose with 2000-deep JSON bracket string -> 200 error toast (never 500)
        resp_str = server.client.post(
            "/api/v2/actions/controlnet/render_pose",
            json={"tab": "txt2img", "values": {}, "unit": 0, "pose": "[" * 2000 + "1" + "]" * 2000},
        )
        assert resp_str.status_code == 200, resp_str.text
        assert resp_str.json()["__toast__"]["level"] == "error"

        # 2. render_pose with 1200-deep dict -> 200 error toast (never 500)
        resp_dict = server.client.post(
            "/api/v2/actions/controlnet/render_pose",
            json={"tab": "txt2img", "values": {}, "unit": 0, "pose": deep_dict},
        )
        assert resp_dict.status_code == 200, resp_dict.text
        assert resp_dict.json()["__toast__"]["level"] == "error"

        # 3. POST /api/v2/jobs with controlnet.unit_0.pose 1200-deep dict -> 201 or 422 (never 500)
        resp_job_enabled = server.client.post(
            "/api/v2/jobs",
            json={"tab": "txt2img", "core": {"prompt": "cat"}, "scripts": {"controlnet": {"unit_0": {"enabled": True, "pose": deep_dict}}}},
        )
        assert resp_job_enabled.status_code in (201, 422), resp_job_enabled.text

        resp_job_disabled = server.client.post(
            "/api/v2/jobs",
            json={"tab": "txt2img", "core": {"prompt": "cat"}, "scripts": {"controlnet": {"unit_0": {"enabled": False, "pose": deep_dict}}}},
        )
        assert resp_job_disabled.status_code == 201, resp_job_disabled.text
        job_detail = server.client.get(f"/api/v2/jobs/{resp_job_disabled.json()['job_id']}")
        assert job_detail.status_code == 200, job_detail.text


# ---------------------------------------------------------------------------
# Non-bytes EXIF UserComment (0x9286 LONG) returns 200 across all 3 metadata routes
# ---------------------------------------------------------------------------


def test_non_bytes_exif_user_comment_across_all_metadata_routes(webui, tmp_path):
    import base64
    import struct

    from remi import prompt_tools
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    tiff_long = (
        b"II"
        + struct.pack("<HI", 42, 8)
        + struct.pack("<H", 1)
        + struct.pack("<HHII", 0x8769, 4, 1, 26)
        + struct.pack("<I", 0)
        + struct.pack("<H", 1)
        + struct.pack("<HHII", 0x9286, 4, 1, 12345)
        + struct.pack("<I", 0)
    )
    jpg_buf = io.BytesIO()
    Image.new("RGB", (8, 8), (4, 5, 6)).save(jpg_buf, format="JPEG", exif=b"Exif\x00\x00" + tiff_long)
    jpg_bytes = jpg_buf.getvalue()
    b64_jpg = base64.b64encode(jpg_bytes).decode("ascii")

    services = Services.create(tmp_path / "data", dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:
            up = client.post("/api/v2/uploads", files={"file": ("bad_exif.jpg", jpg_bytes, "image/jpeg")})
            assert up.status_code == 201, up.text
            file_id = up.json()["id"]

            r_info = client.post("/api/v2/infotext/image", json={"file_id": file_id})
            assert r_info.status_code == 200, r_info.text

            r_pnginfo = client.post("/api/v2/pnginfo", json={"file_id": file_id})
            assert r_pnginfo.status_code == 200, r_pnginfo.text

            r_sdapi = client.post("/sdapi/v1/png-info", json={"image": b64_jpg})
            assert r_sdapi.status_code == 200, r_sdapi.text
    finally:
        prompt_tools.forget(services)
        services.close()


# ---------------------------------------------------------------------------
# POST /api/v2/actions/controlnet/render_pose handles malformed entries & huge coords
# ---------------------------------------------------------------------------


def test_controlnet_render_pose_handles_malformed_entries_and_large_coords(webui, tmp_path, monkeypatch):
    import importlib.util

    from modules import scripts

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "extensions-builtin" / "forge_legacy_preprocessors"))
    harness_spec = importlib.util.spec_from_file_location("remi_port_harness", repo_root / "tests" / "remi_port_harness.py")
    assert harness_spec is not None and harness_spec.loader is not None
    harness = importlib.util.module_from_spec(harness_spec)
    harness_spec.loader.exec_module(harness)

    cn_ext = repo_root / "extensions-builtin" / "sd_forge_controlnet"
    cn_script_path = cn_ext / "scripts" / "controlnet.py"
    cn_mod = harness.load_script(cn_script_path, "cn_malformed_pose_under_test")
    cn_data = scripts.ScriptClassData(cn_mod.ControlNetForForgeOfficial, str(cn_script_path), str(cn_ext), cn_mod)

    malformed_poses = [
        {"canvas_width": 512, "canvas_height": 512, "people": [123]},
        {"canvas_width": 512, "canvas_height": 512, "people": [None]},
        {"canvas_width": 512, "canvas_height": 512, "people": [{}]},
        {"canvas_width": 512, "canvas_height": 512, "people": [], "animals": [123]},
        {"canvas_width": 512, "canvas_height": 512, "people": [], "animals": [[1e15, 1e15, 1.0] * 17]},
        {"canvas_width": 512, "canvas_height": 512, "people": [{"pose_keypoints_2d": [1e200, 1e200, 1.0] + [0.0, 0.0, 1.0] * 17}]},
    ]
    valid_human = {"canvas_width": 64, "canvas_height": 32, "people": [{"pose_keypoints_2d": [10, 10, 1, 20, 20, 1] + [0, 0, 0] * 16}]}
    valid_animal = {
        "canvas_width": 64,
        "canvas_height": 32,
        "people": [],
        "animals": [[coord for i in range(17) for coord in (8 + 3 * i, 4 + i, 1)]],
    }

    with webui_state(script_classes=[cn_data]), server_app(tmp_path / "data") as server:
        for bad_pose in malformed_poses:
            resp = server.client.post(
                "/api/v2/actions/controlnet/render_pose",
                json={"tab": "txt2img", "values": {}, "unit": 0, "pose": bad_pose},
            )
            assert resp.status_code == 200, f"{bad_pose!r} -> {resp.status_code}: {resp.text}"
            payload = resp.json()
            assert payload["__toast__"]["level"] == "error", payload

        for valid_pose in (valid_human, valid_animal):
            ok_resp = server.client.post(
                "/api/v2/actions/controlnet/render_pose",
                json={"tab": "txt2img", "values": {}, "unit": 0, "pose": valid_pose},
            )
            assert ok_resp.status_code == 200, ok_resp.text
            pose_out = ok_resp.json()["__outputs__"]["pose"]
            assert pose_out["unit"] == 0
            assert pose_out["pose"] == valid_pose
            file_resp = server.client.get(f"/api/v2/files/{pose_out['image']}")
            assert file_resp.status_code == 200
            with Image.open(io.BytesIO(file_resp.content)) as rendered:
                assert rendered.size == (64, 32)


# ---------------------------------------------------------------------------
# sd-postprocess-suite preset_save & apply_send input hardening
# ---------------------------------------------------------------------------


def test_postprocess_suite_preset_and_apply_send_hardening(webui, tmp_path, monkeypatch):
    import importlib.util

    from modules import scripts

    repo_root = Path(__file__).resolve().parents[1]
    pp_ext = repo_root / "extensions-builtin" / "sd-postprocess-suite"
    pp_script_path = pp_ext / "scripts" / "postprocess_suite.py"
    monkeypatch.syspath_prepend(str(pp_ext))
    spec = importlib.util.spec_from_file_location("pp_suite_preset_under_test", pp_script_path)
    assert spec is not None and spec.loader is not None
    pp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp_mod)

    monkeypatch.setattr(pp_mod, "_PRESET_DIR", str(tmp_path / "pp_presets"))
    monkeypatch.setattr(pp_mod, "_LEGACY_PRESET_DIR", str(tmp_path / "pp_legacy"))
    pp_data = scripts.ScriptClassData(pp_mod.PostProcessSuite, str(pp_script_path), str(pp_ext), pp_mod)

    with webui_state(script_classes=[pp_data]), server_app(tmp_path / "data") as server:
        # 1. Non-string preset names must return 200 warning toast (never 500 AttributeError)
        for action in ("preset_save", "preset_load", "preset_delete"):
            for bad_preset in (123, ["x"], {"a": 1}):
                resp = server.client.post(
                    f"/api/v2/actions/post_processing_suite/{action}",
                    json={"tab": "txt2img", "values": {"preset": bad_preset}},
                )
                assert resp.status_code == 200, f"{action}({bad_preset!r}) -> {resp.status_code}: {resp.text}"
                assert resp.json()["__toast__"]["level"] == "warning"

        # 2. Happy-path preset save/load/delete still works for valid strings
        save_ok = server.client.post(
            "/api/v2/actions/post_processing_suite/preset_save",
            json={"tab": "txt2img", "values": {"preset": "Smoke Preset", "enable": True, "sepia_enabled": True, "sepia_strength": 0.5}},
        )
        assert save_ok.status_code == 200, save_ok.text
        assert save_ok.json()["__toast__"]["level"] == "success"
        load_ok = server.client.post(
            "/api/v2/actions/post_processing_suite/preset_load",
            json={"tab": "txt2img", "values": {"preset": "Smoke Preset"}},
        )
        assert load_ok.status_code == 200, load_ok.text
        assert load_ok.json()["sepia_strength"]["value"] == 0.5

        # 3. apply_send with .txt upload, corrupt PNG, and decompression-bomb PNG -> 200 warning toast (never 500)
        up_txt = server.client.post("/api/v2/uploads", files={"file": ("notes.txt", b"hello text", "text/plain")})
        assert up_txt.status_code == 201, up_txt.text
        txt_resp = server.client.post(
            "/api/v2/actions/post_processing_suite/apply_send",
            json={"tab": "txt2img", "values": {"enable": True}, "file_id": up_txt.json()["id"]},
        )
        assert txt_resp.status_code == 200, txt_resp.text
        assert txt_resp.json()["__toast__"]["level"] == "warning"

        up_png = server.client.post("/api/v2/uploads", files={"file": ("ok.png", _png_bytes((20, 40, 200)), "image/png")})
        assert up_png.status_code == 201, up_png.text
        png_id = up_png.json()["id"]

        # Valid image happy-path smoke check
        ok_send = server.client.post(
            "/api/v2/actions/post_processing_suite/apply_send",
            json={"tab": "txt2img", "values": {"enable": True, "sepia_enabled": True, "sepia_strength": 1.0}, "file_id": png_id},
        )
        assert ok_send.status_code == 200, ok_send.text
        assert ok_send.json()["__outputs__"]["sent"] != png_id

        # Corrupt PNG on disk -> 200 warning toast (never 500)
        rec = server.services.files.get(png_id)
        assert rec is not None
        Path(rec.path).write_bytes(b"\x89PNG\r\n\x1a\ncorrupt_bytes")
        corrupt_resp = server.client.post(
            "/api/v2/actions/post_processing_suite/apply_send",
            json={"tab": "txt2img", "values": {"enable": True}, "file_id": png_id},
        )
        assert corrupt_resp.status_code == 200, corrupt_resp.text
        assert corrupt_resp.json()["__toast__"]["level"] == "warning"

        # DecompressionBombError -> 200 warning toast (never 500)
        up_bomb = server.client.post("/api/v2/uploads", files={"file": ("bomb.png", _png_bytes((1, 2, 3), (32, 32)), "image/png")})
        assert up_bomb.status_code == 201, up_bomb.text
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
        bomb_resp = server.client.post(
            "/api/v2/actions/post_processing_suite/apply_send",
            json={"tab": "txt2img", "values": {"enable": True}, "file_id": up_bomb.json()["id"]},
        )
        assert bomb_resp.status_code == 200, bomb_resp.text
        assert bomb_resp.json()["__toast__"]["level"] == "warning"


# ---------------------------------------------------------------------------
# anima-lora-stage-scheduler template_save & template_changed hardening
# ---------------------------------------------------------------------------


def test_anima_lora_stage_scheduler_template_actions_hardening(webui, tmp_path, monkeypatch):
    import importlib.util

    from modules import paths_internal, scripts

    repo_root = Path(__file__).resolve().parents[1]
    lss_ext = repo_root / "extensions-builtin" / "anima-lora-stage-scheduler"
    lss_script_path = lss_ext / "scripts" / "anima_lora_stage_scheduler.py"
    monkeypatch.syspath_prepend(str(lss_ext))
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))

    spec = importlib.util.spec_from_file_location("lss_template_under_test", lss_script_path)
    assert spec is not None and spec.loader is not None
    lss_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lss_mod)
    lss_data = scripts.ScriptClassData(lss_mod.Script, str(lss_script_path), str(lss_ext), lss_mod)

    with webui_state(script_classes=[lss_data]), server_app(tmp_path / "data") as server:
        # 1. Non-dict *_timing on template_save -> 422 ValidationError (never 500 ValueError)
        for panel, bad_timing in (("base", 456), ("hr", "bad"), ("base", [1, 2])):
            resp = server.client.post(
                "/api/v2/actions/anima_lora_stage_scheduler/template_save",
                json={"tab": "txt2img", "values": {f"{panel}_timing": bad_timing}, "name": "t1", "panel": panel, "auto": True},
            )
            assert resp.status_code == 422, f"{panel}/{bad_timing!r} -> {resp.status_code}: {resp.text}"
            assert resp.json()["errors"][0]["param"] == f"{panel}_timing"

        # 2. Unhashable *_template on base_template_changed & hr_template_changed -> 200 (never 500 TypeError)
        for prefix in ("base", "hr"):
            for bad_tpl in (["bad"], {"a": 1}, 123):
                resp = server.client.post(
                    f"/api/v2/actions/anima_lora_stage_scheduler/{prefix}_template_changed",
                    json={"tab": "txt2img", "values": {f"{prefix}_template": bad_tpl}},
                )
                assert resp.status_code == 200, f"{prefix}({bad_tpl!r}) -> {resp.status_code}: {resp.text}"
                assert f"{prefix}_timing" in resp.json()

        # 3. Happy-path template_save + base_template_changed
        ok_save = server.client.post(
            "/api/v2/actions/anima_lora_stage_scheduler/template_save",
            json={
                "tab": "txt2img",
                "values": {"base_template": "仅人物特征阶段介入", "base_timing": {"auto": False, "start": 0.1, "end": 0.7, "peak": 0.4, "strength": 1.0}},
                "name": "smoke_tpl",
                "panel": "base",
                "auto": False,
            },
        )
        assert ok_save.status_code == 200, ok_save.text
        assert ok_save.json()["__toast__"]["level"] == "success"
        ok_changed = server.client.post(
            "/api/v2/actions/anima_lora_stage_scheduler/base_template_changed",
            json={"tab": "txt2img", "values": {"base_template": "smoke_tpl"}},
        )
        assert ok_changed.status_code == 200, ok_changed.text
        assert ok_changed.json()["base_timing"]["value"]["start"] == 0.1


# ---------------------------------------------------------------------------
# sd-webui-joycaption-modified load_type & caption hardening
# ---------------------------------------------------------------------------


def test_joycaption_load_type_and_caption_hardening(webui, tmp_path, monkeypatch):
    import importlib.util

    from remi import pages

    repo_root = Path(__file__).resolve().parents[1]
    jc_ext = repo_root / "extensions-builtin" / "sd-webui-joycaption-modified"
    jc_script_path = jc_ext / "scripts" / "joycaption.py"
    monkeypatch.syspath_prepend(str(jc_ext))

    pages.clear()
    spec = importlib.util.spec_from_file_location("jc_caption_under_test", jc_script_path)
    assert spec is not None and spec.loader is not None
    jc_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jc_mod)
    monkeypatch.setattr(jc_mod.jcp, "_user_types_path", lambda: tmp_path / "user_caption_types.json")

    try:
        with server_app(tmp_path / "data") as server:
            # 1. Unhashable/non-string edit_select on load_type -> 200 empty fields (never 500 TypeError)
            for bad_sel in (["bad"], {"a": 1}, 123):
                resp = server.client.post(
                    "/api/v2/actions/page:joycaption/load_type",
                    json={"tab": "txt2img", "values": {"edit_select": bad_sel}},
                )
                assert resp.status_code == 200, f"{bad_sel!r} -> {resp.status_code}: {resp.text}"
                assert resp.json() == {
                    "edit_name": {"value": ""},
                    "edit_any": {"value": ""},
                    "edit_word": {"value": ""},
                    "edit_length": {"value": ""},
                }

            # 2. Happy-path load_type with valid built-in type
            ok_resp = server.client.post(
                "/api/v2/actions/page:joycaption/load_type",
                json={"tab": "txt2img", "values": {"edit_select": "Straightforward"}},
            )
            assert ok_resp.status_code == 200, ok_resp.text
            assert ok_resp.json()["edit_name"] == {"value": "Straightforward"}
            assert ok_resp.json()["edit_any"]["value"].startswith("Write a straightforward caption")

            # 3. Corrupt PNG uploaded then overwritten on disk -> caption returns 422 (never 500)
            up_png = server.client.post("/api/v2/uploads", files={"file": ("corrupt.png", _png_bytes((5, 6, 7)), "image/png")})
            assert up_png.status_code == 201, up_png.text
            png_id = up_png.json()["id"]
            rec = server.services.files.get(png_id)
            assert rec is not None
            Path(rec.path).write_bytes(b"\x89PNG\r\n\x1a\ncorrupt")
            cap_resp = server.client.post(
                "/api/v2/actions/page:joycaption/caption",
                json={"tab": "txt2img", "values": {"image": png_id}},
            )
            assert cap_resp.status_code == 422, cap_resp.text
            assert cap_resp.json()["errors"][0]["param"] == "image"
    finally:
        pages.clear()


# ---------------------------------------------------------------------------
# DecompressionBombError & truncated/corrupt image hardening
# ---------------------------------------------------------------------------


def test_thumb_decompression_bomb_returns_415(webui, tmp_path, monkeypatch):
    with server_app(tmp_path / "data") as server:
        ok_path = tmp_path / "ok.png"
        ok_path.write_bytes(_png_bytes((1, 2, 3), (16, 16)))
        ok_rec = server.services.files.register_output(ok_path)
        assert server.client.get(f"/api/v2/thumbs/{ok_rec.id}").status_code == 200

        bomb_path = tmp_path / "bomb.png"
        bomb_path.write_bytes(_png_bytes((4, 5, 6), (16, 16)))
        bomb_rec = server.services.files.register_output(bomb_path)

        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
        resp = server.client.get(f"/api/v2/thumbs/{bomb_rec.id}")
        assert resp.status_code == 415, resp.text
        assert "too many pixels" in resp.json()["detail"]


def test_action_files_image_and_controlnet_preview_bomb(webui, tmp_path, monkeypatch):
    import importlib.util

    from modules import scripts
    from remi.ui.files import ActionFiles

    repo_root = Path(__file__).resolve().parents[1]
    cn_ext = repo_root / "extensions-builtin" / "sd_forge_controlnet"
    cn_script_path = cn_ext / "scripts" / "controlnet.py"
    legacy_pre = repo_root / "extensions-builtin" / "forge_legacy_preprocessors"
    monkeypatch.syspath_prepend(str(cn_ext))
    monkeypatch.syspath_prepend(str(legacy_pre))
    from lib_controlnet.unit_param import default_value

    import modules_forge.supported_preprocessor  # noqa: F401

    spec = importlib.util.spec_from_file_location("cn_preview_bomb_under_test", cn_script_path)
    assert spec is not None and spec.loader is not None
    cn_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cn_mod)
    cn_data = scripts.ScriptClassData(cn_mod.ControlNetForForgeOfficial, str(cn_script_path), str(cn_ext), cn_mod)

    with webui_state(script_classes=[cn_data]), server_app(tmp_path / "data") as server:
        ok_path = tmp_path / "ok.png"
        ok_path.write_bytes(_png_bytes((1, 2, 3), (4, 4)))
        ok_rec = server.services.files.register_output(ok_path)

        bomb_path = tmp_path / "bomb.png"
        bomb_path.write_bytes(_png_bytes((4, 5, 6), (16, 16)))
        bomb_rec = server.services.files.register_output(bomb_path)

        action_files = ActionFiles(server.services.files)
        loaded_ok = action_files.image(ok_rec.id)
        assert loaded_ok is not None and loaded_ok.size == (4, 4)

        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
        assert action_files.image(bomb_rec.id) is None

        unit = {**default_value(False), "image": bomb_rec.id}
        resp = server.client.post(
            "/api/v2/actions/controlnet/preview",
            json={"tab": "txt2img", "unit": 0, "values": {"unit_0": unit}},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["__toast__"]["level"] == "warning"


def test_page_values_corrupt_and_bomb_raise_validation_error(webui, tmp_path, monkeypatch):
    from remi import pages, ui
    from remi.ui.files import ActionFiles

    class ProbePage(pages.Page):
        def params(self):
            return ui.Column(ui.Image("image"), ui.ImageList("gallery"), ui.Output("size", kind="text"))

        @ui.action("inspect")
        def inspect(self, ctx, values):
            resolved = self.values(ctx, values)
            image = resolved["image"]
            return {"__outputs__": {"size": f"{image.width}x{image.height}" if image is not None else "none"}}

    raw_png = _png_bytes((10, 20, 30), (16, 16))
    idat = raw_png.index(b"IDAT")
    truncated_png = raw_png[: idat + 10]

    pages.clear()
    page = pages.register(ProbePage("t5-r7-probe", "T5 R7 Probe"))
    try:
        with server_app(tmp_path / "data") as server:
            ok_up = server.client.post("/api/v2/uploads", files={"file": ("ok.png", raw_png, "image/png")})
            assert ok_up.status_code == 201, ok_up.text
            ok_id = ok_up.json()["id"]

            trunc_up = server.client.post("/api/v2/uploads", files={"file": ("trunc.png", truncated_png, "image/png")})
            assert trunc_up.status_code == 201, trunc_up.text
            trunc_id = trunc_up.json()["id"]

            bomb_path = tmp_path / "bomb.png"
            bomb_path.write_bytes(raw_png)
            bomb_rec = server.services.files.register_output(bomb_path)

            # Happy path
            ok_resp = server.client.post("/api/v2/actions/page:t5-r7-probe/inspect", json={"tab": "page:t5-r7-probe", "values": {"image": ok_id}})
            assert ok_resp.status_code == 200, ok_resp.text
            assert ok_resp.json()["__outputs__"]["size"] == "16x16"

            # Direct Page.values on truncated PNG upload -> ui.ValidationError (never raw OSError)
            ctx = ui.ActionContext(tab=page.owner, owner=page.owner, files=ActionFiles(server.services.files))
            with pytest.raises(ui.ValidationError) as trunc_exc:
                page.values(ctx, {"image": trunc_id})
            assert trunc_exc.value.errors[0].param == "image"
            assert trunc_exc.value.errors[0].owner == page.owner

            with pytest.raises(ui.ValidationError) as list_exc:
                page.values(ctx, {"gallery": [trunc_id]})
            assert list_exc.value.errors[0].param == "gallery"

            # HTTP action route on truncated PNG -> 422 (never 500)
            trunc_resp = server.client.post("/api/v2/actions/page:t5-r7-probe/inspect", json={"tab": "page:t5-r7-probe", "values": {"image": trunc_id}})
            assert trunc_resp.status_code == 422, trunc_resp.text
            assert trunc_resp.json()["errors"][0]["param"] == "image"

            # DecompressionBombError on registered output -> ui.ValidationError & HTTP 422 (never 500)
            monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
            with pytest.raises(ui.ValidationError) as bomb_exc:
                page.values(ctx, {"image": bomb_rec.id})
            assert bomb_exc.value.errors[0].param == "image"
            assert bomb_exc.value.errors[0].owner == page.owner

            bomb_resp = server.client.post("/api/v2/actions/page:t5-r7-probe/inspect", json={"tab": "page:t5-r7-probe", "values": {"image": bomb_rec.id}})
            assert bomb_resp.status_code == 422, bomb_resp.text
            assert bomb_resp.json()["errors"][0]["param"] == "image"
    finally:
        pages.clear()


# ---------------------------------------------------------------------------
# Prompt upsample ref, outputs/save, and CMYK store_image hardening
# ---------------------------------------------------------------------------


def test_prompt_upsample_corrupt_and_bomb_refs(webui, tmp_path, monkeypatch):
    import base64

    from remi import prompt_tools as pt

    raw_png = _png_bytes((10, 20, 30), (16, 16))
    idat = raw_png.index(b"IDAT")
    truncated_png = raw_png[: idat + 10]

    monkeypatch.setattr(pt.registry, "group_dirs", [str(tmp_path)])
    with server_app(tmp_path / "data") as server:
        trunc_up = server.client.post("/api/v2/uploads", files={"file": ("trunc.png", truncated_png, "image/png")})
        assert trunc_up.status_code == 201, trunc_up.text
        trunc_id = trunc_up.json()["id"]

        trunc_resp = server.client.post(
            "/api/v2/prompt-tools/upsample",
            json={"prompt": "a cat", "reference_file_ids": [trunc_id]},
        )
        assert trunc_resp.status_code == 422, trunc_resp.text
        assert trunc_resp.json()["errors"][0]["param"] == "reference_file_ids"

        bomb_path = tmp_path / "bomb.png"
        bomb_path.write_bytes(raw_png)
        bomb_rec = server.services.files.register_output(bomb_path)

        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
        bomb_ref_resp = server.client.post(
            "/api/v2/prompt-tools/upsample",
            json={"prompt": "a cat", "reference_file_ids": [bomb_rec.id]},
        )
        assert bomb_ref_resp.status_code == 413, bomb_ref_resp.text

        b64_uri = "data:image/png;base64," + base64.b64encode(raw_png).decode("ascii")
        bomb_b64_resp = server.client.post(
            "/api/v2/prompt-tools/upsample",
            json={
                "prompt": "a cat",
                "reference_images": [b64_uri],
                "local_model_path": str(tmp_path / "missing_model"),
            },
        )
        assert bomb_b64_resp.status_code == 413, bomb_b64_resp.text


def test_outputs_save_corrupt_and_bomb_return_415(webui, tmp_path, monkeypatch):
    import shutil

    from modules import shared
    from remi.server import gallery

    raw_png = _png_bytes((10, 20, 30), (16, 16))
    idat = raw_png.index(b"IDAT")
    truncated_png = raw_png[: idat + 10]

    save_dir = tmp_path / "saved"
    monkeypatch.setattr(shared.opts, "outdir_save", str(save_dir), raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)

    with server_app(tmp_path / "data") as server:
        ok_path = tmp_path / "ok.png"
        ok_path.write_bytes(raw_png)
        ok_rec = server.services.files.register_output(ok_path)
        assert server.client.post("/api/v2/outputs/save", json={"ids": [ok_rec.id]}).status_code == 200

        trunc_path = tmp_path / "trunc.png"
        trunc_path.write_bytes(truncated_png)
        trunc_rec = server.services.files.register_output(trunc_path)
        trunc_resp = server.client.post("/api/v2/outputs/save", json={"ids": [trunc_rec.id]})
        assert trunc_resp.status_code == 415, trunc_resp.text
        assert "cannot read the image" in trunc_resp.json()["detail"]

        bomb_path = tmp_path / "bomb.png"
        bomb_path.write_bytes(raw_png)
        bomb_rec = server.services.files.register_output(bomb_path)

        with monkeypatch.context() as mp:
            mp.setattr(Image, "MAX_IMAGE_PIXELS", 10)
            bomb_resp = server.client.post("/api/v2/outputs/save", json={"ids": [bomb_rec.id]})
            assert bomb_resp.status_code == 415, bomb_resp.text
            assert "too many pixels" in bomb_resp.json()["detail"]

        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("hello", encoding="utf-8")
        txt_rec = server.services.files.register_output(txt_path)

        def fail_copy(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr(shutil, "copy2", fail_copy)
        with pytest.raises(OSError, match="disk full"):
            gallery.save_outputs([txt_rec])


def test_store_image_and_apply_send_cmyk(webui, tmp_path, monkeypatch):
    import importlib.util

    from modules import scripts
    from remi.ui.files import ActionFiles

    repo_root = Path(__file__).resolve().parents[1]
    pp_ext = repo_root / "extensions-builtin" / "sd-postprocess-suite"
    pp_script_path = pp_ext / "scripts" / "postprocess_suite.py"
    monkeypatch.syspath_prepend(str(pp_ext))
    spec = importlib.util.spec_from_file_location("pp_suite_t5_r8_03", pp_script_path)
    assert spec is not None and spec.loader is not None
    pp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp_mod)

    monkeypatch.setattr(pp_mod, "_PRESET_DIR", str(tmp_path / "pp_presets"))
    monkeypatch.setattr(pp_mod, "_LEGACY_PRESET_DIR", str(tmp_path / "pp_legacy"))
    pp_data = scripts.ScriptClassData(pp_mod.PostProcessSuite, str(pp_script_path), str(pp_ext), pp_mod)

    cmyk_buf = io.BytesIO()
    with Image.new("CMYK", (16, 16), (0, 128, 255, 10)) as cmyk_src:
        cmyk_src.save(cmyk_buf, format="JPEG")
    cmyk_bytes = cmyk_buf.getvalue()

    with webui_state(script_classes=[pp_data]), server_app(tmp_path / "data") as server:
        up = server.client.post("/api/v2/uploads", files={"file": ("cmyk.jpg", cmyk_bytes, "image/jpeg")})
        assert up.status_code == 201, up.text
        cmyk_id = up.json()["id"]

        action_files = ActionFiles(server.services.files)
        with Image.open(io.BytesIO(cmyk_bytes)) as cmyk_img:
            assert cmyk_img.mode == "CMYK"
            stored_id = action_files.store_image(cmyk_img, "out.png")
        stored_rec = server.services.files.get(stored_id)
        assert stored_rec is not None and stored_rec.mime == "image/png"
        with Image.open(stored_rec.path) as stored_img:
            assert stored_img.format == "PNG"
            assert stored_img.mode in ("RGB", "RGBA")

        with Image.new("LAB", (8, 8), (128, 128, 128)) as lab_img:
            lab_rec = server.services.files.get(action_files.store_image(lab_img, "lab.png"))
            assert lab_rec is not None
            with Image.open(lab_rec.path) as stored_lab:
                assert stored_lab.mode == "RGB"

        with Image.new("PA", (8, 8), (1, 128)) as pa_img:
            pa_rec = server.services.files.get(action_files.store_image(pa_img, "pa.png"))
            assert pa_rec is not None
            with Image.open(pa_rec.path) as stored_pa:
                assert stored_pa.mode == "RGBA"

        for values in ({"enable": False}, {"enable": True, "sepia_enabled": True, "sepia_strength": 0.5}):
            resp = server.client.post(
                "/api/v2/actions/post_processing_suite/apply_send",
                json={"tab": "txt2img", "values": values, "file_id": cmyk_id},
            )
            assert resp.status_code == 200, resp.text
            sent_id = resp.json()["__outputs__"]["sent"]
            sent_rec = server.services.files.get(sent_id)
            assert sent_rec is not None and sent_rec.mime == "image/png"
            with Image.open(sent_rec.path) as sent_img:
                assert sent_img.format == "PNG"
                assert sent_img.mode in ("RGB", "RGBA")


def test_outputs_save_cmyk_and_non_png_modes(webui, tmp_path, monkeypatch):
    from modules import shared

    save_dir = tmp_path / "saved"
    monkeypatch.setattr(shared.opts, "outdir_save", str(save_dir), raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    cmyk_path = out_dir / "cmyk.jpg"
    with Image.new("CMYK", (16, 16), (0, 128, 255, 10)) as img:
        img.save(cmyk_path, format="JPEG")

    lab_path = out_dir / "lab.tif"
    with Image.new("LAB", (12, 12), (128, 128, 128)) as img:
        img.save(lab_path, format="TIFF")

    i_path = out_dir / "int32.tif"
    with Image.new("I", (12, 12), 128) as img:
        img.save(i_path, format="TIFF")

    pa_path = out_dir / "pa.tif"
    with Image.new("PA", (12, 12), (1, 128)) as img:
        img.save(pa_path, format="TIFF")

    pal_path = out_dir / "pal.png"
    with Image.new("P", (12, 12), 0) as img:
        img.putpalette([255, 0, 0, 0, 255, 0])
        img.save(pal_path, format="PNG", transparency=0)

    la_path = out_dir / "la.png"
    with Image.new("LA", (12, 12), (128, 128)) as img:
        img.save(la_path, format="PNG")

    with server_app(tmp_path / "data") as server:
        cmyk_rec = server.services.files.register_output(cmyk_path, tab="txt2img", infotext="cmyk test\nSteps: 5, Seed: 7, Size: 16x16")
        lab_rec = server.services.files.register_output(lab_path, tab="txt2img")
        i_rec = server.services.files.register_output(i_path, tab="txt2img")
        pa_rec = server.services.files.register_output(pa_path, tab="txt2img")
        pal_rec = server.services.files.register_output(pal_path, tab="txt2img")
        la_rec = server.services.files.register_output(la_path, tab="txt2img")

        resp = server.client.post(
            "/api/v2/outputs/save",
            json={"ids": [cmyk_rec.id, lab_rec.id, i_rec.id, pa_rec.id, pal_rec.id, la_rec.id], "zip": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["saved"]) == 6
        assert body["zip"] is not None

        expected_modes = ["RGB", "RGB", "RGB", "RGBA", "P", "LA"]
        for rel_name, expected_mode in zip(body["saved"], expected_modes, strict=True):
            saved_file = save_dir / rel_name
            assert saved_file.is_file()
            with Image.open(saved_file) as saved_img:
                assert saved_img.format == "PNG"
                assert saved_img.mode == expected_mode
                if expected_mode == "P":
                    assert "transparency" in saved_img.info

        monkeypatch.setattr(shared.opts, "samples_format", "jpg", raising=False)
        jpg_resp = server.client.post(
            "/api/v2/outputs/save",
            json={"ids": [cmyk_rec.id, pal_rec.id, la_rec.id, lab_rec.id], "zip": False},
        )
        assert jpg_resp.status_code == 200, jpg_resp.text
        for rel_name in jpg_resp.json()["saved"]:
            saved_file = save_dir / rel_name
            assert saved_file.is_file()
            with Image.open(saved_file) as saved_img:
                assert saved_img.format == "JPEG"
                assert saved_img.mode in ("RGB", "CMYK")


def test_launch_utils_run_extension_installer_timeout_failure_and_shell_metachars(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch.py"])
    from modules import launch_utils

    reported: list[str] = []
    monkeypatch.setattr(launch_utils.errors, "report", lambda message, *a, **kw: reported.append(str(message)))

    # 1. Directory without install.py succeeds as a no-op
    empty_ext = tmp_path / "no_installer"
    empty_ext.mkdir()
    assert launch_utils.run_extension_installer(str(empty_ext)) is True
    assert reported == []

    # 2. Shell metacharacters / quotes in extension_dir do not trigger shell expansion; env & stdin are non-interactive
    pwned_marker = tmp_path / "pwned_marker"
    ok_marker = tmp_path / "ok_marker.json"
    monkeypatch.setenv("REMI_TEST_PWNED_MARKER", str(pwned_marker))
    tricky_name = 'ext & echo pwned > "%REMI_TEST_PWNED_MARKER%"' if os.name == "nt" else 'ext "quoted" $(touch "$REMI_TEST_PWNED_MARKER")'
    tricky_ext = tmp_path / tricky_name
    tricky_ext.mkdir()
    (tricky_ext / "install.py").write_text(
        "import json, os, pathlib, sys\n"
        f"pathlib.Path({str(ok_marker)!r}).write_text(json.dumps({{\n"
        "    'pip_no_input': os.environ.get('PIP_NO_INPUT'),\n"
        "    'git_prompt': os.environ.get('GIT_TERMINAL_PROMPT'),\n"
        "    'stdin': sys.stdin.read(),\n"
        "}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    assert launch_utils.run_extension_installer(str(tricky_ext), timeout=15.0) is True
    assert not pwned_marker.exists()
    recorded_env = json.loads(ok_marker.read_text(encoding="utf-8"))
    assert recorded_env == {"pip_no_input": "1", "git_prompt": "0", "stdin": ""}
    assert reported == []

    # 3. Non-zero exit code returns False and surfaces stderr + exit code via errors.report
    fail_ext = tmp_path / "failing_ext"
    fail_ext.mkdir()
    (fail_ext / "install.py").write_text(
        "import sys\nsys.stderr.write('boom install failed\\n')\nsys.exit(3)\n",
        encoding="utf-8",
    )
    assert launch_utils.run_extension_installer(str(fail_ext), timeout=15.0) is False
    assert len(reported) == 1
    assert "exit code 3" in reported[0]
    assert "boom install failed" in reported[0]

    # 4. Hanging installer + spawned child process group are killed cleanly on timeout
    reported.clear()
    hang_ext = tmp_path / "hanging_ext"
    hang_ext.mkdir()
    child_pid_file = tmp_path / "grandchild.pid"
    (hang_ext / "install.py").write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    assert launch_utils.run_extension_installer(str(hang_ext), timeout=0.4) is False
    assert len(reported) == 1 and "Timed out" in reported[0]
    if child_pid_file.is_file():
        grandchild_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
        deadline = time.monotonic() + 2.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.05)
        assert not alive, f"grandchild process {grandchild_pid} survived installer timeout cleanup"



