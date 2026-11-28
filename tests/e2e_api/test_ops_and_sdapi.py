"""Opaque-box E2E tests for F9 (Workers, Diagnostics, Extensions Admin, Config States, Networks & Prompt Tools) and F10 (/sdapi/v1/* Compatibility)."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Callable
from typing import Any

import pytest
from PIL import Image


def _deep_dict(depth: int) -> dict[str, Any]:
    node: Any = "leaf"
    for _ in range(depth):
        node = {"nested": node}
    assert isinstance(node, dict)
    return node


def _decode_b64_png(b64_text: str) -> tuple[str | None, tuple[int, int]]:
    raw = base64.b64decode(b64_text)
    with Image.open(io.BytesIO(raw)) as img:
        img.load()
        return img.format, img.size


def _assert_admin_error(payload: dict[str, Any]) -> None:
    assert isinstance(payload.get("detail"), str) and payload["detail"]
    assert isinstance(payload.get("template"), str) and payload["template"]
    assert isinstance(payload.get("params"), dict)


# ============================================================================
#
#            Networks & Prompt Tools (Happy Path)
# ============================================================================


def test_workers_telemetry_and_detail(live_server: Any) -> None:
    resp = live_server.http.get("/api/v2/workers")
    assert resp.status_code == 200
    body = resp.json()

    assert body["mode"] == "inprocess"
    assert body["dispatch_active"] is False
    assert body["preview"] is False
    assert isinstance(body["workers"], list) and len(body["workers"]) >= 1
    assert isinstance(body["mlx"], dict)
    assert isinstance(body["mlx"]["available"], bool)
    assert isinstance(body["mlx"]["platform_supported"], bool)

    worker_ids = {w["worker_id"] for w in body["workers"]}
    assert "worker-inprocess" in worker_ids

    detail_resp = live_server.http.get("/api/v2/workers/worker-inprocess")
    assert detail_resp.status_code == 200
    worker = detail_resp.json()
    assert worker["worker_id"] == "worker-inprocess"
    assert worker["backend"] == "inprocess"
    assert worker["state"] == "idle"
    assert worker["restarts"] == 0
    assert isinstance(worker["uptime_s"], (int, float))


def test_worker_inprocess_restart_increments_restarts(live_server: Any) -> None:
    before = live_server.http.get("/api/v2/workers/worker-inprocess").json()
    assert before["restarts"] == 0

    first = live_server.http.post("/api/v2/workers/worker-inprocess/restart")
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["ok"] is True
    assert first_data["dispatch_active"] is False
    assert isinstance(first_data["note"], str) and "in-process" in first_data["note"].lower()
    assert first_data["worker"]["worker_id"] == "worker-inprocess"
    assert first_data["worker"]["restarts"] == 1

    second = live_server.http.post("/api/v2/workers/worker-inprocess/restart")
    assert second.status_code == 200
    assert second.json()["worker"]["restarts"] == 2

    after = live_server.http.get("/api/v2/workers/worker-inprocess").json()
    assert after["restarts"] == 2


def test_extensions_list_and_diagnostics(tiny_anima_server: Any) -> None:
    ext_resp = tiny_anima_server.http.get("/api/v2/extensions")
    assert ext_resp.status_code == 200
    ext_data = ext_resp.json()
    assert isinstance(ext_data["items"], list) and len(ext_data["items"]) > 0
    assert ext_data["disable_all"] in ("none", "extra", "all")
    assert isinstance(ext_data["access"], bool)
    assert isinstance(ext_data["restartable"], bool)
    assert isinstance(ext_data["index_url"], str) and ext_data["index_url"]

    first_ext = ext_data["items"][0]
    assert {"name", "enabled", "builtin", "can_update"} <= set(first_ext)

    diag_resp = tiny_anima_server.http.get("/api/v2/extensions/diagnostics")
    assert diag_resp.status_code == 200
    diag = diag_resp.json()
    assert isinstance(diag["api_version"], str) and diag["api_version"]
    assert isinstance(diag["schema_version"], int) and diag["schema_version"] >= 1

    summary = diag["summary"]
    items = diag["items"]
    assert isinstance(items, list) and len(items) == len(ext_data["items"])
    assert summary["total"] == len(items)
    assert summary["healthy"] + summary["degraded"] + summary["error"] + summary["disabled"] == summary["total"]

    for item in items:
        assert item["health"] in ("healthy", "degraded", "error", "disabled")
        assert item["manifest_status"] in ("valid", "missing", "invalid", "incompatible")
        assert "load_time_ms" in item
        assert isinstance(item["recent_errors"], list)


def test_config_states_lifecycle(live_server: Any) -> None:
    create_resp = live_server.http.post("/api/v2/extensions/config-states", json={"name": "e2e-checkpoint"})
    assert create_resp.status_code == 201
    created = create_resp.json()
    state_id = created["id"]
    assert created["name"] == "e2e-checkpoint"
    assert state_id.endswith(".json")
    assert isinstance(created["extensions"], int)

    list_resp = live_server.http.get("/api/v2/extensions/config-states")
    assert list_resp.status_code == 200
    listed_ids = [entry["id"] for entry in list_resp.json()]
    assert state_id in listed_ids

    detail_resp = live_server.http.get(f"/api/v2/extensions/config-states/{state_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == state_id
    assert {"saved", "current"} <= set(detail["webui"])
    assert isinstance(detail["items"], list)

    restore_resp = live_server.http.post(
        f"/api/v2/extensions/config-states/{state_id}/restore",
        json={"what": "extensions"},
    )
    assert restore_resp.status_code == 200
    assert restore_resp.json() == {"reload": True}

    delete_resp = live_server.http.delete(f"/api/v2/extensions/config-states/{state_id}")
    assert delete_resp.status_code == 204

    gone_resp = live_server.http.get(f"/api/v2/extensions/config-states/{state_id}")
    assert gone_resp.status_code == 404


def test_sysinfo_check_valid_and_tampered(live_server: Any) -> None:
    sysinfo_json_resp = live_server.http.get("/api/v2/sysinfo")
    assert sysinfo_json_resp.status_code == 200
    sysinfo_data = sysinfo_json_resp.json()
    assert {"Platform", "Python", "Checksum", "CPU", "RAM"} <= set(sysinfo_data)

    raw_resp = live_server.http.get("/internal/sysinfo")
    assert raw_resp.status_code == 200
    raw_text = raw_resp.text

    valid_resp = live_server.http.post("/api/v2/sysinfo/check", json={"text": raw_text})
    assert valid_resp.status_code == 200
    assert valid_resp.json() == {"valid": True}

    tampered = raw_text.replace('"Platform": "', '"Platform": "tampered-')
    assert tampered != raw_text
    tampered_resp = live_server.http.post("/api/v2/sysinfo/check", json={"text": tampered})
    assert tampered_resp.status_code == 200
    assert tampered_resp.json() == {"valid": False}


def test_networks_and_prompt_tools_config(
    live_server: Any,
    tiny_anima_server: Any,
) -> None:
    net_resp = live_server.http.get("/api/v2/networks")
    assert net_resp.status_code == 200
    kinds = net_resp.json()
    assert isinstance(kinds, list)
    for kind in kinds:
        assert {"id", "title", "action", "negative", "fields", "builtin"} <= set(kind)
    assert "lora" in {kind["id"] for kind in kinds}

    live_lora_resp = live_server.http.get("/api/v2/networks/lora")
    assert live_lora_resp.status_code == 200
    live_lora_page = live_lora_resp.json()
    assert {"items", "total", "next_cursor"} <= set(live_lora_page)
    assert isinstance(live_lora_page["items"], list)
    assert isinstance(live_lora_page["total"], int)
    assert live_lora_page["total"] >= len(live_lora_page["items"])

    live_facets_resp = live_server.http.get("/api/v2/networks/lora/facets")
    assert live_facets_resp.status_code == 200
    live_facets = live_facets_resp.json()
    assert {"kind", "total", "folder_total", "count", "folders", "archs", "current_arch"} <= set(live_facets)
    assert live_facets["kind"] == "lora"
    assert isinstance(live_facets["folders"], list)
    assert isinstance(live_facets["archs"], list)

    ta_lora_resp = tiny_anima_server.http.get("/api/v2/networks/lora")
    assert ta_lora_resp.status_code == 200
    ta_lora_page = ta_lora_resp.json()
    assert {"items", "total", "next_cursor"} <= set(ta_lora_page)
    assert isinstance(ta_lora_page["items"], list) and len(ta_lora_page["items"]) >= 1
    assert ta_lora_page["total"] >= len(ta_lora_page["items"])
    first_lora = ta_lora_page["items"][0]
    assert {"name", "title", "kind", "path", "folder", "editable", "hidden"} <= set(first_lora)
    assert first_lora["kind"] == "lora"

    ta_search_resp = tiny_anima_server.http.get("/api/v2/networks/lora", params={"q": first_lora["name"], "limit": 1})
    assert ta_search_resp.status_code == 200
    ta_search_page = ta_search_resp.json()
    assert {"items", "total", "next_cursor"} <= set(ta_search_page)
    assert any(item["name"] == first_lora["name"] for item in ta_search_page["items"])

    ta_facets_resp = tiny_anima_server.http.get("/api/v2/networks/lora/facets")
    assert ta_facets_resp.status_code == 200
    ta_facets = ta_facets_resp.json()
    assert {"kind", "total", "folder_total", "count", "folders", "archs", "current_arch"} <= set(ta_facets)
    assert ta_facets["kind"] == "lora"
    assert ta_facets["count"] == ta_lora_page["total"]
    assert ta_facets["total"] == ta_lora_page["total"]
    assert isinstance(ta_facets["folders"], list)
    assert isinstance(ta_facets["archs"], list)

    pt_resp = live_server.http.get("/api/v2/prompt-tools/config")
    assert pt_resp.status_code == 200
    pt_cfg = pt_resp.json()
    assert {"tags", "tools"} <= set(pt_cfg)
    assert {"enabled", "source", "max_results", "default_weight"} <= set(pt_cfg["tags"])
    assert {"enabled", "history", "groups", "language", "toolbar"} <= set(pt_cfg["tools"])


# ============================================================================
#
# ============================================================================


def test_unknown_worker_returns_404(live_server: Any) -> None:
    get_resp = live_server.http.get("/api/v2/workers/unknown_worker")
    assert get_resp.status_code == 404
    assert "detail" in get_resp.json()

    restart_resp = live_server.http.post("/api/v2/workers/unknown_worker/restart")
    assert restart_resp.status_code == 404
    assert "detail" in restart_resp.json()


def test_subprocess_worker_restart_in_preview_mode_returns_409(make_live_server: Callable[..., Any]) -> None:
    server = make_live_server(worker_process=True)
    workers_resp = server.http.get("/api/v2/workers")
    assert workers_resp.status_code == 200
    workers_data = workers_resp.json()
    assert workers_data["mode"] == "subprocess"
    assert workers_data["dispatch_active"] is False
    assert workers_data["preview"] is True
    assert any(w["worker_id"] == "worker-0" for w in workers_data["workers"])

    restart_resp = server.http.post("/api/v2/workers/worker-0/restart")
    assert restart_resp.status_code == 409
    assert "preview" in restart_resp.json()["detail"].lower()


def test_extension_writes_disabled_access_return_403_admin_error(make_live_server: Callable[..., Any]) -> None:
    server = make_live_server(
        cmd_opts_overrides={
            "listen": True,
            "enable_insecure_extension_access": False,
            "disable_extension_access": True,
        },
    )
    status_resp = server.http.get("/api/v2/server/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["extension_access"] is False

    patch_resp = server.http.patch("/api/v2/extensions", json={"disable_all": "extra"})
    assert patch_resp.status_code == 403
    _assert_admin_error(patch_resp.json())

    save_resp = server.http.post("/api/v2/extensions/config-states", json={"name": "blocked"})
    assert save_resp.status_code == 403
    _assert_admin_error(save_resp.json())

    del_resp = server.http.delete("/api/v2/extensions/foo")
    assert del_resp.status_code == 403
    _assert_admin_error(del_resp.json())


def test_delete_builtin_422_and_frozen_settings_409(
    tiny_anima_server: Any,
    make_live_server: Callable[..., Any],
) -> None:
    ext_list = tiny_anima_server.http.get("/api/v2/extensions").json()["items"]
    builtin_ext = next(item for item in ext_list if item["builtin"])

    del_builtin = tiny_anima_server.http.delete(f"/api/v2/extensions/{builtin_ext['name']}")
    assert del_builtin.status_code == 422
    _assert_admin_error(del_builtin.json())

    frozen_server = make_live_server(cmd_opts_overrides={"freeze_settings": True})
    patch_frozen = frozen_server.http.patch("/api/v2/extensions", json={"disable_all": "extra"})
    assert patch_frozen.status_code == 409
    _assert_admin_error(patch_frozen.json())

    saved = frozen_server.http.post("/api/v2/extensions/config-states", json={"name": "frozen-restore-test"})
    assert saved.status_code == 201
    state_id = saved.json()["id"]
    try:
        restore_frozen = frozen_server.http.post(
            f"/api/v2/extensions/config-states/{state_id}/restore",
            json={"what": "extensions"},
        )
        assert restore_frozen.status_code == 409
        _assert_admin_error(restore_frozen.json())
    finally:
        cleanup = frozen_server.http.delete(f"/api/v2/extensions/config-states/{state_id}")
        assert cleanup.status_code == 204


@pytest.mark.parametrize(
    "payload",
    [
        {"url": "file:///etc/passwd"},
        {"url": "ssh://-oProxyCommand=evil/repo"},
        {"url": "https://github.com/example/repo.git", "dirname": "../escape"},
    ],
)
def test_extension_install_rejects_unsafe_inputs_422(live_server: Any, payload: dict[str, Any]) -> None:
    resp = live_server.http.post("/api/v2/extensions/install", json=payload)
    assert resp.status_code == 422
    _assert_admin_error(resp.json())


def test_extension_index_ssrf_and_config_state_traversal(live_server: Any) -> None:
    ssrf_resp = live_server.http.get("/api/v2/extensions/index", params={"url": "http://127.0.0.1:8080/idx.json"})
    assert ssrf_resp.status_code == 422
    _assert_admin_error(ssrf_resp.json())

    traversal_resp = live_server.http.get("/api/v2/extensions/config-states/..%2Fsecret.json")
    assert traversal_resp.status_code == 404


# ============================================================================
#
# ============================================================================


def test_catalogs_and_cmd_flags_redacted(
    tiny_anima_server: Any,
    make_live_server: Callable[..., Any],
    helpers: Any,
) -> None:
    samplers = tiny_anima_server.http.get("/sdapi/v1/samplers")
    assert samplers.status_code == 200
    sampler_items = samplers.json()
    assert isinstance(sampler_items, list) and len(sampler_items) > 0
    assert {"name", "aliases", "options"} <= set(sampler_items[0])

    schedulers = tiny_anima_server.http.get("/sdapi/v1/schedulers")
    assert schedulers.status_code == 200
    scheduler_items = schedulers.json()
    assert isinstance(scheduler_items, list) and len(scheduler_items) > 0
    assert {"name", "label"} <= set(scheduler_items[0])

    sd_models = tiny_anima_server.http.get("/sdapi/v1/sd-models")
    assert sd_models.status_code == 200
    model_items = sd_models.json()
    assert isinstance(model_items, list) and len(model_items) > 0
    assert {"title", "model_name", "filename"} <= set(model_items[0])

    scripts_resp = tiny_anima_server.http.get("/sdapi/v1/scripts")
    assert scripts_resp.status_code == 200
    scripts_data = scripts_resp.json()
    assert isinstance(scripts_data.get("txt2img"), list)
    assert isinstance(scripts_data.get("img2img"), list)

    script_info_resp = tiny_anima_server.http.get("/sdapi/v1/script-info")
    assert script_info_resp.status_code == 200
    assert isinstance(script_info_resp.json(), list)

    secret_srv = make_live_server(
        cmd_opts_overrides={"auth": "alice:top-secret-pw", "api_auth": "apiuser:api-secret-pw"},
    )
    flags_resp = secret_srv.http.get(
        "/sdapi/v1/cmd-flags",
        headers=helpers.basic_auth_header("apiuser", "api-secret-pw"),
    )
    assert flags_resp.status_code == 200
    assert "top-secret-pw" not in flags_resp.text
    assert "api-secret-pw" not in flags_resp.text
    flags_data = flags_resp.json()
    assert flags_data["auth"] and flags_data["auth"] != "alice:top-secret-pw"
    assert flags_data["api_auth"] and flags_data["api_auth"] != "apiuser:api-secret-pw"


def test_options_get_and_post_persists(live_server: Any, helpers: Any) -> None:
    initial_resp = live_server.http.get("/sdapi/v1/options")
    assert initial_resp.status_code == 200
    initial = initial_resp.json()
    assert isinstance(initial["jpeg_quality"], (int, float))
    target_quality = 73 if initial["jpeg_quality"] != 73 else 74

    ws = live_server.ws_connect()
    helpers.recv_ws_until(ws, lambda msg: isinstance(msg, dict) and msg.get("type") == "queue.changed")

    post_resp = live_server.http.post("/sdapi/v1/options", json={"jpeg_quality": target_quality})
    assert post_resp.status_code == 200

    changed_evt = helpers.recv_ws_until(ws, lambda msg: isinstance(msg, dict) and msg.get("type") == "options.changed")
    assert "jpeg_quality" in changed_evt["keys"]

    updated_sdapi = live_server.http.get("/sdapi/v1/options").json()
    assert updated_sdapi["jpeg_quality"] == target_quality

    updated_v2 = live_server.http.get("/api/v2/options").json()
    assert updated_v2["values"]["jpeg_quality"] == target_quality


def test_progress_interrupt_and_skip(tiny_anima_server: Any) -> None:
    progress_resp = tiny_anima_server.http.get("/sdapi/v1/progress")
    assert progress_resp.status_code == 200
    progress_data = progress_resp.json()
    assert {"progress", "eta_relative", "state", "current_image", "textinfo"} <= set(progress_data)
    assert progress_data["progress"] == 0
    assert progress_data["eta_relative"] == 0
    assert progress_data["current_image"] is None
    idle_state = progress_data["state"]
    assert {
        "skipped",
        "interrupted",
        "stopping_generation",
        "job",
        "job_count",
        "job_no",
        "sampling_step",
        "sampling_steps",
    } <= set(idle_state)
    assert idle_state["job_count"] == 0

    skip_resp = tiny_anima_server.http.post("/sdapi/v1/skip")
    assert skip_resp.status_code == 200
    after_skip = tiny_anima_server.http.get("/sdapi/v1/progress")
    assert after_skip.status_code == 200
    assert after_skip.json()["state"]["skipped"] is True

    interrupt_resp = tiny_anima_server.http.post("/sdapi/v1/interrupt")
    assert interrupt_resp.status_code == 200
    assert interrupt_resp.json() == {}
    after_interrupt = tiny_anima_server.http.get("/sdapi/v1/progress")
    assert after_interrupt.status_code == 200
    assert after_interrupt.json()["state"]["interrupted"] is True

    recovery_t2i = tiny_anima_server.http.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": "1girl, idle recovery",
            "steps": 1,
            "width": 64,
            "height": 64,
            "seed": 7,
            "send_images": True,
            "save_images": False,
        },
        timeout=60.0,
    )
    assert recovery_t2i.status_code == 200
    recovery_images = recovery_t2i.json()["images"]
    assert isinstance(recovery_images, list) and len(recovery_images) == 1
    rec_fmt, rec_size = _decode_b64_png(recovery_images[0])
    assert rec_fmt == "PNG"
    assert rec_size == (64, 64)

    final_progress = tiny_anima_server.http.get("/sdapi/v1/progress")
    assert final_progress.status_code == 200
    final_data = final_progress.json()
    assert {"progress", "eta_relative", "state", "current_image", "textinfo"} <= set(final_data)
    assert final_data["progress"] == 0
    assert final_data["state"]["job_count"] == 0
    assert final_data["state"]["job"] == ""
    assert final_data["state"]["interrupted"] is False
    assert final_data["state"]["skipped"] is False


def test_real_inference_txt2img_img2img_extras_and_png_info(tiny_anima_server: Any) -> None:
    t2i_resp = tiny_anima_server.http.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": "1girl, test",
            "steps": 1,
            "width": 64,
            "height": 64,
            "seed": 42,
            "send_images": True,
            "save_images": False,
        },
        timeout=60.0,
    )
    assert t2i_resp.status_code == 200
    t2i_data = t2i_resp.json()
    assert isinstance(t2i_data["images"], list) and len(t2i_data["images"]) == 1
    t2i_b64 = t2i_data["images"][0]
    t2i_fmt, t2i_size = _decode_b64_png(t2i_b64)
    assert t2i_fmt == "PNG"
    assert t2i_size == (64, 64)
    t2i_info = json.loads(t2i_data["info"])
    assert "prompt" in t2i_info

    png_info_resp = tiny_anima_server.http.post("/sdapi/v1/png-info", json={"image": t2i_b64})
    assert png_info_resp.status_code == 200
    png_info = png_info_resp.json()
    assert "Steps: 1" in png_info["info"]
    assert isinstance(png_info["parameters"], dict) and png_info["parameters"]

    i2i_resp = tiny_anima_server.http.post(
        "/sdapi/v1/img2img",
        json={
            "init_images": [t2i_b64],
            "prompt": "1girl, refine",
            "denoising_strength": 0.4,
            "steps": 1,
            "width": 64,
            "height": 64,
            "seed": 42,
            "send_images": True,
            "save_images": False,
        },
        timeout=60.0,
    )
    assert i2i_resp.status_code == 200
    i2i_data = i2i_resp.json()
    assert isinstance(i2i_data["images"], list) and len(i2i_data["images"]) == 1
    i2i_fmt, i2i_size = _decode_b64_png(i2i_data["images"][0])
    assert i2i_fmt == "PNG"
    assert i2i_size == (64, 64)
    i2i_info = json.loads(i2i_data["info"])
    assert "prompt" in i2i_info

    upscalers = [u["name"] for u in tiny_anima_server.http.get("/sdapi/v1/upscalers").json()]
    upscaler_name = "Lanczos" if "Lanczos" in upscalers else "None"
    extras_resp = tiny_anima_server.http.post(
        "/sdapi/v1/extra-single-image",
        json={
            "image": t2i_b64,
            "resize_mode": 0,
            "upscaling_resize": 2,
            "upscaler_1": upscaler_name,
        },
        timeout=60.0,
    )
    assert extras_resp.status_code == 200
    extras_data = extras_resp.json()
    assert isinstance(extras_data["image"], str) and extras_data["image"]
    assert isinstance(extras_data["html_info"], str)
    extras_fmt, extras_size = _decode_b64_png(extras_data["image"])
    assert extras_fmt == "PNG"
    assert extras_size[0] >= 64 and extras_size[1] >= 64


def test_memory_extensions_and_loras(tiny_anima_server: Any) -> None:
    mem_resp = tiny_anima_server.http.get("/sdapi/v1/memory")
    assert mem_resp.status_code == 200
    mem = mem_resp.json()
    assert isinstance(mem["ram"], dict)
    assert isinstance(mem["cuda"], dict)

    ext_resp = tiny_anima_server.http.get("/sdapi/v1/extensions")
    assert ext_resp.status_code == 200
    exts = ext_resp.json()
    assert isinstance(exts, list)
    for ext in exts:
        assert {"name", "remote", "enabled"} <= set(ext)

    loras_resp = tiny_anima_server.http.get("/sdapi/v1/loras")
    assert loras_resp.status_code == 200
    loras = loras_resp.json()
    assert isinstance(loras, list) and len(loras) >= 1
    assert {"name", "alias", "path"} <= set(loras[0])


# ============================================================================
#
# ============================================================================


def test_auth_required_401_and_throttle_429(auth_server: Any, helpers: Any) -> None:
    unauth = auth_server.http.get("/sdapi/v1/options")
    assert unauth.status_code == 401
    assert unauth.headers.get("www-authenticate", "").startswith("Basic")

    bad_headers = helpers.basic_auth_header("alice", "wrong-password")
    for _ in range(10):
        attempt = auth_server.http.get("/sdapi/v1/options", headers=bad_headers)
        assert attempt.status_code == 401

    throttled = auth_server.http.get("/sdapi/v1/options", headers=bad_headers)
    assert throttled.status_code == 429


def test_options_invalid_inputs_return_422(live_server: Any, make_live_server: Callable[..., Any]) -> None:
    unknown = live_server.http.post("/sdapi/v1/options", json={"__no_such_sdapi_option__": 1})
    assert unknown.status_code == 422

    wrong_type = live_server.http.post("/sdapi/v1/options", json={"quicksettings_list": 12345})
    assert wrong_type.status_code == 422

    too_deep = live_server.http.post("/sdapi/v1/options", json={"samples_format": _deep_dict(65)})
    assert too_deep.status_code == 422

    surrogate = live_server.http.post(
        "/sdapi/v1/options",
        content=b'{"samples_format": "\\udc80"}',
        headers={"Content-Type": "application/json"},
    )
    assert surrogate.status_code == 422

    frozen_srv = make_live_server(cmd_opts_overrides={"freeze_settings": True})
    frozen = frozen_srv.http.post("/sdapi/v1/options", json={"jpeg_quality": 77})
    assert frozen.status_code == 422


def test_txt2img_invalid_script_alwayson_and_deep_overrides(tiny_anima_server: Any) -> None:
    bad_script = tiny_anima_server.http.post(
        "/sdapi/v1/txt2img",
        json={"prompt": "test", "script_name": "NoSuchScript", "steps": 1, "width": 64, "height": 64},
    )
    assert bad_script.status_code == 422

    bad_alwayson = tiny_anima_server.http.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": "test",
            "alwayson_scripts": {"NoSuchAlwaysOnScript": {"args": []}},
            "steps": 1,
            "width": 64,
            "height": 64,
        },
    )
    assert bad_alwayson.status_code == 422

    deep_override = tiny_anima_server.http.post(
        "/sdapi/v1/txt2img",
        json={"prompt": "test", "override_settings": {"deep_key": _deep_dict(65)}},
    )
    assert deep_override.status_code == 422


def test_txt2img_unknown_sampler_returns_400(tiny_anima_server: Any) -> None:
    resp = tiny_anima_server.http.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": "test",
            "sampler_name": "NoSuchSampler",
            "steps": 1,
            "width": 64,
            "height": 64,
            "send_images": False,
            "save_images": False,
        },
        timeout=60.0,
    )
    assert resp.status_code == 400
    assert "Sampler not found" in resp.json().get("detail", "")


def test_img2img_null_invalid_base64_and_ssrf_url(tiny_anima_server: Any) -> None:
    null_init = tiny_anima_server.http.post(
        "/sdapi/v1/img2img",
        json={"init_images": None, "prompt": "test"},
    )
    assert null_init.status_code == 404
    assert "Init image not found" in null_init.json().get("detail", "")

    bad_b64 = tiny_anima_server.http.post(
        "/sdapi/v1/img2img",
        json={"init_images": ["!!!not-valid-base64!!!"], "prompt": "test", "steps": 1, "width": 64, "height": 64},
    )
    assert bad_b64.status_code == 500
    assert "Invalid encoded image" in bad_b64.json().get("detail", "")

    ssrf_url = tiny_anima_server.http.post(
        "/sdapi/v1/img2img",
        json={"init_images": ["http://127.0.0.1:1/x.png"], "prompt": "test", "steps": 1, "width": 64, "height": 64},
    )
    assert ssrf_url.status_code == 500
    assert "not allowed" in ssrf_url.json().get("detail", "").lower()


def test_cross_site_post_options_returns_403(live_server: Any) -> None:
    evil_origin = live_server.http.post(
        "/sdapi/v1/options",
        headers={"Origin": "https://evil.com"},
        json={"jpeg_quality": 75},
    )
    assert evil_origin.status_code == 403

    null_origin = live_server.http.post(
        "/sdapi/v1/options",
        headers={"Origin": "null"},
        json={"jpeg_quality": 75},
    )
    assert null_origin.status_code == 403
