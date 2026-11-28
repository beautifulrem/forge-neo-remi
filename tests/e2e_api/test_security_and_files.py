"""Opaque-box E2E tests for F1 (App Bootstrap, Options, Auth & Security) and F2 (Uploads, Files, Thumbnails, Outputs & Housekeeping)."""

from __future__ import annotations

import io
import json
import struct
import time
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image


def _make_built_spa_dir(base: Path) -> Path:
    """Create a minimal built SPA + API docs viewer directory inside ``tmp_path``."""

    spa_dir = base / "built-dist"
    (spa_dir / "assets" / "swagger").mkdir(parents=True, exist_ok=True)
    (spa_dir / "assets" / "redoc").mkdir(parents=True, exist_ok=True)
    (spa_dir / "index.html").write_text(
        "<!doctype html><html><head><title>Remi</title></head><body><div id='root'></div></body></html>",
        encoding="utf-8",
    )
    (spa_dir / "assets" / "swagger" / "swagger-ui-bundle.js").write_text("window.SwaggerUIBundle=function(){};", encoding="utf-8")
    (spa_dir / "assets" / "swagger" / "swagger-ui.css").write_text("body{margin:0;}", encoding="utf-8")
    (spa_dir / "assets" / "redoc" / "redoc.standalone.js").write_text("window.Redoc={};", encoding="utf-8")
    (spa_dir / "api-docs.json").write_text(
        json.dumps(
            {
                "swaggerJs": "assets/swagger/swagger-ui-bundle.js",
                "swaggerCss": "assets/swagger/swagger-ui.css",
                "redocJs": "assets/redoc/redoc.standalone.js",
            }
        ),
        encoding="utf-8",
    )
    return spa_dir


def _make_bomb_png_bytes(width: int = 20000, height: int = 20000) -> bytes:
    """Build a minimal PNG header claiming oversized pixel dimensions."""

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_body = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = struct.pack(">I", len(ihdr_body)) + b"IHDR" + ihdr_body + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_body) & 0xFFFFFFFF)
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return sig + ihdr + iend


def _assert_base_headers(headers: Any) -> None:
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("referrer-policy") == "same-origin"
    assert headers.get("x-robots-tag") == "noindex, nofollow"


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


def test_internal_ping_public_and_stable_instance(live_server: Any, auth_server: Any) -> None:
    r1 = live_server.http.get("/internal/ping")
    r2 = live_server.http.get("/internal/ping")
    assert r1.status_code == 200
    assert r2.status_code == 200
    inst1 = r1.json()["instance"]
    assert isinstance(inst1, str) and len(inst1) == 32
    int(inst1, 16)
    assert r2.json()["instance"] == inst1

    # Public even when auth is enabled
    unauth = auth_server.new_client()
    a1 = unauth.get("/internal/ping")
    a2 = unauth.get("/internal/ping")
    assert a1.status_code == 200
    assert a2.status_code == 200
    auth_inst = a1.json()["instance"]
    assert isinstance(auth_inst, str) and len(auth_inst) == 32
    int(auth_inst, 16)
    assert a2.json()["instance"] == auth_inst
    assert auth_inst != inst1


def test_app_bootstrap_and_base_headers(live_server: Any) -> None:
    resp = live_server.http.get("/api/v2/app")
    assert resp.status_code == 200
    _assert_base_headers(resp.headers)

    data = resp.json()
    assert isinstance(data["version"], str) and data["version"]
    for feat in ("jobs.v1", "ws.v1", "uploads.v1", "outputs.v1", "networks.v1", "options.v1", "styles.v1", "tokens.v1", "presets.v1", "sdapi"):
        assert feat in data["features"]
    assert {"txt2img", "img2img", "extras"}.issubset(set(data["tabs"]))
    assert isinstance(data["options"], dict)
    assert isinstance(data["models"], dict)
    assert {"checkpoint", "vae", "text_encoders", "modules", "preset", "unet_dtype"}.issubset(data["models"].keys())
    assert data["queue"] == {"size": 0, "paused": False, "running": None}
    assert data["auth"]["required"] is False
    assert data["auth"]["user"] is None
    assert isinstance(data["auth"]["source_url"], str) and data["auth"]["source_url"]


def test_server_status_openapi_and_docs(make_live_server: Callable[..., Any], tmp_path: Path) -> None:
    spa_dir = _make_built_spa_dir(tmp_path)
    server = make_live_server(spa_dir=spa_dir)

    status_resp = server.http.get("/api/v2/server/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert isinstance(status_data["restartable"], bool)
    assert isinstance(status_data["extension_access"], bool)
    assert status_data["settings_frozen"] is False
    assert isinstance(status_data["version"], str) and status_data["version"]

    frozen_server = make_live_server(cmd_opts_overrides={"freeze_settings": True})
    frozen_status = frozen_server.http.get("/api/v2/server/status")
    assert frozen_status.status_code == 200
    assert frozen_status.json()["settings_frozen"] is True

    openapi_resp = server.http.get("/openapi.json")
    assert openapi_resp.status_code == 200
    spec = openapi_resp.json()
    assert "openapi" in spec
    assert "/api/v2/app" in spec["paths"]
    assert "/api/v2/jobs" in spec["paths"]
    assert "components" in spec

    docs_resp = server.http.get("/docs")
    assert docs_resp.status_code == 200
    assert "text/html" in docs_resp.headers.get("content-type", "")
    assert "Swagger UI" in docs_resp.text
    assert "/openapi.json" in docs_resp.text
    assert "content-security-policy" in docs_resp.headers
    _assert_base_headers(docs_resp.headers)


def test_auth_cookie_lifecycle(auth_server: Any, helpers: Any) -> None:
    client = auth_server.new_client()

    st_before = client.get("/api/v2/auth/status")
    assert st_before.status_code == 200
    assert st_before.json()["required"] is True
    assert st_before.json()["user"] is None

    login_resp = helpers.login(auth_server, "alice", "s3cret", client=client)
    assert login_resp.status_code == 204
    set_cookie = login_resp.headers.get("set-cookie", "")
    assert "remi_session=" in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "samesite=lax" in set_cookie.lower()
    session_token = client.cookies.get("remi_session")
    assert session_token

    st_after = client.get("/api/v2/auth/status")
    assert st_after.status_code == 200
    assert st_after.json()["user"] == "alice"

    opts_resp = client.get("/api/v2/options")
    assert opts_resp.status_code == 200
    assert "values" in opts_resp.json()

    logout_resp = client.post("/api/v2/auth/logout")
    assert logout_resp.status_code == 204

    after_logout = client.get("/api/v2/options")
    assert after_logout.status_code == 401

    # Replaying the revoked session cookie on a new client must still fail
    replay_client = auth_server.new_client(cookies={"remi_session": session_token})
    replay_resp = replay_client.get("/api/v2/options")
    assert replay_resp.status_code == 401


def test_options_patch_and_validate(live_server: Any) -> None:
    get_resp = live_server.http.get("/api/v2/options")
    assert get_resp.status_code == 200
    schema = get_resp.json()
    assert "values" in schema and "defaults" in schema
    orig_val = bool(schema["values"]["show_progressbar"])
    new_val = not orig_val

    val_resp = live_server.http.post("/api/v2/options/validate", json={"show_progressbar": new_val})
    assert val_resp.status_code == 200
    assert val_resp.json() == {"ok": True, "message": None}

    # Validate must not mutate live options or save config.json
    still_orig = live_server.http.get("/api/v2/options").json()["values"]["show_progressbar"]
    assert still_orig is orig_val

    patch_resp = live_server.http.patch("/api/v2/options", json={"show_progressbar": new_val})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["values"]["show_progressbar"] is new_val

    persisted = live_server.http.get("/api/v2/options").json()["values"]["show_progressbar"]
    assert persisted is new_val
    config_file = live_server.root / "config.json"
    assert config_file.is_file()
    saved_cfg = json.loads(config_file.read_text(encoding="utf-8"))
    assert saved_cfg["show_progressbar"] is new_val


def test_basic_auth_accesses_queue_and_docs(auth_server: Any, helpers: Any) -> None:
    unauth = auth_server.new_client()
    assert unauth.get("/api/v2/queue").status_code == 401
    unauth_docs = unauth.get("/docs")
    assert unauth_docs.status_code == 401
    assert "Basic" in unauth_docs.headers.get("www-authenticate", "")

    basic_client = auth_server.new_client(headers=helpers.basic_auth_header("bob", "pw2"))
    queue_resp = basic_client.get("/api/v2/queue")
    assert queue_resp.status_code == 200
    assert set(queue_resp.json().keys()) == {"size", "paused", "running"}

    status_resp = basic_client.get("/api/v2/auth/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["user"] == "bob"

    openapi_resp = basic_client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    assert "/api/v2/queue" in openapi_resp.json()["paths"]

    # Auth passes for /docs (viewer assets are not built on default auth_server so /docs reaches 503 handler, not 401)
    docs_resp = basic_client.get("/docs")
    assert docs_resp.status_code in (200, 503)


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


def test_unauthenticated_api_and_browser_page_load(auth_server: Any) -> None:
    client = auth_server.new_client()

    api_resp = client.get("/api/v2/options")
    assert api_resp.status_code == 401
    assert api_resp.json()["detail"] == "Not authenticated"

    # Without built index.html, browser page load gets 401 with WWW-Authenticate: Basic challenge
    page_resp = client.get("/", headers={"Accept": "text/html"})
    assert page_resp.status_code == 401
    assert "Basic" in page_resp.headers.get("www-authenticate", "")

    # With built index.html in spa_dir, browser page load gets 401 SPA shell HTML + strict CSP
    spa_dir = auth_server.root / "no-dist"
    spa_dir.mkdir(parents=True, exist_ok=True)
    (spa_dir / "index.html").write_text(
        "<!doctype html><html><head><title>Login</title></head><body><div id='root'></div></body></html>",
        encoding="utf-8",
    )
    spa_resp = client.get("/", headers={"Accept": "text/html"})
    assert spa_resp.status_code == 401
    assert "text/html" in spa_resp.headers.get("content-type", "")
    assert "window.__REMI_ROOT__" in spa_resp.text
    assert "content-security-policy" in spa_resp.headers


def test_login_brute_force_throttle_and_forged_cookie(auth_server: Any, helpers: Any) -> None:
    # Forged / malformed cookies must be rejected with 401
    for bad_cookie in ("YWxpY2U.9999999999.nonce.forgedsig", "not-a-valid-token", "a.b.c"):
        forged_client = auth_server.new_client(cookies={"remi_session": bad_cookie})
        assert forged_client.get("/api/v2/options").status_code == 401

    client = auth_server.new_client()
    for _ in range(10):
        bad = helpers.login(auth_server, "alice", "wrong-password", client=client)
        assert bad.status_code == 401

    # 11th attempt must be throttled (429) even with the correct password
    throttled = helpers.login(auth_server, "alice", "s3cret", client=client)
    assert throttled.status_code == 429

    # Basic auth for the same client IP must also be refused while throttled
    basic_client = auth_server.new_client(headers=helpers.basic_auth_header("alice", "s3cret"))
    assert basic_client.get("/api/v2/options").status_code == 429


def test_dns_rebinding_and_cross_site_posts(live_server: Any) -> None:
    rebind = live_server.http.get("/api/v2/app", headers={"Host": "evil.example:7860"})
    assert rebind.status_code == 403
    assert rebind.json()["detail"] == "Unknown host refused"

    for origin in ("https://evil.com", "null"):
        csrf = live_server.http.post(
            "/api/v2/jobs",
            json={"tab": "txt2img", "core": {"prompt": "test"}, "scripts": {}},
            headers={"Origin": origin},
        )
        assert csrf.status_code == 403
        assert csrf.json()["detail"] == "Cross-site request refused"

    sec_fetch = live_server.http.post(
        "/api/v2/jobs",
        json={"tab": "txt2img", "core": {"prompt": "test"}, "scripts": {}},
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    assert sec_fetch.status_code == 403
    assert sec_fetch.json()["detail"] == "Cross-site request refused"


def test_cross_site_subresource_read_refused(live_server: Any) -> None:
    cors_read = live_server.http.get(
        "/api/v2/app",
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors", "Origin": "https://evil.com"},
    )
    assert cors_read.status_code == 403
    assert cors_read.json()["detail"] == "Cross-site request refused"

    img_read = live_server.http.get(
        "/api/v2/app",
        headers={"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "image"},
    )
    assert img_read.status_code == 403

    # Top-level navigation is allowed
    nav = live_server.http.get(
        "/",
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"},
    )
    assert nav.status_code != 403


def test_options_patch_validation_errors_and_freeze_settings(make_live_server: Callable[..., Any], live_server: Any) -> None:
    unknown_resp = live_server.http.patch("/api/v2/options", json={"nonexistent_setting_xyz": 123})
    assert unknown_resp.status_code == 422
    errs = unknown_resp.json()["errors"]
    assert errs and errs[0]["owner"] == "settings"
    assert errs[0]["param"] == "nonexistent_setting_xyz"

    wrong_type = live_server.http.patch("/api/v2/options", json={"live_preview_fps": "not-a-number"})
    assert wrong_type.status_code == 422
    assert wrong_type.json()["errors"][0]["owner"] == "settings"

    deep: Any = "leaf"
    for _ in range(66):
        deep = {"nested": deep}
    deep_resp = live_server.http.patch("/api/v2/options", json={"show_progressbar": deep})
    assert deep_resp.status_code == 422
    assert deep_resp.json()["errors"][0]["owner"] == "settings"

    surrogate_resp = live_server.http.patch(
        "/api/v2/options",
        content=b'{"show_progressbar":"\\udc80"}',
        headers={"Content-Type": "application/json"},
    )
    assert surrogate_resp.status_code == 422
    assert surrogate_resp.json()["errors"][0]["owner"] == "settings"

    frozen_server = make_live_server(cmd_opts_overrides={"freeze_settings": True})
    frozen_resp = frozen_server.http.patch("/api/v2/options", json={"show_progressbar": False})
    assert frozen_resp.status_code == 422
    assert frozen_resp.json()["errors"][0]["owner"] == "settings"


def test_body_limit_inf_nan_and_surrogate_guards(live_server: Any) -> None:
    oversized_login = json.dumps({"username": "a" * (65 * 1024), "password": "pw"}).encode("utf-8")
    limit_resp = live_server.http.post(
        "/api/v2/auth/login",
        content=oversized_login,
        headers={"Content-Type": "application/json"},
    )
    assert limit_resp.status_code == 413
    assert "larger than 64 KB" in limit_resp.json()["detail"]

    inf_resp = live_server.http.post(
        "/api/v2/jobs",
        content=b'{"tab":"txt2img","core":{"cfg_scale":1e999},"scripts":{}}',
        headers={"Content-Type": "application/json"},
    )
    assert inf_resp.status_code == 422

    nan_resp = live_server.http.post(
        "/api/v2/jobs",
        content=b'{"tab":"txt2img","core":{"cfg_scale":NaN},"scripts":{}}',
        headers={"Content-Type": "application/json"},
    )
    assert nan_resp.status_code == 422

    surrogate_resp = live_server.http.post(
        "/api/v2/jobs",
        content=b'{"tab":"txt2img","core":{"prompt":"\\udc80"},"scripts":{}}',
        headers={"Content-Type": "application/json"},
    )
    assert surrogate_resp.status_code == 422
    assert surrogate_resp.json()["errors"][0]["owner"] == "body"


def test_retired_apis_410_and_wildcard_cors_policy(make_live_server: Callable[..., Any], live_server: Any) -> None:
    for path in ("/agent-scheduler/v1/queue", "/tacapi/v1/test", "/physton_prompt/v1/foo", "/prompt-queue/list"):
        resp = live_server.http.get(path)
        assert resp.status_code == 410
        assert "removed" in resp.json()["detail"]

    cors_server = make_live_server(cmd_opts_overrides={"cors_allow_origins": "*"})
    read_resp = cors_server.http.get(
        "/api/v2/app",
        headers={"Origin": "https://evil.com", "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors"},
    )
    assert read_resp.status_code == 200
    assert read_resp.headers.get("access-control-allow-origin") == "*"
    assert read_resp.headers.get("access-control-allow-credentials") != "true"

    write_resp = cors_server.http.post(
        "/api/v2/jobs",
        json={"tab": "txt2img", "core": {"prompt": "cors"}, "scripts": {}},
        headers={"Origin": "https://evil.com"},
    )
    assert write_resp.status_code == 403


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


def test_upload_png_and_retrieve_with_etag_and_dedup(live_server: Any, helpers: Any) -> None:
    raw = helpers.make_png_bytes(size=(48, 24), color=(10, 20, 30))
    up_resp = live_server.http.post(
        "/api/v2/uploads",
        files={"file": ("sample.png", raw, "image/png")},
    )
    assert up_resp.status_code == 201
    meta = up_resp.json()
    assert meta["width"] == 48
    assert meta["height"] == 24
    assert meta["mime"] == "image/png"
    assert meta["size"] == len(raw)
    assert meta["name"] == "sample.png"

    file_resp = live_server.http.get(f"/api/v2/files/{meta['id']}")
    assert file_resp.status_code == 200
    assert file_resp.content == raw
    assert file_resp.headers["content-type"] == "image/png"
    assert "inline" in file_resp.headers.get("content-disposition", "")
    _assert_base_headers(file_resp.headers)
    etag = file_resp.headers["etag"]

    not_modified = live_server.http.get(f"/api/v2/files/{meta['id']}", headers={"If-None-Match": etag})
    assert not_modified.status_code == 304
    assert not not_modified.content

    # Re-uploading same bytes with same name deduplicates to the same id
    dedup_resp = live_server.http.post(
        "/api/v2/uploads",
        files={"file": ("sample.png", raw, "image/png")},
    )
    assert dedup_resp.status_code == 201
    assert dedup_resp.json()["id"] == meta["id"]


def test_thumbnail_webp_snapping_and_etag(live_server: Any, helpers: Any) -> None:
    uploaded = helpers.upload_png(live_server, size=(400, 200), filename="wide.png")
    file_id = uploaded["id"]

    thumb_resp = live_server.http.get(f"/api/v2/thumbs/{file_id}", params={"w": 200})
    assert thumb_resp.status_code == 200
    assert thumb_resp.headers["content-type"] == "image/webp"
    etag = thumb_resp.headers["etag"]

    with Image.open(io.BytesIO(thumb_resp.content)) as thumb:
        assert thumb.format == "WEBP"
        assert thumb.width <= 256
        assert thumb.width == 256
        assert thumb.height == 128

    cached = live_server.http.get(f"/api/v2/thumbs/{file_id}", params={"w": 200}, headers={"If-None-Match": etag})
    assert cached.status_code == 304
    assert not cached.content


def test_upload_safetensors_and_json_download_attachment(live_server: Any, helpers: Any) -> None:
    st_bytes = helpers.make_safetensors_bytes({"modelspec.title": "demo-lora"})
    st_meta = helpers.upload_bytes(live_server, st_bytes, filename="demo.safetensors", content_type="text/plain")
    assert st_meta["mime"] == "application/octet-stream"
    assert st_meta["width"] is None
    assert st_meta["height"] is None
    assert st_meta["size"] == len(st_bytes)

    json_bytes = b'{"steps": 20, "cfg": 4.5}'
    json_meta = helpers.upload_bytes(live_server, json_bytes, filename="preset.json", content_type="application/octet-stream")
    assert json_meta["mime"] == "application/json"
    assert json_meta["width"] is None

    dl_st = live_server.http.get(f"/api/v2/files/{st_meta['id']}", params={"download": "true"})
    assert dl_st.status_code == 200
    assert dl_st.content == st_bytes
    assert dl_st.headers["content-disposition"].startswith("attachment")
    _assert_base_headers(dl_st.headers)

    dl_json = live_server.http.get(f"/api/v2/files/{json_meta['id']}", params={"download": "true"})
    assert dl_json.status_code == 200
    assert dl_json.content == json_bytes
    assert dl_json.headers["content-disposition"].startswith("attachment")


def test_outputs_save_with_zip_and_log_csv(live_server: Any, helpers: Any) -> None:
    job = helpers.submit_txt2img(live_server, prompt="save zip test")
    assert job["status"] == "completed"
    out_id = job["outputs"][0]["id"]

    save_resp = live_server.http.post("/api/v2/outputs/save", json={"ids": [out_id], "zip": True})
    assert save_resp.status_code == 200
    result = save_resp.json()
    assert len(result["saved"]) >= 1
    assert result["zip"]

    saved_dir = live_server.outputs_dir / "saved"
    assert (saved_dir / result["saved"][0]).is_file()
    log_csv = saved_dir / "log.csv"
    assert log_csv.is_file()
    assert "save zip test" in log_csv.read_text(encoding="utf-8")

    zip_resp = live_server.http.get(f"/api/v2/files/{result['zip']}", params={"download": "true"})
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-disposition"].startswith("attachment")
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as archive:
        names = archive.namelist()
        assert result["saved"][0] in names


def test_outputs_refresh_discovers_new_png(live_server: Any, helpers: Any) -> None:
    ext_path = live_server.outputs_dir / "txt2img-images" / "external_discovery_001.png"
    ext_path.parent.mkdir(parents=True, exist_ok=True)
    ext_path.write_bytes(helpers.make_png_bytes(size=(36, 28), text="external prompt\nSteps: 5, Seed: 42"))

    refresh_resp = live_server.http.post("/api/v2/outputs/refresh")
    assert refresh_resp.status_code == 200
    assert refresh_resp.json()["added"] >= 1

    list_resp = live_server.http.get("/api/v2/outputs", params={"q": "external_discovery_001"})
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "external_discovery_001.png"
    assert items[0]["width"] == 36
    assert items[0]["height"] == 28


def test_delete_output_and_housekeeping_startup_scan(make_live_server: Callable[..., Any], helpers: Any, tmp_path: Path) -> None:
    srv_root = tmp_path / "hk-server"
    pre_dir = srv_root / "outputs" / "txt2img-images"
    pre_dir.mkdir(parents=True, exist_ok=True)
    pre_file = pre_dir / "startup_indexed_001.png"
    pre_file.write_bytes(helpers.make_png_bytes(size=(24, 24)))
    del_file_target = pre_dir / "to_unlink_002.png"
    del_file_target.write_bytes(helpers.make_png_bytes(size=(20, 20)))

    server = make_live_server(housekeeping=True, root=srv_root)
    deadline = time.monotonic() + 5.0
    items_by_name: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        listing = server.http.get("/api/v2/outputs").json()["items"]
        items_by_name = {item["name"]: item for item in listing}
        if "startup_indexed_001.png" in items_by_name and "to_unlink_002.png" in items_by_name:
            break
        time.sleep(0.02)
    assert "startup_indexed_001.png" in items_by_name
    assert "to_unlink_002.png" in items_by_name

    keep_id = items_by_name["startup_indexed_001.png"]["id"]
    unlink_id = items_by_name["to_unlink_002.png"]["id"]

    # delete_file=false removes from index, leaves file on disk, and prevents refresh re-adding
    del_keep = server.http.delete(f"/api/v2/outputs/{keep_id}", params={"delete_file": "false"})
    assert del_keep.status_code == 204
    assert pre_file.is_file()
    assert server.http.get(f"/api/v2/files/{keep_id}").status_code == 404
    assert server.http.post("/api/v2/outputs/refresh").json()["added"] == 0

    # delete_file=true deletes the disk file too
    del_unlink = server.http.delete(f"/api/v2/outputs/{unlink_id}", params={"delete_file": "true"})
    assert del_unlink.status_code == 204
    assert not del_file_target.exists()


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


def test_upload_bad_content_type_missing_field_and_empty_file(live_server: Any, helpers: Any) -> None:
    json_resp = live_server.http.post("/api/v2/uploads", json={"file": "not-multipart"})
    assert json_resp.status_code == 422

    raw = helpers.make_png_bytes()
    wrong_field = live_server.http.post("/api/v2/uploads", files={"image": ("wrong.png", raw, "image/png")})
    assert wrong_field.status_code == 422

    empty_resp = live_server.http.post("/api/v2/uploads", files={"file": ("empty.png", b"", "image/png")})
    assert empty_resp.status_code == 400


def test_upload_spoofed_svg_and_corrupt_png(live_server: Any) -> None:
    spoofed = live_server.http.post(
        "/api/v2/uploads",
        files={"file": ("evil.png", b"<html><script>alert(1)</script></html>", "image/png")},
    )
    assert spoofed.status_code == 415

    svg_resp = live_server.http.post(
        "/api/v2/uploads",
        files={"file": ("vector.svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', "image/svg+xml")},
    )
    assert svg_resp.status_code == 415

    corrupt_png = live_server.http.post(
        "/api/v2/uploads",
        files={"file": ("corrupt.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\x04BAD!", "image/png")},
    )
    assert corrupt_png.status_code == 415

    nul_json = live_server.http.post(
        "/api/v2/uploads",
        files={"file": ("nul.json", b'{"a":"\x00"}', "application/json")},
    )
    assert nul_json.status_code == 415


def test_upload_exceeds_max_bytes_and_decompression_bomb(make_live_server: Callable[..., Any], live_server: Any) -> None:
    capped_server = make_live_server(max_upload_bytes=1024)
    # Exceeds 1024-byte store limit while staying below 64KB multipart overhead
    over_store = capped_server.http.post(
        "/api/v2/uploads",
        files={"file": ("big.json", b'{"data":"' + b"x" * 2048 + b'"}', "application/json")},
    )
    assert over_store.status_code == 413

    # Exceeds Content-Length early guard (limit + 64KB)
    over_header = capped_server.http.post(
        "/api/v2/uploads",
        files={"file": ("huge.json", b'{"data":"' + b"x" * (70 * 1024) + b'"}', "application/json")},
    )
    assert over_header.status_code == 413

    # Decompression bomb PNG header (> 178,956,970 pixels)
    bomb_resp = live_server.http.post(
        "/api/v2/uploads",
        files={"file": ("bomb.png", _make_bomb_png_bytes(20000, 20000), "image/png")},
    )
    assert bomb_resp.status_code == 413


def test_path_traversal_short_id_and_delete_upload_404(live_server: Any, helpers: Any) -> None:
    trav = live_server.http.get("/api/v2/files/..%2F..%2Fetc%2Fpasswd")
    assert trav.status_code == 404

    short_id = live_server.http.get("/api/v2/files/abc")
    assert short_id.status_code == 404

    uploaded = helpers.upload_png(live_server, filename="not_output.png")
    del_upload = live_server.http.delete(f"/api/v2/outputs/{uploaded['id']}")
    assert del_upload.status_code == 404

    save_upload = live_server.http.post("/api/v2/outputs/save", json={"ids": [uploaded["id"]], "zip": False})
    assert save_upload.status_code == 404


def test_thumb_non_image_and_open_folder_proxy_refused(make_live_server: Callable[..., Any], live_server: Any, helpers: Any) -> None:
    text_upload = helpers.upload_bytes(live_server, b'{"k":"v"}', filename="notes.json", content_type="application/json")
    thumb_resp = live_server.http.get(f"/api/v2/thumbs/{text_upload['id']}")
    assert thumb_resp.status_code == 415

    proxied_open = live_server.http.post(
        "/api/v2/outputs/open-folder",
        json={"folder": "txt2img"},
        headers={"X-Forwarded-For": "1.2.3.4"},
    )
    assert proxied_open.status_code == 403

    hidden_server = make_live_server(cmd_opts_overrides={"hide_ui_dir_config": True})
    hidden_open = hidden_server.http.post("/api/v2/outputs/open-folder", json={"folder": "txt2img"})
    assert hidden_open.status_code == 403
    assert "folders.open" not in hidden_server.http.get("/api/v2/app").json()["features"]


def test_thumb_and_outputs_query_boundaries(live_server: Any, helpers: Any) -> None:
    uploaded = helpers.upload_png(live_server, filename="bounds.png")
    file_id = uploaded["id"]

    assert live_server.http.get(f"/api/v2/thumbs/{file_id}", params={"w": 0}).status_code == 422
    assert live_server.http.get(f"/api/v2/thumbs/{file_id}", params={"w": 4097}).status_code == 422

    assert live_server.http.get("/api/v2/outputs", params={"limit": 0}).status_code == 422
    assert live_server.http.get("/api/v2/outputs", params={"limit": 201}).status_code == 422

    assert live_server.http.post("/api/v2/outputs/save", json={"ids": [], "zip": False}).status_code == 422
    assert live_server.http.post("/api/v2/outputs/open-folder", json={"folder": "unknown_folder"}).status_code == 422
