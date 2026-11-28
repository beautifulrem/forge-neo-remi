"""Unit and route tests for ``remi.ui`` v1 API versioning, ``remi-extension.json``
manifest validation, extension diagnostics, action timeout/error isolation, and
public ``remi.testing.verify_extension_contract``."""

from __future__ import annotations

import itertools
import json
import time
from types import SimpleNamespace

import pytest

from remi import extensions_admin, ui
from remi.server.testing import server_app
from remi.testing import assert_extension_contract, verify_extension_contract, webui_state


def test_semver_parsing_and_api_compatibility():
    # 1.0 while the API only grows: a manifest's bare "1.0" (the guide's example) means 1.0.x
    assert ui.REMI_API_VERSION == "1.0"
    assert all(ui.check_api_compatibility(declared, ui.REMI_API_VERSION).compatible for declared in ("1.0", "^1.0", "~1.0", ">=1.0,<2.0"))
    assert ui.parse_semver("1.0") == ui.SemVer(1, 0, 0)
    assert ui.parse_semver("v1.2.3-beta.1") == ui.SemVer(1, 2, 3, "beta.1")

    assert ui.check_api_compatibility("1.0", "1.0").compatible is True
    assert ui.check_api_compatibility("^1.0", "1.4.2").compatible is True
    assert ui.check_api_compatibility("^1.0", "2.0.0").compatible is False
    assert ui.check_api_compatibility("~1.0", "1.0.5").compatible is True
    assert ui.check_api_compatibility("~1.0", "1.1.0").compatible is False
    assert ui.check_api_compatibility(">=1.0,<2.0", "1.5").compatible is True
    assert ui.check_api_compatibility(">=1.0,<2.0", "2.0").compatible is False
    assert ui.check_api_compatibility("1.*", "1.9").compatible is True
    assert ui.check_api_compatibility("2.0", "1.0").compatible is False
    assert ui.check_api_compatibility("0.9 || >=1.0", "1.0").compatible is True
    assert ui.check_api_compatibility("not-a-version", "1.0").compatible is False

    # a space after the operator still reads as one comparator
    assert ui.check_api_compatibility(">= 1.0", "1.0").compatible is True
    assert ui.check_api_compatibility("^ 1.0", "1.4").compatible is True
    assert ui.check_api_compatibility("0.9 || >= 1.0", "1.2").compatible is True
    spaced = ui.check_api_compatibility(">= 1.0, < 2.0", "2.0")
    assert spaced.compatible is False and spaced.reason.startswith("requires")


def test_manifest_loading_and_validation(tmp_path):
    # Missing manifest is allowed (backwards compatible)
    missing = ui.load_manifest(tmp_path)
    assert missing.status == "missing" and missing.compatible is True

    # Valid manifest
    (tmp_path / ui.MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "name": "Demo Extension",
                "version": "1.2.0",
                "api_version": "^1.0",
                "min_remi_version": "1.0.0",
                "requires_capabilities": ["anima_hooks"],
                "requires_widgets": ["image-list"],
                "license": "MIT",
                "homepage": "https://example.com/demo",
            }
        ),
        encoding="utf-8",
    )
    valid = ui.load_manifest(tmp_path, known_widgets=["image-list", "stage-timeline"])
    assert valid.status == "valid" and valid.compatible is True
    assert valid.manifest is not None and valid.manifest.requires_capabilities == ("anima_hooks",)

    # Unknown widget generates warning, not error
    warn_res = ui.load_manifest(tmp_path, known_widgets=["stage-timeline"])
    assert warn_res.status == "valid" and warn_res.warnings

    # Unknown capability: a warning as well (the server passes the ones its models expose)
    caps_res = ui.load_manifest(tmp_path, known_widgets=["image-list"], known_capabilities=["spectrum"])
    assert caps_res.status == "valid" and caps_res.warnings == ("unknown required capabilities: anima_hooks",)
    assert ui.load_manifest(tmp_path, known_widgets=["image-list"], known_capabilities=["anima_hooks"]).warnings == ()

    # Incompatible API version
    incompat = ui.validate_manifest_dict({"api_version": ">=2.0", "license": "MIT"})
    assert incompat.status == "incompatible" and incompat.compatible is False

    # Invalid homepage & missing api_version
    invalid = ui.validate_manifest_dict({"homepage": "ftp://bad.example"})
    assert invalid.status == "invalid" and len(invalid.errors) == 2


REMI_TOO_OLD = "requires Remi {{required}} or newer; this is Remi {{running}}"
API_MISMATCH = "requires remi.ui API {{required}}, but runtime provides {{current}}"


def test_an_extension_that_needs_a_newer_remi_is_incompatible(tmp_path):
    """``min_remi_version`` was only checked for its format, so an extension that needs a newer
    Remi loaded as valid. It is compared with the running Remi (``REMI_VERSION``) as versions,
    not strings, and reported as incompatible, with both versions in a sentence the UI translates."""

    from pathlib import Path

    assert ui.REMI_VERSION == "1.0.0"
    # the guide's example and older requirements stay valid
    for minimum in ("1.0.0", "1.0", "1", "v1.0.0", "0.9.12", "1.0.0-rc.1"):
        assert ui.validate_manifest_dict({"api_version": "^1.0", "min_remi_version": minimum}).status == "valid", minimum
    newer = ui.validate_manifest_dict({"api_version": "^1.0", "min_remi_version": "1.1"})
    assert newer.status == "incompatible" and newer.compatible is False
    assert newer.manifest is not None and newer.manifest.min_remi_version == "1.1"
    (error,) = newer.errors
    assert error == "requires Remi 1.1 or newer; this is Remi 1.0.0"
    assert error.template == REMI_TOO_OLD and error.params == {"required": "1.1", "running": "1.0.0"}

    def runs(minimum, running):
        return ui.validate_manifest_dict({"api_version": "^1.0", "min_remi_version": minimum}, remi_version=running).compatible

    # versions, not strings: 1.10 is newer than 1.9, a release than its prereleases, rc.10 than rc.9
    assert runs("1.9.0", "1.10.0") and not runs("1.10.0", "1.9.0")
    assert runs("2.0.0-rc.1", "2.0.0") and not runs("2.0.0", "2.0.0-rc.1")
    assert runs("2.0.0-rc.9", "2.0.0-rc.10") and not runs("2.0.0-rc.10", "2.0.0-rc.9")
    # SemVer's own order: alpha < alpha.1 < alpha.beta < beta < beta.2 < beta.11 < rc.1, numbers below words
    order = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    for older, later in itertools.pairwise(order):
        assert runs(older, later) and not runs(later, older), (older, later)
    assert runs("1.0.0-9", "1.0.0-alpha") and not runs("1.0.0-alpha", "1.0.0-9")
    # build metadata does not count
    assert runs("1.0.0+build.5", "1.0.0")

    # an api_version mismatch is reported beside it, each a sentence
    both = ui.validate_manifest_dict({"api_version": ">=2.0", "min_remi_version": "3.0"})
    assert both.status == "incompatible" and [e.template for e in both.errors] == [API_MISMATCH, REMI_TOO_OLD]
    assert both.errors[0] == "requires remi.ui API >=2.0, but runtime provides 1.0"

    # read from the extension folder, as the diagnostics and the contract check do
    (tmp_path / ui.MANIFEST_FILENAME).write_text(json.dumps({"api_version": "1.0", "min_remi_version": "99.0.0"}), encoding="utf-8")
    assert ui.load_manifest(tmp_path).status == "incompatible"
    assert ui.load_manifest(tmp_path, remi_version="99.1.0").status == "valid"

    catalog = json.loads((Path(__file__).resolve().parents[1] / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    assert catalog.get(REMI_TOO_OLD) and catalog.get(API_MISMATCH)


def test_the_diagnostics_list_why_a_manifest_is_incompatible_with_its_template(tmp_path, monkeypatch):
    """``manifest_errors`` are English; ``manifest_error_details`` gives each with its template and
    values, so Extensions → Diagnostics shows why in the UI language. The extension still loads:
    it is reported, as one with an incompatible ``api_version`` is."""

    ext = tmp_path / "ext_new"
    ext.mkdir()
    (ext / ui.MANIFEST_FILENAME).write_text(json.dumps({"api_version": "1.0", "min_remi_version": "9.0"}), encoding="utf-8")
    item = SimpleNamespace(name="ext_new", enabled=True, is_builtin=False, path=str(ext))
    monkeypatch.setattr(extensions_admin, "all_extensions", lambda: [item])
    ui.diagnostics.reset()
    with webui_state(), server_app(tmp_path) as server:
        (row,) = server.client.get("/api/v2/extensions/diagnostics").json()["items"]
    assert row["manifest_status"] == "incompatible" and row["health"] == "error" and row["enabled"] is True
    assert row["manifest"]["min_remi_version"] == "9.0"
    assert row["manifest_errors"] == ["requires Remi 9.0 or newer; this is Remi 1.0.0"]
    assert row["manifest_error_details"] == [{"message": "requires Remi 9.0 or newer; this is Remi 1.0.0", "template": REMI_TOO_OLD, "params": {"required": "9.0", "running": "1.0.0"}}]


def test_action_timeout_and_error_isolation():
    ui.diagnostics.reset("test_ext")

    class DemoScript:
        @ui.action("ok")
        def ok(self, ctx, values):
            return {"strength": ui.update(value=0.75)}

        @ui.action("boom")
        def boom(self, ctx, values):
            raise RuntimeError("unexpected crash inside extension")

        @ui.action("slow", timeout=0.05)
        def slow(self, ctx, values):
            time.sleep(0.4)
            return {}

        @ui.action("bad_input")
        def bad_input(self, ctx, values):
            raise ui.ValidationError([ui.ParamError("bad param", param="x")])

    script = DemoScript()
    ctx = ui.ActionContext(owner="demo")

    assert ui.call_action_isolated(script, "ok", ctx, {}, extension="test_ext") == {"strength": {"value": 0.75}}

    # ValidationError still propagates cleanly
    with pytest.raises(ui.ValidationError):
        ui.call_action_isolated(script, "bad_input", ctx, {}, extension="test_ext")

    # Runtime exception is isolated into an error toast and logged in diagnostics
    # (a catalog message: the client translates the sentence around the error)
    patch = ui.call_action_isolated(script, "boom", ctx, {}, extension="test_ext")
    assert patch["__toast__"]["level"] == "error"
    assert patch["__toast__"]["message"]["text"] == "Action {{name}} failed: {{error}}"
    assert "unexpected crash" in patch["__toast__"]["message"]["params"]["error"]

    # Slow action times out cleanly (both toast mode and raise mode)
    timeout_patch = ui.call_action_isolated(script, "slow", ctx, {}, extension="test_ext")
    assert timeout_patch["__toast__"]["level"] == "error"
    assert timeout_patch["__toast__"]["message"] == {"text": "Action {{name}} timed out after {{seconds}} s", "params": {"name": "slow", "seconds": timeout_patch["__toast__"]["message"]["params"]["seconds"]}}

    with pytest.raises(ui.ActionTimeoutError):
        ui.call_action_isolated(script, "slow", ctx, {}, extension="test_ext", return_toast=False)

    rec = ui.diagnostics.get("test_ext")
    assert rec is not None and rec.action_errors == 1 and rec.action_timeouts == 2
    ui.diagnostics.reset("test_ext")


def test_extension_diagnostics_report_and_route(tmp_path, monkeypatch):
    ext_ok = tmp_path / "ext_ok"
    ext_bad = tmp_path / "ext_bad"
    ext_ok.mkdir()
    ext_bad.mkdir()
    (ext_ok / ui.MANIFEST_FILENAME).write_text(json.dumps({"api_version": "1.0", "license": "Apache-2.0"}), encoding="utf-8")
    (ext_bad / ui.MANIFEST_FILENAME).write_text(json.dumps({"api_version": ">=3.0"}), encoding="utf-8")

    ui.diagnostics.reset()
    ui.diagnostics.record_load_time("ext_ok", 14.5)
    ui.diagnostics.record_hook_warning("ext_ok", "minor notice")

    fake_exts = [
        SimpleNamespace(name="ext_ok", enabled=True, is_builtin=True, path=str(ext_ok)),
        SimpleNamespace(name="ext_bad", enabled=True, is_builtin=False, path=str(ext_bad)),
        SimpleNamespace(name="ext_off", enabled=False, is_builtin=False, path=str(tmp_path / "missing")),
    ]
    report = extensions_admin.diagnostics_report(extensions_list=fake_exts)
    assert report["api_version"] == ui.REMI_API_VERSION
    assert report["summary"]["total"] == 3
    assert report["summary"]["degraded"] == 1  # ext_ok has 1 hook_warning
    assert report["summary"]["error"] == 1  # ext_bad has incompatible manifest
    assert report["summary"]["disabled"] == 1
    assert report["summary"]["with_manifest"] == 2

    ext_typo = tmp_path / "ext_typo"
    ext_typo.mkdir()
    (ext_typo / ui.MANIFEST_FILENAME).write_text(json.dumps({"api_version": "1.0", "requires_capabilities": ["anima_hooks", "anima_hook"]}), encoding="utf-8")
    typo = SimpleNamespace(name="ext_typo", enabled=True, is_builtin=False, path=str(ext_typo))
    monkeypatch.setattr(extensions_admin, "all_extensions", lambda: [*fake_exts, typo])
    with webui_state(), server_app(tmp_path) as server:
        resp = server.client.get("/api/v2/extensions/diagnostics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["api_version"] == "1.0"
        by_name = {item["name"]: item for item in body["items"]}
        assert by_name["ext_ok"]["load_time_ms"] == 14.5
        assert by_name["ext_bad"]["manifest_status"] == "incompatible"
        assert by_name["ext_off"]["health"] == "disabled"
        # the route checks required capabilities against the ones models expose
        assert by_name["ext_typo"]["manifest_warnings"] == ["unknown required capabilities: anima_hook"]
        assert by_name["ext_typo"]["health"] == "degraded"
    ui.diagnostics.reset()


def test_diagnostics_parses_startup_timer_exposes_recent_errors_and_validates_widgets(tmp_path, monkeypatch):
    from modules import timer

    ext_demo = tmp_path / "ext_demo"
    (ext_demo / "scripts").mkdir(parents=True)
    (ext_demo / "scripts" / "demo_script.py").write_text("# demo\n", encoding="utf-8")
    (ext_demo / ui.MANIFEST_FILENAME).write_text(
        json.dumps({"api_version": "1.0", "requires_widgets": ["image-list", "nonexistent-widget"]}),
        encoding="utf-8",
    )
    ext_empty = SimpleNamespace(name="ext_empty", enabled=True, is_builtin=False, path="")
    demo_item = SimpleNamespace(name="ext_demo", enabled=True, is_builtin=False, path=str(ext_demo))

    monkeypatch.setattr(
        timer.startup_timer,
        "records",
        {"load scripts/demo_script.py": 0.1234, "load scripts": 0.5, "load scripts/core_only.py": 0.05},
    )
    monkeypatch.setattr(extensions_admin, "all_extensions", lambda: [demo_item, ext_empty])
    monkeypatch.chdir(tmp_path)
    (tmp_path / ui.MANIFEST_FILENAME).write_text(json.dumps({"api_version": "invalid"}), encoding="utf-8")

    ui.diagnostics.reset()
    ui.diagnostics.record_hook_warning("ext_demo", "warn 1")
    ui.diagnostics.record_action_error("ext_demo", "act", "boom", timed_out=False)

    with webui_state(), server_app(tmp_path) as server:
        resp = server.client.get("/api/v2/extensions/diagnostics")
        assert resp.status_code == 200
        by_name = {item["name"]: item for item in resp.json()["items"]}
        demo = by_name["ext_demo"]
        assert demo["load_time_ms"] == 123.4
        assert ui.diagnostics.get("ext_demo").load_time_ms == 123.4
        assert len(demo["recent_errors"]) == 2
        assert [entry["kind"] for entry in demo["recent_errors"]] == ["hook_warning", "action_error"]
        assert all(isinstance(entry["timestamp"], str) and entry["timestamp"] for entry in demo["recent_errors"])
        assert demo["manifest_warnings"] == ["unknown required widgets: nonexistent-widget"]
        # Empty path does not fall back to loading ./remi-extension.json
        assert by_name["ext_empty"]["manifest_status"] == "missing"
    ui.diagnostics.reset()


def test_public_verify_extension_contract(tmp_path):
    ext_dir = tmp_path / "clean_ext"
    (ext_dir / "scripts").mkdir(parents=True)
    (ext_dir / "localizations").mkdir()
    (ext_dir / ui.MANIFEST_FILENAME).write_text(json.dumps({"api_version": "^1.0", "name": "Clean Ext", "license": "MIT"}), encoding="utf-8")
    (ext_dir / "scripts" / "demo.py").write_text("from remi import ui\n", encoding="utf-8")
    (ext_dir / "localizations" / "zh_CN.json").write_text(json.dumps({"Demo Panel": "演示面板", "Enable": "启用", "Strength": "强度"}), encoding="utf-8")

    class CleanScript:
        def show(self, is_img2img):
            return True

        def params(self, is_img2img):
            return ui.Section("Demo Panel", enable=ui.Toggle("enable", False, label="Enable"), children=[ui.Slider("strength", 0.0, 1.0, 0.5, 0.05, label="Strength")])

    report = assert_extension_contract(ext_dir, script_classes=[CleanScript], require_manifest=True, check_locales=["zh_CN"])
    assert report.ok is True and report.manifest_status == "valid"
    assert report.param_counts["CleanScript:txt2img"] == 2

    # Violating extension: imports gradio and remi.server
    bad_dir = tmp_path / "bad_ext"
    bad_dir.mkdir()
    (bad_dir / "bad.py").write_text("import gradio as gr\nfrom remi.server import app\n", encoding="utf-8")
    bad_report = verify_extension_contract(bad_dir, require_manifest=True)
    assert bad_report.ok is False
    assert any("gradio" in err.lower() for err in bad_report.errors)
    assert any("remi.server" in err for err in bad_report.errors)
    assert any("remi-extension.json" in err for err in bad_report.errors)


def test_prompt_tools_upsample_route_wiring(tmp_path, monkeypatch):
    import io
    import sys
    import types

    from PIL import Image

    captured: dict[str, object] = {}

    class FakeUpsampleResult:
        def to_dict(self):
            return {
                "prompt": "upsampled cinematic prompt",
                "original_prompt": str(captured.get("prompt", "")),
                "model_family": str(captured.get("model_family") or "flux3"),
                "mode": str(captured.get("mode") or "i2i"),
                "backend": "openai_vlm",
                "used_reference_count": len(captured.get("reference_images") or []),
                "aspect_ratio": captured.get("aspect_ratio"),
                "layout_rows": None,
                "vgl_json": None,
                "warnings": [],
            }

    def fake_upsample_prompt(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return FakeUpsampleResult()

    fake_mod = types.ModuleType("remi.prompt_tools.pup")
    fake_mod.upsample_prompt = fake_upsample_prompt  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "remi.prompt_tools.pup", fake_mod)
    from remi import prompt_tools as pt

    monkeypatch.setattr(pt.registry, "group_dirs", ["group_tags"])  # the prompt tools are on

    with webui_state(), server_app(tmp_path) as server:
        from remi import prompt_tools

        try:
            buf = io.BytesIO()
            Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="PNG")
            upload = server.client.post("/api/v2/uploads", files={"file": ("ref.png", buf.getvalue(), "image/png")}).json()

            resp = server.client.post(
                "/api/v2/prompt-tools/upsample",
                json={
                    "prompt": "a neon street",
                    "model_family": "flux3",
                    "reference_file_ids": [upload["id"]],
                    "aspect_ratio": "16:9",
                    "mode": "i2i",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["prompt"] == "upsampled cinematic prompt"
            assert data["used_reference_count"] == 1
            assert data["aspect_ratio"] == "16:9"
            assert len(captured["reference_images"]) == 1  # type: ignore[arg-type]

            missing = server.client.post(
                "/api/v2/prompt-tools/upsample",
                json={"prompt": "a neon street", "reference_file_ids": ["nonexistent1234"]},
            )
            assert missing.status_code == 404

            bad_path = server.client.post(
                "/api/v2/prompt-tools/upsample",
                json={"prompt": "a neon street", "reference_images": ["/etc/passwd"]},
            )
            assert bad_path.status_code == 422
        finally:
            prompt_tools.forget(server.services)


def test_semver_manifest_and_actions_edge_cases(tmp_path):
    from remi.ui import actions as ui_actions
    from remi.ui.errors import SchemaError
    from remi.ui.manifest import MAX_RECENT_ERRORS

    # 1. SemVer / version.py complete operator & prerelease coverage
    assert str(ui.SemVer(1, 2, 3)) == "1.2.3"
    assert str(ui.SemVer(1, 2, 3, "rc.1")) == "1.2.3-rc.1"
    with pytest.raises(ValueError):
        ui.parse_semver("")

    assert ui.check_api_compatibility(None, "1.0").compatible is True
    assert ui.check_api_compatibility("||", "1.0").compatible is True
    assert ui.check_api_compatibility("*", "1.0").compatible is True
    assert ui.check_api_compatibility("1.0.x", "1.0.4").compatible is True
    assert ui.check_api_compatibility("2.*", "1.0").compatible is False
    assert ui.check_api_compatibility("bad.*", "1.0").compatible is False
    assert ui.check_api_compatibility("1", "1.5.0").compatible is True
    assert ui.check_api_compatibility("==1.0.0", "1.0.0").compatible is True
    assert ui.check_api_compatibility("!=1.0.0", "1.0.1").compatible is True
    assert ui.check_api_compatibility("<=1.0.0", "1.0.0").compatible is True
    assert ui.check_api_compatibility(">0.9.0", "1.0.0").compatible is True
    assert ui.check_api_compatibility("^0.2.0", "0.2.5").compatible is True
    assert ui.check_api_compatibility("^0.0.2", "0.0.2").compatible is True
    assert ui.check_api_compatibility("^1.5.0", "1.2.0").compatible is False
    assert ui.check_api_compatibility("~1.5.0", "1.2.0").compatible is False
    # Prerelease comparison branches
    assert ui.check_api_compatibility(">1.0.0-alpha", "1.0.0").compatible is True
    assert ui.check_api_compatibility("<1.0.0", "1.0.0-alpha").compatible is True
    assert ui.check_api_compatibility(">1.0.0-alpha", "1.0.0-beta").compatible is True
    assert ui.check_api_compatibility("1.0", "not-semver").compatible is False
    assert ui.check_api_compatibility("1.0", "1.0").to_dict()["compatible"] is True

    # 2. Manifest & DiagnosticsCollector edge cases
    assert ui.validate_manifest_dict(["not", "a", "dict"]).status == "invalid"
    res_bad = ui.validate_manifest_dict(
        {
            "api_version": 123,
            "name": 42,
            "version": "not-semver",
            "min_remi_version": "bad-ver",
            "requires_capabilities": ["ok", ""],
        }
    )
    assert res_bad.status == "invalid" and res_bad.to_dict()["compatible"] is False and res_bad.warnings

    (tmp_path / ui.MANIFEST_FILENAME).write_text("{broken json", encoding="utf-8")
    assert ui.load_manifest(tmp_path).status == "invalid"

    ui.diagnostics.reset()
    assert ui.diagnostics.get("absent") is None
    for i in range(MAX_RECENT_ERRORS + 3):
        ui.diagnostics.record_hook_error("noisy_ext", f"err #{i}")
    snap = ui.diagnostics.snapshot()
    assert snap["noisy_ext"].hook_errors == MAX_RECENT_ERRORS + 3
    assert len(snap["noisy_ext"].recent_errors) == MAX_RECENT_ERRORS
    ui.diagnostics.reset()

    # 3. remi/ui/actions.py full coverage
    class ActionsOwner:
        @ui.action
        def bare(self, ctx, values):
            return {"x": ui.update(value=values.get("x", 1), choices=["a", ("B", "b"), {"value": "c", "label": "C"}])}

        @ui.action("timed_err", timeout=1.0)
        def timed_err(self, ctx, values):
            if values.get("val_err"):
                raise ui.ValidationError([ui.ParamError("bad", param="k")])
            raise RuntimeError("boom in thread")

        @ui.action("timed_ok", timeout=1.0)
        def timed_ok(self, ctx, values):
            return {"ok": ui.update(value=True)}

    class DupActions:
        @ui.action("same")
        def first(self, ctx, values):
            return {}

        @ui.action("same")
        def second(self, ctx, values):
            return {}

    with pytest.raises(SchemaError):
        ui.actions_of(DupActions())

    owner = ActionsOwner()
    ctx = ui.ActionContext(owner="owner")
    assert ui.call_action(owner, "bare", ctx, None)["x"]["value"] == 1
    with pytest.raises(KeyError):
        ui.call_action(owner, "missing", ctx, {})
    with pytest.raises(ui.ValidationError):
        ui.call_action(owner, "bare", ctx, "not-a-dict")  # type: ignore[arg-type]

    # call_action_isolated error & timed branches
    with pytest.raises(KeyError):
        ui.call_action_isolated(owner, "missing", ctx, {})
    with pytest.raises(ui.ValidationError):
        ui.call_action_isolated(owner, "bare", ctx, "bad")  # type: ignore[arg-type]
    assert ui.call_action_isolated(owner, "bare", ctx, None)["x"]["value"] == 1
    assert ui.call_action_isolated(owner, "timed_ok", ctx, {})["ok"]["value"] is True
    with pytest.raises(ui.ValidationError):
        ui.call_action_isolated(owner, "timed_err", ctx, {"val_err": True})
    assert ui.call_action_isolated(owner, "timed_err", ctx, {})["__toast__"]["level"] == "error"
    with pytest.raises(RuntimeError):
        ui.call_action_isolated(owner, "timed_err", ctx, {}, return_toast=False)
    with pytest.raises(RuntimeError):
        ui.call_action_isolated(owner, "timed_err", ctx, {}, timeout=0, return_toast=False)
    ui.diagnostics.reset()

    # update, message, toast, message_of, message_english, check_patch, merge_patches
    with pytest.raises(SchemaError):
        ui.update(unknown_prop=1)
    with pytest.raises(SchemaError):
        ui.message(123)  # type: ignore[arg-type]
    with pytest.raises(SchemaError):
        ui.toast("hi", level="fatal")
    msg_dict = ui.message("Hello {{who}}", who="Remi")
    assert ui.toast(msg_dict)["__toast__"]["message"] == msg_dict
    assert ui_actions.message_english(msg_dict) == "Hello Remi"
    assert ui_actions.message_english("plain") == "plain"
    with pytest.raises(SchemaError):
        ui_actions.message_of({"text": 123})

    checked = ui_actions.check_patch(
        {
            "__advice__": [{"code": "a1", "severity": "info", "message": "tip"}],
            "__capabilities__": {"preset": "anima"},
            "__refresh__": {"keys": ["models"]},
            "__toast__": {"level": "info", "message": "ok"},
            "p": {"value": 2},
        }
    )
    assert checked["__refresh__"] == {"keys": ["models"]}
    for bad_patch in (
        "not-dict",
        {"__advice__": "bad"},
        {"__toast__": "bad"},
        {"__toast__": {"level": "nope", "message": "x"}},
        {"__refresh__": {"keys": [123]}},
        {"__unknown__": {}},
        {"p": "not-dict"},
    ):
        with pytest.raises(SchemaError):
            ui_actions.check_patch(bad_patch)  # type: ignore[arg-type]

    merged = ui_actions.merge_patches(
        {"__toast__": {"level": "info", "message": "one"}, "__advice__": [{"code": "c1", "message": "m1"}], "x": {"value": 1}},
        {"__toast__": {"level": "warning", "message": "two"}, "__advice__": [{"code": "c1", "message": "dup"}, {"code": "c2", "message": "m2"}], "x": {"enabled": False}},
    )
    assert merged["__toast__"] == {"level": "warning", "message": "one; two", "parts": ["one", "two"]}
    assert [a["code"] for a in merged["__advice__"]] == ["c1", "c2"]
    assert merged["x"] == {"value": 1, "enabled": False}



def test_the_extension_an_owner_comes_from(tmp_path, monkeypatch):
    """Scripts and Extras operations by their ``filename``, a module or a class by its
    source file; outside the extension folders: none."""
    import types

    from modules import paths_internal

    builtin = tmp_path / "extensions-builtin"
    monkeypatch.setattr(paths_internal, "extensions_builtin_dir", str(builtin))
    monkeypatch.setattr(paths_internal, "extensions_dir", str(tmp_path / "extensions"))
    script = types.SimpleNamespace(filename=str(builtin / "ext-a" / "scripts" / "a.py"))
    assert ui.extension_of_owner(script) == "ext-a"
    operation = types.SimpleNamespace(filename=str(tmp_path / "extensions" / "ext-b" / "scripts" / "postprocessing_b.py"))
    assert ui.extension_of_owner(operation) == "ext-b"
    module = types.ModuleType("ext_c")
    module.__file__ = str(builtin / "ext-c" / "scripts" / "c.py")
    assert ui.extension_of_owner(module) == "ext-c"
    assert ui.extension_of_owner(types.SimpleNamespace(filename=str(tmp_path / "scripts" / "loopback.py"))) is None
    assert ui.extension_of_owner(object()) is None  # a builtin class: no source file
    # an extension folder linked in from elsewhere (a developer's checkout) is still the extension
    checkout = tmp_path / "dev" / "ext-d"
    (checkout / "scripts").mkdir(parents=True)
    (tmp_path / "extensions").mkdir()
    (tmp_path / "extensions" / "ext-d").symlink_to(checkout, target_is_directory=True)
    assert ui.extension_of_owner(types.SimpleNamespace(filename=str(tmp_path / "extensions" / "ext-d" / "scripts" / "d.py"))) == "ext-d"


def test_an_extension_path_spelled_in_another_case_is_found(tmp_path, monkeypatch):
    """A volume that ignores case (macOS, Windows) keeps the spelling it is given: a script
    loaded as ``Extensions/...`` is still inside ``extensions/``."""
    from modules import paths_internal
    from remi.ui.manifest import extension_of

    (tmp_path / "extensions" / "ext-d" / "scripts").mkdir(parents=True)
    if not (tmp_path / "EXTENSIONS").exists():
        pytest.skip("this volume tells letter case apart")
    monkeypatch.setattr(paths_internal, "extensions_builtin_dir", str(tmp_path / "extensions-builtin"))
    monkeypatch.setattr(paths_internal, "extensions_dir", str(tmp_path / "extensions"))
    assert extension_of(str(tmp_path / "EXTENSIONS" / "ext-d" / "scripts" / "d.py")) == "ext-d"
    assert extension_of(str(tmp_path / "Extensions-builtin2" / "x.py")) is None
