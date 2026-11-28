"""Regression tests for tools/remote/modal_validate.py (BUG-06 & BUG-07)."""

from __future__ import annotations

import fnmatch
import importlib.util
import io
import json
import pickle
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from remi.core.tabs import processing_args
from remi.server.models import JobCreated, JobRequest
from remi.testing import webui_state

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("modal_validate_mod", REPO / "tools" / "remote" / "modal_validate.py")
assert SPEC is not None and SPEC.loader is not None
mv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mv)


def _matches_ignore(rel_path: str, patterns: tuple[str, ...]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
        if pat.endswith("/**") and (rel_path == pat[:-3] or rel_path.startswith(pat[:-3] + "/")):
            return True
        if pat.startswith("**/") and fnmatch.fnmatch(rel_path.rsplit("/", 1)[-1], pat[3:]):
            return True
    return False


@pytest.mark.parametrize(
    "sample",
    [
        "weights.pt",
        "nested/weights.pt",
        "weights.pth",
        "nested/weights.pth",
        "model.bin",
        "nested/model.bin",
        "model.onnx",
        "nested/model.onnx",
        "cert.pem",
        "keys/priv.pem",
        "secret.key",
        "keys/secret.key",
        "config.json",
        "extensions/ext1/a.py",
        "state.db",
        "data/files.db-wal",
        "state.sqlite",
        "data/state.sqlite3",
        "webui/test-results/a.json",
    ],
)
def test_ignore_patterns_exclude_sensitive_and_heavy_artifacts(sample: str):
    assert _matches_ignore(sample, mv.IGNORE_PATTERNS) is True


@pytest.mark.parametrize(
    "keep",
    [
        "remi/server/app.py",
        "tools/remote/modal_validate.py",
        "backend/huggingface/Qwen/Qwen-Image/transformer/config.json",
        "webui/tsconfig.json",
    ],
)
def test_ignore_patterns_keep_runtime_and_hf_configs(keep: str):
    assert _matches_ignore(keep, mv.IGNORE_PATTERNS) is False


@pytest.mark.parametrize(
    ("spent", "projected", "soft", "hard", "exc_type"),
    [
        (0.0, 25.0, 20.0, 30.0, RuntimeError),
        (19.0, 2.0, 20.0, 30.0, RuntimeError),
        (20.0, 0.1, 20.0, 30.0, RuntimeError),
        (0.0, 31.0, 30.0, 30.0, RuntimeError),
        (float("nan"), 1.0, 20.0, 30.0, ValueError),
        (0.0, float("nan"), 20.0, 30.0, ValueError),
        (0.0, 1.0, float("nan"), 30.0, ValueError),
        (0.0, 1.0, 20.0, float("inf"), ValueError),
        (-1.0, 1.0, 20.0, 30.0, ValueError),
        (0.0, -5.0, 20.0, 30.0, ValueError),
        (0.0, 1.0, 20.0, 35.0, ValueError),
        (0.0, 1.0, 25.0, 20.0, ValueError),
        (True, 1.0, 20.0, 30.0, ValueError),
    ],
)
def test_check_budget_guard_enforces_finiteness_and_projected_soft_stop(spent, projected, soft, hard, exc_type):
    with pytest.raises(exc_type):
        mv.check_budget_guard(spent, projected, soft_stop_usd=soft, hard_cap_usd=hard)


def test_check_budget_guard_allows_within_budget():
    mv.check_budget_guard(5.0, 10.0, soft_stop_usd=20.0, hard_cap_usd=30.0)
    plan = mv.build_validation_plan(tier="cpu", soft_stop_usd=20.0, hard_cap_usd=30.0)
    assert plan["soft_stop_usd"] == 20.0
    assert plan["hard_cap_usd"] == 30.0


def _setup_anima_env(tmp_path, monkeypatch, *, ckpt_name: str = "anima-turbo-v1.1.safetensors", include_adetailer: bool = False):
    import torch

    cache_dir = tmp_path / "hf_cache"
    diff_dir = cache_dir / "models--x" / "snapshots" / "1" / "diffusion_models"
    diff_dir.mkdir(parents=True)
    (diff_dir / ckpt_name).write_bytes(b"ckpt")
    (diff_dir.parent / "qwen_3_06b_base.safetensors").write_bytes(b"te")
    (diff_dir.parent / "qwen_image_vae.safetensors").write_bytes(b"vae")
    if include_adetailer:
        ad_dir = cache_dir / "models--Bingsu--adetailer" / "snapshots" / "1"
        ad_dir.mkdir(parents=True)
        (ad_dir / "face_yolov8n.pt").write_bytes(b"yolo")

    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))

    # Map /tmp/remi-remote-* paths inside _execute_cuda_subrun to tmp_path so nothing touches /tmp
    real_path = Path

    class _SandboxPath(type(Path())):
        def __new__(cls, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0].startswith("/tmp/remi-remote-"):
                rel = args[0].removeprefix("/tmp/")
                return real_path(tmp_path / rel, *args[1:])
            return real_path(*args, **kwargs)

    monkeypatch.setattr(mv, "Path", _SandboxPath)
    monkeypatch.setattr(mv, "REMOTE_APP_DIR", str(tmp_path / "remote_app"))
    (tmp_path / "remote_app" / "tools").mkdir(parents=True)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _dev: "Mock L4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _dev: SimpleNamespace(total_memory=24 * 1024**3))

    from backend import memory_management
    from backend.nn import anima_quant

    monkeypatch.setattr(memory_management, "supports_fp8_compute", lambda _dev: True)
    monkeypatch.setattr(anima_quant, "scale_fp8_state_dict", lambda sd, **_kw: 1)
    monkeypatch.setattr(anima_quant, "quantize_int8_convrot_state_dict", lambda sd, **_kw: 1)
    return cache_dir


@pytest.mark.parametrize(
    ("ckpt_name", "expected_cfg", "expected_dcfg"),
    [
        ("anima-turbo-v1.1.safetensors", 1.0, 3.5),
        ("anima-base-v1.0.safetensors", 4.0, 3.0),
    ],
)
def test_execute_cuda_subrun_anima_uses_options_and_valid_v2_job_contract(
    tmp_path, monkeypatch, ckpt_name: str, expected_cfg: float, expected_dcfg: float
):
    _setup_anima_env(tmp_path, monkeypatch, ckpt_name=ckpt_name, include_adetailer=True)

    # Build a synthetic high-variance 512x512 PNG for GET /api/v2/files/{id}
    arr = np.indices((512, 512)).sum(axis=0).astype(np.uint8)
    rgb = np.stack([arr, 255 - arr, (arr * 3) % 255], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    waited_ids: list[str] = []
    requests: list[tuple[str, str, object]] = []

    class _FakeServer:
        def __init__(self, *_args, **_kwargs):
            self.stopped = False

        def wait_ready(self, _timeout: float) -> bool:
            return True

        def request(self, method: str, path: str, body=None):
            requests.append((method, path, body))
            if method == "PATCH" and path == "/api/v2/options":
                assert isinstance(body, dict)
                assert body["sd_model_checkpoint"] == ckpt_name
                module_names = {Path(str(m)).name for m in body["forge_additional_modules"]}
                assert module_names == {"qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"}
                return {
                    "version": 1,
                    "categories": [],
                    "quicksettings": [],
                    "values": {
                        "sd_model_checkpoint": ckpt_name,
                        "forge_additional_modules": list(body["forge_additional_modules"]),
                    },
                }
            if method == "POST" and path == "/api/v2/jobs":
                assert isinstance(body, dict)
                req = JobRequest(**body)
                assert "seed" not in req.core
                assert "sd_model_checkpoint" not in req.core
                assert "forge_additional_modules" not in req.core
                assert req.core["cfg_scale"] == expected_cfg
                assert req.core["distilled_cfg_scale"] == expected_dcfg
                assert req.scripts["seed"] == {"seed": 12345}
                with webui_state():
                    core_args, script_args = processing_args(req.tab, req.core, req.scripts)
                assert core_args["cfg_scale"] == expected_cfg
                assert core_args["distilled_cfg_scale"] == expected_dcfg
                assert len(script_args) > 0
                return JobCreated(job_id="j-anima-1", job_ids=["j-anima-1"], seq=7).model_dump()
            if method == "GET" and path == "/api/v2/files/out-1":
                return png_bytes
            raise AssertionError(f"Unexpected request: {method} {path}")

        def stop(self) -> None:
            self.stopped = True

    def _fake_wait_job(_server, job_id: str, timeout: float = 300):
        waited_ids.append(job_id)
        return {"id": job_id, "status": "completed", "outputs": [{"id": "out-1"}]}

    monkeypatch.setitem(sys.modules, "smoke_real", SimpleNamespace(Server=_FakeServer, wait_job=_fake_wait_job))

    result = mv._execute_cuda_subrun("anima")
    assert waited_ids == ["j-anima-1"]
    assert [r[:2] for r in requests] == [
        ("PATCH", "/api/v2/options"),
        ("POST", "/api/v2/jobs"),
        ("GET", "/api/v2/files/out-1"),
    ]
    assert result["ok"] is True
    assert result["checkpoint"] == ckpt_name
    assert result["image_size"] == [512, 512]
    assert (tmp_path / "remi-remote-models" / "adetailer" / "face_yolov8n.pt").is_symlink()


@pytest.mark.parametrize(
    ("job_payload", "expected_status", "expected_err_substr"),
    [
        ({"id": "j-fail-1", "status": "failed", "outputs": [], "error": "CUDA OOM"}, "failed", "CUDA OOM"),
        ({"id": "j-empty-1", "status": "completed", "outputs": [], "error": None}, "completed", "0 outputs"),
    ],
)
def test_execute_cuda_subrun_anima_guards_failed_or_empty_outputs(
    tmp_path, monkeypatch, job_payload: dict, expected_status: str, expected_err_substr: str
):
    _setup_anima_env(tmp_path, monkeypatch)

    class _FakeServer:
        def __init__(self, *_args, **_kwargs):
            pass

        def wait_ready(self, _timeout: float) -> bool:
            return True

        def request(self, method: str, path: str, body=None):
            if method == "PATCH" and path == "/api/v2/options":
                return {"values": body}
            if method == "POST" and path == "/api/v2/jobs":
                return JobCreated(job_id=job_payload["id"], job_ids=[job_payload["id"]], seq=1).model_dump()
            raise AssertionError(f"Unexpected request: {method} {path}")

        def stop(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "smoke_real", SimpleNamespace(Server=_FakeServer, wait_job=lambda _s, _id, timeout=300: job_payload))

    result = mv._execute_cuda_subrun("anima")
    assert result["ok"] is False
    assert result["status"] == expected_status
    assert expected_err_substr in result["error"]


def test_select_cached_anima_ckpt_deterministic_and_respects_anima_file(tmp_path):
    assert mv.ANIMA_DEFAULT_CKPT_CANDIDATES[0] == "split_files/diffusion_models/anima-turbo-v1.1.safetensors"
    assert "split_files/diffusion_models/anima-preview3-base.safetensors" in mv.ANIMA_DEFAULT_CKPT_CANDIDATES

    diff_dir = tmp_path / "models--circlestone-labs--Anima" / "snapshots" / "abc" / "split_files" / "diffusion_models"
    diff_dir.mkdir(parents=True)
    for name in (
        "anima-preview.safetensors",
        "anima-preview3-base.safetensors",
        "anima-base-v1.0.safetensors",
        "anima-turbo-v1.1.safetensors",
    ):
        (diff_dir / name).write_bytes(b"ckpt")

    assert mv._select_cached_anima_ckpt(tmp_path).name == "anima-turbo-v1.1.safetensors"
    assert mv._select_cached_anima_ckpt(tmp_path, "split_files/diffusion_models/anima-base-v1.0.safetensors").name == "anima-base-v1.0.safetensors"
    assert mv._select_cached_anima_ckpt(tmp_path, "anima-preview3-base.safetensors").name == "anima-preview3-base.safetensors"
    assert mv._select_cached_anima_ckpt(tmp_path, "split_files/diffusion_models/missing.safetensors") is None


def test_execute_prefetch_validates_explicit_anima_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    class _FakeApi:
        def list_repo_files(self, *, repo_id: str):
            assert repo_id == mv.ANIMA_HF_REPO
            return [
                "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
                mv.ANIMA_TEXT_ENCODER_FILE,
                mv.ANIMA_VAE_FILE,
            ]

    def _fake_download(*, repo_id: str, filename: str, cache_dir: str):
        p = Path(cache_dir) / Path(filename).name
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=_FakeApi, hf_hub_download=_fake_download))

    with pytest.raises(FileNotFoundError, match="missing-ckpt.safetensors"):
        mv._execute_prefetch(mv.ANIMA_HF_REPO, "split_files/diffusion_models/missing-ckpt.safetensors", False)

    ok_res = mv._execute_prefetch(mv.ANIMA_HF_REPO, "", False)
    assert ok_res["ok"] is True
    assert ok_res["files"][0]["filename"] == "split_files/diffusion_models/anima-turbo-v1.1.safetensors"


def test_fetch_billing_report_workspace_sdk_dict_legacy_and_cli_fallback(monkeypatch):
    # 1. Modern Workspace.from_context().billing.report() dataclass rows
    ts = datetime(2026, 11, 28, 1, 0, tzinfo=UTC)
    fake_ws = SimpleNamespace(
        billing=SimpleNamespace(
            report=lambda **_kw: [
                SimpleNamespace(
                    object_id="ap-1",
                    description="remi-remote-validation",
                    environment_name="main",
                    interval_start=ts,
                    cost=Decimal("0.12500000"),
                    tags={"project": "remi-remote-validation"},
                )
            ]
        )
    )
    monkeypatch.setattr(mv, "modal", SimpleNamespace(Workspace=SimpleNamespace(from_context=lambda: fake_ws)))
    res_ws = mv.fetch_billing_report()
    assert res_ws["ok"] is True
    assert res_ws["source"] == "modal.Workspace.billing.report"
    assert res_ws["total_cost_usd"] == 0.125
    assert res_ws["rows"] == [
        {
            "object_id": "ap-1",
            "description": "remi-remote-validation",
            "environment_name": "main",
            "interval_start": "2026-11-28T01:00:00+00:00",
            "cost": "0.12500000",
            "tags": {"project": "remi-remote-validation"},
        }
    ]

    # 2. Legacy modal.billing.workspace_billing_report() returning list[dict]
    monkeypatch.setattr(
        mv,
        "modal",
        SimpleNamespace(
            billing=SimpleNamespace(
                workspace_billing_report=lambda **_kw: [
                    {
                        "object_id": "ap-2",
                        "description": "remi-remote-validation",
                        "environment_name": "main",
                        "interval_start": ts,
                        "cost": Decimal("0.25000000"),
                        "tags": {"project": "remi-remote-validation"},
                    }
                ]
            )
        ),
    )
    res_dict = mv.fetch_billing_report()
    assert res_dict["ok"] is True
    assert res_dict["source"] == "modal.billing.workspace_billing_report"
    assert res_dict["total_cost_usd"] == 0.25
    assert res_dict["rows"][0]["object_id"] == "ap-2"
    assert res_dict["rows"][0]["cost"] == "0.25000000"

    # 3. DeprecationError / exception in SDK falls back cleanly to `modal billing report --for "this week" --json`
    class _DeprecationError(Exception):
        pass

    def _raise_dep(**_kw):
        raise _DeprecationError("deprecated")

    monkeypatch.setattr(
        mv,
        "modal",
        SimpleNamespace(
            Workspace=SimpleNamespace(from_context=lambda: SimpleNamespace(billing=SimpleNamespace(report=_raise_dep))),
            billing=SimpleNamespace(workspace_billing_report=_raise_dep),
        ),
    )
    monkeypatch.setattr(mv.shutil, "which", lambda name: "/usr/local/bin/modal" if name == "modal" else None)

    recorded_cmds: list[list[str]] = []

    def _fake_run(cmd, **_kw):
        recorded_cmds.append(cmd)
        cli_rows = [
            {
                "object_id": "ap-3",
                "description": "remi-remote-validation",
                "environment": "main",
                "interval_start": "2026-11-28T00:00:00",
                "cost": "0.37500000",
            }
        ]
        return SimpleNamespace(returncode=0, stdout=json.dumps(cli_rows), stderr="")

    monkeypatch.setattr(mv.subprocess, "run", _fake_run)
    res_cli = mv.fetch_billing_report()
    assert recorded_cmds == [["/usr/local/bin/modal", "billing", "report", "--for", "this week", "--json"]]
    assert res_cli["ok"] is True
    assert res_cli["total_cost_usd"] == 0.375
    assert res_cli["rows"][0]["object_id"] == "ap-3"
    assert res_cli["rows"][0]["environment_name"] == "main"

    # 4. Unavailable modal + CLI error returns structured ok=False without crashing
    monkeypatch.setattr(mv, "modal", None)
    monkeypatch.setattr(mv.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout="", stderr="auth missing"))
    res_fail = mv.fetch_billing_report()
    assert res_fail["ok"] is False
    assert res_fail["rows"] == []
    assert res_fail["total_cost_usd"] == 0.0


def test_fetch_billing_report_aligns_end_to_next_utc_hour(monkeypatch):
    captured: dict[str, object] = {}

    def _report(**kw):
        captured.update(kw)
        return []

    fake_ws = SimpleNamespace(billing=SimpleNamespace(report=_report))
    monkeypatch.setattr(mv, "modal", SimpleNamespace(Workspace=SimpleNamespace(from_context=lambda: fake_ws)))
    before = datetime.now(UTC)
    res = mv.fetch_billing_report(days=1)
    assert res["ok"] is True
    start = captured["start"]
    end = captured["end"]
    assert isinstance(start, datetime) and isinstance(end, datetime)
    assert end.minute == 0 and end.second == 0 and end.microsecond == 0
    assert end > before
    assert end - start == timedelta(days=1)


def test_container_entrypoint_import_at_depth_one_path_does_not_raise_index_error(tmp_path, monkeypatch):
    """Simulate Modal mounting tools/remote/modal_validate.py at /root/modal_validate.py (parents len == 2)."""
    copied = tmp_path / "modal_validate.py"
    shutil.copy2(REPO / "tools" / "remote" / "modal_validate.py", copied)

    class _FakeImage:
        @classmethod
        def debian_slim(cls, **_kw):
            return cls()

        def apt_install(self, *_a, **_kw):
            return self

        def uv_pip_install(self, *_a, **_kw):
            return self

        def env(self, *_a, **_kw):
            return self

        def add_local_dir(self, *_a, **_kw):
            return self

    class _FakeApp:
        def __init__(self, *_a, **_kw):
            pass

        def function(self, **_kw):
            return lambda fn: fn

        def local_entrypoint(self):
            return lambda fn: fn

    class _FakeVolume:
        def with_mount_options(self, **_kw):
            return self

        def commit(self):
            pass

    fake_modal = SimpleNamespace(
        App=_FakeApp,
        Image=_FakeImage,
        Volume=SimpleNamespace(from_name=lambda *_a, **_kw: _FakeVolume()),
        FilePatternMatcher=lambda *patterns: patterns,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    real_resolve = Path.resolve

    def _container_resolve(self, *args, **kwargs):
        if self == copied:
            return Path("/root/modal_validate.py")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _container_resolve)

    spec = importlib.util.spec_from_file_location("modal_validate_container_sim", copied)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.REPO_ROOT == Path(mod.REMOTE_APP_DIR) == Path("/app")
    assert callable(mod.remote_cpu_suite)
    assert callable(mod.remote_prefetch)
    assert callable(mod.remote_cuda_subrun)

    # Restore real resolve and point REPO_ROOT/REMOTE_APP_DIR at a temporary /app-like tree
    monkeypatch.setattr(Path, "resolve", real_resolve)
    app_dir = tmp_path / "app"
    for rel_target in mod.CPU_PYTEST_TARGETS:
        target_path = app_dir / rel_target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("# stub\n", encoding="utf-8")

    monkeypatch.setattr(mod, "REPO_ROOT", app_dir)
    monkeypatch.setattr(mod, "REMOTE_APP_DIR", str(app_dir))
    assert mod.resolve_cpu_pytest_targets(mod.REPO_ROOT) == list(mod.CPU_PYTEST_TARGETS)
    plan = mod.build_validation_plan(tier="cpu")
    assert plan["tier"] == "cpu"

    mod._ensure_app_sys_path(str(app_dir))
    assert sys.path[0] == str(app_dir)
    assert sys.path[1] == f"{app_dir}/modules_forge/packages"


def test_check_snapshot_line_endings_skips_ignored_missing_and_catches_lf_or_missing_included(tmp_path):
    app_dir = tmp_path / "app"
    (app_dir / "tools").mkdir(parents=True)
    shutil.copy2(REPO / "tools" / "check_line_endings.py", app_dir / "tools" / "check_line_endings.py")
    (app_dir / "modules").mkdir(parents=True)
    (app_dir / "modules" / "a.py").write_bytes(b"line1\r\nline2\r\n")
    (app_dir / "tools" / "upstream_crlf.txt").write_text(
        "extensions/Put Extensions here.txt\nmodels/VAE/Put VAE here.txt\nmodules/a.py\n",
        encoding="utf-8",
    )

    ok_res = mv._check_snapshot_line_endings(app_dir)
    assert ok_res["returncode"] == 0
    assert ok_res["checked_count"] == 1
    assert ok_res["skipped_ignored"] == ["extensions/Put Extensions here.txt", "models/VAE/Put VAE here.txt"]

    # Bare LF in a snapshot-included file must fail
    (app_dir / "modules" / "a.py").write_bytes(b"line1\nline2\n")
    lf_res = mv._check_snapshot_line_endings(app_dir)
    assert lf_res["returncode"] == 1
    assert "LF line ending" in lf_res["stdout_tail"]

    # Missing non-ignored file must also fail
    (app_dir / "modules" / "a.py").unlink()
    missing_res = mv._check_snapshot_line_endings(app_dir)
    assert missing_res["returncode"] == 1
    assert "listed but missing" in missing_res["stdout_tail"]


def test_execute_cuda_subrun_anima_replaces_stale_broken_symlink(tmp_path, monkeypatch):
    _setup_anima_env(tmp_path, monkeypatch, ckpt_name="anima-turbo-v1.1.safetensors")
    sd_dir = tmp_path / "remi-remote-models" / "Stable-diffusion"
    sd_dir.mkdir(parents=True)
    stale = sd_dir / "anima-turbo-v1.1.safetensors"
    stale.symlink_to(tmp_path / "nonexistent-target.safetensors")
    assert stale.is_symlink() and not stale.exists()

    class _FakeServer:
        def __init__(self, *_args, **_kwargs):
            pass

        def wait_ready(self, _timeout: float) -> bool:
            return False

        def stop(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "smoke_real", SimpleNamespace(Server=_FakeServer, wait_job=lambda *_a, **_kw: {}))
    (tmp_path / "remi-remote-server.log").write_text("boot", encoding="utf-8")

    res = mv._execute_cuda_subrun("anima")
    assert res["ok"] is False  # FakeServer.wait_ready returned False, after symlinks were set up
    assert stale.is_symlink() and stale.exists()


def _assert_strict_builtins(value: object) -> None:
    if value is None:
        return
    if type(value) in (bool, int, float, str, bytes):
        return
    if type(value) is list:
        for item in value:
            _assert_strict_builtins(item)
        return
    if type(value) is dict:
        for k, v in value.items():
            assert type(k) in (bool, int, float, str)
            _assert_strict_builtins(v)
        return
    raise AssertionError(f"Non-builtin type survived sanitization: {type(value)!r} ({value!r})")


def _unpickle_without_torch(payload: bytes) -> dict:
    script = (
        "import importlib.abc, json, pickle, sys\n"
        "class _BlockTorch(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "            raise ModuleNotFoundError(f\"No module named {fullname!r}\")\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockTorch())\n"
        "sys.modules['torch'] = None\n"
        "obj = pickle.loads(sys.stdin.buffer.read())\n"
        "print(json.dumps(obj))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], input=payload, capture_output=True, check=True)
    return json.loads(proc.stdout.decode("utf-8"))


def test_to_builtin_primitives_strips_torch_numpy_and_str_subclasses(tmp_path):
    import torch
    from torch.torch_version import TorchVersion

    class _WeirdStr(str):
        def __str__(self):
            return self

    sample = {
        TorchVersion("2.14.0+cu130"): _WeirdStr("custom"),
        "torch": TorchVersion("2.14.0+cu130"),
        "dtype": torch.bfloat16,
        "path": tmp_path / "weights.safetensors",
        "np_scalar": np.float32(1.25),
        "np_int": np.int64(7),
        "np_arr": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "tensor": torch.tensor([1.0, 2.0]),
        "tuple": (True, 1, 2.5, TorchVersion("1.0")),
        "set": {Path("a"), Path("b")},
    }
    cleaned = mv._to_builtin_primitives(sample)
    _assert_strict_builtins(cleaned)
    assert cleaned["2.14.0+cu130"] == "custom"
    assert cleaned["torch"] == "2.14.0+cu130"
    assert cleaned["dtype"] == "torch.bfloat16"
    assert cleaned["np_scalar"] == 1.25
    assert cleaned["np_int"] == 7
    assert cleaned["np_arr"] == [[1, 2], [3, 4]]
    assert cleaned["tensor"] == [1.0, 2.0]

    unpickled = _unpickle_without_torch(pickle.dumps(cleaned))
    assert unpickled["torch"] == "2.14.0+cu130"


def test_execute_cuda_subrun_families_and_others_unpickle_without_torch(tmp_path, monkeypatch):
    import torch
    from torch.torch_version import TorchVersion

    class _GpuNameStr(str):
        def __str__(self):
            return self

    app_dir = tmp_path / "remote_app"
    (app_dir / "extensions-builtin" / "sd_forge_spectrum" / "tests").mkdir(parents=True)
    real_device = torch.device
    monkeypatch.setattr(mv, "REMOTE_APP_DIR", str(app_dir))
    monkeypatch.setattr(torch, "__version__", TorchVersion("2.14.0+cu130"))
    monkeypatch.setattr(torch, "device", lambda *_a, **_kw: real_device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _dev: _GpuNameStr("NVIDIA L4"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (np.int64(8), np.int64(9)))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _dev: SimpleNamespace(total_memory=np.int64(24 * 1024**3)))

    from backend import memory_management

    monkeypatch.setattr(memory_management, "supports_fp8_compute", lambda _dev: False)
    monkeypatch.setattr(mv, "_run_cmd", lambda cmd, **_kw: {"cmd": cmd, "returncode": 0, "elapsed_s": 0.01, "stdout_tail": "ok", "stderr_tail": ""})

    # families uses real scale_fp8_state_dict (proves non-sensitive probe key quantizes to float8_e4m3fn)
    c2_res = mv._execute_cuda_subrun("families")
    _assert_strict_builtins(c2_res)
    assert c2_res["ok"] is True
    assert c2_res["fp8_quantized"] == 1
    assert type(c2_res["gpu"]["torch"]) is str
    assert c2_res["gpu"]["torch"] == "2.14.0+cu130"
    assert c2_res["gpu"]["name"] == "NVIDIA L4"
    assert c2_res["gpu"]["capability"] == [8, 9]

    c2_unpickled = _unpickle_without_torch(pickle.dumps(c2_res))
    assert c2_unpickled["ok"] is True
    assert c2_unpickled["gpu"]["torch"] == "2.14.0+cu130"

    # extensions also unpickles cleanly without torch
    extensions_res = mv._execute_cuda_subrun("extensions")
    _assert_strict_builtins(extensions_res)
    assert extensions_res["ok"] is True
    assert _unpickle_without_torch(pickle.dumps(extensions_res))["gpu"]["torch"] == "2.14.0+cu130"

    # bench with mocked bench output path
    bench_file = tmp_path / "remi-remote-cuda-bench.json"
    bench_file.write_text(json.dumps({"schema_version": 1, "suites": {"inference": {"ok": True}}}), encoding="utf-8")
    real_path = Path

    class _BenchPath(type(Path())):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "/tmp/remi-remote-cuda-bench.json":
                return real_path(bench_file)
            return real_path(*args, **kwargs)

    monkeypatch.setattr(mv, "Path", _BenchPath)
    d1_res = mv._execute_cuda_subrun("bench")
    _assert_strict_builtins(d1_res)
    assert d1_res["ok"] is True
    assert _unpickle_without_torch(pickle.dumps(d1_res))["results"]["schema_version"] == 1


def test_execute_cuda_subrun_extensions_rejects_empty_targets(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(mv, "REMOTE_APP_DIR", str(tmp_path / "empty_app"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _dev: "Mock L4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _dev: SimpleNamespace(total_memory=24 * 1024**3))

    called = False

    def _fail_if_called(*_a, **_kw):
        nonlocal called
        called = True
        return {"returncode": 0}

    monkeypatch.setattr(mv, "_run_cmd", _fail_if_called)
    res = mv._execute_cuda_subrun("extensions")
    assert res["ok"] is False
    assert called is False
    assert "No extension test directories" in res["error"]

