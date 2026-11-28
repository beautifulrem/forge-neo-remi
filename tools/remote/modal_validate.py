"""Ephemeral Modal SDK 1.6.4 cross-platform validation harness (Linux CPU & CUDA).

Strictly non-runtime: lives under ``tools/remote/`` with zero imports from
``modal`` inside ``remi/`` or ``backend/``. Runs via ``modal run`` (never
``modal deploy``) using ``copy=False`` workspace snapshots, an isolated v2 HF
cache Volume (``remi-hf-cache``), and strict wall-clock + billing spend
guards (``$20`` soft stop, ``<= $30`` hard cap).

Examples::

    # Local dry-run / plan inspection (free, works without Modal credentials)
    .venv/bin/python tools/remote/modal_validate.py --tier cpu --dry-run
    .venv/bin/python tools/remote/modal_validate.py --tier cuda --subrun all --dry-run

    # Ephemeral Modal runs (incur Modal compute charges)
    modal run tools/remote/modal_validate.py --tier cpu
    modal run tools/remote/modal_validate.py --tier prefetch
    modal run tools/remote/modal_validate.py --tier cuda --subrun anima --gpu L4
    modal run tools/remote/modal_validate.py --tier cuda --subrun bench --gpu L4
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

try:
    modal: Any = importlib.import_module("modal")
except ImportError:
    modal = None

REMOTE_APP_DIR = "/app"
_MODULE_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _MODULE_PARENTS[2] if len(_MODULE_PARENTS) > 2 else Path(REMOTE_APP_DIR)
APP_NAME = "remi-remote-validation"
APP_TAGS = {"project": "remi-remote-validation"}
HF_CACHE_VOLUME_NAME = "remi-hf-cache"
HF_CACHE_MOUNT_PATH = "/cache/hf"

DEFAULT_TIMEOUT_S = 600
DEFAULT_SCALEDOWN_WINDOW_S = 30
DEFAULT_RETRIES = 0
DEFAULT_GPU = "L4"
SUPPORTED_GPUS = ("T4", "L4", "A10G", "L40S")

# Verified Modal per-second rates (USD/sec, global pool region=None).
GPU_RATES_PER_SEC: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10G": 0.000306,
    "A10": 0.000306,
    "L40S": 0.000542,
}
CPU_RATE_PER_CORE_SEC = 0.0000131
RAM_RATE_PER_GIB_SEC = 0.00000222

DEFAULT_SOFT_STOP_USD = 20.0
HARD_CAP_USD = 30.0

ANIMA_HF_REPO = "circlestone-labs/Anima"
ANIMA_TEXT_ENCODER_FILE = "split_files/text_encoders/qwen_3_06b_base.safetensors"
ANIMA_VAE_FILE = "split_files/vae/qwen_image_vae.safetensors"
ANIMA_DEFAULT_CKPT_CANDIDATES = (
    "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
    "split_files/diffusion_models/anima-turbo-v1.0.safetensors",
    "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "split_files/diffusion_models/anima-preview3-base.safetensors",
    "split_files/diffusion_models/anima-preview2.safetensors",
    "split_files/diffusion_models/anima-preview.safetensors",
)
ADETAILER_HF_REPO = "Bingsu/adetailer"
ADETAILER_DETECTOR_FILE = "face_yolov8n.pt"

# Exact workspace snapshot exclusions for Image.add_local_dir(..., copy=False).
IGNORE_PATTERNS: tuple[str, ...] = (
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    "venv",
    "venv/**",
    "webui/node_modules",
    "webui/node_modules/**",
    "webui/dist",
    "webui/dist/**",
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "models",
    "models/**",
    "outputs",
    "outputs/**",
    "output",
    "output/**",
    "remi-data",
    "remi-data/**",
    "logs",
    "logs/**",
    "*.safetensors",
    "**/*.safetensors",
    "*.ckpt",
    "**/*.ckpt",
    "*.gguf",
    "**/*.gguf",
    "*.pt",
    "**/*.pt",
    "*.pth",
    "**/*.pth",
    "*.bin",
    "**/*.bin",
    "*.onnx",
    "**/*.onnx",
    "*.pem",
    "**/*.pem",
    "*.key",
    "**/*.key",
    "config.json",
    "extensions",
    "extensions/**",
    "*.db",
    "**/*.db",
    "*.db*",
    "**/*.db*",
    "*.sqlite",
    "**/*.sqlite",
    "*.sqlite*",
    "**/*.sqlite*",
    "webui/test-results",
    "webui/test-results/**",
    ".agents",
    ".agents/**",
    "**/__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    "tmp",
    "tmp/**",
    "cache",
    "cache/**",
    "extension-data",
    "extension-data/**",
)

CPU_PYTEST_TARGETS: tuple[str, ...] = (
    "tests/test_remi_server_app.py",
    "tests/test_remi_server_jobs.py",
    "tests/test_remi_server_files.py",
    "tests/test_remi_chains.py",
    "tests/test_remi_chain_lifecycle.py",
    "tests/test_remi_worker_protocol.py",
    "tests/test_remi_server_ws.py",
    "tests/test_remi_security.py",
    "tests/test_remi_server_auth.py",
    "tests/test_remi_sdapi_contract.py",
    "tests/test_remi_upstream_forks.py",
    "tests/test_scripts_ported.py",
    "tests/test_remi_ml_runtime_fixes.py",
)


def resolve_cpu_pytest_targets(app_dir: Path, pytest_targets: list[str] | tuple[str, ...] | None = None) -> list[str]:
    targets = list(pytest_targets) if pytest_targets is not None else list(CPU_PYTEST_TARGETS)
    missing = [t for t in targets if not (app_dir / t).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing configured CPU pytest target(s) under {app_dir}: {missing}")
    return targets

CUDA_SUBRUNS: tuple[str, ...] = ("anima", "extensions", "families", "bench")
TIERS: tuple[str, ...] = ("cpu", "prefetch", "cuda", "billing")


def estimate_cost_usd(
    elapsed_s: float,
    *,
    gpu: str | None = None,
    cpu_cores: float = 4.0,
    memory_gib: float = 16.0,
) -> float:
    """Estimate Modal container cost in USD from wall-clock seconds and resource shape."""
    seconds = max(0.0, float(elapsed_s))
    gpu_rate = GPU_RATES_PER_SEC.get(gpu.upper(), 0.0) if gpu else 0.0
    per_sec = gpu_rate + max(0.125, cpu_cores) * CPU_RATE_PER_CORE_SEC + max(0.125, memory_gib) * RAM_RATE_PER_GIB_SEC
    return round(seconds * per_sec, 6)


def check_budget_guard(
    spent_usd: float,
    projected_usd: float,
    *,
    soft_stop_usd: float = DEFAULT_SOFT_STOP_USD,
    hard_cap_usd: float = HARD_CAP_USD,
) -> None:
    """Raise ``RuntimeError`` before exceeding soft-stop or authorized hard cap."""
    for name, value in (
        ("spent_usd", spent_usd),
        ("projected_usd", projected_usd),
        ("soft_stop_usd", soft_stop_usd),
        ("hard_cap_usd", hard_cap_usd),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    if hard_cap_usd > HARD_CAP_USD:
        raise ValueError(f"hard_cap_usd ({hard_cap_usd}) cannot exceed authorized ${HARD_CAP_USD:.2f} USD limit")
    if soft_stop_usd > hard_cap_usd:
        raise ValueError(f"soft_stop_usd ({soft_stop_usd}) cannot exceed hard_cap_usd ({hard_cap_usd})")
    if spent_usd >= soft_stop_usd:
        raise RuntimeError(f"Budget soft-stop reached: ${spent_usd:.4f} >= ${soft_stop_usd:.2f} USD")
    if spent_usd + projected_usd > hard_cap_usd:
        raise RuntimeError(f"Projected spend ${spent_usd + projected_usd:.4f} exceeds hard cap ${hard_cap_usd:.2f} USD")
    if spent_usd + projected_usd > soft_stop_usd:
        raise RuntimeError(f"Projected spend ${spent_usd + projected_usd:.4f} exceeds soft-stop ${soft_stop_usd:.2f} USD")


def build_validation_plan(
    *,
    tier: str,
    subrun: str = "all",
    gpu: str = DEFAULT_GPU,
    soft_stop_usd: float = DEFAULT_SOFT_STOP_USD,
    hard_cap_usd: float = HARD_CAP_USD,
    anima_repo: str = ANIMA_HF_REPO,
    anima_file: str = "",
    include_adetailer: bool = False,
) -> dict[str, Any]:
    """Build a serializable validation plan and worst-case cost estimate."""
    check_budget_guard(0.0, 0.0, soft_stop_usd=soft_stop_usd, hard_cap_usd=hard_cap_usd)
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {TIERS}")
    if gpu not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU {gpu!r}; expected one of {SUPPORTED_GPUS}")
    selected_subruns = list(CUDA_SUBRUNS) if subrun == "all" else [subrun]
    for item in selected_subruns:
        if item not in CUDA_SUBRUNS:
            raise ValueError(f"Unknown CUDA subrun {item!r}; expected 'all' or one of {CUDA_SUBRUNS}")

    if tier == "cpu":
        resolve_cpu_pytest_targets(REPO_ROOT)
        max_cost = estimate_cost_usd(DEFAULT_TIMEOUT_S, gpu=None, cpu_cores=4.0, memory_gib=8.0)
    elif tier == "prefetch":
        max_cost = estimate_cost_usd(DEFAULT_TIMEOUT_S, gpu=None, cpu_cores=2.0, memory_gib=4.0)
    elif tier == "cuda":
        max_cost = round(len(selected_subruns) * estimate_cost_usd(DEFAULT_TIMEOUT_S, gpu=gpu, cpu_cores=4.0, memory_gib=16.0), 4)
    else:
        max_cost = 0.0

    return {
        "app_name": APP_NAME,
        "tags": APP_TAGS,
        "tier": tier,
        "subruns": selected_subruns if tier == "cuda" else [],
        "gpu": gpu if tier == "cuda" else None,
        "region": None,
        "scaledown_window_s": DEFAULT_SCALEDOWN_WINDOW_S,
        "timeout_s": DEFAULT_TIMEOUT_S,
        "retries": DEFAULT_RETRIES,
        "hf_cache_volume": HF_CACHE_VOLUME_NAME,
        "hf_cache_mount_mode": "read-write" if tier == "prefetch" else "read-only",
        "ignore_patterns": list(IGNORE_PATTERNS),
        "prefetch_targets": {
            "anima_repo": anima_repo,
            "anima_file": anima_file or "<auto-first-candidate>",
            "text_encoder": ANIMA_TEXT_ENCODER_FILE,
            "vae": ANIMA_VAE_FILE,
            "include_adetailer": include_adetailer,
        },
        "soft_stop_usd": soft_stop_usd,
        "hard_cap_usd": hard_cap_usd,
        "worst_case_estimate_usd": max_cost,
    }


def _is_ignored_path(rel_path: str, patterns: tuple[str, ...] = IGNORE_PATTERNS) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
        if pat.endswith("/**") and (rel_path == pat[:-3] or rel_path.startswith(pat[:-3] + "/")):
            return True
        if pat.startswith("**/") and fnmatch.fnmatch(rel_path.rsplit("/", 1)[-1], pat[3:]):
            return True
    return False


def _ensure_app_sys_path(app_dir: str = REMOTE_APP_DIR) -> None:
    for entry in (f"{app_dir}/modules_forge/packages", app_dir):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _to_builtin_primitives(value: Any) -> Any:
    """Recursively convert ``value`` into pure Python builtins safe for cross-env pickling."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, os.PathLike):
        return _to_builtin_primitives(os.fspath(value))
    if isinstance(value, Mapping):
        return {_to_builtin_primitives(k): _to_builtin_primitives(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_builtin_primitives(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _to_builtin_primitives(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _to_builtin_primitives(value.item())
    return _to_builtin_primitives(str(value))


def _check_snapshot_line_endings(app_dir: Path) -> dict[str, Any]:
    """Run ``tools/check_line_endings.py`` across manifest paths included in the snapshot."""
    t0 = time.monotonic()
    tool_path = app_dir / "tools" / "check_line_endings.py"
    manifest_path = app_dir / "tools" / "upstream_crlf.txt"
    cmd = [sys.executable, "tools/check_line_endings.py", "--snapshot"]
    spec = importlib.util.spec_from_file_location("remi_check_line_endings", tool_path)
    if spec is None or spec.loader is None:
        return {
            "cmd": cmd,
            "returncode": 1,
            "elapsed_s": round(time.monotonic() - t0, 3),
            "stdout_tail": "",
            "stderr_tail": f"Unable to load {tool_path}",
        }
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = mod.read_manifest(str(manifest_path))
    skipped_ignored = [name for name in files if _is_ignored_path(name) and not (app_dir / name).is_file()]
    included = [name for name in files if name not in skipped_ignored]
    found = mod.problems(included, root=str(app_dir))
    elapsed_s = round(time.monotonic() - t0, 3)
    stdout_tail = (
        "\n".join(found[-40:])
        if found
        else f"{len(included)} snapshot-included upstream CRLF files keep their line endings (skipped {len(skipped_ignored)} snapshot-ignored)"
    )
    return {
        "cmd": cmd,
        "returncode": 1 if found else 0,
        "elapsed_s": elapsed_s,
        "stdout_tail": stdout_tail,
        "stderr_tail": "",
        "checked_count": len(included),
        "skipped_ignored": skipped_ignored,
    }


def _run_cmd(cmd: list[str], *, cwd: str = REMOTE_APP_DIR, env: dict[str, str] | None = None) -> dict[str, Any]:
    merged_env = {**os.environ, **(env or {})}
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, env=merged_env, capture_output=True, text=True, check=False)
    elapsed_s = round(time.monotonic() - t0, 3)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
    }


def _execute_cpu_suite(pytest_targets: list[str] | None = None) -> dict[str, Any]:
    """Run Linux CPU checks inside the Modal CPU container."""
    _ensure_app_sys_path(REMOTE_APP_DIR)

    gradio_found = [name for name in ("gradio", "gradio_client", "gradio_rangeslider") if importlib.util.find_spec(name) is not None]
    steps: dict[str, Any] = {
        "env:gradio-free": {
            "ok": not gradio_found,
            "found": gradio_found,
        }
    }

    targets = resolve_cpu_pytest_targets(Path(REMOTE_APP_DIR), pytest_targets)
    steps["pytest:cpu_subset"] = _run_cmd([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets])

    has_git = (Path(REMOTE_APP_DIR) / ".git").exists()
    if has_git:
        sdapi = _run_cmd([sys.executable, "tools/sdapi_contract.py"])
        steps["contract:sdapi"] = {**sdapi, "status": "PASS" if sdapi["returncode"] == 0 else ("SKIPPED" if sdapi["returncode"] == 77 else "FAIL")}
    else:
        steps["contract:sdapi"] = {
            "status": "SKIPPED",
            "reason": "excluded .git in copy=False snapshot; unit suite covered by tests/test_remi_sdapi_contract.py",
            "returncode": 77,
        }

    steps["contract:upstream-budget"] = _run_cmd([sys.executable, "tools/upstream_forks.py", "budget"])
    if has_git:
        steps["contract:line-endings"] = _run_cmd([sys.executable, "tools/check_line_endings.py"])
    else:
        steps["contract:line-endings"] = _check_snapshot_line_endings(Path(REMOTE_APP_DIR))

    bench_out = "/tmp/remi-remote-cpu-bench.json"
    bench = _run_cmd([sys.executable, "tools/perf/bench_inference.py", "--synthetic", "--device", "cpu", "--out", bench_out])
    bench_payload: dict[str, Any] | None = None
    if bench["returncode"] == 0 and Path(bench_out).is_file():
        bench_payload = json.loads(Path(bench_out).read_text(encoding="utf-8"))
    steps["bench:cpu_synthetic"] = {**bench, "summary": (bench_payload or {}).get("suites")}

    ok = (
        steps["env:gradio-free"]["ok"]
        and steps["pytest:cpu_subset"]["returncode"] == 0
        and steps["contract:upstream-budget"]["returncode"] == 0
        and steps["contract:line-endings"]["returncode"] == 0
        and steps["bench:cpu_synthetic"]["returncode"] == 0
        and steps["contract:sdapi"]["returncode"] in (0, 77)
    )
    return _to_builtin_primitives({"tier": "cpu", "ok": ok, "steps": steps})


def _execute_prefetch(anima_repo: str, anima_file: str, include_adetailer: bool) -> dict[str, Any]:
    """C0: selective ``hf_hub_download`` into ``HF_CACHE_MOUNT_PATH`` (never ``snapshot_download``)."""
    from huggingface_hub import HfApi, hf_hub_download

    cache_dir = Path(os.environ.get("HF_HUB_CACHE", HF_CACHE_MOUNT_PATH))
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repo_files = set(api.list_repo_files(repo_id=anima_repo))

    chosen_ckpt = anima_file.strip()
    if not chosen_ckpt:
        chosen_ckpt = next((cand for cand in ANIMA_DEFAULT_CKPT_CANDIDATES if cand in repo_files), "")
        if not chosen_ckpt:
            safetensors = sorted(name for name in repo_files if name.endswith(".safetensors") and "diffusion_models" in name)
            if not safetensors:
                raise FileNotFoundError(f"No Anima diffusion checkpoint found in {anima_repo}")
            chosen_ckpt = safetensors[0]
    elif chosen_ckpt not in repo_files:
        raise FileNotFoundError(f"Requested Anima checkpoint {chosen_ckpt!r} not found in {anima_repo}")

    downloads: list[tuple[str, str]] = [
        (anima_repo, chosen_ckpt),
        (anima_repo, ANIMA_TEXT_ENCODER_FILE),
        (anima_repo, ANIMA_VAE_FILE),
    ]
    if include_adetailer:
        downloads.append((ADETAILER_HF_REPO, ADETAILER_DETECTOR_FILE))

    fetched: list[dict[str, Any]] = []
    for repo_id, filename in downloads:
        local_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir)))
        size_mb = round(local_path.stat().st_size / (1024 * 1024), 2)
        fetched.append({"repo_id": repo_id, "filename": filename, "path": str(local_path), "size_mb": size_mb})

    return _to_builtin_primitives({"tier": "prefetch", "ok": True, "cache_dir": str(cache_dir), "files": fetched})


def _find_cached_safetensors(cache_dir: Path, filename_suffix: str) -> Path | None:
    matches = sorted(cache_dir.glob(f"models--*/snapshots/*/**/{filename_suffix}"), key=lambda p: p.as_posix())
    if matches:
        return matches[0]
    direct = sorted(cache_dir.rglob(filename_suffix), key=lambda p: p.as_posix())
    return direct[0] if direct else None


def _select_cached_anima_ckpt(cache_dir: Path, anima_file: str = "") -> Path | None:
    """Deterministically select a prefetched Anima diffusion checkpoint from ``cache_dir``."""
    requested = anima_file.strip().lstrip("/")
    all_safetensors = sorted(cache_dir.rglob("*.safetensors"), key=lambda p: p.as_posix())
    if requested:
        exact = [p for p in all_safetensors if p.as_posix().endswith(f"/{requested}") or p.as_posix() == requested]
        if exact:
            return exact[0]
        req_name = Path(requested).name
        by_name = sorted(
            (p for p in all_safetensors if p.name == req_name),
            key=lambda p: (0 if "diffusion_models" in p.as_posix() else 1, p.as_posix()),
        )
        return by_name[0] if by_name else None

    for candidate in ANIMA_DEFAULT_CKPT_CANDIDATES:
        cand_name = Path(candidate).name
        preferred = sorted(
            (p for p in all_safetensors if p.name == cand_name),
            key=lambda p: (0 if "diffusion_models" in p.as_posix() else 1, p.as_posix()),
        )
        if preferred:
            return preferred[0]

    diffusion_ckpts = [p for p in all_safetensors if "diffusion_models" in p.as_posix()]
    return diffusion_ckpts[0] if diffusion_ckpts else None


def _execute_cuda_subrun(subrun: str, *, anima_file: str = "") -> dict[str, Any]:
    """Run one focused CUDA validation sub-run (read-only HF cache)."""
    return _to_builtin_primitives(_execute_cuda_subrun_impl(subrun, anima_file=anima_file))


def _execute_cuda_subrun_impl(subrun: str, *, anima_file: str = "") -> dict[str, Any]:
    """Internal implementation for ``_execute_cuda_subrun`` before builtin sanitization."""
    import torch

    _ensure_app_sys_path(REMOTE_APP_DIR)

    if not torch.cuda.is_available():
        return {"subrun": subrun, "ok": False, "error": "CUDA is unavailable inside container"}

    device = torch.device("cuda", torch.cuda.current_device())
    raw_torch_version = torch.__version__
    gpu_info = {
        "name": str.__str__(raw_name) if isinstance((raw_name := torch.cuda.get_device_name(device)), str) else str(raw_name),
        "capability": [int(x) for x in torch.cuda.get_device_capability(device)],
        "vram_gb": round(float(torch.cuda.get_device_properties(device).total_memory) / (1024**3), 2),
        "torch": str.__str__(raw_torch_version) if isinstance(raw_torch_version, str) else str(raw_torch_version),
    }

    if subrun == "bench":
        bench_out = "/tmp/remi-remote-cuda-bench.json"
        res = _run_cmd([sys.executable, "tools/perf/bench_inference.py", "--synthetic", "--device", "cuda", "--out", bench_out])
        payload = json.loads(Path(bench_out).read_text(encoding="utf-8")) if res["returncode"] == 0 and Path(bench_out).is_file() else None
        return {"subrun": subrun, "ok": res["returncode"] == 0, "gpu": gpu_info, "bench": res, "results": payload}

    if subrun == "families":
        targets = [t for t in ("tests/test_remi_ml_runtime_fixes.py", "tests/test_anima_semantic_connector.py", "tests/test_anima_reference_size.py") if (Path(REMOTE_APP_DIR) / t).is_file()]
        pytest_res = _run_cmd([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets]) if targets else {"returncode": 0, "stdout_tail": "no matching family test files", "stderr_tail": ""}
        # Direct CUDA synthetic shape check for DiT quantization & capability registry (blocks.2 is outside Anima sensitive_prefixes)
        from backend import memory_management
        from backend.nn.anima_quant import scale_fp8_state_dict

        probe_key = "blocks.2.self_attn.q_proj.weight"
        probe = {probe_key: torch.randn(256, 256, dtype=torch.bfloat16)}
        quantized = int(scale_fp8_state_dict(probe, fp8_matmul=bool(memory_management.supports_fp8_compute(device)), device=device))
        ok = pytest_res["returncode"] == 0 and quantized == 1 and probe[probe_key].dtype == torch.float8_e4m3fn
        return {"subrun": subrun, "ok": ok, "gpu": gpu_info, "fp8_quantized": quantized, "pytest": pytest_res}

    cache_dir = Path(os.environ.get("HF_HUB_CACHE", HF_CACHE_MOUNT_PATH))
    te_path = _find_cached_safetensors(cache_dir, "qwen_3_06b_base.safetensors")
    vae_path = _find_cached_safetensors(cache_dir, "qwen_image_vae.safetensors")
    ckpt_path = _select_cached_anima_ckpt(cache_dir, anima_file)
    adetailer_path = _find_cached_safetensors(cache_dir, ADETAILER_DETECTOR_FILE)

    if subrun == "anima":
        if not (ckpt_path and te_path and vae_path):
            return {
                "subrun": subrun,
                "ok": False,
                "gpu": gpu_info,
                "error": f"Missing prefetched Anima weights in {cache_dir}; run --tier prefetch first",
                "found": {"ckpt": str(ckpt_path), "te": str(te_path), "vae": str(vae_path)},
            }
        model_ref = Path("/tmp/remi-remote-models")
        data_dir = Path("/tmp/remi-remote-data")
        data_dir.mkdir(parents=True, exist_ok=True)
        for folder in ("Stable-diffusion", "text_encoder", "VAE", "adetailer"):
            (model_ref / folder).mkdir(parents=True, exist_ok=True)
        links = [
            (ckpt_path, model_ref / "Stable-diffusion" / ckpt_path.name),
            (te_path, model_ref / "text_encoder" / te_path.name),
            (vae_path, model_ref / "VAE" / vae_path.name),
        ]
        if adetailer_path is not None:
            links.append((adetailer_path, model_ref / "adetailer" / adetailer_path.name))
        for src, dst in links:
            if dst.is_symlink() and not dst.exists():
                dst.unlink()
            if not dst.exists():
                dst.symlink_to(src)

        # Verify quantization formats (FP8 / INT8-ConvRot) on CUDA (blocks.2 is outside Anima sensitive_prefixes)
        from backend import memory_management
        from backend.nn.anima_quant import quantize_int8_convrot_state_dict, scale_fp8_state_dict

        probe_key = "blocks.2.self_attn.q_proj.weight"
        w = torch.randn(512, 512, dtype=torch.bfloat16)
        fp8_probe = {probe_key: w.clone()}
        fp8_count = int(scale_fp8_state_dict(fp8_probe, fp8_matmul=bool(memory_management.supports_fp8_compute(device)), device=device))
        int8_probe = {probe_key: w.clone()}
        int8_count = int(quantize_int8_convrot_state_dict(int8_probe, device=device))

        # Run real Anima txt2img generation through the Gradio-free FastAPI server
        tools_dir = f"{REMOTE_APP_DIR}/tools"
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import io

        import numpy as np
        from PIL import Image
        from smoke_real import Server, wait_job

        server = Server(sys.executable, 7879, data_dir, Path("/tmp/remi-remote-server.log"), model_ref=model_ref)
        try:
            if not server.wait_ready(300):
                log_tail = Path("/tmp/remi-remote-server.log").read_text(errors="replace")[-4000:]
                return {"subrun": subrun, "ok": False, "gpu": gpu_info, "error": "server failed to start", "log_tail": log_tail}
            options_resp = server.request(
                "PATCH",
                "/api/v2/options",
                {
                    "sd_model_checkpoint": ckpt_path.name,
                    "forge_additional_modules": [
                        str(model_ref / "VAE" / vae_path.name),
                        str(model_ref / "text_encoder" / te_path.name),
                    ],
                },
            )
            if isinstance(options_resp, dict) and isinstance(options_resp.get("values"), dict):
                applied_values = options_resp["values"]
                applied_modules = {Path(str(m)).name for m in (applied_values.get("forge_additional_modules") or [])}
                if not applied_values.get("sd_model_checkpoint") or not {vae_path.name, te_path.name}.issubset(applied_modules):
                    return {
                        "subrun": subrun,
                        "ok": False,
                        "gpu": gpu_info,
                        "checkpoint": ckpt_path.name,
                        "error": f"Options patch did not retain expected checkpoint/modules: {applied_values!r}",
                    }

            is_turbo = "turbo" in ckpt_path.name.lower()
            submitted = server.request(
                "POST",
                "/api/v2/jobs",
                {
                    "tab": "txt2img",
                    "core": {
                        "prompt": "1girl, anime illustration, clean lineart, smiling",
                        "negative_prompt": "lowres, blurry",
                        "width": 512,
                        "height": 512,
                        "cfg_scale": 1.0 if is_turbo else 4.0,
                        "distilled_cfg_scale": 3.5 if is_turbo else 3.0,
                        "batch_size": 1,
                        "batch_count": 1,
                    },
                    "scripts": {
                        "sampler": {"sampler_name": "Euler", "scheduler": "Simple", "steps": 8},
                        "seed": {"seed": 12345},
                    },
                },
            )
            if not isinstance(submitted, dict) or not (submitted.get("job_id") or submitted.get("id")):
                return {
                    "subrun": subrun,
                    "ok": False,
                    "gpu": gpu_info,
                    "checkpoint": ckpt_path.name,
                    "error": f"Unexpected job submission response: {submitted!r}",
                }
            job_id = str(submitted.get("job_id") or submitted["id"])
            job = wait_job(server, job_id, timeout=300)
            status = job.get("status")
            outputs = job.get("outputs") or []
            if status != "completed" or not outputs or not isinstance(outputs[0], dict) or not outputs[0].get("id"):
                return {
                    "subrun": subrun,
                    "ok": False,
                    "gpu": gpu_info,
                    "checkpoint": ckpt_path.name,
                    "job_id": job_id,
                    "status": status,
                    "error": job.get("error") or f"Job {job_id} finished with status={status!r} and {len(outputs)} outputs",
                    "errors": job.get("errors"),
                }
            raw_png = server.request("GET", f"/api/v2/files/{outputs[0]['id']}")
            assert isinstance(raw_png, (bytes, bytearray))
            img = Image.open(io.BytesIO(raw_png)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32)
            pixel_std = round(float(arr.std()), 3)
            ok = fp8_count == 1 and int8_count == 1 and img.size == (512, 512) and pixel_std > 5.0
            return {
                "subrun": subrun,
                "ok": ok,
                "gpu": gpu_info,
                "checkpoint": ckpt_path.name,
                "fp8_quantized": fp8_count,
                "int8_convrot_quantized": int8_count,
                "image_size": [int(d) for d in img.size],
                "pixel_std": pixel_std,
            }
        finally:
            server.stop()

    if subrun == "extensions":
        ext_targets = [
            t
            for t in (
                "extensions-builtin/sd_forge_spectrum/tests",
                "extensions-builtin/sd_forge_controlnet/tests",
                "extensions-builtin/ADetailer-Neo/tests",
            )
            if (Path(REMOTE_APP_DIR) / t).is_dir()
        ]
        if not ext_targets:
            return {"subrun": subrun, "ok": False, "gpu": gpu_info, "error": "No extension test directories found"}
        res = _run_cmd([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *ext_targets])
        return {"subrun": subrun, "ok": res["returncode"] == 0, "gpu": gpu_info, "pytest": res}

    raise ValueError(f"Unhandled subrun {subrun!r}")


def _serialize_billing_row(item: Any) -> dict[str, Any]:
    """Normalize Modal billing row (either ``Mapping`` or dataclass/object) to a plain dict."""
    if isinstance(item, Mapping):
        object_id = item.get("object_id")
        description = item.get("description")
        env_name = item.get("environment_name") if item.get("environment_name") is not None else item.get("environment")
        interval = item.get("interval_start", "")
        cost_raw = item.get("cost", "")
        tags_raw = item.get("tags")
    else:
        object_id = getattr(item, "object_id", None)
        description = getattr(item, "description", None)
        env_name = getattr(item, "environment_name", None)
        if env_name is None:
            env_name = getattr(item, "environment", None)
        interval = getattr(item, "interval_start", "")
        cost_raw = getattr(item, "cost", "")
        tags_raw = getattr(item, "tags", None)

    interval_str = interval.isoformat() if isinstance(interval, datetime) else str(interval or "")
    tags = {str(k): str(v) for k, v in tags_raw.items()} if isinstance(tags_raw, Mapping) else {}
    return {
        "object_id": object_id,
        "description": description,
        "environment_name": env_name,
        "interval_start": interval_str,
        "cost": str(cost_raw if cost_raw is not None else ""),
        "tags": tags,
    }


def _sum_billing_cost(rows: list[dict[str, Any]]) -> float:
    total = Decimal("0")
    for row in rows:
        raw = str(row.get("cost") or "").strip()
        if not raw:
            continue
        try:
            total += Decimal(raw)
        except InvalidOperation:
            continue
    return float(round(total, 8))


def fetch_billing_report(*, days: int = 1) -> dict[str, Any]:
    """Query Modal workspace billing via ``Workspace.billing.report``, legacy SDK, or CLI fallback."""
    now = datetime.now(UTC)
    end = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    start = end - timedelta(days=max(1, days))
    notes: list[str] = []

    if modal is not None:
        workspace_cls = getattr(modal, "Workspace", None)
        from_context = getattr(workspace_cls, "from_context", None) if workspace_cls is not None else None
        if callable(from_context):
            try:
                workspace = from_context()
                billing = getattr(workspace, "billing", None)
                report_fn = getattr(billing, "report", None) if billing is not None else None
                if callable(report_fn):
                    rows = report_fn(start=start, end=end, resolution="h", tag_names=["project"])
                    serialized = [_serialize_billing_row(item) for item in rows]
                    return {
                        "ok": True,
                        "source": "modal.Workspace.billing.report",
                        "rows": serialized,
                        "total_cost_usd": _sum_billing_cost(serialized),
                    }
            except Exception as exc:
                notes.append(f"Workspace.billing.report unavailable ({type(exc).__name__}: {exc})")

        billing_mod = getattr(modal, "billing", None)
        legacy_fn = getattr(billing_mod, "workspace_billing_report", None) if billing_mod is not None else None
        if callable(legacy_fn):
            try:
                rows = legacy_fn(start=start, end=end, resolution="h", tag_names=["project"])
                serialized = [_serialize_billing_row(item) for item in rows]
                result: dict[str, Any] = {
                    "ok": True,
                    "source": "modal.billing.workspace_billing_report",
                    "rows": serialized,
                    "total_cost_usd": _sum_billing_cost(serialized),
                }
                if notes:
                    result["note"] = "; ".join(notes)
                return result
            except Exception as exc:
                notes.append(f"workspace_billing_report unavailable ({type(exc).__name__}: {exc})")
        elif not notes:
            notes.append("modal billing API unavailable in current SDK")
    else:
        notes.append("modal Python SDK not installed in current interpreter")

    modal_bin = shutil.which("modal")
    note = "; ".join([*notes, "falling back to CLI"])
    if not modal_bin:
        return {"ok": False, "source": "unavailable", "rows": [], "total_cost_usd": 0.0, "note": f"{note}; `modal` CLI not on PATH"}

    for_range = "this week" if days <= 7 else "this month"
    try:
        proc = subprocess.run(
            [modal_bin, "billing", "report", "--for", for_range, "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "source": "modal billing report", "rows": [], "total_cost_usd": 0.0, "note": f"{note}; CLI failed ({type(exc).__name__}: {exc})"}

    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout or "[]")
        except ValueError:
            return {"ok": True, "source": f"modal billing report --for {for_range!r}", "rows": [], "total_cost_usd": 0.0, "raw": proc.stdout, "note": note}
        if isinstance(parsed, list):
            serialized = [_serialize_billing_row(item) for item in parsed]
            return {
                "ok": True,
                "source": f"modal billing report --for {for_range!r} --json",
                "rows": serialized,
                "total_cost_usd": _sum_billing_cost(serialized),
                "note": note,
            }
        return {"ok": True, "source": f"modal billing report --for {for_range!r} --json", "rows": [], "total_cost_usd": 0.0, "raw": parsed, "note": note}
    return {"ok": False, "source": "modal billing report", "rows": [], "total_cost_usd": 0.0, "note": f"{note}; CLI exited {proc.returncode}: {proc.stderr.strip()}"}


if modal is not None:
    app = modal.App(APP_NAME, tags=APP_TAGS)
    hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True, version=2)
    _matcher = modal.FilePatternMatcher(*IGNORE_PATTERNS)

    cpu_image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
        .uv_pip_install("torch==2.14.0+cpu", "torchvision==0.29.0+cpu", index_url="https://download.pytorch.org/whl/cpu")
        .uv_pip_install(
            "torch==2.14.0+cpu",
            "torchvision==0.29.0+cpu",
            "ultralytics==8.3.253",
            requirements=[str(REPO_ROOT / "requirements.txt"), str(REPO_ROOT / "requirements-dev.txt")],
            extra_index_url="https://download.pytorch.org/whl/cpu",
            extra_options="--index-strategy unsafe-best-match",
        )
        .env({"HF_HUB_CACHE": HF_CACHE_MOUNT_PATH, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{REMOTE_APP_DIR}:{REMOTE_APP_DIR}/modules_forge/packages"})
        .add_local_dir(str(REPO_ROOT), REMOTE_APP_DIR, copy=False, ignore=_matcher)
    )

    prefetch_image = (
        modal.Image.debian_slim(python_version="3.13")
        .uv_pip_install("huggingface-hub==0.36.2", "hf_transfer")
        .env({"HF_HUB_CACHE": HF_CACHE_MOUNT_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    )

    cuda_image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1", "ffmpeg")
        .uv_pip_install(
            "torch==2.14.0+cu130",
            "torchvision==0.29.0+cu130",
            extra_index_url="https://download.pytorch.org/whl/cu130",
            extra_options="--index-strategy unsafe-best-match",
        )
        .uv_pip_install(
            "torch==2.14.0+cu130",
            "torchvision==0.29.0+cu130",
            "ultralytics==8.3.253",
            requirements=[str(REPO_ROOT / "requirements.txt"), str(REPO_ROOT / "requirements-dev.txt")],
            extra_index_url="https://download.pytorch.org/whl/cu130",
            extra_options="--index-strategy unsafe-best-match",
        )
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_MOUNT_PATH,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": f"{REMOTE_APP_DIR}:{REMOTE_APP_DIR}/modules_forge/packages",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        .add_local_dir(str(REPO_ROOT), REMOTE_APP_DIR, copy=False, ignore=_matcher)
    )

    @app.function(
        image=cpu_image,
        cpu=4.0,
        memory=8192,
        timeout=DEFAULT_TIMEOUT_S,
        scaledown_window=DEFAULT_SCALEDOWN_WINDOW_S,
        retries=DEFAULT_RETRIES,
        region=None,
    )
    def remote_cpu_suite(pytest_targets: list[str] | None = None) -> dict[str, Any]:
        return _to_builtin_primitives(_execute_cpu_suite(pytest_targets))

    @app.function(
        image=prefetch_image,
        cpu=2.0,
        memory=4096,
        volumes={HF_CACHE_MOUNT_PATH: hf_cache_vol},
        timeout=DEFAULT_TIMEOUT_S,
        scaledown_window=DEFAULT_SCALEDOWN_WINDOW_S,
        retries=DEFAULT_RETRIES,
        region=None,
    )
    def remote_prefetch(anima_repo: str = ANIMA_HF_REPO, anima_file: str = "", include_adetailer: bool = False) -> dict[str, Any]:
        res = _execute_prefetch(anima_repo, anima_file, include_adetailer)
        hf_cache_vol.commit()
        return _to_builtin_primitives(res)

    @app.function(
        image=cuda_image,
        gpu=DEFAULT_GPU,
        cpu=4.0,
        memory=16384,
        volumes={HF_CACHE_MOUNT_PATH: hf_cache_vol.with_mount_options(read_only=True)},
        timeout=DEFAULT_TIMEOUT_S,
        scaledown_window=DEFAULT_SCALEDOWN_WINDOW_S,
        retries=DEFAULT_RETRIES,
        region=None,
    )
    def remote_cuda_subrun(subrun: str, anima_file: str = "") -> dict[str, Any]:
        return _to_builtin_primitives(_execute_cuda_subrun(subrun, anima_file=anima_file))

    @app.local_entrypoint()
    def entrypoint(
        tier: str = "cpu",
        subrun: str = "all",
        gpu: str = DEFAULT_GPU,
        dry_run: bool = False,
        soft_stop_usd: float = DEFAULT_SOFT_STOP_USD,
        hard_cap_usd: float = HARD_CAP_USD,
        anima_repo: str = ANIMA_HF_REPO,
        anima_file: str = "",
        include_adetailer: bool = False,
        report_out: str = "",
    ) -> None:
        plan = build_validation_plan(
            tier=tier,
            subrun=subrun,
            gpu=gpu,
            soft_stop_usd=soft_stop_usd,
            hard_cap_usd=hard_cap_usd,
            anima_repo=anima_repo,
            anima_file=anima_file,
            include_adetailer=include_adetailer,
        )
        if dry_run:
            print(json.dumps(plan, indent=2))
            return

        if tier == "billing":
            billing_payload = json.dumps(fetch_billing_report(), indent=2)
            if report_out:
                out_path = Path(report_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(billing_payload + "\n", encoding="utf-8")
            print(billing_payload)
            return

        spent_usd = 0.0
        results: list[dict[str, Any]] = []
        if tier == "cpu":
            check_budget_guard(spent_usd, estimate_cost_usd(DEFAULT_TIMEOUT_S, cpu_cores=4.0, memory_gib=8.0), soft_stop_usd=soft_stop_usd, hard_cap_usd=hard_cap_usd)
            t0 = time.monotonic()
            res = remote_cpu_suite.remote()
            elapsed = time.monotonic() - t0
            cost = estimate_cost_usd(elapsed, cpu_cores=4.0, memory_gib=8.0)
            spent_usd += cost
            results.append({"wall_s": round(elapsed, 2), "estimated_cost_usd": cost, **res})
        elif tier == "prefetch":
            check_budget_guard(spent_usd, estimate_cost_usd(DEFAULT_TIMEOUT_S, cpu_cores=2.0, memory_gib=4.0), soft_stop_usd=soft_stop_usd, hard_cap_usd=hard_cap_usd)
            t0 = time.monotonic()
            res = remote_prefetch.remote(anima_repo=anima_repo, anima_file=anima_file, include_adetailer=include_adetailer)
            elapsed = time.monotonic() - t0
            cost = estimate_cost_usd(elapsed, cpu_cores=2.0, memory_gib=4.0)
            spent_usd += cost
            results.append({"wall_s": round(elapsed, 2), "estimated_cost_usd": cost, **res})
        elif tier == "cuda":
            fn = remote_cuda_subrun if gpu == DEFAULT_GPU else remote_cuda_subrun.with_options(gpu=gpu)
            for item in plan["subruns"]:
                projected = estimate_cost_usd(DEFAULT_TIMEOUT_S, gpu=gpu, cpu_cores=4.0, memory_gib=16.0)
                check_budget_guard(spent_usd, projected, soft_stop_usd=soft_stop_usd, hard_cap_usd=hard_cap_usd)
                t0 = time.monotonic()
                res = fn.remote(item, anima_file=anima_file) if item == "anima" else fn.remote(item)
                elapsed = time.monotonic() - t0
                cost = estimate_cost_usd(elapsed, gpu=gpu, cpu_cores=4.0, memory_gib=16.0)
                spent_usd += cost
                results.append({"wall_s": round(elapsed, 2), "estimated_cost_usd": cost, **res})

        summary = {
            "plan": plan,
            "estimated_total_usd": round(spent_usd, 4),
            "billing": fetch_billing_report(),
            "results": results,
        }
        payload = json.dumps(summary, indent=2)
        if report_out:
            out_path = Path(report_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        if any(not entry.get("ok", False) for entry in results):
            raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remi ephemeral Modal validation harness (run remote tiers via `modal run`).",
    )
    parser.add_argument("--tier", choices=TIERS, default="cpu", help="Validation tier (default: %(default)s)")
    parser.add_argument("--subrun", choices=("all", *CUDA_SUBRUNS), default="all", help="Focused CUDA sub-run for --tier cuda")
    parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default=DEFAULT_GPU, help="Modal GPU type for --tier cuda (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved execution plan and cost ceiling without calling Modal")
    parser.add_argument("--soft-stop-usd", type=float, default=DEFAULT_SOFT_STOP_USD, help="Conservative local budget soft stop in USD")
    parser.add_argument("--hard-cap-usd", type=float, default=HARD_CAP_USD, help="Authorized hard spend cap in USD (<= 30)")
    parser.add_argument("--anima-repo", default=ANIMA_HF_REPO, help="Hugging Face repo for C0 selective weight prefetch")
    parser.add_argument("--anima-file", default="", help="Specific diffusion checkpoint path inside --anima-repo")
    parser.add_argument("--include-adetailer", action="store_true", help="Also prefetch face_yolov8n.pt during C0")
    parser.add_argument("--report-out", default="", help="Optional path to write JSON report")
    args = parser.parse_args(argv)

    if args.tier == "billing":
        billing_payload = json.dumps(fetch_billing_report(), indent=2)
        if args.report_out:
            out_path = Path(args.report_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(billing_payload + "\n", encoding="utf-8")
        print(billing_payload)
        return 0

    plan = build_validation_plan(
        tier=args.tier,
        subrun=args.subrun,
        gpu=args.gpu,
        soft_stop_usd=args.soft_stop_usd,
        hard_cap_usd=args.hard_cap_usd,
        anima_repo=args.anima_repo,
        anima_file=args.anima_file,
        include_adetailer=args.include_adetailer,
    )
    if args.dry_run:
        payload = json.dumps(plan, indent=2)
        if args.report_out:
            out_path = Path(args.report_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    print(
        "Direct python execution only supports --dry-run, --tier billing, or --help.\n"
        f"To launch an ephemeral Modal run, execute:\n  modal run tools/remote/modal_validate.py --tier {args.tier}"
        + (f" --subrun {args.subrun} --gpu {args.gpu}" if args.tier == "cuda" else ""),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
