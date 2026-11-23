"""Cross-platform inference, memory, quantization, acceleration & soak benchmark harness.

Reproducibly benchmarks Forge Neo Remi against upstream Forge Neo and ComfyUI across
NVIDIA CUDA, Apple Silicon MPS, and CPU-synthetic CI environments with a standardized
JSON schema and Markdown comparison table generator.

Examples:
    # Fast CI / CPU-synthetic run (all suites + 100-job SQLite queue soak test)
    venv/bin/python tools/perf/bench_inference.py --synthetic --out bench_cpu.json

    # Apple Silicon MPS micro-benchmarks (quantized ops, attention, memory, soak)
    venv/bin/python tools/perf/bench_inference.py --synthetic --device mps --out bench_mps.json

    # Real-checkpoint run against a live Remi, Forge Neo, or ComfyUI server
    venv/bin/python tools/perf/bench_inference.py --real --target remi \\
        --base-url http://127.0.0.1:7860 --checkpoint anima-turbo-v1.1.safetensors \\
        --second-checkpoint anima-base.safetensors --out remi.json

    # Generate Markdown comparison tables across Remi vs Forge Neo vs ComfyUI runs
    venv/bin/python tools/perf/bench_inference.py --compare remi.json neo.json comfyui.json \\
        --markdown-out comparison.md

    # Static diagnosis of the historical tools/smoke_real.py txt2img pixel-parity drift
    venv/bin/python tools/perf/bench_inference.py --diagnose-parity
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
DEFAULT_SUITES = ("inference", "model_lifecycle", "quantization", "optimizations", "soak")
ALL_SUITES = (*DEFAULT_SUITES, "parity_diagnosis")
ALL_QUANT_FORMATS = ("bf16", "fp8_e4m3", "fp8_rowwise", "int8_convrot", "gguf_q8_0", "svdquant_w4a4", "nvfp4")
ALL_ATTN_BACKENDS = ("basic", "sdpa", "sage", "flash", "xformers")

# Loopback / direct HTTP without picking up desktop SOCKS/HTTP proxies.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# region Helpers & Device Telemetry


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto|cuda|mps|cpu`` to an available ``torch.device``."""
    req = (requested or "auto").strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {requested!r} is unavailable (CUDA not available)")
        return torch.device(req)
    if req == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' is unavailable on this host")
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device selector: {requested!r}")


def sync_device(device: torch.device) -> None:
    """Synchronize accelerator work for accurate wall-clock timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def empty_device_cache(device: torch.device) -> None:
    """Flush allocator cache on the active accelerator."""
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    elif device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def sample_memory_mb(device: torch.device) -> dict[str, float]:
    """Snapshot process RSS and accelerator allocated/driver memory in MiB."""
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    out: dict[str, float] = {
        "rss_mb": round(rss_mb, 2),
        "device_allocated_mb": 0.0,
        "device_reserved_mb": 0.0,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        out["device_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2)
        out["device_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2)
        out["device_peak_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
    elif device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        alloc = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
        driver = torch.mps.driver_allocated_memory() if hasattr(torch.mps, "driver_allocated_memory") else alloc
        rec = torch.mps.recommended_max_memory() if hasattr(torch.mps, "recommended_max_memory") else 0
        out["device_allocated_mb"] = round(alloc / (1024 * 1024), 2)
        out["device_reserved_mb"] = round(driver / (1024 * 1024), 2)
        out["mps_recommended_max_mb"] = round(rec / (1024 * 1024), 2)
    return out


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank linear interpolation percentile for non-empty sequences."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(idx)
    hi = min(len(ordered) - 1, lo + 1)
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def linear_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope dy/dx (0.0 for fewer than 2 points or constant x)."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    x_mean = sum(xs[:n]) / n
    y_mean = sum(ys[:n]) / n
    denom = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if denom == 0.0:
        return 0.0
    numer = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    return numer / denom


def compute_tensor_fidelity(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Relative L2 error, MAE, max abs diff and PSNR (dB) between two float tensors."""
    a = candidate.detach().float().cpu()
    b = reference.detach().float().cpu()
    diff = a - b
    ref_norm = float(torch.linalg.vector_norm(b).item())
    err_norm = float(torch.linalg.vector_norm(diff).item())
    rel_l2 = err_norm / max(ref_norm, 1e-12)
    mse = float(torch.mean(diff * diff).item())
    mae = float(torch.mean(diff.abs()).item())
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    dyn_range = float((b.max() - b.min()).item()) if b.numel() else 1.0
    dyn_range = max(dyn_range, 1e-6)
    psnr = 99.99 if mse <= 1e-18 else min(99.99, 20.0 * math.log10(dyn_range / math.sqrt(mse)))
    return {
        "rel_l2_error": round(rel_l2, 6),
        "mae": round(mae, 6),
        "max_abs_diff": round(max_abs, 6),
        "psnr_db": round(psnr, 2),
    }


def compute_image_fidelity(image_a: Any, image_b: Any) -> dict[str, float | None]:
    """Compute PSNR (dB), global SSIM, MAE, max RGB diff, and optional LPIPS for two PIL Images."""
    x = np.asarray(image_a.convert("RGB"), dtype=np.float64)
    y = np.asarray(image_b.convert("RGB"), dtype=np.float64)
    if x.shape != y.shape:
        return {"psnr_db": 0.0, "ssim": 0.0, "mae": 255.0, "max_abs_diff": 255.0, "lpips": None}
    diff = np.abs(x - y)
    mse = float(np.mean((x - y) ** 2))
    mae = float(np.mean(diff))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    psnr = 99.99 if mse <= 1e-12 else min(99.99, 10.0 * math.log10((255.0**2) / mse))

    # Luminance-weighted SSIM (Wang et al. constants for 8-bit dynamic range L=255)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))

    lpips_score: float | None = None
    try:
        import lpips  # type: ignore[import-not-found]

        loss_fn = lpips.LPIPS(net="alex", verbose=False)
        ta = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float() / 127.5 - 1.0
        tb = torch.from_numpy(y.transpose(2, 0, 1)).unsqueeze(0).float() / 127.5 - 1.0
        with torch.inference_mode():
            lpips_score = round(float(loss_fn(ta, tb).item()), 5)
    except Exception:
        lpips_score = None

    return {
        "psnr_db": round(psnr, 2),
        "ssim": round(float(ssim), 5),
        "mae": round(mae, 4),
        "max_abs_diff": round(max_abs, 2),
        "lpips": lpips_score,
    }


def _git_revision() -> str:
    try:
        env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"}
        res = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10, check=False)
        return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        return ""


def _probe_available_backends(device: torch.device) -> tuple[dict[str, bool], dict[str, bool]]:
    from backend import memory_management

    attn: dict[str, bool] = {
        "basic": True,
        "sdpa": hasattr(F, "scaled_dot_product_attention"),
        "sage": False,
        "flash": False,
        "xformers": False,
        "comfy_kitchen_int8": False,
    }
    if device.type == "cuda":
        attn["sage"] = bool(memory_management.sage_enabled())
        attn["flash"] = bool(memory_management.flash_enabled())
        attn["xformers"] = bool(memory_management.xformers_enabled())
        attn["comfy_kitchen_int8"] = bool(memory_management.ck_enabled())

    gguf_available = False
    try:
        import gguf  # noqa: F401

        gguf_available = True
    except Exception:
        gguf_available = False

    svdquant_available = False
    try:
        from backend.quant_ops import ck

        svdquant_available = hasattr(ck, "quantize_svdquant_w4a4")
    except Exception:
        svdquant_available = False

    quant: dict[str, bool] = {
        "bf16": device.type != "cpu" or hasattr(torch, "bfloat16"),
        "fp8_e4m3": hasattr(torch, "float8_e4m3fn"),
        "fp8_rowwise": hasattr(torch, "float8_e4m3fn"),
        "fp8_native_gemm": bool(memory_management.supports_fp8_compute(device)),
        "int8_convrot": True,
        "int8_native_gemm": bool(memory_management.supports_int8_compute(device)),
        "gguf_q8_0": gguf_available,
        "svdquant_w4a4": svdquant_available,
        "nvfp4": hasattr(torch, "float8_e4m3fn"),
        "nvfp4_native_gemm": bool(memory_management.supports_nvfp4_compute(device)),
    }
    return attn, quant


def collect_system_info(device: torch.device, system_label: str = "remi") -> dict[str, Any]:
    """Collect reproducible platform, PyTorch, hardware, and backend capability metadata."""
    attn_backends, quant_backends = _probe_available_backends(device)
    device_name = device.type
    cuda_cap: list[int] | None = None
    mps_rec_mb: float | None = None

    if device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)
        cuda_cap = list(torch.cuda.get_device_capability(device))
    elif device.type == "mps":
        device_name = f"Apple Silicon MPS ({platform.machine()})"
        if hasattr(torch.mps, "recommended_max_memory"):
            try:
                mps_rec_mb = round(torch.mps.recommended_max_memory() / (1024 * 1024), 1)
            except Exception:
                mps_rec_mb = None
    else:
        device_name = platform.processor() or platform.machine() or "cpu"

    return {
        "system_label": system_label,
        "platform": platform.system(),
        "os_version": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "backend": device.type,
        "device_name": device_name,
        "cuda_capability": cuda_cap,
        "mps_recommended_max_mb": mps_rec_mb,
        "git_revision": _git_revision(),
        "available_attention_backends": attn_backends,
        "available_quant_backends": quant_backends,
    }


# endregion

# region Synthetic DiT Workload


class SyntheticDiTBlock(nn.Module):
    """Compact DiT transformer block mirroring Anima / Qwen-2.1 / FLUX.3 compute patterns."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio, bias=False)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(attn)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))
        return x


class SyntheticDiT(nn.Module):
    """Multi-block synthetic DiT with explicit pre-head seam for Spectrum & compile tests."""

    def __init__(self, *, dim: int = 256, num_heads: int = 4, num_blocks: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([SyntheticDiTBlock(dim, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(dim)
        self.final_head = nn.Linear(dim, dim, bias=False)

    def forward_features(self, x: torch.Tensor, timestep: float) -> torch.Tensor:
        h = x + (timestep * 0.01)
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.final_head(self.final_norm(features))

    def forward(self, x: torch.Tensor, timestep: float = 0.5) -> torch.Tensor:
        return self.forward_head(self.forward_features(x, timestep))


def _preferred_compute_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


# endregion

# region Suite 1: Inference Throughput & Memory


def run_synthetic_inference_suite(
    device: torch.device,
    *,
    steps: int = 12,
    warmup_steps: int = 2,
    runs: int = 2,
    tokens: int = 256,
    dim: int = 256,
    num_blocks: int = 4,
    num_heads: int = 4,
    seed: int = 12345,
) -> dict[str, Any]:
    """Measure step latency (median/p95), throughput (it/s), first/steady image latency & memory."""
    dtype = _preferred_compute_dtype(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model = SyntheticDiT(dim=dim, num_heads=num_heads, num_blocks=num_blocks).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    model = model.to(device=device, dtype=dtype)

    x0 = (torch.randn(1, tokens, dim, generator=gen) * 0.1).to(device=device, dtype=dtype)
    empty_device_cache(device)
    reset_peak_memory(device)
    mem_before = sample_memory_mb(device)

    step_times_ms: list[float] = []
    image_times_s: list[float] = []
    peak_alloc_mb = mem_before["device_allocated_mb"]
    peak_driver_mb = mem_before["device_reserved_mb"]
    peak_rss_mb = mem_before["rss_mb"]

    with torch.inference_mode():
        # Warm-up passes (unrecorded)
        w = x0.clone()
        for idx in range(max(1, warmup_steps)):
            w = model(w, timestep=1.0 - idx / max(1, warmup_steps))
        sync_device(device)

        for _run_idx in range(max(1, runs)):
            x = x0.clone()
            sync_device(device)
            t_img_start = time.perf_counter()
            for step in range(max(1, steps)):
                t_step_start = time.perf_counter()
                x = model(x, timestep=1.0 - step / max(1, steps))
                sync_device(device)
                step_times_ms.append((time.perf_counter() - t_step_start) * 1000.0)
            image_times_s.append(time.perf_counter() - t_img_start)
            snap = sample_memory_mb(device)
            peak_alloc_mb = max(peak_alloc_mb, snap.get("device_peak_mb", snap["device_allocated_mb"]))
            peak_driver_mb = max(peak_driver_mb, snap["device_reserved_mb"])
            peak_rss_mb = max(peak_rss_mb, snap["rss_mb"])

    del model, x0
    empty_device_cache(device)

    med_ms = statistics.median(step_times_ms)
    p95_ms = percentile(step_times_ms, 95.0)
    it_per_s = 1000.0 / med_ms if med_ms > 0 else 0.0
    steady_s = statistics.median(image_times_s[1:]) if len(image_times_s) > 1 else image_times_s[0]
    return {
        "status": "ok",
        "dtype": str(dtype).removeprefix("torch."),
        "tokens": tokens,
        "dim": dim,
        "num_blocks": num_blocks,
        "steps": steps,
        "runs": runs,
        "ms_per_step_median": round(med_ms, 3),
        "ms_per_step_p95": round(p95_ms, 3),
        "it_per_s": round(it_per_s, 3),
        "first_image_s": round(image_times_s[0], 4),
        "steady_image_s": round(steady_s, 4),
        "peak_device_mb": round(peak_alloc_mb, 2),
        "peak_driver_mb": round(peak_driver_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "delta_rss_mb": round(max(0.0, peak_rss_mb - mem_before["rss_mb"]), 2),
    }


# endregion

# region Suite 2: Model Lifecycle (Cold / Warm Load & Checkpoint Switch)


def run_synthetic_lifecycle_suite(
    device: torch.device,
    *,
    dim: int = 256,
    num_blocks: int = 4,
    num_heads: int = 4,
    seed: int = 12345,
) -> dict[str, Any]:
    """Measure cold state_dict load from disk, warm reload, A<->B switch, and unload reclaim."""
    from safetensors.torch import load_file, save_file

    dtype = _preferred_compute_dtype(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def _build_state(scale: float) -> dict[str, torch.Tensor]:
        m = SyntheticDiT(dim=dim, num_heads=num_heads, num_blocks=num_blocks)
        return {k: (torch.randn(v.shape, generator=gen) * scale).to(dtype=torch.bfloat16) for k, v in m.state_dict().items()}

    with tempfile.TemporaryDirectory(prefix="remi-bench-ckpt-") as tmpdir:
        ckpt_a = Path(tmpdir) / "model_a.safetensors"
        ckpt_b = Path(tmpdir) / "model_b.safetensors"
        save_file(_build_state(0.02), str(ckpt_a))
        save_file(_build_state(0.03), str(ckpt_b))

        def _load_ckpt(path: Path) -> SyntheticDiT:
            sd = load_file(str(path), device="cpu")
            model = SyntheticDiT(dim=dim, num_heads=num_heads, num_blocks=num_blocks).eval()
            model.load_state_dict(sd, assign=True)
            model = model.to(device=device, dtype=dtype)
            sync_device(device)
            return model

        empty_device_cache(device)
        mem_start = sample_memory_mb(device)

        t0 = time.perf_counter()
        model_a = _load_ckpt(ckpt_a)
        cold_load_s = time.perf_counter() - t0
        mem_loaded = sample_memory_mb(device)

        # Unload A
        t0 = time.perf_counter()
        del model_a
        empty_device_cache(device)
        unload_s = time.perf_counter() - t0
        mem_unloaded = sample_memory_mb(device)

        # Warm reload A (OS page cache warm)
        t0 = time.perf_counter()
        model_a = _load_ckpt(ckpt_a)
        warm_load_s = time.perf_counter() - t0

        # Switch A -> B
        t0 = time.perf_counter()
        del model_a
        empty_device_cache(device)
        model_b = _load_ckpt(ckpt_b)
        switch_a_to_b_s = time.perf_counter() - t0

        # Switch B -> A
        t0 = time.perf_counter()
        del model_b
        empty_device_cache(device)
        model_a = _load_ckpt(ckpt_a)
        switch_b_to_a_s = time.perf_counter() - t0

        mem_peak_dev = max(mem_loaded["device_allocated_mb"], sample_memory_mb(device)["device_allocated_mb"])
        mem_peak_rss = max(mem_loaded["rss_mb"], sample_memory_mb(device)["rss_mb"])
        del model_a
        empty_device_cache(device)
        mem_final = sample_memory_mb(device)

    ckpt_size_mb = sum(p.numel() * 2 for p in SyntheticDiT(dim=dim, num_heads=num_heads, num_blocks=num_blocks).parameters()) / (1024 * 1024)
    return {
        "status": "ok",
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "cold_load_s": round(cold_load_s, 4),
        "warm_load_s": round(warm_load_s, 4),
        "switch_s": round((switch_a_to_b_s + switch_b_to_a_s) / 2.0, 4),
        "unload_s": round(unload_s, 4),
        "peak_device_mb": round(mem_peak_dev, 2),
        "post_unload_device_mb": round(mem_final["device_allocated_mb"], 2),
        "peak_rss_mb": round(mem_peak_rss, 2),
        "rss_reclaimed_mb": round(max(0.0, mem_loaded["rss_mb"] - mem_unloaded["rss_mb"]), 2),
        "rss_start_mb": round(mem_start["rss_mb"], 2),
    }


# endregion

# region Suite 3: Quantization Matrix (BF16 / FP8 / INT8-ConvRot / GGUF / SVDQuant / NVFP4)


def _measure_linear_callable(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    steps: int,
) -> tuple[torch.Tensor, list[float], dict[str, float]]:
    times_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(max(1, warmup)):
            out = fn()
        sync_device(device)
        reset_peak_memory(device)
        for _ in range(max(1, steps)):
            t0 = time.perf_counter()
            out = fn()
            sync_device(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    mem = sample_memory_mb(device)
    return out, times_ms, mem


def _build_mixed_precision_linear(
    weight_bf16: torch.Tensor,
    bias_bf16: torch.Tensor,
    quant_format: str,
    device: torch.device,
    *,
    full_precision_mm: bool,
    extra_conf: dict[str, Any] | None = None,
) -> tuple[nn.Module, int]:
    """Instantiate a real ``MixedPrecisionOps.Linear`` layer and return ``(module, storage_bytes)``."""
    from backend import operations  # noqa: F401 - initialize operations before operations_mixed_precision
    from backend.operations_mixed_precision import mixed_precision_ops
    from backend.quant_ops import RowWiseFP8E4M3Layout, TensorCoreFP8E4M3Layout, TensorCoreNVFP4Layout_, TensorWiseINT8Layout

    out_f, in_f = weight_bf16.shape
    ops = mixed_precision_ops(compute_dtype=weight_bf16.dtype, full_precision_mm=full_precision_mm, disabled=[])
    layer = ops.Linear(in_f, out_f, bias=True, device=device, dtype=weight_bf16.dtype)

    sd: dict[str, torch.Tensor] = {"bias": bias_bf16.to(device=device)}
    conf: dict[str, Any] = {"format": quant_format, **(extra_conf or {})}
    if full_precision_mm:
        conf["full_precision_matrix_mult"] = True

    w_cpu = weight_bf16.detach().float().cpu()
    if quant_format == "float8_e4m3fn":
        qdata, params = TensorCoreFP8E4M3Layout.quantize(w_cpu, scale="recalculate")
        sd["weight"] = qdata
        sd["weight_scale"] = params.scale
        storage_bytes = qdata.nbytes + params.scale.nbytes
    elif quant_format == "float8_e4m3fn_rowwise":
        qdata, params = RowWiseFP8E4M3Layout.quantize(w_cpu, scale="recalculate")
        sd["weight"] = qdata
        sd["weight_scale"] = params.scale
        storage_bytes = qdata.nbytes + params.scale.nbytes
    elif quant_format == "int8_tensorwise":
        convrot = bool((extra_conf or {}).get("convrot", False))
        gs = int((extra_conf or {}).get("convrot_groupsize", 256))
        qdata, params = TensorWiseINT8Layout.quantize(w_cpu, is_weight=True, per_channel=True, convrot=convrot, convrot_groupsize=gs)
        sd["weight"] = qdata
        sd["weight_scale"] = params.scale.float()
        storage_bytes = qdata.nbytes + params.scale.nbytes
    elif quant_format == "nvfp4":
        qdata, params = TensorCoreNVFP4Layout_.quantize(w_cpu, scale="recalculate")
        sd["weight"] = qdata
        sd["weight_scale"] = params.block_scale.view(torch.uint8)
        sd["weight_scale_2"] = params.scale
        storage_bytes = qdata.nbytes + params.block_scale.nbytes + params.scale.nbytes
    else:
        raise ValueError(f"Unsupported synthetic quant format: {quant_format}")

    sd["comfy_quant"] = torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)
    layer.load_state_dict(sd)
    layer = layer.to(device=device)
    return layer, storage_bytes


def run_synthetic_quantization_suite(
    device: torch.device,
    *,
    formats: tuple[str, ...] = ALL_QUANT_FORMATS,
    in_features: int = 1024,
    out_features: int = 1024,
    tokens: int = 512,
    steps: int = 16,
    warmup: int = 3,
    seed: int = 12345,
) -> dict[str, Any]:
    """Benchmark BF16, FP8, RowWise FP8, INT8-ConvRot, GGUF Q8_0, SVDQuant W4A4, and NVFP4."""
    from backend import memory_management, operations  # noqa: F401 - initialize operations before operations_mixed_precision
    from backend.operations_mixed_precision import clear_mps_dequant_cache, mps_dequant_cache_stats

    compute_dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Realistic DiT Linear weight distribution with mild channel outliers
    w_ref_fp32 = torch.randn(out_features, in_features, generator=gen) * 0.02
    w_ref_fp32[:, ::64] *= 2.5
    b_ref_fp32 = torch.randn(out_features, generator=gen) * 0.01
    x_cpu = torch.randn(1, tokens, in_features, generator=gen) * 0.25

    w_bf16 = w_ref_fp32.to(dtype=torch.bfloat16)
    b_bf16 = b_ref_fp32.to(dtype=torch.bfloat16)
    x_dev = x_cpu.to(device=device, dtype=compute_dtype)

    # BF16 reference (cast to compute_dtype on CPU where bfloat16 F.linear can be slower)
    w_base_dev = w_bf16.to(device=device, dtype=compute_dtype)
    b_base_dev = b_bf16.to(device=device, dtype=compute_dtype)
    bf16_storage_bytes = w_bf16.nbytes
    ref_out, bf16_times, bf16_mem = _measure_linear_callable(
        lambda: F.linear(x_dev, w_base_dev, b_base_dev),
        device,
        warmup=warmup,
        steps=steps,
    )
    bf16_med_ms = statistics.median(bf16_times)
    results: dict[str, Any] = {}

    if "bf16" in formats:
        results["bf16"] = {
            "status": "ok",
            "storage_mb": round(bf16_storage_bytes / (1024 * 1024), 3),
            "compression_vs_bf16": 1.0,
            "ms_per_step_median": round(bf16_med_ms, 4),
            "ms_per_step_p95": round(percentile(bf16_times, 95.0), 4),
            "speedup_vs_bf16": 1.0,
            "peak_device_mb": bf16_mem.get("device_peak_mb", bf16_mem["device_allocated_mb"]),
            "peak_rss_mb": bf16_mem["rss_mb"],
            **compute_tensor_fidelity(ref_out, ref_out),
        }

    # Mixed-precision formats via backend.operations_mixed_precision
    mp_specs = {
        "fp8_e4m3": ("float8_e4m3fn", not memory_management.supports_fp8_compute(device), {}),
        "fp8_rowwise": ("float8_e4m3fn_rowwise", not memory_management.supports_fp8_compute(device), {}),
        "int8_convrot": ("int8_tensorwise", not memory_management.supports_int8_compute(device), {"convrot": True, "convrot_groupsize": 256}),
        "nvfp4": ("nvfp4", not memory_management.supports_nvfp4_compute(device), {}),
    }

    for fmt in formats:
        if fmt == "bf16":
            continue
        if fmt in mp_specs:
            qfmt, full_mm, extra = mp_specs[fmt]
            try:
                clear_mps_dequant_cache()
                layer, storage_bytes = _build_mixed_precision_linear(
                    w_bf16,
                    b_bf16,
                    qfmt,
                    device,
                    full_precision_mm=full_mm,
                    extra_conf=extra,
                )
                # First measure steady-state (with MPS cache enabled when on MPS)
                from backend.operations_mixed_precision import _mps_dequant_cache

                _mps_dequant_cache.reset_stats()
                out, times, mem = _measure_linear_callable(lambda _l=layer: _l(x_dev), device, warmup=warmup, steps=steps)
                med_ms = statistics.median(times)
                cache_stats = mps_dequant_cache_stats()
                entry: dict[str, Any] = {
                    "status": "ok",
                    "native_gemm": not full_mm,
                    "storage_mb": round(storage_bytes / (1024 * 1024), 3),
                    "compression_vs_bf16": round(bf16_storage_bytes / max(storage_bytes, 1), 2),
                    "ms_per_step_median": round(med_ms, 4),
                    "ms_per_step_p95": round(percentile(times, 95.0), 4),
                    "speedup_vs_bf16": round(bf16_med_ms / max(med_ms, 1e-6), 3),
                    "peak_device_mb": mem.get("device_peak_mb", mem["device_allocated_mb"]),
                    "peak_rss_mb": mem["rss_mb"],
                    **compute_tensor_fidelity(out, ref_out),
                }
                if device.type == "mps":
                    total_lookups = cache_stats["hits"] + cache_stats["misses"]
                    entry["mps_cache_hit_rate"] = round(cache_stats["hits"] / max(total_lookups, 1), 4)
                    # Also measure uncached baseline on MPS to quantify cache speedup directly
                    prev_env = os.environ.get("REMI_MPS_DEQUANT_CACHE_MB")
                    os.environ["REMI_MPS_DEQUANT_CACHE_MB"] = "0"
                    clear_mps_dequant_cache()
                    try:
                        _, uncached_times, _ = _measure_linear_callable(lambda _l=layer: _l(x_dev), device, warmup=1, steps=max(4, steps // 2))
                        uncached_med = statistics.median(uncached_times)
                        entry["mps_uncached_ms_median"] = round(uncached_med, 4)
                        entry["mps_cache_speedup"] = round(uncached_med / max(med_ms, 1e-6), 3)
                    finally:
                        if prev_env is None:
                            os.environ.pop("REMI_MPS_DEQUANT_CACHE_MB", None)
                        else:
                            os.environ["REMI_MPS_DEQUANT_CACHE_MB"] = prev_env
                results[fmt] = entry
                del layer
                clear_mps_dequant_cache()
            except Exception as exc:
                results[fmt] = {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
        elif fmt == "gguf_q8_0":
            try:
                import gguf

                from backend.operations_gguf import ParameterGGUF, dequantize_tensor

                q_np = gguf.quants.quantize(w_ref_fp32.numpy(), gguf.GGMLQuantizationType.Q8_0)
                q_t = ParameterGGUF(
                    torch.from_numpy(np.array(q_np, copy=True)).to(device=device),
                    tensor_type=gguf.GGMLQuantizationType.Q8_0,
                    tensor_shape=torch.Size((out_features, in_features)),
                )
                q_t.computation_dtype = torch.float16
                if q_t.gguf_cls is not None:
                    q_t.gguf_cls.bake(q_t)
                storage_bytes = q_t.data.nbytes

                def _gguf_run(_q=q_t) -> torch.Tensor:
                    w_dq = dequantize_tensor(_q).to(dtype=compute_dtype)
                    return F.linear(x_dev, w_dq, b_base_dev)

                out, times, mem = _measure_linear_callable(_gguf_run, device, warmup=warmup, steps=steps)
                med_ms = statistics.median(times)
                results[fmt] = {
                    "status": "ok",
                    "native_gemm": False,
                    "storage_mb": round(storage_bytes / (1024 * 1024), 3),
                    "compression_vs_bf16": round(bf16_storage_bytes / max(storage_bytes, 1), 2),
                    "ms_per_step_median": round(med_ms, 4),
                    "ms_per_step_p95": round(percentile(times, 95.0), 4),
                    "speedup_vs_bf16": round(bf16_med_ms / max(med_ms, 1e-6), 3),
                    "peak_device_mb": mem.get("device_peak_mb", mem["device_allocated_mb"]),
                    "peak_rss_mb": mem["rss_mb"],
                    **compute_tensor_fidelity(out, ref_out),
                }
                del q_t
            except Exception as exc:
                results[fmt] = {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
        elif fmt == "svdquant_w4a4":
            try:
                from backend.quant_ops import TensorCoreConvRotW4A4Layout, ck

                if hasattr(ck, "quantize_svdquant_w4a4") and hasattr(ck, "scaled_mm_svdquant_w4a4") and device.type == "cuda":
                    # SVDQuant W4A4 uses FP4 + low-rank branch on CUDA Blackwell/Ada kernels
                    raise RuntimeError("SVDQuant W4A4 benchmark requires CUDA checkpoint calibration factors")
                # Fallback to ConvRot W4A4 quantized layout available across CPU/MPS/CUDA
                qdata, params = TensorCoreConvRotW4A4Layout.quantize(w_ref_fp32.cpu(), convrot_groupsize=256)
                qdata = qdata.to(device=device)
                scale = params.scale.to(device=device)
                params_dev = TensorCoreConvRotW4A4Layout.Params(
                    scale=scale,
                    orig_dtype=compute_dtype,
                    orig_shape=(out_features, in_features),
                    convrot_groupsize=params.convrot_groupsize,
                    quant_group_size=params.quant_group_size,
                    linear_dtype=params.linear_dtype,
                )
                storage_bytes = qdata.nbytes + scale.nbytes

                def _w4a4_run(_qd=qdata, _pd=params_dev) -> torch.Tensor:
                    w_dq = TensorCoreConvRotW4A4Layout.dequantize(_qd, _pd).to(dtype=compute_dtype)
                    return F.linear(x_dev, w_dq, b_base_dev)

                out, times, mem = _measure_linear_callable(_w4a4_run, device, warmup=warmup, steps=max(4, steps // 2))
                med_ms = statistics.median(times)
                results[fmt] = {
                    "status": "ok",
                    "native_gemm": False,
                    "layout": "TensorCoreConvRotW4A4Layout",
                    "storage_mb": round(storage_bytes / (1024 * 1024), 3),
                    "compression_vs_bf16": round(bf16_storage_bytes / max(storage_bytes, 1), 2),
                    "ms_per_step_median": round(med_ms, 4),
                    "ms_per_step_p95": round(percentile(times, 95.0), 4),
                    "speedup_vs_bf16": round(bf16_med_ms / max(med_ms, 1e-6), 3),
                    "peak_device_mb": mem.get("device_peak_mb", mem["device_allocated_mb"]),
                    "peak_rss_mb": mem["rss_mb"],
                    **compute_tensor_fidelity(out, ref_out),
                }
                del qdata, scale
            except Exception as exc:
                results[fmt] = {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
        else:
            results[fmt] = {"status": "skipped", "reason": f"Unknown format {fmt!r}"}

    empty_device_cache(device)
    return {
        "status": "ok",
        "shape": {"tokens": tokens, "in_features": in_features, "out_features": out_features},
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "formats": results,
    }


# endregion

# region Suite 4: Attention & Acceleration Backends (SDPA / Sage / Flash / Compile / Spectrum)


def _chebyshev_forecast(history_steps: list[float], history_feats: list[torch.Tensor], target_step: float, degree: int = 2, lam: float = 1e-3) -> torch.Tensor:
    """Lightweight Chebyshev ridge predictor mirroring Remi's Spectrum SEA feature forecaster."""
    k = len(history_steps)
    m = min(degree, k - 1)
    s_min, s_max = min(history_steps), max(history_steps)
    span = max(s_max - s_min, 1e-6)

    def _norm(s: float) -> float:
        return 2.0 * (s - s_min) / span - 1.0

    def _cheb_row(u: float) -> list[float]:
        row = [1.0]
        if m >= 1:
            row.append(u)
        for _ in range(2, m + 1):
            row.append(2.0 * u * row[-1] - row[-2])
        return row

    design = torch.tensor([_cheb_row(_norm(s)) for s in history_steps], dtype=torch.float64)
    target_vec = torch.tensor(_cheb_row(_norm(target_step)), dtype=torch.float64)
    gram = design.T @ design + lam * torch.eye(m + 1, dtype=torch.float64)
    weights = torch.linalg.solve(gram, design.T)  # (m+1, k)
    coeffs = (target_vec @ weights).to(dtype=torch.float32)  # (k,)
    out = torch.zeros_like(history_feats[0], dtype=torch.float32)
    for idx, feat in enumerate(history_feats):
        out = out + float(coeffs[idx].item()) * feat.float()
    return out.to(dtype=history_feats[0].dtype)


def run_synthetic_optimizations_suite(
    device: torch.device,
    *,
    tokens: int = 512,
    heads: int = 8,
    head_dim: int = 64,
    steps: int = 12,
    warmup: int = 2,
    seed: int = 12345,
) -> dict[str, Any]:
    """Benchmark attention backends (Basic / SDPA / Sage / Flash / xFormers), per-block compile, and Spectrum."""
    from backend import attention as attn_mod
    from backend import memory_management

    dtype = _preferred_compute_dtype(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dim = heads * head_dim

    q_fp32 = torch.randn(1, tokens, dim, generator=gen) * 0.1
    k_fp32 = torch.randn(1, tokens, dim, generator=gen) * 0.1
    v_fp32 = torch.randn(1, tokens, dim, generator=gen) * 0.1

    q = q_fp32.to(device=device, dtype=dtype)
    k = k_fp32.to(device=device, dtype=dtype)
    v = v_fp32.to(device=device, dtype=dtype)

    # Exact float32 reference
    with torch.inference_mode():
        qh_ref = q_fp32.view(1, tokens, heads, head_dim).transpose(1, 2)
        kh_ref = k_fp32.view(1, tokens, heads, head_dim).transpose(1, 2)
        vh_ref = v_fp32.view(1, tokens, heads, head_dim).transpose(1, 2)
        ref_attn = F.scaled_dot_product_attention(qh_ref, kh_ref, vh_ref).transpose(1, 2).reshape(1, tokens, dim)

    attention_results: dict[str, Any] = {}
    basic_med_ms: float | None = None

    # 1. Basic attention
    out_basic, times_basic, mem_basic = _measure_linear_callable(
        lambda _q=q, _k=k, _v=v: attn_mod.attention_basic(_q, _k, _v, heads),
        device,
        warmup=warmup,
        steps=steps,
    )
    basic_med_ms = statistics.median(times_basic)
    attention_results["basic"] = {
        "status": "ok",
        "ms_per_call_median": round(basic_med_ms, 4),
        "ms_per_call_p95": round(percentile(times_basic, 95.0), 4),
        "speedup_vs_basic": 1.0,
        "peak_device_mb": mem_basic.get("device_peak_mb", mem_basic["device_allocated_mb"]),
        **compute_tensor_fidelity(out_basic, ref_attn),
    }

    # 2. PyTorch SDPA
    out_sdpa, times_sdpa, mem_sdpa = _measure_linear_callable(
        lambda _q=q, _k=k, _v=v: attn_mod.attention_pytorch(_q, _k, _v, heads),
        device,
        warmup=warmup,
        steps=steps,
    )
    sdpa_med = statistics.median(times_sdpa)
    attention_results["sdpa"] = {
        "status": "ok",
        "ms_per_call_median": round(sdpa_med, 4),
        "ms_per_call_p95": round(percentile(times_sdpa, 95.0), 4),
        "speedup_vs_basic": round(basic_med_ms / max(sdpa_med, 1e-6), 3),
        "peak_device_mb": mem_sdpa.get("device_peak_mb", mem_sdpa["device_allocated_mb"]),
        **compute_tensor_fidelity(out_sdpa, ref_attn),
    }

    # 3. Optional CUDA attention kernels (Sage / Flash / xFormers)
    optional_attn = (
        ("sage", memory_management.sage_enabled, "attention_sage", "Requires NVIDIA CUDA + sageattention"),
        ("flash", memory_management.flash_enabled, "attention_flash", "Requires NVIDIA CUDA + flash_attn"),
        ("xformers", memory_management.xformers_enabled, "attention_xformers", "Requires GPU + xformers"),
    )
    for name, check_fn, fn_name, reason in optional_attn:
        if device.type == "cuda" and check_fn():
            try:
                fn = getattr(attn_mod, fn_name)
                out_opt, times_opt, mem_opt = _measure_linear_callable(lambda _fn=fn, _q=q, _k=k, _v=v: _fn(_q, _k, _v, heads), device, warmup=warmup, steps=steps)
                opt_med = statistics.median(times_opt)
                attention_results[name] = {
                    "status": "ok",
                    "ms_per_call_median": round(opt_med, 4),
                    "ms_per_call_p95": round(percentile(times_opt, 95.0), 4),
                    "speedup_vs_basic": round(basic_med_ms / max(opt_med, 1e-6), 3),
                    "peak_device_mb": mem_opt.get("device_peak_mb", mem_opt["device_allocated_mb"]),
                    **compute_tensor_fidelity(out_opt, ref_attn),
                }
            except Exception as exc:
                attention_results[name] = {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
        else:
            attention_results[name] = {"status": "unsupported", "reason": reason}

    # Execution optimizations: eager vs per-block compile vs Spectrum forecasting
    accel_results: dict[str, Any] = {}
    dit = SyntheticDiT(dim=256, num_heads=4, num_blocks=4).eval()
    with torch.no_grad():
        for p in dit.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    dit = dit.to(device=device, dtype=dtype)
    x_dit = (torch.randn(1, min(tokens, 256), 256, generator=gen) * 0.1).to(device=device, dtype=dtype)

    # Eager multi-step trajectory
    def _run_trajectory(model_obj: SyntheticDiT, total_steps: int, x_init: torch.Tensor = x_dit) -> torch.Tensor:
        h = x_init.clone()
        for step in range(total_steps):
            h = h - 0.05 * model_obj(h, timestep=1.0 - step / max(1, total_steps))
        return h

    traj_steps = max(8, steps)
    with torch.inference_mode():
        _run_trajectory(dit, 2)
        sync_device(device)
        t0 = time.perf_counter()
        eager_out = _run_trajectory(dit, traj_steps)
        sync_device(device)
        eager_total_ms = (time.perf_counter() - t0) * 1000.0

    accel_results["eager_baseline"] = {
        "status": "ok",
        "steps": traj_steps,
        "total_ms": round(eager_total_ms, 3),
        "ms_per_step": round(eager_total_ms / traj_steps, 3),
        "speedup_vs_eager": 1.0,
        **compute_tensor_fidelity(eager_out, eager_out),
    }

    # Per-block torch.compile (only when CUDA Inductor or supported backend is present; skipped on CPU/MPS if inductor unsupported)
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            compiled = SyntheticDiT(dim=256, num_heads=4, num_blocks=4).eval().to(device=device, dtype=dtype)
            compiled.load_state_dict(dit.state_dict())
            t_compile_start = time.perf_counter()
            for i in range(len(compiled.blocks)):
                compiled.blocks[i] = torch.compile(compiled.blocks[i], mode="default")
            with torch.inference_mode():
                _run_trajectory(compiled, 2)
                sync_device(device)
                compile_warmup_s = time.perf_counter() - t_compile_start
                t0 = time.perf_counter()
                comp_out = _run_trajectory(compiled, traj_steps)
                sync_device(device)
                comp_total_ms = (time.perf_counter() - t0) * 1000.0
            accel_results["per_block_compile"] = {
                "status": "ok",
                "steps": traj_steps,
                "compile_warmup_s": round(compile_warmup_s, 3),
                "total_ms": round(comp_total_ms, 3),
                "ms_per_step": round(comp_total_ms / traj_steps, 3),
                "speedup_vs_eager": round(eager_total_ms / max(comp_total_ms, 1e-6), 3),
                **compute_tensor_fidelity(comp_out, eager_out),
            }
            del compiled
        except Exception as exc:
            accel_results["per_block_compile"] = {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
    else:
        accel_results["per_block_compile"] = {
            "status": "unsupported",
            "reason": f"Per-block Inductor torch.compile is disabled on {device.type} in synthetic mode (CUDA Inductor target)",
        }

    # Spectrum pre-head feature forecasting (warmup 4 steps, forecast every 2nd step thereafter)
    with torch.inference_mode():
        t0 = time.perf_counter()
        h = x_dit.clone()
        hist_t: list[float] = []
        hist_f: list[torch.Tensor] = []
        real_forwards = 0
        forecast_steps = 0
        for step in range(traj_steps):
            ts = 1.0 - step / max(1, traj_steps)
            if step < 4 or step % 2 == 0:
                feat = dit.forward_features(h, ts)
                hist_t.append(float(step))
                hist_f.append(feat)
                if len(hist_t) > 6:
                    hist_t.pop(0)
                    hist_f.pop(0)
                real_forwards += 1
            else:
                feat = _chebyshev_forecast(hist_t, hist_f, float(step), degree=2)
                forecast_steps += 1
            pred = dit.forward_head(feat)
            h = h - 0.05 * pred
        sync_device(device)
        spec_total_ms = (time.perf_counter() - t0) * 1000.0

    accel_results["spectrum_forecast"] = {
        "status": "ok",
        "steps": traj_steps,
        "real_forwards": real_forwards,
        "forecast_steps": forecast_steps,
        "total_ms": round(spec_total_ms, 3),
        "ms_per_step": round(spec_total_ms / traj_steps, 3),
        "speedup_vs_eager": round(eager_total_ms / max(spec_total_ms, 1e-6), 3),
        **compute_tensor_fidelity(h, eager_out),
    }

    del dit, q, k, v, x_dit
    empty_device_cache(device)
    return {
        "status": "ok",
        "attention_shape": {"tokens": tokens, "heads": heads, "head_dim": head_dim},
        "attention": attention_results,
        "accelerators": accel_results,
    }


# endregion

# region Suite 5: 100-Job Queue Soak Test


class _FakeState:
    """Minimal ``shared.state`` stand-in for deterministic queue soak verification."""

    def __init__(self) -> None:
        self.interrupted = False
        self.skipped = False
        self.job_count = 1
        self.job_no = 0
        self.sampling_step = 0
        self.sampling_steps = 1

    def begin(self, job: str = "") -> None:
        self.interrupted = False
        self.skipped = False

    def end(self) -> None:
        pass

    def interrupt(self) -> None:
        self.interrupted = True

    def skip(self) -> None:
        self.skipped = True


def _open_fd_count() -> int:
    try:
        return psutil.Process(os.getpid()).num_fds()
    except Exception:
        return -1


def run_synthetic_soak_suite(
    device: torch.device,
    *,
    num_jobs: int = 100,
    seed: int = 12345,
) -> dict[str, Any]:
    """Run a mixed-workload queue soak test through Remi's real SQLite ``JobStore`` / ``JobQueue``.

    Exercises txt2img, img2img, simulated model switches, mid-run interrupts, queued cancels,
    and transient OOM/error recovery while tracking RSS slope, device memory stability,
    file descriptor counts, and WAL size with zero leaked connections.
    """
    from PIL import Image

    from remi.files import FileStore
    from remi.jobs.queue import JobQueue, Submission
    from remi.jobs.runner import JobResult
    from remi.jobs.store import JobStore
    from remi.realtime import EventBus

    num_jobs = max(4, int(num_jobs))
    empty_device_cache(device)
    state = _FakeState()
    lock = threading.Lock()
    rss_series: list[float] = []
    dev_series: list[float] = []
    fd_start = _open_fd_count()
    mem_start = sample_memory_mb(device)

    rng = np.random.default_rng(seed)
    active_model = {"name": "model_a"}
    model_switches = 0
    executed_count = 0

    with tempfile.TemporaryDirectory(prefix="remi-bench-soak-") as tmpdir:
        tmp = Path(tmpdir)
        store: JobStore | None = None
        files: FileStore | None = None
        queue: JobQueue | None = None
        try:
            store = JobStore(tmp / "jobs.db")
            files = FileStore(tmp)
            bus = EventBus()
            def executor(job, ctx):
                nonlocal model_switches, executed_count
                executed_count += 1
                want_model = job.core.get("_bench_model", "model_a")
                if want_model != active_model["name"]:
                    active_model["name"] = want_model
                    model_switches += 1
                    # Simulate a small tensor allocation + release across switch
                    buf = torch.empty((128, 128), device=device, dtype=torch.float32)
                    del buf

                action = job.core.get("_bench_action", "normal")
                if action == "fail":
                    raise RuntimeError("Simulated transient OOM / execution failure")
                if action == "interrupt":
                    if ctx.on_begin is not None:
                        ctx.on_begin()
                    assert queue is not None
                    queue.interrupt(job.id)
                    return JobResult(images=[], info="interrupted")

                # Small tensor work on the target device
                t = torch.randn((64, 64), device=device, dtype=torch.float32)
                y = (t @ t.T).mean()
                _ = float(y.item())
                del t, y

                arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(arr, mode="RGB")
                info = f"{job.core.get('prompt', '')}\nSteps: 4, Model: {active_model['name']}"
                return JobResult(images=[img], infotexts=[info], info=info)

            queue = JobQueue(
                store,
                files,
                bus,
                executor=executor,
                dispatch=lambda fn, *a, **kw: fn(*a, **kw),
                lock=lock,
                validate=lambda tab, core, scripts: None,
                state=lambda: state,
                hash_layout=lambda tab: "bench-v1",
                option=lambda name, default: default,
                set_option=lambda name, value: None,
                when_done=lambda action: None,
            )

            import logging

            q_logger = logging.getLogger("remi.jobs.queue")
            prev_level = q_logger.level
            q_logger.setLevel(logging.CRITICAL)
            try:
                t0 = time.perf_counter()

                job_ids: list[str] = []
                cancel_targets: list[str] = []
                for idx in range(num_jobs):
                    tab = "img2img" if idx % 5 == 1 else "txt2img"
                    model_name = "model_b" if (idx // 10) % 2 == 1 else "model_a"
                    action = "normal"
                    if idx > 0 and idx % 25 == 0:
                        action = "fail"
                    elif idx > 0 and idx % 20 == 0:
                        action = "interrupt"
                    created = queue.submit(
                        Submission(
                            tab=tab,
                            core={
                                "prompt": f"soak bench job #{idx}",
                                "_bench_model": model_name,
                                "_bench_action": action,
                            },
                            scripts={},
                        )
                    )
                    jid = created[0].id
                    job_ids.append(jid)
                    if idx > 5 and idx % 33 == 0:
                        cancel_targets.append(jid)

                # Cancel a small subset while queued to exercise cancel cleanup paths
                for jid in cancel_targets:
                    try:
                        queue.interrupt(jid)
                    except Exception:
                        pass
                queue.start()

                # Wait for all jobs to settle and sample memory periodically
                deadline = time.monotonic() + max(60.0, num_jobs * 1.5)
                last_done = -1
                while time.monotonic() < deadline:
                    done = sum(store.count(st) for st in ("completed", "failed", "cancelled", "interrupted"))
                    if done != last_done and (done % max(1, num_jobs // 20) == 0 or done == num_jobs):
                        snap = sample_memory_mb(device)
                        rss_series.append(snap["rss_mb"])
                        dev_series.append(snap["device_allocated_mb"])
                        last_done = done
                    if done >= num_jobs:
                        break
                    time.sleep(0.01)

                elapsed_s = time.perf_counter() - t0
                status_counts = {st: store.count(st) for st in ("queued", "running", "completed", "failed", "cancelled", "interrupted")}
                wal_path = tmp / "jobs.db-wal"
                wal_size_kb = round(wal_path.stat().st_size / 1024.0, 2) if wal_path.is_file() else 0.0
            finally:
                q_logger.setLevel(prev_level)
        finally:
            if queue is not None:
                queue.stop()
            if store is not None:
                store.close()
            if files is not None:
                files.close()

    empty_device_cache(device)
    mem_end = sample_memory_mb(device)
    fd_end = _open_fd_count()

    # Compute RSS trend per 100 jobs over the steady window (after first sample)
    steady_rss = rss_series[1:] if len(rss_series) >= 3 else rss_series
    xs = [float(i) * (num_jobs / max(len(steady_rss), 1)) for i in range(len(steady_rss))]
    rss_slope_per_100 = linear_slope(xs, steady_rss) * 100.0
    dev_delta_mb = mem_end["device_allocated_mb"] - mem_start["device_allocated_mb"]
    fd_delta = (fd_end - fd_start) if (fd_start >= 0 and fd_end >= 0) else 0

    all_settled = status_counts["queued"] == 0 and status_counts["running"] == 0
    leak_detected = (not all_settled) or (rss_slope_per_100 > 50.0) or (dev_delta_mb > 32.0) or (fd_delta > 16)

    return {
        "status": "ok" if (all_settled and not leak_detected) else "warn",
        "num_jobs": num_jobs,
        "elapsed_s": round(elapsed_s, 3),
        "jobs_per_s": round(num_jobs / max(elapsed_s, 1e-6), 2),
        "model_switches": model_switches,
        "status_counts": status_counts,
        "rss_start_mb": round(mem_start["rss_mb"], 2),
        "rss_peak_mb": round(max(rss_series) if rss_series else mem_end["rss_mb"], 2),
        "rss_end_mb": round(mem_end["rss_mb"], 2),
        "rss_slope_mb_per_100_jobs": round(rss_slope_per_100, 3),
        "device_mem_start_mb": round(mem_start["device_allocated_mb"], 2),
        "device_mem_peak_mb": round(max(dev_series) if dev_series else mem_end["device_allocated_mb"], 2),
        "device_mem_end_mb": round(mem_end["device_allocated_mb"], 2),
        "fd_start": fd_start,
        "fd_end": fd_end,
        "fd_delta": fd_delta,
        "wal_size_kb": wal_size_kb,
        "leak_detected": leak_detected,
    }


# endregion

# region Real-Checkpoint HTTP Benchmark Harness (Remi / Forge Neo / ComfyUI)


def _http_json(method: str, url: str, body: Any = None, timeout: float = 600.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("content-type", "")
    except urllib.error.HTTPError as err:
        detail = err.read()[:1500].decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url}: HTTP {err.code}: {detail}") from None
    return json.loads(raw) if "json" in ctype else raw


def _decode_pil_image(payload: str | bytes):
    from PIL import Image

    raw = base64.b64decode(payload.split(",", 1)[-1]) if isinstance(payload, str) else payload
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


@dataclass
class RealBenchConfig:
    target: str
    base_url: str
    checkpoint: str
    second_checkpoint: str | None = None
    prompt: str = "masterpiece, best quality, 1girl, silver hair, blue eyes, cherry blossoms"
    negative_prompt: str = "worst quality, low quality"
    width: int = 768
    height: int = 768
    steps: int = 8
    cfg_scale: float = 1.0
    distilled_cfg_scale: float = 3.5
    sampler: str = "Euler"
    scheduler: str = "Simple"
    seed: int = 12345
    warmup_runs: int = 1
    runs: int = 3
    comfy_workflow: Path | None = None
    reference_image: Path | None = None
    timeout: float = 900.0


def _wait_remi_job(base_url: str, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = _http_json("GET", f"{base_url}/api/v2/jobs/{job_id}", timeout=30.0)
        if job.get("status") not in ("queued", "running"):
            return job
        time.sleep(0.25)
    raise TimeoutError(f"Remi job {job_id} timed out after {timeout}s")


def _run_remi_once(cfg: RealBenchConfig) -> tuple[float, Any]:
    base = cfg.base_url.rstrip("/")
    body = {
        "tab": "txt2img",
        "core": {
            "prompt": cfg.prompt,
            "negative_prompt": cfg.negative_prompt,
            "width": cfg.width,
            "height": cfg.height,
            "cfg_scale": cfg.cfg_scale,
            "distilled_cfg_scale": cfg.distilled_cfg_scale,
            "batch_size": 1,
            "batch_count": 1,
        },
        "scripts": {
            "sampler": {"sampler_name": cfg.sampler, "scheduler": cfg.scheduler, "steps": cfg.steps},
            "seed": {"seed": cfg.seed},
        },
    }
    t0 = time.perf_counter()
    queued = _http_json("POST", f"{base}/api/v2/jobs", body, timeout=60.0)
    job = _wait_remi_job(base, queued["job_id"], cfg.timeout)
    elapsed = time.perf_counter() - t0
    if job.get("status") != "completed" or not job.get("outputs"):
        raise RuntimeError(f"Remi job failed: {job.get('error') or job.get('status')}")
    raw = _http_json("GET", f"{base}/api/v2/files/{job['outputs'][0]['id']}", timeout=60.0)
    return elapsed, _decode_pil_image(raw)


def _run_forge_neo_once(cfg: RealBenchConfig) -> tuple[float, Any]:
    base = cfg.base_url.rstrip("/")
    body = {
        "prompt": cfg.prompt,
        "negative_prompt": cfg.negative_prompt,
        "width": cfg.width,
        "height": cfg.height,
        "steps": cfg.steps,
        "cfg_scale": cfg.cfg_scale,
        "distilled_cfg_scale": cfg.distilled_cfg_scale,
        "sampler_name": cfg.sampler,
        "scheduler": cfg.scheduler,
        "seed": cfg.seed,
        "batch_size": 1,
        "n_iter": 1,
    }
    t0 = time.perf_counter()
    ans = _http_json("POST", f"{base}/sdapi/v1/txt2img", body, timeout=cfg.timeout)
    elapsed = time.perf_counter() - t0
    return elapsed, _decode_pil_image(ans["images"][0])


def _run_comfyui_once(cfg: RealBenchConfig) -> tuple[float, Any]:
    if cfg.comfy_workflow is None or not cfg.comfy_workflow.is_file():
        raise ValueError("--comfy-workflow must point to a valid ComfyUI API prompt JSON file for target='comfyui'")
    base = cfg.base_url.rstrip("/")
    prompt_graph = json.loads(cfg.comfy_workflow.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    queued = _http_json("POST", f"{base}/prompt", {"prompt": prompt_graph}, timeout=60.0)
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + cfg.timeout
    history: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        res = _http_json("GET", f"{base}/history/{prompt_id}", timeout=30.0)
        if prompt_id in res:
            history = res[prompt_id]
            break
        time.sleep(0.25)
    elapsed = time.perf_counter() - t0
    if history is None:
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {cfg.timeout}s")
    image = None
    for node_out in (history.get("outputs") or {}).values():
        images = node_out.get("images") or []
        if images:
            meta = images[0]
            qs = f"filename={meta['filename']}&subfolder={meta.get('subfolder', '')}&type={meta.get('type', 'output')}"
            raw = _http_json("GET", f"{base}/view?{qs}", timeout=60.0)
            image = _decode_pil_image(raw)
            break
    return elapsed, image


def run_real_checkpoint_benchmark(cfg: RealBenchConfig, device: torch.device) -> dict[str, Any]:
    """Execute standardized real-checkpoint load, switch, and steady-state runs against a live server."""
    base = cfg.base_url.rstrip("/")
    target = cfg.target.lower()
    suites: dict[str, Any] = {}

    # 1. Cold checkpoint load & optional A<->B switch
    cold_load_s: float | None = None
    switch_s: float | None = None
    if cfg.checkpoint and target in ("remi", "forge-neo"):
        opt_method, opt_path = ("PATCH", "/api/v2/options") if target == "remi" else ("POST", "/sdapi/v1/options")
        t0 = time.perf_counter()
        _http_json(opt_method, f"{base}{opt_path}", {"sd_model_checkpoint": cfg.checkpoint}, timeout=cfg.timeout)
        cold_load_s = time.perf_counter() - t0
        if cfg.second_checkpoint:
            t0 = time.perf_counter()
            _http_json(opt_method, f"{base}{opt_path}", {"sd_model_checkpoint": cfg.second_checkpoint}, timeout=cfg.timeout)
            sw1 = time.perf_counter() - t0
            t0 = time.perf_counter()
            _http_json(opt_method, f"{base}{opt_path}", {"sd_model_checkpoint": cfg.checkpoint}, timeout=cfg.timeout)
            sw2 = time.perf_counter() - t0
            switch_s = (sw1 + sw2) / 2.0

    if cold_load_s is not None:
        suites["model_lifecycle"] = {
            "status": "ok",
            "checkpoint": cfg.checkpoint,
            "second_checkpoint": cfg.second_checkpoint,
            "cold_load_s": round(cold_load_s, 3),
            "switch_s": round(switch_s, 3) if switch_s is not None else None,
        }

    # 2. Warmup + timed inference runs
    runner = {"remi": _run_remi_once, "forge-neo": _run_forge_neo_once, "comfyui": _run_comfyui_once}[target]
    for _ in range(max(0, cfg.warmup_runs)):
        runner(cfg)

    durations_s: list[float] = []
    last_img = None
    for _ in range(max(1, cfg.runs)):
        dur, img = runner(cfg)
        durations_s.append(dur)
        last_img = img

    med_s = statistics.median(durations_s)
    steady_s = statistics.median(durations_s[1:]) if len(durations_s) > 1 else durations_s[0]
    it_per_s = cfg.steps / med_s if med_s > 0 else 0.0
    inf_entry: dict[str, Any] = {
        "status": "ok",
        "target": target,
        "checkpoint": cfg.checkpoint,
        "resolution": f"{cfg.width}x{cfg.height}",
        "steps": cfg.steps,
        "runs": len(durations_s),
        "first_image_s": round(durations_s[0], 3),
        "steady_image_s": round(steady_s, 3),
        "ms_per_step_median": round((med_s / max(1, cfg.steps)) * 1000.0, 2),
        "ms_per_step_p95": round((percentile(durations_s, 95.0) / max(1, cfg.steps)) * 1000.0, 2),
        "it_per_s": round(it_per_s, 3),
    }

    if cfg.reference_image is not None and cfg.reference_image.is_file() and last_img is not None:
        from PIL import Image

        with Image.open(cfg.reference_image) as ref_img:
            ref_img.load()
            inf_entry["fidelity"] = compute_image_fidelity(last_img, ref_img)

    # Query remote server memory stats when exposed
    if target == "comfyui":
        try:
            stats = _http_json("GET", f"{base}/system_stats", timeout=15.0)
            devs = stats.get("devices") or []
            if devs:
                vram_total = devs[0].get("vram_total", 0)
                vram_free = devs[0].get("vram_free", 0)
                inf_entry["peak_device_mb"] = round(max(0, vram_total - vram_free) / (1024 * 1024), 1)
        except Exception:
            pass
    elif target in ("remi", "forge-neo"):
        try:
            mem_info = _http_json("GET", f"{base}/sdapi/v1/memory", timeout=15.0)
            ram = mem_info.get("ram") or {}
            if "used" in ram:
                inf_entry["peak_rss_mb"] = round(ram["used"] / (1024 * 1024), 1)
            cuda_m = mem_info.get("cuda") or {}
            alloc = (cuda_m.get("allocated") or {}).get("peak")
            if alloc:
                inf_entry["peak_device_mb"] = round(alloc / (1024 * 1024), 1)
        except Exception:
            pass

    suites["inference"] = inf_entry
    return suites


# endregion

# region Static Diagnosis of Historical smoke_real.py Pixel-Parity Drift


def diagnose_smoke_real_parity_drift(repo_root: Path = ROOT) -> dict[str, Any]:
    """Statically inspect the codebase for root causes of the historical ``tools/smoke_real.py``
    txt2img pixel-parity drift (mean 8.3677, max 247 reported in ``docs/upstream-sync-2026-11-22.md``).
    """
    attention_src = (repo_root / "backend" / "attention.py").read_text(encoding="utf-8", errors="replace")
    mem_src = (repo_root / "backend" / "memory_management.py").read_text(encoding="utf-8", errors="replace")
    wan_cache_path = repo_root / "backend" / "nn" / "_wan_cache.py"
    wan_cache_src = wan_cache_path.read_text(encoding="utf-8", errors="replace") if wan_cache_path.is_file() else ""
    fixture_path = repo_root / "tests" / "fixtures" / "parity" / "anima_turbo_txt2img_seed12345.json"
    fixture_meta = json.loads(fixture_path.read_text(encoding="utf-8")) if fixture_path.is_file() else {}
    sync_doc_path = repo_root / "docs" / "upstream-sync-2026-11-22.md"
    sync_doc = sync_doc_path.read_text(encoding="utf-8", errors="replace") if sync_doc_path.is_file() else ""

    has_return_in_get_attn_precision = bool(
        re.search(r"def get_attn_precision\([^)]*\).*?:\s+return\s+memory_management\.force_upcast_attention_dtype\(\)", attention_src, re.DOTALL)
    )
    upcasts_fp16_on_macos = bool(re.search(r"macos_version\s*>=\s*\(14,\s*5\)", mem_src))
    mps_uses_basic_attention = "elif is_device_mps" not in mem_src and "ENABLE_PYTORCH_ATTENTION = True" in mem_src
    has_wan_strip_apply = "def strip_apply(" in wan_cache_src and "STRIP_ELEMS" in wan_cache_src
    drift_match = re.search(r"mean\s+([\d.]+),\s*max\s+(\d+)", sync_doc)

    causes: list[dict[str, str]] = []
    if has_return_in_get_attn_precision and upcasts_fp16_on_macos:
        causes.append(
            {
                "rank": "primary",
                "commit": "b59500ed (merged 2026-11-18 in 27cbc152)",
                "file": "backend/attention.py:get_attn_precision",
                "mechanism": (
                    "Before commit b59500ed, get_attn_precision() lacked a `return` statement and always returned None. "
                    "On Apple Silicon MPS, ENABLE_PYTORCH_ATTENTION is False by default, so Anima DiT and Qwen3-0.6B run "
                    "through attention_basic() in float16. On macOS >= 14.5, force_upcast_attention_dtype() maps "
                    "{float16: float32}. Fixing the missing `return` activated fp16->fp32 upcast in Q@K^T and softmax "
                    "across all Qwen3-0.6B and 28 Anima DiT attention layers; across 8 Euler flow steps this deterministic "
                    "precision upgrade shifts pixels (mean ~8.37, max ~247) while staying bitwise deterministic run-to-run."
                ),
            }
        )
    if has_wan_strip_apply:
        causes.append(
            {
                "rank": "secondary",
                "commit": "865255e6 (merged 2026-11-18 in 27cbc152, ComfyUI PR #17270)",
                "file": "backend/nn/_wan_cache.py:strip_apply + backend/nn/wan_vae.py",
                "mechanism": (
                    "WanVAE (qwen_image_vae) switched residual/head convolutions to striped spatial tiling whenever "
                    "x.numel() > STRIP_ELEMS (2^24 = 16.78M). Decoding 768x768 reaches 1*96*1*768*768 = 56.6M elements, "
                    "triggering striped Conv3d evaluation whose floating-point reduction order differs slightly (~1-2 LSB) "
                    "from full-frame evaluation."
                ),
            }
        )

    return {
        "status": "diagnosed" if causes else "inconclusive",
        "fixture_baseline": {
            "file": str(fixture_path.relative_to(repo_root)) if fixture_path.is_file() else None,
            "resolution": f"{fixture_meta.get('width')}x{fixture_meta.get('height')}" if fixture_meta else None,
            "steps": fixture_meta.get("steps"),
            "seed": fixture_meta.get("seed"),
            "note": fixture_meta.get("note"),
        },
        "observed_drift": {
            "mean_rgb_diff": float(drift_match.group(1)) if drift_match else None,
            "max_rgb_diff": int(drift_match.group(2)) if drift_match else None,
            "deterministic_between_runs": "maximum difference 0" in sync_doc,
        },
        "static_checks": {
            "get_attn_precision_returns_upcast": has_return_in_get_attn_precision,
            "macos_14_5_fp16_upcast_enabled": upcasts_fp16_on_macos,
            "mps_defaults_to_basic_attention": mps_uses_basic_attention,
            "wan_vae_striped_tiling_present": has_wan_strip_apply,
        },
        "root_causes": causes,
        "recommended_action": (
            "Commit b59500ed is an intentional numerical accuracy fix (preventing fp16 attention overflow on macOS >= 14.5). "
            "Verify on an Apple Silicon host with Anima weights by toggling get_attn_precision(), then refresh "
            "tests/fixtures/parity/anima_turbo_txt2img_seed12345.png and track perceptual parity (PSNR/SSIM) alongside strict bitwise parity."
        ),
    }


# endregion

# region Comparison Table Generator (Remi vs Forge Neo vs ComfyUI)


def _fmt_num(val: Any, suffix: str = "", digits: int = 2) -> str:
    if val is None or isinstance(val, bool):
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.{digits}f}{suffix}"
    return str(val)


def _pct_delta(val: float | None, base: float | None, *, higher_is_better: bool = True) -> str:
    if val is None or base is None or base == 0:
        return "—"
    pct = ((val - base) / abs(base)) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def generate_comparison_markdown(reports: list[dict[str, Any]]) -> str:
    """Render standardized GitHub-Flavored Markdown comparison tables from 1..N benchmark JSON reports."""
    if not reports:
        return "_No benchmark reports provided._\n"

    lines: list[str] = ["# Inference & Memory Benchmark Comparison", ""]

    # 1. Environment & Compatibility Check
    lines.append("## 1. Environment Summary")
    lines.append("")
    lines.append("| System | Mode | Backend / Device | PyTorch | RAM (MB) | Revision |")
    lines.append("|---|---|---|---|---|---|")
    backends = set()
    resolutions = set()
    for rep in reports:
        sys_info = rep.get("system") or {}
        cfg = rep.get("config") or {}
        label = sys_info.get("system_label", "unknown")
        mode = rep.get("mode", "unknown")
        backend = f"{sys_info.get('backend', '?')} ({sys_info.get('device_name', '?')})"
        backends.add(sys_info.get("backend"))
        if cfg.get("resolution"):
            resolutions.add(cfg["resolution"])
        lines.append(
            f"| **{label}** | `{mode}` | {backend} | `{sys_info.get('torch_version', '?')}` | "
            f"{_fmt_num(sys_info.get('ram_total_mb'), '', 0)} | `{sys_info.get('git_revision') or 'n/a'}` |"
        )
    lines.append("")
    if len(backends) > 1 or len(resolutions) > 1:
        lines.append("> **Note:** Input reports span different hardware backends or resolutions; compare relative ratios carefully.")
        lines.append("")

    # 2. End-to-End Inference, Memory & Model Lifecycle
    inf_rows = [(rep.get("system", {}).get("system_label", f"run-{i}"), rep.get("suites") or {}) for i, rep in enumerate(reports)]
    if any("inference" in suites or "model_lifecycle" in suites for _, suites in inf_rows):
        lines.append("## 2. Inference, Memory & Model Lifecycle (Remi vs Forge Neo vs ComfyUI)")
        lines.append("")
        lines.append("| System | Cold Load (s) | Warm Switch (s) | First Image (s) | Steady Image (s) | Throughput (it/s) | Step p95 (ms) | Peak Device (MB) | Peak RSS (MB) | vs Baseline it/s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        base_ips = None
        for _, suites in inf_rows:
            ips = (suites.get("inference") or {}).get("it_per_s")
            if isinstance(ips, (int, float)) and ips > 0:
                base_ips = float(ips)
                break
        for label, suites in inf_rows:
            inf = suites.get("inference") or {}
            life = suites.get("model_lifecycle") or {}
            ips = inf.get("it_per_s")
            delta = "baseline" if (base_ips is not None and ips == base_ips) else _pct_delta(ips, base_ips)
            lines.append(
                f"| **{label}** | {_fmt_num(life.get('cold_load_s'), '', 3)} | {_fmt_num(life.get('switch_s'), '', 3)} | "
                f"{_fmt_num(inf.get('first_image_s'), '', 3)} | {_fmt_num(inf.get('steady_image_s'), '', 3)} | "
                f"{_fmt_num(ips, '', 2)} | {_fmt_num(inf.get('ms_per_step_p95'), '', 2)} | "
                f"{_fmt_num(inf.get('peak_device_mb'), '', 1)} | {_fmt_num(inf.get('peak_rss_mb'), '', 1)} | {delta} |"
            )
        lines.append("")

    # 3. Quantization Matrix (rendered per report containing quantization results)
    for label, suites in inf_rows:
        qsuite = suites.get("quantization")
        if not qsuite or not qsuite.get("formats"):
            continue
        lines.append(f"## 3. Quantization Matrix — `{label}`")
        lines.append("")
        lines.append("| Format | Status | Storage (MB) | Compression | Median (ms) | Speedup vs BF16 | Rel L2 Error | PSNR (dB) | MPS Cache Hit / Speedup |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for fmt, row in qsuite["formats"].items():
            if row.get("status") != "ok":
                lines.append(f"| `{fmt}` | `{row.get('status')}` | — | — | — | — | — | — | {row.get('reason', '')[:60]} |")
                continue
            mps_note = "—"
            if "mps_cache_hit_rate" in row:
                mps_note = f"{row['mps_cache_hit_rate'] * 100:.0f}% / {row.get('mps_cache_speedup', 1.0):.2f}x"
            lines.append(
                f"| `{fmt}` | ok | {_fmt_num(row.get('storage_mb'), '', 3)} | {_fmt_num(row.get('compression_vs_bf16'), 'x', 2)} | "
                f"{_fmt_num(row.get('ms_per_step_median'), '', 3)} | {_fmt_num(row.get('speedup_vs_bf16'), 'x', 2)} | "
                f"{_fmt_num(row.get('rel_l2_error'), '', 5)} | {_fmt_num(row.get('psnr_db'), '', 2)} | {mps_note} |"
            )
        lines.append("")

    # 4. Attention & Acceleration Backends
    for label, suites in inf_rows:
        osuite = suites.get("optimizations")
        if not osuite:
            continue
        lines.append(f"## 4. Attention & Acceleration Backends — `{label}`")
        lines.append("")
        if osuite.get("attention"):
            lines.append("| Attention Backend | Status | Median (ms) | p95 (ms) | Speedup vs Basic | Max Abs Diff vs FP32 |")
            lines.append("|---|---|---|---|---|---|")
            for name, row in osuite["attention"].items():
                if row.get("status") != "ok":
                    lines.append(f"| `{name}` | `{row.get('status')}` | — | — | — | {row.get('reason', '')[:60]} |")
                else:
                    lines.append(
                        f"| `{name}` | ok | {_fmt_num(row.get('ms_per_call_median'), '', 3)} | "
                        f"{_fmt_num(row.get('ms_per_call_p95'), '', 3)} | {_fmt_num(row.get('speedup_vs_basic'), 'x', 2)} | "
                        f"{_fmt_num(row.get('max_abs_diff'), '', 5)} |"
                    )
            lines.append("")
        if osuite.get("accelerators"):
            lines.append("| Execution Accelerator | Status | Total (ms) | ms/step | Speedup vs Eager | PSNR (dB) |")
            lines.append("|---|---|---|---|---|---|")
            for name, row in osuite["accelerators"].items():
                if row.get("status") != "ok":
                    lines.append(f"| `{name}` | `{row.get('status')}` | — | — | — | {row.get('reason', '')[:60]} |")
                else:
                    lines.append(
                        f"| `{name}` | ok | {_fmt_num(row.get('total_ms'), '', 2)} | "
                        f"{_fmt_num(row.get('ms_per_step'), '', 3)} | {_fmt_num(row.get('speedup_vs_eager'), 'x', 2)} | "
                        f"{_fmt_num(row.get('psnr_db'), '', 2)} |"
                    )
            lines.append("")

    # 5. Queue Soak Test Summary
    soak_rows = [(label, suites["soak"]) for label, suites in inf_rows if "soak" in suites]
    if soak_rows:
        lines.append("## 5. Queue Soak Stability Summary")
        lines.append("")
        lines.append("| System | Jobs | Completed / Int / Cancel / Fail | RSS Start → Peak → End (MB) | RSS Slope (MB/100 jobs) | FD Delta | Leak Detected |")
        lines.append("|---|---|---|---|---|---|---|")
        for label, soak in soak_rows:
            sc = soak.get("status_counts") or {}
            breakdown = f"{sc.get('completed', 0)} / {sc.get('interrupted', 0)} / {sc.get('cancelled', 0)} / {sc.get('failed', 0)}"
            rss_trip = f"{_fmt_num(soak.get('rss_start_mb'), '', 1)} → {_fmt_num(soak.get('rss_peak_mb'), '', 1)} → {_fmt_num(soak.get('rss_end_mb'), '', 1)}"
            lines.append(
                f"| **{label}** | {soak.get('num_jobs', 0)} | {breakdown} | {rss_trip} | "
                f"{_fmt_num(soak.get('rss_slope_mb_per_100_jobs'), '', 2)} | {soak.get('fd_delta', 0)} | "
                f"{'YES' if soak.get('leak_detected') else 'NO'} |"
            )
        lines.append("")

    return "\n".join(lines)


# endregion

# region CLI Entry Point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-platform inference, memory, quantization & soak benchmark harness for Forge Neo Remi.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="Run fast self-contained synthetic benchmark suites (CI/CPU/MPS/CUDA)")
    mode.add_argument("--real", action="store_true", help="Run real-checkpoint benchmarks against a live Remi, Forge Neo, or ComfyUI server")
    mode.add_argument("--compare", nargs="+", type=Path, metavar="JSON", help="Generate Markdown comparison tables from 1..N benchmark JSON files")
    mode.add_argument("--diagnose-parity", action="store_true", help="Statically diagnose the historical tools/smoke_real.py txt2img pixel-parity drift")

    parser.add_argument("--device", default="auto", help="Execution device: auto, cuda, mps, or cpu (default: auto)")
    parser.add_argument("--system", default="remi", help="System label recorded in JSON output (e.g. remi, forge-neo, comfyui)")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES), help=f"Comma-separated synthetic suites ({','.join(ALL_SUITES)})")
    parser.add_argument("--quant-formats", default=",".join(ALL_QUANT_FORMATS), help="Comma-separated quantization formats to benchmark")
    parser.add_argument("--steps", type=int, default=12, help="Sampling / micro-benchmark steps (default: 12)")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warm-up steps before timing (default: 2)")
    parser.add_argument("--runs", type=int, default=2, help="Timed benchmark repetitions (default: 2)")
    parser.add_argument("--tokens", type=int, default=256, help="Synthetic sequence length (default: 256)")
    parser.add_argument("--dim", type=int, default=256, help="Synthetic DiT hidden dimension (default: 256)")
    parser.add_argument("--linear-dim", type=int, default=1024, help="Quantization micro-benchmark Linear width (multiple of 256; default: 1024)")
    parser.add_argument("--soak-jobs", type=int, default=100, help="Number of jobs for the queue soak test (default: 100)")
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic RNG seed (default: 12345)")

    # Real-checkpoint server options
    parser.add_argument("--target", choices=("remi", "forge-neo", "comfyui"), default="remi", help="Server protocol for --real mode")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860", help="Base URL of the running server for --real mode")
    parser.add_argument("--checkpoint", default="", help="Primary checkpoint name for --real mode")
    parser.add_argument("--second-checkpoint", default=None, help="Second checkpoint name for A<->B switch timing in --real mode")
    parser.add_argument("--comfy-workflow", type=Path, default=None, help="ComfyUI API prompt JSON template for --target comfyui")
    parser.add_argument("--reference-image", type=Path, default=None, help="Optional reference PNG for PSNR/SSIM comparison in --real mode")
    parser.add_argument("--width", type=int, default=768, help="Generation width for --real mode (default: 768)")
    parser.add_argument("--height", type=int, default=768, help="Generation height for --real mode (default: 768)")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale for --real mode (default: 1.0)")
    parser.add_argument("--sampler", default="Euler", help="Sampler name for --real mode")
    parser.add_argument("--scheduler", default="Simple", help="Scheduler name for --real mode")

    parser.add_argument("--out", type=Path, default=None, help="Write standardized JSON report to this path")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Write rendered Markdown summary/comparison table to this path")
    return parser


def run_synthetic_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    selected = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    unknown = [s for s in selected if s not in ALL_SUITES]
    if unknown:
        raise ValueError(f"Unknown benchmark suite(s): {', '.join(unknown)}")
    quant_formats = tuple(q.strip() for q in args.quant_formats.split(",") if q.strip())

    suites: dict[str, Any] = {}
    if "inference" in selected:
        suites["inference"] = run_synthetic_inference_suite(
            device,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            runs=args.runs,
            tokens=args.tokens,
            dim=args.dim,
            seed=args.seed,
        )
    if "model_lifecycle" in selected:
        suites["model_lifecycle"] = run_synthetic_lifecycle_suite(
            device,
            dim=args.dim,
            seed=args.seed,
        )
    if "quantization" in selected:
        lin_dim = max(256, (args.linear_dim // 256) * 256)
        suites["quantization"] = run_synthetic_quantization_suite(
            device,
            formats=quant_formats,
            in_features=lin_dim,
            out_features=lin_dim,
            tokens=args.tokens,
            steps=args.steps,
            warmup=args.warmup_steps,
            seed=args.seed,
        )
    if "optimizations" in selected:
        suites["optimizations"] = run_synthetic_optimizations_suite(
            device,
            tokens=args.tokens,
            steps=args.steps,
            warmup=args.warmup_steps,
            seed=args.seed,
        )
    if "soak" in selected:
        suites["soak"] = run_synthetic_soak_suite(
            device,
            num_jobs=args.soak_jobs,
            seed=args.seed,
        )
    if "parity_diagnosis" in selected:
        suites["parity_diagnosis"] = diagnose_smoke_real_parity_drift(ROOT)

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/perf/bench_inference.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": "synthetic",
        "system": collect_system_info(device, system_label=args.system),
        "config": {
            "suites": list(selected),
            "quant_formats": list(quant_formats),
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "runs": args.runs,
            "tokens": args.tokens,
            "dim": args.dim,
            "linear_dim": args.linear_dim,
            "soak_jobs": args.soak_jobs,
            "seed": args.seed,
        },
        "suites": suites,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.diagnose_parity:
        diag = diagnose_smoke_real_parity_drift(ROOT)
        text = json.dumps(diag, indent=2)
        print(text)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        return 0

    if args.compare:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.compare]
        markdown = generate_comparison_markdown(reports)
        print(markdown)
        if args.markdown_out is not None:
            args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            summary = {"schema_version": SCHEMA_VERSION, "compared": [str(p) for p in args.compare], "reports": reports}
            args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return 0

    if args.synthetic:
        report = run_synthetic_benchmark(args)
    else:
        device = resolve_device(args.device)
        real_cfg = RealBenchConfig(
            target=args.target,
            base_url=args.base_url,
            checkpoint=args.checkpoint,
            second_checkpoint=args.second_checkpoint,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            scheduler=args.scheduler,
            seed=args.seed,
            warmup_runs=max(0, args.warmup_steps),
            runs=args.runs,
            comfy_workflow=args.comfy_workflow,
            reference_image=args.reference_image,
        )
        suites = run_real_checkpoint_benchmark(real_cfg, device)
        cfg_dict = asdict(real_cfg)
        cfg_dict["comfy_workflow"] = str(real_cfg.comfy_workflow) if real_cfg.comfy_workflow else None
        cfg_dict["reference_image"] = str(real_cfg.reference_image) if real_cfg.reference_image else None
        cfg_dict["resolution"] = f"{real_cfg.width}x{real_cfg.height}"
        report = {
            "schema_version": SCHEMA_VERSION,
            "tool": "tools/perf/bench_inference.py",
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": "real",
            "system": collect_system_info(device, system_label=args.system or args.target),
            "config": cfg_dict,
            "suites": suites,
        }

    markdown = generate_comparison_markdown([report])
    print(markdown)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
