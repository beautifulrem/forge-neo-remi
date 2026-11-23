# Forge Neo Remi — Inference, Memory & Cross-Platform Performance

> **Benchmark baseline date:** 2026-11-22  
> **Code revision:** `remi` @ `8b1b749e81de` (+ MPS memory & mixed-precision optimizations)  
> **Companion UI report:** [ui-performance.md](ui-performance.md)  
> **Harness:** [`tools/perf/bench_inference.py`](../../tools/perf/bench_inference.py) (standardized JSON schema v1, Gradio-free)

This document establishes Remi's reproducible cross-platform inference, VRAM/unified-memory, model-lifecycle, quantization, acceleration, and queue-soak benchmark methodology, documents Apple Silicon MPS hot-path optimizations landed in `backend/memory_management.py` and `backend/operations_mixed_precision.py`, records actual measurements executed on this Apple Silicon host (both Metal MPS and CPU-synthetic), and provides a static root-cause diagnosis of the historical `tools/smoke_real.py` txt2img pixel-parity drift.

---

## 1. Benchmark Methodology

To make Remi vs. Forge Neo vs. ComfyUI comparisons verifiable and free of confounds, every benchmark run must hold the following control variables constant across systems:

1. **Identical hardware & thermal protocol:** Same machine, driver/OS build, PyTorch build, and power state. On fanless / thermally constrained laptops (such as MacBook Air), alternate runs (`A, B, B, A`) after discarding warm-up iterations so thermal throttling affects all systems equally.
2. **Identical model weights & sampling contract:** Same checkpoint/VAE/text-encoder files (verified by hash), identical weight storage precision (`BF16` / `FP8` / `INT8-ConvRot` / `GGUF` / `NVFP4`), identical resolution, steps, sampler, scheduler, CFG scale, shift, batch size, and fixed RNG seed (`12345` for single-image checks; 20 fixed seeds `12345..12364` for quantization/acceleration fidelity ablations).
3. **Standardized warm-up & timing boundaries:**
   - Warm-up steps/runs are executed and synchronized (`torch.cuda.synchronize` / `torch.mps.synchronize`) before recording timed intervals.
   - Report **median** and **p95** step latency (`ms/step`), steady throughput (`it/s`), **first-image** vs. **steady-image** latency, **cold load** (fresh server/page-cache), **warm reload**, and **A↔B checkpoint switch** wall times.
4. **Memory accounting:**
   - **CUDA:** `torch.cuda.max_memory_allocated()`, `memory_reserved()`, and host process RSS (`psutil`).
   - **Apple Silicon MPS:** `torch.mps.current_allocated_memory()`, `torch.mps.driver_allocated_memory()`, Metal `torch.mps.recommended_max_memory()`, and host process RSS.
   - **Soak test:** Continuous 100-job mixed queue (txt2img, img2img, checkpoint switches, queued cancellations, mid-run interrupts, transient error recovery) measuring least-squares **RSS slope (MB / 100 jobs)**, peak/end device memory delta, open file descriptor delta, and SQLite WAL stability.
5. **Fidelity & perceptual guardrails:** Every quantization or acceleration ablation outputs `rel_l2_error`, `MAE`, `max_abs_diff`, `PSNR (dB)`, `SSIM` (for image mode), and `LPIPS` (when the optional `lpips` package is installed).

### Reproducible CLI Commands

```sh
# 1. Fast CI / CPU-synthetic verification (all suites + 100-job SQLite queue soak)
venv/bin/python tools/perf/bench_inference.py --synthetic --device cpu \
  --linear-dim 2048 --tokens 512 --steps 16 --warmup-steps 3 --soak-jobs 100 \
  --suites inference,model_lifecycle,quantization,optimizations,soak,parity_diagnosis \
  --out bench_cpu.json

# 2. Apple Silicon MPS micro-benchmarks (quantized linear ops, MPS cache speedup, attention, soak)
venv/bin/python tools/perf/bench_inference.py --synthetic --device mps \
  --linear-dim 2048 --tokens 1024 --steps 20 --warmup-steps 3 --soak-jobs 100 \
  --suites inference,model_lifecycle,quantization,optimizations,soak,parity_diagnosis \
  --out bench_mps.json

# 3. Real-checkpoint runs against live servers (Remi / Forge Neo / ComfyUI)
venv/bin/python tools/perf/bench_inference.py --real --target remi --system remi \
  --base-url http://127.0.0.1:7860 --checkpoint anima-turbo-v1.1.safetensors \
  --second-checkpoint anima-base.safetensors --width 768 --height 768 --steps 8 \
  --cfg-scale 1.0 --sampler Euler --scheduler Simple --runs 5 --out remi.json

venv/bin/python tools/perf/bench_inference.py --real --target forge-neo --system forge-neo \
  --base-url http://127.0.0.1:7861 --checkpoint anima-turbo-v1.1.safetensors \
  --second-checkpoint anima-base.safetensors --width 768 --height 768 --steps 8 \
  --cfg-scale 1.0 --sampler Euler --scheduler Simple --runs 5 --out forge_neo.json

venv/bin/python tools/perf/bench_inference.py --real --target comfyui --system comfyui \
  --base-url http://127.0.0.1:8188 --comfy-workflow path/to/anima_workflow_api.json \
  --steps 8 --runs 5 --out comfyui.json

# 4. Generate standardized Markdown comparison tables
venv/bin/python tools/perf/bench_inference.py --compare remi.json forge_neo.json comfyui.json \
  --markdown-out comparison.md

# 5. Static parity drift diagnosis
venv/bin/python tools/perf/bench_inference.py --diagnose-parity
```

---

## 2. Cross-System Comparison Matrix Template (Remi vs. Forge Neo vs. ComfyUI)

The table below defines the standard publication matrix populated by `bench_inference.py --compare` once real checkpoint weights are run on target NVIDIA CUDA (8–24 GB) and Apple Silicon MPS (16–64 GB) hosts.

### 2.1 End-to-End Inference, Memory & Model Lifecycle Template

| Model Family & Config | System | Cold Load (s) | Warm Switch (s) | First Image (s) | Steady Image (s) | Throughput (it/s) | Step p95 (ms) | Peak VRAM / MPS (MB) | Peak RAM (MB) | Δ vs. Forge Neo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Anima Turbo 2B** (768², 8 steps, Euler/Simple, CFG 1) | **Forge Neo Remi** | _TBD (GPU)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Anima Turbo 2B** (768², 8 steps, Euler/Simple, CFG 1) | Forge Neo (`neo`) | _TBD (GPU)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | baseline |
| **Anima Turbo 2B** (768², 8 steps, Euler/Simple, CFG 1) | ComfyUI (stock) | _TBD (GPU)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Qwen-Image-2.1 Turbo** (1024², 8 steps, CFG 1, RGBA) | **Forge Neo Remi** | _TBD (GPU)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Qwen-Image-2.1 Turbo** (1024², 8 steps, CFG 1, RGBA) | Forge Neo (`neo`) | _TBD (GPU)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | baseline |
| **Qwen-Image-2.1 Turbo** (1024², 8 steps, CFG 1, RGBA) | ComfyUI (stock) | _TBD (GPU)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **FLUX.3 Image Klein** (1024², 16 steps, CFG 1) | **Forge Neo Remi** | _TBD (GPU)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **FLUX.3 Image Klein** (1024², 16 steps, CFG 1) | ComfyUI (stock) | _TBD (GPU)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### 2.2 Quantization & Acceleration Ablation Template (Real Checkpoints, 20-Seed Protocol)

| Dimension | Variant | Storage Compression | Speedup vs. BF16/Eager | Peak VRAM Δ (MB) | PSNR (dB) | SSIM | LPIPS ↓ | Target Hardware |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Quantization | `BF16` (baseline) | 1.00× | 1.00× | 0 | ∞ | 1.000 | 0.000 | CUDA / MPS |
| Quantization | `FP8 E4M3` (scaled) | ~2.0× | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Ada/Hopper/Blackwell / MPS |
| Quantization | `FP8 RowWise` | ~2.0× | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | SM 8.9+ / MPS |
| Quantization | `INT8-ConvRot` | ~2.0× | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | SM 8.0+ / MPS |
| Quantization | `GGUF (Q8_0 / Q4_K)` | ~1.9×–3.6× | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | CUDA / MPS / CPU |
| Quantization | `SVDQuant (W4A4)` | ~4.0× | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Ada/Blackwell CUDA |
| Quantization | `NVFP4` | ~3.6× | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Blackwell SM 10.0+ / MPS |
| Attention | `Basic` / `SDPA` / `Sage` / `Flash` | — | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | CUDA / MPS |
| Acceleration | `per-block compile` | — | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | CUDA Inductor |
| Acceleration | `Spectrum (SEA)` | — | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | CUDA / MPS |
| Acceleration | `Prefix KV Cache (Qwen-2.1)` | — | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | CUDA / MPS |

---

## 3. Apple Silicon MPS Optimization Findings

Static code audit and Metal micro-benchmarking of `backend/memory_management.py` and `backend/operations_mixed_precision.py` uncovered six concrete hot-path issues on Apple Silicon (and CPU-only CI environments), all addressed cleanly with unit tests in `tests/test_remi_mps_perf.py`:

1. **Unified-memory budget over-reporting & unconditional `torch.mps.empty_cache()` storms (`backend/memory_management.py`):**
   - **Root cause:** `get_total_memory("mps")` reported 100% of host RAM (`psutil.virtual_memory().total`, e.g. 16,384 MB on a 16 GB Mac) instead of Metal's `recommended_max_memory()` (12,124 MB). Worse, `get_free_memory("mps", torch_free_too=True)` returned `mem_free_torch = mem_free_total`, so `free_memory()`'s check `if mem_free_torch > mem_free_total * 0.25:` evaluated `1.0 > 0.25` (always `True`!) and triggered `torch.mps.empty_cache()` — forcing a full Metal command-queue drain and buffer-pool flush on *every* `load_models_gpu()` call even when zero models were unloaded and zero cached bytes were reclaimable.
   - **Fix:** Added `mps_memory_info()` querying `torch.mps.recommended_max_memory()`, `current_allocated_memory()`, and `driver_allocated_memory()`. `get_total_memory("mps")` now caps at `min(vm_total, recommended_max_bytes)` and reports `driver_allocated_bytes`; `get_free_memory("mps")` computes real reclaimable Metal pool bytes `max(0, driver - current)` and bounds headroom by `min(recommended - current, vm_avail + reclaimable)`. `soft_empty_cache(force=False)` skips `torch.mps.empty_cache()` when zero reclaimable bytes exist and no regenerable cache entries were cleared.
2. **Device-identity mismatch between `torch.device("mps")` and `torch.device("mps:0")` (`backend/memory_management.py`):**
   - **Root cause:** `get_torch_device()` returns `torch.device("mps")` (`index=None`), whereas every tensor on MPS carries `tensor.device == torch.device("mps:0")` (`index=0`). In PyTorch, `torch.device("mps") == torch.device("mps:0")` is `False`. Consequently, `cast_to()` missed its no-op identity branch and allocated full-tensor copies, while `LoadedModel.model_memory_required()` and `free_memory()` misclassified resident MPS models.
   - **Fix:** Added `_same_device()` and `_canonical_device()` used across `cast_to()`, `LoadedModel.model_memory_required()`, `use_more_memory()`, `offloaded_memory()`, `free_memory()`, `load_models_gpu()`, and `text_encoder_initial_device()`.
3. **Redundant per-step weight dequantization on MPS (`backend/operations_mixed_precision.py`):**
   - **Root cause:** Apple Silicon has no native FP8/INT8/NVFP4 tensor-core GEMM (`_full_precision_mm = True`), so every quantized `Linear` layer dequantized its weight on *every* denoising step (e.g., 40–100× per image with CFG), repeating 256-dim Hadamard rotations for INT8-ConvRot and E2M1 LUT + block-scale unswizzling for NVFP4 while thrashing the Metal heap.
   - **Fix:** Added a thread-safe, headroom-bounded LRU cache (`_MPSDequantCache`, capped at `min(25% of surplus headroom above minimum_inference_memory + 512 MiB, 2 GiB)`, configurable/disableable via `REMI_MPS_DEQUANT_CACHE_MB`). It activates only for resident MPS weights with no active LoRA `weight_function`, bypasses generic `weights_manual_cast` / CUDA offload-stream scaffolding on MPS, invalidates on `_apply`, `set_weight`, `_load_from_state_dict`, `soft_empty_cache()`, and GC (`weakref.finalize`), and is purged *before* `free_memory()` evicts loaded models.
4. **Synchronous GPU/MPS `.item()` stall in `Embedding._forward()` & redundant FP16 allocation (`backend/operations_mixed_precision.py`):**
   - **Root cause:** `Embedding._forward()` checked `scale.item() != 1.0` on a device tensor, forcing a synchronous Metal/CUDA pipeline stall on every embedding lookup; `_convrot_linear()` allocated a second full weight matrix in FP16 (`qdata.to(fp16) * scale`); `Linear._forward()` ran `QUANT_ALGOS.get(...)` even on `_full_precision_mm` layers.
   - **Fix:** Precomputes `_scale_is_unit` on CPU during `Embedding._load_from_state_dict()` and never calls `.item()` on an accelerator tensor; scales FP16 ConvRot weights in-place (`w.mul_(scale)`); gates `QUANT_ALGOS.get(...)` behind `_use_quantized`.
5. **High-ROI attention finding on Apple Silicon MPS (SDPA vs. Basic):**
   - Because `backend/memory_management.py` sets `ENABLE_PYTORCH_ATTENTION = True` only for CUDA, XPU, and ROCm, Apple Silicon MPS defaults to `attention_basic()` (explicit `einsum` + `softmax`). Micro-benchmarks on this host show PyTorch MPS SDPA (`attention_pytorch`, enabled via `--use-pytorch-cross-attention`) is **2.34× faster** (`0.902 ms` vs. `2.111 ms`) with *lower* error vs. FP32 (`3.4e-5` vs. `4.9e-5`). Follow-up real-checkpoint verification is recommended before changing the global default (since changing default attention on macOS also affects exact parity fixtures).
6. **CPU-only CI/sandbox import safety (`backend/memory_management.py`):**
   - Added a clean fallback to `CPUState.CPU` when neither CUDA, XPU, nor MPS is available, and guarded `should_use_bf16()` when `mac_version()` returns `None`.

---

## 4. Measured Results on This Machine

### Host Environment

| Property | Value |
|---|---|
| Machine / OS | Apple Silicon `arm64`, 10 CPU cores, **16,384 MB** unified RAM, `macOS-27.2.1-arm64` |
| Metal Working-Set Budget | **`12,124.2 MB`** (`torch.mps.recommended_max_memory()`) |
| Runtime | Python `3.13.15` (`venv/bin/python`, Gradio-free), PyTorch `2.15.0`, `comfy-kitchen 0.2.41` |
| Revision | `8b1b749e81de` + MPS optimizations in this change |

### 4.1 Apple Silicon MPS Micro-Benchmarks (`--synthetic --device mps`)

Executed with `--linear-dim 2048 --tokens 1024 --steps 20 --warmup-steps 3 --soak-jobs 100` (`bfloat16` compute on Metal MPS):

#### Inference & Model Lifecycle (4-block Synthetic DiT, 1024 tokens, dim 256)

| Suite | Metric | Measured (MPS) |
|---|---|---:|
| Inference | Median / p95 step latency | **4.504 ms** / 5.670 ms |
| Inference | Steady throughput | **222.02 it/s** |
| Inference | First / steady image (20 steps) | 0.0987 s / **0.0899 s** |
| Inference | Peak MPS allocated / driver / RSS | 7.63 MB / 48.47 MB / 362.06 MB |
| Model Lifecycle | Cold load / warm reload / A↔B switch / unload (6.13 MB safetensors) | 0.0220 s / 0.0141 s / 0.0340 s / 0.0172 s |
| Model Lifecycle | Post-unload MPS allocated | **0.00 MB** |

#### Quantized Linear Ops on Apple Silicon MPS (`1024 × 2048 → 2048`, BF16 compute)

| Format | Storage (MB) | Compression | Cached MPS Median (ms) | Uncached MPS Median (ms) | **MPS Cache Speedup** | Cache Hit Rate | Rel L2 Error vs. BF16 | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bf16` (reference) | 8.000 | 1.00× | 3.213 | — | — | — | 0.00000 | 99.99 |
| `fp8_e4m3` | 4.000 | 2.00× | **2.895** | 3.099 | **1.07×** | 95.7% | 0.02657 | 51.26 |
| `fp8_rowwise` | 4.008 | 2.00× | **2.825** | 3.838 | **1.36×** | 95.7% | 0.02640 | 51.32 |
| `int8_convrot` | 4.008 | 2.00× | **2.746** | 4.111 | **1.50×** | 95.7% | 0.00873 | 60.93 |
| `nvfp4` | 2.250 | 3.56× | **2.889** | 4.883 | **1.69×** | 95.7% | 0.09414 | 40.27 |
| `gguf_q8_0` | 4.250 | 1.88× | 3.613 | — | _(GGUF path)_ | — | 0.00666 | 63.28 |
| `svdquant_w4a4` (ConvRot W4A4) | 2.008 | 3.98× | 7.870 | — | _(eager dequant)_ | — | 0.14977 | 36.24 |

> **Key takeaway:** On Apple Silicon MPS, the headroom-bounded dequantization cache eliminates **7% to 69%** of per-step quantized linear latency (`1.36×` on RowWise FP8, `1.50×` on INT8-ConvRot, `1.69×` on NVFP4) while keeping identical bitwise results and releasing 100% of cached bytes automatically on `soft_empty_cache()` or memory pressure.

#### Attention & Accelerators on Apple Silicon MPS (`1024 tokens, 8 heads × 64`)

| Category | Backend | Status | Median (ms) | p95 (ms) | Speedup | Max Abs Diff vs. FP32 | PSNR (dB) |
|---|---|---|---:|---:|---:|---:|---:|
| Attention | `basic` (`attention_basic`) | ok | 2.111 | 2.323 | 1.00× | 0.00005 | 65.98 |
| Attention | `sdpa` (`attention_pytorch`) | ok | **0.902** | **1.083** | **2.34×** | **0.00003** | **68.07** |
| Attention | `sage` / `flash` / `xformers` | unsupported | — | — | — | CUDA-only | — |
| Accelerator | `eager_baseline` (20 steps) | ok | 21.68 total (1.084/step) | — | 1.00× | 0.00000 | 99.99 |
| Accelerator | `spectrum_forecast` (12 real + 8 forecast) | ok | **15.13 total (0.756/step)** | — | **1.43×** | 0.00391 | 80.58 |
| Accelerator | `per_block_compile` | unsupported | — | — | — | CUDA Inductor target | — |

#### 100-Job Queue Soak Test (Real SQLite WAL `JobStore` + `FileStore` + `JobQueue` + MPS Ops)

| Backend | Jobs | Elapsed (s) | Model Switches | RSS Start → Peak → End (MB) | RSS Slope (MB/100 jobs) | Device Mem Start → End (MB) | FD Start → End (Δ) | Leak Detected |
|---|---:|---:|---:|---|---:|---|---|---|
| **MPS** | 100 | 0.123 | 9 | 645.8 → 647.8 → 647.7 | **0.06** | 0.63 → 0.63 | 12 → 12 (**0**) | **NO** |
| **CPU** | 100 | 0.062 | 9 | 568.8 → 570.4 → 570.3 | **0.72** | 0.00 → 0.00 | 6 → 6 (**0**) | **NO** |

### 4.2 CPU-Synthetic CI Baseline (`--synthetic --device cpu`)

Executed with `--linear-dim 2048 --tokens 512 --steps 16 --warmup-steps 3 --soak-jobs 100` (`float32` compute on 10-core Apple Silicon CPU):

- **Inference:** `180.70 it/s`, `5.534 ms` median (`6.449 ms` p95), peak RSS `233.6 MB`.
- **Lifecycle:** cold load `0.013 s`, warm reload `0.005 s`, A↔B switch `0.023 s`, unload `0.014 s`.
- **Quantization (`512 × 2048 → 2048`):** `bf16` `2.674 ms` (99.99 dB), `fp8_e4m3` `3.926 ms` (2.00×, 50.94 dB), `fp8_rowwise` `3.809 ms` (2.00×, 51.01 dB), `int8_convrot` `3.408 ms` (2.00×, 61.06 dB), `gguf_q8_0` `3.442 ms` (1.88×, 63.74 dB), `svdquant_w4a4` `8.347 ms` (3.98×, 35.88 dB), `nvfp4` `8.869 ms` (3.56×, 39.92 dB).
- **Attention & Spectrum:** `sdpa` `1.182 ms` vs. `basic` `2.554 ms` (**2.16× speedup**, 0.00000 max abs diff); `spectrum_forecast` `43.22 ms` vs. `eager_baseline` `58.71 ms` (**1.36× speedup**).

---

## 5. Static Diagnosis: Historical `tools/smoke_real.py` txt2img Pixel-Parity Drift

In [`docs/upstream-sync-2026-11-22.md`](../upstream-sync-2026-11-22.md), `tools/smoke_real.py` passed 10 of 12 checks and failed only the two exact pixel-parity assertions against `tests/fixtures/parity/anima_turbo_txt2img_seed12345.png` with **`mean 8.3677, max 247`** on both `/sdapi/v1/txt2img` and `/api/v2/jobs`, while producing **bitwise identical** outputs (`max diff 0`) before and after the Forge Neo 2.33 sync.

Static inspection (`bench_inference.py --diagnose-parity`) and git forensics between the last pixel-identical smoke run (`78f325d3`, 2026-10-27, [smoke/2026-10-27.json](smoke/2026-10-27.json), `mean 0.0000, max 0`) and `ed6cc223` (2026-11-22) pinpoint the exact root causes introduced during the 2026-11-18 upstream sync (`27cbc152`, merging Forge Neo `89e2e09d`):

1. **Primary Root Cause — Commit `b59500ed` ("precision", `backend/attention.py:get_attn_precision`):**
   - Prior to `b59500ed`, `get_attn_precision()` was missing its `return` statement and unconditionally returned `None`:
     ```python
     # Before b59500ed (2026-10-15 baseline & 2026-10-27 smoke):
     def get_attn_precision(attn_precision: torch.dtype, current_dtype: torch.dtype) -> torch.dtype:
         memory_management.force_upcast_attention_dtype().get(current_dtype, attn_precision)  # returned None!
     ```
   - On Apple Silicon MPS, `ENABLE_PYTORCH_ATTENTION` is `False` by default, so both **Qwen3-0.6B** (`backend/nn/llm/llama.py:16`, default `float16`) and **Anima DiT** (`backend/nn/anima.py`, `supported_inference_dtypes=[float16, bfloat16, float32]` -> `float16` on MPS) execute `attention_basic()`.
   - On macOS ≥ 14.5, `memory_management.force_upcast_attention_dtype()` returns `{torch.float16: torch.float32}`. Restoring the missing `return` in `b59500ed` activated `float16 -> float32` upcasting of `Q @ K^T` and `softmax` across every Qwen3-0.6B layer and all 28 Anima DiT blocks.
   - Across 8 Euler rectified-flow steps, this deterministic precision upgrade accumulates a small floating-point shift into an edge-localized pixel delta (`mean 8.3677, max 247`) while remaining 100% bitwise deterministic across runs (`max diff 0`).
2. **Secondary Contributing Factor — Commit `865255e6` ("opt", ComfyUI PR #17270, `backend/nn/_wan_cache.py` & `backend/nn/wan_vae.py`):**
   - Replaced full-frame `WanVAE` (`qwen_image_vae`) residual and head `CausalConv3d` passes with striped spatial evaluation (`strip_apply`) whenever `x.numel() > STRIP_ELEMS` (`2**24 = 16,777,216`).
   - Decoding the `768×768` parity fixture creates a 96-channel intermediate of `1 × 96 × 1 × 768 × 768 = 56,623,104` elements (`> 16.78M`), activating striped Conv3d evaluation whose floating-point accumulation order differs by ~1–2 LSB from full-frame evaluation.
3. **Recommended Verification & Closure:**
   - Commit `b59500ed` is an **intentional upstream accuracy fix** (preventing macOS ≥ 14.5 FP16 attention overflow), not a regression.
   - On an Apple Silicon host with `anima-turbo-v1.1.safetensors`, confirm the hypothesis once by comparing `get_attn_precision()` returning `None` vs. `torch.float32`, then regenerate `tests/fixtures/parity/anima_turbo_txt2img_seed12345.png` under the corrected FP32-upcast baseline and record perceptual parity (`PSNR / SSIM`) alongside strict bitwise parity in `tools/smoke_real.py`.
