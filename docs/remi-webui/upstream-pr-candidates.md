# Upstream PR Roadmap & Code Bloat Audit

> **Audit baseline:** `remi` `9a7e665ec` vs Forge Neo `upstream/neo` merge base `5998fb39` (`2026-11-23`)
> **Governance tool:** `python tools/upstream_forks.py budget [--max-forked 6] [--target-diverged 20] [--json] [--verbose]`
> **Policy reference:** [upstream-absorption.md](upstream-absorption.md) · [roadmap-2026-q4.zh-CN.md](../research/roadmap-2026-q4.zh-CN.md) (P1.4 Divergence Budget)

---

## Executive Summary

1. **Divergence budget snapshot (`9a7e665ec`):**
   - `FORKED`: **6 / max 6** (hard guardrail enforced by `tools/upstream_forks.py budget`).
   - `DIVERGED`: **32 / target ≤ 20** (+12 over target). All 32 pinning test files declared in [upstream-absorption.md §1.2](upstream-absorption.md#12-upstream-bugs-remi-fixes-divergences) exist on disk and pass.
   - `upstream_crlf.txt`: **80** shared files preserving upstream CRLF line endings.
2. **Concrete path from `DIVERGED` 32 → ≤ 20 (down to 12):**
   - Code-level diff inspection against `5998fb39` classifies **20 of the 32** files as **`upstream-pr-candidate`** (pure deterministic Forge Neo bug fixes whose §1.2 row drops completely once accepted upstream) and **12** as **`remi-architecture-hook`** (files carrying Remi-specific DiT caches, Anima 2.9B/3.8B hooks, Spectrum seams, or `remi.ui`/`remi.model_selection` wiring).
   - **18 of the 20** PR candidates require **zero prerequisite hooks** in upstream Forge Neo. Landing Wave 1 (8 quick-win PRs) + Wave 2 (10 standalone ControlNet/img2img/infotext/cache PRs) reduces `DIVERGED` from **32 → 14** (6 files below the ≤ 20 target), and Wave 3 (a tiny 12-line `Script.cleanup` teardown hook + 2 PRs) lowers it further to **12**.
   - Even across the **12 retained `remi-architecture-hook`** files, 12 discrete deterministic bug fixes (such as `comfy/lora.py`'s mutable `key_map={}` default, `sd_schedulers.py`'s `turbo_scheduler`, `krea.py`'s in-place `TextFusionBlock` mutation, `sampling_function.py`'s MPS/CPU nondeterministic batch merge, and `processing.py`'s 5D `fill_latent_noise` broadcast bug) can be upstreamed as partial PRs to shrink merge-conflict surface.
3. **Code bloat & redundancy hotspots:**
   - The 5 `backend/nn/*_diffusers.py` modules total **1,415 lines** with heavily copy-pasted directory/sharding helpers and cross-family imports into `anima_diffusers.py`; `backend/nn/dit_registry.py` (**890 lines**) carries **29 thin lazy-import wrapper functions (115 lines)**; `_tensor_version` is duplicated **4×** and `_broadcast_ref_batch` **2×** across DiT models.
   - Five Remi Python modules exceed **800 lines** (`remi/networks.py` 1,307L, `remi/ui/nodes.py` 1,141L, `remi/extensions_admin.py` 922L, `remi/jobs/importers.py` 865L, `remi/jobs/queue.py` 811L), and five Remi-owned built-in extension scripts range from **816 to 2,511 lines**.
   - In `webui/src/` (626 files, 71,936 lines), `SearchBox`, `Thumb`, and `clamp` are each re-implemented **3×**, `ImageListInput` / `ImageListWidget` duplicate image-list rendering (470L combined), and four production views/engines exceed **600 lines** (`EditorBody.tsx` 761L, `document.ts` 649L, `json-builder/index.tsx` 627L, `NetworkBrowser.tsx` 600L).

---

## Part 1 — Exact Upstream PR Roadmap (`DIVERGED` 32 → ≤ 20)

### 1.1 Reduction Trajectory

| Wave | Scope | Files Dropped | Remaining `DIVERGED` | Upstream Prerequisite |
| --- | --- | ---: | ---: | --- |
| Baseline (`9a7e665ec`) | Current registry (`tools/upstream_forks.py`) | — | **32** | — |
| **Wave 1 (Quick Wins)** | Security, MPS OOM crash, flow samplers/schedulers, tiny 1–17L pure fixes | **8** | **24** | None |
| **Wave 2 (Standalone Runtime)** | ControlNet preprocessors/filter/cache, LLLite step window, ImageStitch, infotext/img2img/merger/refiner | **10** | **14** (target ≤ 20 achieved) | None |
| **Wave 3 (Guaranteed Teardown)** | `Script.cleanup` (~12L hook in `modules/scripts.py` + `finally` in `processing.py`) → Radial + Refiner | **2** | **12** | 1 small `Script.cleanup` PR |
| **Wave 4 (Hook Slimming)** | Partial bug-fix PRs extracted from the 12 retained `remi-architecture-hook` files | 0 rows dropped (smaller diffs) | **12** | None |

---

### 1.2 Upstream PR Candidates (20 Files — Full Row Removal)

Every file below has zero Remi-only runtime dependency for its §1.2 fix (for built-in scripts that also replaced `ui()` with `params()`, that UI port is tracked by [upstream-absorption.md §1](upstream-absorption.md#1-where-upstream-files-went) and does not require a §1.2 `DIVERGED` row). Diff sizes are measured via `git diff --numstat 5998fb39 HEAD -- <path>` (total file diff vs merge base; `fix scope` notes the runtime bug-fix lines when the file also carries a `params()` UI port).

#### Wave 1 — Immediate Quick-Win PRs (8 Files; 1–17 Lines Each, Zero Prerequisites)

| # | File | Diff vs `5998fb39` | Deterministic Fix Summary | Pinning Test | Priority / Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `modules/images.py` | +6 / -25 (fix: 4L) | **Security:** verify `exif_data` is `bytes` starting with JPEG/TIFF/Exif magic bytes before `piexif.load()`, preventing uploaded EXIF chunks from being opened as arbitrary server file paths. | `tests/test_remi_security.py` | **P0 Security** — trivial 4-line guard; highest urgency to upstream. |
| 2 | `backend/memory_management.py` | +8 / -0 | Guard `discard_cuda_async_error()` behind `is_device_cuda(get_torch_device())` (on MPS `torch.cuda.synchronize()` raised inside `is_oom`, breaking tiled-VAE fallback); `@functools.lru_cache(maxsize=1)` on `mac_version()`. | `tests/test_memory_management_oom.py` | **P0 Stability** — fixes real Apple Silicon MPS OOM crash. |
| 3 | `backend/attention.py` | +6 / -2 | Extend `attention_basic` bool mask reshaping to accept 2D `(Q, K)` and 3D/4D boolean masks (e.g. Qwen-Image-2.1 block-causal prefix mask) alongside `(B, K)` padding masks. | `tests/test_remi_qwen21.py` | **P0 Correctness** — fixes fallback attention crash on 4D bool masks. |
| 4 | `modules_forge/packages/k_diffusion/external.py` | +6 / -2 | `ForgeScheduleLinker.get_sigmas`: space predictor timesteps between `sigma_to_t(sigmas[-1])` and `sigma_to_t(sigmas[0])` instead of `999..0` (preventing division by zero / black outputs on flow predictors). | `tests/test_flow_schedules.py` | **P0 Correctness** — affects every flow model using Uniform scheduler. |
| 5 | `modules/sd_samplers.py` | +17 / -0 | Refuse VP/VE samplers (`DDIM`, `PLMS`, `Kohaku LoNyu Yog`) in `create_sampler()` when `predictor.prediction_type == "const"`, matching upstream's existing `no_sdxl` check. | `tests/test_flow_sampler_support.py` | **P0 UX/Correctness** — prevents garbled outputs on Flux/Anima/Qwen. |
| 6 | `modules/sd_samplers_common.py` | +9 / -0 | In `apply_refiner`: call `clear_prompt_cache()` before `setup_conds()` (refiner reused base conds when loading params matched), call `calculate_shorthash()` before writing infotext, and record `"Refiner CFG scale"`. | `tests/test_refiner_switch.py` | **P0 Correctness** — fixes broken refiner conditioning reuse. |
| 7 | `extensions-builtin/forge_preprocessor_tile/scripts/preprocessor_tile.py` | +7 / -2 | Broadcast per-image sigmas across chunks (`row_sigmas`) in `PreprocessorTileColorFix` so batch size ≥ 2 no longer crashes. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P0 Crash Fix** — clean 7-line standalone patch. |
| 8 | `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py` | +81 / -103 (fix: 1L) | Replace `masks_for_overlay[i] = converted_mask` with `masks_for_overlay.append(converted_mask)` in `apply_masks()` (fixes `IndexError` on empty list). | `extensions-builtin/soft-inpainting/tests/test_soft_inpainting_params.py` | **P0 One-liner** — rest of diff is standard `params()` UI port. |

#### Wave 2 — Standalone Runtime & Built-in Extension PRs (10 Files; Zero Prerequisites)

| # | File | Diff vs `5998fb39` | Deterministic Fix Summary | Pinning Test | Priority / Notes |
| --- | --- | --- | --- | --- | --- |
| 9 | `extensions-builtin/sd_forge_lora_schedule/scripts/control.py` | +5 / -6 | Move `available_network_aliases[alias]` lookup inside `try`, reject malformed schedules atomically (`all(len(pt) == 2)`), and default weight before first point to `schedule[0][0]`. | `extensions-builtin/sd_forge_lora_schedule/tests/test_lora_schedule_params.py` | **P1 Correctness** — tiny 5-line diff. |
| 10 | `modules/img2img.py` | +190 / -87 (fix: 4L) | Wrap `parse_generation_parameters(geninfo)` in `process_batch` with `try/finally` so a malformed batch image's PNG Info never leaves `shared.opts.infotext_styles` stuck at `"Ignore"`. | `tests/test_img2img_batch_png_info.py` | **P1 State Leak** — rest of diff is §1 `run_img2img` entrypoint port. |
| 11 | `modules/extras.py` | +88 / -52 (fix: 3L) | Refuse a Checkpoint Merger output filename whose resolved `realpath` matches any input checkpoint being merged (prevents corrupting an input model mid-merge). | `tests/test_remi_merger.py` | **P1 Data Safety** — rest of diff is §1 merger UI-contract translation. |
| 12 | `modules/infotext_utils.py` | +28 / -278 (fix: 12L) | Fix `"Hires Module 1"` paste handling (`"Built-in"`, `"Use same choices"`, missing modules) and linear-time atomic-group `re_param_code` (eliminating multi-minute backtracking on 200k-char prompt lines). | `tests/test_infotext_paste_roundtrip.py` | **P1 Correctness + DoS** — also covered by `tests/test_remi_image_meta.py`. |
| 13 | `extensions-builtin/forge_preprocessor_inpaint/scripts/preprocessor_inpaint.py` | +22 / -4 | Store `inpaint_only_image`/`mask` per unit on `params` (two units no longer clobber each other); blend using `hr_image`/`hr_mask` when a unit skipped the hires pass; interpolate mask/raw to output size instead of downscaling hires output. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 Correctness** — zero Remi imports; clean standalone PR. |
| 14 | `extensions-builtin/forge_preprocessor_reference/scripts/forge_reference.py` | +16 / -5 | Wrap style-recording forward pass in `try/finally: self.is_recording_style = False`; add `process_after_every_sampling` + `release()` to drop recorded GPU self-attention tensors after sampling. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 VRAM Leak** — stops GPU tensor retention across generations. |
| 15 | `extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite.py` | +41 / -4 | Add `sampler_progress(process)` and `step_active(progress, start_percent, end_percent)` so LLLite step windows track sampler steps rather than raw forward calls (fixes img2img, Hires, MultiDiffusion, 2nd-order samplers). | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 Correctness** — purely additive optional `progress=None` parameter. |
| 16 | `extensions-builtin/sd_forge_controlnet/lib_controlnet/global_state.py` | +41 / -2 | Add cached safetensors key inspection (`control_model_family`) so `get_filtered_controlnet_names()` only offers Anima DiT control models on Anima and UNet ControlNets on UNet checkpoints. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 UX/Correctness** — uses upstream `dynamic_args.anima` + `safetensors`. |
| 17 | `modules_forge/supported_controlnet.py` | +22 / -4 | Key `_CONTROL_MODEL_CACHE` by `(realpath, mtime_ns, size, *dtypes)` via `cache_key()` so replaced files / changed dtypes reload cleanly, and honor `control_net_model_cache_size == 0`. | `tests/test_controlnet_model_cache.py` | **P1 Cache Bug** — 2 one-line base-class hooks (`anima_compatible`, `process_before_conditioning`) remain as normal §1 hooks. |
| 18 | `extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py` | +60 / -132 (fix: 18L) | Track `cached_model = weakref.ref(p.sd_model)` so references re-encode after model unload/reload; preserve 4-channel RGBA references for `dynamic_args.qwen21` and hash alpha channel in `hash_image`. | `extensions-builtin/sd_forge_image_stitch/tests/test_image_stitch_params.py` | **P1 Correctness** — also pinned by `tests/test_remi_qwen21.py`; rest is `params()` UI port. |

#### Wave 3 — Guaranteed-Teardown PRs (2 Files; Unblocked by One 12-Line `Script.cleanup` PR)

Upstream Forge Neo's `ScriptRunner` currently has `postprocess()`, which does not execute when sampling raises an exception (e.g. OOM or user interrupt). Submitting a minimal prerequisite PR adding `Script.cleanup(self, p, *args)` called inside `finally:` of `process_images_inner` unblocks both rows below (and shrinks `controlnet.py` in Wave 4):

| # | File | Diff vs `5998fb39` | Deterministic Fix Summary | Pinning Test | Priority / Notes |
| --- | --- | --- | --- | --- | --- |
| 19 | `extensions-builtin/sd_forge_radial/scripts/forge_radial.py` | +33 / -20 (fix: 10L) | Match batched `timestep[0]` against `sigmas` (standalone 3L fix!) and restore `wan.attention = ORIG_ATTENTION` in `cleanup()` so errors or `last_dense_timestep == 0` do not leak radial attention into later Wan runs. | `extensions-builtin/sd_forge_radial/tests/test_radial_params.py` | **P2 Leak + Crash** — batched `timestep[0]` can even ship in Wave 1. |
| 20 | `modules/processing_scripts/refiner.py` | +63 / -87 (fix: 10L) | Clear `p.refiner_cfg = None` when disabled in `setup()`, and call `restore_base_weights()` from `cleanup()` so an exception after fast-swap restores the base checkpoint weights. | `tests/test_refiner_switch.py` | **P2 State Corruption** — rest of diff is standard `params()` UI port. |

---

### 1.3 Retained Remi Architecture Hooks (12 Files — With Extractable Partial PRs)

These 12 files carry Remi-only DiT architecture hooks, Anima 2.9B/3.8B features, Spectrum pre-head seams, or `remi.*` integration and therefore stay in `DIVERGED` even after Wave 1–3. However, **all 12 contain clean sub-fixes** that can still be extracted into upstream PRs (Wave 4) to keep Remi's merge diffs thin:

| # | File | Diff vs `5998fb39` | Why Retained as `remi-architecture-hook` | Pinning Test | Extractable Partial Upstream PR (Wave 4) |
| --- | --- | --- | --- | --- | --- |
| 1 | `backend/nn/krea.py` | +172 / -37 | Step-invariant `_txt_cache`/`_ref_cache`/`_freqs_cache`, `single_block` SLG replace hook, Spectrum `last_input`/`compute_last` seam. | `tests/test_remi_krea2.py` | `TextFusionBlock` out-of-place `x = x + ...` (fixes cached conditioning mutation) + `_broadcast_ref_batch` & odd-dim unpad. |
| 2 | `backend/nn/qwen21.py` | +393 / -11 | Lossless block-causal prefix KV cache (VRAM budget guard, eager prefill for per-block compile), target RoPE cache, Spectrum `final_input`/`compute_final` seam. | `tests/test_remi_qwen21.py` | 4D `(1, 1, Q, K)` causal mask + `_broadcast_ref_batch` safe reference broadcast. |
| 3 | `backend/sampling/sampling_function.py` | +62 / -5 | Anima-3.8B `_ANIMA_SEMANTIC_KEYS` batching, `batch_indices`, `edit_strength` forwarding, `moss_ref_kv` / `dit.clear_caches()` cleanup. | `tests/test_sampling_batch_indices.py` | `merges_by_free_memory(device)` (stops nondeterministic MPS/CPU cond/uncond batch merging causing image drift). |
| 4 | `backend/text_processing/anima_engine.py` | +161 / -33 | Remi `PromptChunk`, raw tokenizers for NegPiP/FreeFuse, Anima-3.8B Qwen3.5 semantic connector inputs (`semantic_inputs`). | `tests/test_anima_text_engine_compat.py` | Treat parser `BREAK` chunk marker as a space rather than literal `"BREAK"` at weight `-1`. |
| 5 | `extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite_anima.py` | +82 / -15 | Imports Remi-only `backend/nn/anima_layout.py` for 2.9B/3.8B depth remapping. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | Chain-unhook `restore()` fix (2 units on same `Linear`), `_check_width` mismatch error, `step_active` window, once-per-sampling skip warning. |
| 6 | `extensions-builtin/sd_forge_controlllite/scripts/forge_controllllite.py` | +263 / -11 | Registers Remi-only `AnimaControlLoraPatcher`, `AnimaControlVacePatcher`, and `_request_spectrum_window`. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | Copy control image before applying soft mask (prevents compounding across batches/hires) + follow Hires checkpoint DiT. |
| 7 | `extensions-builtin/sd_forge_controlnet/scripts/controlnet.py` | +130 / -46 | Carries Remi `unit_param`/`unit_actions` UI, `process_before_conditioning` text-patch hook (Anima control-LoRA), and `cleanup()` teardown. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | Remove failed unit from `extra_generation_params` infotext + release reference preprocessor recordings on teardown. |
| 8 | `extensions-builtin/sd_forge_lora/networks.py` | +285 / -55 | Imports `anima_layout` & `RESOLVED_LORA_DEPTHS`, mirrors LLM adapter LoRA into Anima-3.8B DiT, tracks `anima_sources`/`remap_notes`/`anima_inserted_blocks_policy`. | `extensions-builtin/sd_forge_lora/tests/test_networks_apply_failure.py` | `LoraNotApplied` cleanup so a failed LoRA apply does not leave the previous LoRA set active and re-reports failures on cached hashes. |
| 9 | `modules/api/api.py` | +190 / -174 | Rewritten around `remi.ui` layouts, `remi.model_selection` deferred model switching, and `modules.redact` flag redaction. | `tests/test_api_pasted_value.py` | `pasted_value` overflow guard (`Steps: 1e999`), HTTP 422 on unknown `/sdapi/v1/options` keys, `stop_webui(self)` method signature fix. |
| 10 | `modules/processing.py` | +104 / -23 | Core runtime hub carrying Remi's `after_model_load`/`cleanup` hooks, `keep_negative_conds` (Anima NAG), `collect_workflow_advice`, and Qwen-2.1 RGBA pipeline. | `tests/test_prompt_cache_clear.py` | 4 standalone PRs: `fill_latent_noise` 5D broadcast bug, `clear_prompt_cache` on reload + hires conds, `ini_latent` edit-model clear/restore, `skips_negative_conds` for `CFG++` samplers. |
| 11 | `modules/sd_schedulers.py` | +69 / -11 | Adds Remi's `beta57_scheduler`, `anima_flow_match_scheduler`, and `all_schedulers_map` alias table for Anima Flow scheduler locking. | `tests/test_flow_schedules.py` | 3-line `turbo_scheduler` fix (`inner_model.sigmas[timesteps]` instead of `predictor.sigma(timesteps)` on flow models). |
| 12 | `modules_forge/packages/comfy/lora.py` | +101 / -25 | Adds `SplitTargets`/`SlicedWeightAdapter` and Qwen-2.1/MOSS/Krea2/FLUX.3 multi-trainer LoRA key dispatchers. | `tests/test_anima_lora_formats.py` | 4-line mutable-default bug fix (`key_map=None` in `model_lora_keys_clip` / `model_lora_keys_unet`). |

### 1.4 Upstream PR Execution Checklist

When upstreaming any candidate from §1.2 or §1.3:
1. Branch directly from `upstream/neo` (`5998fb39` or latest `upstream/neo`) and port **only** the minimal bug-fix hunk — zero `remi.*` imports, zero `params()` UI changes, preserving upstream CRLF line endings (`tools/check_line_endings.py`).
2. Include a minimal standalone reproduction / unit test in the upstream PR description.
3. Keep Remi's pinning test in place permanently (even after upstream merges the fix!).
4. On the next Remi upstream sync that absorbs the merged PR, drop the file from `DIVERGED`, `DIVERGED_CLASSIFICATION` (`tools/upstream_forks.py`), and [upstream-absorption.md §1.2](upstream-absorption.md#12-upstream-bugs-remi-fixes-divergences), then run `python tools/upstream_forks.py budget`.

---

## Part 2 — Code Bloat & Efficiency Audit (`backend/`, `modules/`, `remi/`, `extensions-builtin/`, `webui/src/`)

### 2.1 Codebase Size Snapshot

| Subsystem | Files | Total LOC | Largest File (LOC) | Governance Boundary |
| --- | ---: | ---: | --- | --- |
| `remi/` (Python control plane & UI schema) | 75 | 20,928 | `remi/networks.py` (1,307) | Remi-owned (refactor freely) |
| `webui/src/` (React SPA, excl. `api/gen/`) | 626 | 71,936 | `webui/src/api/mock/fixtures.ts` (842); prod: `canvas/ui/EditorBody.tsx` (761) | Remi-owned (refactor freely) |
| `backend/` (Inference & DiT models) | 113 | 30,907 | `backend/memory_management.py` (1,590) | Mixed: upstream-shared vs Remi-only (`*_diffusers.py`, `dit_registry.py`, `anima_*.py`) |
| `modules/` & `modules_forge/` (excl. `packages/`) | 122 | 25,555 | `modules/processing.py` (2,013) | Mostly upstream-shared (keep diffs minimal!) |
| `extensions-builtin/` (incl. vendored annotators) | 968 | 174,531 | `sd-forge-prompt-workshop/scripts/prompt_workshop.py` (2,511) | Mixed: Remi Anima/workshop extensions vs upstream built-ins |

---

### 2.2 Prioritized Refactoring Recommendations

#### P0 — High-ROI Duplication & Maintenance-Debt Eliminators (Remi-Owned Backend)

1. **Consolidate the 5 `backend/nn/*_diffusers.py` modules into a shared `backend/nn/diffusers_io.py` (1,415 LOC → ~900 LOC; saves ~450–550 LOC)**
   - **Files & line counts:**
     - `backend/nn/anima_diffusers.py` (**384L**)
     - `backend/nn/qwen21_diffusers.py` (**338L**)
     - `backend/nn/flux3_diffusers.py` (**280L**)
     - `backend/nn/krea_diffusers.py` (**259L**)
     - `backend/nn/moss_diffusers.py` (**154L**)
   - **Verified duplication:**
     - All 4 newer files (`flux3_diffusers.py`, `krea_diffusers.py`, `moss_diffusers.py`, `qwen21_diffusers.py`) reach into `anima_diffusers.py` for generic HuggingFace sharding utilities (`_read_json`, `is_sharded_hf_dir`, `find_sharded_index`, `load_sharded_safetensors`), coupling every DiT family to Anima.
     - Every family re-implements near-identical `_dir_primary` / `_dir_primary_safetensors` (`flux3_diffusers.py:L119-L122`, `qwen21_diffusers.py:L48-L51`, `moss_diffusers.py:L29-L44`), `*_diffusers_primary_file`, `is_*_diffusers_dir`, `is_sharded_*_transformer_dir`, and `load_*_diffusers_dir` (~500 lines of repeated directory/shard boilerplate across the 5 files).
     - `anima_diffusers.py` (`is_diffusers_wan_vae` / `_map_wan_resnet` / `convert_diffusers_wan_vae`, L112–L176, 61L) and `qwen21_diffusers.py` (`is_diffusers_qwen21_vae` / `_map_qwen21_resnet` / `convert_diffusers_qwen21_vae`, L57–L157, 91L) duplicate Wan-family VAE ResNet conversion logic (9 overlapping 8-line windows).
   - **Recommendation:** Extract a family-agnostic `backend/nn/diffusers_io.py` providing `DiffusersDirSpec` (component map, class predicates, primary-file lookup, sharded transformer detection, and generic `load_diffusers_dir`) plus a shared `convert_diffusers_wan_vae_family(..., channels=...)` converter. Each model's `*_diffusers.py` shrinks to only its architecture-specific key translation table.

2. **Slim `backend/nn/dit_registry.py` (890 LOC) and extract shared DiT tensor/batch helpers**
   - **Files & line counts:**
     - `backend/nn/dit_registry.py` (**890L**): contains **29 thin private wrapper functions** (`_is_*_dir`, `_*_primary_file`, `_load_*_dir`, `_is_*_single_file`, `_convert_*_single_file`, L544–L855, totaling **115 lines** of repetitive `from backend.nn.X import fn; return fn(...)` glue) plus **4 parallel `*_workflow_advice` functions** (L312–L493, **168 lines**).
     - `_tensor_version` duplicated **4×**: `backend/nn/flux3.py`, `backend/nn/krea.py`, `backend/nn/moss_image.py`, `backend/nn/qwen21.py`.
     - `_broadcast_ref_batch` duplicated **2×**: `backend/nn/krea.py` (L68–L75), `backend/nn/qwen21.py` (L147–L154).
     - `timestep_embedding` duplicated **4×**: `backend/nn/anima_connector.py`, `backend/nn/flux.py`, `backend/nn/lumina.py`, `backend/nn/unet.py` (keep upstream `flux.py`/`lumina.py`/`unet.py` untouched; reuse from a shared helper in Remi-owned files).
   - **Recommendation:** Add a 6-line `_lazy(module_path, attr)` descriptor helper in `dit_registry.py` (replacing all 29 wrapper functions), consolidate `*_workflow_advice` into declarative rule predicates aligned with P0.2 `AdviceItem`, and hoist `_tensor_version` + `_broadcast_ref_batch` into a tiny `backend/nn/_dit_common.py` (note: keep `krea.py` / `qwen21.py` self-contained if targeting upstream PRs, or import from `_dit_common.py` for Remi-only caches).

3. **Eliminate dual-track architecture branching (`dynamic_args.*` & `PresetArch`)**
   - **Files & line counts:**
     - `modules/processing.py` (**2,013L**): **17** `dynamic_args.*` branches.
     - `extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py` (**130L**): **7** `dynamic_args.*` branches.
     - `modules_forge/presets.py` (**274L**): **7 parallel dicts** (`SAMPLERS`, `SCHEDULERS`, `STEPS`, `CFG`, `DISTILL`, `SHIFT`, `FRAMES`) separate from `DiTCapability`.
   - **Recommendation:** Route capability checks (RGBA preservation, reference max count, default sampling profiles) through `DiTCapability` / `ModelCapability` (P0.2) rather than adding further `if dynamic_args.<model>:` switches in shared upstream files.

---

#### P1 — Modularize Oversized `remi/` & `webui/src/` Files and Deduplicate Shared UI Primitives

4. **Split the 5 largest `remi/` Python monoliths (5,046 LOC combined)**
   - `remi/networks.py` (**1,307L**, 14 classes, 37 top-level functions): `NetworkIndex` (L888–L1277, 390L) + `PlaceholderCache` (L712–L834, 123L) + `ExtraPageProvider` + `LoraProvider` + disk scanner + filter/sort live in one module. **Split into** `remi/networks/` package: `models.py` (`NetworkEntry`), `providers.py`, `placeholders.py`, `index.py`.
   - `remi/ui/nodes.py` (**1,141L**, 33 classes, 15 functions): `Input` base (179L) + all primitive, choice, media, and container nodes in a single file. **Split into** `remi/ui/nodes_base.py`, `nodes_inputs.py`, `nodes_layout.py` (re-exporting all public names from `remi/ui/nodes.py` to preserve 100% API compatibility).
   - `remi/extensions_admin.py` (**922L**, 55 top-level functions): mixes git subprocess wrappers, SSRF-safe URL redirect validation (`_PublicRedirects`), remote index caching (`_IndexCache`), install/update operations, and config-state backup/restore. **Split into** `remi/extensions/` (`git_ops.py`, `index.py`, `config_states.py`, `admin.py`).
   - `remi/jobs/importers.py` (**865L**): separates cleanly into `SafeUnpickler` + legacy format readers (`legacy_readers.py`), `Mapper` (`mapper.py`), and `Importer` (`importers.py`).
   - `remi/jobs/queue.py` (**811L**, `JobQueue` alone **601L** L192–L792): extract scheduling/worker loop vs submission validation (`submission.py`) as prep for P2.1 `WorkerProtocol`.
   - **Small duplicated helpers inside `remi/`** to unify in a tiny `remi/_runtime_helpers.py` (or `remi/core/`):
     - `_main_entry` (**3×**: `remi/core/presets.py`, `remi/model_selection.py`, `remi/server/runtime.py`)
     - `_opts` / `_option` (**4×**: `remi/core/presets.py`, `remi/networks.py`, `remi/jobs/queue.py`)
     - `_generation_lock` (**2×**: `remi/model_selection.py`, `remi/pages/__init__.py`)
     - `_like_terms` (**2×**: `remi/files.py`, `remi/prompt_tools/store.py`), `_is_image` (**2×**: `remi/jobs/importers.py`, `remi/ui/nodes.py`), `iso` (**2×**: `remi/extensions_admin.py`, `remi/server/deps.py`).

5. **Consolidate duplicated `webui/src/` primitives & decompose >600L React files**
   - **Duplicated components/helpers (quick, zero-risk cleanup):**
     - `SearchBox` (**3×**): `webui/src/features/library/GalleryToolbar.tsx:L49`, `webui/src/features/networks/NetworkBrowser.tsx:L156`, `webui/src/features/prompt-tools/panels/panel.tsx:L17` → unify into a single `webui/src/components/SearchBox.tsx` (matching the 28px toolbar design system in [frontend.md](frontend.md)).
     - `Thumb` (**3×**): `webui/src/canvas/img2img/SourcePanel.tsx:L115`, `webui/src/ui/inputs/Image.tsx:L16`, `webui/src/widgets/image-list/index.tsx:L38` → extract shared `webui/src/components/FileThumb.tsx` with optional `onSize` callback.
     - `clamp` (**3×**): `webui/src/features/networks/grid-layout.ts:L37`, `webui/src/ui/values.ts:L6`, `webui/src/widgets/region-boxes/geometry.ts:L19` → import one `clamp` from `webui/src/lib/math.ts` (or `@/ui/values`).
     - Parallel image lists: `webui/src/ui/inputs/Image.tsx` (`ImageListInput`, **202L**) vs `webui/src/widgets/image-list/index.tsx` (`ImageListWidget`, **268L**) → share upload/paste/drop & thumbnail grid primitives.
   - **Oversized TypeScript production files (>500L):**
     - `webui/src/canvas/ui/EditorBody.tsx` (**761L**): extract `colorUtils.ts` (`cssToHex`), `ReplaceButton.tsx`, and toolbar/layer sub-components.
     - `webui/src/canvas/engine/document.ts` (**649L**) + `interaction.ts` (**455L**): split snapshot/history management (`history.ts`) from raster/layer mutation (`document.ts`).
     - `webui/src/pages/custom/json-builder/index.tsx` (**627L**): already defines 9 distinct components inline (`ResolutionSection`, `LoadPromptDialog`, `FileSection`, `SummarySection`, `ItemRow`, `ItemsSection`); split into sibling files under `pages/custom/json-builder/`.
     - `webui/src/features/networks/NetworkBrowser.tsx` (**600L**): extract `SortMenu`, `NetworkFilters`, and `EmptyState`/`ErrorState` into `features/networks/BrowserChrome.tsx`.
     - `webui/src/api/mock/fixtures.ts` (**842L**) + `server.ts` (**799L**) + `admin.ts` (**380L**) = **2,021L** mock harness: split `server.ts` by route domain (`jobs.ts`, `models.ts`, `generation.ts`) so contract edits touch small focused modules.

---

#### P2 — Remi-Owned Extension Monoliths & Upstream-Shared Guardrails

6. **Modularize Remi-owned built-in extension monoliths**
   - `extensions-builtin/sd-forge-prompt-workshop/scripts/prompt_workshop.py` (**2,511L**, `_hunt_gelbooru_family` **168L** L1667–L1834): largest non-vendored Python file in the entire repository; split tag-hunting scrapers, storage, and Remi page wiring into `lib_prompt_workshop/`.
   - `extensions-builtin/anima-artist-scheduled-mixer/scripts/anima_artist_scheduled_mixer.py` (**1,269L**): extract `params()` declaration & preset parsing into `lib_artist_mixer/`.
   - `extensions-builtin/sd_forge_anima_guidance/scripts/anima_guidance.py` (**933L**, `guidance_params` alone **228L** L596–L823): move declarative `params()` tree into `lib_anima_guidance/params.py` (matching `anima-lora-stage-scheduler/lib_stage_scheduler/params.py`).
   - `extensions-builtin/anima-lora-stage-scheduler/lib_stage_scheduler/runtime.py` (**851L**) and `extensions-builtin/anima-lora-layer-weight/scripts/anima_lora_layer_weight.py` (**816L**): extract schedule/weight parsing helpers into separate modules.

7. **Upstream-Shared Monoliths — Do NOT Refactor Locally (Governance Guardrail)**
   The following large files are shared with upstream Forge Neo. Refactoring or splitting them in Remi would create permanent whole-file merge conflicts on every sync and violate [remi-architecture-notes.md](../remi-architecture-notes.md):
   - `modules/processing.py` (**2,013L**, `process_images_inner` 342L, L900–L1241) — keep Remi additions to minimal 1–3 line hooks delegating into `backend/nn/dit_registry.py` or `remi/`.
   - `backend/memory_management.py` (**1,590L**, `load_models_gpu` 88L) — keep patches surgical and upstreamable (like Wave 1 PR #2).
   - `backend/loader.py` (**1,206L**, `load_huggingface_component` **565L**, L57–L621) — keep Remi model discovery/loading behind `dit_registry.py` 1–3 line hooks rather than expanding `load_huggingface_component`.
   - `backend/nn/svdq.py` (**1,060L**) & `backend/nn/qwen.py` (**483L**) — share 10 overlapping 8-line transformer blocks from upstream Nunchaku/ComfyUI; do not diverge locally.
   - `extensions-builtin/forge_legacy_preprocessors/annotator/` (OneFormer / Detectron2 / MMCV, multiple 850–1,350L files) — vendored upstream third-party code; leave untouched.
