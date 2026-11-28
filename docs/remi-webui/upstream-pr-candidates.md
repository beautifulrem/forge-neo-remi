# In-Tree Runtime Divergence Inventory & Code-Bloat Audit

> **Audit baseline:** `remi` in-tree runtime governance vs historical Forge Neo merge base `5998fb39`
> **Governance tool:** `python tools/upstream_forks.py budget [--max-forked N] [--target-diverged N] [--json] [--verbose]`
> **Policy reference:** [upstream-absorption.md](upstream-absorption.md) · [roadmap-2026-q4.zh-CN.md](../research/roadmap-2026-q4.zh-CN.md) (P1.4 Progressive Neo Decoupling)

---

## Executive Summary

1. **Progressive Neo decoupling policy (Zero Upstream Issues / PRs):**
   - Remi is progressively decoupling from Forge Neo (`neo`) into a self-contained codebase.
   - **No issues or pull requests are submitted to upstream Forge Neo.** All shared-origin runtime fixes (`remi-runtime-fix`) and architecture hooks (`remi-architecture-hook`) are owned, evolved, and maintained directly in Remi.
   - `python tools/upstream_forks.py budget` enforces 100% parity between code (`FORKED`, `DIVERGED`), documentation ([upstream-absorption.md](upstream-absorption.md)), category classifications, and on-disk `test_*.py` regression pinning tests (**52 / 52** verified on disk), without artificial `FORKED` or `DIVERGED` caps.
   - `upstream_crlf.txt`: **80** tracked CRLF files keep clean line endings (`tools/check_line_endings.py`).
2. **In-tree runtime divergence breakdown (`52` files):**
   - **33 `remi-runtime-fix` files:** Self-contained deterministic correctness, security, memory, and cross-platform fixes maintained directly in Remi (covering EXIF/path security, attention masks, flow schedulers/samplers, zero-denoise `sample_img2img`, off-GPU `sd_vae` backups + `loaded_vae_file` tracking, bounded `shell=False` `launch_utils` extension installers, IP-Adapter validation, ControlNet/LLLite/MultiDiffusion/Radial/LoRA-Schedule fixes, and atomic `styles.py`/`config_states.py` persistence).
   - **19 `remi-architecture-hook` files:** Shared-origin runtime modules hosting Remi's DiT caches, Anima 2.9B/3.8B hooks, Spectrum pre-head seams, Apple Silicon MPS hooks (`_MPSDequantCache`, FP8 CPU-staging fallback, `REMI_MPS_SDPA=1`), and `remi.ui` / `remi.model_selection` / `remi.realtime` integration.
3. **Code bloat & redundancy hotspots:**
   - The 5 `backend/nn/*_diffusers.py` modules total **1,415 lines** with heavily copy-pasted directory/sharding helpers and cross-family imports into `anima_diffusers.py`; `backend/nn/dit_registry.py` (**890 lines**) carries **29 thin lazy-import wrapper functions (115 lines)**; `_tensor_version` is duplicated **4×** and `_broadcast_ref_batch` **2×** across DiT models.
   - Five Remi Python modules exceed **800 lines** (`remi/networks.py` 1,307L, `remi/ui/nodes.py` 1,141L, `remi/extensions_admin.py` 922L, `remi/jobs/importers.py` 865L, `remi/jobs/queue.py` 811L), and five Remi-owned built-in extension scripts range from **816 to 2,511 lines**.
   - In `webui/src/` (626 files, 71,936 lines), `SearchBox`, `Thumb`, and `clamp` are each re-implemented **3×**, `ImageListInput` / `ImageListWidget` duplicate image-list rendering (470L combined), and four production views/engines exceed **600 lines** (`EditorBody.tsx` 761L, `document.ts` 649L, `json-builder/index.tsx` 627L, `NetworkBrowser.tsx` 600L).

---

## Part 1 — In-Tree Runtime Divergence Inventory (`52` Files)

### 1.1 Ownership Summary

| Category | Count | Scope & Maintenance Policy |
| --- | ---: | --- |
| **`remi-runtime-fix`** | **33** | Self-contained correctness, security, memory, and cross-platform fixes maintained directly in Remi (`0` upstream issues/PRs). |
| **`remi-architecture-hook`** | **19** | Shared-origin runtime modules hosting Remi DiT caches, Anima 2.9B/3.8B hooks, Spectrum pre-head seams, Apple Silicon MPS hooks, and `remi.*` integration. |
| **Total `DIVERGED`** | **52** | **52 / 52** verified on disk with `test_*.py` regression pinning tests (`python tools/upstream_forks.py budget`). |

---

### 1.2 In-Tree Runtime Fixes (`remi-runtime-fix`, 33 Files)

Every file below carries a deterministic bug, security, memory, or platform fix maintained directly in Remi (for built-in scripts that also replaced `ui()` with `params()`, that UI port is tracked by [upstream-absorption.md §1](upstream-absorption.md#1-where-upstream-files-went)). Diff sizes are measured via `git diff --numstat 5998fb39 HEAD -- <path>`.

#### Group A — Core Security, Attention, Samplers & Engine Correctness (19 Files)

| # | File | Diff vs `5998fb39` | Deterministic Fix Summary | Pinning Test | Priority / Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `modules/images.py` | +6 / -25 (fix: 4L) | **Security:** verify `exif_data` is `bytes` starting with JPEG/TIFF/Exif magic bytes before `piexif.load()`, preventing uploaded EXIF chunks from being opened as arbitrary server file paths, and guard non-`bytes` EXIF `UserComment` values. | `tests/test_remi_security.py` | **P0 Security** — pinned by `test_remi_security.py` & `test_remi_adversarial.py`. |
| 2 | `backend/attention.py` | +15 / -4 | Extend `attention_basic` bool mask reshaping for 2D `(Q, K)` and 3D/4D boolean masks, and reshape `(B, K)` padding masks to `(B, 1, 1, K)` in `attention_pytorch`. | `tests/test_remi_qwen21.py` | **P0 Correctness** — fixes fallback and SDPA mask crashes. |
| 3 | `modules_forge/packages/k_diffusion/external.py` | +6 / -2 | `ForgeScheduleLinker.get_sigmas`: space predictor timesteps between `sigma_to_t(sigmas[-1])` and `sigma_to_t(sigmas[0])` instead of `999..0` (preventing division by zero / black outputs on flow predictors). | `tests/test_flow_schedules.py` | **P0 Correctness** — affects every flow model using Uniform scheduler. |
| 4 | `modules/sd_samplers.py` | +17 / -0 | Refuse VP/VE samplers (`DDIM`, `PLMS`, `Kohaku LoNyu Yog`) in `create_sampler()` when `predictor.prediction_type == "const"`, matching `no_sdxl` validation. | `tests/test_flow_sampler_support.py` | **P0 UX/Correctness** — prevents garbled outputs on Flux/Anima/Qwen. |
| 5 | `modules/sd_samplers_common.py` | +23 / -9 | In `apply_refiner`: call `clear_prompt_cache()` before `setup_conds()`, call `calculate_shorthash()`, record `"Refiner CFG scale"`, and guard `denoising_strength <= 0` in `setup_img2img_steps()`. | `tests/test_refiner_switch.py` | **P0 Correctness** — fixes broken refiner conditioning reuse and zero-strength img2img. |
| 6 | `modules/sd_samplers_kdiffusion.py` | +6 / -0 | Short-circuit `sample_img2img()` when `denoising_strength <= 0` or `len(sigma_sched) < 2`, avoiding `sigma_sched[-2]` `IndexError` on `discard_next_to_last_sigma` samplers and zero-denoise noise injection. | `tests/test_remi_ml_runtime_fixes.py` | **P1 Correctness** — guards k-diffusion img2img zero-denoise path. |
| 7 | `modules/sd_vae.py` | +22 / -4 | Offload `base_vae` backups to CPU (`v.detach().cpu().clone()`), guard `None` models and repeated `reload_vae_weights()` calls from clobbering `base_vae`, and maintain `loaded_vae_file` across override/restore for `p.sd_vae_name`/`[vae]`. | `tests/test_remi_ml_runtime_fixes.py` | **P1 Memory + Metadata** — prevents doubled GPU VAE memory on `_vae_override`. |
| 8 | `modules/launch_utils.py` | +56 / -10 | Run extension `install.py` via `subprocess.Popen(..., shell=False, stdin=DEVNULL)` with `PIP_NO_INPUT=1`, `GIT_TERMINAL_PROMPT=0`, process-group timeout (`REMI_INSTALLER_TIMEOUT`, default `300s`), and exit-code reporting; bound `git_pull_recursive`. | `tests/test_remi_adversarial.py` | **P1 Security + Reliability** — bounds installer subprocesses with `shell=False`. |
| 9 | `extensions-builtin/forge_preprocessor_tile/scripts/preprocessor_tile.py` | +7 / -2 | Broadcast per-image sigmas across chunks (`row_sigmas`) in `PreprocessorTileColorFix` so batch size ≥ 2 no longer crashes. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P0 Crash Fix** — 7-line batched tile fix. |
| 10 | `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py` | +81 / -103 (fix: 1L) | Replace `masks_for_overlay[i] = converted_mask` with `masks_for_overlay.append(converted_mask)` in `apply_masks()` (fixes `IndexError` on empty list). | `extensions-builtin/soft-inpainting/tests/test_soft_inpainting_params.py` | **P0 One-liner** — rest of diff is standard `params()` UI port. |
| 11 | `backend/diffusion_engine/qwen.py` | +3 / -2 | Format per-item prompts in `QwenImage.get_learned_conditioning_with_image()` instead of flattening batched prompts into a single string. | `tests/test_remi_ml_runtime_fixes.py` | **P0 Correctness** — 3-line batched conditioning fix. |
| 12 | `backend/patcher/vae.py` | +6 / -0 | Copy `upscale_ratio`, `upscale_index_formula`, `downscale_index_formula`, `latent_dim`, and `pad_channel_value` in `VAE.clone()`. | `tests/test_remi_ml_runtime_fixes.py` | **P0 Correctness** — prevents lost VAE geometry metadata on clone. |
| 13 | `backend/operations.py` | +30 / -11 | Restore `(current_device, current_dtype, current_manual_cast_enabled)` on `using_forge_operations` exit; avoid offload stream sync on resident weights; generalize `fp8_linear` ndim. | `tests/test_remi_ml_runtime_fixes.py` | **P0 State Leak** — prevents FP8 dtype leaks across model loads. |
| 14 | `modules/ngrok.py` | +15 / -2 | Redact ngrok authtoken, `NGROK_AUTHTOKEN`, basic auth, and secret-like `--ngrok-options` keys in error messages. | `tests/test_remi_ngrok_token.py` | **P0 Security** — credential redaction guard. |
| 15 | `modules/config_states.py` | +16 / -5 | Disambiguate same-second `list_config_states()` keys; fix `restore_extension_config()` to set `ext.enabled` (not non-existent `ext.disabled`) and preserve disabled built-ins. | `tests/test_remi_config_states_restore.py` | **P1 Correctness** — fixes broken extension enablement restore. |
| 16 | `backend/diffusion_engine/krea.py` | +46 / -13 | Use `blake2b` + `(shape, dtype)` `_vision_cache_key`, clear `ini_latent`/`ref_latents` on `clear_references()`, expand 1-channel references, and ground edit conditioning. | `tests/test_remi_ml_runtime_fixes.py` | **P1 Correctness** — eliminates stale `id()` vision cache hits. |
| 17 | `backend/diffusion_engine/qwen21.py` | +55 / -11 | Use `blake2b` + `(shape, dtype)` + `reference_max_side` `_vision_cache_key`, preserve 4-ch RGBA in `encode_vision`, guard `use_dynamic_shifting`, and reset state in `clear_references()`. | `tests/test_remi_ml_runtime_fixes.py` | **P1 Correctness** — fixes stale `id()`/`data_ptr()` vision cache hits and RGBA encoding. |
| 18 | `extensions-builtin/sd_forge_controlnet/lib_controlnet/api.py` | +9 / -3 | Clamp `/controlnet/detect` thresholds/resolution to preprocessor slider bounds and return HTTP 422 on unknown modules instead of `KeyError` (500). | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 Security/API** — prevents unbounded resize DoS. |
| 19 | `extensions-builtin/sd_forge_ipadapter/scripts/forge_ipadapter.py` | +24 / -12 | Remove duplicate `sorting_priority`, support `.safetensors`/`.sft`/flat `ip_adapter.*` dicts via exact `removeprefix` in `try_build_from_state_dict`, validate `cond` dict in `process_before_every_sampling`, and guard mask channel squeeze. | `extensions-builtin/sd_forge_ipadapter/tests/test_ipadapter_patcher.py` | **P2 Correctness** — hardens IP-Adapter state dict and mask handling. |

#### Group B — Built-in Extensions, Persistence & Guaranteed Teardown (14 Files)

| # | File | Diff vs `5998fb39` | Deterministic Fix Summary | Pinning Test | Priority / Notes |
| --- | --- | --- | --- | --- | --- |
| 20 | `extensions-builtin/sd_forge_lora_schedule/scripts/control.py` | +51 / -26 | Move `available_network_aliases` lookup inside `try`, reject malformed schedules atomically, scan `p.all_prompts`, clear `LoRAControl.mapping` in `postprocess`/`cleanup`, guard `t2 == t1` and `has_online_lora`. | `extensions-builtin/sd_forge_lora_schedule/tests/test_lora_schedule_params.py` | **P1 Correctness** — prevents stale mapping and division by zero. |
| 21 | `modules/img2img.py` | +190 / -87 (fix: 4L) | Wrap `parse_generation_parameters(geninfo)` in `process_batch` with `try/finally` so a malformed batch image's PNG Info never leaves `shared.opts.infotext_styles` stuck at `"Ignore"`. | `tests/test_img2img_batch_png_info.py` | **P1 State Leak** — rest of diff is §1 `run_img2img` entrypoint port. |
| 22 | `modules/extras.py` | +88 / -52 (fix: 3L) | Refuse a Checkpoint Merger output filename whose resolved `realpath` matches any input checkpoint being merged (prevents corrupting an input model mid-merge). | `tests/test_remi_merger.py` | **P1 Data Safety** — rest of diff is §1 merger UI-contract translation. |
| 23 | `modules/infotext_utils.py` | +28 / -278 (fix: 12L) | Fix `"Hires Module 1"` paste handling (`"Built-in"`, `"Use same choices"`, missing modules) and linear-time atomic-group `re_param_code` (eliminating multi-minute backtracking on 200k-char prompt lines). | `tests/test_infotext_paste_roundtrip.py` | **P1 Correctness + DoS** — also covered by `tests/test_remi_image_meta.py`. |
| 24 | `extensions-builtin/forge_preprocessor_inpaint/scripts/preprocessor_inpaint.py` | +22 / -4 | Store `inpaint_only_image`/`mask` per unit on `params` (two units no longer clobber each other); blend using `hr_image`/`hr_mask` when a unit skipped the hires pass; interpolate mask/raw to output size instead of downscaling hires output. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 Correctness** — multi-unit & hires inpaint fix. |
| 25 | `extensions-builtin/forge_preprocessor_reference/scripts/forge_reference.py` | +16 / -5 | Wrap style-recording forward pass in `try/finally: self.is_recording_style = False`; add `process_after_every_sampling` + `release()` to drop recorded GPU self-attention tensors after sampling. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 VRAM Leak** — stops GPU tensor retention across generations. |
| 26 | `extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite.py` | +41 / -4 | Add `sampler_progress(process)` and `step_active(progress, start_percent, end_percent)` so LLLite step windows track sampler steps rather than raw forward calls (fixes img2img, Hires, MultiDiffusion, 2nd-order samplers). | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 Correctness** — purely additive optional `progress=None` parameter. |
| 27 | `extensions-builtin/sd_forge_controlnet/lib_controlnet/global_state.py` | +44 / -2 | Add cached safetensors key inspection (`control_model_family`) so `get_filtered_controlnet_names()` only offers Anima/DiT control models on matching DiTs and UNet ControlNets on UNet checkpoints. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | **P1 UX/Correctness** — uses `safetensors` key inspection. |
| 28 | `modules_forge/supported_controlnet.py` | +24 / -4 | Key `_CONTROL_MODEL_CACHE` by `(realpath, mtime_ns, size, *dtypes)` via `cache_key()` so replaced files / changed dtypes reload cleanly, and honor `control_net_model_cache_size == 0`. | `tests/test_controlnet_model_cache.py` | **P1 Cache Bug** — 2 one-line base-class hooks remain as normal §1 hooks. |
| 29 | `extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py` | +60 / -132 (fix: 18L) | Track `cached_model = weakref.ref(p.sd_model)` so references re-encode after model unload/reload; preserve 4-channel RGBA references for `dynamic_args.qwen21` and hash alpha channel in `hash_image`. | `extensions-builtin/sd_forge_image_stitch/tests/test_image_stitch_params.py` | **P1 Correctness** — also pinned by `tests/test_remi_qwen21.py`; rest is `params()` UI port. |
| 30 | `extensions-builtin/sd_forge_multidiffusion/lib_multidiffusion/tiled_diffusion.py` | +61 / -13 | Derive latent scale factor from `vae.downscale_ratio` (with `processing.opt_f`/8 fallback) instead of hardcoding `8` in `process_controlnet`, and chain prior `model_function_wrapper` hooks. | `extensions-builtin/sd_forge_multidiffusion/tests/test_multidiffusion_params.py` | **P1 Correctness** — supports 16× VAEs and wrapper composition. |
| 31 | `modules/styles.py` | +63 / -43 | Guard `StyleDatabase` writes with `RLock` + atomic temp/`fsync`/`os.replace` (preserving CRLF) and handle missing/`None` CSV fields safely in `load_from_csv()`. | `tests/test_remi_styles_save.py` | **P1 Data Safety** — prevents truncated/corrupt `styles.csv` on concurrent writes. |
| 32 | `extensions-builtin/sd_forge_radial/scripts/forge_radial.py` | +41 / -21 (fix: 14L) | Match batched `timestep[0]` against `sigmas`, chain prior `model_function_wrapper`, and restore `wan.attention = ORIG_ATTENTION` in `try/finally` and `cleanup()`. | `extensions-builtin/sd_forge_radial/tests/test_radial_params.py` | **P2 Leak + Crash** — guaranteed teardown via `Script.cleanup`. |
| 33 | `modules/processing_scripts/refiner.py` | +76 / -89 (fix: 10L) | Clear `p.refiner_cfg = None` when disabled in `setup()`, handle LoRA replacements safely under frozen settings, and call `restore_base_weights()` from `cleanup()`. | `tests/test_refiner_switch.py` | **P2 State Corruption** — guaranteed teardown via `Script.cleanup`. |

---

### 1.3 Remi Architecture Hooks (`remi-architecture-hook`, 19 Files)

These 19 files carry Remi-only DiT architecture hooks, Anima 2.9B/3.8B features, Spectrum pre-head seams, MPS runtime hooks, or `remi.*` integration alongside embedded runtime fixes:

| # | File | Diff vs `5998fb39` | Why Classified as `remi-architecture-hook` | Pinning Test | Embedded Runtime Fixes |
| --- | --- | --- | --- | --- | --- |
| 1 | `backend/nn/krea.py` | +176 / -37 | Step-invariant `_txt_cache`/`_ref_cache`/`_freqs_cache` (`weakref` weight probes), `single_block` SLG replace hook, Spectrum `last_input`/`compute_last` seam. | `tests/test_remi_krea2.py` | `TextFusionBlock` out-of-place `x = x + ...` (fixes cached conditioning mutation) + `_broadcast_ref_batch` & odd-dim unpad. |
| 2 | `backend/nn/qwen21.py` | +403 / -11 | Lossless block-causal prefix KV cache (`weakref` weight probes, VRAM budget guard, eager prefill for per-block compile), target RoPE cache, Spectrum `final_input`/`compute_final` seam. | `tests/test_remi_qwen21.py` | 4D `(1, 1, Q, K)` causal mask + `_broadcast_ref_batch` safe reference broadcast. |
| 3 | `backend/sampling/sampling_function.py` | +62 / -5 | Anima-3.8B `_ANIMA_SEMANTIC_KEYS` batching, `batch_indices`, `edit_strength` forwarding, `moss_ref_kv` / `dit.clear_caches()` cleanup. | `tests/test_sampling_batch_indices.py` | `merges_by_free_memory(device)` (stops nondeterministic MPS/CPU cond/uncond batch merging causing image drift). |
| 4 | `backend/text_processing/anima_engine.py` | +161 / -33 | Remi `PromptChunk`, raw tokenizers for NegPiP/FreeFuse, Anima-3.8B Qwen3.5 semantic connector inputs (`semantic_inputs`). | `tests/test_anima_text_engine_compat.py` | Treat parser `BREAK` chunk marker as a space rather than literal `"BREAK"` at weight `-1`. |
| 5 | `backend/memory_management.py` | +124 / -12 | Apple Silicon MPS `_same_device`, `mps_memory_info()`, cache-clear hooks, `REMI_MPS_SDPA=1` opt-in gate, and MPS FP8 CPU `cast_to` fallback. | `tests/test_memory_management_oom.py` | Guard `discard_cuda_async_error` on CUDA + `@functools.lru_cache` on `mac_version()`. |
| 6 | `backend/diffusion_engine/zimage.py` | +51 / -0 | Imports `backend.nn.dit_capabilities_ext` and hosts Omni/Edit `ref_latents`/`ini_latent` hooks + `dynamic_args.zimage` reset. | `tests/test_remi_ml_runtime_fixes.py` | Reset `dynamic_args.zimage = False` on `clear_references()` and no-ref paths. |
| 7 | `backend/loader.py` | +263 / -20 | Hosts Remi DiT/Diffusers loading, `backend.nn.anima_quant` on-load quantization, `dit_registry` hooks, and 3.8B connector validation. | `tests/test_remi_ml_runtime_fixes.py` | Guard unsupported `float8_e4m3fn`/`float8_e5m2` `storage_dtype` on MPS (`load_huggingface_component`). |
| 8 | `backend/operations_mixed_precision.py` | +274 / -23 | Implements Remi `_MPSDequantCache` plus MPS CPU-dequant load guard and FP8 `Embedding._forward` CPU fallback. | `tests/test_remi_ml_runtime_fixes.py` | CPU-dequant fallback in `_load_quantized_module` when `not supports_cast(device, dtype)`. |
| 9 | `extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite_anima.py` | +82 / -15 | Imports Remi-only `backend/nn/anima_layout.py` for 2.9B/3.8B depth remapping. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | Chain-unhook `restore()` fix (2 units on same `Linear`), `_check_width` mismatch error, `step_active` window, once-per-sampling skip warning. |
| 10 | `extensions-builtin/sd_forge_controlllite/scripts/forge_controllllite.py` | +263 / -11 | Registers Remi-only `AnimaControlLoraPatcher`, `AnimaControlVacePatcher`, and `_request_spectrum_window`. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | Copy control image before applying soft mask (prevents compounding across batches/hires) + follow Hires checkpoint DiT. |
| 11 | `extensions-builtin/sd_forge_controlnet/scripts/controlnet.py` | +148 / -49 | Carries Remi `unit_param`/`unit_actions` UI, `process_before_conditioning` text-patch hook (Anima control-LoRA), and `cleanup()` teardown. | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` | Guard `control_net_disabled` & missing `current_params` / `preprocessor` hooks; remove failed unit from infotext. |
| 12 | `extensions-builtin/sd_forge_lora/networks.py` | +285 / -55 | Imports `anima_layout` & `RESOLVED_LORA_DEPTHS`, mirrors LLM adapter LoRA into Anima-3.8B DiT, tracks `anima_sources`/`remap_notes`/`anima_inserted_blocks_policy`. | `extensions-builtin/sd_forge_lora/tests/test_networks_apply_failure.py` | `LoraNotApplied` cleanup so a failed LoRA apply does not leave the previous LoRA set active and re-reports failures on cached hashes. |
| 13 | `modules/api/api.py` | +414 / -183 | Rewritten around `remi.ui` layouts, `remi.model_selection` deferred model switching, DNS-pinned `fetch_url_image_bytes`, and `modules.redact` flag redaction. | `tests/test_api_pasted_value.py` | `pasted_value` overflow guard (`Steps: 1e999`), `data:image/` `IndexError` fix, `--api-auth` colon parsing, `stop_webui` cleanup. |
| 14 | `modules/initialize.py` | +14 / -23 | Gradio-free initialization and `remi.extensions_admin.record_startup_load_times()` hook. | `tests/test_remi_config_states_restore.py` | Re-run `extensions.list_extensions()` after `restore_config_state_file()` when restoring. |
| 15 | `modules/options.py` | +306 / -52 | `Options.hold_for_run`, `REMI_TEST_HELPER` / `_REPOSITORY_SETTINGS` test isolation guard, atomic `write_whole` save, `RLock` sync, and `cast_value` bool parsing. | `tests/test_remi_options_save.py` | Atomic `write_whole` save + corrupt JSON load recovery + `cast_value` bool string fix. |
| 16 | `modules/processing.py` | +108 / -25 | Core runtime hub carrying Remi's `after_model_load`/`cleanup` hooks, `keep_negative_conds` (Anima NAG), `collect_workflow_advice`, and Qwen-2.1 RGBA pipeline. | `tests/test_prompt_cache_clear.py` | `fill_latent_noise` 5D broadcast bug, 3-slot `cached_hr_*` reset in `close()`, `clear_prompt_cache`, `ini_latent` clear/restore, `skips_negative_conds`. |
| 17 | `modules/progress.py` | +29 / -44 | Gradio-free `/internal/progress` (`progressapi`) delegating live preview snapshots and animated seeks to `remi.realtime`. | `tests/test_remi_server_ws.py` | Gradio removal + `remi.realtime` integration + `shared.state` zero-division guard. |
| 18 | `modules/sd_schedulers.py` | +78 / -12 | Adds Remi's `beta57_scheduler`, `anima_flow_match_scheduler`, `all_schedulers_map`, and `n <= 0` guards across schedulers. | `tests/test_flow_schedules.py` | `turbo_scheduler` sigma lookup fix + `n <= 0` scheduler guards. |
| 19 | `modules_forge/packages/comfy/lora.py` | +101 / -25 | Adds `SplitTargets`/`SlicedWeightAdapter` and Qwen-2.1/MOSS/Krea2/FLUX.3 multi-trainer LoRA key dispatchers. | `tests/test_anima_lora_formats.py` | 4-line mutable-default bug fix (`key_map=None` in `model_lora_keys_clip` / `model_lora_keys_unet`). |

### 1.4 In-Tree Maintenance Checklist

When modifying any shared-origin runtime file in `backend/`, `modules/`, `modules_forge/`, or `extensions-builtin/`:
1. Implement the root-cause fix directly in Remi — do **not** open upstream issues or PRs, and do not defer needed fixes for upstream parity.
2. Preserve CRLF line endings on files listed in `tools/upstream_crlf.txt` (`python tools/check_line_endings.py`).
3. Add or extend a `test_*.py` regression pinning test on disk.
4. Register newly diverged shared-origin files in `DIVERGED` & `DIVERGED_CLASSIFICATION` (`tools/upstream_forks.py`) and [upstream-absorption.md §1.2](upstream-absorption.md#12-upstream-bugs-remi-fixes-divergences), then verify `python tools/upstream_forks.py budget`.

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
   - **Recommendation:** Add a 6-line `_lazy(module_path, attr)` descriptor helper in `dit_registry.py` (replacing all 29 wrapper functions), consolidate `*_workflow_advice` into declarative rule predicates aligned with P0.2 `AdviceItem`, and hoist `_tensor_version` + `_broadcast_ref_batch` into a tiny `backend/nn/_dit_common.py`.

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
     - `SearchBox` (**3×**): `webui/src/features/library/GalleryToolbar.tsx:L43`, `webui/src/features/networks/NetworkBrowser.tsx:L164`, `webui/src/features/prompt-tools/panels/panel.tsx:L17` → unify into a single `webui/src/components/SearchBox.tsx` (matching the 28px toolbar design system in [frontend.md](frontend.md)).
     - `Thumb` (**3×**): `webui/src/canvas/img2img/SourcePanel.tsx`, `webui/src/ui/inputs/Image.tsx`, `webui/src/widgets/image-list/index.tsx` (each a local `function Thumb`) → extract shared `webui/src/components/FileThumb.tsx` with optional `onSize` callback.
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

7. **Shared-Origin Runtime Monoliths — Refactor Surgically via Seam Extraction**
   The following large runtime modules originated from Forge Neo (`5998fb39`). As Remi progressively decouples, extract new behavior into focused modules under `remi/` or `backend/nn/` behind clean seams rather than growing these monoliths further:
   - `modules/processing.py` (**2,013L**, `process_images_inner` 342L, L900–L1241) — delegate new model/workflow capabilities into `backend/nn/dit_registry.py` or `remi/`.
   - `backend/memory_management.py` (**1,590L**, `load_models_gpu` 88L) — keep device/offload fixes focused and covered by deterministic pinning tests.
   - `backend/loader.py` (**1,206L**, `load_huggingface_component` **565L**, L57–L621) — keep Remi model discovery/loading behind `dit_registry.py` hooks rather than expanding `load_huggingface_component`.
   - `backend/nn/svdq.py` (**1,060L**) & `backend/nn/qwen.py` (**483L**) — share 10 overlapping 8-line transformer blocks from Nunchaku/ComfyUI; keep model-specific kernel code isolated.
   - `extensions-builtin/forge_legacy_preprocessors/annotator/` (OneFormer / Detectron2 / MMCV, multiple 850–1,350L files) — vendored third-party annotator code; leave untouched unless fixing a concrete bug.
