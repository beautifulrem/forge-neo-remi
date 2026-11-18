# Remi upstream sync — 2026-11-18

This sync merges Forge Neo `neo` at
[`89e2e09d`](https://github.com/Haoming02/sd-webui-forge-classic/commit/89e2e09d38b56c4f15558f06ee3773045e44f97e)
into `remi`. The previous merge base was `c6c2b43d`
([2026-10-27](upstream-sync-2026-10-27.md)). It follows
[upstream-absorption.md](remi-webui/upstream-absorption.md): the runtime is a
Git merge, and the changes to upstream's Gradio UI are translated into
settings widgets, server code and SPA code. All 51 upstream commits are
classified below. `tools/upstream_forks.py review` reported no upstream
commit touching a Remi fork.

Notable for users:

- **GGUF** is back: the vendored `modules_forge/packages/gguf` reader is
  replaced by the `gguf` package (`gguf==0.19.0` in `requirements.txt`) and a
  Triton K-quant dequantizer (`kquant_triton.py`, used where Triton exists).
- **Anima RGBA** (transparent images) with its
  [LoRA and VAE](https://huggingface.co/localai0301/Anima-RGBA); for img2img
  of a transparent image set `img2img_background_color` to `(255, 255, 255)`.
- **Reference images** of every Edit model (img2img and ImageStitch) are now
  resized by one setting, **Settings → Stable Diffusion → Maximum Side Length
  for Reference Images** (`reference_max_side`, default 1024, 0 = no limit).
  It replaces ImageStitch's own "Maximum Side Length" slider and the
  `qwen_vae_resize` setting. An `/sdapi` call that still sends ImageStitch's
  third argument is accepted; the value is ignored.
- **Checkpoint cards apply preferred VAE / text encoders**: the checkpoint
  metadata editor has a "Preferred VAE / text encoders" field (sidecar key
  `vae_te`, the same key upstream writes), and loading the checkpoint from its
  card also selects those modules.
- **Extensions → Available** loads the third-party
  [Forge Neo extension index](https://github.com/eduardoabreu81/forge-neo-extensions)
  by default (`WEBUI_EXTENSIONS_INDEX` still overrides it) and shows a "2025+"
  filter, on by default, that hides extensions without a commit after
  2025-01-01.
- New launch options `--disable-preset …`, `--sort-preset` and `--force-w4a8`.
- `docker/Dockerfile.rocm`: the Remi image for AMD GPUs.
- **`--directml` was removed** upstream; a launch command that still passes it
  stops with an argparse error.
- **CodeFormer** needs `spandrel_extra_arches`, which is no longer in
  `requirements.txt`: install it by hand to keep using CodeFormer face
  restoration (GFPGAN is unaffected).
- Dependency bumps: GitPython 3.2.1, comfy-kitchen 0.2.41, protobuf 6.33.6.

## Integrated as upstream

Merged unchanged, or with conflicts that only kept Remi's hooks:

| Upstream commit | Change |
| --- | --- |
| `ed74b625` amd, `57c7977f` cu130 | No fp8 compute on AMD gfx11; a CUDA 13 runtime check on NVIDIA (`backend/memory_management.py`) |
| `c30f02e0`, `be910116`, `865255e6`, `753a40d7` opt; `398279fe` dtype | Moss-Image, mixed-precision, Wan / Qwen 2.1 VAE (new `backend/nn/_wan_cache.py`) and Krea speed-ups; dtype fixes |
| `49bc706b`, `2d79fc2f` batching | Cond / uncond batching of conditions with different lengths (`backend/sampling/`, `modules/prompt_parser.py`) |
| `b72495f7` fallback, `60479cb1` meta, `fa5ffcd5` a8, `9653c15b` nvfp4, `14223a4b` pin | Operations fallbacks, Nunchaku detection, `--force-w4a8`, nvfp4 and GGUF pinning fixes |
| `b59500ed` precision, `96e8fec2` device, `51c0b26a` safe, `4f4284b0` head | Attention precision, Qwen vision tower on the text-encoder device, VAE config fixes |
| `10d7bc5e` Anima RGBA (backend) | RGBA VAE output (`backend/loader.py`, `backend/nn/wan_vae.py`, `backend/patcher/vae.py`). The send-to-img2img change in `modules/infotext_utils.py` is already Remi's behaviour: the img2img image input is RGBA (`remi/core/forms.py`) |
| `6017d53d` gguf, `ad4d06b8`, `24761676` bump, `cda2e50b` extra | GGUF loader on the `gguf` package; requirements as upstream (still no Gradio); `spandrel_extra_arches` imported only by CodeFormer |
| `caaa2cca` references (backend) | `backend/misc/image_resize.resize_reference`, `ForgeDiffusionEngine.encode_first_stage(x, *, multiple)`, reference handling in every engine |
| `42c1a92f` sorting | Model, module and LoRA lists natural-sorted where they are built (`modelloader`, `sd_models`, `sd_vae`, LoRA `networks`); the `sorted()` calls downstream removed, also in Remi's `params()` of PiD and Refiner |
| `936d9dac` xyz: lora, `980cd471` xyz: module, `7d6fd36c` shush, `197ea3c0` oops | X/Y/Z: **Prompt S/R (LoRA)** axis (labels `name:weight`), VAE and **Text Encoder** axes through `override_settings`, no loading logs for each grid cell (`disable_log`). `tests/test_scripts_xyz_grid.py` follows |
| `286a244e` duplicate | Duplicate `ScriptRunner.process_before_every_sampling` removed |
| `1b5c396b`, `79b4104c`, `197bb4ff` yeet | `EnumAction`, `--directml` and the "might be outdated" extension notice removed. `tests/test_quant_platform_policy.py` no longer stubs `is_directml_enabled` |
| `dc86121b` check | The launcher's GPU test uses `torch.accelerator` |
| `a31a8ba0` onnx | Legacy preprocessors' onnxruntime install |
| `8406f5d0` log | Log lines no longer break tqdm bars |
| `11a5f14c` module, `89e2e09d` permission | Unknown modules in loading parameters skipped; an unreadable checkpoint is reported and skipped instead of stopping the scan |
| `308fa6f9` donacdum | `huggingface_guess` model list |
| `d4a4265f` lint (`modules/sysinfo.py`) | Formatting merged; Remi's `redact.url` of the remote kept |
| `2c25fd9c` update README | Forge Neo version 2.32 (`forge_version.py`); the README additions go into Remi's READMEs (en, zh-CN) |

## Translated into the Remi UI

| Upstream commit / file | Remi change |
| --- | --- |
| `caaa2cca` references, `507e66fb` donacdum (`image_stitch.py`) | `params()` drops the "Maximum Side Length" slider; `process(p, enable, references, *_legacy)` takes upstream's cache key (all reference settings) and loop. `_EDIT_MODELS` lists Moss Image Edit. `extensions-builtin/sd_forge_image_stitch/tests/test_image_stitch_params.py` and the legacy fixtures (`tests/fixtures/legacy_ui_{txt2img,img2img}.json`) drop the slider |
| `caaa2cca` settings | `reference_max_side` as `ui.Slider` (0–2048, step 256) replaces `qwen_vae_resize`; zh_CN labels; settings contract regenerated |
| `1f2f068d` checkpoints (`ui_extra_networks_checkpoints*.py`, `javascript/ui.js`) | `remi/network_metadata.py` `preferred_modules()` and validation of `vae_te` (file names, at most 16; upstream's old single `vae` key is read too); the checkpoints kind offers the field and items carry `preferred_modules`; `webui/src/features/networks/EditMetadataDialog.tsx` "Preferred VAE / text encoders" checklist ("Use the current selection", "Clear"); `actions.ts` `selectCheckpoint()` writes the checkpoint and the modules in one settings patch (no preferred modules: the selection is kept). Upstream's `sd_arch` key is not adopted: Remi's existing `sd version` field is the architecture |
| `6a8cbc01` rounding (`javascript/dimRounding.js`) | Typed integer slider values snap to the step on commit (`webui/src/ui/inputs/Slider.tsx`), so Width / Height become multiples of the resolution step (`res_step`) as upstream rounds them on blur. "Use the image's size" already rounded |
| `e67193d9` rewrite Extensions tab (`modules/ui_extensions.py`) | `remi/extensions_admin.py` `DEFAULT_INDEX_URL` is the Forge Neo index; the Available view says the index is maintained by third parties, and has the "2025+" toggle (`isRecent` / `RECENT_SINCE` in `extensions-model.ts`, on by default). Upstream's or / and tag matching is Remi's existing any / all |
| `7f00fbb4` presets (`modules_forge/shared.py`, `presets.py`) | `--disable-preset` / `--sort-preset` merged; `remi/core/presets.ui_preset_choices()` applies them to the model bar's preset list, `/api/v2` models and preset validation |
| `fc69a4f7` lora (`extra_networks_lora.py`, `networks.py`) | Upstream's failed-LoRA cache (`failed_networks`, cleared on rescan) and per-LoRA reasons merged into Remi's reporting: `load_networks()` returns name → reason, and every generation that names a failed LoRA still gets the job warning "LoRA not found or failed to load: name (reason)" (the SPA and e2e rely on this text); upstream's later "LoRAs with Error" comment is folded into it |
| `2900d594` docker | `docker/Dockerfile.rocm` is written like Remi's `docker/Dockerfile` (SPA built in a Node.js stage, this checkout copied, no clone, data in `/home/remi/data`), with the ROCm PyTorch wheels (`TORCH_INDEX=rocm7.2`); `docker/README.md`, both READMEs and `tests/test_remi_docker.py` cover it |

## Adapted for Remi

| File | Resolution |
| --- | --- |
| `modules_forge/main_entry.py` (`980cd471`, `11a5f14c`, `7f00fbb4`) | Upstream's `refresh_model_loading_parameters(*, refresh, disable_log)` and sorted module list merged into Remi's structure; Anima's online-LoRA setting hook kept |
| `modules_forge/presets.py` | The disabled-preset lookup tolerates a `cmd_opts` without the flag (test environments that do not load `modules_forge.shared`) |
| `extensions-builtin/sd_forge_anima_freefuse/tests/test_patch_abi.py` | Imports the `gguf` package instead of the removed vendored module |
| `pyproject.toml` (`156f5faf` project) | Remi's kept: its ruff configuration is the gate of `tools/check.sh`; upstream's `[project]` metadata names the upstream repository |
| `README.md`, `README.zh-CN.md`, `docker/README.md` | Remi's kept; added Anima RGBA, GGUF, AMD Docker, the extension index, preferred modules and a "Models and presets" table of launch options |
| `backend/nn/anima.py`, `backend/diffusion_engine/anima.py` (`caaa2cca`) | Anima stacks its references along the frames, so they must have the generation's size: upstream's `reference_max_side` resize broke ImageStitch on Anima (`Sizes of tensors must match except in dimension 2`). The img2img image keeps its own latent as the first reference; other references are fitted to the generation by `fit_reference` (`tests/test_anima_reference_size.py`) |
| NegPiP (`lib_negpip/anima.py`), `modules/anima_support.py` (`2d79fc2f`) | No logic change: `compile_conditions` now groups rows of different lengths and recurses through the hooked function. Tests and comments follow the per-row conditions |
| `backend/diffusion_engine/qwen.py`, `backend/nn/qwen21_vae.py` | Unused / duplicate imports from upstream removed for Remi's ruff gate |
| XYZ plot (`980cd471`, `936d9dac`) | "Text Encoder" and "Prompt S/R (LoRA)" axes added to the zh_CN translation and the expected axis list |
| PiD (`42c1a92f`) | Lists are natural-sorted at the source; the parameter tests give them in that order |
| `tools/legacy_ui_recordings.py` (`caaa2cca`) | ImageStitch's legacy Gradio fixtures are recorded from `89e2e09d` (two arguments, no "Maximum Side Length") instead of the pre-port revision |
| line endings | `modules/processing_scripts/refiner.py`, `modules/sysinfo.py`, `scripts/xyz_grid.py` restored to upstream's CRLF |

## Skipped

| Upstream commit | Why |
| --- | --- |
| `939a6473` last (`style.css`), `99550df2` newline (Gradio text areas) | Gradio styling; the files stay deleted, the SPA has its own layout |
| `e67193d9`: half the CPU cores for update checks | Remi's update check is a server job with its own worker count and progress events |
| `1f2f068d`: `sd_arch` | Remi's metadata keeps `sd version` (shared with LoRA and embeddings) |

## Validation

`PYTHON=venv/bin/python tools/check.sh --strict` on macOS (Apple Silicon):
all 59 steps passed, among them ruff, every pytest suite, the API / settings /
schema contracts, line endings, i18n, the SPA build, lint, vitest, the
end-to-end run (ImageStitch on Anima included) and the legacy Gradio recorder
(`tools/legacy_ui_env.sh` builds its interpreter in `../remi-gradio-env`).
The e2e run needs `ultralytics` in the venv (ADetailer) and Playwright's
`chromium-headless-shell`. Pyright findings on the lines this sync changed
were fixed in the follow-up commit (the remaining ones are upstream's own).

Still to check by hand on a GPU: a GGUF checkpoint, Anima RGBA, an Edit model
on a large reference image (resized to `reference_max_side`), Anima Edit with
an img2img image of another size than 1024 x 1024, and the ROCm
image on an AMD card.
