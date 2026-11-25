# Absorbing Forge Neo updates into Remi

Remi started as a downstream branch of
[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
(`neo`) and still shares its **runtime**: `backend/`, the processing
pipeline, samplers, model loading, the scripts runtime, the extensions'
processing code and the `/sdapi/v1` API. Its **user interface** no longer
shares anything: the Gradio UI (`modules/ui*.py`, `javascript/`, `html/`,
`style.css`, `script.js`, ForgeCanvas, every `ui()`) was deleted and replaced
by the declarative parameter model (`remi/ui`), the FastAPI server
(`remi/server`, `remi/jobs`) and the React SPA (`webui/`)
([architecture.md](architecture.md)).

So an upstream update is two kinds of work:

- **runtime changes** are merged as before (Git merge, conflicts resolved in
  favour of Remi's Anima hooks; see [remi-architecture-notes.md](../remi-architecture-notes.md));
- **UI changes** cannot be merged: their *intent* is translated into
  `params()`, `remi.core` forms, actions, settings widgets or SPA code, and
  the upstream file stays deleted.

History of earlier syncs:
[2026-10-02](../upstream-sync-2026-10-02.md) (selective port from reviewed
diffs), [2026-10-14](../upstream-sync-2026-10-14.md) (Git merge of `neo`
at `46af880c`, the last upstream commit absorbed before the UI rewrite) and
[2026-10-27](../upstream-sync-2026-10-27.md) (merge of `c6c2b43d`, the first
one done with this procedure),
[2026-11-18](../upstream-sync-2026-11-18.md) (merge of `89e2e09d`) and
[2026-11-22](../upstream-sync-2026-11-22.md) (merge of `5998fb39`).
Each new sync adds a `docs/upstream-sync-YYYY-MM-DD.md` from the template in
[§5](#5-sync-note-template).

## 1. Where upstream files went

| upstream file(s) | in Remi | what to do with an upstream change |
|---|---|---|
| `backend/**`, `modules_forge/packages/**` | same files (with Remi's Anima / DiT capability hooks: `dit_registry.py`, `dit_capabilities_ext.py`, `model_capabilities.py`, `dit_control_patch.py`, Z-Image/ERNIE/Fibo engine & `lumina.py`/`svdq.py` hooks, lazy `loader.py` capability hook, and MPS dequant cache in `operations_mixed_precision.py`) | merge; keep Remi's hooks and run `test_remi_mps_perf.py`, `test_remi_zimage.py`, `test_remi_ernie.py`, `test_remi_fibo.py`, `test_remi_dit_control_patch.py` |
| `modules/processing.py`, `modules/sd_models.py`, `modules/sd_samplers*.py`, `modules/sd_schedulers.py`, `modules/sd_vae*.py`, `modules/sd_unet.py`, `modules/prompt_parser.py`, `modules/images.py`, `modules/masking.py`, `modules/rng*.py`, `modules/devices.py`, `modules/upscaler*.py`, `modules/*_model.py`, `modules/textual_inversion/**`, `modules/extra_networks.py` | same files | merge; check Remi's hooks listed in remi-architecture-notes.md |
| `modules/scripts.py`, `modules/scripts_postprocessing.py`, `modules/scripts_auto_postprocessing.py` | same files: hooks unchanged, `ui()` / Gradio wiring replaced by `params()` layouts (`build_layouts`, `script_args_from_values`, `disable_script`), plus `Script.requires_capabilities` applicability metadata in `scripts_schema()` | merge hook and runner changes; translate anything about controls into the layout code |
| `modules/script_callbacks.py` | same file without the UI-building callbacks (`on_ui_tabs`, `on_ui_train_tabs`, `on_before_component`, `on_after_component`) | merge non-UI callbacks; a new UI callback has no equivalent: translate what it builds into params, a page or a widget ([extension-guide.md §14](extension-guide.md#14-porting-an-a1111--forge-extension)) |
| `modules/txt2img.py`, `modules/img2img.py`, `modules/postprocessing.py` | same files; entry points take named values (`run_txt2img(core, scripts, job)`, `run_img2img`, `run_extras_named`); the Gradio handler signatures are gone | merge changes to how `p` is built; a new argument of the old handler is a new core param (§2.3) |
| `modules/infotext_utils.py` | same file (parsing, `create_override_settings_dict`; Remi made `re_param` linear-time with the same matches); paste bindings are `infotext=` / `paste=` on inputs and `remi/core/infotext.py` | merge parsing changes, keeping the `re_param_code` lookbehind and atomic group (`tests/test_remi_image_meta.py` times a 200,000-character line); new paste fields of `ui.py` become `infotext=` keys in `remi/core/forms.py` |
| `modules/shared_state.py` | same file plus `State.live_preview_listeners`: called after each `assign_current_image`, it calls Remi's preview encoder (`remi/realtime.py`), which encodes a small preview right there or wakes its thread, so a live preview leaves as soon as it exists | merge; keep the listener list and its call in `assign_current_image` |
| `modules/errors.py` | same file; while a job runs, `remi/jobs/queue.py` (`script_errors`) wraps `errors.report(message, *, exc_info)` to turn "Error running <hook>: <script>" reports into the job's warnings | merge; keep `report`'s name and signature (or adapt `script_errors`) |
| `modules/call_queue.py`, `modules/progress.py` | `queue_lock` and the progress registry kept for `/sdapi` and `/internal/progress`; jobs run in `remi/jobs` (queue, worker) with `remi/core/job.py` (`job_state`, what `wrap_gradio_gpu_call` did) | merge lock / progress fixes into both places |
| `modules/shared_options.py`, `modules_forge/shared_options.py`, options in extensions | same files; `component` is a `remi.ui` class (`ui.Slider`, `ui.Choice`, …) | merge; convert `gr.*` components (§2.2) |
| `modules/shared_items.py`, `modules/shared.py`, `modules/options.py`, `modules/shared_cmd_options.py`, `modules/cmd_args.py` | same files (no Gradio types; `--auth`, `--worker-process`, `--worker-gpus`, `--worker-mlx`, etc. added; Gradio-only flags are no-ops) | merge; a new flag that configures Gradio needs a server equivalent or a no-op |
| `modules/api/api.py`, `modules/api/models.py` | same files, mounted on the Remi app; script args and infotext go through the `remi.ui` layouts (`init_default_script_args`, `apply_infotext`) | merge; keep the layout-based script-argument code |
| `modules/launch_utils.py`, `launch.py`, `webui.py`, `requirements*.txt`, `webui.sh`, `webui.bat`, `webui-user.sh`, `webui-user.bat`, `download_configs.py` | same files: no Gradio install, `build_webui()` builds the SPA, `webui.py` serves `remi.server`; `webui-user.sh` keeps a `COMMANDLINE_ARGS` from the environment and passes its arguments on | merge installer changes; drop Gradio pins and Gradio launch code |
| `modules/initialize.py`, `modules/initialize_util.py` | same files without Gradio setup | merge |
| `modules/extensions.py` | same file (discovery, metadata, `Extension` objects); the git work of `Extension.check_updates` / `fetch_and_reset_hard` is delegated to `remi/extensions_admin.py` (`check_repo`, `reset_repo`: no terminal prompt, timeouts, per-repo locks) | merge discovery and metadata changes; port changes to the git commands into `extensions_admin.py` |
| `modules/config_states.py` | same file; `remi/extensions_admin.py` uses it for the backups of `/api/v2/extensions/config-states` | merge |
| `modules/sysinfo.py` | same file; credentials are redacted with `modules/redact.py` (config, `COMMANDLINE_ARGS`, URLs); served by `GET /api/v2/sysinfo` | merge; keep the redaction on every new field that can hold a secret |
| `modules/styles.py` | same file (`StyleDatabase`); `/api/v2/styles` and the style editor read and write through it | merge |
| `modules/localization.py` | same file without `localization_js`; `remi/server/i18n.py` merges the catalogs for `GET /api/v2/i18n/<locale>` | merge; a change to how catalogs are found goes into `i18n.py` too |
| `modules/fifo_lock.py` | same file plus `acquire(timeout=)`, used by `remi/jobs` | merge; keep the timeout |
| `modules/util.py`, `modules/profiling.py` | same files without the helpers that served Gradio HTML / file paths (`html()`, `webpath()`) | merge; a new helper for Gradio has no use |
| other `modules/*.py` (`script_loading`, `shared_init`, `restart`, `timer`, `paths`, `paths_internal`, `hashes`, `cache`, `modelloader`, `memmon`, `safe`, `gitpython_hack`, `torch_utils`, `logging_config`, `face_restoration*`, `codeformer_model`, `gfpgan_model`, `extras`, …), `modules/uni_pc/`, `modules/Roboto-Regular.ttf` | same files, unchanged or with small hooks | merge |
| `modules_forge/*.py` (`initialization`, `main_thread`, `shared`, `config`, `uv_hook`, `cuda_malloc`, `patch_basic`, `utils`, `supported_controlnet`, `supported_preprocessor`, `forge_alter_samplers`, `diffusers_patcher`, `forge_version`) | same files; `main_thread` runs the jobs of `remi/jobs`, `patch_basic` lost its Gradio URL patch, `cuda_malloc` appends `backend:cudaMallocAsync` to whichever `PYTORCH_*_ALLOC_CONF` variable torch reads | merge; drop anything that patches Gradio |
| `modules/web/` (Source Sans fonts and CSS for Gradio) | deleted: the SPA bundles its own fonts | none |
| `modules_forge/main_entry.py` | the model-selection logic only (`checkpoint_change`, `modules_change`, `dtype_change`, loading parameters); the model bar is the SPA's `features/model-bar` with `/api/v2/models` and `remi/core/presets.py` | merge selection logic; translate model-bar UI changes into the SPA |
| `modules_forge/presets.py` | same file; applied to forms by `remi/core/presets.py` (`apply_forge_preset`, `forge_preset_patch`) | merge; a new preset value that changes a control becomes a form patch entry |
| `modules/ui.py` (txt2img / img2img / PNG Info tabs, paste fields, top-level layout) | `remi/core/forms.py` (core params, infotext keys, presets, slots), `remi/core/tabs.py` (schema, validation), `remi/core/actions.py` (size swap, Hires resolution), `webui/src/pages/generate`, `features/*` | translate (§2.3) |
| `modules/ui_toprow.py` | `webui/src/features/prompt-editor` (prompt fields, token counter, styles), `features/progress` (Generate / Interrupt / Skip, forever) | translate |
| `modules/ui_common.py` | `webui/src/features/send-to` (result actions), `POST /api/v2/outputs/save`, `features/lightbox` | translate |
| `modules/ui_components.py`, `modules/gradio_extensions.py`, `modules/ui_gradio_extensions.py`, `modules/shared_gradio_themes.py` | `remi.ui` nodes (`InputAccordion` → `Section(enable=)`, `ToolButton` → `Button(variant="tool")`, `FormRow` → `Row`, `DropdownMulti` → `MultiChoice`), the SPA design system and appearance menu | none (a new component is a new node kind or widget only if a feature needs it) |
| `modules/ui_postprocessing.py` | `remi/core/forms.py` / `remi/core/extras.py` (the `extras` form), `webui/src/pages/extras` | translate |
| `modules/ui_extra_networks*.py`, `html/extra-networks-*.html`, `javascript/extraNetworks.js` | `remi/networks.py`, `remi/network_metadata.py`, `/api/v2/networks/*`, `webui/src/features/networks`; the extension-facing `ExtraNetworksPage` data API is Gradio-free in `modules/extra_networks_pages.py` (`modules/ui_extra_networks.py` re-exports it for A1111 extensions; `remi.networks` adapts registered pages) | translate UI; merge changes to item data (`list_items`, metadata) |
| `modules/ui_extensions.py`, `javascript/extensions.js` | `remi/extensions_admin.py`, `/api/v2/extensions/*`, `webui/src/pages/extensions` | translate (git logic changes: port into `extensions_admin.py`) |
| `modules/ui_settings.py`, `javascript/settings.js`, `javascript/ui_settings_hints.js` | `remi/ui/settings.py` (settings schema), `remi/server/options.py`, `PATCH /api/v2/options`, `webui/src/pages/settings` | translate |
| `modules/ui_prompt_styles.py` | `/api/v2/styles`, the style editor dialog in `features/prompt-editor` | translate |
| `modules/ui_loadsave.py` (`ui-config.json`) | not carried over: forms start from schema defaults, UI presets and browser drafts | none |
| `modules/ui_tempdir.py` | `remi/files.py` (uploads, outputs, thumbnails by opaque id) | translate if it is about saving or serving files |
| `modules/ui_checkpoint_merger.py` | `remi/core/merger.py` (the Checkpoint Merger page, registered by `scripts/checkpoint_merger.py`); `modules/extras.run_modelmerger` is the processing code | merge processing changes in `extras.py`; new or changed controls in `CheckpointMerger.params()` (keep `tests/test_remi_merger.py`'s legacy order), new texts in `localizations/zh_CN.json` |
| `modules_forge/forge_canvas/**` | `webui/src/canvas` (Konva editor) | translate (upstream's 2026-10-02 single-click drawing is already covered) |
| `javascript/*.js` (other) | SPA features: `edit-attention.js` / `edit-order.js` / `promptField.js` / `token-counters.js` → prompt editor; `imageviewer*.js` → lightbox; `progressbar.js` → WebSocket progress; `contextMenus.js` (generate forever) → run controller; `dragdrop.js` / `imageDragAndDrop.js` → drop zones; `aspectRatioOverlay.js` / `imageMaskFix.js` → canvas; `hires_fix.js` → `hr_resolution` action; `localization.js` → i18n; `notification.js` → completion notifications; `resizeHandle.js` → resizable panes; `ui.js` → app shell | translate |
| `html/licenses.html` | `GET /api/v2/licenses` (parsed into data; kept in `html/`) | merge the license list |
| `html/footer.html`, `html/ui.webp`, other `html/*` | none: `html/licenses.html` is the only file Remi keeps in `html/`. The others were Gradio page templates and assets (`footer.html`, the `extra-networks-*.html` cards, `card-no-preview.jpg`) or a screenshot of the old Gradio UI (`ui.webp`), and the SPA has its own | none: delete them again when a merge brings them back; a new runtime data file (read by Python, like `licenses.html`) is merged |
| `scripts/*.py`, `modules/processing_scripts/*.py` | same files; `ui()` replaced by `params()` with the same arguments | merge processing; translate `ui()` changes into `params()` (§2.1) |
| `extensions-builtin/<upstream ext>/` (ControlNet, LoRA, MultiDiffusion, Spectrum, …) | same folders; `ui()` → `params()`, `on_ui_tabs` → `remi.pages`, extension JS → widgets | merge `lib_*` / processing, **except the files Remi forked ([§1.1](#11-built-ins-whose-processing-remi-forked)): keep Remi's and port upstream fixes selectively**; translate UI (§2.1); upstream `mobile` and `prompt-bracket-checker` are SPA features and stay deleted |
| `docker/` | Remi's own image recipe: `docker/Dockerfile` builds this checkout (build context: the repository root, `.dockerignore`), the SPA in a Node.js stage, the server with `requirements.txt` and the launcher's PyTorch build; one data volume (`--data-dir`), no Gradio-era `ui-config.json`. Upstream's recipe cloned Forge Neo (the Gradio UI) at build time | translate: carry over base-image, apt-package and user changes by hand; never the `git clone` of upstream or the Gradio-era config files. `tests/test_remi_docker.py` keeps the torch pins and pnpm in step with the launcher |
| `README.md`, `pyproject.toml`, `.gitignore`, `.github/`, `LICENSE`, `models/**/Put … here.txt`, `extensions/`, `localizations/` placeholders | Remi's own README (en, zh-CN), lint config and CI; placeholders unchanged | carry over what applies (README model-family list; launch options, Docker and other user-facing details go into [usage.md](usage.md) and [usage.zh-CN.md](usage.zh-CN.md); new placeholder folders); keep Remi's versions otherwise |
| **any other file** (including files upstream adds) | — | **merge** when it is runtime code or data; **translate** when it builds UI (imports `gradio`, defines `ui()` / `on_ui_tabs`, or is JavaScript / CSS / HTML for the page) and delete it again |

### 1.1 Built-ins whose processing Remi forked

Most upstream built-ins only lost their UI, and their processing merges as
the table says. A few had their processing rewritten or extended by Remi;
a merge must not let upstream's version replace Remi's. For these files the
rule is **keep Remi's version; read upstream's diff and port its fixes
selectively** (the ones Remi's version does not already cover, with a test);
then name each upstream commit and what was taken or left in the sync note
(§5, "Adapted for Remi"). Conflicts are not the only risk: Git merges a
forked file **without a conflict** whenever Remi left it as it was at the
merge base, and upstream's version then replaces Remi's silently (Remi's
`lib_spectrum/__init__.py` holds only a logger; upstream's 2026-10 Spectrum
rewrite fills it with code that imports a `Spectrum` class Remi's
`forecaster.py` does not have, so the extension stops importing). So right
after `git merge`, `python tools/upstream_forks.py restore` puts **every**
file under these paths that the merge touched (merged cleanly, conflicted,
added or deleted by upstream) back to Remi's version, and prints the
`git diff` of each to read for the selective port (§3). Never take
`--theirs` for them. `tools/upstream_forks.py` holds the same list of paths
as the first column below (`tests/test_remi_upstream_forks.py` keeps them
equal).

| upstream file(s) | Remi's version | seams it relies on ([remi-architecture-notes.md](../remi-architecture-notes.md#seams)) |
|---|---|---|
| `extensions-builtin/sd_forge_spectrum/lib_spectrum/` (`forecaster.py`, `presets.py`, `__init__.py`, Remi's `pass_context.py`, `sea.py` and `modes.py`) and `extensions-builtin/sd_forge_spectrum/scripts/spectrum.py` (its hooks: `process_before_every_sampling`, …) | a fork: per-branch forecasters that rebuild on shape changes and handle cond/uncond split calls, Remi-only `pass_context.py` (the steps and parameters of the pass being sampled: hires, img2img), `sea.py` (the SEA cache-decision metric with per-configuration calibration) and `modes.py` (opt-in TaylorSeer and MagCache forecasting modes), presets and the controls upstream lacks. Upstream's `34022261` "rewrite Spectrum" (after `46af880c`) conflicts in `forecaster.py` and `spectrum.py` and moves `SpectrumNode` into `lib_spectrum/__init__.py`, which merges *cleanly* (Remi's holds only the logger; Remi's `SpectrumNode` stays in `forecaster.py`), so only `upstream_forks.py restore` keeps it out; the 2026-10-14 sync already kept Remi's `forecaster.py` over upstream's shape guard | "Spectrum \"run a real forward\" requests" (`modules/spectrum_force.py`), "`model_function_wrapper` chain", "Script-argument check" (defaults for the controls upstream lacks), "Replaced blocks in a branch's signature" (`_replaced_blocks`: since the 2026-10-27 sync, the block keys of `transformer_options["patches_replace"]` are part of a branch's signature, for Skip Layer Guidance; check Spectrum when upstream changes `modules/processing_scripts/slg.py` or how `patches_replace` is built) |
| `extensions-builtin/sd_forge_compile/scripts/compile.py` (`process_batch` and the compile presets) | upstream's presets plus Remi's "Anima per-block" preset (Remi-only `anima_block_compile.py`), the Windows compiler probe and the eager fallback while a forward override is active | "Temporary forward overrides", "Per-call block forward pre-hooks" (Modulation keeps the compiled blocks) |
| `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/` (deleted in Remi, stays deleted), `extensions-builtin/sd_forge_controlnet/lib_controlnet/infotext.py`, `extensions-builtin/sd_forge_controlnet/lib_controlnet/external_code.py` (infotext and API unit parsing) | the Gradio unit UI is replaced by Remi-only `unit_param.py` / `unit_actions.py` / `openpose.py` / `region_canvas.py`; `infotext.py` parses an entry into unit fields (`parse_unit_fields`) instead of wiring Gradio components; `external_code.py` also reads the A1111 API's `{"image", "mask"}` dict | none (UI port); processing in `lib_controlnet` other than these files merges normally |

Small Remi hooks inside otherwise upstream files (for example the Anima
key formats and the failed-name list of `sd_forge_lora/networks.py`, or
`backend/nn/anima.py`) are not forks: merge, keeping the hooks.

To see what a sync must look at here, list the upstream commits and the
diffstat on these paths before merging (`git diff "$BASE" upstream/neo --
<path>` for each file then shows the change to port):

```bash
python tools/upstream_forks.py review    # commits after the merge base and diffstat on the forked paths
python tools/upstream_forks.py budget    # enforce FORKED <= 6, verify §1.2 pinning tests, classify DIVERGED
```

When Remi forks another upstream file, add its row here and its path to
`FORKED` in `tools/upstream_forks.py` in the same commit (guardrail: `FORKED <= 6`,
enforced by `tools/upstream_forks.py budget`; see
[upstream-pr-candidates.md](upstream-pr-candidates.md) for the `DIVERGED` 32 → ≤ 20
upstream PR roadmap).

### 1.2 Upstream bugs Remi fixes (divergences)

**Policy: deterministic upstream bugs are fixed in Remi and recorded as
divergences; report them upstream when practical.** Remi does not inherit a
bug it can reproduce (a fixed input that fails or gives a wrong result every
time) just to stay close to upstream. Each fix keeps upstream's code around it
and is small enough to merge through; it gets a regression test and a row
here.

These files are not forks: a merge takes upstream's changes to them as usual,
and `tools/upstream_forks.py restore` never touches them (it lists §1.1 only).
`tools/upstream_forks.py review` lists the upstream commits on them
(`DIVERGED`, kept equal to the first column below by
`tests/test_remi_upstream_forks.py`). On a conflict there, keep Remi's fix and
take the rest of upstream's change; if upstream fixed the same bug, take
upstream's fix, keep the test, and drop the row once nothing of Remi's is left.
A clean merge can still undo a fix (upstream rewrites the function around it):
the test the row names fails then, so run it after every sync.

| upstream file | Remi's fix | pinning test |
|---|---|---|
| `backend/attention.py` | `attention_basic` supports 2D `(Q, K)` and 3D/4D boolean attention masks (such as Qwen-Image-2.1's block-causal prefix mask) in addition to `(B, K)` padding masks | `tests/test_remi_qwen21.py` |
| `backend/memory_management.py` | `discard_cuda_async_error` only on CUDA: on MPS its `torch.cuda.synchronize()` raised inside `is_oom`, so an out-of-memory error skipped the tiled-VAE fallback and left the job running; `mac_version()` is `@functools.lru_cache`d; canonicalizes `mps` vs `mps:0` device comparisons (`_same_device`), exposes `mps_memory_info()`, and supports cache-clear hooks (`tests/test_remi_mps_perf.py`) | `tests/test_memory_management_oom.py` |
| `backend/nn/krea.py` | `TextFusionBlock` avoids in-place mutation (`x += ...`) of cached Qwen3-VL conditioning tensors across steps; Edit mode broadcasts single-batch reference latents for arbitrary batch sizes and unpads odd latent dimensions cleanly; `single_block` SLG replace hook and `last_input` / `compute_last` Spectrum pre-head seam | `tests/test_remi_krea2.py` |
| `backend/nn/qwen21.py` | lossless cross-step block-causal prefix KV cache (with fresh-tensor `torch.equal` lookup, target odd/even parity key, VRAM budget guard, eager prefill for per-block `torch.compile`) + target RoPE cache, 4D `(1, 1, Q, K)` causal mask, safe reference batch broadcast, and `final_input` / `compute_final` Spectrum pre-head seam | `tests/test_remi_qwen21.py` |
| `backend/sampling/sampling_function.py` | `transformer_options["batch_indices"]`: the image rows of a call, for per-image state in model wrappers; on MPS and CPU the cond / uncond chunks are never merged by free memory (`merges_by_free_memory`): "free" there is the system's available RAM, so the same request ran as one batch-2 or two batch-1 calls from run to run and gave a different image (batch-2 kernels round differently in bf16, CFG amplifies it) | `tests/test_sampling_batch_indices.py` |
| `backend/text_processing/anima_engine.py` | `BREAK` (the parser's chunk marker) is a space for Anima's single chunk, not the text "BREAK" at weight -1 | `tests/test_anima_text_engine_compat.py` |
| `extensions-builtin/forge_preprocessor_inpaint/scripts/preprocessor_inpaint.py` | inpaint_only with Hires fix: a unit that sat out the hires pass blends with the hires original instead of scaling the image back down; two inpaint_only units each blend with their own image and mask (the state is per unit, not on the one shared preprocessor) | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` |
| `extensions-builtin/forge_preprocessor_reference/scripts/forge_reference.py` | an error while recording no longer leaves it recording for every later run; the recorded tensors are released after sampling, and by ControlNet's `cleanup` after an error (`release()`) | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` |
| `extensions-builtin/forge_preprocessor_tile/scripts/preprocessor_tile.py` | tile_colorfix scales each row by its own image's sigma (a batch of 2+ failed) | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` |
| `extensions-builtin/sd_forge_controlllite/scripts/forge_controllllite.py` | Anima LLLite follows a Hires checkpoint's DiT, skips one it does not fit, restores after an error, is placed on the original blocks of a 2.9B / 3.8B (`backend/nn/anima_layout.py`) and refuses a depth or width it does not fit by name; the step window counts sampler steps (`sampler_progress`), not forward calls; the mask multiplies a copy of the unit's control image, not the cached one (a soft mask compounded over batches and the hires pass) | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` |
| `extensions-builtin/sd_forge_controlnet/lib_controlnet/global_state.py` | the model list holds only the control models the loaded architecture runs, read from the tensor names (`control_model_family` / `detect_dit_control_family` for `anima` / `qwen21` / `zimage` vs `unet`): upstream filtered by file name, so an SD ControlNet named "depth" was offered for DiT models that ignore it (`tests/test_remi_dit_control_patch.py`) | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` |
| `extensions-builtin/sd_forge_controlnet/scripts/controlnet.py` | `cleanup` runs each model's after-sampling hook (a generation that raised no longer keeps LLLite's patch in every later one) and releases a reference preprocessor's recordings; a unit whose control model fails to load is removed from the infotext (it never ran) | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` |
| `extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite.py` | LLLite runs in its window of sampler steps (`step_active`; img2img, hires and second-order samplers counted forward calls instead) | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` |
| `extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite_anima.py` | two units on the same Linear restore in any order (the second restore put the first's wrapper back, and the next apply recursed); a control it cannot use on a pass (token count, batch) is reported once per sampling instead of silently doing nothing; sampler-step window as above | `extensions-builtin/sd_forge_controlnet/tests/test_controlnet_params.py` |
| `extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py` | the encoded references are cached per engine (re-encoded after unload/reload); Qwen-Image-2.1 preserves 4-channel RGBA references and includes alpha in `hash_image` (`tests/test_remi_qwen21.py`) | `extensions-builtin/sd_forge_image_stitch/tests/test_image_stitch_params.py` |
| `extensions-builtin/sd_forge_lora/networks.py` | a failed LoRA apply no longer leaves the previous set active; the failure is reported again for the same set without a reload, named by the alias this prompt uses; an Anima LoRA whose depth no block layout maps is reported as failed instead of only logged, and the LoRA cache key includes the inserted-blocks policy (a changed policy reused the old patches) | `extensions-builtin/sd_forge_lora/tests/test_networks_apply_failure.py` |
| `extensions-builtin/sd_forge_lora_schedule/scripts/control.py` | a malformed schedule is rejected whole; before its first point a LoRA takes that point's weight; the step is found with a batch of timesteps | `extensions-builtin/sd_forge_lora_schedule/tests/test_lora_schedule_params.py` |
| `extensions-builtin/sd_forge_radial/scripts/forge_radial.py` | Wan's global attention is put back in `cleanup` (no dense last step, or an error, left radial attention for the next Wan run) | `extensions-builtin/sd_forge_radial/tests/test_radial_params.py` |
| `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py` | `apply_masks` (samples a script already decoded) appends each image's mask; it indexed an empty list (IndexError). No built-in path reaches it today | `extensions-builtin/soft-inpainting/tests/test_soft_inpainting_params.py` |
| `modules/api/api.py` | a pasted value that overflows (`Steps: 1e999`) is ignored instead of failing the request; `POST /sdapi/v1/options` holds a model pick while a generation runs (`tests/test_remi_model_selection.py`) and checks a changed value as the settings page does (`remi.ui.settings.check_api_values`; `tests/test_remi_server_runtime.py`); setting values nested deeper than 64 levels are refused there and in a generation's `override_settings` (`refuse_deep_settings`); an infotext choice named by label fills its request field with the index the processing takes | `tests/test_api_pasted_value.py` |
| `modules/extras.py` | the Merger refuses an output name that is one of the models it reads | `tests/test_remi_merger.py` |
| `modules/images.py` | PNG Info never lets piexif open an EXIF chunk as a file name on the server | `tests/test_remi_security.py` |
| `modules/img2img.py` | a batch image whose infotext fails to parse no longer leaves `infotext_styles` at "Ignore" | `tests/test_img2img_batch_png_info.py` |
| `modules/infotext_utils.py` | pasting "Hires Module 1" ("Built-in", "Use same choices", modules not installed here) | `tests/test_infotext_paste_roundtrip.py` |
| `modules/ngrok.py` | a failed `--ngrok` connection names the token by its last 4 characters; the authtoken (also `NGROK_AUTHTOKEN`), the basic-auth credentials and any `--ngrok-options` value under a secret-like name (`auth`, `secret`, `token`, `password`, `key`) are kept out of the printed reason | `tests/test_remi_ngrok_token.py` |
| `modules/options.py` | `Options.save` encodes the settings first, checks they read back, and swaps the file in whole (`write_whole`: a temporary file next to the link's target, its mode kept; written in place where the swap is refused or the file is hard-linked; a read-only file still refuses the save): dumped straight into the open file, a value `json` cannot write (an unpaired surrogate, nesting past the recursion limit) left `config.json` cut short | `tests/test_remi_options_save.py` |
| `modules/processing.py` | the prompt cache (the hires conds too) is cleared after a model reload; edit models: an `ini_latent` an earlier run left is cleared before `init()`, and the input-image reference is restored before each batch's `setup_conds()` (these two need an edit model and have no unit test: check them by reading after a merge); Qwen-Image-2.1 preserves 4-channel RGBA inputs and alpha channels across resize, `masking.fill`, `uncrop`, `apply_overlay` (both standard and precise mask), and color correction (`tests/test_remi_qwen21.py`); the CFG++ samplers keep the negative prompt at CFG 1 (`skips_negative_conds`): they read the unconditional prediction at every scale and got zeros (pinned by `tests/test_processing_negative_conds.py`) | `tests/test_prompt_cache_clear.py` |
| `modules/processing_scripts/refiner.py` | `cleanup` puts the base weights back after a generation that raised after the fast swap; `setup` clears `p.refiner_cfg = None` when disabled | `tests/test_refiner_switch.py` |
| `modules/sd_samplers.py` | `create_sampler` refuses DDIM, PLMS and Kohaku LoNyu Yog on flow-matching models (`prediction_type == "const"`), as `no_sdxl` samplers are refused on SDXL: they assume a VP / VE schedule and garbled Anima's images | `tests/test_flow_sampler_support.py` |
| `modules/sd_samplers_common.py` | `apply_refiner` clears the prompt cache before `setup_conds()`: the cache key holds the base model's loading parameters again by then, so the refiner reused the base's conds (Anima-3.8B conds failed on a 2B refiner); records `"Refiner CFG scale"` in infotext when `p.refiner_cfg` is set | `tests/test_refiner_switch.py` |
| `modules/sd_schedulers.py` | `turbo_scheduler` takes the schedule's sigmas at its indices instead of `predictor.sigma(index)`: a flow predictor's timesteps are not the indices, and Anima's images came out as noise | `tests/test_flow_schedules.py` |
| `modules_forge/packages/k_diffusion/external.py` | `ForgeScheduleLinker.get_sigmas` (the "Uniform" scheduler) spaces the predictor's own timesteps between sigma_max and sigma_min instead of the indices 999..0: a flow predictor's timesteps are not the indices (Anima's run to 1), and its images came out black | `tests/test_flow_schedules.py` |
| `modules_forge/packages/comfy/lora.py` | `model_lora_keys_unet` / `model_lora_keys_clip` take a fresh key map per call: the shared `key_map={}` default kept every earlier model's keys (a 3.8B's blocks 28-51 matched on a 2B); dispatches Anima / MOSS / Krea 2 / FLUX.3 / Qwen-Image-2.1 multi-trainer LoRA key maps | `tests/test_anima_lora_formats.py` |
| `modules_forge/supported_controlnet.py` | the built ControlNet cache is keyed on the file's contents (real path, mtime, size) and the dtypes, not the path (a replaced file kept its old weights until a restart); the setting's 0 caches nothing | `tests/test_controlnet_model_cache.py` |

When Remi fixes another upstream bug, add its row here and its path to
`DIVERGED` in the same commit.

## 2. Translating each kind of change

### 2.1 A script or extension changes its `ui()`

Upstream adds, removes or reorders controls in a `ui()` (a built-in script,
a processing script, a bundled extension).

1. Read the upstream diff of `ui()` **and** of the processing hook signature
   (`process`, `process_before_every_sampling`, …). The hook signature is the
   contract: `params()` must declare the inputs in exactly that positional
   order (use `param_order` when the visual order differs).
2. Edit `params()`: the component mapping is in
   [extension-guide.md §14](extension-guide.md#14-porting-an-a1111--forge-extension)
   (`gr.Slider` → `ui.Slider`, `gr.Radio` → `ui.Choice(style="radio")`,
   `.change` visibility → `visible_if`, buttons → actions, …). Carry the
   infotext key (`infotext=`), paste logic (`paste=`) and Anima preset field.
3. Update the extension's tests: the argument-order assertion
   (`layout.names` / `values_to_args({})`) and, for ports with a legacy
   snapshot, list the change in `assert_matches_snapshot(..., changes=...)`
   instead of re-recording the snapshot.
4. Add the new strings to `localizations/zh_CN.json` (the localization test
   fails otherwise).
5. If a control moved between scripts or a positional index changed,
   imported Agent Scheduler tasks and API clients are affected: keep the
   order compatible with upstream's `ui()` so `/sdapi` `alwayson_scripts`
   payloads written for Forge Neo still work.

### 2.2 A new or changed setting

Upstream `shared_options.py` uses Gradio classes. Convert while merging:

| upstream | Remi |
|---|---|
| `gr.Checkbox` / no component with a `bool` default | `ui.Toggle` or no component |
| `gr.Slider, {"minimum", "maximum", "step"}` | `ui.Slider, {same}` |
| `gr.Number` | `ui.Number` |
| `gr.Textbox` | `ui.Text` (`{"lines": n}`) |
| `gr.Dropdown, {"choices"}` | `ui.Choice, {"choices"}` (`"allow_custom": True` for an editable dropdown) |
| `gr.Radio, {"choices"}` | `ui.Choice, {"choices", "style": "radio"}` |
| `ui_components.DropdownMulti`, `gr.Dropdown(multiselect=True)` | `ui.MultiChoice` |
| `gr.CheckboxGroup` | `ui.MultiChoice, {"style": "checkboxes"}` |
| `gr.ColorPicker` | `ui.Color` |
| `gr.HTML` | `ui.Markdown` (a help block: write it as `OptionMarkdown`) |
| `OptionHTML("…<b>…</b><br><a href=…>…</a>")` | `OptionMarkdown("…**…**\n\n[…](…)")`: the same text in Markdown, with its `localizations/zh_CN.json` entry re-keyed (`tests/test_remi_ui_schema.py` fails on an `html` node or `OptionHTML` in the core settings) |

Keep `.info()`, `.link()`, `.needs_restart()`, `.needs_reload_ui()`,
`refresh=` and `onchange=`. Then check the setting in the SPA (Settings page
search), add its label and help to `localizations/zh_CN.json`, and, if the
SPA itself must react to it (a client-side behaviour such as a lightbox
option), add the key to the bootstrap `options` subset in
`remi/server/routes/core.py` and implement it in `webui/src`. Settings that
only configured Gradio (themes, queue, `gradio_*`) have no effect in Remi:
leave them out or keep them hidden.

### 2.3 A change to the generation forms (`modules/ui.py`)

A new core control (a new txt2img/img2img parameter) touches:

1. `modules/processing.py` / `txt2img.py` / `img2img.py`: merged as usual.
2. `remi/core/forms.py`: the param (name = the processing attribute), its
   infotext key, preset field, slot/category, UI-preset patch if any.
3. `remi/core/tabs.py` / `forms.check_core`: cross-param rules, if any.
4. `modules/api/api.py` mapping (`INFOTEXT_API_FIELDS`) when the infotext
   key differs from the API field.
5. The SPA only when the control needs more than the schema renderer (the
   curated generate view frames blocks by content: `pages/generate/curate.ts`).
6. Refresh what the frontend tests and the typed client read, from the
   repository root: `python webui/tools/dump_schemas.py` (the tab schemas),
   `python tools/dump_settings.py` for settings (`tools/check.sh` runs both
   with `--check`), and `python tools/export_openapi.py` followed by
   `pnpm gen:api` in `webui/` when a route or model changed
   ([frontend.md "Scripts"](frontend.md#scripts) lists the command,
   ["API layer"](frontend.md#api-layer-srcapi) what it generates).

### 2.4 A JavaScript or styling change

Find the SPA feature that replaced the file (§1 table, or
[inventory/core-frontend-features.md](inventory/core-frontend-features.md),
which maps every upstream JS function to its feature) and implement the
behaviour there with a test. Cosmetic upstream CSS changes are ignored: the
SPA has its own design system.

### 2.5 A new built-in extension upstream

Port it like Remi's own ports ([extension-guide.md §14](extension-guide.md#14-porting-an-a1111--forge-extension),
architecture.md "Built-in extension porting rule"): snapshot its `ui()` with
`tools/record_legacy_ui.py` (it runs the `ui()` in an export of the last
Gradio-era revision, with an interpreter that has that era's Gradio, set up
once outside the checkout from the requirements `record_legacy_ui.py
--requirements` derives from the git history, as
[extension-guide.md §13](extension-guide.md#13-testing) shows; the current
`requirements.txt` cannot be installed next to that Gradio):

```bash
tools/legacy_ui_env.sh        # once (network): builds ../remi-gradio-env, then runs --self-test
../remi-gradio-env/bin/python tools/record_legacy_ui.py --self-test   # the setup still records a known
                              # script (Spectrum before its port) exactly as its committed fixtures
../remi-gradio-env/bin/python tools/record_legacy_ui.py \
    extensions-builtin/<ext>/scripts/<script>.py --script-rev upstream/neo
# -> extensions-builtin/<ext>/tests/fixtures/legacy_ui_txt2img.json, legacy_ui_img2img.json
```

`REMI_LEGACY_PYTHON=../remi-gradio-env/bin/python tools/check.sh --only legacy-ui`
runs the self-test as a gate, and `legacy-ui:fixtures` records every committed
fixture again (`record_legacy_ui.py --fixtures --check`); add the new
fixture to `RECORDINGS` in `tools/legacy_ui_recordings.py` (its script,
the revision, its format and context), which
`tests/test_remi_legacy_ui_fixtures.py` requires. A script already ported to `params()` at the
given revision is refused (its `ui()` would record nothing): record the
revision before its port.

then write `params()`, move tabs to pages and JavaScript to widgets or core
features, add `zh_CN` strings and the parity (`assert_matches_snapshot`) /
localization / standalone-import tests, and list it in
[inventory/extensions.md](inventory/extensions.md). Record before merging
(`--script-rev upstream/neo` reads the upstream file, whatever the working
tree holds).

### 2.6 A new upstream model family, sampler or scheduler

Merge. The SPA picks up new samplers, schedulers, checkpoints, modules and UI
presets from `/api/v2/models` and the forms' callable choices; a new UI
preset (`forge_preset`) needs its defaults in `modules_forge/presets.py` and
appears in the model bar automatically.

## 3. Procedure

Once per clone, add Forge Neo as the `upstream` remote:

```bash
git remote add upstream https://github.com/Haoming02/sd-webui-forge-classic.git
```

Then review what is new (bash or zsh, from the repository root):

```bash
git fetch upstream
BASE=$(git merge-base HEAD upstream/neo)             # last absorbed upstream commit (c6c2b43d after the 2026-10-27 sync)
git log --oneline --no-merges "$BASE"..upstream/neo  # what is new

# Review by path group (the §1 rows); "other" is everything the groups leave out
review() { echo "== $1"; shift; git diff --stat=200 "$BASE" upstream/neo -- "$@"; }
RUNTIME=(backend modules_forge/packages modules/processing.py 'modules/sd_*.py' modules/prompt_parser.py modules/images.py
         modules/masking.py modules/textual_inversion modules/extra_networks.py modules/devices.py 'modules/rng*.py'
         'modules/upscaler*.py' 'modules/*_model.py' modules/uni_pc)
RUNNER=(modules/scripts.py modules/scripts_postprocessing.py modules/scripts_auto_postprocessing.py modules/script_callbacks.py
        modules/txt2img.py modules/img2img.py modules/postprocessing.py modules/infotext_utils.py modules/call_queue.py
        modules/progress.py modules/shared_state.py modules/errors.py modules/fifo_lock.py modules/api)
SETTINGS=(modules/shared_options.py modules_forge/shared_options.py modules/shared_items.py modules/shared.py modules/options.py
          modules/shared_cmd_options.py modules/cmd_args.py modules_forge/presets.py modules_forge/main_entry.py)
DELEGATED=(modules/extensions.py modules/config_states.py modules/sysinfo.py modules/styles.py modules/localization.py)
UI=(modules/ui.py 'modules/ui_*.py' modules/gradio_extensions.py modules/shared_gradio_themes.py javascript html style.css
    script.js modules_forge/forge_canvas modules/web)
SCRIPTS=(scripts modules/processing_scripts extensions-builtin)
INSTALL=(launch.py webui.py webui.sh webui.bat webui-user.sh webui-user.bat download_configs.py modules/launch_utils.py
         modules/initialize.py modules/initialize_util.py 'requirements*.txt')
review runtime   "${RUNTIME[@]}"
review runner    "${RUNNER[@]}"
review settings  "${SETTINGS[@]}"
review delegated "${DELEGATED[@]}"   # Remi moved part of these into remi/ (§1)
review ui        "${UI[@]}"
review scripts   "${SCRIPTS[@]}"
review install   "${INSTALL[@]}"
GROUPED=("${RUNTIME[@]}" "${RUNNER[@]}" "${SETTINGS[@]}" "${DELEGATED[@]}" "${UI[@]}" "${SCRIPTS[@]}" "${INSTALL[@]}")
review other . "${GROUPED[@]/#/:(exclude)}"  # modules_forge/*.py, other modules/*.py, docker/, README, …: §1 rows or the default row

git diff "$BASE" upstream/neo -- "${UI[@]}"                                                          # translated, not merged
# ui() changes: the whole diff of every script file, because a ui() also shows module-level texts and
# constants (image_stitch.py's t2i_info / i2i_info HTML, choice lists) that a pattern on ui() lines misses
git diff "$BASE" upstream/neo -- scripts modules/processing_scripts 'extensions-builtin/*/scripts/*' \
    'extensions-builtin/*/javascript/*' 'extensions-builtin/*.css'
git diff "$BASE" upstream/neo -G'gradio|gr\.|def ui\(|on_ui_tabs|ui_components' -- extensions-builtin \
    ':(exclude)extensions-builtin/*/scripts/*'                                                       # UI built in lib_* code
git diff --name-status --diff-filter=A "$BASE" upstream/neo                                          # new files
python tools/upstream_forks.py review                                # files Remi forked (§1.1): port, never merge;
                                                                     # files with Remi's upstream-bug fixes (§1.2): merge, keep each fix
```

Record the legacy snapshot of every new upstream `Script` class that has a
`ui()` now, before merging (§2.5). Two kinds of new items have no snapshot:

- **Tab-only or JavaScript-only extensions** (`on_ui_tabs`, `javascript/`,
  no `Script.ui()`): port the tab to a `remi.pages` page (or an SPA feature
  for pure JavaScript behaviour), list it in
  [inventory/extensions.md](inventory/extensions.md), and delete the Gradio and
  JavaScript files after the port.
- **Scripts without UI** (an `AlwaysVisible` script whose controls are
  settings, or with no controls): register its options through §2.2
  (`OptionInfo` with `ui.*` widgets); nothing to snapshot.

`record_legacy_ui.py` refuses both kinds: a file with no `Script` class
reports `found [] (use --class)`, and a script with no Gradio `ui()` at the
upstream revision reports `has no Gradio ui() at upstream/neo`. For a new
upstream item either message is the signal to take one of these two routes
(the `--script-rev` hint in the second one applies only to a script Remi
already ported).

Then merge on a branch. **Step A — merge and list the work:**

```bash
git switch -c sync/neo-$(date +%Y-%m-%d)
git merge --no-ff --no-commit upstream/neo
python tools/upstream_forks.py restore   # first, before anything else: every file Remi forked (§1.1)
                                   # that the merge touched, clean merge or conflict, back to Remi's
                                   # version; then port the printed `git diff`s selectively, with tests

git status --short | grep '^DU'     # upstream edited a file Remi deleted (modules/ui*.py, javascript/, …):
                                   # translate the change (§2), then: git rm <file>
git diff --name-only --diff-filter=A --cached | grep -E '(^|/)(javascript/|style\.css$)|^(html/|modules/ui|script\.js$)'
                                   # new upstream UI files (core or extension): translate, then git rm them
git diff --name-only --diff-filter=A --cached -- '*.py' | xargs -r git grep --cached -lE 'import gradio|from gradio|on_ui_tabs' --
                                   # new upstream Python that builds Gradio UI: translate (§2, §2.5), then
                                   # git rm it or remove its Gradio code
git diff --name-only --diff-filter=U                       # content conflicts: resolve keeping Remi's hooks
```

Now work through those lists: resolve every content conflict (keeping
Remi's hooks), translate each `DU` file and new upstream UI file (§2) and
`git rm` it, and `git add` the results. The checks below read and import the
merged tree, so they only mean something once nothing is left:
`git diff --name-only --diff-filter=U` prints nothing and no file holds a
`<<<<<<<` marker. Before that, ruff reports the conflict markers as syntax
errors and `tools/sdapi_contract.py` stops with the list of unresolved files.

**Step B — after every conflict is resolved and the upstream UI files are
translated and removed:**

```bash
git diff --name-only --diff-filter=U  # must print nothing
git grep -nE '^(<<<<<<<|>>>>>>>) '   # leftover conflict markers, also in files already added: must print nothing
ruff check . --select TID251 --output-format concise  # Gradio imports (banned in pyproject.toml): must report none;
                                     # a hit is upstream UI still to translate (§2) and then git rm
python tools/check_line_endings.py   # upstream files keep upstream's line endings (below); --fix repairs
python tools/sdapi_contract.py --rev upstream/neo   # /sdapi/v1 routes and models still match upstream's
```

`tools/sdapi_contract.py` compares the merged `modules/api/api.py` and
`modules/api/models.py` with upstream's: every upstream route (method, path,
response model, endpoint parameters) and every model field (name, default,
required, a type that accepts upstream's) must be in Remi's; differences Remi
keeps on purpose are listed with their reason in its `ALLOWED`. Without
`--rev` it compares with the last absorbed commit (the merge base with
`upstream/neo`, or `LAST_ABSORBED` in a clone without that remote), as
`tests/test_remi_sdapi_contract.py` does in `tools/check.sh`; set
`LAST_ABSORBED` to the new upstream commit when committing the merge (the
test checks it against the merge base).

**Line endings.** Forge Neo commits about a hundred files with CRLF line
endings (most of `modules/`, `launch.py`, …) and the rest with LF. Remi
keeps every file it shares with upstream byte for byte in upstream's line
endings, and its own files in LF. There is no `.gitattributes` and no
renormalization: the repository stores what upstream stores, so `git merge`
compares real lines and a conflict is a real conflict. An editor or tool
that saves a CRLF file as LF turns every line into a change and makes the
next merge conflict on the whole file; `tools/check_line_endings.py` (run by
`tools/check.sh` as `contract:line-endings`) fails on that, and `--fix`
converts the listed files back. The list, `tools/upstream_crlf.txt`, holds
the files that are CRLF at the merge base; after committing the merge,
regenerate it with `python tools/check_line_endings.py --update` (it reads
`git merge-base HEAD upstream/neo`), so new upstream CRLF files are guarded
and deleted ones dropped. The UI rewrite had saved 19 of these files as LF
(`modules/scripts.py`, `scripts_postprocessing.py`,
`scripts_auto_postprocessing.py`, `script_callbacks.py`, `launch_utils.py`,
`shared_options.py`, `shared_items.py`, `shared.py`, `options.py`,
`cmd_args.py`, `extensions.py`, `sysinfo.py`, `localization.py`,
`postprocessing.py`, `txt2img.py`, `img2img.py`,
`processing_scripts/comments.py`, `html/licenses.html` and
`extensions-builtin/extra-options-section/scripts/extra_options_section.py`);
they were converted back to CRLF on 2026-10-19, so `git diff 46af880c --
modules/extensions.py` shows Remi's 37 changed lines instead of all 603.

With step B passing, test (`tools/check.sh` runs every Python suite, the
OpenAPI check, i18n coverage and the frontend build, lint and tests; then the
real-model smoke test `tools/smoke_real.py` on a GPU or Apple Silicon
machine), write the sync note (§5), and commit the merge. Then refresh the
line-ending list and commit it with the sync:

```bash
python tools/check_line_endings.py --update   # tools/upstream_crlf.txt from the new merge base
python tools/check_line_endings.py            # must pass
```

Because it is a real merge, `git log HEAD..upstream/neo` is empty afterwards
and the next sync starts from the new merge base.

A selective port (cherry-picking reviewed commits, as on 2026-10-02) is
possible when only a few fixes are wanted; record the commits in the sync
note, because the merge base does not move.

## 4. Checklist

- [ ] `git log $BASE..upstream/neo` reviewed; every commit classified (merge /
      translate / skip) in the sync note.
- [ ] Runtime conflicts resolved without dropping Remi's Anima hooks
      ([remi-architecture-notes.md](../remi-architecture-notes.md) "Seams").
- [ ] Files Remi forked (§1.1) keep Remi's version: `python
      tools/upstream_forks.py restore` ran right after `git merge` (clean
      merges included, not only conflicts); the upstream fixes they lacked
      are ported with tests, and each upstream commit that
      `tools/upstream_forks.py review` lists is in the sync note's "Adapted
      for Remi" table. A new fork is in §1.1 and in `FORKED`.
- [ ] Remi's fixes of upstream bugs (§1.2) survived the merge: the test
      each row names passes; a row whose bug upstream fixed itself is
      dropped (the test stays). A new fix is in §1.2 and in `DIVERGED`, and
      `python tools/upstream_forks.py budget` passes (`FORKED <= 6`, every
      pinning test verified on disk, every `DIVERGED` entry classified).
- [ ] No Gradio: the `ruff --select TID251` check above reports nothing;
      `requirements.txt` and `modules/launch_utils.py` install no Gradio.
- [ ] Line endings: `tools/check_line_endings.py` passes, and
      `tools/upstream_crlf.txt` was regenerated with `--update` after the
      merge commit.
- [ ] Upstream UI files are deleted again (`DU` conflicts and new UI files
      resolved with `git rm` after translating).
- [ ] Each `ui()` change is in `params()` with the hook's argument order;
      parity tests and `zh_CN` catalogs updated; new upstream extensions or
      scripts have a legacy snapshot (`tools/record_legacy_ui.py`, whose
      `--self-test` and `--fixtures --check` pass in the recorder's
      environment), listed in `tools/legacy_ui_recordings.py`.
- [ ] New settings use `ui.*` widgets, have `zh_CN` labels, and are in the
      bootstrap `options` subset when the SPA reads them.
- [ ] New core params are in `remi/core/forms.py` with infotext keys;
      schema fixtures (`webui/tools/dump_schemas.py`) and the OpenAPI
      document (`tools/export_openapi.py`, then `pnpm gen:api`) regenerated.
- [ ] `/sdapi/v1` request and response models still match upstream's:
      `python tools/sdapi_contract.py --rev upstream/neo` passes (a
      difference kept on purpose is in its `ALLOWED` with the reason), and
      `LAST_ABSORBED` in that file is the new upstream commit.
- [ ] `tools/check.sh` passes; the real-model smoke test passes (txt2img
      pixel parity with the previous baseline where the change should not
      alter images).
- [ ] `docs/upstream-sync-YYYY-MM-DD.md` written; README model-family list and
      [remi-architecture-notes.md](../remi-architecture-notes.md) updated when
      the change adds a family or a seam.

## 5. Sync note template

Copy into `docs/upstream-sync-YYYY-MM-DD.md` (examples:
[2026-10-14](../upstream-sync-2026-10-14.md), a merge, and
[2026-10-02](../upstream-sync-2026-10-02.md), a selective port):

```markdown
# Remi upstream sync — YYYY-MM-DD

This sync merges Forge Neo `neo` at
[`<short sha>`](https://github.com/Haoming02/sd-webui-forge-classic/commit/<full sha>)
into `remi`. The previous merge base was `<short sha>` (<previous sync note>).
<For a selective port: the commits taken, and why the rest waits.>

## Integrated as upstream

Merged unchanged, or with conflicts that only kept Remi's hooks:

| Upstream commit | Change |
| --- | --- |
| `<sha>` <subject> | <what users get; where it shows up in Remi> |

## Translated into the Remi UI

| Upstream commit / file | Remi change |
| --- | --- |
| `<sha>` `modules/ui.py` <subject> | <param in remi/core/forms.py, SPA feature, setting widget, …> |

## Adapted for Remi

| File | Resolution |
| --- | --- |
| `<path>` | <conflict and how it was resolved, which Remi hook it keeps; for a file Remi forked (§1.1): Remi's version kept, and which upstream fixes were ported or left out> |

## Skipped

| Upstream commit | Why |
| --- | --- |
| `<sha>` <subject> | <Gradio-only, superseded by a Remi feature, …> |

## Validation

`tools/check.sh` (all steps), `tools/smoke_real.py` on <machine> (txt2img
pixel parity with <baseline>), and what still needs a manual check.
```
