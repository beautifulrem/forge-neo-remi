# remi architecture notes

How the Anima features hook into Forge Neo, and the trade-offs that were made
on purpose. Read this before "fixing" one of them.

## Seams

| Seam | Where | Used by |
|---|---|---|
| Attention modifier chain | `backend/nn/anima_attention.py` (`ANIMA_ATTENTION_MODIFIERS`, `ANIMA_PROJECT_KV`) | NAG |
| Masked attention (a call with a mask runs on PyTorch SDPA when the active backend is Sage, which ignores masks, or Flash, which rejects them) | `mask_honouring_attention` in `backend/nn/anima_attention.py` | FreeFuse's concept bias |
| Temporary forward overrides | `modules/forward_override.py` | Regional, Artist Mixer, FreeFuse, DAVE, cross-attention boost, NegPiP's DiT hook (`lib_negpip/anima.py`) and its SD1/SDXL cross-attention hook (`lib_negpip/sd.py`) |
| Per-call block forward pre-hooks (registered inside a `model_function_wrapper`, removed in its `finally`) | `_block_deltas` in `extensions-builtin/sd_forge_anima_guidance/lib_anima_guidance/modulation.py` | Modulation (adds its AdaLN-LoRA delta to the modulated blocks' input) |
| `compute_qkv(value_context=...)` / `compute_kv(value_context=...)` (cross-attention only) | `backend/nn/anima.py` | NegPiP, FreeFuse, NAG's negative-prompt K/V via `ANIMA_PROJECT_KV` (V from the masked context) |
| `model_function_wrapper` chain | `call_model_wrapper` in `modules/anima_support.py` | Modulation, Regional, FreeFuse, Artist Mixer, DAVE, cross-attention boost, CFG-Zero* zero-init (skips the forward of zero-init steps), Spectrum |
| Windowed forward-override patches (clone, gated `model_function_wrapper`, per-call overrides under a lock with guaranteed cleanup, wrapper flags, Spectrum request); `outermost` installs around the model function so the overrides wrap every other override of the same modules in either wrapper order | `windowed_override_patch` in `modules/anima_support.py` | Regional, DAVE (`outermost`: attenuates the cross-attention boost's norm-matched block output whichever hook runs first), cross-attention boost |
| Generated frames vs edit reference frames (an edit model's reference latents are appended after the generated frames along T, so their tokens follow the generated ones) | `generated_token_frames` / `pad_to_sequence` in `modules/anima_support.py` | Regional (region weight 0 on references), DAVE (mean over generated frames only), FreeFuse (LoRA routing factor 1 and no concept bias on references) |
| Spectrum "run a real forward" requests | `modules/spectrum_force.py` | every feature Spectrum cannot forecast |
| Replaced blocks in a branch's signature (the block keys of `transformer_options["patches_replace"]` are part of each cond/uncond branch's conditioning signature, so a branch whose blocks are replaced for part of the pass rebuilds its history at the switch instead of forecasting across it; a script that sets a replace patch on every call of a pass, as IP-Adapter's `attn2` and the generic patcher do, changes nothing) | `_replaced_blocks` in `extensions-builtin/sd_forge_spectrum/lib_spectrum/forecaster.py` (read in Spectrum's `model_function_wrapper`) | Spectrum, against Skip Layer Guidance (`modules/processing_scripts/slg.py` replaces Moss-Image `single_block`s in the uncond call between two percents); a change to `slg.py` or to the `patches_replace` structure is checked against Spectrum (`extensions-builtin/sd_forge_spectrum/tests/test_replaced_blocks.py`) |
| Stateful guidance registry | `modules/anima_guidance_state.py` | DCW, SMC-CFG, APG, CFG-Zero*, Momentum register; samplers that change what the history describes reset it (FreeFuse after its probe pass, SPEED at its resolution switch) |
| Active-feature registry | `set_feature_active` / `is_feature_active` in `modules/anima_feature_conflicts.py` | FreeFuse, Regional, Artist Mixer |
| Online-LoRA leases | `acquire_online_lora` / `release_online_lora` in `modules/anima_online_lora.py` (the last release restores the user's setting) | Stage scheduler, FreeFuse |
| Post-load validation | `Script.after_model_load` in `modules/scripts.py`, run by `process_images` once checkpoint/VAE overrides have loaded | features that check the loaded model or the final settings (Hires Guard) |
| Guaranteed teardown | `Script.cleanup` in `modules/scripts.py`, run from `process_images`' `finally` | features that must undo global state even when a run raises |
| Infotext paste keys (a `str` key that is also the paste function: `table_params` makes it the input's `infotext` key and its `paste` function, so the SPA paste, "Send to" and the `/sdapi` infotext replay all read the value through it via `ScriptLayout.paste`) | `PasteKey` and `table_params` in `modules/anima_support.py` | Artist Mixer (one recorded chain fills many row controls), Layer Weight (a text the run left out pastes empty) |
| Script-argument check (one policy, applied before a hook changes anything: more values than names raise `TypeError` naming the panel and the counts; fewer are filled from the panel's defaults when it has them, for argument lists stored before a control was added such as an imported Agent Scheduler task or a positional `/sdapi` request, else `TypeError`. The API itself fills every slot from the UI defaults first. A float value that is not finite (NaN, ±Infinity: the API reads JSON `NaN` / `Infinity` as floats, and a recorded "nan" replays as one) raises `ValueError` naming the panel and each such argument; text that reads as "nan" is kept. Forge's runner reports the exception and skips the panel for that hook) | `script_arguments` in `modules/anima_support.py`; panels that take their args as named parameters call its finite check, `require_finite`, first | Guidance (defaults for its later controls), Stage Scheduler, Layer Weight (split by adapter kind afterwards), FreeFuse, Regional, Spectrum (defaults for the controls upstream lacks), Artist Mixer (exact count: its layout has not changed); `require_finite`: Hires Guard, DAVE, cross-attention boost |
| Bounding a setting (`clamp(to_float(value, default), low, high)`, no upper bound by default: `to_float` gives `default` for anything that is not a finite number, where `min(max(nan, low), high)` is NaN and `max(low, min(high, nan))` is `high`) | `clamp` / `to_float` in `modules/anima_support.py` | every remi panel and lib that bounds a setting (Hires Guard, DAVE, cross-attention boost, Regional, Guidance, NAG, FreeFuse, Artist Mixer, Stage Scheduler, LoRA stage windows) |
| Load-time quantization | `quantize_on_load` in `backend/nn/anima_quant.py`, via "Diffusion in Low Bits" | scaled FP8, INT8 ConvRot |
| DiT capability registry (declarative per-family rules for precision-sensitive prefixes, FP8/INT8-ConvRot quantization, Diffusers/sharded checkpoint discovery and single-file conversion, per-block `torch.compile` attribute/preset/engine checks, Spectrum pre-head capture/fast-forward, and non-blocking workflow advice; keeps upstream `loader.py`, `sd_models.py`, `comfy/utils.py`, and `slg.py` to 1–3 line hooks) | `DiTCapability` / `register_dit_capability` in `backend/nn/dit_registry.py` | Anima, MOSS-Image, Krea 2, FLUX.3 Image Klein (and future DiT families: `anima_quant.py`, `loader.py`, `sd_models.py`, `comfy/utils.py`, `sd_forge_compile`, `sd_forge_spectrum`, `slg.py`) |
| Anima depth layouts (28 / 40 / 52 blocks: which blocks of a 2.9B / 3.8B are the 2B's, which were inserted) | `backend/nn/anima_layout.py` | LoRA re-mapping (`networks.process_anima`), Anima LLLite (`map_blocks`), LoRA file scans (`modules/anima_lora_files.py`), and through `modules/anima_lora_support.py` Layer Weight and the stage scheduler |
| Per-step text conditioning (an extra conditioning tensor reaches the DiT as a keyword: a cond dict's keys other than `crossattn` become `Condition`s in `compile_conditions`; cond and uncond with different keys are not batched) | `Anima.semantic_context` in `backend/nn/anima.py`, `compile_conditions` in `backend/sampling/condition.py`, `Qwen06Engine.semantic_inputs`; MultiDiffusion repeats the `anima_*` keys and NegPiP's mask per tile (`tiled_diffusion.py`) | Anima-3.8B's Semantic Connector (`backend/nn/anima_connector.py`, Qwen3.5 4B: `backend/nn/anima_qwen35.py`); NegPiP keeps the extra keys (`lib_negpip/anima.py`) |
| Text-side patches of a control model (before each batch's prompts are encoded; the prompt cache is cleared when the set of such units changes) | `ControlModelPatcher.process_before_conditioning`, ControlNet's `process_batch` | Anima control-LoRA's LLM adapter LoRA |
| Control on the patch embedding (a forward hook on `x_embedder`, removed after sampling and by ControlNet's `cleanup`) | `ControlInjection` in `extensions-builtin/sd_forge_controlllite/lib_controllllite/anima_control_lora.py` | Anima control-LoRA |

## Deliberate trade-offs

- **Upstream runtime files stay close to upstream.** (The UI is not shared
  with upstream any more: [upstream-absorption.md](remi-webui/upstream-absorption.md)
  maps upstream UI files to their Remi equivalents.) Changes to Forge Neo files
  (`backend/loader.py`, `modules/processing.py`, `modules/scripts.py`,
  `extensions-builtin/sd_forge_lora/networks.py`, ...) are kept to small hooks
  so upstream merges stay easy. For the same reason the per-block modulation
  locals in `Block.forward` (`backend/nn/anima.py`) mirror upstream line for
  line, and the 2B to 2.9B/3.8B block tables are duplicated in
  `modules/anima_lora_support.py` rather than edited out of `networks.py`; a
  test keeps the two identical. Shared files also keep upstream's line
  endings (CRLF for most of `modules/`), with no `.gitattributes`
  conversion, so a merge compares real lines; `tools/check_line_endings.py`
  (a `tools/check.sh` step) fails when an editor saves one of them as LF
  ([upstream-absorption.md §3](remi-webui/upstream-absorption.md#3-procedure)).
  The few files Remi forked outright (Spectrum's `lib_spectrum`, the compile
  script, ControlNet's unit parsing) are the exception: a merge must not
  replace them even when it merges cleanly, so `tools/upstream_forks.py
  restore` puts them back after `git merge` and their upstream fixes are
  ported by hand ([upstream-absorption.md §1.1](remi-webui/upstream-absorption.md#11-built-ins-whose-processing-remi-forked)).
  `/sdapi/v1` stays compatible with Forge Neo's: `tools/sdapi_contract.py`
  (run by `tests/test_remi_sdapi_contract.py`) compares the routes and models
  with upstream's.
- **Three LoRA extensions wrap `ModelPatcher.add_patches`** (layer weight,
  stage scheduler, FreeFuse metadata) through `backend/patcher/function_chain.py`
  instead of a patch-hook registry in `sd_forge_lora`. A registry would be
  cleaner but means rewriting ~5k lines of bundled extensions whose effect
  can only be checked on a GPU.
- **Regional, Artist Mixer and FreeFuse override cross-attention `forward`**
  rather than being attention modifiers. They swap in whole contexts (with
  their own NegPiP masks) and blend outputs; moving them onto the modifier
  chain is possible but changes numerics that need GPU A/B tests first.
- **Modulation uses per-call forward pre-hooks, not forward overrides.**
  Per-block compile runs any call with a forward override inside the blocks
  eager (`patched_inside` in `sd_forge_compile/anima_block_compile.py`), so
  an override would cost modulation users the compiled blocks for the whole
  generation. Dynamo traces a pre-hook as a frame of its own and keeps the
  compiled blocks, and adding or removing the hooks between calls
  recompiles nothing (`sd_forge_anima_guidance/tests/test_modulation_compile.py`). The hooks live
  only for the model call, like an override: between calls, and whenever
  modulation is off, the blocks carry none.
- **NegPiP patches `compile_conditions`** to carry its mask through
  conditioning. This is the bundled Forge NegPiP extension's own design;
  changing it would mean diverging from that extension and upstream sampling.
- **Prompt rewrites are opt-in** (`modules/prompt_rewrites.py`): a global
  restore in `process_images` would also undo Loopback's intentional edits.
- **`ScriptAbort`** is only re-raised from `before_process_batch`, where the
  per-batch LoRA rules are validated; other hooks keep upstream's
  report-and-continue behaviour.
- **Not cached, on purpose:** FreeFuse's per-LoRA contributions, since the
  stage scheduler changes LoRA strengths every step through the same patch
  objects (plain LoRAs run as two low-rank matmuls each call; other adapters
  rebuild a dense delta); Spectrum's per-call prompt signature (split cond/uncond calls
  carry different branches within one step).
