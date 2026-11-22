# Remi upstream sync — 2026-11-22

This sync merges Forge Neo `neo` at
[`5998fb39`](https://github.com/Haoming02/sd-webui-forge-classic/commit/5998fb39ea347dfe3a72a8ec88a971e909ed8738)
into `remi` (runtime commit `d6f18e36`, UI commit `db117cb2`, docs commit
`5998fb39`). The previous merge base was `89e2e09d`
([2026-11-18](upstream-sync-2026-11-18.md)). It follows
[upstream-absorption.md](remi-webui/upstream-absorption.md): the runtime is a
Git merge, and the changes to upstream's Gradio UI are translated into SPA
code. `tools/upstream_forks.py restore` reported no upstream commit touching
a Remi fork.

Notable for users:

- **GGUF loading keeps quantization on reload**: weights that land in
  GGUF-native layers (`ForgeOperationsGGUF` Linear / Conv2d / Embedding) stay
  `ParameterGGUF` through `load_state_dict` — first and second load alike —
  while ordinary torch layers dequantize them on load, including a
  meta-device assign at the parameter's dtype (`backend/operations.py`,
  `backend/state_dict.py`).
- **JSON Builder reworked** (below): per-item colours, canvas backgrounds,
  a field editor for captions and texts, direct load from the generation
  tabs, an expandable item list, a help dialog, proportional box rescaling
  on committed size changes, and "Extra" on its own prompt line.
- Forge Neo version **2.33** (`modules_forge/forge_version.py`).

## Integrated as upstream

| Upstream commit | Change |
| --- | --- |
| `d6f18e36` | `backend/operations.py`: `ForgeOperationsGGUF` layers hold `_initial_device` / `_computation_dtype` instead of the one-shot `dummy` dict, so a second `load_state_dict` re-wraps and registers the parameters (with the nvfp4 repack where supported) instead of assigning raw weights; `tensor2parameter` keeps `ParameterGGUF`. `backend/state_dict.py`: GGUF entries dequantize only when their target module is not a GGUF-native layer; the meta-assign dtype conversion skips `gguf_cls` entries. Merged verbatim; no Remi hooks touched |

## Translated into the Remi UI

| Upstream commit | Remi change |
| --- | --- |
| `db117cb2` | The upstream JSON Builder rework lands on Remi's standalone page: indexed spawn offset so new items step down-right (`defaultBbox`/`newItem`, `addItem` passes the item count); `rescaleBox` scales boxes proportionally on a committed size change (`setSize(w, h, true)` from `committedWidth`/`committedHeight`, which a `load` and a restore also set — a typed size only previews, and the committed size is what persists); per-item colours (`itemColor`, `--chart-N`) on the box border/tag and the list number; canvas backgrounds (file blob URL or the selected session result through `api.fileUrl`, never persisted or compiled, revoked on replace/clear/unmount); an expandable item list (`itemsExpanded`, `aria-expanded`); a help dialog on the canvas |

The upstream "prompt link" (`addPromptLink`, its styles dropdown and the
`/internal/json_builder_prompt_styles` endpoint) becomes a shared
`LazyPromptEditor` inside the lazy `FieldEditor` dialog instead of syncing
fields to the main prompt: double-click a caption/text field (canvas or
list) or its pencil button opens it (`scope={{ form: null, extension:
true }}`, `kind="positive"`, `lint={false}`). Styles come from the existing
`stylesQuery` + `api.styles.apply` (`negative_prompt: ""`); only the
returned prompt is written back, and only when the draft's revision and the
field's value still match what was sent — a deletion, load or close drops
the answer. The `/internal` endpoint and the extra-networks page
registration are not ported (the styles API already exists); the
Gradio-only `style.css` and `main.js` stay deleted.

`db117cb2`'s "Load from Prompt" becomes `FileSection` buttons "Load from
txt2img / img2img" reading `getFormStore(tab).getState().values.core`
(persisted drafts included): a nonempty prompt and finite positive size go
through `splitPrompt`/`decompilePrompt`; anything else leaves the draft
alone and toasts why. No navigation, the form untouched. Its final-prompt
layout change lands too: the compact JSON and "Extra" on separate lines,
with newlines inside captions/meta/text flattened to spaces before
`JSON.stringify` (upstream's raw `\\n` replace would corrupt literal
backslashes — escaping stays the compiler's).

## Adapted for Remi

| Upstream commit | Resolution |
| --- | --- |
| `5998fb39` | `forge_version.py` 2.33 taken; its README additions adapted into Remi's READMEs (en, zh-CN): the JSON Builder line in the feature list and the GGUF reload note |

| File | Resolution |
| --- | --- |
| `README.md` (conflict) | Remi's kept; upstream's JSON Builder line adapted into the feature list |
| `extensions-builtin/sd_forge_json_builder/scripts/builder.py` (conflict) | Remi's kept: the page stays the standalone `pages.register(Page("json-builder", …, widget="json-builder"))` custom page — not moved into the networks view |
| `javascript/main.js`, `style.css` (deleted by Remi, modified upstream) | Stay deleted; the SPA page and the design system replace them |

## Validation

- `PYTHON=venv/bin/python tools/check.sh --strict`: the full run passed 55
  of 59 steps, with no skipped steps. The four failures were corrected:
  `openapi.json` was regenerated for version 2.33, an unused test import was
  removed, and a test fixture was given its parsed `LayoutDraft` type.
  The failed `ruff:tests`, `contract:openapi` and `pnpm:build` steps then
  passed via `tools/check.sh --strict --only
  ruff:tests,contract:openapi,pnpm:build`; the sole failing core pytest
  case, `test_openapi_covers_the_api_and_the_export_is_current`, passed
  separately. The complete gate was not rerun. The original run included
  1,929 passing core Python tests, 60 passing subtests and two skipped
  Python tests, all extension suites, 1,345 passing Vitest tests and 148
  passing Playwright tests. `pnpm gen:api` produced no type changes.
- `tools/check_line_endings.py --update` regenerated the list from the new
  merge base without changing its contents; all 80 upstream CRLF files
  pass the check. The `/sdapi/v1` contract matches `5998fb39`.
- Targeted tests (before the gate): `pytest` on the GGUF adapter, the new
  `test_gguf_loading.py` and the JSON Builder extension tests — 14 passed;
  `vitest` on `src/pages/custom/json-builder` + locales — 97 passed;
  Playwright `flows.spec.ts` "JSON Builder" e2e — 1 passed (real CodeMirror
  field edit, a mock style applied through the real combobox, background
  upload, committed-size rescale, send → direct load back, expand/collapse).
- Visual/accessibility: an audit of 16 rendered states (light/dark
  × phone/desktop: canvas, background, expanded list, field dialog, help)
  plus `@axe-core/playwright` on the field and help dialogs — 8 analyses,
  zero violations. The ui-audit probe's only new finding (pencil-button row
  alignment) was fixed.
- Post-sync smoke (`tools/smoke_real.py`, Anima Turbo): 10 checks pass, 2
  fail — the **same** txt2img pixel-parity failure as pre-sync, with the
  identical numbers (mean 8.3677, max 247) on both `/sdapi/v1` and
  `/api/v2/jobs`. Infotext checks pass, allowing the version change;
  img2img, inpaint, extras and server lifecycle checks pass. Comparing the
  five output images with the captured pre-sync run gives identical RGBA
  pixels in every image (maximum difference 0). The historical parity
  drift predates this sync; this sync does not diagnose it.

Still untested: a real GGUF checkpoint load and model switch on a GPU (the
regression tests use synthetic Q8_0 weights on CPU and do not exercise the
forward path).
