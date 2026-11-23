# Remi documentation

Start with the [project README](../../README.md) ([简体中文](../../README.zh-CN.md)):
what the Remi WebUI is, screenshots and a quick start. The
[usage guide](usage.md) ([简体中文](usage.zh-CN.md)) has the details: every page,
installation, launch options, data locations, keyboard shortcuts, migration
from the Gradio UI, security and troubleshooting.

## Using Remi

| document | for |
|---|---|
| [README](../../README.md) / [README.zh-CN](../../README.zh-CN.md) | users: what Remi is, screenshots, quick start, supported model families |
| [usage.md](usage.md) / [usage.zh-CN.md](usage.zh-CN.md) | users: every page of the interface (Workbench, Recipes, Linear Pipelines, Gallery Boards/Lineage/Compare Diff, Canvas Staging, JSON Builder VGL), Anima & DiT families (`Fun-Controlnet-Union`, Z-Image, ERNIE, Fibo 2.0, Kromoss), install, launch flags (incl. `--worker-process` / `--worker-gpus` / `--worker-mlx`), data locations, shortcuts, migration, security and troubleshooting |
| [Remi Anima generation and parameter guide](../remi-anima-generation-guide.zh-CN.md) (zh-CN) | choosing and combining the Anima samplers, guidance, spatial and LoRA features |
| [api.md](api.md) | script and tool authors: REST `/api/v2`, the WebSocket protocol and binary previews, `/sdapi/v1` compatibility, auth, curl and Python examples |
| [security.md](security.md) | administrators: the threat model, every security control, how to run the server exposed (`--listen`, TLS, proxies), residual risks |
| Built-in extension READMEs (`extensions-builtin/*/README.md`) | what each bundled extension does, e.g. [Agent Scheduler (superseded by the Queue)](../../extensions-builtin/agent-scheduler/README.md) and [Prompt Tools](../../extensions-builtin/sd-webui-prompt-all-in-one-neo/README.md) |

## Developing for Remi

| document | for |
|---|---|
| [extension-guide.md](extension-guide.md) | writing or porting extensions: `REMI_API_VERSION` v1 & `remi-extension.json` manifest, `requires_capabilities`, script hooks, `params()` inputs and layout, conditions, isolated `@ui.action`s, pages, settings, widgets, infotext / presets, localization, `remi.testing` contract tests, the `gr.X` → `ui.Y` porting table |
| [architecture.md](architecture.md) | the layers (`remi.ui`, `remi.core`, `remi.server`, `remi.jobs`, `remi.worker`, `remi.pages`, `webui/`), full-stack `ModelCapability`, Recipes & Linear Pipelines, `WorkerProtocol`, the full `/api/v2` route table, WebSocket events, server, auth, jobs and the porting rule |
| [schema-format.md](schema-format.md) | the JSON contract between `remi.ui` and the SPA renderer: nodes, conditions, values and validation errors, tab / page / settings schemas, action patches |
| [frontend.md](frontend.md) | the React SPA (`webui/`): scripts, structure, design system, API layer, renderer rules, adding widgets and pages, every page's behaviour, i18n, mock mode, tests |
| [../../webui/README.md](../../webui/README.md) | frontend quick commands |
| [remi-architecture-notes.md](../remi-architecture-notes.md) | how the Anima features hook into the Forge runtime (seams) and the trade-offs made on purpose |
| [scorecard.md](scorecard.md) | the acceptance criteria of the Gradio-free rewrite |
| [score-history.md](score-history.md) | independent scoring rounds and the final scores |
| [testing.md](testing.md) / [integration-checklist.md](integration-checklist.md) | the quality gates (`tools/check.sh`), the real-server smoke test and browser audit, and the last full browser walk-through with its evidence |

## Maintaining Remi against Forge Neo

| document | for |
|---|---|
| [upstream-absorption.md](upstream-absorption.md) | where every upstream UI and runtime file went, how to translate each kind of upstream change, `tools/upstream_forks.py` (`review` / `restore` / `budget`), the merge procedure and checklist |
| [upstream-pr-candidates.md](upstream-pr-candidates.md) | code-level audit of all 32 `DIVERGED` files (Wave 1–4 upstream PR roadmap from 32 → ≤ 20) and Python/TypeScript code-bloat refactoring priorities |
| [upstream-sync-2026-11-22.md](../upstream-sync-2026-11-22.md) · [2026-11-18](../upstream-sync-2026-11-18.md) · [2026-10-27](../upstream-sync-2026-10-27.md) | recent Forge Neo merges into `remi` |
| [upstream-sync-2026-10-14.md](../upstream-sync-2026-10-14.md) | the last Git merge of `neo` (`46af880c`) before the UI rewrite |
| [upstream-sync-2026-10-02.md](../upstream-sync-2026-10-02.md) | the earlier selective port |

## Rewrite records

Reference material from the Gradio-to-SPA rewrite. These describe the old
Gradio UI that was replaced; line numbers refer to the pre-rewrite tree.

| document | content |
|---|---|
| [research/webui-replatform-plan.zh-CN.md](../research/webui-replatform-plan.zh-CN.md) | the original replatform plan (its "keep Gradio as /legacy" parts are superseded by architecture.md) |
| [inventory/core-frontend-features.md](inventory/core-frontend-features.md) | every feature of the old core frontend (JS, CSS, HTML) and the SPA checklist (§10) |
| [inventory/extensions.md](inventory/extensions.md) | the built-in extensions and scripts, their UI entry points and port classes |
| [inventory/gradio-api-surface.md](inventory/gradio-api-surface.md) | the Gradio API surface the code used |
| [inventory/runtime-mechanics.md](inventory/runtime-mechanics.md) | the runtime behaviour of the old Gradio app |
| [ports/scripts-postprocess-detail-daemon.md](ports/scripts-postprocess-detail-daemon.md) | ports of `scripts/`, Post Processing Suite and Detail Daemon; the "UI-only params come last" rule |
| [ports/forge-panels.md](ports/forge-panels.md) | ports of the Forge panels (compile, MultiDiffusion, Never OOM, PiD, Radial, Sparse, Spectrum, …) |
| [ports/adetailer.md](ports/adetailer.md) | the ADetailer port |
| [ports/json-builder.md](ports/json-builder.md) | the JSON Builder port (MOSS-Image Layout, FLUX.3 BBox & Fibo 2.0 VGL JSON; dedicated SPA page) |

## Research & Benchmarks

| document | content |
|---|---|
| [research/inference-performance.md](../research/inference-performance.md) | cross-platform CUDA / Apple Silicon MPS / CPU-synthetic inference, memory, quantization, acceleration & 100-job soak benchmarks (`tools/perf/bench_inference.py`) |
| [research/ui-performance.md](../research/ui-performance.md) | SPA startup, bundle budget, 10k gallery / 5k network scrolling, INP and live-preview overhead measurements |
| [research/competitive-and-architecture-research-2026-11.zh-CN.md](../research/competitive-and-architecture-research-2026-11.zh-CN.md) | competitive benchmarking and code-level architecture audit |
| [research/roadmap-2026-q4.zh-CN.md](../research/roadmap-2026-q4.zh-CN.md) | strategic positioning, guardrails and P0–P2 evolution roadmap |
| [research/anima-generation-optimizations.md](../research/anima-generation-optimizations.md) | Anima generation optimization research |
| [research/readme-format-reference.md](../research/readme-format-reference.md) (zh-CN) | how the READMEs are structured |
