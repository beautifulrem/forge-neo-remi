# Using Forge Neo Remi

[English](usage.md) | [简体中文](usage.zh-CN.md)

The reference for users: every page of the interface, the Anima features, installation, launch options, where data is stored, keyboard shortcuts, migration from the Gradio UI, security and troubleshooting. The [project README](../../README.md) is the short overview; the other documents are in the [documentation index](README.md).

**Contents:** [The Remi WebUI](#the-remi-webui) · [Anima Integration](#anima-integration) · [Quick Start](#quick-start) · [Launch Options](#launch-options) · [Where Data Is Stored](#where-data-is-stored) · [Keyboard Shortcuts](#keyboard-shortcuts) · [Migrating from the Gradio UI](#migrating-from-the-gradio-ui) · [Built-in Extensions](#built-in-extensions) · [Security and Remote Access](#security-and-remote-access) · [Troubleshooting](#troubleshooting) · [Compatibility Notes](#compatibility-notes)

## The Remi WebUI

The browser app (`webui/`) talks to the server only through a documented API ([`/api/v2`](api.md) plus a WebSocket for progress and live previews); the server never renders HTML. Every page is described below.

**Shell.** A slim navigation rail on the left (a bottom bar and a "More" sheet on phones) leads to the pages; extension pages appear between the built-in pages and Settings. The header shows the page title, the **model bar** (UI preset, checkpoint with architecture, size, and auto-detected capability/format badges such as `Diffusers`, `GGUF`, `INT8-ConvRot`, `FP8`, `NVFP4`, `Turbo`, `RGBA`, `Control`, VAE / text encoders, "Diffusion in Low Bits"), a **worker health pill** (in-process, subprocess, multi-GPU pool, or MLX sidecar status), the ⚡ **quick settings** you pinned, a **Networks** button, the **command palette** (Ctrl/⌘+K: every page, command and shortcut, searchable), a connection indicator and the **Appearance** menu (light / dark / system theme, comfortable or compact density, accent color, language). A thin progress line runs under the header while a job generates.

**Generate (txt2img and img2img).** One view with a txt2img / img2img switch, each with its own form. On wide screens it has three resizable, collapsible panes:

- *Parameters*: the run controls at the top (Generate, Add to queue, Interrupt, Skip, **Recipes** picker to save/apply cross-model parameter subsets, and a menu with generate forever, generate N times and completion notifications), pre-submit **Workflow Advice** chips offering one-click fixes for mismatched settings (e.g. Turbo 8 steps / CFG 1.0, Kromoss 8-step DMD2, RGBA → PNG output), the prompt and negative prompt in a syntax-highlighting editor (attention weights, LoRA tags, schedules, wildcards, bracket checks, tag autocompletion, token counter, styles, and the **Prompt Upsampler (PUP)** button with multimodal reference/BBox support and diff preview), then the parameter cards in your configured order: sampler, size (aspect presets, "W x H" paste), guidance, Hires. fix, seed, batch, and every extension panel as a collapsible section. Panels declare model capabilities (`requires_capabilities`), so panels that do not apply to the active model family (e.g. Anima-only panels on FLUX.3 or Qwen-Image-2.1) collapse into an "inapplicable for current model" drawer with an explanation; a searchable list also supports "only modified" filtering and per-panel reset. A dot next to a changed value resets it to its default. Values you type are remembered per browser.
- *Stage*: the live preview while a job runs (step, batch, ETA, queue position), then the selected result with actions: download, save, copy parameters, reuse seed, use all parameters, send to img2img / inpaint / **Reference Images (`ImageStitch`)** / Extras. Reference lists display model-aware slot tokens (`<image_1>` for Qwen-Image-2.1, `ref_image_0` for FLUX.3, `ref_0` for Fibo 2.0) that click-insert into the prompt and support drag reordering. In img2img the stage is the **image editor** for the source image: brush, eraser, fill, rectangle and lasso selections, eyedropper, layers for sketch and inpaint masks, undo/redo, zoom and pan, pen pressure, **in-place Staging** (generate 1–4 inpaint candidates over the active region, cycle through them in place, and accept or discard without leaving the canvas), and overlays that show the resize frame and the "only masked" area.
- *History*: this session's results as a virtualized thumbnail strip; a full-screen viewer with zoom, pan, tile preview, **RGBA 4-backdrop** switcher (checkerboard / black / white / green via toolbar or `B`), and gamepad support.

Below 72 rem the history becomes a strip; on phones the stage comes first and the run controls stick to the bottom of the screen. Drop an image anywhere on the prompt to read its generation parameters.

**Extras.** Upscaling, face restoration and the other postprocessing operations on a single image, a batch, a folder or a video, with a before/after comparison slider.

**PNG Info.** Drop, paste or choose an image to see its generation parameters, every metadata entry (text chunks, EXIF, PNG chunks), and send the parameters or the image to another page.

**Gallery.** Every image in your output folders, indexed: search the prompt, file name or model; filter by form, folder, model, **project Boards**, date range and stars; justified or masonry layout; select many to star, add to a board, send a whole board to Reference Images, download as zip, **compare two** (pixel slider/side-by-side plus a structured **Parameter Diff** drawer highlighting prompt, seed, sampler, CFG, steps, LoRA, and extension differences with one-click copy to form), or delete. The viewer and details drawer show **asset lineage** (parent jobs, retries, img2img/inpaint/upscale descendants), parameters, RGBA backdrop switching, and can reveal the file in your file manager.

**Networks.** One browser for LoRA / LyCORIS, embeddings, checkpoints, VAE and text encoders, as a full page or a drawer next to the prompt (Alt+N): folder tree, architecture filters ("compatible with the current preset"), sort, card and list views, previews (images or videos), metadata and training tags, an editor for each item's user metadata (description, activation text, preferred weight, a checkpoint's preferred VAE / text encoders that its card selects with it, …), setting a preview from your results, and a refresh button that rescans the folders (it also updates the model bar). Thousands of items scroll smoothly.

**Queue.** The running job with its live preview, the queued jobs in run order (drag or keyboard to reorder, high / normal / low priority, duplicate, edit, cancel), pause / resume, and the searchable history of every job with its outputs, stars and "queue again". Also hosts **Linear Multi-Stage Pipelines** (`Prompt Upsample → Generate → Refine → Finish`: save reusable templates and re-run after changing a downstream stage while automatically reusing cached outputs from unchanged upstream stages) and the **Recipes** library (save, edit, import/export, and apply cross-model parameter patches). Imports tasks from the old Agent Scheduler and Prompt Queue.

**Settings.** All settings in one searchable, virtualized list with categories, "changed" / "unsaved" / "pinned" filters, per-setting reset, export and import (also of an A1111 `config.json`), plus server tools: Reload UI, restart, unload or reload the model, show the weights in memory, hash all checkpoints, system info and licenses. A setting that holds a value of the wrong kind (a hand-edited `config.json`, an extension's assignment; `/sdapi/v1/options` checks a changed value as the settings page does, see api.md), a value its fixed choices do not offer (a list that accepts a typed value has none), or null reads as its default, and the server resets it to the default when it starts, with a warning in the log. A number outside its slider's range reads as the nearest end of that range. Upstream code and extensions read such a value set while the server runs (by an extension) as it is, until the next start.

**Extensions.** Installed extensions (enable, disable, check for updates, update, uninstall), **Diagnostics** (per-extension `remi-extension.json` manifest compatibility, load time, recent hook warnings/errors, isolated action errors/timeouts, and worker status), the extension index (by default the third-party [Forge Neo index](https://github.com/eduardoabreu81/forge-neo-extensions); verify an extension before installing it) with search, tag filters and a recent-updates filter ("2025+"), install from a git URL, and configuration backups you can restore.

**Extension pages.** Extensions add their own pages: **JoyCaption** (image captioning, single or folder, with progress and cancel), **Prompt Workshop** and **JSON Builder** ship with Remi.

The JSON Builder lays out structured prompts for **MOSS-Image Layout**, **FLUX.3 Image Klein BBox**, and **Bria Fibo 2.0 VGL 2.0** (displaying the CC BY-NC 4.0 non-commercial weights notice when VGL mode is selected) as boxes on a canvas: drag a box to place it, its corner to resize it (arrow keys nudge the focused box, Shift for 16 px), scroll to zoom and middle-drag to pan. Each item has its own colour, shared by its box and its number in the list. Changing the canvas size keeps every box at the same relative position and size. The item list can expand over the rest of the side panel for long layouts, and the ? button on the canvas shows a short manual.

A caption or text field opens in a **field editor** — double-click it, or use its pencil button: the same prompt editor as the generation fields, with completions, plus a styles picker whose "Apply styles" writes the styled prompt into that field only. The buttons at the canvas's top left can put an image under the boxes as a drawing guide — a file or the selected session result; it is never saved or sent.

"Send to" writes the compiled prompt — the JSON on one line, "Extra" on its own line — and the size into the tab; "Load prompt…" pastes one back, and "Load from txt2img / img2img" reads the tab's current prompt and size.

**Prompt tools.** Under every prompt: a tag view (chips you can reorder, reweight, disable or delete), quick tag groups, prompt history, favorites in collections, optional translation through a provider you configure, and, with an OpenAI-compatible provider, "Write a prompt" (an LLM writes or changes the prompt from a description in any language). Nothing is sent anywhere until you choose a provider.

**API.** The same server offers [`/api/v2`](api.md) (what the SPA uses: named parameters, validation, jobs, WebSocket events) and, with `--api`, the A1111-compatible `/sdapi/v1`. Interactive docs are at `/docs`.

## Anima Integration

| Area | Included capabilities |
| --- | --- |
| Sampling | Beta57, Flow Euler, UniPC2, PC3, SPEED (experimental half-res early steps; switch sigma adjustable in Settings), ER SDE, Dynamic Shift |
| Guidance | NAG, Momentum Guidance, Skimmed CFG, SMC, CFG-Zero*, APG, TCFG, FDG, DCW, CNS, CFG active range (CFG only within a model-time window: start/end are fractions of the model's denoising time, converted to sigma with `percent_to_sigma`, not step fractions; outside it the negative pass is skipped unless another hook reads it), CLIP Modulation guidance (on first use by default it downloads a pinned adapter and CLIP-L; local files also work) |
| Spatial control | Regional Conditioning, FreeFuse multi-LoRA routing, Artist Scheduled Mixer |
| Diversity | DAVE (early-step DC attenuation for same-prompt layout diversity across seeds) |
| Prompt adherence | Cross-Attention Boost (positive-prompt cross-attention gain in the high-noise steps, norm-matched, for weak tags and bindings) |
| LoRA workflow | Layer weights, stage scheduling, LoRAs across 2B/2.9B/3.8B (28/40/52 blocks), prompt rescaling |
| Reliability | Feature conflict resolver, Hires Guard, generation metadata, safe fallbacks |
| Acceleration | Spectrum forecasting and Anima per-block `torch.compile` (after the first recompile, a changed resolution or batch size reuses one dynamic-shape graph; batch 1 (CFG 1, or steps outside the CFG range) and reference frames still get graphs of their own) |
| Low-bit loading | **Diffusion in Low Bits**: `float8-e4m3fn (Anima scaled)` (per-tensor scaled FP8, same VRAM as plain FP8, more precise) and `int8 (Anima ConvRot)` (INT8 GEMMs; CUDA 13 or `--enable-triton-backend`, SM 8.0+) |

Read the [Remi Anima generation and parameter guide](../remi-anima-generation-guide.zh-CN.md) before combining experimental methods.

### Anima 2.9B and 3.8B checkpoints

**28/40/52-block support:** detects original (28 blocks), 2.9B (40) and 3.8B (52) Anima DiT layouts. Layer Weight, stage-scheduler block weights and FreeFuse's collect block keep following the 28-block layout on the deeper models.

| Model | Files | Preset |
|---|---|---|
| Anima 2B (preview, base, aesthetic, turbo) | checkpoint in `models/Stable-diffusion`; `qwen_image_vae` and `qwen_3_06b_base` as VAE / Text Encoder modules | Anima presets |
| [Anima-2.9B](https://huggingface.co/Gazingstars123/Anima-2.9B) preview-v1 | the same modules | Anima-2.9B (Euler, SGM Uniform, 30 steps, CFG 4) |
| [Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B) v1.1 | the same modules **plus `qwen35_4b.safetensors`** (in `models/text_encoder`, selected as a third module) | Anima-3.8B (Res Multistep, Beta, 32 steps, CFG 5) |

**Anima-3.8B.** The v1.1 checkpoint bundles a Semantic Connector (v2) that rewrites the text conditioning at every denoising step from the Qwen3.5 4B encoder's hidden states (layers 7, 15, 23, 31), around the model's own LLM adapter. Remi runs it natively: the connector stays in the diffusion model, the Qwen3.5 encoder (FP8 weights, kept as stored) encodes the prompt once, and the context is recomputed at each step. Without `qwen35_4b` selected, a 3.8B generation stops with a message naming the file; selecting it with another model only logs a warning. As in the model's reference workflow, the negative prompt uses only Anima's own encoder; **Settings → Stable Diffusion → [Anima 3.8B] Semantic connector for the negative prompt** sends it through the connector too. NegPiP and MultiDiffusion work on the connector's output; the extra prompts of Regional and Artist Mixer are encoded by Anima's own encoder only (the main prompt still goes through the connector). The v1 preview's separate progressive adapter, MXFP8 and GGUF 3.8B files are not supported.

**LoRA across depths.** A LoRA trained on one depth is re-mapped onto another; the infotext records it (`Lora remap: name 28→40`). The depth it was trained on comes from the file's metadata, then from its block indices; when that guess is wrong (a 40-block LoRA that only touches early blocks reads as 28), name it in the tag: `<lora:name:1:src=40>`. On a deeper model, **Settings → Extra Networks → Anima LoRA on a deeper model: inserted blocks** decides whether the inserted blocks receive the LoRA of the block they were copied from (`copy`, the default) or nothing (`skip`). On a shallower model the inserted blocks' weights are dropped. A LoRA that cannot be placed at all is listed in the job's warnings instead of being silently ignored. LoRA weights on the LLM adapter also reach the 3.8B connector's copy of the adapter.

**Control.** For Anima the ControlNet model list only holds what runs on its DiT: ControlNet-LLLite models (re-mapped across depths, like LoRAs) and control-LoRAs.

- **Control-LoRA** ([Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose), `anima_pose_preview2.safetensors` in `models/ControlNet`): give the unit a skeleton image (an OpenPose render on black, or the openpose preprocessor) and pick the model. The control image is VAE-encoded and added to the DiT's patch embedding, scaled by the unit's **weight** (0 = no control, 1 follows the skeleton, up to 2) within its start / end. The LoRA trained with it is applied at 1.0 to the DiT for the whole pass and to the LLM adapter before the prompts are encoded, so start / end and weight do not change it. It follows best at about 1024 px with weight 1 to 1.5, legs more readily than raised arms, and it was trained on its own thin skeleton (COCO body points, white joints, face and hand dots on black): a skeleton without the face dots can come out as a back view. The model card asks for detailed prompts; its weights are for non-commercial use.

### DiT Families, ControlNet-Union & Acceleration

Beyond Anima, Remi's declarative `ModelCapability` layer integrates:

- **`Fun-Controlnet-Union` (`DiTControlPatch`):** single-file zero-gated `control_blocks` support for **Qwen-Image-2.1-Fun-Controlnet-Union** (16 even layers, 64/129-ch structural + inpaint) and **Z-Image-Turbo-Fun-Controlnet-Union** (16/33-ch), automatically filtered by the loaded DiT family in ControlNet.
- **Z-Image (Base / Turbo / Omni / Edit)** & **ERNIE-Image:** Diffusers/sharded directory loading, sensitive-layer quantization guards, cross-step text/RoPE caching, per-block `torch.compile`, and Spectrum seams.
- **Bria Fibo 2.0:** VGL 2.0 structured JSON editing (integrated with JSON Builder and PUP) and 4-step Edit Turbo workflow advice (note: Fibo 2.0 weights are CC BY-NC 4.0 non-commercial).
- **MOSS-Image / Kromoss:** native 8-step DMD2 LoRA workflow advice (`steps=8, cfg=1.0`) alongside Layout JSON editing.
- **Spectrum forecasting modes:** alongside the default **Chebyshev** forecaster, Spectrum exposes opt-in **TaylorSeer** (higher-order finite-difference extrapolation) and **MagCache** (magnitude-decay adaptive skip caching).
- **Apple Silicon MPS & benchmarking:** bounded MPS dequantized-weight caching (`backend/operations_mixed_precision.py`) and unified-memory budget awareness (`backend/memory_management.py`); run `python tools/perf/bench_inference.py --synthetic` (or `--real`) to benchmark CUDA, MPS, or CPU-synthetic throughput, memory, and 100-job queue stability (see [inference-performance.md](../research/inference-performance.md)).

## Quick Start

### Requirements

- [Git](https://git-scm.com/downloads)
- [Python 3.13](https://www.python.org/downloads/) or [uv](https://docs.astral.sh/uv/)
- A supported PyTorch device; an NVIDIA GPU is the most widely tested option (Apple Silicon works through MPS)
- For the web interface: [Node.js](https://nodejs.org/) 20.19+ or 22.12+ (needed once per update to build it; see [The web interface build](#the-web-interface-build)). pnpm is used if installed; otherwise the launcher runs it through Corepack or `npx`, which come with Node.js.
- Optional: [FFmpeg](https://ffmpeg.org/) for video export

### Install and run

```bash
git clone --branch remi https://github.com/beautifulrem/forge-neo-remi.git forge-neo-remi
cd forge-neo-remi
```

**Windows:** run `webui-user.bat`. For the recommended `uv` setup, create the environment first and add `--uv` to `COMMANDLINE_ARGS` in `webui-user.bat`:

```bat
uv venv venv --python 3.13 --seed
```

**Linux and macOS:** run `./webui.sh`. Fresh installations prefer `python3.13`; otherwise `uv` finds or installs Python 3.13 and creates the virtual environment. Existing environments are reused without being deleted or rebuilt. On macOS the launcher installs native `torch` and `torchvision` from PyPI instead of CUDA wheels. Put your own settings in `webui.settings.sh` at the repository root (ignored by Git):

```bash
export PYTHON="/path/to/python3.13"
export COMMANDLINE_ARGS="--uv"
# Optional: override the automatically selected PyTorch installation command
# export TORCH_COMMAND="pip install torch torchvision"
```

The first launch installs the Python packages, builds the web interface and prints `Running on local URL:  http://127.0.0.1:7860/`; open it in a browser (or start with `--autolaunch`). Put checkpoints and companion modules in the matching `models/` subfolders (`Stable-diffusion`, `VAE`, `text_encoder`, `Lora`, …), then pick them in the model bar. Files added while the WebUI runs appear after the refresh button of the Networks browser (Alt+N) on the matching tab.

See the upstream [Unix installation guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Unix) for platform packages. Apple Silicon can use PyTorch MPS, but CUDA-only attention and compilation backends are unavailable and generation is generally slower than on a dedicated GPU.

### The web interface build

The interface is a static app built from `webui/` into `webui/dist`. The launcher (`launch.py`, run by `webui.sh` / `webui-user.bat`) builds it automatically when `webui/dist` is missing or older than its sources: after a fresh clone and after every update that touched `webui/`. It runs `pnpm install --frozen-lockfile` and `pnpm build` with the pnpm it finds (installed pnpm, then Corepack, then `npx pnpm@12`), and stamps the build so unchanged sources are not rebuilt.

Without Node.js on the machine:

- An existing build is used as it is, with a warning that it may be out of date.
- With no build at all the server still starts (the APIs work) and the browser shows a page explaining how to build the interface.
- To run Remi on a machine without Node.js, build once on any machine that has it (the output is plain static files, the same for every OS) and copy the folder over:

  ```bash
  cd webui
  pnpm install --frozen-lockfile   # or: npx pnpm@12 install --frozen-lockfile
  pnpm build                       # writes webui/dist
  ```

  Copy `webui/dist` into the same place of the target installation. Rebuild and copy again after each update that changes `webui/`.

`REMI_SKIP_WEBUI_BUILD=1` turns the automatic build off (any non-empty value does, `0` included); `--skip-install` and `--skip-prepare-environment` skip it too.

### Docker (NVIDIA or AMD GPU)

`docker/Dockerfile` builds an image of this checkout (the build context is the repository root): the web interface is built in a Node.js stage, the server image gets Python 3.13, the launcher's PyTorch build and `requirements.txt`, and no Gradio. Every piece of user data lives in one volume, `/home/remi/data`. See [docker/README.md](../../docker/README.md).

```bash
docker build -f docker/Dockerfile -t remi-webui .
docker run --gpus all -p 127.0.0.1:7860:7860 -v remi-data:/home/remi/data remi-webui
```

For AMD GPUs, `docker/Dockerfile.rocm` is the same recipe with the ROCm build of PyTorch:

```bash
docker build -f docker/Dockerfile.rocm -t remi-webui-rocm .
docker run --device /dev/kfd --device /dev/dri --group-add video -p 127.0.0.1:7860:7860 -v remi-data:/home/remi/data remi-webui-rocm
```

The container listens on every interface and has no login by default, so these commands publish the port on this machine only. To reach it from other machines, set a login with `--auth-file` (see [docker/README.md](../../docker/README.md#run)) and read [Security and Remote Access](#security-and-remote-access).

### Recommended first Anima run

1. Choose a supported Anima checkpoint and its text encoder and VAE in the model bar.
2. Open the **Anima Remi Presets** panel in the Generate view.
3. Apply **Official Base/Aesthetic — Safe (Recommended)**.
4. Generate once with optional guidance, spatial controls, Hires and compilation disabled.
5. Add one advanced feature at a time and compare with the same seed.

The safe preset starts from `ER SDE`, `36` steps, `CFG 4.5` and the model's standard shift behavior. It configures inactive modules reproducibly but does not silently enable experimental stacks. For Anima-Turbo checkpoints pick **Official Turbo** (`Euler`, `10` steps, `CFG 1`), or **Turbo + NAG** to keep the negative prompt at CFG 1 through Normalized Attention Guidance, which needs no unconditional model pass. A warning is logged when a Turbo checkpoint runs with Base settings or the reverse.

## Launch Options

Pass options in `COMMANDLINE_ARGS` (`webui-user.bat`, `webui.settings.sh`) or to `python launch.py`. `python launch.py --help` lists all of them, including the ones bundled and installed extensions add (such as `--agent-scheduler-sqlite-file`, `--controlnet-loglevel` and `--lora-dir`).

**Server and access**

| Option | Purpose |
| --- | --- |
| `--port N` | Port to listen on. Default: the first free port from 7860 (7861 with `--nowebui`) |
| `--listen` | Listen on all interfaces (`0.0.0.0`) so other machines can connect |
| `--server-name HOST` | Listen on a specific address or host name |
| `--subpath /sd` | Serve the app under a path prefix, for a reverse proxy (with or without prefix stripping) |
| `--forwarded-allow-ips 127.0.0.1` | Trust `X-Forwarded-For` / `X-Forwarded-Proto` from these reverse proxies (addresses or networks, comma-separated), so failed logins are counted per real client. Without it the headers are ignored and every client behind a proxy shares one counter |
| `--tls-keyfile KEY --tls-certfile CERT` | Serve HTTPS (session cookies become `Secure`) |
| `--auth user:pass[,user2:pass2]` | Require a login for the web interface, `/api/v2`, files and the WebSocket (formerly `--gradio-auth`) |
| `--auth-file PATH` | Read logins in the `--auth` format from a file, one or more per line (formerly `--gradio-auth-path`) |
| `--api` | Also serve the A1111-compatible `/sdapi/v1` API |
| `--api-auth user:pass[,…]` | HTTP Basic login for `/sdapi/v1` (and for extension routes from API clients) |
| `--api-no-auth-on-localhost` | With `--auth` and no `--api-auth`: let API clients on this machine use `/sdapi/v1` and extension API routes without a login (the old WebUI behaviour) |
| `--nowebui` | API only (`/api/v2`, `/sdapi/v1`, no web interface), default port 7861 |
| `--cors-allow-origins URLS`, `--cors-allow-origins-regex RE` | Allow browser front-ends on other origins, and extra host names for a loopback-only server |
| `--enable-insecure-extension-access` | Allow installing and updating extensions from the UI while listening on the network |
| `--hide-ui-dir-config` | Hide folder settings and "open folder" / "show in folder" actions |
| `--freeze-settings`, `--freeze-settings-in-sections`, `--freeze-specific-settings` | Make settings read-only |
| `--autolaunch` | Open the browser at start |
| `--ngrok TOKEN` | Publish through an ngrok tunnel |
| `--timeout-keep-alive N` | HTTP keep-alive timeout (seconds) |
| `--api-server-stop` | Enable `/sdapi/v1/server-kill|restart|stop` |

**Paths and installation**

| Option | Purpose |
| --- | --- |
| `--data-dir PATH` | Base folder for user data: settings, styles, outputs, extensions, `models/` and `remi-data/` (default: the installation folder) |
| `--model-ref PATH` | Replace the local `models` directory with a central model directory |
| `--ckpt-dirs PATH` (repeatable) | Extra checkpoint folders |
| `--forge-ref-a1111-home PATH`, `--forge-ref-comfy-home PATH`, `--forge-ref-comfy-yaml PATH` | Reuse model folders of an AUTOMATIC1111 or ComfyUI installation, or a ComfyUI `extra_model_paths.yaml` |
| `--uv`, `--uv-symlink`, `--uv-local-cache` | Install packages with `uv` |
| `--skip-install`, `--skip-prepare-environment` | Skip package installation (and the web interface build) |
| `--sage` / `--flash` | Install optional attention packages; availability does not guarantee selection |
| `--disable-extra-extensions`, `--disable-all-extensions` | Start without user extensions, or without any |

**Models, presets and workers**

| Option | Purpose |
| --- | --- |
| `--disable-preset sd xl …` | Remove these UI presets from the model bar and the preset settings |
| `--sort-preset` | List the UI presets alphabetically |
| `--force-w4a8` | Load `convrot_w4a4` checkpoints as `w4a8` (better quality) |
| `--use-ck-attention`, `--enable-triton-backend` | Comfy-Kitchen attention (overrides other attention packages); Triton kernels in `comfy-kitchen` |
| `--pynvml` | Read the actual free VRAM through the NVIDIA Management Library |
| `--worker-process` | Run PyTorch inference in an isolated supervised child process (`SubprocessWorker`) for OOM/crash recovery and zero UI/WS GIL contention (default: in-process) |
| `--worker-gpus 0,1` | Spawn one isolated inference worker per listed GPU index in `WorkerPool` |
| `--worker-mlx` | Enable the optional Apple Silicon MLX / `mflux` sidecar worker alongside the standard UI, queue, and gallery |

The three `--worker-*` options are a preview: they list the workers in the Workers monitor (health, restart), but the queue does not dispatch to them and still runs every job in-process. `REMI_WORKER_PROCESS=1`, `REMI_WORKER_GPUS=0,1` and `REMI_WORKER_MLX=1` set the same options when the flags are absent (only `1` turns the switches on).

On Apple Silicon, `REMI_MPS_DEQUANT_CACHE_MB` sets how many MiB of dequantized weights stay cached between steps (`0` turns the cache off). Without it the cache takes at most a quarter of the unified memory left above what inference needs, up to 2 GiB. A value you set is used as given, without that cap; one that is not a whole number is ignored.

**Changed options.** The Gradio-only options are gone: replace `--gradio-auth` / `--gradio-auth-path` with `--auth` / `--auth-file`, and drop `--gradio-debug`, `--gradio-allowed-path` and `--no-gradio-queue`. The WebUI refuses to start with an unknown option, so an old login option can never be silently ignored. `--share` is accepted but only logs a warning (use `--listen`, a reverse proxy or `--ngrok`); `--use-textbox-seed` has no effect; `--theme light|dark|system` is the theme of browsers that have not picked one in the Appearance menu (`?__theme=light` in the URL picks one, as before); `--agent-scheduler-sqlite-file` tells the queue importer where an old Agent Scheduler database is. Do not install every attention backend blindly; native PyTorch attention is often the most compatible choice, see the upstream [extra installation guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations).

## Where Data Is Stored

Paths are relative to the data folder (`--data-dir`, by default the installation folder).

| What | Where |
| --- | --- |
| Settings | `config.json` (`--ui-settings-file`) |
| Prompt styles | `styles.csv` |
| Generated images | `output/txt2img-images`, `output/img2img-images`, `output/extras-images`, grids and saved images next to them (Settings → Paths for saving) |
| Models | `models/` (or `--model-ref`) |
| User extensions | `extensions/` |
| Built-in extensions' data | `extension-data/`: Spectrum presets, LoRA stage and artist mixer templates, the prompt blacklist, post-processing presets, the prompt workshop's settings |
| **Remi data** | `remi-data/`: `jobs.db` (queue, history, multi-stage chains & templates), `files.db` (file index, gallery stars, project boards & lineage), `recipes.db` (saved cross-model Recipes), `uploads/` (images you upload, pruned after `remi_upload_retention_days`), `outputs/` (results the save settings did not write to the output folders, pruned the same way), `thumbs/` (thumbnail cache), `networks/` (extracted previews and placeholders), `prompt-tools.db` (prompt history, favorites, completion usage), `secret.key` and `revoked-sessions.json` (login sessions) |
| Your own CSS | `user.css` (applied to the web interface when present) |
| Completion sound | `notification.mp3` in the installation folder |
| Per browser | form values (drafts), appearance, language, pinned layout sizes: the browser's local storage. "Reset all parameters" in the Generate menu returns a form to its defaults |

Keep `remi-data/` when you move an installation or a Docker volume; deleting it clears the queue, the history, the gallery stars and the prompt tools data, never your images in `output/`.

## Keyboard Shortcuts

The command palette (Ctrl+K, ⌘K on a Mac) lists every command with its shortcut; `?` outside a text field shows the shortcuts of the whole app (the image editor shows its own list with `?`).

| Where | Keys |
| --- | --- |
| Everywhere | Ctrl+K / ⌘K command palette · ? keyboard shortcuts · Alt+N networks drawer |
| Generate | Ctrl/⌘+Enter generate (while running: interrupt and generate again, or only interrupt with the setting "Revert [Ctrl + Enter] to only interrupt the generation") · Ctrl/⌘+Shift+Enter add to queue · Alt+Enter skip · Esc interrupt (outside text fields and dialogs) · results: ← / → select, Enter open, Delete remove |
| Prompt editor | Ctrl/⌘+↑ / ↓ change the weight of the selection · Alt+← / → move the tag · Tab or Enter accept a completion · Ctrl/⌘+Z undo · Ctrl+Y / ⌘⇧Z redo |
| Image editor | B brush · E eraser · G fill · M rectangle · L lasso · I eyedropper · H hand · [ / ] brush size · Shift (hold) erase · Ctrl/⌘+Z / Y undo / redo · Ctrl/⌘+X clear layer · Ctrl/⌘+I invert mask · R or 0 fit · 1 actual size · + / − zoom · F maximize · Space+drag pan · ? help |
| Lightbox | ← / → previous / next · Home / End · + / − / 0 / 1 zoom · S download · T tile preview · Esc close |
| Gallery | arrows move · Enter open · Space select (Shift extends) · F star · Ctrl/⌘+A select all · Delete remove · Esc clear selection |
| Networks | / search · arrows move · Enter apply · I metadata · E edit · Shift+F10 item menu |
| Queue | Space pick up a job, ↑ / ↓ move it, Space drop, Esc cancel |
| Extras | Ctrl/⌘+Enter process · Esc interrupt |
| Settings | / search · Ctrl/⌘+S apply |

## Migrating from the Gradio UI

Remi's interface is new; the generation engine, models, settings file, styles, output folders and infotext format are the same, so your files and images carry over unchanged.

| In the old UI | In Remi |
| --- | --- |
| txt2img and img2img tabs | **Generate** page, txt2img / img2img switch in the header |
| img2img sub-tabs (img2img, Sketch, Inpaint, Inpaint sketch, Inpaint upload, Batch) | mode picker in the **Source** card; the image editor is on the stage |
| Checkpoint, VAE / text encoder and preset dropdowns at the top | **model bar** in the header |
| Quicksettings | ⚡ quick settings in the header; pin settings from their row in Settings |
| Extra Networks tabs under the prompt | **Networks** drawer (Alt+N) and **Networks** page |
| "Generate forever" context menu, Repeat Generate | the menu next to Generate: generate forever, generate N times |
| Agent Scheduler tab and "Enqueue" | **Queue** page and **Add to queue** next to Generate (Ctrl+Shift+Enter); its queue settings are kept, the layout ones are listed in [its README](../../extensions-builtin/agent-scheduler/README.md) |
| Prompt Queue tab | **Queue** page |
| Prompt All-in-One | prompt tools bar under every prompt (history and favorites are imported automatically on first use); what each old feature became is in [its README](../../extensions-builtin/sd-webui-prompt-all-in-one-neo/README.md) |
| TagComplete | built-in autocompletion (its settings are kept, except those listed with the reason in [its README](../../extensions-builtin/sd-webui-tagcomplete-neo/README.md)) |
| Bracket checker, mobile layout | built in |
| PNG Info tab | **PNG Info** page |
| Settings tab, "Defaults" and "Actions" | **Settings** page; server actions under "Server and models". `ui-config.json` is no longer read: forms start from defaults, the UI preset and your browser's saved values |
| Extensions tab | **Extensions** page |
| JoyCaption, Prompt Workshop and JSON Builder tabs | pages of the same name |
| Checkpoint Merger tab | **Checkpoint Merger** page (the same controls; merges and saves run behind generation with progress and Cancel) |
| `/file=` links | files are served by id (`/api/v2/files/{id}`) |
| `/agent-scheduler/v1/*`, `/prompt-queue/*` APIs | `POST /api/v2/jobs` and `/api/v2/queue/*` ([API guide](api.md)); the old routes answer `410 Gone` |
| `/tacapi/v1/*`, `/physton_prompt/*` APIs | `/api/v2/prompt-tools` (the old routes answer `410 Gone`) |

**Bringing your queue over.** Open **Queue → ⋯ → Import from Agent Scheduler or Prompt Queue…**. The importer reads the old data without changing it and can run again safely (tasks imported before are skipped). Agent Scheduler pending tasks are queued (the queue is paused first so you can review them), finished tasks go to the history with their images, bookmarks become stars. Script arguments stored by position are named through the current panels; tasks whose arguments cannot all be named are kept as *not replayable* and listed in the report: open one in its form to check and queue it. The Agent Scheduler database is found at `extension-data/agent-scheduler/task_scheduler.sqlite3` or wherever `--agent-scheduler-sqlite-file` points; Prompt Queue's `queue.json` under `extension-data/sd-webui-prompt-queue/`.

**Third-party extensions.** Extensions that only process images (no UI) usually keep working. Extensions with a Gradio UI (`ui()`, `on_ui_tabs`, JavaScript) must be ported: a script that still declares `ui()` is disabled and a file that imports Gradio is skipped, with the reason in the console and in `GET /api/v2/app` (`not_ported`); the rest of the WebUI keeps working. The [extension guide](extension-guide.md) explains the port. API clients written for `/sdapi/v1` keep working with `--api`.

## Built-in Extensions

Remi ships a curated set of integrated extensions, all ported to the new interface:

- ADetailer Neo, NegPiP, Detail Daemon, Prompt Anchor, Prompt Blacklist, Post Processing Suite, Lock img2img Steps
- JoyCaption, Prompt Workshop and JSON Builder (pages), prompt tools (successor of Prompt All-in-One) and tag completion, localization packs
- Forge ControlNet with preprocessors, ControlLLLite and IP-Adapter, MultiDiffusion, Image Stitch, Soft Inpainting, Never OOM, Sparse / Radial attention, compile and PiD
- Remi's Anima guidance, Regional, FreeFuse, Artist Mixer, DAVE, Cross-Attention Boost, LoRA layer weights and stage scheduling, prompt rescale, resolutions, Dynamic Shift, Hires Guard, Spectrum and presets

Agent Scheduler, Prompt Queue and Repeat Generate became the core Queue. Built-in status means these extensions are versioned with Remi and covered by its tests; it does not imply that every external extension is compatible.

## Security and Remote Access

By default the server listens on `127.0.0.1` only.

- **Logins.** `--auth` / `--auth-file` protect everything: the interface, `/api/v2`, files, the WebSocket, `/internal` and the API docs. The browser shows a sign-in form (part of the interface, translated like it); sessions last 30 days and survive restarts, and logging out revokes them. Scripts can send HTTP Basic credentials instead. Ten failed logins from one client within a minute lock it out for a minute. `/sdapi/v1` uses `--api-auth` when set; without it, the `--auth` login (HTTP Basic for scripts), also on a loopback-only server, since a reverse proxy or tunnel on the same machine connects from loopback too. Local tools that cannot send credentials can be let in with `--api-no-auth-on-localhost` (requests from this machine that came through no proxy).
- **Exposing the server.** Use `--listen` (or `--server-name`) together with `--auth`, ideally behind a TLS reverse proxy or with `--tls-keyfile` / `--tls-certfile`. `--share` is not supported. Installing and updating extensions from the UI is disabled while listening on the network unless `--enable-insecure-extension-access` is given; "Show in folder" in the Gallery only works for a browser on the same machine, and `--hide-ui-dir-config` removes every action that opens or reveals folders on the server.
- **Cross-site requests are refused.** State-changing requests and WebSocket connections from pages on another site are rejected, so a malicious page cannot drive your WebUI. Scripts and API tools send no `Origin` header and are unaffected; browser front-ends on another origin must be allowed with `--cors-allow-origins`.
- **DNS rebinding is refused.** A loopback-only server answers only to `localhost` / loopback host names (or hosts allowed with `--cors-allow-origins`). A local reverse proxy must pass one of those as `Host`, or allow its public origin.
- **Files** are served by opaque id only, uploads are type- and size-checked (`remi_upload_max_mb`), every other request body is capped at twice that size, and extension HTML, infotext and metadata are rendered as text or sanitized.
- **Extension routes** (added by extensions to the server) get the same cross-site and host checks; with `--auth` they also require the login or `--api-auth` credentials (local API clients are exempt only with `--api-no-auth-on-localhost`).
- **Startup warnings** name risky combinations: reachable without a login, `--cors-allow-origins=*`, insecure extension access, logins without TLS. See [docs/remi-webui/security.md](security.md) for the threat model and a checklist for running exposed.
- **Outbound requests** happen only for what you use or configure: the extension index on the Extensions page, a translation provider you set up (it also receives the descriptions you send with "Write a prompt" when it is an OpenAI-compatible one), and model downloads of features that need them on first use (CLIP Modulation, JoyCaption, ControlNet preprocessors, ADetailer).

## Troubleshooting

**"The web interface is not built"** in the browser. Node.js was not found when the launcher ran, the build failed, or it was skipped (`--skip-install`, `--skip-prepare-environment`, `REMI_SKIP_WEBUI_BUILD`); the console says which. An older Node.js gets the warning "Node.js vX is too old to build the web interface" and the build usually fails. Install Node.js 20.19+ / 22.12+ and restart, or build `webui/dist` elsewhere and copy it ([The web interface build](#the-web-interface-build)).

**The build fails** (`pnpm install` or `pnpm build` exits with an error). Run `pnpm install --frozen-lockfile && pnpm build` in `webui/` to see the full message. Behind a proxy, configure it for npm/pnpm (`HTTPS_PROXY`). The server still starts; the API works meanwhile.

**The page looks outdated after an update.** Reload the page; the app also reloads itself once when it notices a new build. If it persists, the build did not run (see above).

**403 "Unknown host refused".** You opened a loopback-only server through another host name (a LAN name, a reverse proxy, a tunnel). Use `--listen` / `--server-name`, or allow the name with `--cors-allow-origins=https://your.host`, or make the proxy send `Host: localhost`.

**403 "Cross-site request refused".** A page on another origin (a separate front-end, a userscript site) sent a write request. Allow that origin with `--cors-allow-origins`. Behind a reverse proxy, forward the public host and port (`X-Forwarded-Host`, `X-Forwarded-Port`, `X-Forwarded-Proto`).

**No live preview or progress; the connection dot in the header is red.** The WebSocket (`/api/v2/ws`) is blocked. A reverse proxy must forward WebSocket upgrades (nginx: `proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`). Also check that live previews are enabled in Settings.

**Under a reverse proxy path the page is blank.** Start with `--subpath /your/prefix` matching the proxy location.

**The port is taken.** Without `--port` the server tries 7860, 7861, … up to 7879; with `--port` it fails if that port is busy. Look for `Running on local URL:` in the console.

**An extension's panel or page is missing.** Check the console: a script that declares Gradio `ui()` instead of `params()`, or imports Gradio, is disabled with a message (also listed in `GET /api/v2/app` under `not_ported`). Disabled extensions are managed on the Extensions page; changes apply after "Reload UI".

**"Extension access is disabled".** The server listens on the network; start with `--enable-insecure-extension-access` if you trust everyone who can reach it.

**A form keeps odd values.** Form values are saved per browser. Use "Reset all parameters" in the Generate menu, or the reset dot next to a single value.

**Login does not stick.** The session cookie is `Secure` with `--tls-*` or behind HTTPS; open the HTTPS address. Cookies are scoped to `--subpath`.

**Out of memory or slow generation.** These are engine matters as in Forge Neo: try the Never OOM panel, a low-bit dtype in "Diffusion in Low Bits", smaller batches, and native PyTorch attention.

## Compatibility Notes

- Training features, legacy model families and older WebUI components removed by Forge Neo are not restored by Remi.
- Extensions written for the Gradio UI need a port; processing-only extensions and `/sdapi/v1` clients usually work unchanged.
- CUDA-only packages such as FlashAttention are not available on every platform.
- Experimental guidance methods can alter image character as well as quality; preserve a baseline seed when comparing them.
- Torch compile (whole-model presets and **Anima per-block**) runs the affected steps uncompiled while NAG, NegPiP, Regional, FreeFuse, Artist Mixer, DAVE or Cross-Attention Boost (during their window steps), or Sparse Attention are active, so these features are never silently ignored by a cached graph; expect no compile speed-up for those runs.
- **Anima FlowMatch** is only meaningful for flow-matching models; on other models (e.g. left selected after switching from an Anima Flow sampler) it falls back to the Simple schedule.
- On flow-matching models (Anima, Flux): Karras, Exponential, Polyexponential, KL Optimal and Align Your Steps are made for SD's noise scale and wash the image out, so the job finishes with a warning (use Simple, Beta, SGM Uniform or Normal); DDIM, PLMS and Kohaku LoNyu Yog are refused with a message. The CFG++ samplers keep the negative prompt at CFG 1, where they are meant to run.
- Real checkpoint output remains the final compatibility test. Startup or unit tests do not prove generation quality for every checkpoint or quantization.
