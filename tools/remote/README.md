# Remi Remote Validation Harness (`tools/remote/`)

Non-runtime ephemeral Modal (`1.6.4`) validation runner for cross-platform Linux CPU and CUDA verification.

## Guardrails

1. **Zero Runtime Coupling**: `remi/` and `backend/` never import `modal`. All Modal SDK code lives strictly in `tools/remote/`.
2. **Ephemeral Only (`modal run`)**: Never uses `modal deploy` or persistent services. Every function sets `scaledown_window=30`, `timeout=600`, `retries=0`, `region=None`, and App tag `{"project": "remi-remote-validation"}`.
3. **Zero-Rebuild Filtered Snapshot**: `Image.add_local_dir(..., copy=False, ignore=FilePatternMatcher(...))` excludes `.git`, `.venv`, `venv`, `webui/node_modules`, `webui/dist`, `.env*`, `models/`, `outputs/`, `remi-data/`, `logs/`, `*.safetensors`, `*.ckpt`, `*.gguf`, and `.agents/`.
4. **Isolated HF Cache Volume**: Uses `modal.Volume.from_name("remi-hf-cache", create_if_missing=True, version=2)` mounted at `/cache/hf` (`HF_HUB_CACHE=/cache/hf`). Mounted writable only during `--tier prefetch` (selective `hf_hub_download`, never `snapshot_download`) and mounted `read_only=True` during CUDA runs. Never touches pre-existing user Volumes.
5. **Spend Cap (`<= $30 USD`)**: Local wall-clock estimator enforces a `$20` soft stop and `$30` hard cap, paired with `modal.billing.workspace_billing_report(...)` (falling back to `modal billing report` CLI on Starter workspaces).

## Usage

### Local Dry-Run (Free)

```bash
.venv/bin/python tools/remote/modal_validate.py --help
.venv/bin/python tools/remote/modal_validate.py --tier cpu --dry-run
.venv/bin/python tools/remote/modal_validate.py --tier prefetch --dry-run
.venv/bin/python tools/remote/modal_validate.py --tier cuda --subrun all --gpu L4 --dry-run
```

### Linux CPU

```bash
modal run tools/remote/modal_validate.py --tier cpu --report-out /tmp/modal_cpu.json
```

Runs Gradio-free check, Linux CPU pytest subset, `/sdapi/v1` contract tests, `tools/upstream_forks.py budget`, `tools/check_line_endings.py`, and `tools/perf/bench_inference.py --synthetic --device cpu`.

### Selective Weight Prefetch

```bash
modal run tools/remote/modal_validate.py --tier prefetch
```

Selectively downloads one Anima checkpoint from `circlestone-labs/Anima` plus `qwen_3_06b_base.safetensors` and `qwen_image_vae.safetensors` into `remi-hf-cache` and commits the Volume.

### Linux CUDA (`L4` Primary, `$0.000222/s`)

```bash
modal run tools/remote/modal_validate.py --tier cuda --subrun anima --gpu L4
modal run tools/remote/modal_validate.py --tier cuda --subrun extensions --gpu L4
modal run tools/remote/modal_validate.py --tier cuda --subrun families --gpu L4
modal run tools/remote/modal_validate.py --tier cuda --subrun bench --gpu L4
```

### Billing & Post-Run Cleanup Verification

```bash
.venv/bin/python tools/remote/modal_validate.py --tier billing
modal app list
modal volume ls remi-hf-cache
```
