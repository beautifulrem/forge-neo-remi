# Remi WebUI (Docker)

> [!Warning]
> Requires an **NVIDIA** GPU and the NVIDIA Container Toolkit (`docker/Dockerfile`:
> the PyTorch CUDA 13.0 build the launcher installs on Linux, driver `580+`), or an
> **AMD** GPU with ROCm (`docker/Dockerfile.rocm`: the ROCm 7.2 PyTorch wheels).

The image is built from this checkout, never from a clone: what it runs is the
code of the folder you build it in, without Gradio (the WebUI's requirements do
not include it).

## Build

From the repository root (the build context must be the whole checkout):

```bash
docker build -f docker/Dockerfile -t remi-webui .            # NVIDIA
docker build -f docker/Dockerfile.rocm -t remi-webui-rocm .  # AMD (ROCm)
```

- Stage `webui` builds the web interface (`webui/dist`) with Node.js and
  `pnpm install --frozen-lockfile && pnpm build`.
- The server stage installs Python 3.13 (uv), PyTorch, `requirements.txt`,
  copies the checkout and the built interface, and runs the built-in extensions'
  installers once (`launch.py --exit`). Node.js is not in the final image;
  `REMI_SKIP_WEBUI_BUILD=1` keeps the launcher from rebuilding the interface.
- `.dockerignore` (repository root) leaves models, outputs, virtual
  environments, `webui/node_modules` and `webui/dist` out of the build context.

Build arguments: `CUDA_VERSION` (base image, default `13.0.0`), `NODE_VERSION`
(default `24`), `PNPM_VERSION` (default `12`), `TORCH_INDEX_URL` and
`TORCH_PACKAGES` (default: the launcher's Linux PyTorch build). The ROCm recipe
uses `ubuntu:22.04` as its base and takes `TORCH_INDEX` (default `rocm7.2`, an
index of <https://download.pytorch.org/whl/>) and `TORCH_PACKAGES` (default
`torch torchvision`) instead of the CUDA arguments.

## Run

```bash
# NVIDIA
docker run --gpus all -p 7860:7860 -v remi-data:/home/remi/data remi-webui
# AMD (ROCm): pass the kernel driver and the GPU devices
docker run --device /dev/kfd --device /dev/dri --group-add video \
    -p 7860:7860 -v remi-data:/home/remi/data remi-webui-rocm
```

Open `http://localhost:7860/`. Extra launch options go after the image name or
in `COMMANDLINE_ARGS` (for example `-e COMMANDLINE_ARGS="--auth user:password"`);
the container always adds `--listen` and `--data-dir /home/remi/data`. Read
[Security and Remote Access](../docs/remi-webui/usage.md#security-and-remote-access) before
exposing the port beyond your machine.

## Data

Everything the WebUI keeps is in the data folder, `/home/remi/data` (`--data-dir`):

| Path in the volume | Purpose |
| --- | --- |
| `models/` | Checkpoints (`Stable-diffusion`), `text_encoder`, `VAE`, `Lora`, `ControlNet`, `ESRGAN`, `embeddings` |
| `output/` | Generated images |
| `extensions/` | User-installed extensions |
| `config.json`, `styles.csv`, `user.css` | Settings, prompt styles, custom CSS |
| `remi-data/` | The queue, history, gallery stars and prompt tools data |

Bind subfolders separately when they live elsewhere, for example
`-v /mnt/models:/home/remi/data/models`. The container runs as **UID 99** /
**GID 100** (`nobody:users`, Unraid's default share permissions): bound host
folders must be writable by it.

## Image Details

| | |
| --- | --- |
| Base | `nvidia/cuda:13.0.0-runtime-ubuntu22.04` (ROCm: `ubuntu:22.04`) |
| Python | `3.13` via **uv** |
| PyTorch | the launcher's Linux build (`torch==2.14.0+cu130`); ROCm: the `rocm7.2` wheels |
| User | `remi` (UID 99 / GID 100) |
| Port | 7860 |
