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
# NVIDIA
docker build --build-arg REMI_REVISION="$(git rev-parse HEAD)" -f docker/Dockerfile -t remi-webui .
# AMD (ROCm)
docker build --build-arg REMI_REVISION="$(git rev-parse HEAD)" -f docker/Dockerfile.rocm -t remi-webui-rocm .
```

`REMI_REVISION` records the commit in the image's
`org.opencontainers.image.revision` label (`unknown` without it: the build
context has no `.git`); `docker inspect --format
'{{index .Config.Labels "org.opencontainers.image.revision"}}' remi-webui` reads it.

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
`TORCH_PACKAGES` (default: the launcher's Linux PyTorch build). Changing
`CUDA_VERSION` needs the matching `TORCH_INDEX_URL` (`…/whl/cuXY` for CUDA X.Y)
and `TORCH_PACKAGES` (the `+cuXY` builds): the base image's CUDA must be the
one the PyTorch wheels are built for. The ROCm recipe
uses `ubuntu:22.04` as its base and takes `TORCH_INDEX` (default `rocm7.2`, an
index of <https://download.pytorch.org/whl/>) and `TORCH_PACKAGES` (default
`torch torchvision`) instead of the CUDA arguments.

## Run

```bash
# NVIDIA
docker run --gpus all -p 127.0.0.1:7860:7860 -v remi-data:/home/remi/data remi-webui
# AMD (ROCm): pass the kernel driver and the GPU devices
docker run --device /dev/kfd --device /dev/dri --group-add video \
    -p 127.0.0.1:7860:7860 -v remi-data:/home/remi/data remi-webui-rocm
```

Open `http://localhost:7860/`. Extra launch options go after the image name or
in `COMMANDLINE_ARGS` (split like shell words: quote a value with spaces); the
container always adds `--listen` and `--data-dir "$REMI_DATA_DIR"` ([Data](#data)).
`-p 127.0.0.1:7860:7860` publishes the port on this machine only, because the
server inside listens on every interface and has no login until you add one. Before publishing it further (`-p 7860:7860`), read
[Security and Remote Access](../docs/remi-webui/usage.md#security-and-remote-access)
and set a login with `--auth-file` from a mounted file (for example
`-v ./logins.txt:/run/remi/logins.txt:ro remi-webui --auth-file /run/remi/logins.txt`;
the file must be readable by the container's user, uid 99):
a password in `--auth` or `COMMANDLINE_ARGS` shows in `docker inspect` and the
process list.

Because of `--listen`, the Extensions page refuses to install, update or remove
extensions (the server is reachable from other machines). Put an extension into
the volume's `extensions/` folder instead (clone its repository there from the
host or through `docker exec`) and restart the container, or add
`--enable-insecure-extension-access` together with a login: it lets everyone
who can sign in run code in the container.

The image's health check asks `http://localhost:7860/internal/ping`. With another
`--port`, with `--nowebui` (the API alone listens on 7861), with `--server-name`
(the server listens on that one address, not on `localhost`) or with
`--tls-keyfile` / `--tls-certfile` it fails and the container shows as unhealthy
while it works: keep the server on plain HTTP at 7860 on every interface and
terminate TLS at a reverse proxy, or run the container with `--no-healthcheck`.

## Data

Everything the WebUI keeps is in the data folder: `REMI_DATA_DIR`, by default
`/home/remi/data` (the image's volume), which the container passes as `--data-dir`.
To keep it at another path in the container, set the variable
(`-e REMI_DATA_DIR=/data -v remi-data:/data`) rather than adding `--data-dir`
to the launch options; one there still wins (the last `--data-dir` does), and the
container prepares the folders below in whichever folder the WebUI uses.

| Path in the volume | Purpose |
| --- | --- |
| `models/` | Checkpoints (`Stable-diffusion`), `text_encoder`, `VAE`, `Lora`, `ControlNet`, `ESRGAN`, `embeddings` |
| `output/` | Generated images |
| `extensions/` | User-installed extensions |
| `extension-data/` | Presets, templates and lists saved by the built-in extensions |
| `config_states/` | Backups of the extensions' versions (Extensions → Backup and restore) |
| `config.json`, `styles.csv`, `user.css`, `notification.mp3` | Settings, prompt styles, custom CSS, the completion sound |
| `remi-data/` | The queue, history, gallery stars and prompt tools data |

Bind subfolders separately when they live elsewhere, for example
`-v /mnt/models:/home/remi/data/models`. The container runs as **UID 99** /
**GID 100** (`nobody:users`, Unraid's default share permissions): bound host
folders must be writable by it.

To keep the data folder at another path in the container, set the variable
rather than passing `--data-dir` (for example
`-e REMI_DATA_DIR=/data -v /srv/remi:/data`): before it starts the WebUI, the
entrypoint creates the `models/` subfolders, `output/` and `extensions/` in
`$REMI_DATA_DIR`, which must therefore be writable by UID 99.

## Image Details

| | |
| --- | --- |
| Base | `nvidia/cuda:13.0.0-runtime-ubuntu22.04` (ROCm: `ubuntu:22.04`) |
| Python | `3.13` via **uv** |
| PyTorch | the launcher's Linux build (`torch==2.14.0+cu130`); ROCm: the `rocm7.2` wheels |
| User | `remi` (UID 99 / GID 100) |
| Port | 7860 |
