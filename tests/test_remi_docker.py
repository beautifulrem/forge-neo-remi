"""docker/: the image recipe builds this checkout, Gradio-free.

The image cannot be built in the test suite (it needs a Docker daemon, the
network and gigabytes of CUDA wheels), so these tests read the recipe: it
copies the build context instead of cloning another repository, installs
``requirements.txt`` (which has no Gradio) and the launcher's PyTorch build,
builds the SPA with the launcher's pnpm, keeps all user data in ``--data-dir``
and leaves Gradio-era files (``ui-config.json``) out.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
DOCKERFILE = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
ROCM_DOCKERFILE = (DOCKER / "Dockerfile.rocm").read_text(encoding="utf-8")
ENTRYPOINT = (DOCKER / "entrypoint.sh").read_text(encoding="utf-8")
README = (DOCKER / "README.md").read_text(encoding="utf-8")
IGNORE = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def instructions(recipe: str = DOCKERFILE) -> list[tuple[str, str]]:
    """``(INSTRUCTION, arguments)`` of the Dockerfile, continuation lines joined."""

    joined = re.sub(r"\\\n", " ", recipe)
    found = []
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keyword, _, rest = line.partition(" ")
            found.append((keyword.upper(), rest.strip()))
    return found


def arg_default(name: str, recipe: str = DOCKERFILE) -> str:
    for keyword, rest in instructions(recipe):
        if keyword == "ARG" and rest.startswith(f"{name}="):
            return rest.split("=", 1)[1].strip('"')
    raise AssertionError(f"no ARG {name}")


def test_the_image_is_built_from_this_checkout():
    assert "git clone" not in DOCKERFILE and "github.com" not in DOCKERFILE
    copies = [rest for keyword, rest in instructions() if keyword == "COPY"]
    # the whole build context, and the SPA from the image's own Node.js stage
    assert "--chown=99:100 . ./" in copies
    assert "--from=webui --chown=99:100 /src/webui/dist ./webui/dist" in copies
    # every path copied from the context exists in the checkout
    for rest in copies:
        if "--from=" in rest:
            continue
        sources = [part for part in shlex.split(rest) if not part.startswith("--")][:-1]
        for source in sources:
            assert (ROOT / source).exists(), source


def test_the_python_side_is_the_webuis_requirements():
    runs = " ".join(rest for keyword, rest in instructions() if keyword == "RUN")
    assert "uv pip install -r requirements.txt" in runs
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "gradio" not in requirements
    assert not any("gradio" in rest.lower() for _keyword, rest in instructions())
    # the built-in extensions' installers run through the launcher, which then exits
    assert "python launch.py --exit --skip-torch-cuda-test --data-dir /tmp/remi-build" in runs


def test_pytorch_and_pnpm_follow_the_launcher():
    from modules import launch_utils

    command = launch_utils.default_torch_command(platform="linux")
    packages = arg_default("TORCH_PACKAGES")
    assert f"pip install {packages} --extra-index-url {arg_default('TORCH_INDEX_URL')}" == command
    assert f"pnpm@{arg_default('PNPM_VERSION')}" == launch_utils.PNPM_FALLBACK
    assert launch_utils.node_version_problem(f"v{arg_default('NODE_VERSION')}.0.0") is None
    # the image ships the build: the launcher must not try to rebuild it without Node.js
    assert "REMI_SKIP_WEBUI_BUILD=1" in DOCKERFILE
    assert "pnpm install --frozen-lockfile" in DOCKERFILE and "pnpm run build" in DOCKERFILE
    # the base image's CUDA is the wheels' CUDA
    major, minor = arg_default("CUDA_VERSION").split(".")[:2]
    assert arg_default("TORCH_INDEX_URL").endswith(f"/cu{major}{minor}")


def test_all_user_data_is_in_the_data_folder():
    assert "ui-config.json" not in ENTRYPOINT and "ln -s" not in ENTRYPOINT
    assert '--data-dir "$REMI_DATA_DIR"' in ENTRYPOINT and "--listen" in ENTRYPOINT
    assert "REMI_DATA_DIR=/home/remi/data" in DOCKERFILE and 'VOLUME ["/home/remi/data"]' in DOCKERFILE
    # the healthcheck uses the one route that answers without a login
    from remi.server.auth import PUBLIC_PATHS

    assert "/internal/ping" in PUBLIC_PATHS and "localhost:7860/internal/ping" in DOCKERFILE


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_entrypoint_parses():
    assert subprocess.run(["bash", "-n", str(DOCKER / "entrypoint.sh")], capture_output=True).returncode == 0


def run_entrypoint(tmp_path: Path, commandline_args: str | None, *args: str) -> tuple[subprocess.CompletedProcess[str], list[str] | None]:
    """docker/entrypoint.sh with a ``python`` on PATH that records its arguments: the result
    and what the launcher would have received (None when it was not started)."""

    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    record = tmp_path / "launcher-args"
    record.unlink(missing_ok=True)
    stub = stubs / "python"
    stub.write_text(f'#!/bin/sh\nfor arg in "$@"; do printf \'%s\\0\' "$arg"; done > "{record}"\n', encoding="utf-8")
    stub.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "COMMANDLINE_ARGS"}
    env["PATH"] = f"{stubs}{os.pathsep}{env.get('PATH', '')}"
    env["REMI_DATA_DIR"] = str(tmp_path / "data")
    if commandline_args is not None:
        env["COMMANDLINE_ARGS"] = commandline_args
    result = subprocess.run(["bash", str(DOCKER / "entrypoint.sh"), *args], env=env, capture_output=True, text=True, timeout=60)
    received = record.read_bytes().decode().split("\0")[:-1] if record.exists() else None
    return result, received


@pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="needs bash")
def test_the_entrypoint_splits_commandline_args_like_a_shell(tmp_path):
    """COMMANDLINE_ARGS was split on blanks (``read -ra``), so a quoted value with a space
    (a folder name, a prompt) reached the launcher as several arguments."""

    base = ["/home/remi/app/launch.py", "--listen", "--data-dir", str(tmp_path / "data")]
    result, received = run_entrypoint(tmp_path, "--x \"a b\" --y 'c d' e\\ f", "--last")
    assert result.returncode == 0, result.stderr
    assert received == [*base, "--x", "a b", "--y", "c d", "e f", "--last"]
    assert (tmp_path / "data" / "models" / "Lora").is_dir() and (tmp_path / "data" / "extensions").is_dir()
    for empty in (None, "", "   "):
        result, received = run_entrypoint(tmp_path, empty)
        assert result.returncode == 0 and received == base, (empty, result.stderr)
    # an unbalanced quote stops the container instead of starting without some of the options
    result, received = run_entrypoint(tmp_path, '--auth-file "/run/remi/logins.txt')
    assert result.returncode != 0 and received is None
    assert "COMMANDLINE_ARGS" in result.stderr and "logins.txt" not in result.stderr


@pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="needs bash")
def test_the_entrypoint_prepares_the_data_folder_the_launcher_uses(tmp_path):
    """A --data-dir among the launch options wins (argparse takes the last one), so the model and
    output folders are made there, not under $REMI_DATA_DIR."""

    for commandline_args, args, folder in ((f"--data-dir {tmp_path / 'other'}", (), "other"), (None, (f"--data-dir={tmp_path / 'third'}",), "third"), (f"--data-dir {tmp_path / 'other'}", ("--data-dir", str(tmp_path / "last")), "last")):
        result, received = run_entrypoint(tmp_path, commandline_args, *args)
        assert result.returncode == 0, result.stderr
        assert received is not None and received[:4] == ["/home/remi/app/launch.py", "--listen", "--data-dir", str(tmp_path / "data")]
        assert (tmp_path / folder / "models" / "Stable-diffusion").is_dir() and (tmp_path / folder / "output").is_dir(), folder
        assert not (tmp_path / "data").exists(), folder


def test_the_readme_names_the_data_folder_variable():
    """The entrypoint passes ``--data-dir "$REMI_DATA_DIR"`` and prepares that folder (or the last
    --data-dir of the launch options): the README says which variable names it and its default."""

    data = section(README, "## Data")
    assert "`REMI_DATA_DIR`" in data and "`/home/remi/data`" in data and "-e REMI_DATA_DIR=" in data
    assert "REMI_DATA_DIR=/home/remi/data" in DOCKERFILE and '--data-dir "$REMI_DATA_DIR"' in ENTRYPOINT


def test_the_entrypoint_runs_from_a_windows_checkout():
    """Git for Windows checks text files out with CRLF (core.autocrlf), and the repository has no
    .gitattributes: "#!/bin/bash\\r" is no interpreter, so the image strips the CRs itself."""
    for dockerfile in (DOCKERFILE, ROCM_DOCKERFILE):
        assert "RUN sed -i 's/\\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh" in dockerfile


def test_the_build_context_leaves_out_data_and_build_output():
    for pattern in ("/models", "/extensions", "/output", "/remi-data", "/config.json", "/venv", "/.venv", "/webui/node_modules", "/webui/dist", ".git", "**/*.safetensors"):
        assert pattern in IGNORE, pattern


def docker_builds(text: str) -> list[str]:
    """The ``docker build`` commands in ``text``, continuation lines joined, without their comment."""

    return [line.split(" #", 1)[0].strip() for line in re.sub(r"\\\n", " ", text).splitlines() if line.lstrip().startswith("docker build")]


def test_the_readme_describes_this_image():
    assert "Forge - Neo" not in README and "git clone" not in README
    assert any(build.endswith("-f docker/Dockerfile -t remi-webui .") for build in docker_builds(README))
    assert "--data-dir" in README


def test_the_docs_name_the_variables_the_entrypoint_reads():
    """The environment variables the entrypoint reads (not the ones it sets itself) are
    documented: REMI_DATA_DIR in the usage guide too, as only the image reads it."""

    referenced = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", ENTRYPOINT))
    assigned = set(re.findall(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=", ENTRYPOINT, flags=re.M)) | {"EXTRA_ARGS"}
    read = referenced - assigned
    assert {"COMMANDLINE_ARGS", "REMI_DATA_DIR"} <= read
    for name in sorted(read):
        assert f"`{name}`" in README, name
    for guide in ("usage.md", "usage.zh-CN.md"):
        assert "`REMI_DATA_DIR`" in (ROOT / "docs" / "remi-webui" / guide).read_text(encoding="utf-8"), guide


def test_the_image_records_the_commit_it_was_built_from():
    """The build context leaves .git out, so the image cannot tell which commit it runs: the
    documented builds pass it as REMI_REVISION, which becomes the OCI revision label. The ARG
    comes after the last RUN, so a new commit does not rebuild the PyTorch and requirements layers."""

    for recipe in (DOCKERFILE, ROCM_DOCKERFILE):
        steps = instructions(recipe)
        assert arg_default("REMI_REVISION", recipe) == "unknown"
        assert ("LABEL", "org.opencontainers.image.revision=$REMI_REVISION") in steps
        last_run = max(index for index, (keyword, _rest) in enumerate(steps) if keyword == "RUN")
        assert steps.index(("ARG", "REMI_REVISION=unknown")) > last_run
    usage = (ROOT / "docs" / "remi-webui" / "usage.md").read_text(encoding="utf-8")
    usage_zh = (ROOT / "docs" / "remi-webui" / "usage.zh-CN.md").read_text(encoding="utf-8")
    for text in (README, usage, usage_zh):
        builds = docker_builds(text)
        assert builds
        for build in builds:
            assert '--build-arg REMI_REVISION="$(git rev-parse HEAD)"' in build, build


def readme_paragraph(start: str) -> str:
    """The paragraph of docker/README.md that starts with ``start``, its lines joined."""

    paragraphs = [" ".join(block.split()) for block in README.split("\n\n")]
    found = [paragraph for paragraph in paragraphs if paragraph.startswith(start)]
    assert len(found) == 1, start
    return found[0]


def test_the_readme_names_every_option_the_health_check_does_not_follow():
    """The health check asks http://localhost:7860: another port (``--port``; ``--nowebui``
    defaults to 7861), one bound address (``--server-name``) or HTTPS fails it while the
    server works."""

    from remi.server.launch import DEFAULT_API_PORT, DEFAULT_PORT

    assert f"localhost:{DEFAULT_PORT}/internal/ping" in DOCKERFILE and DEFAULT_API_PORT != DEFAULT_PORT
    paragraph = readme_paragraph("The image's health check")
    for option in ("--port", "--nowebui", "--server-name", "--tls-keyfile", "--no-healthcheck"):
        assert f"`{option}" in paragraph, option


def section(text: str, heading: str) -> str:
    """The section of a Markdown ``text`` under the heading line ``heading``, up to the next heading of its level."""

    level = heading.split(" ", 1)[0]
    start = text.index(f"\n{heading}\n")
    end = text.find(f"\n{level} ", start + 1)
    return text[start : end if end >= 0 else len(text)]


def test_the_docs_say_the_container_refuses_extension_installs():
    """The entrypoint always adds --listen, and a server reachable from other machines refuses to
    install or update extensions (security.md, "Extensions"): the Docker docs say so and name the
    two ways around it."""

    assert "--listen" in ENTRYPOINT
    usage = (ROOT / "docs" / "remi-webui" / "usage.md").read_text(encoding="utf-8")
    usage_zh = (ROOT / "docs" / "remi-webui" / "usage.zh-CN.md").read_text(encoding="utf-8")
    for text in (section(README, "## Run"), section(usage, "### Docker (NVIDIA or AMD GPU)"), section(usage_zh, "### Docker（NVIDIA 或 AMD GPU）")):
        assert "--enable-insecure-extension-access" in text and "extensions/" in text, text[:80]


def test_the_rocm_recipe_is_the_same_image_with_rocm_pytorch():
    """docker/Dockerfile.rocm (upstream 2900d594, translated): the same
    stages, data folder, launcher run and entrypoint; only the base image and
    the PyTorch wheels differ (the ROCm wheels bundle the ROCm runtime)."""

    assert "git clone" not in ROCM_DOCKERFILE and "github.com" not in ROCM_DOCKERFILE
    assert "nvidia/cuda" not in ROCM_DOCKERFILE
    rocm = instructions(ROCM_DOCKERFILE)
    assert re.fullmatch(r"rocm\d+\.\d+", arg_default("TORCH_INDEX", ROCM_DOCKERFILE))
    assert "TORCH_INDEX_URL=https://download.pytorch.org/whl/${TORCH_INDEX}" in ROCM_DOCKERFILE
    runs = " ".join(rest for keyword, rest in rocm if keyword == "RUN")
    assert 'uv pip install ${TORCH_PACKAGES} --index-url "${TORCH_INDEX_URL}"' in runs
    assert "uv pip install -r requirements.txt" in runs
    # everything but the base image and PyTorch is the NVIDIA recipe's
    same = ("COPY", "VOLUME", "EXPOSE", "HEALTHCHECK", "ENTRYPOINT", "WORKDIR", "USER", "LABEL")
    assert [i for i in rocm if i[0] in same] == [i for i in instructions() if i[0] in same]
    for keyword in ("NODE_VERSION", "PNPM_VERSION"):
        assert arg_default(keyword, ROCM_DOCKERFILE) == arg_default(keyword)
    assert "python launch.py --exit --skip-torch-cuda-test --data-dir /tmp/remi-build" in runs
    assert "REMI_SKIP_WEBUI_BUILD=1" in ROCM_DOCKERFILE and "REMI_DATA_DIR=/home/remi/data" in ROCM_DOCKERFILE
    assert any(build.endswith("-f docker/Dockerfile.rocm -t remi-webui-rocm .") for build in docker_builds(README))



def test_uv_comes_from_a_pinned_digest():
    """A tag alone can be moved; the digest pins the exact uv binary copied into the image."""

    for recipe in (DOCKERFILE, ROCM_DOCKERFILE):
        sources = [rest.split()[0] for keyword, rest in instructions(recipe) if keyword == "COPY" and "astral-sh/uv" in rest]
        assert sources, "uv is not copied from its image"
        for source in sources:
            assert re.fullmatch(r"--from=ghcr\.io/astral-sh/uv:[\w.]+@sha256:[0-9a-f]{64}", source), source


def docker_runs(text: str) -> list[str]:
    """The ``docker run`` commands in ``text``, continuation lines joined."""

    return [line.strip() for line in re.sub(r"\\\n", " ", text).splitlines() if line.lstrip().startswith("docker run")]


def wide_publishes(text: str) -> list[str]:
    """The ports the ``docker run`` commands in ``text`` publish beyond this machine:
    ``-p``/``--publish`` in every spelling (``-p7860:7860``, ``--publish=…``) without
    ``127.0.0.1:``, and ``-P``/``--publish-all`` (every exposed port, on every interface)."""

    found = []
    for run in docker_runs(text):
        found += [publish for publish in re.findall(r"(?:^|\s)(?:-p\s*|--publish(?:=|\s+))(\S+)", run) if not publish.startswith("127.0.0.1:")]
        found += re.findall(r"(?:^|\s)(-P|--publish-all)(?=\s|$)", run)
    return found


def test_the_publish_check_reads_every_spelling():
    for run in ("docker run -p 7860:7860 x", "docker run -p7860:7860 x", "docker run --publish 7860:7860 x", "docker run --publish=7860:7860 x", "docker run --gpus all \\\n    -p 7860:7860 x"):
        assert wide_publishes(run) == ["7860:7860"], run
    assert wide_publishes("docker run -P x") == ["-P"] and wide_publishes("docker run --publish-all x") == ["--publish-all"]
    assert wide_publishes("docker run -p 127.0.0.1:7860:7860 --publish=127.0.0.1:7861:7861 x") == []


def test_every_documented_run_publishes_on_this_machine_only():
    """The server inside listens on every interface without a login: the documented
    commands publish the port on 127.0.0.1, and only the remote-access section widens it."""

    usage = (ROOT / "docs" / "remi-webui" / "usage.md").read_text(encoding="utf-8")
    usage_zh = (ROOT / "docs" / "remi-webui" / "usage.zh-CN.md").read_text(encoding="utf-8")
    for text in (README, usage, usage_zh):
        assert docker_runs(text)
        assert wide_publishes(text) == []


def test_the_build_stage_node_meets_the_build_tools():
    """@babel/core (loaded by the build) needs Node ^22.18 || >=24.11; a floating major tag of 24 or
    later resolves to a release past 24.11."""

    for recipe in (DOCKERFILE, ROCM_DOCKERFILE):
        assert int(arg_default("NODE_VERSION", recipe)) >= 24
