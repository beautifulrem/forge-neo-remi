"""`python launch.py --help` lists every flag, including the ones extensions add.

docs/remi-webui/usage.md / usage.zh-CN.md ("Launch Options") promise that `--help` lists all
launch options. Bundled extensions register theirs in ``preload.py``, which runs
after the launcher's own early parse, so this guards against `--help` exiting
before the preloads have run.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]

# flags registered by extensions-builtin/*/preload.py
EXTENSION_FLAGS = ("--agent-scheduler-sqlite-file", "--controlnet-loglevel", "--lora-dir", "--ad-no-huggingface")


def run_launch(*argv: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "COMMANDLINE_ARGS"}
    with tempfile.TemporaryDirectory() as data_dir:
        return subprocess.run(
            [sys.executable, "launch.py", "--data-dir", data_dir, *argv],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )


@pytest.mark.parametrize("flag", ["--help", "-h", "--hel"])
def test_help_lists_core_and_extension_flags(flag):
    result = run_launch(flag)

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: launch.py")
    for option in ("--port", "--listen", "--api", "--data-dir", *EXTENSION_FLAGS):
        assert f"  {option}" in result.stdout, option
    # the help text of an extension flag is shown, not just its name
    assert "Agent Scheduler database to import into the Queue" in " ".join(result.stdout.split())


def test_help_with_other_flags_still_exits_cleanly():
    result = run_launch("--port", "7001", "--help")

    assert result.returncode == 0, result.stderr
    assert "--controlnet-loglevel" in result.stdout


USAGE = (ROOT / "docs" / "remi-webui" / "usage.md", "**Changed options.**")
USAGE_ZH = (ROOT / "docs" / "remi-webui" / "usage.zh-CN.md", "**已变更的参数。**")


def test_flags_without_effect_say_so_and_the_usage_guide_names_them():
    from modules.cmd_args import parser

    # nothing reads these any more: --api-log logged /sdapi requests in the Gradio-era API
    # middleware, --ui-config-file named the Gradio UI's ui-config.json
    for dest in ("api_log", "ui_config_file"):
        for folder in ("modules", "modules_forge", "remi", "backend"):
            for path in (ROOT / folder).rglob("*.py"):
                assert f"cmd_opts.{dest}" not in path.read_text(encoding="utf-8", errors="replace"), (dest, path)
    no_effect = sorted(option for action in parser._actions if (action.help or "").startswith("does not do anything") for option in action.option_strings)
    assert {"--api-log", "--ui-config-file"} <= set(no_effect)
    for path, heading in (USAGE, USAGE_ZH):
        paragraph = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(heading))
        assert [flag for flag in no_effect if f"`{flag}`" not in paragraph] == [], path.name


def test_the_worker_options_say_they_are_a_preview():
    """The --worker-* options start workers the Workers monitor lists, but the queue does not
    dispatch to them yet (architecture.md): their help and the usage tables said they bring crash
    recovery and multi-GPU dispatch."""

    from modules import cmd_args

    helps = {option: action.help for action in cmd_args.parser._actions for option in action.option_strings}
    workers = [option for option in helps if option.startswith("--worker-")]
    assert sorted(workers) == ["--worker-gpus", "--worker-mlx", "--worker-process"]
    for option in workers:
        assert "(preview: the queue still runs jobs in-process)" in helps[option], option
        assert "recovery" not in helps[option] and "multi-GPU" not in helps[option], option
    docs = ROOT / "docs" / "remi-webui"
    for name, word in (("usage.md", "Preview"), ("usage.zh-CN.md", "预览")):
        rows = [line for line in (docs / name).read_text(encoding="utf-8").splitlines() if line.startswith("| `--worker-")]
        assert len(rows) == 3, name
        for row in rows:
            assert word in row and "recovery" not in row and "自愈" not in row, row
