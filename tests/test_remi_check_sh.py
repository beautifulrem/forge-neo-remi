"""``tools/check.sh``, the one command that runs every quality gate.

Every step is always registered, so a run cannot silently leave one out: the
``legacy-ui`` steps (the recorder's self-test, and ``legacy-ui:fixtures``, which
records every committed fixture again), whose Gradio 4.40 interpreter is
optional, are reported as ``skip`` with the reason when that interpreter is missing (a failure when
``--only`` names it, or with ``--strict``). The script runs from a copy in a
temporary folder, so its sibling ``remi-gradio-env`` is under the test's
control, with every other step skipped.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "tools" / "check.sh"
OTHER_STEPS = "env,ruff,pytest,contract,i18n,pnpm,perf,e2e"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="tools/check.sh is a bash script")


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A folder holding only ``tools/check.sh``, inside an otherwise empty parent."""

    root = tmp_path / "checkout"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(CHECK, root / "tools" / "check.sh")
    return root


def fake_python(path: Path, record: Path, status: int = 0) -> Path:
    """An executable that appends its arguments to ``record`` (a line per call) and exits with ``status``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "$@" >> "{record}"\nexit {status}\n', encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def run(checkout: Path, *args: str, legacy: str | None = None, python: str = sys.executable) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS"}}
    env["PYTHON"] = python
    env["REMI_CHECK_LOGS"] = str(checkout.parent / "logs")
    if legacy is not None:
        env["REMI_LEGACY_PYTHON"] = legacy
    return subprocess.run(["bash", str(checkout / "tools" / "check.sh"), *args], cwd=checkout, env=env, capture_output=True, text=True, timeout=120)


def summary_row(output: str, name: str) -> list[str]:
    rows = [line.split() for line in output.splitlines() if line.startswith(name + " ")]
    assert rows, output
    return rows[-1]


def test_list_names_every_step_including_legacy_ui():
    listed = subprocess.run(["bash", str(CHECK), "--list"], cwd=REPO, env={**os.environ, "PYTHON": sys.executable}, capture_output=True, text=True, timeout=120)
    assert listed.returncode == 0, listed.stderr
    steps = listed.stdout.split()
    assert "legacy-ui" in steps and "legacy-ui:fixtures" in steps
    assert {"env:gradio-free", "ruff:remi", "pytest:tests", "contract:openapi", "pnpm:build", "pnpm:lint", "pnpm:vitest", "e2e"} <= set(steps)


def test_missing_legacy_interpreter_is_reported_as_skipped(checkout: Path):
    result = run(checkout, "--skip", OTHER_STEPS)
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary_row(result.stdout, "legacy-ui")[:2] == ["legacy-ui", "skip"]
    assert summary_row(result.stdout, "legacy-ui:fixtures")[:2] == ["legacy-ui:fixtures", "skip"]
    assert "legacy-ui: no Gradio 4.40 interpreter: build it with tools/legacy_ui_env.sh" in result.stdout
    assert "0 of 2 steps passed, 2 skipped." in result.stdout
    assert "All 2 steps passed" not in result.stdout


def test_strict_fails_on_a_skipped_step(checkout: Path):
    result = run(checkout, "--strict", "--skip", OTHER_STEPS)
    assert result.returncode == 1
    assert summary_row(result.stdout, "legacy-ui")[1] == "skip"


def test_only_legacy_ui_without_interpreter_fails(checkout: Path):
    result = run(checkout, "--only", "legacy-ui", "--skip", "legacy-ui:fixtures")
    assert result.returncode == 1
    assert summary_row(result.stdout, "legacy-ui")[1] == "FAIL"
    assert "no Gradio 4.40 interpreter" in result.stdout


def test_runs_the_self_test_with_remi_legacy_python(checkout: Path, tmp_path: Path):
    record = tmp_path / "args.txt"
    legacy = fake_python(tmp_path / "gradio" / "python", record)
    result = run(checkout, "--skip", OTHER_STEPS, legacy=str(legacy))
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary_row(result.stdout, "legacy-ui")[1] == "pass"
    assert summary_row(result.stdout, "legacy-ui:fixtures")[1] == "pass"
    # the self-test's and the fixtures check's JSON results go with the logs
    logs = checkout.parent / "logs"
    assert [line.split() for line in record.read_text(encoding="utf-8").splitlines()] == [
        ["tools/record_legacy_ui.py", "--self-test", "--report", str(logs / "legacy-ui.json")],
        ["tools/record_legacy_ui.py", "--fixtures", "--check", "--report", str(logs / "legacy-ui-fixtures.json")],
    ]
    assert "All 2 steps passed." in result.stdout


def test_finds_the_default_legacy_environment(checkout: Path, tmp_path: Path):
    record = tmp_path / "args.txt"
    fake_python(checkout.parent / "remi-gradio-env" / "bin" / "python", record)
    result = run(checkout, "--skip", OTHER_STEPS)
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary_row(result.stdout, "legacy-ui")[1] == "pass"
    assert record.is_file()


def test_failing_self_test_fails_the_run(checkout: Path, tmp_path: Path):
    legacy = fake_python(tmp_path / "gradio" / "python", tmp_path / "args.txt", status=1)
    result = run(checkout, "--skip", OTHER_STEPS, legacy=str(legacy))
    assert result.returncode == 1
    assert summary_row(result.stdout, "legacy-ui")[1] == "FAIL"
    assert summary_row(result.stdout, "legacy-ui:fixtures")[1] == "FAIL"
    assert "2 of 2 steps failed (0 passed, 0 skipped)." in result.stdout


def test_a_step_that_cannot_run_here_is_a_skip_with_its_reason(checkout: Path, tmp_path: Path):
    """tools/sdapi_contract.py exits 77 when a shallow clone (CI) lacks the upstream commit and
    cannot fetch it: the summary lists the step as skipped, with the tool's last line as the reason."""

    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text('#!/bin/sh\necho "$@" >> "' + str(tmp_path / "args.txt") + '"\necho "cannot compare /sdapi/v1 with upstream 5998fb39ea: fetching it from origin failed" >&2\nexit 77\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    result = run(checkout, "--only", "contract:sdapi", python=str(python))
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary_row(result.stdout, "contract:sdapi")[:2] == ["contract:sdapi", "skip"]
    assert "contract:sdapi: cannot compare /sdapi/v1 with upstream 5998fb39ea: fetching it from origin failed" in result.stdout
    assert "0 of 1 steps passed, 1 skipped." in result.stdout
    assert (tmp_path / "args.txt").read_text(encoding="utf-8").splitlines()[-1] == "tools/sdapi_contract.py"  # after the ruff probe
    assert run(checkout, "--strict", "--only", "contract:sdapi", python=str(python)).returncode == 1
    # any other failure is still a failure
    python.write_text("#!/bin/sh\necho 'route GET /sdapi/v1/options: missing in Remi' >&2\nexit 1\n", encoding="utf-8")
    failed = run(checkout, "--only", "contract:sdapi", python=str(python))
    assert failed.returncode == 1 and summary_row(failed.stdout, "contract:sdapi")[1] == "FAIL"


@pytest.mark.parametrize(("ci", "reporter"), [(None, ["--reporter=line"]), ("1", []), ("true", [])])
def test_the_e2e_step_keeps_the_ci_reporters(checkout: Path, tmp_path: Path, ci: str | None, reporter: list[str]):
    """playwright.config.ts writes the HTML report CI uploads (node_modules/.e2e-report) only with its
    own reporters, which --reporter=line replaced: the step forces the line reporter only outside CI
    (``Boolean(process.env.CI)``, so any non-empty CI)."""

    (checkout / "webui").mkdir()
    record = tmp_path / "pnpm-args.txt"
    fake_python(tmp_path / "bin" / "pnpm", record)
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS", "CI"}}
    env.update(PYTHON=sys.executable, REMI_CHECK_LOGS=str(checkout.parent / "logs"), PATH=f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}")
    if ci is not None:
        env["CI"] = ci
    result = subprocess.run(["bash", str(checkout / "tools" / "check.sh"), "--only", "e2e"], cwd=checkout, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary_row(result.stdout, "e2e")[1] == "pass"
    assert record.read_text(encoding="utf-8").split() == ["exec", "playwright", "test", *reporter]


def test_the_steps_do_not_see_commandline_args(checkout: Path, tmp_path: Path):
    """modules.paths_internal adds COMMANDLINE_ARGS to the WebUI's command line: exported by a
    shell, its --data-dir would give the test suites the owner's settings file (tests/conftest.py
    drops it for tests/ only, and none of the extension suites' own conftests does)."""

    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    seen = tmp_path / "seen.txt"
    python.write_text(f'#!/bin/sh\necho "${{COMMANDLINE_ARGS-unset}}" >> "{seen}"\nexit 0\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS"}}
    env.update(PYTHON=str(python), REMI_CHECK_LOGS=str(checkout.parent / "logs"), COMMANDLINE_ARGS="--data-dir /the/owners/data")
    result = subprocess.run(["bash", str(checkout / "tools" / "check.sh"), "--only", "contract:sdapi"], cwd=checkout, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert set(seen.read_text(encoding="utf-8").splitlines()) == {"unset"}
