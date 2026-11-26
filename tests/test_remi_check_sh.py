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
OTHER_STEPS = "env,ruff,pytest,contract,i18n,pnpm,api-docs,perf,e2e"

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
    assert {"env:gradio-free", "ruff:remi", "pytest:tests", "contract:openapi", "pnpm:build", "api-docs", "pnpm:lint", "pnpm:vitest", "e2e"} <= set(steps)
    assert steps.index("api-docs") == steps.index("pnpm:build") + 1


def test_the_tests_of_the_build_run_after_it_and_not_beside_it(checkout: Path, tmp_path: Path):
    """pnpm:build empties webui/dist while it runs, so pytest:tests (another lane) leaves out the
    tests that read it (marker spa_build), and api-docs runs them after the build, in its lane,
    where a missing build fails them instead of skipping them (REMI_SPA_BUILT)."""

    (checkout / "webui").mkdir()
    record = tmp_path / "record.txt"
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text(f'#!/bin/sh\necho "python ${{REMI_SPA_BUILT-unset}} $*" >> "{record}"\nexit 0\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    pnpm = tmp_path / "bin" / "pnpm"
    pnpm.write_text(f'#!/bin/sh\nsleep 1\necho "pnpm $*" >> "{record}"\nexit 0\n', encoding="utf-8")
    pnpm.chmod(pnpm.stat().st_mode | stat.S_IXUSR)
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS", "REMI_SPA_BUILT"}}
    env.update(PYTHON=str(python), REMI_CHECK_LOGS=str(checkout.parent / "logs"), PATH=f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}")
    command = ["bash", str(checkout / "tools" / "check.sh"), "--only", "pytest:tests,pnpm:build,api-docs"]
    result = subprocess.run(command, cwd=checkout, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = record.read_text(encoding="utf-8").splitlines()
    tests = next(call for call in calls if " -m pytest tests " in call and "--cov=remi" in call)
    assert "-m not spa_build " in tests and tests.startswith("python unset "), tests
    # (pytest:tests runs beside the build, so its calls may come before or after it)
    api_docs = [i for i, call in enumerate(calls) if "-m spa_build " in call]
    assert len(api_docs) == 1 and api_docs[0] > calls.index("pnpm build"), calls
    assert calls[api_docs[0]].startswith("python 1 -m pytest tests -m spa_build "), calls


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


def rendezvous_tool(path: Path, role: str, other: str, record: Path, wait: float, when: str | None = None) -> Path:
    """An executable that records its start, waits up to ``wait`` seconds for ``other`` to start
    (only when its arguments contain ``when``), then records its end and exits 0."""

    path.parent.mkdir(parents=True, exist_ok=True)
    folder = record.parent
    rounds = int(wait * 10)
    body = f'echo "{role} start" >> "{record}"\ntouch "{folder / role}.started"\ni=0\nwhile [ $i -lt {rounds} ] && [ ! -e "{folder / other}.started" ]; do sleep 0.1; i=$((i + 1)); done\necho "{role} end" >> "{record}"\n'
    if when is not None:
        body = f'case "$*" in\n  *{when}*) ;;\n  *) exit 0 ;;\nesac\n' + body
    path.write_text("#!/bin/sh\n" + body + "exit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def run_two_lanes(checkout: Path, tmp_path: Path, *args: str, wait: float) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """contract:sdapi (a Python lane) and pnpm:lint (the frontend lane), each waiting for the other."""

    (checkout / "webui").mkdir()
    record = tmp_path / "record.txt"
    python = rendezvous_tool(tmp_path / "bin" / "python", "python", "pnpm", record, wait, when="sdapi_contract")
    rendezvous_tool(tmp_path / "bin" / "pnpm", "pnpm", "python", record, wait)
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS"}}
    env.update(PYTHON=str(python), REMI_CHECK_LOGS=str(checkout.parent / "logs"), PATH=f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}")
    command = ["bash", str(checkout / "tools" / "check.sh"), "--only", "contract:sdapi,pnpm:lint", *args]
    result = subprocess.run(command, cwd=checkout, env=env, capture_output=True, text=True, timeout=120)
    return result, record.read_text(encoding="utf-8").splitlines()


def test_the_python_and_frontend_lanes_run_side_by_side(checkout: Path, tmp_path: Path):
    result, record = run_two_lanes(checkout, tmp_path, wait=30)
    assert result.returncode == 0, result.stdout + result.stderr
    # each saw the other start before it ended: run one after the other, the first would wait 30 s
    assert sorted(record[:2]) == ["pnpm start", "python start"], record
    # the summary keeps the order the steps are declared in, whichever ends first
    rows = [line.split()[0] for line in result.stdout.splitlines() if line.startswith(("contract:sdapi ", "pnpm:lint "))]
    assert rows == ["contract:sdapi", "pnpm:lint"], result.stdout
    assert "All 2 steps passed." in result.stdout


def test_serial_runs_one_step_at_a_time(checkout: Path, tmp_path: Path):
    result, record = run_two_lanes(checkout, tmp_path, "--serial", wait=1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert record == ["python start", "python end", "pnpm start", "pnpm end"]
    assert "Running 2 steps one at a time;" in result.stdout


@pytest.mark.parametrize(
    ("only", "first"),
    [
        ("contract:sdapi", "Running 1 steps one at a time;"),
        ("pytest:tests,contract:sdapi", "Running 2 steps in two lanes at once (python, suites);"),
        ("pytest:tests,contract:sdapi,pnpm:lint", "Running 3 steps in three lanes at once (python, suites, frontend);"),
        ("pnpm:install,pnpm:lint", "Running 2 steps one at a time;"),
        ("pytest:tests,pytest:coverage", "Running 2 steps one at a time;"),
        ("pytest:coverage,pnpm:lint", "Running 2 steps in two lanes at once (frontend, after);"),
    ],
    ids=["one step", "python and suites", "all three", "frontend alone", "after follows python", "after beside frontend"],
)
def test_the_first_line_names_the_lanes_that_start_at_once(checkout: Path, tmp_path: Path, only: str, first: str):
    """A subset can leave lanes empty (CI's SPA job runs the frontend lane alone): the line that
    starts the run says which lanes run side by side, and "one at a time" when only one has steps."""

    (checkout / "webui").mkdir()
    record = tmp_path / "record.txt"
    python = fake_python(tmp_path / "bin" / "python", record)
    fake_python(tmp_path / "bin" / "pnpm", record)
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS"}}
    env.update(PYTHON=str(python), REMI_CHECK_LOGS=str(checkout.parent / "logs"), PATH=f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}")
    result = subprocess.run(["bash", str(checkout / "tools" / "check.sh"), "--only", only], cwd=checkout, env=env, capture_output=True, text=True, timeout=120)
    started = [line for line in result.stdout.splitlines() if line.startswith("Running ")]
    assert len(started) == 1 and started[0].startswith(first), result.stdout + result.stderr


@pytest.mark.parametrize(("only", "build", "flag"), [("pnpm:build,api-docs", 0, "1"), ("pnpm:build,api-docs", 1, "unset"), ("api-docs", None, "unset")], ids=["built", "build failed", "no build"])
def test_a_missing_build_fails_api_docs_only_after_this_runs_build(checkout: Path, tmp_path: Path, only: str, build: int | None, flag: str):
    """REMI_SPA_BUILT turns the spa_build tests' skip (no build) into a failure, but only once
    pnpm:build has passed in this run: without a build (--skip pnpm, a machine without Node.js)
    they skip as they did in pytest:tests."""

    (checkout / "webui").mkdir()
    record = tmp_path / "record.txt"
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text(f'#!/bin/sh\necho "${{REMI_SPA_BUILT-unset}} $*" >> "{record}"\nexit 0\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    pnpm = tmp_path / "bin" / "pnpm"
    pnpm.write_text(f"#!/bin/sh\nexit {build or 0}\n", encoding="utf-8")
    pnpm.chmod(pnpm.stat().st_mode | stat.S_IXUSR)
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS", "REMI_SPA_BUILT"}}
    env.update(PYTHON=str(python), REMI_CHECK_LOGS=str(checkout.parent / "logs"), PATH=f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}")
    result = subprocess.run(["bash", str(checkout / "tools" / "check.sh"), "--only", only], cwd=checkout, env=env, capture_output=True, text=True, timeout=120)
    assert summary_row(result.stdout, "api-docs")[1] == "pass", result.stdout + result.stderr
    calls = [call.split()[0] for call in record.read_text(encoding="utf-8").splitlines() if "-m spa_build" in call]
    assert calls == [flag], calls


@pytest.mark.parametrize("slow", ['*"-m pytest tests "*', "*test_remi_core_zh_cn*"], ids=["pytest:tests", "i18n:server"])
def test_pytest_coverage_waits_for_both_python_lanes(checkout: Path, tmp_path: Path, slow: str):
    """pytest:coverage combines the coverage data of every pytest step, so it starts only after
    pytest:tests (one lane) and the other Python steps (the other lane) have ended, whichever
    lane takes longer."""

    record = tmp_path / "args.txt"
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text(f'#!/bin/sh\ncase "$*" in\n  {slow}) sleep 2 ;;\nesac\necho "$@" >> "{record}"\nexit 0\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    result = run(checkout, "--only", "pytest,i18n:server", python=str(python))
    assert result.returncode == 0, result.stdout + result.stderr
    calls = record.read_text(encoding="utf-8").splitlines()
    strict = next(i for i, call in enumerate(calls) if "check_coverage.py strict" in call)
    assert any("-m pytest tests " in call for call in calls[:strict]), calls
    assert any("test_remi_core_zh_cn" in call for call in calls[:strict]), calls


def lane_tool(path: Path, pids: Path) -> Path:
    """A python or pnpm that, for pytest:tests, contract:sdapi and pnpm:lint (one step in each
    lane), starts a child process, records both process ids and waits; other calls end at once."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\ncase "$*" in\n  *"-m pytest tests "*|*sdapi_contract*|lint) ;;\n  *) exit 0 ;;\nesac\nsleep 60 &\necho "$$ $!" >> "{pids}"\nwait\n', encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="signals")
@pytest.mark.parametrize("stop", ["SIGTERM", "SIGINT"])
def test_stopping_the_run_stops_every_lane(checkout: Path, tmp_path: Path, stop: str):
    """A stopped run (a terminal's Ctrl-C, a tool stopping it) leaves no step running in the
    background, in any lane, nor what a step started; the script then ends by that signal, as a
    stopped command does (a shell loop running it stops at Ctrl-C)."""

    import signal
    import time

    sent = getattr(signal, stop)
    (checkout / "webui").mkdir()
    pids = tmp_path / "pids.txt"
    python = lane_tool(tmp_path / "bin" / "python", pids)
    lane_tool(tmp_path / "bin" / "pnpm", pids)
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS"}}
    env.update(PYTHON=str(python), REMI_CHECK_LOGS=str(checkout.parent / "logs"), PATH=f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}")
    command = ["bash", str(checkout / "tools" / "check.sh"), "--only", "pytest:tests,contract:sdapi,pnpm:lint"]

    def default_sigint() -> None:
        # a shell cannot trap a signal ignored when it started, and this test may run in a lane
        # of tools/check.sh itself, where (a background job) SIGINT is ignored
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    process = subprocess.Popen(command, cwd=checkout, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=default_sigint)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (pids.is_file() and len(pids.read_text(encoding="utf-8").split()) == 6):
            time.sleep(0.1)
        started = [int(pid) for pid in pids.read_text(encoding="utf-8").split()]
        assert len(started) == 6, started  # a step and its child in each of the three lanes
        process.send_signal(sent)
        assert process.wait(timeout=30) == -sent
        deadline = time.monotonic() + 10
        alive = started
        while alive and time.monotonic() < deadline:
            alive = [pid for pid in started if _alive(pid)]
            time.sleep(0.1)
        assert not alive, f"still running: {alive} of {started}"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="signals")
def test_a_lane_whose_script_was_killed_starts_no_further_step(checkout: Path, tmp_path: Path):
    """SIGKILL cannot be trapped: the lanes outlive the script. The step running goes on (as it did
    when one step ran at a time), but its lane starts no other, even while the killed script is a
    zombie its parent has not reaped."""

    import signal
    import time

    record = tmp_path / "record.txt"
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text(f'#!/bin/sh\ncase "$*" in\n  *sdapi_contract*) echo "sdapi start" >> "{record}"; sleep 2; echo "sdapi end" >> "{record}" ;;\n  *dump_settings*) echo "settings" >> "{record}" ;;\nesac\nexit 0\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    env = {key: value for key, value in os.environ.items() if key not in {"REMI_LEGACY_PYTHON", "REMI_CHECK_LOGS"}}
    env.update(PYTHON=str(python), REMI_CHECK_LOGS=str(checkout.parent / "logs"))
    command = ["bash", str(checkout / "tools" / "check.sh"), "--only", "contract:sdapi,contract:settings"]
    process = subprocess.Popen(command, cwd=checkout, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (record.is_file() and "sdapi start" in record.read_text(encoding="utf-8")):
            time.sleep(0.1)
        process.send_signal(signal.SIGKILL)  # not reaped until the end: a zombie meanwhile
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and "sdapi end" not in record.read_text(encoding="utf-8"):
            time.sleep(0.1)
        time.sleep(1)  # the lane would start contract:settings at once
        assert record.read_text(encoding="utf-8").splitlines() == ["sdapi start", "sdapi end"]
    finally:
        process.kill()
        process.wait()


def test_a_step_whose_lane_died_is_a_failure(checkout: Path, tmp_path: Path):
    """A lane that died (killed) leaves its step and the steps after it without a result: each
    is a failure in the summary, never a pass."""

    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text('#!/bin/sh\ncase "$*" in\n  *sdapi_contract*) kill -9 $PPID ;;\nesac\nexit 0\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    result = run(checkout, "--only", "contract:sdapi,contract:settings", python=str(python))
    assert result.returncode == 1, result.stdout + result.stderr
    assert summary_row(result.stdout, "contract:sdapi")[1] == "FAIL"
    assert summary_row(result.stdout, "contract:settings")[1] == "FAIL"
    assert "2 of 2 steps failed (0 passed, 0 skipped)." in result.stdout


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie (ended, not yet reaped by its parent) is not running
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(status.stdout.strip()) and not status.stdout.strip().startswith("Z")
