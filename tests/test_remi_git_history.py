"""The gates that read the git history fetch what a shallow clone lacks.

CI checks out one commit (``actions/checkout``'s depth 1): the ``/sdapi/v1``
contract (``tools/sdapi_contract.py`` reads upstream's API at the last
absorbed commit), the legacy-UI recorder and the pinned-revision test found
nothing to read there and skipped without a word. ``tools/git_history.py``
fetches the missing commits (or the whole history) first and returns the
reason only when that fails. Each test builds a tiny repository and a
``git clone --depth 1 file://`` of it: no network.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


history = _load("tool_git_history_test", REPO / "tools" / "git_history.py")
contract = _load("tool_sdapi_contract_history_test", REPO / "tools" / "sdapi_contract.py")
recorder = _load("tool_record_legacy_ui_history_test", REPO / "tools" / "record_legacy_ui.py")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout.strip()


@pytest.fixture
def repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list[str]]:
    """``(source, shallow clone, [old, middle, head])``: three commits of
    ``modules/api/api.py`` and ``requirements.txt``, the clone holding the last."""

    # no user or system git configuration (signing, URL rewrites, hooks), here and in the tools
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for key, value in {"GIT_AUTHOR_NAME": "Remi test", "GIT_AUTHOR_EMAIL": "test@example.invalid", "GIT_COMMITTER_NAME": "Remi test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}.items():
        monkeypatch.setenv(key, value)
    source = tmp_path / "source"
    (source / "modules" / "api").mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=source)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=source)  # a commit by hash, as GitHub serves it
    commits = []
    for number in range(3):
        (source / "modules" / "api" / "api.py").write_text(f"class Api:  # revision {number}\n    pass\n", encoding="utf-8")
        (source / "requirements.txt").write_text(f"package=={number}\n", encoding="utf-8")
        git("add", "-A", cwd=source)
        git("commit", "-q", "-m", f"revision {number}", cwd=source)
        commits.append(git("rev-parse", "HEAD", cwd=source))
    clone = tmp_path / "clone"
    git("clone", "-q", "--depth", "1", source.as_uri(), str(clone), cwd=tmp_path)
    assert git("rev-parse", "--is-shallow-repository", cwd=clone) == "true"
    return source, clone, commits


def test_the_sdapi_contract_fetches_the_upstream_revision_a_shallow_clone_lacks(repos, monkeypatch):
    _source, clone, (old, middle, _head) = repos
    assert not history.has_commit(old, clone)  # what the contract test skipped on
    monkeypatch.setattr(contract, "ROOT", str(clone))
    assert contract.show(old, "modules/api/api.py") == "class Api:  # revision 0\n    pass\n"
    # that one commit, not the whole history
    assert history.is_shallow(clone) and not history.has_commit(middle, clone)


def test_the_legacy_ui_recorder_fetches_its_revision(repos, monkeypatch):
    _source, clone, (old, _unused, _head) = repos
    monkeypatch.setattr(recorder, "REPO", clone)
    assert recorder.git_show(old, "requirements.txt") == "package==0\n"


def test_an_abbreviated_hash_fetches_the_whole_history(repos):
    _source, clone, (old, middle, _head) = repos
    assert history.ensure_commits(old[:9], repo=clone) is None
    assert history.has_commit(middle, clone) and not history.is_shallow(clone)


def test_the_ancestry_check_gets_the_whole_history(repos):
    _source, clone, (old, _unused, _head) = repos
    assert history.ensure_commits(old, repo=clone) is None  # a second shallow root, below HEAD
    assert history.ensure_history(clone) is None
    assert not history.is_shallow(clone)
    assert subprocess.run(["git", "merge-base", "--is-ancestor", old, "HEAD"], cwd=clone).returncode == 0
    assert history.ensure_history(clone) is None  # nothing left to fetch


# git, except that a fetch first holds the clone's shallow.lock for a second, as a fetch of a long
# history holds it while the pack arrives
SLOW_FETCH = """#!/bin/sh
if [ "$1" = fetch ]; then
  lock="$("{git}" rev-parse --git-path shallow.lock)"
  if ! (set -C; : >"$lock") 2>/dev/null; then
    echo "fatal: Unable to create '$PWD/$lock': File exists." >&2
    exit 128
  fi
  touch "{fetching}"
  sleep 1
  rm -f "$lock"
fi
exec "{git}" "$@"
"""

# another gate's process: ensure_history or ensure_commits, printing what it returned
OTHER_GATE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("remi_tools_git_history", sys.argv[1])
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)
repo, which, rev = sys.argv[2:5]
print(history.ensure_history(repo) if which == "history" else history.ensure_commits(rev, repo=repo))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="fetches are serialized with fcntl; the stand-in git is a shell script")
@pytest.mark.parametrize(("first", "second"), [("history", "commits"), ("commits", "history"), ("history", "history")])
def test_a_fetch_waits_for_another_processes_fetch_into_the_clone(repos, tmp_path, monkeypatch, first, second):
    """tools/check.sh runs pytest:tests (the pinned-revision test fetches the whole history) and
    contract:sdapi (one commit) side by side. A fetch into the shallow clone while the other
    process's ran failed on git's shallow.lock, and the pinned-revision test skipped. It now waits
    for the other fetch, then looks again: a --depth=1 fetch into the clone the other process has
    just given its whole history would make it shallow again, and --unshallow fails on it."""

    _source, clone, (old, _middle, _head) = repos
    fetching = tmp_path / "fetching"
    stand_in = tmp_path / "bin" / "git"
    stand_in.parent.mkdir()
    stand_in.write_text(SLOW_FETCH.format(git=shutil.which("git"), fetching=fetching), encoding="utf-8")
    stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")
    command = [sys.executable, "-c", OTHER_GATE, str(REPO / "tools" / "git_history.py"), str(clone), first, old]
    other = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 60
        while not fetching.exists() and other.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert fetching.exists(), other.communicate()  # the other process's fetch is under way
        mine = history.ensure_history(clone) if second == "history" else history.ensure_commits(old, repo=clone)
        out, err = other.communicate(timeout=60)
    finally:
        if other.poll() is None:
            other.kill()
            other.communicate()
    assert (out.strip(), mine) == ("None", None), err
    assert history.has_commit(old, clone) and not history.is_shallow(clone)


def test_a_failed_fetch_names_its_reason(repos, monkeypatch, capsys):
    _source, clone, (old, middle, _head) = repos
    git("remote", "set-url", "origin", (clone.parent / "gone").as_uri(), cwd=clone)
    reason = history.ensure_commits(old, repo=clone)
    assert reason is not None and reason.startswith(f"{old[:10]} is not in this shallow clone and fetching it from origin failed: ")
    assert "gone" in reason  # git's own error, naming the remote it could not read
    assert history.ensure_commits(old, repo=clone) == reason  # remembered: one attempt per process
    assert history.ensure_history(clone).startswith("this shallow clone lacks the history and fetching it from origin failed: ")
    # the contract tool reports it, with the status tools/check.sh shows as a skip
    monkeypatch.setattr(contract, "ROOT", str(clone))
    with pytest.raises(contract.RevisionUnavailable, match="fetching it from origin failed"):
        contract.show(old, "modules/api/api.py")
    assert contract.main(["--rev", old]) == contract.UNAVAILABLE
    assert f"cannot compare /sdapi/v1 with upstream {old[:10]}: {reason}" in capsys.readouterr().err
    monkeypatch.setattr(recorder, "REPO", clone)
    with pytest.raises(SystemExit, match=f"cannot read requirements.txt at {old[:10]}: .*fetching it from origin failed"):
        recorder.git_show(old, "requirements.txt")
    git("remote", "remove", "origin", cwd=clone)
    assert history.ensure_commits(middle, repo=clone) == f"{middle[:10]} is not in this shallow clone, which has no remote to fetch it from"


def fetches(monkeypatch) -> list[tuple[str, ...]]:
    """The ``git fetch`` commands the helper runs from now on."""

    seen: list[tuple[str, ...]] = []
    real = history._git

    def counting(repo, *args, **kwargs):
        if args[:1] == ("fetch",):
            seen.append(args)
        return real(repo, *args, **kwargs)

    monkeypatch.setattr(history, "_git", counting)
    return seen


def test_a_remote_that_failed_to_give_the_history_is_not_asked_for_it_again(repos, monkeypatch):
    """Each gate asks for its own commits; once the remote failed to give the
    whole history, the others still fetch their commits by hash but not the
    history again (three fetches per gate otherwise)."""

    _source, clone, (old, middle, _head) = repos
    git("remote", "set-url", "origin", (clone.parent / "gone").as_uri(), cwd=clone)
    reason = history.ensure_commits(old, repo=clone)
    assert reason is not None
    seen = fetches(monkeypatch)
    later = history.ensure_commits(middle, repo=clone)
    assert later.startswith(f"{middle[:10]} is not in this shallow clone and fetching it from origin failed: ")
    assert later.split(" failed: ", 1)[1] in reason  # the first failure's own words
    assert history.ensure_history(clone).startswith("this shallow clone lacks the history and fetching it from origin failed: ")
    assert seen == [("fetch", "--no-tags", "--depth=1", "origin", middle)]


def test_a_fetch_that_timed_out_is_not_retried(repos, monkeypatch):
    _source, clone, (old, _unused, _head) = repos
    real = subprocess.run
    seen = []

    def hanging(command, **kwargs):
        if command[1:2] == ["fetch"]:
            seen.append(command)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return real(command, **kwargs)

    monkeypatch.setattr(history.subprocess, "run", hanging)
    reason = history.ensure_commits(old, repo=clone)
    assert reason is not None and "timed out" in reason
    assert len(seen) == 1  # neither the whole history nor HEAD's branch after it
    assert history.ensure_history(clone) is not None and len(seen) == 1


def test_a_full_clone_is_never_fetched_into(repos, tmp_path):
    source, _clone, (old, _unused, head) = repos
    full = tmp_path / "full"
    git("clone", "-q", source.as_uri(), str(full), cwd=tmp_path)
    assert history.ensure_commits(old, head, repo=full) is None and history.ensure_history(full) is None
    gone = "0123456789" * 4
    assert history.ensure_commits(gone, repo=full) == "0123456789 is not in this clone, which has its whole history (nothing to fetch: was the history rewritten?)"
    plain = tmp_path / "plain"
    plain.mkdir()
    assert history.ensure_commits(gone, repo=plain) == "0123456789: not a git checkout"
    assert history.ensure_history(plain) == "not a git checkout"


def test_git_never_waits_for_input(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(history.subprocess, "run", fake_run)
    result = history._git(REPO, "fetch", "origin", timeout=1)
    assert result.returncode == history.TIMED_OUT and result.stderr == "git fetch timed out after 1 s"
    assert seen["stdin"] is subprocess.DEVNULL and seen["env"]["GIT_TERMINAL_PROMPT"] == "0"
