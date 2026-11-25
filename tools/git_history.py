"""Commits a quality gate reads from the git history, fetched into a shallow clone.

CI checks out one commit (``actions/checkout`` fetches with depth 1). The
gates that read an older revision (``tools/sdapi_contract.py``'s last absorbed
upstream commit, the legacy-UI recorder's revisions, the extension settings
record) or walk the history (``tests/test_remi_pinned_revisions.py``) found
nothing there and skipped without a word. :func:`ensure_commits` and
:func:`ensure_history` fetch what a shallow clone is missing from its remote
first (the pinned commits themselves, else the whole history), and return the
reason only when that fails; the gate then skips naming it, and
``tools/check.sh`` lists the skip in its summary (step ``contract:sdapi``).

A clone with its whole history is never fetched into: a commit missing there
is gone (a history rewrite), which fetching could hide.

Standard library and git only: the legacy-UI recorder runs this under its own
Gradio interpreter. ``tools/`` is not a package: the other tools load this file
by path under :data:`MODULE` (one copy per process, so a failed fetch is tried once).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = "remi_tools_git_history"
FETCH_TIMEOUT = 600
"""Seconds a fetch may take (the whole history of this repository is about 110 MB)."""
LOCAL_TIMEOUT = 30
"""Seconds a local git command may take."""
FULL_HASH = re.compile(r"[0-9a-f]{40}")
"""Only a full hash can be fetched by itself; an abbreviated one needs the whole history."""

_failures: dict[tuple[str, ...], str] = {}
"""Fetches that failed in this process, by repository and revisions: their reason, not a second attempt."""


def _git(repo: Path | str, *args: str, timeout: float = LOCAL_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """``git`` in ``repo`` that never waits for input (no credential or terminal prompt)."""

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(["git", *args], cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(["git", *args], 1, "", str(error) or type(error).__name__)


def _error(result: subprocess.CompletedProcess[str]) -> str:
    """Git's message, on one line (its first line names the remote or the commit it could not get)."""

    lines = dict.fromkeys(line.strip() for line in result.stderr.splitlines() if line.strip())
    return " ".join(lines) if lines else f"git {result.args[1]} exited with {result.returncode}"


def has_commit(rev: str, repo: Path | str = ROOT) -> bool:
    return _git(repo, "cat-file", "-e", f"{rev}^{{commit}}").returncode == 0


def is_shallow(repo: Path | str = ROOT) -> bool | None:
    """Whether ``repo`` is a shallow clone; None outside a git checkout."""

    found = _git(repo, "rev-parse", "--is-shallow-repository")
    return None if found.returncode != 0 else found.stdout.strip() == "true"


def remote(repo: Path | str = ROOT) -> str | None:
    """The remote to fetch from: ``origin``, else the first one."""

    names = _git(repo, "remote").stdout.split()
    return "origin" if "origin" in names else (names[0] if names else None)


def _shallow_ancestors(repo: Path | str) -> list[str]:
    """The shallow boundary commits HEAD's history still stops at."""

    path = _git(repo, "rev-parse", "--git-path", "shallow").stdout.strip()
    boundary = Path(repo, path) if path else None
    if boundary is None or not boundary.is_file():
        return []
    return [commit for commit in boundary.read_text(encoding="utf-8").split() if _git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode == 0]


def _unshallow(repo: Path | str, name: str) -> str | None:
    """Fetch HEAD's whole history from ``name``; the error, or None."""

    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    # HEAD itself first: a pull request's merge commit is on no branch the default refspec fetches
    fetched = _git(repo, "fetch", "--no-tags", "--unshallow", name, *([head] if head else []), timeout=FETCH_TIMEOUT)
    if fetched.returncode != 0 and head:
        fetched = _git(repo, "fetch", "--no-tags", "--unshallow", name, timeout=FETCH_TIMEOUT)
    if fetched.returncode != 0:
        return _error(fetched)
    if left := _shallow_ancestors(repo):
        return f"the history still stops at {left[0][:10]}"
    return None


def ensure_commits(*revs: str, repo: Path | str = ROOT) -> str | None:
    """Make every commit of ``revs`` readable in ``repo``.

    In a shallow clone a missing commit is fetched from the remote (by its
    hash, at depth 1; an abbreviated hash, or a remote that does not serve a
    commit by hash, needs the whole history). Returns None when every commit is
    there, else why not (for the gate's skip)."""

    missing = [rev for rev in revs if not has_commit(rev, repo)]
    if not missing:
        return None
    names = ", ".join(rev[:10] for rev in missing)
    shallow = is_shallow(repo)
    if shallow is None:
        return f"{names}: not a git checkout"
    if not shallow:
        return f"{names} is not in this clone, which has its whole history (nothing to fetch: was the history rewritten?)"
    key = (str(repo), *missing)
    if key in _failures:
        return _failures[key]
    name = remote(repo)
    if name is None:
        reason = f"{names} is not in this shallow clone, which has no remote to fetch it from"
    else:
        errors = []
        by_hash = [rev for rev in missing if FULL_HASH.fullmatch(rev)]
        if by_hash:
            fetched = _git(repo, "fetch", "--no-tags", "--depth=1", name, *by_hash, timeout=FETCH_TIMEOUT)
            if fetched.returncode != 0:
                errors.append(_error(fetched))
        if not all(has_commit(rev, repo) for rev in missing) and (error := _unshallow(repo, name)):
            errors.append(error)
        if all(has_commit(rev, repo) for rev in missing):
            return None
        reason = f"{names} is not in this shallow clone and fetching it from {name} failed: {'; '.join(dict.fromkeys(errors)) or 'the remote does not have it'}"
    _failures[key] = reason
    return reason


def ensure_history(repo: Path | str = ROOT) -> str | None:
    """Make HEAD's whole history readable in ``repo`` (``git fetch --unshallow``
    in a shallow clone). Returns None when it is, else why not."""

    shallow = is_shallow(repo)
    if shallow is None:
        return "not a git checkout"
    if not shallow or not _shallow_ancestors(repo):
        return None
    key = (str(repo), "--unshallow")
    if key in _failures:
        return _failures[key]
    name = remote(repo)
    if name is None:
        reason = "this shallow clone lacks the history and has no remote to fetch it from"
    elif (error := _unshallow(repo, name)) is None:
        return None
    else:
        reason = f"this shallow clone lacks the history and fetching it from {name} failed: {error}"
    _failures[key] = reason
    return reason
