"""Where users get the source code of the running server: the offer section 13
of the GNU AGPL v3 asks of a modified program that users reach over a network
(``docs/remi-webui/licensing.md``).

The link is ``source_url`` of ``GET /api/v2/auth/status`` (a public path, so
the sign-in screen has it too) and of the bootstrap's ``auth``. By default it
is Remi's public repository: the tree of the running commit when that commit
is published there (a remote-tracking ref of a remote pointing at the
repository contains it), else the ``remi`` branch. The operator of a modified
copy names its source with ``--source-url`` or ``REMI_SOURCE_URL``; that URL
is linked as given.

Before a login only the repository link is offered, never the running commit:
the commit tells anyone who reaches the server which fixes it lacks.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import subprocess
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_SOURCE_URL = "https://github.com/beautifulrem/forge-neo-remi"
DEFAULT_BRANCH = "remi"
SOURCE_URL_ENV = "REMI_SOURCE_URL"
GIT_TIMEOUT = 10.0


@dataclass(frozen=True)
class SourceOffer:
    public: str
    """offered to everyone who reaches the server (the sign-in screen)"""
    running: str
    """offered to its users: the running version when it is known"""


def configured_source_url(cmd_opts: Any, environ: Mapping[str, str] = os.environ) -> str | None:
    """``--source-url``, else ``REMI_SOURCE_URL``, else None (the default)."""

    for value in (getattr(cmd_opts, "source_url", None), environ.get(SOURCE_URL_ENV)):
        if value and value.strip():
            return value.strip()
    return None


def web_address(value: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _repository(remote: str) -> str:
    """A git remote as ``host/owner/repo`` (lower case), to compare remotes."""

    remote = remote.strip()
    match = re.match(r"^[\w.-]+@([^:/\s]+):(.+)$", remote)  # git@github.com:owner/repo.git
    if match:
        host, path = match.groups()
    else:
        parts = urllib.parse.urlsplit(remote)
        host, path = parts.hostname or "", parts.path
    path = path.strip("/").removesuffix(".git")
    return f"{host}/{path}".lower()


def _git(root: Path, *args: str) -> str | None:
    """The output of a git command in ``root``, None when it fails."""

    git = os.environ.get("GIT", "git")
    try:
        result = subprocess.run(
            [git, "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def published_commit(root: Path, repository: str = DEFAULT_SOURCE_URL) -> str | None:
    """The checkout's HEAD when ``repository`` has it: a remote-tracking ref
    of a remote that points there contains it. None otherwise (not a git
    checkout, git missing, a local commit, another repository's commit)."""

    head = _git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
    remotes = _git(root, "config", "--get-regexp", r"^remote\..*\.url$")
    if not head or not remotes:
        return None
    wanted = _repository(repository)
    for line in remotes.splitlines():
        key, _, url = line.partition(" ")
        name = key.removeprefix("remote.").removesuffix(".url")
        if _repository(url) == wanted and _git(root, "for-each-ref", "--contains", head, "--count=1", "--format=%(refname)", f"refs/remotes/{name}"):
            return head
    return None


def resolve_source(configured: str | None, *, root: Path) -> SourceOffer:
    """The source offer of the checkout in ``root`` (``configured``: the
    operator's URL, see :func:`configured_source_url`)."""

    if configured is not None:
        if web_address(configured):
            return SourceOffer(public=configured, running=configured)
        log.warning("--source-url / %s must be an http(s) address, not %r: offering %s instead", SOURCE_URL_ENV, configured, DEFAULT_SOURCE_URL)
    branch = f"{DEFAULT_SOURCE_URL}/tree/{DEFAULT_BRANCH}"
    commit = published_commit(root)
    return SourceOffer(public=branch, running=f"{DEFAULT_SOURCE_URL}/tree/{commit}" if commit else branch)


@functools.cache
def _current(configured: str | None, root: str) -> SourceOffer:
    return resolve_source(configured, root=Path(root))


def current_source() -> SourceOffer:
    """The offer of this process (``--source-url`` / ``REMI_SOURCE_URL`` and
    the WebUI checkout), resolved once."""

    from modules.paths_internal import script_path
    from modules.shared_cmd_options import cmd_opts

    return _current(configured_source_url(cmd_opts), script_path)
