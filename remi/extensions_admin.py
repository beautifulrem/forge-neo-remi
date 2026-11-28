"""Extension administration without a UI: the installed list, enabling and
disabling, update checks and updates, installing from a git URL or the
extension index, uninstalling, and config-state backups ("Backup/Restore").

The HTTP routes (``remi/server/routes/extensions.py``) call this module.
Functions raise :class:`AdminError` subclasses carrying an HTTP status and a
translatable message template; nothing here touches the request layer.

Extensions are arbitrary code: every function that changes anything first
calls :func:`check_access`, which refuses when the server is reachable from
other machines (``--listen`` & co.) without
``--enable-insecure-extension-access``.

Git runs non-interactively (:func:`run_git`): no terminal, no password or
host-key prompt (``GIT_TERMINAL_PROMPT=0``, ssh ``BatchMode=yes``) and a
timeout, so a private repository or an unknown ssh host fails the request
instead of blocking it (and the management lock) forever.

Backups restore the extensions only. The WebUI checkout itself is never
reset from here: a backup made before the Remi rewrite points at an
upstream commit, and ``git reset --hard`` would drop the working tree.
"""

from __future__ import annotations

import contextlib
import errno
import http.client
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# The Forge Neo extension index (upstream e67193d9), maintained by a third party:
# the Extensions page says so next to the URL.
DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/eduardoabreu81/forge-neo-extensions/main/index.json"
INDEX_TIMEOUT = 20
INDEX_MAX_BYTES = 16 * 1024 * 1024
INDEX_CACHE_SECONDS = 600

GIT_TIMEOUT = 120
"""Seconds a network git command (fetch) may take."""
GIT_CLONE_TIMEOUT = 900
"""Seconds a clone (with submodules) may take."""
GIT_LOCAL_TIMEOUT = 60
"""Seconds a local git command (rev-parse, reset) may take."""
INSTALLER_TIMEOUT = 300
"""Seconds an extension's ``install.py`` may run."""
MAX_CONFIG_STATES = 50
"""Maximum number of config-state backup files retained in ``config_states/``."""

DIRNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
GIT_URL_RE = re.compile(r"^(https?|ssh|git)://[^\s]+$|^git@[^\s:]+:[^\s]+$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
STATE_ID_RE = re.compile(r"^[^/\\]{1,200}\.json$")
WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{n}" for n in range(1, 10)), *(f"lpt{n}" for n in range(1, 10))}
INSTALL_PROTOCOLS = "https:http:ssh:git"
"""transports a clone (and its submodules) may use: no ``file``/``ext::`` helpers"""
RESTORE_TARGETS = ("extensions",)

log = logging.getLogger(__name__)

lock = threading.RLock()
"""One management operation (install, update, uninstall, restore) at a time."""


# region Errors


def render(template: str, params: dict[str, Any]) -> str:
    """``"git {{command}} failed"`` with ``command="fetch"`` → ``"git fetch failed"``."""

    from remi.ui.sentence import Sentence

    return str(Sentence(template, **params))


class AdminError(Exception):
    """A refused or failed operation. ``template`` is English text with
    ``{{name}}`` placeholders (the key clients translate), ``params`` fills
    them, ``str(error)`` is the rendered English message."""

    status = 422

    def __init__(self, template: str, status: int | None = None, **params: Any) -> None:
        from remi.ui.sentence import Sentence

        sentence = Sentence(template, **params)
        self.sentence = sentence
        self.template = sentence.template
        self.params = {key: str(value) for key, value in sentence.params.items()}
        super().__init__(str(sentence))
        if status is not None:
            self.status = status

    def to_json(self) -> dict[str, Any]:
        return {"detail": str(self), "template": self.template, "params": self.params}


class AccessDenied(AdminError):
    status = 403


class NotFound(AdminError):
    status = 404


class Conflict(AdminError):
    status = 409


class GitError(AdminError):
    """Git failed or timed out (the remote is unreachable, needs credentials, ...)."""

    status = 502


# endregion

# region Access and helpers


def access_allowed() -> bool:
    from modules import shared

    return not shared.cmd_opts.disable_extension_access


def check_access() -> None:
    if not access_allowed():
        raise AccessDenied("extension access is disabled (the server is reachable from other machines; see --enable-insecure-extension-access)")


def default_index_url() -> str:
    return os.environ.get("WEBUI_EXTENSIONS_INDEX") or DEFAULT_INDEX_URL


def normalize_git_url(url: str | None) -> str:
    """``https://host/user/repo.git/`` → ``https://host/user/repo`` (for comparing remotes)."""

    if not url:
        return ""
    url = url.strip().rstrip("/")
    return url[:-4] if url.endswith(".git") else url


def dirname_from_url(url: str) -> str:
    return normalize_git_url(url).rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def valid_git_url(url: str) -> bool:
    """An https/http/ssh/git URL or ``user@host:path``, whose host cannot be
    read as a command-line option (``ssh://-oProxyCommand=...``)."""

    if not GIT_URL_RE.match(url) or any(ord(char) < 32 or ord(char) == 127 for char in url):
        return False
    if "://" in url:
        try:
            parsed = urllib.parse.urlsplit(url)
            _ = parsed.port
        except ValueError:
            return False
        host = parsed.hostname or ""
        username = parsed.username or ""
    else:  # scp-like user@host:path
        username, _, rest = url.partition("@")
        host = rest.partition(":")[0]
    return bool(host) and not host.startswith("-") and not username.startswith("-")


def valid_dirname(name: str) -> bool:
    stem = name.split(".", 1)[0].lower()
    return bool(DIRNAME_RE.match(name)) and stem not in WINDOWS_RESERVED and not name.endswith(".")


def valid_branch(branch: str) -> bool:
    """A plausible ref name (git still checks it): no option, no ``..``."""

    return bool(BRANCH_RE.match(branch)) and not branch.startswith(("-", "/")) and ".." not in branch and not branch.endswith(("/", ".lock", ".")) and "//" not in branch


def _host(parsed: urllib.parse.SplitResult) -> str:
    host = parsed.hostname or ""
    if ":" in host:  # IPv6
        host = f"[{host}]"
    return host + (f":{parsed.port}" if parsed.port else "")


def web_url(remote: str | None) -> str | None:
    """A browsable https URL for a git remote (``git@github.com:a/b.git`` too),
    credentials removed; None when the remote is not a web host."""

    if not remote:
        return None
    remote = normalize_git_url(remote)
    match = re.match(r"^git@([^:/\s]+):(.+)$", remote)
    if match:
        return f"https://{match.group(1)}/{match.group(2)}"
    if remote.startswith(("https://", "http://")):
        try:
            parsed = urllib.parse.urlsplit(remote)
            host = _host(parsed)
        except ValueError:
            return None
        if not parsed.hostname:
            return None
        return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return None


def strip_credentials(remote: str | None) -> str | None:
    """A git remote as shown to clients: no password or token in any form.

    ``scheme://user:secret@host/p`` → ``scheme://host/p`` (http(s), git, ftp:
    any user part may be a token; ssh keeps a plain user name);
    ``user:secret@host:path`` → ``user@host:path``; local paths unchanged."""

    if not remote:
        return None
    remote = remote.strip()
    if "://" in remote:
        try:
            parsed = urllib.parse.urlsplit(remote)
            host = _host(parsed)
        except ValueError:
            return remote.split("://", 1)[0] + "://"
        user = parsed.username if parsed.scheme in ("ssh", "git+ssh", "ssh+git") else None
        netloc = f"{user}@{host}" if user else host
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    match = re.match(r"^([^@/\s]+)@([^:/\s]+):(.*)$", remote)
    if match:
        user = match.group(1).split(":", 1)[0]
        return f"{user}@{match.group(2)}:{match.group(3)}"
    return remote


def display_remote(remote: str | None) -> str | None:
    """A browsable URL when the remote has one, else the remote without credentials."""

    return web_url(remote) or strip_credentials(remote)


def commit_url(remote: str | None, commit: str | None) -> str | None:
    base = web_url(remote)
    if not base or not commit:
        return None
    host = urllib.parse.urlsplit(base).hostname or ""
    if host in ("github.com", "gitlab.com", "codeberg.org") or host.endswith(".github.com"):
        return f"{base}/commit/{commit}"
    return None


def _valid_timestamp(timestamp: Any) -> bool:
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        return False
    try:
        datetime.fromtimestamp(timestamp, UTC)
        return True
    except (OverflowError, OSError, ValueError):
        return False


def iso(timestamp: float | None) -> str | None:
    if not timestamp or not _valid_timestamp(timestamp):
        return None
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


# endregion

# region Git


def git_executable() -> str:
    return os.environ.get("GIT", "git")


def git_environment() -> dict[str, str]:
    """The environment of every git command: never ask for anything.

    No credential prompt (``GIT_TERMINAL_PROMPT``, Git Credential Manager),
    no ssh askpass, ssh in batch mode (no password or host-key question).
    A ``GIT_SSH_COMMAND`` of the user's keeps its options."""

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env["SSH_ASKPASS_REQUIRE"] = "never"
    if not env.get("GIT_SSH"):
        ssh = env.get("GIT_SSH_COMMAND") or "ssh"
        if "batchmode" not in ssh.lower():
            env["GIT_SSH_COMMAND"] = f"{ssh} -o BatchMode=yes"
    return env


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill git and what it started (ssh, remote helpers): they hold its pipes."""

    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, timeout=10, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    with contextlib.suppress(OSError):
        process.kill()


def _reason(output: str) -> str:
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    return " ".join(lines[-3:])[:500] or "no output"


def run_git(args: list[str], *, cwd: str | os.PathLike | None = None, timeout: float = GIT_TIMEOUT, extra_env: dict[str, str] | None = None) -> str:
    """Run ``git <args>`` without a terminal; its stdout, or :class:`GitError`
    (502) when it fails, cannot start or runs longer than ``timeout``."""

    command = next((arg for arg in args if not arg.startswith("-")), "git")
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        options["start_new_session"] = True  # no controlling terminal: ssh cannot prompt
    try:
        process = subprocess.Popen(
            [git_executable(), *args],
            cwd=cwd,
            env={**git_environment(), **(extra_env or {})},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **options,
        )
    except OSError as error:
        raise GitError("git could not be started: {{reason}}", reason=error) from error
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        with contextlib.suppress(subprocess.TimeoutExpired, OSError, ValueError):
            process.communicate(timeout=5)
        raise GitError("git {{command}} timed out after {{seconds}} s", command=command, seconds=int(timeout)) from None
    if process.returncode != 0:
        raise GitError("git {{command}} failed: {{reason}}", command=command, reason=_reason(stderr or stdout))
    return stdout


_repo_locks: dict[str, threading.Lock] = {}
_repo_locks_guard = threading.Lock()


@contextlib.contextmanager
def repo_lock(path: str | os.PathLike) -> Iterator[None]:
    """One git operation per repository at a time (a check never races an update)."""

    key = os.path.realpath(path)
    with _repo_locks_guard:
        held = _repo_locks.setdefault(key, threading.Lock())
    with held:
        yield


def _remote_name(path: str | os.PathLike) -> str:
    names = run_git(["remote"], cwd=path, timeout=GIT_LOCAL_TIMEOUT).split()
    if not names:
        raise GitError("the repository has no remote")
    return "origin" if "origin" in names else names[0]


def _tracking_ref(remote: str, branch: str | None) -> str:
    return f"{remote}/{branch}" if branch else f"{remote}/HEAD"


def check_repo(path: str | os.PathLike, branch: str | None) -> tuple[bool, str]:
    """Fetch a repository's remote and compare: ``(can_update, status)``,
    status ``"latest"``, ``"new commits"`` (the remote is ahead) or
    ``"behind HEAD"`` (local commits the remote does not have; an update
    resets to the remote)."""

    remote = _remote_name(path)
    run_git(["fetch", "--quiet", remote], cwd=path)
    target = run_git(["rev-parse", "--verify", f"{_tracking_ref(remote, branch)}^{{commit}}"], cwd=path, timeout=GIT_LOCAL_TIMEOUT).strip()
    head = run_git(["rev-parse", "HEAD"], cwd=path, timeout=GIT_LOCAL_TIMEOUT).strip()
    if head == target:
        return False, "latest"
    behind = run_git(["rev-list", "--count", f"HEAD..{target}"], cwd=path, timeout=GIT_LOCAL_TIMEOUT).strip()
    if behind.isdigit() and int(behind) > 0:
        return True, "new commits"
    return True, "behind HEAD"


def reset_repo(path: str | os.PathLike, branch: str | None, commit: str | None = None) -> None:
    """Fetch every remote, then reset the checkout to ``commit`` (default: the
    remote's tip of ``branch``)."""

    remote = _remote_name(path)
    run_git(["fetch", "--all", "--quiet"], cwd=path)
    run_git(["reset", "--hard", commit or _tracking_ref(remote, branch)], cwd=path, timeout=GIT_LOCAL_TIMEOUT)


# endregion

# region Installed extensions


def all_extensions() -> list[Any]:
    from modules import extensions

    return list(extensions.extensions)


def find(name: str) -> Any:
    for ext in all_extensions():
        if ext.name == name:
            return ext
    raise NotFound("no extension {{name}}", name=name)


def info(ext: Any) -> dict[str, Any]:
    """One extension as JSON (reads its git metadata once)."""

    if not ext.have_info_from_repo:
        ext.read_info_from_repo()
    commit = getattr(ext, "commit_hash", "") or None
    return {
        "name": ext.name,
        "enabled": bool(ext.enabled),
        "builtin": bool(ext.is_builtin),
        "remote": display_remote(ext.remote),
        "remote_url": web_url(ext.remote),
        "branch": ext.branch,
        "version": ext.version or None,
        "commit_hash": commit,
        "commit_url": commit_url(ext.remote, commit),
        "commit_date": iso(ext.commit_date),
        "status": ext.status or None,
        "can_update": bool(ext.can_update),
    }


def disabled_reason() -> str | None:
    """Why extensions are not loaded at all (a command-line flag or the option)."""

    from modules import shared

    if shared.cmd_opts.disable_all_extensions:
        return "--disable-all-extensions"
    if _disable_all() == "all":
        return "all"
    if shared.cmd_opts.disable_extra_extensions:
        return "--disable-extra-extensions"
    if _disable_all() == "extra":
        return "extra"
    return None


def _disable_all() -> str:
    """The "disable extensions" setting: "none", "extra" or "all"."""

    from remi.setting_values import setting

    value = setting("disable_all_extensions", "none")
    return value if value in ("none", "extra", "all") else "none"


def listing() -> dict[str, Any]:
    from modules import restart

    return {
        "items": [info(ext) for ext in all_extensions()],
        "disable_all": _disable_all(),
        "access": access_allowed(),
        "restartable": restart.is_restartable(),
        "disabled_by": disabled_reason(),
        "index_url": default_index_url(),
    }


def disabled_extensions() -> list[str]:
    """The names of the disabled extensions: a list of names whatever the
    setting holds (5, null or a name as a string after an /sdapi/v1/options
    write failed the extensions page, or matched names by substring)."""

    from remi.setting_values import setting

    return [name for name in setting("disabled_extensions", []) if isinstance(name, str)]


def _refuse_frozen(*keys: str) -> None:
    """409 when a ``--freeze-settings*`` flag keeps one of these settings as
    it is (the setter would fail an assertion after the checks passed)."""

    from modules import shared
    from remi.setting_values import frozen_reason

    for key in keys:
        if (reason := frozen_reason(shared.opts, key, shared.cmd_opts)) is not None:
            raise AdminError(reason, status=409)


def configure(disabled: Iterable[str] | None = None, disable_all: str | None = None, *, config_filename: str | None = None) -> None:
    """Set which extensions are disabled (the others are enabled) and the
    "disable all" mode; saved to the config file; loaded on the next reload."""

    from modules import shared

    check_access()
    _refuse_frozen(*(["disabled_extensions"] if disabled is not None else []), *(["disable_all_extensions"] if disable_all is not None else []))
    known = {ext.name for ext in all_extensions()}
    if disabled is not None:
        disabled = list(dict.fromkeys(disabled))
        unknown = sorted(set(disabled) - known)
        if unknown:
            raise AdminError("unknown extension(s): {{names}}", names=", ".join(unknown))
        shared.opts.disabled_extensions = disabled
    if disable_all is not None:
        if disable_all not in ("none", "extra", "all"):
            raise AdminError("disable_all must be none, extra or all, not {{value}}", value=disable_all)
        shared.opts.disable_all_extensions = disable_all
    shared.opts.save(config_filename or shared.config_filename)
    disabled_now = disabled_extensions()
    for ext in all_extensions():
        ext.enabled = ext.name not in disabled_now


def check_one(ext: Any) -> None:
    """Fetch an extension's remote and set its update status (a failure is
    the status ``"unknown (remote error)"``, never a prompt or a hang)."""

    from modules import errors

    if ext.remote is None:
        return
    with repo_lock(ext.path):
        try:
            ext.check_updates()
        except Exception as error:
            if isinstance(error, GitError):
                log.warning("Error checking updates for %s: %s", ext.name, error)
            else:
                errors.report(f"Error checking updates for {ext.name}", exc_info=True)
            ext.can_update = False
            ext.status = "unknown (remote error)"


def check_updates(names: Iterable[str] | None = None, *, on_progress: Callable[[Any], None] | None = None) -> list[Any]:
    """Check the given extensions (default: every enabled one with a remote),
    16 at a time. ``on_progress(ext)`` runs after each check."""

    check_access()
    if names is None:
        targets = [ext for ext in all_extensions() if ext.remote is not None and ext.enabled]
    else:
        targets = [find(name) for name in dict.fromkeys(names)]
    progress_lock = threading.Lock()

    def run(ext: Any) -> None:
        check_one(ext)
        if on_progress is not None:
            with progress_lock:
                on_progress(ext)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(run, targets))
    return targets


def update(names: Iterable[str]) -> None:
    """Update extensions to their remote's latest commit, after backing up
    the config state. Every one is tried; :class:`GitError` (502) names those
    that failed."""

    from modules import errors

    check_access()
    targets = [find(name) for name in dict.fromkeys(names)]
    builtin = [ext.name for ext in targets if ext.is_builtin or ext.remote is None]
    if builtin:
        raise AdminError("not updatable (built-in or not a git repository): {{names}}", names=", ".join(builtin))
    failed: list[tuple[str, str]] = []
    with lock:
        save_config_state("Backup (pre-update)")
        for ext in targets:
            try:
                with repo_lock(ext.path):
                    ext.fetch_and_reset_hard()
                ext.can_update = False
                ext.status = "latest"
                ext.read_info_from_repo()
            except Exception as error:
                if not isinstance(error, GitError):
                    errors.report(f"Error getting updates for {ext.name}", exc_info=True)
                failed.append((ext.name, str(error) or type(error).__name__))
    if failed:
        raise GitError("could not update {{names}}: {{reason}}", names=", ".join(name for name, _ in failed), reason=failed[0][1])


_installing_urls: set[str] = set()


def _rmtree(path: str) -> None:
    if not os.path.lexists(path):
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_make_writable)
    else:
        shutil.rmtree(path, onerror=_make_writable)


def run_installer(extension_dir: str, *, timeout: float = INSTALLER_TIMEOUT) -> None:
    """Run ``<extension_dir>/install.py`` bounded by ``timeout`` seconds.
    Raises :class:`AdminError` (502) if it cannot start, times out, or exits non-zero."""

    from modules import launch_utils, paths_internal

    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    python = getattr(launch_utils, "python", None) or sys.executable
    script_path = paths_internal.script_path
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{script_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PIP_NO_INPUT"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"

    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        options["start_new_session"] = True
    try:
        process = subprocess.Popen(
            [python, path_installer],
            cwd=script_path,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **options,
        )
    except OSError as error:
        raise AdminError("extension installer could not be started: {{reason}}", status=502, reason=error) from error
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        with contextlib.suppress(subprocess.TimeoutExpired, OSError, ValueError):
            process.communicate(timeout=5)
        raise AdminError("extension installer timed out after {{seconds}} s", status=502, seconds=int(timeout)) from None
    if process.returncode != 0:
        raise AdminError("extension installer failed: {{reason}}", status=502, reason=_reason(stderr or stdout))


def install(url: str, branch: str | None = None, dirname: str | None = None) -> Any:
    """Clone ``url`` into ``extensions/<dirname>`` and run its installer.
    The extension loads after the next reload."""

    from modules import extensions, paths

    check_access()
    url = (url or "").strip()
    branch = (branch or "").strip() or None
    if not valid_git_url(url):
        raise AdminError("expected a git URL (https://, ssh://, git:// or git@host:path)")
    dirname = (dirname or "").strip() or dirname_from_url(url)
    if not valid_dirname(dirname):
        raise AdminError("not a valid folder name: {{name}}", name=dirname)
    if branch is not None and not valid_branch(branch):
        raise AdminError("not a valid branch name: {{name}}", name=branch)
    norm_url = normalize_git_url(url)
    with lock:
        target = os.path.join(extensions.extensions_dir, dirname)
        if os.path.exists(target):
            raise Conflict("the folder {{name}} already exists", name=dirname)
        if norm_url in _installing_urls or any(normalize_git_url(ext.remote) == norm_url for ext in all_extensions() if ext.remote):
            raise Conflict("an extension with this URL is installed")
        _installing_urls.add(norm_url)
        temp = os.path.join(paths.data_path, "tmp", dirname)
        try:
            shutil.rmtree(temp, True)
            os.makedirs(os.path.dirname(temp), exist_ok=True)
            args = ["clone", "--filter=blob:none", "--recurse-submodules"]
            if branch:
                args += ["--branch", branch]
            run_git([*args, "--", url, temp], timeout=GIT_CLONE_TIMEOUT, extra_env={"GIT_ALLOW_PROTOCOL": INSTALL_PROTOCOLS})
            try:
                os.rename(temp, target)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.move(temp, target)  # tmp/ and extensions/ on different file systems
        except BaseException:
            _installing_urls.discard(norm_url)
            raise
        finally:
            shutil.rmtree(temp, True)

    try:
        with repo_lock(target):
            run_installer(target)
        with lock:
            extensions.list_extensions()
    except BaseException:
        with lock, repo_lock(target):
            with contextlib.suppress(OSError):
                _rmtree(target)
            extensions.list_extensions()
        raise
    finally:
        with lock:
            _installing_urls.discard(norm_url)
    return find(dirname)


def _make_writable(function: Callable[..., Any], path: str, _excinfo: Any) -> None:
    # git marks pack files read-only; Windows refuses to delete them otherwise
    os.chmod(path, stat.S_IWRITE)
    function(path)


def uninstall(name: str) -> None:
    """Delete an installed (not built-in) extension's folder. Its scripts
    stay loaded until the next reload."""

    from modules import extensions, shared
    from remi.setting_values import frozen_reason

    check_access()
    ext = find(name)
    if ext.is_builtin:
        raise AdminError("built-in extensions cannot be uninstalled; disable them instead")
    root = os.path.realpath(extensions.extensions_dir)
    path = os.path.realpath(ext.path)
    if os.path.dirname(path) != root:
        raise AdminError("the extension is not inside the extensions folder")
    with lock, repo_lock(path):
        _rmtree(path)
        extensions.extensions[:] = [item for item in extensions.extensions if item.name != name]
        # its name in the disabled list is harmless where the settings are frozen
        if name in (disabled := disabled_extensions()) and frozen_reason(shared.opts, "disabled_extensions", shared.cmd_opts) is None:
            shared.opts.disabled_extensions = [item for item in disabled if item != name]
            shared.opts.save(shared.config_filename)


# endregion

# region Extension index ("Available")


@dataclass
class _IndexCache:
    url: str
    fetched: float
    data: dict[str, Any]


_index_cache: _IndexCache | None = None


def _addresses_are_public(addresses: Iterable[str]) -> bool:
    seen = False
    for address in addresses:
        seen = True
        ip = ipaddress.ip_address(address)
        if not (getattr(ip, "ipv4_mapped", None) or ip).is_global:
            return False
    return seen


def host_is_public(host: str | None) -> bool:
    """Every address ``host`` resolves to (IPv4 and IPv6, mapped IPv4
    unwrapped) is a global internet address: no loopback, private, link-local
    or reserved target."""

    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        addresses = {info[4][0].split("%", 1)[0] for info in infos}
        return _addresses_are_public(addresses)
    except (OSError, UnicodeError, ValueError):
        return False


def check_index_url(url: str, *, public: bool) -> None:
    """An index URL is http(s); with ``public`` (a URL the client named, or
    a redirect from one) its host must be a public internet address, so the
    server cannot be made to probe its own network (``WEBUI_EXTENSIONS_INDEX``
    may name a local index)."""

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise AdminError("the index URL must be http(s)://")
    if public and not host_is_public(parsed.hostname):
        raise AdminError("the index URL must be a public internet address (a local index can be set with WEBUI_EXTENSIONS_INDEX)")


def _create_public_connection(
    address: tuple[str, int],
    timeout: float = socket._GLOBAL_DEFAULT_TIMEOUT,  # type: ignore[assignment]
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """Resolve ``address`` once, reject if any returned IP is non-global, and
    connect directly to the validated ``sockaddr`` without a second DNS lookup."""

    host, port = address
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    addresses = [info[4][0].split("%", 1)[0] for info in infos]
    try:
        is_public = _addresses_are_public(addresses)
    except ValueError:
        is_public = False
    if not is_public:
        raise AdminError("the index URL must be a public internet address (a local index can be set with WEBUI_EXTENSIONS_INDEX)")

    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in infos:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(family, socktype, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as error:
            last_error = error
            if sock is not None:
                sock.close()
    if last_error is not None:
        raise last_error
    raise OSError("getaddrinfo returns an empty list")


class _PublicHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _create_public_connection  # type: ignore[method-assign]


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _create_public_connection  # type: ignore[method-assign]


class _PublicHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(_PublicHTTPConnection, req)


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(_PublicHTTPSConnection, req, context=self._context)


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to another public http(s) URL."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        check_index_url(newurl, public=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_public_url(request: urllib.request.Request, timeout: float) -> Any:
    """``urlopen`` for client-named URLs: every redirect hop is checked with
    :func:`check_index_url`, and each socket pins the IP resolved and validated
    at connect time so DNS rebinding cannot reach loopback or private targets."""

    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _PublicRedirects,
        _PublicHTTPHandler,
        _PublicHTTPSHandler,
    ).open(request, timeout=timeout)


def fetch_index(url: str, *, opener: Callable[..., Any] | None = None, public: bool = True) -> dict[str, Any]:
    """Download and parse an extension index (``{"tags", "extensions"}``).
    ``public``: the URL came from the client, so it and every redirect must
    be public addresses (:func:`check_index_url`)."""

    check_index_url(url, public=public)
    request = urllib.request.Request(url, headers={"User-Agent": "Remi-WebUI", "Accept": "application/json"})
    try:
        with (opener or (open_public_url if public else urllib.request.urlopen))(request, timeout=INDEX_TIMEOUT) as response:
            text = response.read(INDEX_MAX_BYTES + 1)
    except AdminError:
        raise
    except Exception as error:
        raise AdminError("could not load the index: {{reason}}", status=502, reason=error) from error
    if len(text) > INDEX_MAX_BYTES:
        raise AdminError("the index is too large", status=502)
    try:
        data = json.loads(text)
    except ValueError as error:
        raise AdminError("the index is not JSON: {{reason}}", status=502, reason=error) from error
    if not isinstance(data, dict) or not isinstance(data.get("extensions"), list):
        raise AdminError("the index has no extension list", status=502)
    return data


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def index_entries(data: dict[str, Any]) -> dict[str, Any]:
    """The index normalised: entries with ``installed`` marked, and the tag
    descriptions (``installed`` added)."""

    names = {ext.name for ext in all_extensions()}
    remotes = {normalize_git_url(ext.remote) for ext in all_extensions() if ext.remote}
    tags = {str(key): _text(value) for key, value in (data.get("tags") or {}).items()} if isinstance(data.get("tags"), dict) else {}
    items = []
    for order, entry in enumerate(data["extensions"]):
        if not isinstance(entry, dict):
            continue
        url = _text(entry.get("url")).strip()
        if not url or not valid_git_url(url):
            continue
        entry_tags = [str(tag) for tag in entry.get("tags") or [] if isinstance(tag, (str, int))]
        for tag in entry_tags:
            tags.setdefault(tag, "")
        try:
            stars = int(entry.get("stars") or 0)
        except (TypeError, ValueError):
            stars = 0
        items.append(
            {
                "name": _text(entry.get("name")) or dirname_from_url(url),
                "url": url,
                "description": _text(entry.get("description")),
                "tags": entry_tags,
                "added": _text(entry.get("added")) or None,
                "created_at": _text(entry.get("created_at")) or None,
                "commit_time": _text(entry.get("commit_time")) or None,
                "stars": stars,
                "order": order,
                "installed": dirname_from_url(url) in names or normalize_git_url(url) in remotes,
            }
        )
    tags.setdefault("installed", "extensions that are already installed")
    return {"items": items, "tags": tags}


def load_index(url: str | None = None, *, refresh: bool = False, opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    """The index at ``url`` (default: ``WEBUI_EXTENSIONS_INDEX`` or the
    Forge Neo index), cached for ten minutes. Another URL the client names must
    be a public internet address (:func:`check_index_url`)."""

    global _index_cache

    check_access()  # the server fetches a URL the client names
    url = (url or "").strip() or default_index_url()
    cache = _index_cache
    if refresh or cache is None or cache.url != url or time.time() - cache.fetched > INDEX_CACHE_SECONDS:
        data = fetch_index(url, opener=opener, public=url != default_index_url())
        cache = _index_cache = _IndexCache(url, time.time(), data)
    return {"url": url, "fetched_at": iso(cache.fetched), **index_entries(cache.data)}


# endregion

# region Config states (backup / restore)


def _states_dir() -> str:
    from modules import config_states

    return config_states.config_states_dir


def _legacy_states_dir() -> str | None:
    """``config_states/`` in the installation folder, where an installation run with
    ``--data-dir`` kept its backups before they moved to the data folder; None when that
    is the data folder's (a default installation)."""

    from modules import paths_internal

    legacy = os.path.join(paths_internal.script_path, "config_states")
    return None if os.path.realpath(legacy) == os.path.realpath(_states_dir()) else legacy


def save_config_state(name: str = "", *, max_states: int = MAX_CONFIG_STATES) -> dict[str, Any]:
    """Write the current WebUI and extension commits to ``config_states/`` atomically."""

    from modules import config_states

    with lock:
        state = config_states.get_config()
        if not isinstance(state, dict) or not _valid_timestamp(state.get("created_at")):
            raise AdminError("invalid config state timestamp")
        state["name"] = os.path.basename((name or "").strip()) or "Config"
        folder = _states_dir()
        os.makedirs(folder, exist_ok=True)
        safe = re.sub(r"[^\w .()-]+", "_", state["name"])[:80]
        now = datetime.now()
        suffix = secrets.token_hex(3)
        filename = os.path.join(folder, f"{now.strftime('%Y_%m_%d-%H_%M_%S')}_{now.microsecond:06d}_{suffix}_{safe}.json")
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, prefix=".tmp_", suffix=".tmp", delete=False) as tmp:
                tmp_path = tmp.name
                json.dump(state, tmp, indent=4, ensure_ascii=False, allow_nan=False)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, filename)
            tmp_path = None
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

        if max_states > 0:
            existing = sorted(
                _read_states(folder),
                key=lambda item: (float(item["created_at"]), os.path.basename(item["filepath"])),
                reverse=True,
            )
            for stale in existing[max_states:]:
                stale_path = stale["filepath"]
                if os.path.realpath(stale_path) != os.path.realpath(filename):
                    with contextlib.suppress(OSError):
                        os.remove(stale_path)

        config_states.list_config_states()
        state["filepath"] = filename
        return state_summary(state)


def _read_states(folder: str) -> list[dict[str, Any]]:
    """The backups in ``folder`` (files that are no backup skipped), each with its ``filepath``."""

    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    states = []
    for name in names:
        path = os.path.join(folder, name)
        if not name.endswith(".json") or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as file:
                state = json.load(file)
        except (OSError, ValueError):
            continue
        created = state.get("created_at") if isinstance(state, dict) else None
        if _valid_timestamp(created):
            states.append({**state, "filepath": path})
    return states


def _load_states() -> dict[str, dict[str, Any]]:
    """State file name → state (newest first; unreadable files skipped): the data folder's
    backups, and those an older version left in the installation folder (read only; the data
    folder's file wins a name both hold)."""

    from modules import config_states

    config_states.list_config_states()
    states = {os.path.basename(state["filepath"]): state for state in _read_states(_states_dir())}
    legacy = _legacy_states_dir()
    if legacy is None:
        return dict(sorted(states.items(), key=lambda item: item[1]["created_at"], reverse=True))
    for state in _read_states(legacy):
        states.setdefault(os.path.basename(state["filepath"]), state)
    return dict(sorted(states.items(), key=lambda item: item[1]["created_at"], reverse=True))


def _is_legacy(state: dict[str, Any]) -> bool:
    legacy = _legacy_states_dir()
    return legacy is not None and os.path.dirname(os.path.realpath(state["filepath"])) == os.path.realpath(legacy)


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    webui = state.get("webui") if isinstance(state.get("webui"), dict) else {}
    extensions = state.get("extensions") if isinstance(state.get("extensions"), dict) else {}
    return {
        "id": os.path.basename(state.get("filepath", "")),
        "name": str(state.get("name") or "Config"),
        "created_at": iso(state.get("created_at")),
        "webui_commit": (webui or {}).get("commit_hash"),
        "extensions": len(extensions or {}),
    }


def list_config_states() -> list[dict[str, Any]]:
    return [state_summary(state) for state in _load_states().values()]


def _state(state_id: str) -> dict[str, Any]:
    if not STATE_ID_RE.match(state_id or ""):
        raise NotFound("no config state {{name}}", name=state_id)
    state = _load_states().get(state_id)
    if state is None:
        raise NotFound("no config state {{name}}", name=state_id)
    return state


def _repo_entry(entry: dict[str, Any]) -> dict[str, Any]:
    # Saved states record "" where git info is missing (built-ins outside a
    # checkout) while the live extension reports None: normalise both to None
    # so a fresh backup compares equal to the current state.
    remote = entry.get("remote") or None
    commit = entry.get("commit_hash") or None
    return {
        "remote": display_remote(remote),
        "branch": entry.get("branch") or None,
        "commit_hash": commit,
        "commit_url": commit_url(remote, commit),
        "commit_date": iso(entry.get("commit_date")),
    }


def _current_entry(ext: Any) -> dict[str, Any]:
    if not ext.have_info_from_repo:
        ext.read_info_from_repo()
    return {**_repo_entry({"remote": ext.remote, "branch": ext.branch, "commit_hash": ext.commit_hash, "commit_date": ext.commit_date}), "enabled": bool(ext.enabled)}


def config_state_detail(state_id: str) -> dict[str, Any]:
    """A backup next to the current state: each extension's saved and
    current commit, branch, remote and enablement."""

    from modules import config_states

    state = _state(state_id)
    current_webui = config_states.get_webui_config()
    raw_saved_webui = state.get("webui")
    saved_webui = raw_saved_webui if isinstance(raw_saved_webui, dict) else {}
    current = {ext.name: ext for ext in all_extensions()}
    rows = []
    raw_saved_exts = state.get("extensions")
    saved_exts = raw_saved_exts if isinstance(raw_saved_exts, dict) else {}
    for name, entry in sorted(saved_exts.items(), key=lambda item: item[0].lower()):
        if not isinstance(entry, dict):
            continue
        ext = current.get(name)
        rows.append({"name": name, "builtin": bool(entry.get("is_builtin")), "saved": {**_repo_entry(entry), "enabled": bool(entry.get("enabled"))}, "current": _current_entry(ext) if ext is not None else None})
    for name, ext in current.items():
        if name not in saved_exts and not ext.is_builtin:
            rows.append({"name": name, "builtin": False, "saved": None, "current": _current_entry(ext)})
    return {
        **state_summary(state),
        "webui": {"saved": _repo_entry(saved_webui or {}), "current": _repo_entry(current_webui)},
        "items": rows,
    }


def delete_config_state(state_id: str) -> None:
    check_access()
    state = _state(state_id)
    if _is_legacy(state):
        raise AdminError("the backup {{name}} is in the installation folder's config_states/, which is only read: move it to the data folder to delete it here", name=state_id)
    with lock:
        os.remove(state["filepath"])


def restore_config_state(state_id: str, what: str = "extensions") -> dict[str, Any]:
    """Restore the extensions of a backup: every extension is reset to its
    recorded commit (and enablement) while the next "Reload UI" runs
    (``restore_config_state_file``). Returns ``{"reload": True}``.

    The WebUI checkout is never restored (see the module docstring)."""

    from modules import shared

    check_access()
    if what not in RESTORE_TARGETS:
        raise AdminError("only the extensions of a backup can be restored; the WebUI checkout is never reset from here")
    _refuse_frozen("restore_config_state_file")
    state = _state(state_id)
    with lock:
        shared.opts.restore_config_state_file = state["filepath"]
        shared.opts.save(shared.config_filename)
    return {"reload": True}


# endregion

# region Diagnostics


def _disabled_scripts_by_extension() -> dict[str, list[dict[str, Any]]]:
    by_ext: dict[str, list[dict[str, Any]]] = {}
    try:
        from remi.ui.manifest import not_ported

        entries = not_ported()
    except Exception:
        entries = []
    for entry in entries:
        ext_name = entry.get("extension")
        if ext_name:
            by_ext.setdefault(str(ext_name), []).append(
                {
                    "kind": entry.get("kind", "script"),
                    "id": entry.get("id", ""),
                    "title": entry.get("title", ""),
                    "tab": entry.get("tab"),
                    "reason": entry.get("reason"),
                }
            )
    return by_ext


def _script_to_extension_map(exts: Iterable[Any]) -> dict[str, str]:
    """Map script filenames (e.g. ``controlnet.py``) to extension folder names."""

    from remi.ui.manifest import extension_of

    mapping: dict[str, str] = {}
    for ext in exts:
        ext_name = str(ext.name)
        ext_path = getattr(ext, "path", "") or ""
        if ext_path and os.path.isdir(ext_path):
            scripts_dir = os.path.join(ext_path, "scripts")
            if os.path.isdir(scripts_dir):
                try:
                    for fname in os.listdir(scripts_dir):
                        if fname.lower().endswith(".py") and os.path.isfile(os.path.join(scripts_dir, fname)):
                            mapping.setdefault(fname, ext_name)
                except OSError:
                    pass
    try:
        from modules import scripts

        for entry in (*getattr(scripts, "scripts_data", ()), *getattr(scripts, "postprocessing_scripts_data", ())):
            path = getattr(entry, "path", None)
            if isinstance(path, str) and path and (ext_name := extension_of(path)):
                mapping[os.path.basename(path)] = ext_name
        for scriptfile, _reason in getattr(scripts, "scripts_not_loaded", ()):
            path = getattr(scriptfile, "path", None)
            if isinstance(path, str) and path and (ext_name := extension_of(path)):
                fname = getattr(scriptfile, "filename", None) or os.path.basename(path)
                mapping[str(fname)] = ext_name
    except Exception:
        pass
    return mapping


def _startup_load_times_ms(extensions_list: Iterable[Any] | None = None) -> dict[str, float]:
    """Read per-extension script load times from ``modules.timer`` when recorded."""

    times: dict[str, float] = {}
    try:
        from modules import timer

        records = getattr(timer.startup_timer, "records", None) or {}
        if not isinstance(records, dict):
            return times
        exts = list(extensions_list) if extensions_list is not None else all_extensions()
        ext_names = {str(ext.name) for ext in exts}
        file_to_ext = _script_to_extension_map(exts)

        for key, value in records.items():
            if not isinstance(key, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if key.startswith("load scripts/"):
                script_key = key[len("load scripts/") :]
            elif key.startswith("scripts/"):
                script_key = key[len("scripts/") :]
            else:
                continue
            ext_name = file_to_ext.get(script_key) or file_to_ext.get(os.path.basename(script_key))
            if ext_name is None and script_key in ext_names:
                ext_name = script_key
            if ext_name is not None:
                times[ext_name] = round(times.get(ext_name, 0.0) + max(0.0, float(value)) * 1000.0, 2)

        scripts_rec = records.get("scripts") or records.get("load scripts")
        if isinstance(scripts_rec, dict):
            for name, value in scripts_rec.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    ext_name = file_to_ext.get(str(name), str(name))
                    times[ext_name] = round(max(0.0, float(value)) * 1000.0, 2)
    except Exception:
        pass
    return times


def record_startup_load_times(extensions_list: Iterable[Any] | None = None) -> dict[str, float]:
    """Populate :data:`remi.ui.manifest.diagnostics` with startup script load times."""

    from remi.ui.manifest import diagnostics as collector

    times = _startup_load_times_ms(extensions_list=extensions_list)
    for ext_name, duration_ms in times.items():
        collector.record_load_time(ext_name, duration_ms)
    return times


def diagnostics_report(
    *,
    extensions_list: Iterable[Any] | None = None,
    known_widgets: Iterable[str] | None = None,
    known_capabilities: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Collect per-extension health, manifest status, load times and error counts.
    A manifest's required widgets and capabilities are checked against
    ``known_widgets`` / ``known_capabilities`` when given."""

    from remi.ui.manifest import ManifestValidationResult, load_manifest
    from remi.ui.manifest import diagnostics as collector
    from remi.ui.schema import SCHEMA_VERSION
    from remi.ui.sentence import sentence_fields
    from remi.ui.version import REMI_API_VERSION

    exts = list(extensions_list) if extensions_list is not None else all_extensions()
    disabled_map = _disabled_scripts_by_extension()
    startup_times = _startup_load_times_ms(extensions_list=exts)
    for ext_name, duration_ms in startup_times.items():
        existing = collector.get(ext_name)
        if existing is None or existing.load_time_ms is None:
            collector.record_load_time(ext_name, duration_ms)
    recorded = collector.snapshot()
    widgets = None if known_widgets is None else frozenset(known_widgets)
    capabilities = None if known_capabilities is None else frozenset(known_capabilities)

    items: list[dict[str, Any]] = []
    summary = {"total": 0, "healthy": 0, "degraded": 0, "error": 0, "disabled": 0, "with_manifest": 0}

    for ext in exts:
        name = str(ext.name)
        enabled = bool(getattr(ext, "enabled", True))
        builtin = bool(getattr(ext, "is_builtin", False))
        path = getattr(ext, "path", "") or ""
        if not path or not os.path.isdir(path):
            mres = ManifestValidationResult(None, "missing", (), (), True)
        else:
            mres = load_manifest(path, known_widgets=widgets, known_capabilities=capabilities)

        rec = recorded.get(name)
        load_time_ms = rec.load_time_ms if rec and rec.load_time_ms is not None else startup_times.get(name)
        hook_warnings = rec.hook_warnings if rec else 0
        hook_errors = rec.hook_errors if rec else 0
        action_errors = rec.action_errors if rec else 0
        action_timeouts = rec.action_timeouts if rec else 0
        last_error = rec.last_error if rec else None
        last_error_at = iso(rec.last_error_at) if rec and rec.last_error_at else None
        recent_errors = [
            {
                "kind": str(entry.get("kind") or "error"),
                "message": str(entry.get("message") or ""),
                "timestamp": iso(entry["timestamp"]) if isinstance(entry.get("timestamp"), (int, float)) else (entry.get("timestamp") if isinstance(entry.get("timestamp"), str) else None),
            }
            for entry in (rec.recent_errors if rec else ())
        ]
        disabled_scripts = disabled_map.get(name, [])

        if not enabled:
            health = "disabled"
        elif mres.status in ("invalid", "incompatible") or hook_errors > 0 or action_errors > 0 or action_timeouts > 0:
            health = "error"
        elif hook_warnings > 0 or disabled_scripts or mres.warnings:
            health = "degraded"
        else:
            health = "healthy"

        summary["total"] += 1
        summary[health] += 1
        if mres.manifest is not None:
            summary["with_manifest"] += 1

        items.append(
            {
                "name": name,
                "enabled": enabled,
                "builtin": builtin,
                "health": health,
                "load_time_ms": load_time_ms,
                "hook_warnings": hook_warnings,
                "hook_errors": hook_errors,
                "action_errors": action_errors,
                "action_timeouts": action_timeouts,
                "last_error": last_error,
                "last_error_at": last_error_at,
                "recent_errors": recent_errors,
                "manifest_status": mres.status,
                "manifest": mres.manifest.to_dict() if mres.manifest is not None else None,
                "manifest_errors": list(mres.errors),
                "manifest_error_details": [{"message": str(error), **sentence_fields(error)} for error in mres.errors],
                "manifest_warnings": list(mres.warnings),
                "disabled_scripts": disabled_scripts,
            }
        )

    return {
        "api_version": REMI_API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "items": items,
    }


# endregion
