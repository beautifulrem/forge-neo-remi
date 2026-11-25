"""Startup: scripts that cannot be loaded, launch helpers, the SPA build
step, model selection through PATCH /options and /sdapi on the same app."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules import scripts
from remi.server import runtime
from remi.server.testing import server_app
from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]


class LegacyScript(scripts.Script):
    """An A1111 script: Gradio controls in ui(), no params()."""

    def title(self):
        return "Legacy thing"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        raise AssertionError("ui() is never called")

    def process(self, p, on, amount):
        raise AssertionError("a disabled script must not run")


# region Scripts that cannot be loaded


def test_ui_only_scripts_are_disabled_and_listed(capsys):
    with webui_state(script_classes=[LegacyScript]) as state:
        assert "legacy_thing" not in state.txt2img.by_id and "legacy_thing" not in state.img2img.by_id
        assert "Legacy thing" not in state.txt2img.titles
        assert "declares ui() (Gradio controls) instead of params()" in capsys.readouterr().out
        found = [item for item in runtime.not_ported() if item["title"] == "Legacy thing"]
        assert {item["tab"] for item in found} == {"txt2img", "img2img"} and all(item["status"] == "disabled" and item["reason"] == "declares ui() instead of params()" for item in found)
        # the generation forms still validate and build their arguments
        from remi.core.tabs import processing_args

        assert processing_args("txt2img", {}, {})[1][0] == 0


def test_script_files_that_fail_to_import_are_listed(monkeypatch):
    with webui_state():
        folder = ROOT / "extensions" / "old-tab"
        broken = scripts.ScriptFile(str(folder), "tab.py", str(folder / "scripts" / "tab.py"))
        monkeypatch.setattr(scripts, "scripts_not_loaded", [(broken, "needs the module 'missing_toolkit', which is not installed")])
        assert {"kind": "script", "id": "tab.py", "title": "tab.py", "extension": "old-tab", "status": "disabled", "reason": "needs the module 'missing_toolkit', which is not installed"} in runtime.not_ported()


def test_a_script_that_imports_a_missing_module_is_skipped(tmp_path, monkeypatch):
    script = tmp_path / "scripts" / "needs_toolkit.py"
    script.parent.mkdir()
    script.write_text("import missing_toolkit_for_test\n")
    monkeypatch.setattr(scripts, "list_scripts", lambda dirname, ext, include_extensions=True: [scripts.ScriptFile(str(tmp_path), script.name, str(script))] if dirname == "scripts" else [])
    saved = (scripts.scripts_txt2img, scripts.scripts_img2img, scripts.scripts_postproc, list(scripts.scripts_data), list(scripts.postprocessing_scripts_data))
    try:
        scripts.load_scripts()
        assert [(item.filename, reason) for item, reason in scripts.scripts_not_loaded] == [("needs_toolkit.py", "needs the module 'missing_toolkit_for_test', which is not installed")]
    finally:
        scripts.scripts_txt2img, scripts.scripts_img2img, scripts.scripts_postproc = saved[:3]
        scripts.scripts_data[:], scripts.postprocessing_scripts_data[:] = saved[3], saved[4]
        scripts.scripts_not_loaded.clear()


# endregion

# region Startup


def run_python(code: str, timeout: int = 300) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "modules_forge" / "packages")])}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


GRADIO = ("gradio", "gradio_client", "gradio_rangeslider")


def test_the_webui_starts_and_generates(tmp_path):
    # Startup (forge init, imports, scripts), the app with /sdapi and every
    # module a job runs, in a fresh interpreter (the suites also run in an
    # environment with nothing but requirements.txt installed). Gradio is
    # made unimportable first (as if it were not installed, whatever this
    # interpreter has): nothing on the startup path, in remi/ or in a built-in
    # extension may need it. Startup makes no outbound connection without a
    # user action: the data folder is empty and no opt-out flag is given (the
    # stock ADetailer detectors are downloaded on first use, not at load), so
    # any connection to a non-local host fails the test (proxy settings, from
    # the environment or the system, are dropped, so a download cannot pass
    # as a local connection to a proxy).
    code = f"""
import os, socket, sys, traceback, urllib.request
for key in [key for key in os.environ if key.lower().endswith("_proxy")]:
    del os.environ[key]
urllib.request.getproxies = dict
for name in {GRADIO!r}:
    sys.modules[name] = None  # "import gradio" raises ImportError
remote = []
_connect, _getaddrinfo = socket.socket.connect, socket.getaddrinfo
LOCAL = ("127.0.0.1", "localhost", "::1", None)
def connect(self, address):
    if isinstance(address, tuple) and address[0] not in LOCAL:
        remote.append((address, traceback.format_stack(limit=8)))
        raise OSError("no network in this test")
    return _connect(self, address)
def getaddrinfo(host, *args, **kwargs):
    if host not in LOCAL:
        remote.append((host, traceback.format_stack(limit=8)))
        raise socket.gaierror("no network in this test")
    return _getaddrinfo(host, *args, **kwargs)
socket.socket.connect, socket.getaddrinfo = connect, getaddrinfo
sys.argv = ["launch.py", "--skip-prepare-environment", "--skip-version-check", "--skip-torch-cuda-test", "--data-dir", {str(tmp_path)!r}]
import webui
from modules import scripts
from remi.jobs import run_inline
from remi.server import Services, create_app, runtime
from remi.server.auth import Auth
from modules import shared
# a setting of the wrong kind (hand-edited config.json) is repaired before the app reads it
shared.opts.data.update(ui_tab_order=5, remi_upload_max_mb="x")
runtime.prepare()
assert shared.opts.data["ui_tab_order"] == [] and shared.opts.data["remi_upload_max_mb"] == 64, "settings not repaired"
for name in ("modules.txt2img", "modules.img2img", "modules.postprocessing", "modules.extras", "modules.api.api", "remi.jobs.runner", "remi.server.gallery", "remi.server.routes.extensions"):
    __import__(name)
app = create_app(services=Services.create({str(tmp_path / "remi")!r}, dispatch=run_inline), auth=Auth(), sdapi=True, start_services=False)
assert any(getattr(route, "path", "") == "/sdapi/v1/txt2img" for route in app.routes)
# every built-in script and Extras operation loads
assert runtime.not_ported() == [], runtime.not_ported()
# every module of the remi package imports
import importlib, pkgutil, remi
for module in pkgutil.walk_packages(remi.__path__, "remi."):
    importlib.import_module(module.name)
loaded = sorted(name for name, module in sys.modules.items() if name.split(".")[0] in {GRADIO!r} and module is not None)
assert not loaded, loaded
assert not remote, "".join(f"{{host}}\\n{{''.join(stack)}}" for host, stack in remote)
print("ok")
"""
    result = run_python(code)
    assert result.returncode == 0 and "ok" in result.stdout, result.stdout[-3000:] + result.stderr[-6000:]
    assert "Error loading script" not in result.stdout + result.stderr


def test_the_server_path_starts_alone():
    # The API app itself (queue, files, networks) starts without the WebUI runtime
    code = """
import sys, tempfile
sys.argv = sys.argv[:1]
from remi.jobs import run_inline
from remi.server import Services, create_app
from remi.server.auth import Auth
import remi.networks, remi.realtime, remi.files, remi.server.launch, remi.server.runtime
services = Services.create(tempfile.mkdtemp(), dispatch=run_inline)
app = create_app(services=services, auth=Auth(), sdapi=False, start_services=False)
from fastapi.testclient import TestClient
with TestClient(app, base_url="http://127.0.0.1:7860") as client:
    assert client.get("/api/v2/queue").status_code == 200
for name in ("modules.progress", "modules.initialize_util", "modules.script_callbacks", "modules.call_queue", "modules.cmd_args"):
    __import__(name)
print("ok")
"""
    result = run_python(code)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stdout[-2000:] + result.stderr[-4000:]


# endregion

# region Launch helpers


def test_bind_host_and_ports(monkeypatch):
    import socket
    from types import SimpleNamespace

    from remi.server import launch

    assert launch.bind_host(SimpleNamespace(server_name=None, listen=False)) == "127.0.0.1"
    assert launch.bind_host(SimpleNamespace(server_name=None, listen=True)) == "0.0.0.0"
    assert launch.bind_host(SimpleNamespace(server_name="me.local", listen=True)) == "me.local"
    assert launch.root_path(SimpleNamespace(subpath="/remi/")) == "/remi" and launch.root_path(SimpleNamespace(subpath=None)) == ""

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert not launch.port_free("127.0.0.1", port)
        assert not launch.port_free("0.0.0.0", port)  # the wildcard bind would succeed next to it (SO_REUSEADDR)
        assert launch.pick_port("127.0.0.1", None, port) != port
        assert launch.pick_port("127.0.0.1", 1234, port) == 1234


def test_the_server_thread_serves_and_stops(webui_tmp):
    import httpx

    from remi.server import launch

    with server_app(webui_tmp) as server:
        port = launch.pick_port("127.0.0.1", None, 18760)
        thread = launch.ServerThread(server.app, host="127.0.0.1", port=port, root="/sub")
        thread.start()
        try:
            assert thread.url == f"http://127.0.0.1:{port}/sub/"
            assert launch.ServerThread(server.app, host="::1", port=port).url == f"http://[::1]:{port}/"
            with httpx.Client(trust_env=False, timeout=5) as client:  # no proxy for loopback
                # behind a proxy that keeps the prefix, and one that strips it
                assert set(client.get(f"http://127.0.0.1:{port}/sub/internal/ping").json()) == {"instance"}
                assert set(client.get(f"http://127.0.0.1:{port}/internal/ping").json()) == {"instance"}
                assert client.get(f"http://127.0.0.1:{port}/sub/api/v2/queue").status_code == 200
        finally:
            thread.stop()
        assert not thread.thread.is_alive()


def test_websockets_are_not_deflated():
    """The WebSocket's big frames are JPEG previews: the server declines
    permessage-deflate (a browser always asks for it), so neither side spends
    CPU per preview compressing incompressible bytes during generation."""

    import socket

    from fastapi import FastAPI, WebSocket

    from remi.server import launch

    app = FastAPI()

    @app.websocket("/ws")
    async def ws(socket_: WebSocket):
        await socket_.accept()
        await socket_.send_bytes(b"\xff\xd8" + bytes(range(256)) * 64)
        await socket_.close()

    port = launch.pick_port("127.0.0.1", None, 18800)
    server = launch.ServerThread(app, host="127.0.0.1", port=port, graceful_shutdown=0.5)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
            connection.sendall(
                b"GET /ws HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n"
                b"Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits\r\n\r\n"
            )
            head = b""
            while b"\r\n\r\n" not in head:
                head += connection.recv(4096)
        assert head.startswith(b"HTTP/1.1 101")
        assert b"sec-websocket-extensions" not in head.lower()
    finally:
        assert server.stop()


def test_stopping_does_not_wait_forever_for_a_blocked_request_and_frees_the_port():
    import threading
    import time

    import httpx
    from fastapi import FastAPI

    from remi.server import launch

    app = FastAPI()
    entered, release = threading.Event(), threading.Event()

    @app.get("/block")
    def block():  # e.g. waiting for the generation lock
        entered.set()
        release.wait(20)
        return {}

    port = launch.pick_port("127.0.0.1", None, 18790)
    server = launch.ServerThread(app, host="127.0.0.1", port=port, graceful_shutdown=0.5)
    server.start()

    def call():
        try:
            httpx.get(f"http://127.0.0.1:{port}/block", timeout=20, trust_env=False)
        except httpx.HTTPError:
            pass

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    try:
        assert entered.wait(5)
        started = time.monotonic()
        assert server.stop()
        assert time.monotonic() - started < 8
        assert launch.wait_port_free("127.0.0.1", port, 5)
        again = launch.ServerThread(app, host="127.0.0.1", port=port, graceful_shutdown=0.5)
        again.start()
        assert again.stop()
    finally:
        release.set()


def test_restart_answers_and_leaves_the_exit_to_the_serve_loop(webui_tmp, monkeypatch):
    from modules import restart, shared
    from modules.shared_state import State
    from remi.server import launch

    monkeypatch.setattr(shared, "state", State())
    monkeypatch.setattr(restart, "is_restartable", lambda: True)
    exited = []
    monkeypatch.setattr(restart, "restart_program", lambda: exited.append(True))
    with server_app(webui_tmp) as server:
        assert server.client.post("/api/v2/server/restart").json()["ok"] is True
        assert not exited
        assert shared.state.wait_for_server_command(1) == launch.RESTART_PROGRAM


def test_reload_asks_the_serve_loop_without_interrupting_the_running_job(webui_tmp, monkeypatch):
    from modules import shared
    from modules.shared_state import State

    monkeypatch.setattr(shared, "state", State())
    with server_app(webui_tmp) as server:
        assert server.client.post("/api/v2/server/reload").json()["ok"] is True
        assert shared.state.wait_for_server_command(1) == "restart"
        assert not shared.state.interrupted  # the reload waits for the job (queue.hold) instead


def test_reload_holds_a_job_that_waits_for_the_lock(webui_tmp):
    import threading
    import time

    lock = threading.Lock()
    with server_app(webui_tmp, lock=lock) as server:
        queue = server.services.queue
        with lock:
            server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "x"}})
            assert wait_for(lambda: queue.busy)
            held = threading.Event()

            def reload():
                with queue.hold():
                    held.set()

            threading.Thread(target=reload, daemon=True).start()
            time.sleep(0.2)
            assert not held.is_set()  # the claimed job runs first
        assert held.wait(5)


def wait_for(predicate, timeout=5):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# endregion

# region Extensions and user files


def test_extensions_are_listed_and_toggled_only_with_access(webui_tmp, monkeypatch):
    from modules import extensions, shared

    fake = SimpleNamespace(name="ext-one", path=str(webui_tmp), enabled=True, is_builtin=False, remote=None, branch=None, version="", commit_date=None, status="", can_update=False, have_info_from_repo=True)
    monkeypatch.setattr(extensions, "extensions", [fake])
    monkeypatch.setattr(shared.opts, "disabled_extensions", [], raising=False)
    monkeypatch.setattr(shared, "config_filename", str(webui_tmp / "config.json"))
    with server_app(webui_tmp) as server:
        listing = server.client.get("/api/v2/extensions").json()
        assert listing["items"][0]["name"] == "ext-one" and listing["items"][0]["enabled"] is True
        monkeypatch.setattr(shared.cmd_opts, "disable_extension_access", True)
        assert server.client.patch("/api/v2/extensions", json={"disabled": ["ext-one"]}).status_code == 403
        assert server.client.post("/api/v2/extensions/install", json={"url": "https://example.com/x.git"}).status_code == 403
        monkeypatch.setattr(shared.cmd_opts, "disable_extension_access", False)
        assert server.client.patch("/api/v2/extensions", json={"disabled": ["nope"]}).status_code == 422
        toggled = server.client.patch("/api/v2/extensions", json={"disabled": ["ext-one"]}).json()
        assert toggled["items"][0]["enabled"] is False and shared.opts.disabled_extensions == ["ext-one"]
        assert server.client.post("/api/v2/extensions/install", json={"url": "file:///etc"}).status_code == 422
        assert server.client.post("/api/v2/extensions/install", json={"url": "https://example.com/x.git", "dirname": "../up"}).status_code == 422


def test_user_css_and_notification_sound_are_served_when_present(webui_tmp, monkeypatch):
    from modules import paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(webui_tmp))
    monkeypatch.setattr(paths_internal, "script_path", str(webui_tmp))
    with server_app(webui_tmp) as server:
        assert server.client.get("/api/v2/user-assets/user.css").status_code == 404
        assert server.client.get("/api/v2/app").json()["user_assets"] == []
        (webui_tmp / "user.css").write_text("body{color:red}")
        (webui_tmp / "notification.mp3").write_bytes(b"ID3")
        css = server.client.get("/api/v2/user-assets/user.css")
        assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
        assert server.client.get("/api/v2/app").json()["user_assets"] == ["notification.mp3", "user.css"]
        assert server.client.get("/api/v2/user-assets/..%2Fsecret").status_code == 404


@pytest.fixture
def webui_tmp(tmp_path):
    with webui_state():
        yield tmp_path


def test_webui_build_always_uses_pnpm_and_its_lockfile_or_explains(tmp_path, capsys, monkeypatch):
    from modules import launch_utils

    tools = {"node": "/n/node", "pnpm": "/n/pnpm", "npm": "/n/npm"}
    assert launch_utils.webui_build_commands(which=tools.get) == [["/n/pnpm", "install", "--frozen-lockfile"], ["/n/pnpm", "run", "build"]]
    via_npx = launch_utils.webui_build_commands(which={"node": "n", "npm": "/n/npm", "npx": "/n/npx"}.get)
    assert via_npx == [["/n/npx", "--yes", launch_utils.PNPM_FALLBACK, "install", "--frozen-lockfile"], ["/n/npx", "--yes", launch_utils.PNPM_FALLBACK, "run", "build"]]
    assert launch_utils.webui_build_commands(which={"pnpm": "p"}.get) is None

    webui = tmp_path / "webui"
    webui.mkdir()
    (webui / "package.json").write_text("{}")
    monkeypatch.delenv("REMI_SKIP_WEBUI_BUILD", raising=False)
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)  # a machine without Node.js
    assert launch_utils.build_webui(webui) is False
    assert "Node.js was not found" in capsys.readouterr().out

    ran = []
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: [["build"]])

    def fake_run(command, cwd, shell):
        ran.append((command, cwd))
        (webui / "dist").mkdir()
        (webui / "dist" / "index.html").write_text("<html>")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch_utils.subprocess, "run", fake_run)
    assert launch_utils.build_webui(webui) is True and ran == [(["build"], webui)]
    assert launch_utils.build_webui(webui) is True and len(ran) == 1  # already built

    # an update changes the sources: the old build is rebuilt
    (webui / "src").mkdir()
    (webui / "src" / "main.ts").write_text("export {}")
    ran.clear()

    def rebuild(command, cwd, shell):
        ran.append((command, cwd))
        (webui / "dist" / "index.html").write_text("<html>new")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch_utils.subprocess, "run", rebuild)
    assert launch_utils.build_webui(webui) is True and len(ran) == 1
    assert (webui / "dist" / "index.html").read_text() == "<html>new"
    assert launch_utils.build_webui(webui) is True and len(ran) == 1
    # without Node.js an existing (maybe stale) build is still used
    (webui / "src" / "main.ts").write_text("export const changed = 1")
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: None)
    assert launch_utils.build_webui(webui) is True and "out of date" in capsys.readouterr().out


def test_the_build_stamp_covers_every_file_the_build_reads():
    """vite.config.ts imports tools/notices.mjs (the licence notices): a change there alone
    kept the old build, as the stamp hashed only WEBUI_SOURCES."""
    import re

    from modules import launch_utils

    config = (launch_utils.WEBUI_DIR / "vite.config.ts").read_text(encoding="utf-8")
    local = re.findall(r'from "\./([^"]+)"', config)
    assert "tools/notices.mjs" in local
    for path in local:
        assert any(path == name or path.startswith(name + "/") for name in launch_utils.WEBUI_SOURCES), path


@pytest.mark.parametrize(("value", "skips"), [("1", True), ("true", True), ("yes", True), ("0", False), ("false", False), ("no", False), ("off", False), ("", False)])
def test_remi_skip_webui_build_reads_as_a_flag(tmp_path, monkeypatch, value, skips):
    """``REMI_SKIP_WEBUI_BUILD=0`` skipped the build like ``1`` (any non-empty value did)."""
    from modules import launch_utils

    webui = tmp_path / "webui"
    webui.mkdir()
    (webui / "package.json").write_text("{}")
    monkeypatch.setenv("REMI_SKIP_WEBUI_BUILD", value)
    ran = []
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: [["build"]])
    monkeypatch.setattr(launch_utils, "node_version", lambda node: "v24.11.0")
    monkeypatch.setattr(shutil, "which", lambda name: "/n/" + name)

    def fake_run(command, cwd, shell):
        ran.append(command)
        (webui / "dist").mkdir(exist_ok=True)
        (webui / "dist" / "index.html").write_text("<html>")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch_utils.subprocess, "run", fake_run)
    assert launch_utils.build_webui(webui) is not skips
    assert ran == ([] if skips else [["build"]])


def test_the_node_help_says_which_node_versions_ship_corepack():
    """Node.js 25 and later no longer ship Corepack: the launcher's help and the usage guide said
    the launcher runs pnpm through Corepack or npx, "which come with Node.js"."""

    from modules import launch_utils

    docs = Path(launch_utils.script_path) / "docs" / "remi-webui"
    usage = [line for line in (docs / "usage.md").read_text(encoding="utf-8").splitlines() if line.startswith("- For the web interface:")]
    usage_zh = [line for line in (docs / "usage.zh-CN.md").read_text(encoding="utf-8").splitlines() if line.startswith("- 网页界面需要")]
    assert len(usage) == 1 and len(usage_zh) == 1
    for text in (launch_utils.NODE_HELP, usage[0]):
        assert "Corepack (bundled with Node.js 24 and earlier)" in text and "which come with Node.js" not in text, text
    assert "Node.js 24 及更早版本自带的 Corepack" in usage_zh[0] and "Node.js 自带的 Corepack" not in usage_zh[0]


_ENGINE_COMPARATOR = re.compile(r"(\^|~|>=|<=|>|<|=)?v?(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def node_range_allows(spec: str, version: tuple[int, int, int]) -> bool:
    """Whether npm's semver range ``spec`` (as ``engines.node`` writes it: ``^20.19.0 || >=22.12.0``,
    ``>= 20``, ``^6 || ^7``) allows ``version``; any other syntax fails the test."""

    def bounds(op: str, major: int, minor: str, patch: str) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        low = (major, int(minor or 0), int(patch or 0))
        if op == "^":
            return low, (major + 1, 0, 0) if major or not minor else (0, int(minor) + 1, 0)
        if op == "~" or op in ("", "="):
            if not minor:
                return low, (major + 1, 0, 0)
            if op == "~" or not patch:
                return low, (major, int(minor) + 1, 0)
            return low, (major, int(minor), int(patch) + 1)
        raise AssertionError(op)

    for alternative in spec.split("||"):
        tokens = re.sub(r"([<>=^~]+)\s+", r"\1", alternative.strip()).split()
        assert tokens, spec
        allowed = True
        for token in tokens:
            match = _ENGINE_COMPARATOR.fullmatch(token)
            assert match, f"engines range {spec!r}: unknown comparator {token!r}"
            op, major, minor, patch = match[1] or "", int(match[2]), match[3], match[4]
            point = (major, int(minor or 0), int(patch or 0))
            if op == ">=":
                allowed &= version >= point
            elif op == "<":
                allowed &= version < point
            elif op == ">":
                allowed &= version >= (major + 1, 0, 0) if not minor else version >= (major, int(minor) + 1, 0) if not patch else version > point
            elif op == "<=":
                allowed &= version < (major + 1, 0, 0) if not minor else version < (major, int(minor) + 1, 0) if not patch else version <= point
            else:
                low, high = bounds(op, major, minor, patch)
                allowed &= low <= version < high
        if allowed:
            return True
    return False


def test_the_node_range_reader_reads_engines_ranges():
    allows = node_range_allows
    assert allows("^20.19.0 || >=22.12.0", (20, 19, 0)) and not allows("^20.19.0 || >=22.12.0", (21, 0, 0)) and allows("^20.19.0 || >=22.12.0", (26, 1, 0))
    assert allows("^22.18.0 || >=24.11.0", (22, 18, 0)) and not allows("^22.18.0 || >=24.11.0", (23, 11, 0)) and not allows("^22.18.0 || >=24.11.0", (24, 10, 0))
    assert allows(">= 20", (20, 0, 0)) and not allows(">= 20", (19, 9, 0)) and allows("^6 || ^12", (12, 9, 1)) and not allows("^6 || ^12", (13, 0, 0))
    assert allows("~1.2.3", (1, 2, 9)) and not allows("~1.2.3", (1, 3, 0)) and allows(">=0.10.0", (0, 10, 4))
    with pytest.raises(AssertionError, match="unknown comparator"):
        allows("1.2.3 - 2.3.4", (2, 0, 0))


def build_tool_engines() -> dict[str, str]:
    """``engines.node`` of every package the SPA build (``tsc -b && vite build``) runs: the
    packages vite.config.ts imports and TypeScript, with everything they depend on (peers
    included), as webui/pnpm-lock.yaml pins them. Read from the lockfile, which is in the
    repository: webui/node_modules may not be installed when pytest runs."""

    import yaml

    from modules import launch_utils

    webui = launch_utils.WEBUI_DIR
    lock = yaml.safe_load((webui / "pnpm-lock.yaml").read_text(encoding="utf-8"))
    assert json.loads((webui / "package.json").read_text(encoding="utf-8"))["scripts"]["build"] == "tsc -b && vite build"
    config = (webui / "vite.config.ts").read_text(encoding="utf-8")
    roots = {name for name in re.findall(r'from "([^"]+)"', config) if not name.startswith(("node:", "."))} | {"typescript"}
    importer = lock["importers"]["."]
    pinned = {**importer.get("dependencies", {}), **importer["devDependencies"]}
    queue = [f"{name}@{pinned[name]['version']}" for name in sorted(roots)]
    seen: set[str] = set()
    engines: dict[str, str] = {}
    while queue:
        key = queue.pop()
        if key in seen:
            continue
        seen.add(key)
        package = key.split("(", 1)[0]
        node = (lock["packages"][package].get("engines") or {}).get("node")
        if node:
            engines[package] = str(node)
        snapshot = lock["snapshots"][key] or {}
        for name, version in {**snapshot.get("dependencies", {}), **snapshot.get("optionalDependencies", {})}.items():
            queue.append(f"{name}@{version}")
    assert any(package.startswith("vite@") for package in engines) and any(package.startswith("@babel/core@") for package in engines)
    return engines


def test_webui_build_states_the_node_requirement_of_its_build_tools(tmp_path, capsys, monkeypatch):
    """The launcher accepts exactly the Node.js versions every build tool's ``engines`` allows
    (Vite alone said 20.19+ or 22.12+, but the build also runs @babel/core 8, through the React
    Compiler preset: ``^22.18.0 || >=24.11.0``), and the READMEs and the usage guide name them."""

    from modules import launch_utils

    engines = build_tool_engines()
    for major in range(16, 31):
        for minor in range(31):
            supported = all(node_range_allows(spec, (major, minor, 0)) for spec in engines.values())
            problem = launch_utils.node_version_problem(f"v{major}.{minor}.0")
            assert (problem is None) == supported, (f"v{major}.{minor}.0", problem, {package: spec for package, spec in engines.items() if not node_range_allows(spec, (major, minor, 0))})
    assert launch_utils.node_version_problem("garbage") is None
    assert launch_utils.NODE_REQUIREMENT == "Node.js 22.18+ or 24.11+"
    root = Path(launch_utils.script_path)
    for name in ("README.md", "docs/remi-webui/usage.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "22.18+ or 24.11+" in text and "20.19" not in text, name
    for name in ("README.zh-CN.md", "docs/remi-webui/usage.zh-CN.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "22.18+ 或 24.11+" in text and "20.19" not in text, name

    webui = tmp_path / "webui"
    webui.mkdir()
    (webui / "package.json").write_text("{}")
    monkeypatch.delenv("REMI_SKIP_WEBUI_BUILD", raising=False)
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert launch_utils.build_webui(webui) is False
    out = capsys.readouterr().out
    assert "Node.js was not found" in out and "Node.js 22.18+ or 24.11+" in out and "pnpm is optional" in out
    assert "`pnpm install --frozen-lockfile && pnpm build`" in out
    monkeypatch.setattr(shutil, "which", lambda name: "/n/node" if name == "node" else None)
    assert launch_utils.build_webui(webui) is False
    out = capsys.readouterr().out
    assert "none of pnpm, corepack or npx" in out and "`pnpm install --frozen-lockfile && pnpm build`" in out

    ran = []
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: [["build"]])
    monkeypatch.setattr(launch_utils, "node_version", lambda node: "v20.11.1")

    def fake_run(command, cwd, shell):
        ran.append(command)
        (webui / "dist").mkdir(exist_ok=True)
        (webui / "dist" / "index.html").write_text("<html>")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch_utils.subprocess, "run", fake_run)
    assert launch_utils.build_webui(webui) is True and ran == [["build"]]
    assert "Node.js v20.11.1 cannot build the web interface: it needs Node.js 22.18+ or 24.11+" in capsys.readouterr().out


# endregion

# region Model selection and /sdapi


def test_model_settings_go_through_the_model_manager(webui_tmp, monkeypatch):
    from modules import sd_models, shared

    calls = []
    monkeypatch.setattr(runtime, "select_checkpoint", lambda name: calls.append(("checkpoint", name)) or True)
    monkeypatch.setattr(runtime, "select_modules", lambda names: calls.append(("modules", names)) or True)
    monkeypatch.setattr(runtime, "select_unet_dtype", lambda dtype: calls.append(("dtype", dtype)) or True)
    monkeypatch.setattr(runtime, "unet_dtypes", lambda: ["Automatic", "float8-e4m3fn"])
    monkeypatch.setattr(sd_models, "get_closet_checkpoint_match", lambda name: object() if name == "anima.safetensors" else None)
    monkeypatch.setattr("remi.networks.module_files", lambda kind: [("qwen_image_vae.safetensors", "/m/VAE/qwen_image_vae.safetensors", "/m/VAE")] if kind == "vae" else [("qwen.safetensors", "/m/te/qwen.safetensors", "/m/te")])
    monkeypatch.setattr(shared, "config_filename", str(webui_tmp / "config.json"))
    with server_app(webui_tmp) as server:
        body = {"sd_model_checkpoint": "anima.safetensors", "forge_additional_modules": ["/elsewhere/qwen.safetensors", "qwen_image_vae.safetensors"], "forge_unet_storage_dtype": "float8-e4m3fn"}
        assert server.client.patch("/api/v2/options", json=body).status_code == 200
        assert calls == [("checkpoint", "anima.safetensors"), ("modules", ["qwen.safetensors", "qwen_image_vae.safetensors"]), ("dtype", "float8-e4m3fn")]
        bad = server.client.patch("/api/v2/options", json={"sd_model_checkpoint": "missing", "forge_additional_modules": ["nope.bin"], "forge_unet_storage_dtype": "int3"})
        assert {error["param"] for error in bad.json()["errors"]} == {"sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype"}


def test_sdapi_is_mounted_on_the_same_app(webui_tmp):
    from fastapi.testclient import TestClient

    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    services = Services.create(webui_tmp, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860") as client:
            assert client.get("/sdapi/v1/samplers").status_code == 200
            info = client.get("/sdapi/v1/script-info").json()
            assert any(item["name"] == "sampler" for item in info)
            assert client.get("/api/v2/queue").status_code == 200
    finally:
        services.close()


def test_sdapi_refuses_text_utf8_cannot_write(webui_tmp, monkeypatch):
    """``/sdapi/v1/options`` stores what it is given and saves the settings: text with an
    unpaired surrogate stopped ``Options.save`` half way and left config.json cut short
    (the next start lost every setting). /sdapi bodies are checked as /api/v2's are."""
    from fastapi.testclient import TestClient

    from modules import shared
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    config = webui_tmp / "config.json"
    monkeypatch.setattr(shared, "config_filename", str(config))
    monkeypatch.setitem(shared.opts.data, "samples_filename_pattern", shared.opts.data.get("samples_filename_pattern", ""))
    shared.opts.save(str(config))
    before = config.read_bytes()
    services = Services.create(webui_tmp, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:
            for body in (b'{"samples_filename_pattern": "x\\udc80"}', '{"samples_filename_pattern": "x\udc80"}'.encode("utf-8", "surrogatepass")):
                response = client.post("/sdapi/v1/options", content=body, headers={"content-type": "application/json"})
                assert response.status_code == 422, response.text
                assert config.read_bytes() == before
                assert shared.opts.data["samples_filename_pattern"] != "x\udc80"
            assert client.post("/sdapi/v1/options", json={"samples_filename_pattern": "[seed]"}).status_code == 200
            assert json.loads(config.read_text(encoding="utf-8"))["samples_filename_pattern"] == "[seed]"
    finally:
        services.close()


def test_sdapi_options_checks_values_as_the_settings_page_does(webui_tmp, monkeypatch):
    """``/sdapi/v1/options`` checks a changed value as ``PATCH /api/v2/options`` does (its
    widget's kind, range and choices). It stored anything: a value of the wrong kind written
    while the server ran reached upstream readers (``ui_reorder_list``, the postprocessing
    scripts) and made the schema a 500 until the next start. A refused request sets nothing;
    a value equal to the stored one passes as it is (a client posting back what GET answered)."""
    from fastapi.testclient import TestClient

    from modules import shared
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    config = webui_tmp / "config.json"
    monkeypatch.setattr(shared, "config_filename", str(config))
    for key in ("jpeg_quality", "ui_reorder_list", "postprocessing_disable_in_extras", "enable_pnginfo"):
        monkeypatch.setitem(shared.opts.data, key, shared.opts.data.get(key, shared.opts.data_labels[key].default))
    shared.opts.save(str(config))
    before, stored = config.read_bytes(), dict(shared.opts.data)
    services = Services.create(webui_tmp, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:
            refused = [({"ui_reorder_list": 5}, "ui_reorder_list"), ({"postprocessing_disable_in_extras": "x"}, "postprocessing_disable_in_extras"), ({"jpeg_quality": "high"}, "jpeg_quality"), ({"enable_pnginfo": "maybe"}, "enable_pnginfo"), ({"jpeg_quality": 90, "ui_reorder_list": 5}, "ui_reorder_list")]
            for body, key in refused:
                response = client.post("/sdapi/v1/options", json=body)
                assert response.status_code == 422, (body, response.text)
                assert key in response.json()["detail"]
                assert config.read_bytes() == before and shared.opts.data == stored  # nothing set, nothing saved
            assert client.post("/sdapi/v1/options", json={"jpeg_quality": 90, "enable_pnginfo": "false"}).status_code == 200
            assert (shared.opts.data["jpeg_quality"], shared.opts.data["enable_pnginfo"]) == (90, False)  # stored as their kind
            assert client.post("/sdapi/v1/options", json={"jpeg_quality": 1000}).status_code == 200
            assert shared.opts.data["jpeg_quality"] == 100  # within the slider, as the settings page sets it
            # posted back unchanged, a stored value the check would refuse passes
            shared.opts.data["jpeg_quality"] = "legacy"
            assert client.post("/sdapi/v1/options", json={"jpeg_quality": "legacy", "enable_pnginfo": True}).status_code == 200
    finally:
        services.close()


def test_sdapi_options_leaves_alone_what_the_settings_page_does_not_check(webui_tmp, monkeypatch):
    """What the check leaves as upstream's ``set_config`` had it: a setting set elsewhere
    (``sd_vae``, ``interactive=False``: "None" is A1111's usual value, and refusing it lost the
    checkpoint switch posted with it), an extension option the settings page cannot read either
    (it was a 500 for the whole request) and a ``restrict_api`` setting (the API may not change
    it: ignored with a 200 as before, not a 422). A list setting without a widget (a ``Json``
    node, which checks nothing) still takes only a list: ``disabled_extensions: 5`` was stored
    and the next start failed in ``launch.list_extensions``."""
    from fastapi.testclient import TestClient

    from modules import shared
    from modules.options import OptionInfo
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    def boom():
        raise RuntimeError("an extension's component_args")

    config = webui_tmp / "config.json"
    monkeypatch.setattr(shared, "config_filename", str(config))
    monkeypatch.setitem(shared.opts.data_labels, "ext_throwing_args", OptionInfo("a", "Broken extension option", component_args=boom))
    monkeypatch.setitem(shared.opts.data_labels, "my-ext.opt", OptionInfo("a", "Extension option with a dotted name"))
    for key in ("sd_vae", "jpeg_quality", "api_forbid_local_requests", "disabled_extensions", "ext_throwing_args", "my-ext.opt"):
        monkeypatch.setitem(shared.opts.data, key, shared.opts.data.get(key, shared.opts.data_labels[key].default))
    shared.opts.save(str(config))
    services = Services.create(webui_tmp, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:
            response = client.post("/sdapi/v1/options", json={"sd_vae": "None", "jpeg_quality": 91})
            assert response.status_code == 200, response.text
            assert (shared.opts.data["sd_vae"], shared.opts.data["jpeg_quality"]) == ("None", 91)
            response = client.post("/sdapi/v1/options", json={"ext_throwing_args": "b", "my-ext.opt": "c", "jpeg_quality": 92})
            assert response.status_code == 200, response.text
            assert (shared.opts.data["ext_throwing_args"], shared.opts.data["my-ext.opt"], shared.opts.data["jpeg_quality"]) == ("b", "c", 92)
            response = client.post("/sdapi/v1/options", json={"api_forbid_local_requests": "maybe", "jpeg_quality": 93})
            assert response.status_code == 200, response.text
            assert (shared.opts.data["api_forbid_local_requests"], shared.opts.data["jpeg_quality"]) == (True, 93)
            stored = dict(shared.opts.data)
            for value in (5, "ext", [1], [["nested"]]):
                response = client.post("/sdapi/v1/options", json={"disabled_extensions": value})
                assert response.status_code == 422 and "disabled_extensions" in response.json()["detail"], (value, response.text)
                assert shared.opts.data == stored
            assert client.post("/sdapi/v1/options", json={"disabled_extensions": ["some-extension"]}).status_code == 200
            assert shared.opts.data["disabled_extensions"] == ["some-extension"]
    finally:
        services.close()


def test_a_setting_nested_too_deep_is_refused_on_both_routes(webui_tmp, monkeypatch):
    """A list nested about a thousand levels deep passed the checks (each one recursion-deep
    enough to read it) and stopped ``Options.save``'s ``json.dump`` half way, deeper in the
    stack: ``config.json`` was cut short and every setting lost at the next start. Either
    route now refuses a value nested deeper than ``MAX_NESTING`` levels before anything else,
    a setting the check otherwise leaves alone included."""
    from fastapi.testclient import TestClient

    from modules import shared
    from modules.options import OptionInfo
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth
    from remi.setting_values import MAX_NESTING

    def boom():
        raise RuntimeError("an extension's component_args")

    config = webui_tmp / "config.json"
    monkeypatch.setattr(shared, "config_filename", str(config))
    monkeypatch.setitem(shared.opts.data_labels, "ext_json", OptionInfo(None, "Extension JSON value"))
    monkeypatch.setitem(shared.opts.data_labels, "ext_throwing_args", OptionInfo("a", "Broken extension option", component_args=boom))
    for key in ("ext_json", "ext_throwing_args", "disabled_extensions"):
        monkeypatch.setitem(shared.opts.data, key, shared.opts.data_labels[key].default)
    shared.opts.save(str(config))
    before, stored = config.read_bytes(), dict(shared.opts.data)

    def nested(depth):
        value: list = []
        for _ in range(depth - 1):
            value = [value]
        return value

    services = Services.create(webui_tmp, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:
            for key in ("ext_json", "ext_throwing_args", "disabled_extensions"):
                for depth in (MAX_NESTING + 1, 700):
                    response = client.post("/sdapi/v1/options", json={key: nested(depth)})
                    assert response.status_code == 422 and key in response.json()["detail"], (key, depth, response.text)
                    if key != "ext_throwing_args":  # the settings page shows that one as an error, not a field
                        response = client.patch("/api/v2/options", json={key: nested(depth)})
                        assert response.status_code == 422, (key, depth, response.text)
                    assert config.read_bytes() == before and shared.opts.data == stored
            # the model selection, which the route takes out before the check (a dtype went straight to opts.save)
            from remi import model_selection

            picks = []
            monkeypatch.setattr(model_selection, "select", lambda key, value: picks.append(key))
            for key in model_selection.MODEL_OPTIONS:
                response = client.post("/sdapi/v1/options", json={key: nested(MAX_NESTING + 1), "jpeg_quality": 77})
                assert response.status_code == 422 and key in response.json()["detail"], (key, response.text)
                assert picks == [] and config.read_bytes() == before and shared.opts.data == stored
            assert client.post("/sdapi/v1/options", json={"ext_json": nested(MAX_NESTING)}).status_code == 200
            assert client.patch("/api/v2/options", json={"ext_json": {"a": [1, [2]]}}).status_code == 200
            assert shared.opts.data["ext_json"] == {"a": [1, [2]]}
    finally:
        services.close()


def test_sdapi_options_checks_the_model_selection_as_the_settings_page_does(webui_tmp, monkeypatch):
    """The route takes the model selection out before the value check and applied it as
    sent: a dtype of another kind (``["x"]``: ``forge_unet_storage_dtype_options.get`` raised
    TypeError at the next model load) or modules that are not a list of known names were
    stored. They are checked as ``PATCH /api/v2/options`` checks them (422, nothing set)."""
    from fastapi.testclient import TestClient

    from modules import shared
    from remi import model_selection
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    config = webui_tmp / "config.json"
    monkeypatch.setattr(shared, "config_filename", str(config))
    shared.opts.save(str(config))
    picks = []
    monkeypatch.setattr(model_selection, "select", lambda key, value: picks.append((key, value)))
    services = Services.create(webui_tmp, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:
            refused = [
                {"forge_unet_storage_dtype": ["x"]},
                {"forge_unet_storage_dtype": 5},
                {"forge_unet_storage_dtype": "no such dtype"},
                {"forge_additional_modules": "ae.safetensors"},
                {"forge_additional_modules": ["no-such-module.safetensors"]},
            ]
            for body in refused:
                response = client.post("/sdapi/v1/options", json={**body, "jpeg_quality": 66})
                assert response.status_code == 422 and next(iter(body)) in response.json()["detail"], (body, response.text)
                assert picks == [] and shared.opts.data.get("jpeg_quality") != 66
            assert client.post("/sdapi/v1/options", json={"forge_unet_storage_dtype": "Automatic", "forge_additional_modules": []}).status_code == 200
            assert sorted(picks) == [("forge_additional_modules", []), ("forge_unet_storage_dtype", "Automatic")]
            # a Windows path names the module as the check reads it (its file name): on a POSIX server
            # os.path.basename kept the whole path, matched nothing and emptied the selection
            from remi import networks

            picks.clear()
            monkeypatch.setattr(networks, "module_files", lambda kind: [("ae.safetensors", "/m/VAE/ae.safetensors", "/m/VAE")] if kind == "vae" else [])
            assert client.post("/sdapi/v1/options", json={"forge_additional_modules": ["C:\\SD\\models\\VAE\\ae.safetensors"]}).status_code == 200
            assert picks == [("forge_additional_modules", ["ae.safetensors"])]
            # posted back as GET answered, a stored pick no longer offered (a dtype of another Forge, a
            # module file since removed) passes as it did: only a changed pick is checked
            picks.clear()
            monkeypatch.setitem(shared.opts.data, "forge_unet_storage_dtype", "bnb-nf4")
            monkeypatch.setitem(shared.opts.data, "forge_additional_modules", ["/gone/VAE/missing.safetensors"])
            body = {"forge_unet_storage_dtype": "bnb-nf4", "forge_additional_modules": ["/gone/VAE/missing.safetensors"], "jpeg_quality": 67}
            response = client.post("/sdapi/v1/options", json=body)
            assert response.status_code == 200, response.text
            assert shared.opts.data["jpeg_quality"] == 67 and len(picks) == 2
            # the same modules in another order are the same selection (modules_change sorts them)
            monkeypatch.setitem(shared.opts.data, "forge_additional_modules", ["/gone/VAE/missing.safetensors", "/m/VAE/ae.safetensors"])
            body = {"forge_additional_modules": ["/m/VAE/ae.safetensors", "/gone/VAE/missing.safetensors"]}
            assert client.post("/sdapi/v1/options", json=body).status_code == 200
    finally:
        services.close()


def test_sdapi_options_names_every_refused_value_in_the_order_sent(webui_tmp, monkeypatch):
    """A refused model pick answered alone and first: a client fixing it learned of its other
    refused values one request at a time. Every refused value is named, in the request's order."""
    from fastapi.testclient import TestClient

    from modules import shared
    from remi import model_selection
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    config = webui_tmp / "config.json"
    monkeypatch.setattr(shared, "config_filename", str(config))
    shared.opts.save(str(config))
    monkeypatch.setattr(model_selection, "select", lambda key, value: None)
    services = Services.create(webui_tmp, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:
            body = {"jpeg_quality": "loud", "sd_model_checkpoint": "missing.safetensors", "samples_save": "maybe", "forge_unet_storage_dtype": 5, "forge_additional_modules": "x"}
            response = client.post("/sdapi/v1/options", json=body)
            assert response.status_code == 422, response.text
            detail = response.json()["detail"]
            assert detail.startswith("invalid settings: ")
            assert [detail.index(f"settings.{key}:") for key in body] == sorted(detail.index(f"settings.{key}:") for key in body), detail
            assert "unknown checkpoint 'missing.safetensors'" in detail
    finally:
        services.close()


def test_sdapi_override_settings_nested_too_deep_are_refused():
    """A generation's ``override_settings`` go into ``opts.data`` (``processing.py``), kept
    there with ``override_settings_restore_afterwards: false`` or while another route saves:
    one nested about a thousand levels deep cut ``config.json`` short at the next save. They
    are refused (422) before anything else of the request is read."""
    import pytest
    from fastapi import HTTPException

    from modules.api import api, models
    from remi.setting_values import MAX_NESTING

    deep: list = []
    for _ in range(MAX_NESTING):
        deep = [deep]
    requests = [
        (api.Api.text2imgapi, models.StableDiffusionTxt2ImgProcessingAPI(override_settings={"forge_checkpoint_sd": deep})),
        (api.Api.img2imgapi, models.StableDiffusionImg2ImgProcessingAPI(override_settings={"forge_checkpoint_sd": deep})),
    ]
    for handler, request in requests:
        with pytest.raises(HTTPException) as error:
            handler(None, request)  # refused before the handler uses its Api
        assert error.value.status_code == 422 and "forge_checkpoint_sd" in error.value.detail


def test_sdapi_infotext_fills_request_fields_and_script_args(webui_tmp):
    from modules.api import api, models

    with server_app(webui_tmp):
        request = models.StableDiffusionTxt2ImgProcessingAPI(infotext="a cat\nNegative prompt: dog\nSteps: 9, Sampler: Euler, CFG scale: 4, Seed: 77, Size: 640x512", cfg_scale=7)
        mentioned = {}
        api.Api.apply_infotext(None, request, "txt2img", script_runner=scripts.scripts_txt2img, mentioned_script_args=mentioned)
        assert request.prompt == "a cat" and request.negative_prompt == "dog" and request.steps == 9 and request.seed == 77
        assert (request.width, request.height) == (640, 512) and request.cfg_scale == 7  # set by the request: kept


def test_sdapi_infotext_sets_the_inpaint_choices_by_their_index(webui_tmp):
    """Mask mode, Masked content and Inpaint area are written by label; the img2img request
    takes the index (``inpaint_full_res`` is index 1, "Only masked"). Read as text, "Whole
    picture" made ``inpaint_full_res`` True and "latent noise" failed ``int()`` and was dropped,
    so the three had no effect (api.md: the same code as the SPA's paste)."""
    from modules.api import api, models

    with server_app(webui_tmp):
        text = "a cat\nSteps: 9, Size: 512x512, Mask mode: Inpaint not masked, Masked content: latent noise, Inpaint area: Whole picture, Masked area padding: 16"
        request = models.StableDiffusionImg2ImgProcessingAPI(infotext=text)
        api.Api.apply_infotext(None, request, "img2img", script_runner=scripts.scripts_img2img, mentioned_script_args={})
        assert (request.inpainting_mask_invert, request.inpainting_fill, request.inpaint_full_res, request.inpaint_full_res_padding) == (1, 2, False, 16)
        assert request.steps == 9
        # a field the request sets wins over the infotext
        request = models.StableDiffusionImg2ImgProcessingAPI(infotext=text, inpainting_fill=1)
        api.Api.apply_infotext(None, request, "img2img", script_runner=scripts.scripts_img2img, mentioned_script_args={})
        assert (request.inpainting_fill, request.inpaint_full_res) == (1, False)


# endregion
