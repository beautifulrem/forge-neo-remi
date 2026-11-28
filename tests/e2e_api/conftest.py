"""End-to-end HTTP/WebSocket API test harness for ``tests/e2e_api/``.

Every test interacts with a real FastAPI ``create_app()`` instance over live
TCP/HTTP and RFC 6455 WebSocket connections, backed by either an isolated
``ServerThread`` + ``StubExecutor`` or an isolated ``TinyAnima`` subprocess.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import itertools
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets.sync.client

_UNSET = object()
_WORKER_ENV_KEYS = ("REMI_WORKER_PROCESS", "REMI_WORKER_GPUS", "REMI_WORKER_MLX")
_PORT_CURSOR = itertools.count(18900, 10)
_SERVER_SEQ = itertools.count(1)


def _load_helpers() -> type[Any]:
    path = Path(__file__).with_name("_helpers.py")
    spec = importlib.util.spec_from_file_location("_remi_opaque_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OpaqueHelpers


OpaqueHelpers = _load_helpers()


def _clear_worker_env() -> dict[str, str | None]:
    saved = {key: os.environ.get(key) for key in _WORKER_ENV_KEYS}
    for key in _WORKER_ENV_KEYS:
        os.environ.pop(key, None)
    return saved


def _restore_worker_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _build_output_options(out_root: Path) -> dict[str, str]:
    mapping = {
        "outdir_samples": "",
        "outdir_txt2img_samples": str(out_root / "txt2img-images"),
        "outdir_img2img_samples": str(out_root / "img2img-images"),
        "outdir_extras_samples": str(out_root / "extras-images"),
        "outdir_hires_samples": str(out_root / "hires-images"),
        "outdir_videos": str(out_root / "videos"),
        "outdir_grids": "",
        "outdir_txt2img_grids": str(out_root / "txt2img-grids"),
        "outdir_img2img_grids": str(out_root / "img2img-grids"),
        "outdir_save": str(out_root / "saved"),
        "outdir_init_images": str(out_root / "init-images"),
        "temp_dir": str(out_root / "tmp"),
    }
    for value in mapping.values():
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)
    return mapping


def _pick_free_port(host: str = "127.0.0.1") -> int:
    from remi.server import launch

    start = next(_PORT_CURSOR)
    if start > 45000:
        start = 18900
    try:
        return launch.pick_port(host, None, start)
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            return int(probe.getsockname()[1])


def _merge_ws_headers(
    headers: Mapping[str, str] | Sequence[tuple[str, str]] | None,
    cookies: Mapping[str, str] | httpx.Cookies | None,
    client: httpx.Client | None,
) -> list[tuple[str, str]] | None:
    pairs: list[tuple[str, str]] = []
    if headers is not None:
        items = headers.items() if isinstance(headers, Mapping) else headers
        pairs.extend((str(k), str(v)) for k, v in items)
    has_cookie_header = any(k.lower() == "cookie" for k, _ in pairs)
    cookie_source = cookies if cookies is not None else (client.cookies if client is not None else None)
    if cookie_source is not None and not has_cookie_header:
        cookie_pairs = [f"{k}={v}" for k, v in cookie_source.items()]
        if cookie_pairs:
            pairs.append(("Cookie", "; ".join(cookie_pairs)))
    return pairs or None


@dataclass
class OpaqueServer:
    """Live HTTP + WebSocket handle for an isolated opaque-box test server."""

    base_url: str
    ws_url: str
    root: Path
    outputs_dir: Path
    config_states_dir: Path
    http: httpx.Client
    executor: Any = None
    state: Any = None
    gate: threading.Event | None = None
    services: Any = None
    _http_stack: contextlib.ExitStack = field(default_factory=contextlib.ExitStack, repr=False)
    _ws_stack: contextlib.ExitStack = field(default_factory=contextlib.ExitStack, repr=False)

    def __post_init__(self) -> None:
        self._http_stack.enter_context(self.http)

    def new_client(self, **kwargs: Any) -> httpx.Client:
        opts: dict[str, Any] = {
            "base_url": self.base_url,
            "trust_env": False,
            "timeout": 15.0,
            **kwargs,
        }
        return self._http_stack.enter_context(httpx.Client(**opts))

    def ws_connect(
        self,
        headers: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
        origin: str | object | None = _UNSET,
        *,
        cookies: Mapping[str, str] | httpx.Cookies | None = None,
        client: httpx.Client | None = None,
        path: str = "/api/v2/ws",
        **kwargs: Any,
    ) -> Any:
        url = self.ws_url if path == "/api/v2/ws" else f"ws://{self.base_url.split('://', 1)[1]}{path}"
        effective_origin: Any = self.base_url if origin is _UNSET else origin
        merged_headers = _merge_ws_headers(headers, cookies, client)
        connect_kwargs: dict[str, Any] = {
            "origin": effective_origin,
            "additional_headers": merged_headers,
            "open_timeout": 5.0,
            "close_timeout": 2.0,
            "proxy": None,
            **kwargs,
        }
        return self._ws_stack.enter_context(websockets.sync.client.connect(url, **connect_kwargs))

    def wait_job(
        self,
        job_id: str,
        *,
        timeout: float = 10.0,
        poll_interval: float = 0.02,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        http = client or self.http
        deadline = time.monotonic() + timeout
        while True:
            response = http.get(f"/api/v2/jobs/{job_id}")
            response.raise_for_status()
            data = response.json()
            if data.get("status") not in ("queued", "running"):
                return data
            if time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} stayed {data.get('status')} for >{timeout:.1f}s")
            time.sleep(poll_interval)

    def wait_chain(
        self,
        chain_id: str,
        *,
        timeout: float = 10.0,
        poll_interval: float = 0.02,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        http = client or self.http
        deadline = time.monotonic() + timeout
        while True:
            response = http.get(f"/api/v2/chains/{chain_id}")
            response.raise_for_status()
            data = response.json()
            if data.get("status") not in ("queued", "running"):
                return data
            if time.monotonic() >= deadline:
                raise TimeoutError(f"chain {chain_id} stayed {data.get('status')} for >{timeout:.1f}s")
            time.sleep(poll_interval)

    def expire_files_and_run_housekeeping(
        self,
        file_ids: Sequence[str],
        *,
        age_seconds: int = 3 * 86400,
    ) -> None:
        if self.services is None:
            raise RuntimeError("expire_files_and_run_housekeeping requires an in-process server with services")
        stale_ts = time.time() - age_seconds
        with self.services.files._lock:
            for fid in file_ids:
                row = self.services.files._db.execute("SELECT path FROM files WHERE id = ?", (fid,)).fetchone()
                if row is None:
                    raise KeyError(f"unknown file_id: {fid}")
                self.services.files._db.execute("UPDATE files SET created = ? WHERE id = ?", (stale_ts, fid))
                if os.path.isfile(row[0]):
                    os.utime(row[0], (stale_ts, stale_ts))
        self.services.housekeeping()

    def close_clients(self) -> None:
        try:
            self._ws_stack.close()
        finally:
            self._http_stack.close()


@pytest.fixture(scope="session")
def helpers() -> type[Any]:
    """Pure HTTP/WebSocket helper namespace for opaque-box tests."""

    return OpaqueHelpers


@pytest.fixture(scope="module")
def webui_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """Module-scoped isolated WebUI state with all output/temp/config-states paths redirected."""

    from modules import config_states, paths_internal
    from remi.testing import webui_state

    saved_env = _clear_worker_env()
    env_root = tmp_path_factory.mktemp("remi-opaque-env")
    outputs_dir = env_root / "outputs"
    config_states_dir = env_root / "config_states"
    data_dir = env_root / "data"
    config_states_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_cs_dir = config_states.config_states_dir
    saved_pi_cs_dir = paths_internal.config_states_dir
    saved_pi_data_path = paths_internal.data_path
    try:
        with webui_state(options=_build_output_options(outputs_dir)) as state:
            config_states.config_states_dir = str(config_states_dir)
            paths_internal.config_states_dir = str(config_states_dir)
            paths_internal.data_path = str(data_dir)
            state.root = env_root
            state.outputs_dir = outputs_dir
            state.config_states_dir = config_states_dir
            state.data_dir = data_dir
            yield state
    finally:
        config_states.config_states_dir = saved_cs_dir
        paths_internal.config_states_dir = saved_pi_cs_dir
        paths_internal.data_path = saved_pi_data_path
        _restore_worker_env(saved_env)


@pytest.fixture
def make_live_server(webui_env: Any, tmp_path: Path) -> Iterator[Callable[..., OpaqueServer]]:
    """Factory fixture booting isolated ``create_app()`` + ``ServerThread`` servers."""

    from modules import config_states, paths_internal, shared, styles
    from remi.jobs import run_inline
    from remi.server import Services, create_app, launch
    from remi.server.auth import API_AUTH_THROTTLE, Auth
    from remi.server.services import configured_upload_limit
    from remi.server.testing import FakeState, StubExecutor

    teardowns: list[Callable[[], None]] = []

    def _factory(
        *,
        count: int = 1,
        steps: int = 2,
        step_time: float = 0.01,
        gate: threading.Event | None = None,
        error: Exception | None = None,
        executor: Any = None,
        state: Any = None,
        realtime: bool = True,
        housekeeping: bool = False,
        worker_process: bool | None = False,
        worker_gpus: str | None = "",
        worker_mlx: bool | None = False,
        auth: Auth | None = None,
        sdapi: bool = True,
        webui: bool = True,
        spa_dir: Path | None = None,
        start_services: bool = True,
        max_upload_bytes: int | Callable[[], int] | None = None,
        options: dict[str, Any] | None = None,
        cmd_opts_overrides: dict[str, Any] | None = None,
        validate: Callable[..., None] | None = None,
        lock: Any = None,
        root: Path | None = None,
    ) -> OpaqueServer:
        server_root = root or (tmp_path / f"srv-{next(_SERVER_SEQ)}")
        server_root.mkdir(parents=True, exist_ok=True)
        outputs_dir = server_root / "outputs"
        config_states_dir = server_root / "config_states"
        data_dir = server_root / "data"
        config_states_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        saved_env = _clear_worker_env()
        saved_opts_data = copy.deepcopy(shared.opts.data)
        saved_config_filename = shared.config_filename
        saved_prompt_styles = shared.prompt_styles
        saved_cs_dir = config_states.config_states_dir
        saved_pi_cs_dir = paths_internal.config_states_dir
        saved_pi_data_path = paths_internal.data_path
        saved_cmd_opts: dict[str, tuple[bool, Any]] = {}
        for key, value in (cmd_opts_overrides or {}).items():
            has_attr = hasattr(shared.cmd_opts, key)
            saved_cmd_opts[key] = (has_attr, getattr(shared.cmd_opts, key, None))
            setattr(shared.cmd_opts, key, value)

        services: Services | None = None
        thread: launch.ServerThread | None = None
        server: OpaqueServer | None = None

        def _cleanup() -> None:
            try:
                if server is not None:
                    server.close_clients()
            finally:
                try:
                    if thread is not None:
                        thread.stop()
                finally:
                    try:
                        if services is not None:
                            services.close()
                    finally:
                        shared.config_filename = saved_config_filename
                        shared.opts.data.clear()
                        shared.opts.data.update(saved_opts_data)
                        shared.prompt_styles = saved_prompt_styles
                        config_states.config_states_dir = saved_cs_dir
                        paths_internal.config_states_dir = saved_pi_cs_dir
                        paths_internal.data_path = saved_pi_data_path
                        for key, (had_attr, old_val) in saved_cmd_opts.items():
                            if had_attr:
                                setattr(shared.cmd_opts, key, old_val)
                            elif hasattr(shared.cmd_opts, key):
                                delattr(shared.cmd_opts, key)
                        API_AUTH_THROTTLE._failures.clear()
                        _restore_worker_env(saved_env)

        try:
            shared.config_filename = str(server_root / "config.json")
            shared.prompt_styles = styles.StyleDatabase([str(server_root / "styles.csv")])
            config_states.config_states_dir = str(config_states_dir)
            paths_internal.config_states_dir = str(config_states_dir)
            paths_internal.data_path = str(data_dir)
            shared.opts.data.update(_build_output_options(outputs_dir))
            if options:
                shared.opts.data.update(options)

            fake_state = state if state is not None else FakeState()
            stub_executor = executor if executor is not None else StubExecutor(fake_state, count=count, steps=steps, step_time=step_time, gate=gate, error=error)
            queue_opts: dict[str, Any] = {
                "executor": stub_executor,
                "dispatch": run_inline,
                "lock": lock if lock is not None else threading.Lock(),
                "state": lambda: fake_state,
            }
            if validate is not None:
                queue_opts["validate"] = validate

            upload_limit: int | Callable[[], int] = configured_upload_limit if max_upload_bytes is None else max_upload_bytes
            services = Services.create(
                server_root / "remi-data",
                max_upload_bytes=upload_limit,
                realtime=realtime,
                housekeeping=housekeeping,
                worker_process=worker_process,
                worker_gpus=worker_gpus,
                worker_mlx=worker_mlx,
                **queue_opts,
            )
            app = create_app(
                services=services,
                auth=auth or Auth(),
                webui=webui,
                sdapi=sdapi,
                spa_dir=spa_dir or (server_root / "no-dist"),
                start_services=start_services,
            )

            subpath = launch.root_path(shared.cmd_opts)
            forwarded_allow_ips = getattr(shared.cmd_opts, "forwarded_allow_ips", None)
            last_exc: Exception | None = None
            bound_port = 0
            for _attempt in range(3):
                candidate_port = _pick_free_port("127.0.0.1")
                candidate_thread = launch.ServerThread(
                    app,
                    host="127.0.0.1",
                    port=candidate_port,
                    root=subpath,
                    graceful_shutdown=0.5,
                    forwarded_allow_ips=forwarded_allow_ips,
                )
                try:
                    candidate_thread.start(timeout=15)
                    thread = candidate_thread
                    bound_port = candidate_port
                    break
                except (RuntimeError, TimeoutError, OSError) as exc:
                    last_exc = exc
                    candidate_thread.stop(timeout=2)
            if thread is None:
                raise RuntimeError(f"failed to start ServerThread: {last_exc}") from last_exc

            base_url = f"http://127.0.0.1:{bound_port}{subpath}"
            ws_url = f"ws://127.0.0.1:{bound_port}{subpath}/api/v2/ws"
            http_client = httpx.Client(base_url=base_url, trust_env=False, timeout=15.0)
            server = OpaqueServer(
                base_url=base_url,
                ws_url=ws_url,
                root=server_root,
                outputs_dir=outputs_dir,
                config_states_dir=config_states_dir,
                http=http_client,
                executor=stub_executor,
                state=fake_state,
                gate=getattr(stub_executor, "gate", gate),
                services=services,
            )
        except BaseException:
            _cleanup()
            raise

        teardowns.append(_cleanup)
        return server

    try:
        yield _factory
    finally:
        for cleanup in reversed(teardowns):
            cleanup()


@pytest.fixture
def live_server(make_live_server: Callable[..., OpaqueServer]) -> OpaqueServer:
    """Fast default opaque-box server with realtime WS and ``/sdapi/v1`` enabled."""

    return make_live_server(steps=2, step_time=0.01, realtime=True, sdapi=True)


@pytest.fixture
def gated_server(make_live_server: Callable[..., OpaqueServer]) -> OpaqueServer:
    """Server whose ``StubExecutor`` pauses at ``server.gate`` until ``server.gate.set()``."""

    return make_live_server(steps=2, step_time=0.01, gate=threading.Event(), realtime=True, sdapi=True)


@pytest.fixture
def failing_server(make_live_server: Callable[..., OpaqueServer]) -> OpaqueServer:
    """Server whose ``StubExecutor`` raises ``RuntimeError('synthetic boom')``."""

    return make_live_server(steps=0, error=RuntimeError("synthetic boom"), realtime=True, sdapi=True)


@pytest.fixture
def auth_server(make_live_server: Callable[..., OpaqueServer]) -> OpaqueServer:
    """Authenticated server with users ``alice`` and ``bob``."""

    from remi.server.auth import Auth

    return make_live_server(
        auth=Auth({"alice": "s3cret", "bob": "pw2"}, secret=b"x" * 32),
        steps=2,
        step_time=0.01,
        realtime=True,
        sdapi=True,
    )


@pytest.fixture(scope="module")
def tiny_anima_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OpaqueServer]:
    """Module-scoped isolated subprocess running real ``TinyAnima`` on CPU with ``--api``."""

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = tmp_path_factory.mktemp("remi-tiny-anima")
    outputs_dir = data_dir / "output"
    (outputs_dir / "images").mkdir(parents=True, exist_ok=True)
    config_states_dir = data_dir / "config_states"
    config_states_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "server.log"

    env = dict(os.environ)
    for key in _WORKER_ENV_KEYS:
        env.pop(key, None)
    env["YOLO_OFFLINE"] = "1"
    env["REMI_TEST_HELPER"] = "tiny_anima_server"

    port = _pick_free_port("127.0.0.1")
    base_url = f"http://127.0.0.1:{port}"
    ws_url = f"ws://127.0.0.1:{port}/api/v2/ws"

    log_file = log_path.open("w", encoding="utf-8")
    proc: subprocess.Popen[str] | None = None
    server: OpaqueServer | None = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-W",
                "ignore",
                str(repo_root / "tools" / "tiny_anima.py"),
                "serve",
                "--data-dir",
                str(data_dir),
                "--port",
                str(port),
                "--cpu",
                "--api",
                "--skip-version-check",
                "--skip-torch-cuda-test",
            ],
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        http_client = httpx.Client(base_url=base_url, trust_env=False, timeout=60.0)
        server = OpaqueServer(
            base_url=base_url,
            ws_url=ws_url,
            root=data_dir,
            outputs_dir=outputs_dir,
            config_states_dir=config_states_dir,
            http=http_client,
        )

        deadline = time.monotonic() + 120.0
        while True:
            rc = proc.poll()
            if rc is not None:
                log_file.flush()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"tiny_anima_server exited early with code {rc}:\n{tail}")
            try:
                ping = http_client.get("/internal/ping", timeout=2.0)
                if ping.status_code == 200:
                    app_resp = http_client.get("/api/v2/app", timeout=5.0)
                    if app_resp.status_code == 200:
                        break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                log_file.flush()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise TimeoutError(f"tiny_anima_server did not become ready within 120s:\n{tail}")
            time.sleep(0.2)

        yield server
    finally:
        try:
            if server is not None:
                server.close_clients()
        finally:
            try:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
            finally:
                log_file.close()
