"""The FastAPI application of the Remi WebUI (no Gradio on this path)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import HTTPConnection
from starlette.types import Scope

from remi.files import FileError
from remi.jobs import JobError
from remi.server.api_docs import add_docs_routes
from remi.server.auth import BASE_HEADERS, PUBLIC_PATHS, Auth, SecurityMiddleware, credentials_from_cmd_opts, load_secret, route_path
from remi.server.limits import PUBLIC_BODY_LIMIT, BodyLimitMiddleware, body_limit
from remi.server.services import Services, default_services
from remi.server.spa import DIST, Spa
from remi.server.ws import Hub
from remi.setting_values import encodable
from remi.ui import ParamError, ValidationError

log = logging.getLogger(__name__)

API_PREFIX = "/api/v2"


def version() -> str:
    try:
        from modules_forge import forge_version

        return f"{forge_version.version} {forge_version.release}"
    except Exception:
        return "remi"


def auth_from_cmd_opts(root: Path) -> Auth:
    from modules.shared_cmd_options import cmd_opts

    credentials = credentials_from_cmd_opts(cmd_opts)
    secure = bool(cmd_opts.tls_keyfile and cmd_opts.tls_certfile)
    path = f"/{cmd_opts.subpath.strip('/')}" if cmd_opts.subpath else "/"
    if not credentials:
        return Auth(cookie_path=path)
    return Auth(credentials, load_secret(root / "secret.key"), secure_cookies=secure, cookie_path=path, revoked_path=root / "revoked-sessions.json")


def _phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def json_safe(value: Any) -> Any:
    """``value`` with every non-finite float (``inf``, ``nan``: what a JSON
    body like ``1e999`` parses to) as its text, so it can be encoded."""

    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class TextSafeResponse(JSONResponse):
    """The API's JSON. Text with an unpaired surrogate ("\\udc80") cannot be
    UTF-8: a setting an extension stored (``/sdapi/v1/options`` once did) and
    read back by a tab's schema or the system info failed those requests until
    the next start. Such text is written with JSON escapes."""

    def render(self, content: Any) -> bytes:
        try:
            return super().render(content)
        except UnicodeEncodeError:
            return json.dumps(content, ensure_ascii=True, allow_nan=False, indent=None, separators=(",", ":")).encode("ascii")


class ErrorResponse(TextSafeResponse):
    """The error handlers' JSON: errors echo the request's values, which may
    be non-finite numbers (strict JSON cannot encode them) or text with an
    unpaired surrogate: the plain response would fail with a 500 and a logged
    traceback instead."""

    def render(self, content: Any) -> bytes:
        return super().render(json_safe(content))


def handle_exception(request: Request, error: Exception) -> JSONResponse:
    """Unhandled errors as JSON. A server error (5xx) answers only a generic
    message and a correlation id; the exception, its message and any body it
    carries (a ``ResponseValidationError`` holds the whole response) go to
    the server log under that id, never to the client. An ``HTTPException``
    that reaches this handler (raised where Starlette's own handler does not
    run, as ``BodyTooLarge`` from a body read) keeps its status, and a 4xx
    keeps its ``detail``: it is meant for the client. A status on any other
    exception is not Remi's answer: a remote service's 401 that an extension
    let escape is a server error, not a refused session."""

    status = error.status_code if isinstance(error, StarletteHTTPException) else None
    status = status if isinstance(status, int) and 400 <= status < 600 else 500
    if status < 500:
        detail = getattr(error, "detail", None)
        return ErrorResponse(status_code=status, content={"detail": detail if isinstance(detail, str) and detail else _phrase(status)})
    error_id = secrets.token_hex(6)
    log.error("%s %s failed (error id %s)", request.method, request.url.path, error_id, exc_info=error)
    # this handler runs in Starlette's outermost middleware, outside the one that adds the base headers
    headers = {name.decode(): value.decode() for name, value in BASE_HEADERS}
    return ErrorResponse(status_code=status, content={"error": _phrase(status), "detail": f"Internal server error (id {error_id}); the details are in the server log", "id": error_id}, headers=headers)


# Longest raw body text an error echoes back (an upload can be megabytes)
ECHO_LIMIT = 1024


def _echoed_bytes(value: bytes) -> str:
    """Raw body bytes a validation error echoes as its ``input`` (a multipart
    upload sent to a JSON route): short UTF-8 text as is, anything else as a
    placeholder. ``jsonable_encoder`` decodes bytes as UTF-8, so binary data
    (a PNG) raised inside the error handler and answered 500."""

    if len(value) <= ECHO_LIMIT:
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return f"<{len(value)} bytes of binary data>"


def _request_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    """422 for a request that fails its model (also for bodies holding
    ``1e999`` / ``NaN`` or binary data, which the default handler could not
    encode: a 500 and a traceback per request from any client). A client
    error: logged at debug level, without a traceback."""

    log.debug("%s %s: invalid request (%d errors)", request.method, request.url.path, len(error.errors()))
    encoders = {bytes: _echoed_bytes, bytearray: lambda value: _echoed_bytes(bytes(value)), memoryview: lambda value: _echoed_bytes(value.tobytes())}
    return ErrorResponse(status_code=422, content={"detail": jsonable_encoder(error.errors(), custom_encoder=encoders)})


def _validation_error(request: Request, error: ValidationError) -> JSONResponse:
    return ErrorResponse(status_code=422, content=error.to_json())


def _job_error(request: Request, error: JobError) -> JSONResponse:
    return ErrorResponse(status_code=error.status, content={"detail": str(error)})


def _file_error(request: Request, error: FileError) -> JSONResponse:
    return ErrorResponse(status_code=error.status, content={"detail": str(error)})


BROKEN_TEXT = "the text has a broken character (an unpaired surrogate)"
# routes that check their values themselves and report each setting (remi/server/options.py)
CHECKED_PER_SETTING = (f"{API_PREFIX}/options", f"{API_PREFIX}/options/validate")
# An unpaired surrogate in a JSON body: escaped (\ud800 to \udfff) or as raw bytes, which `json.loads`
# takes (0xED 0xA0-0xBF; Hangul is 0xED 0x80-0x9F)
_MAYBE_BROKEN = re.compile(rb"\\u[dD][89a-fA-F]|\xed[\xa0-\xbf]")


def _broken_text_at(value: Any) -> tuple[str, ...] | None:
    """Where in ``value`` the first text UTF-8 cannot write is, or None
    (without recursion: a body can be nested as deep as the parser goes)."""

    pending: list[tuple[Any, tuple[str, ...]]] = [(value, ())]
    while pending:
        item, path = pending.pop()
        if isinstance(item, str):
            if not encodable(item):
                return path
            continue
        children = list(item.items()) if isinstance(item, dict) else list(enumerate(item)) if isinstance(item, list) else []
        for key, _child in children:
            if not encodable(key):
                return (*path, str(key).encode("utf-8", "backslashreplace").decode("utf-8"))
        pending.extend((child, (*path, str(key))) for key, child in reversed(children))
    return None


async def refuse_broken_text(connection: HTTPConnection) -> None:
    """Every JSON body (``/api/v2``, ``/sdapi/v1``): text UTF-8 cannot write is
    refused (422) before a route keeps it. JSON can carry an unpaired surrogate
    (escaped, "\\udc80", half of an emoji, or as raw bytes); UTF-8 cannot hold
    one, so a value of any kind (a setting, a recipe's patch, a pipeline, a
    model's metadata) failed in the middle of its write to a file or to SQLite;
    ``/sdapi/v1/options`` stored it and left config.json cut short, every
    setting lost at the next start.
    Typed text fields refused it already (pydantic); this covers values of
    any kind. FastAPI reads a body as JSON without a content type, and with
    one in any case, and in any of the encodings JSON allows."""

    if not isinstance(connection, Request):
        return
    kind = connection.headers.get("content-type", "").lower()
    if kind and "json" not in kind:
        return  # uploads and forms: their routes read text of their own
    if connection.scope["path"].endswith(CHECKED_PER_SETTING):
        return  # these name each refused setting for the settings page
    body = await connection.body()
    # `json.loads` reads UTF-16 and UTF-32 too, whose bytes the patterns never match: parse those
    if json.detect_encoding(body) in ("utf-8", "utf-8-sig") and not _MAYBE_BROKEN.search(body):
        return
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        return  # the route's own validation answers (a body nested too deep: 400)
    path = _broken_text_at(data)
    if path is not None:
        raise ValidationError([ParamError(BROKEN_TEXT, owner="body", param=".".join(path) or "body")])


def snapshot_events(services: Services) -> list[dict[str, Any]]:
    """What a client connecting mid-job needs: the queue and the running job."""

    events: list[dict[str, Any]] = [{"type": "queue.changed", **services.queue.status()}]
    ref = services.queue.current_ref()
    if ref is not None:
        events.append({"type": "job.started", **ref.fields()})
    if ref is not None and ref.begun:  # before that, the progress state is still the previous task's
        try:
            events.append({"type": "job.progress", **ref.fields(), **services.bridge.progress_snapshot()})
        except Exception:
            pass
    return events


def _no_credentials_for_every_origin(app: FastAPI) -> None:
    """With a wildcard CORS origin (``--cors-allow-origins=*`` or a regex that
    admits every origin) the CORS middleware sends no ``Allow-Credentials``:
    with it, Starlette answers a request that carries a cookie with the
    requesting origin and ``Allow-Credentials: true``, so every web site could
    read the answers with the user's login. Only origins listed exactly get
    credentials, and then only without a wildcard next to them. (The stack is
    built at the first request, from ``user_middleware``.)"""

    from starlette.middleware.cors import CORSMiddleware

    from modules import extension_route_guard
    from modules.shared_cmd_options import cmd_opts

    if not extension_route_guard.cors_allows_every_origin(cmd_opts):
        return
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            middleware.kwargs["allow_credentials"] = False
    if app.middleware_stack is not None:
        app.middleware_stack = app.build_middleware_stack()


def create_app(
    *,
    services: Services | None = None,
    auth: Auth | None = None,
    webui: bool = True,
    sdapi: bool | None = None,
    spa_dir: Path | None = None,
    start_services: bool = True,
) -> FastAPI:
    """The app: ``/api/v2`` (REST + WebSocket), ``/internal``, ``/sdapi/v1``
    (``sdapi``; default: ``--api`` or ``--nowebui``) and, with ``webui``,
    the SPA from ``webui/dist``. ``services`` default to the process-wide
    ones (worker, file index, ...), started with the app."""

    services = services or default_services()
    auth = auth or auth_from_cmd_opts(services.root)
    if sdapi is None:
        from modules.shared_cmd_options import cmd_opts

        sdapi = bool(cmd_opts.api or cmd_opts.nowebui)
    hub = Hub(version=version(), snapshot=lambda: snapshot_events(services))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hub.bind(asyncio.get_running_loop())
        unsubscribe = services.bus.subscribe(hub)
        if start_services:
            services.start()
        try:
            yield
        finally:
            unsubscribe()
            await hub.close_all()

    app = FastAPI(
        title="Remi WebUI",
        version=version(),
        lifespan=lifespan,
        # FastAPI's docs pages load their viewers from a CDN: ours serve the build's copies
        docs_url=None,
        redoc_url=None,
        default_response_class=TextSafeResponse,
        exception_handlers={Exception: handle_exception, RequestValidationError: _request_validation_error, ValidationError: _validation_error, JobError: _job_error, FileError: _file_error},
        # every route's JSON body, /sdapi's (added below) included
        dependencies=[Depends(refuse_broken_text)],
    )
    app.state.services = services
    app.state.auth = auth
    app.state.hub = hub
    app.state.sdapi = sdapi
    # extension_route_guard asks the app who is logged in
    app.auth_dependency = auth.user_for  # type: ignore[attr-defined]

    from remi.server.routes import admin, boards, chains, core, extensions, files, jobs, library, networks, prompt_tools, queue_tools, recipes, system, workers

    api = APIRouter(prefix=API_PREFIX)
    for module in (core, jobs, files, networks, extensions, system, admin, queue_tools, library, boards, recipes, chains, workers, prompt_tools):
        api.include_router(module.router)
    app.include_router(api)
    app.include_router(system.internal, prefix="/internal")
    system.add_progress_routes(app)
    add_docs_routes(app, spa_dir or DIST, serve_assets=not webui)

    if sdapi:
        from modules.api.api import Api
        from modules.call_queue import queue_lock

        Api(app, queue_lock)

    # (inside the security checks: a refused request is never read)
    def request_body_limit(scope: Scope) -> int:
        return PUBLIC_BODY_LIMIT if route_path(scope).startswith(PUBLIC_PATHS) else body_limit(services.files.max_upload_bytes)

    app.add_middleware(BodyLimitMiddleware, limit=request_body_limit)
    spa = None
    if webui:
        from modules import shared

        spa = Spa(auth, spa_dir, theme=getattr(shared.cmd_opts, "theme", None))
    app.add_middleware(SecurityMiddleware, auth=auth, login_page=spa.login_page if spa is not None else None)
    from modules import initialize_util

    initialize_util.setup_middleware(app)
    _no_credentials_for_every_origin(app)

    if spa is not None:
        app.router.default = spa  # type: ignore[assignment]
    return app
