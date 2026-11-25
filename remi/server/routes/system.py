"""Login, server control, sysinfo, the WebSocket, and the ``/internal``
routes API clients use (``/internal/progress`` & co.)."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
import threading
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse

from remi.server import models as m
from remi.server.auth import LOGIN_FAILURE_DELAY, client_key, reached_over_https
from remi.server.deps import UserDep

router = APIRouter()
internal = APIRouter()


# region Auth


@router.get("/auth/status", response_model=m.AuthStatus, summary="Whether a login is required, and who is logged in")
def auth_status(request: Request, user: UserDep) -> dict[str, Any]:
    auth = request.app.state.auth
    if user is None and auth.required:
        # HTTP Basic, checked like a login: failures count towards the
        # client's throttle, and a throttled client is not told
        user = auth.basic_user(request.headers, client_key(request))
    return {"required": auth.required, "user": user, "source_url": source_url(request, user)}


def source_url(request: Request, user: str | None) -> str:
    """The source offer (``remi/server/source.py``): the running version for
    its users, the repository for a client that has not signed in."""

    source = request.app.state.source
    return source.running if user is not None or not request.app.state.auth.required else source.public


@router.post("/auth/login", status_code=204, responses={401: {"model": m.Detail}, 429: {"model": m.Detail}}, summary="Log in (sets the session cookie)")
async def login(body: m.LoginRequest, request: Request) -> Response:
    auth = request.app.state.auth
    client = client_key(request)
    if not auth.required:
        return Response(status_code=204)
    if auth.throttled(client):
        raise HTTPException(status_code=429, detail="too many failed logins; wait a minute")
    if not auth.check_password(body.username, body.password):
        auth.record_failure(client)
        await asyncio.sleep(LOGIN_FAILURE_DELAY)
        raise HTTPException(status_code=401, detail="wrong username or password")
    response = Response(status_code=204)
    auth.set_cookie(response, body.username, https=reached_over_https(request))
    return response


@router.post("/auth/logout", status_code=204, summary="Log out (revokes and clears the session cookie)")
def logout(request: Request) -> Response:
    from remi.server.auth import COOKIE_NAME

    auth = request.app.state.auth
    session = request.cookies.get(COOKIE_NAME)
    auth.revoke(session)
    hub = getattr(request.app.state, "hub", None)
    if session and hub is not None:
        hub.disconnect(session)  # its open sockets would stream on
    response = Response(status_code=204)
    auth.clear_cookie(response)
    return response


# endregion

# region Server


@router.post("/server/reload", response_model=m.Ok, summary="Reload the UI: reload scripts and rebuild the app (in-process)")
def reload_server() -> dict[str, Any]:
    from modules import shared

    # not request_restart(): that interrupts the running job, which the reload waits for (queue.hold)
    shared.state.server_command = "restart"
    return {"ok": True, "message": "reloading"}


@router.post("/server/restart", response_model=m.Ok, responses={501: {"model": m.Detail}}, summary="Restart the WebUI process (when started by the launcher script); answers, then stops")
def restart_server() -> dict[str, Any]:
    from modules import restart, shared
    from remi.server.launch import RESTART_PROGRAM

    if not restart.is_restartable():
        raise HTTPException(status_code=501, detail="the WebUI was not started by a launcher that can restart it")
    # The serve loop stops the server (after this response), interrupts the
    # running job, stops the worker and exits for the launcher to restart.
    shared.state.server_command = RESTART_PROGRAM
    return {"ok": True, "message": "restarting"}


def user_assets() -> dict[str, tuple[str, str]]:
    """The user's customisation files that exist: name → (path, media type);
    ``user.css`` in the data folder, ``notification.mp3`` in the data folder
    or else the WebUI folder (where the Gradio UI read it: outside the data
    folder, so outside a Docker volume)."""

    from modules.paths_internal import data_path, script_path

    candidates = {"user.css": [os.path.join(data_path, "user.css")], "notification.mp3": [os.path.join(data_path, "notification.mp3"), os.path.join(script_path, "notification.mp3")]}
    media_types = {"user.css": "text/css", "notification.mp3": "audio/mpeg"}
    found = {}
    for name, paths in candidates.items():
        path = next((path for path in paths if os.path.isfile(path)), None)
        if path is not None:
            found[name] = (path, media_types[name])
    return found


@router.get("/user-assets/{name}", response_class=FileResponse, responses={200: {"content": {"text/css": {}, "audio/mpeg": {}}}, 404: {"model": m.Detail}}, summary="user.css or notification.mp3, when the user added them")
def user_asset(name: str) -> FileResponse:
    entry = user_assets().get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no {name}")
    path, media_type = entry
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})


@router.get("/sysinfo", response_model=dict[str, Any], summary="System information (as the sysinfo download)")
def sysinfo() -> dict[str, Any]:
    from modules import sysinfo as info

    return info.get_dict()


@router.websocket("/ws")
async def websocket(websocket: WebSocket) -> None:
    await websocket.app.state.hub.serve(websocket)


# endregion

# region /internal


_instance_lock = threading.Lock()


@internal.get("/ping", summary="Answers once the server is up (polled after a reload); `instance` differs for every app (it changes with a reload or restart)")
def ping(request: Request) -> dict[str, Any]:
    state = request.app.state
    with _instance_lock:
        instance = getattr(state, "instance", None)
        if instance is None:
            instance = state.instance = uuid.uuid4().hex
    return {"instance": instance}


@internal.get("/quicksettings-hint")
def quicksettings_hint() -> list[dict[str, str]]:
    from modules import shared

    return [{"name": key, "label": info.label} for key, info in shared.opts.data_labels.items()]


@internal.get("/profile-startup")
def profile_startup() -> Any:
    from modules import timer

    return timer.startup_record


def _sysinfo_text(attachment: bool) -> PlainTextResponse:
    from modules import sysinfo as info

    filename = f"sysinfo-{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d-%H-%M')}-UTC.json"
    try:
        text = info.get()
    except UnicodeEncodeError:
        # a setting holding an unpaired surrogate (an /sdapi/v1/options write) cannot be UTF-8:
        # the same file with JSON escapes, its checksum taken the way sysinfo.check reads it
        text = json.dumps(info.get_dict(), ensure_ascii=True, indent=4)
        text = text.replace(info.checksum_token, hashlib.sha256(text.encode("utf8")).hexdigest())
    return PlainTextResponse(text, headers={"Content-Disposition": f'{"attachment" if attachment else "inline"}; filename="{filename}"'})


@internal.get("/sysinfo")
def internal_sysinfo() -> PlainTextResponse:
    return _sysinfo_text(False)


@internal.get("/sysinfo-download")
def internal_sysinfo_download() -> PlainTextResponse:
    return _sysinfo_text(True)


def add_progress_routes(app: Any) -> None:
    """``/internal/progress`` and ``/internal/pending-tasks``."""

    from modules import progress

    progress.setup_progress_api(app)


# endregion
