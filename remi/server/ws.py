"""The WebSocket hub (``/api/v2/ws``).

Each client gets a bounded event queue and a single preview slot (a newer
frame replaces one not sent yet). A client that falls so far behind that its
queue overflows gets ``{"type": "resync", "dropped": n}`` and a fresh
snapshot (queue state, running job) instead of the stale backlog; it should
refetch what it shows. Events arrive from any thread through the
:class:`remi.realtime.EventBus`; the hub hands them to its event loop. A
failed send closes the socket (the client reconnects).

Protocol: the server sends ``{"type": "hello", "features": [...],
"version": ...}`` on connect, then the current queue state; the client may
send ``{"type": "hello", "features": [...], "visible": bool}`` (previews are
only sent in the formats it lists, when it lists any), ``{"type":
"visibility", "visible": bool}`` when its page is hidden or shown again (a
hidden page gets no previews; while no client is visible none are encoded)
and ``{"type": "ping"}`` (answered with ``pong``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Callable
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from remi.realtime import FORMAT_NAMES, PreviewFrame
from remi.setting_values import encodable

log = logging.getLogger(__name__)

SERVER_FEATURES = ("events.v1", "preview.binary", "preview.jpeg", "preview.webp", "jobs.v1")
MAX_EVENTS = 256
SEND_TIMEOUT = 10.0


Snapshot = Callable[[], list[dict[str, Any]]]


class Client:
    def __init__(self, websocket: WebSocket, max_events: int = MAX_EVENTS, snapshot: Snapshot | None = None, session: str | None = None, send_timeout: float = SEND_TIMEOUT) -> None:
        self.websocket = websocket
        self.session = session
        """the session cookie of the handshake: logging that session out closes the socket"""
        self.send_timeout = float(send_timeout)
        self.close_code: int | None = None
        """set to close the socket from its own send loop (see :meth:`Hub.disconnect`)"""
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.preview: PreviewFrame | None = None
        self.features: set[str] | None = None
        self.visible = True
        """the client's page is shown (``visibility`` messages): a hidden one gets no previews"""
        self.wake = asyncio.Event()
        self.snapshot = snapshot
        self.dropped = 0
        """events lost since the last resync"""

    def push_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "job.progress":
            job_id = event.get("job_id")
            for idx in range(len(self.events) - 1, -1, -1):
                existing = self.events[idx]
                if existing.get("type") == "job.progress" and existing.get("job_id") == job_id:
                    del self.events[idx]
                    break
        if len(self.events) == self.events.maxlen:
            self.dropped += 1
        self.events.append(event)
        self.wake.set()

    def push_frame(self, frame: PreviewFrame) -> None:
        if self.accepts(frame):
            self.preview = frame
            self.wake.set()

    def accepts_previews(self) -> bool:
        if self.features is None:
            return True
        return "preview.binary" in self.features and any(f"preview.{name}" in self.features for name in FORMAT_NAMES.values())

    def accepts(self, frame: PreviewFrame) -> bool:
        if self.features is None:
            return True
        return "preview.binary" in self.features and f"preview.{FORMAT_NAMES[frame.fmt]}" in self.features

    async def send_loop(self) -> None:
        """Send until the socket fails (then close it and return)."""

        try:
            while True:
                await self.wake.wait()
                self.wake.clear()
                if self.close_code is not None:
                    await asyncio.wait_for(self.websocket.close(code=self.close_code), timeout=min(self.send_timeout, 2.0))
                    return
                if self.dropped:
                    await self._resync()
                while self.events:
                    if self.close_code is not None:
                        await asyncio.wait_for(self.websocket.close(code=self.close_code), timeout=min(self.send_timeout, 2.0))
                        return
                    await self._send(self.events.popleft())
                frame, self.preview = self.preview, None
                if frame is not None:
                    if self.close_code is not None:
                        await asyncio.wait_for(self.websocket.close(code=self.close_code), timeout=min(self.send_timeout, 2.0))
                        return
                    await asyncio.wait_for(self.websocket.send_bytes(frame.pack()), timeout=self.send_timeout)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.debug("websocket send failed: %s", error)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.websocket.close(code=1011), timeout=min(self.send_timeout, 2.0))

    async def _send(self, event: dict[str, Any]) -> None:
        try:
            text = json.dumps(event, ensure_ascii=False, allow_nan=False, default=str)
        except ValueError:  # NaN/inf in an event: not JSON
            log.warning("dropped an event that is not JSON: %s", event.get("type"))
            return
        if not encodable(text):
            # An unpaired surrogate (a job's error naming a file that is not UTF-8, an extension's
            # progress text): the frame is UTF-8, so it goes as a JSON escape, as the API writes it
            text = json.dumps(event, ensure_ascii=True, allow_nan=False, default=str)
        await asyncio.wait_for(self.websocket.send_text(text), timeout=self.send_timeout)

    async def _resync(self) -> None:
        dropped, self.dropped = self.dropped, 0
        self.events.clear()
        await self._send({"type": "resync", "dropped": dropped})
        try:
            snapshot = self.snapshot() if self.snapshot else []
        except Exception:
            log.exception("websocket snapshot")
            snapshot = []
        for event in snapshot:
            await self._send(event)


class Hub:
    """Subscribes to the bus while its app runs (see ``create_app``)."""

    def __init__(self, *, version: str = "", snapshot: Callable[[], list[dict[str, Any]]] | None = None, send_timeout: float = SEND_TIMEOUT) -> None:
        self.version = version
        self.snapshot = snapshot
        self.send_timeout = float(send_timeout)
        self.clients: set[Client] = set()
        self._pending: set[Client] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    # Subscriber (any thread)

    def _call(self, func: Callable[..., None], *args: Any) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(func, *args)
        except RuntimeError:  # the loop is shutting down
            pass

    def send_event(self, event: dict[str, Any]) -> None:
        self._call(self._fan_event, event)

    def send_frame(self, frame: PreviewFrame) -> None:
        self._call(self._fan_frame, frame)

    def disconnect(self, session: str) -> None:
        """Close the sockets opened with the session cookie ``session`` (it
        was logged out: they would stream on). Close code 1008."""

        self._call(self._disconnect, session)

    def _disconnect(self, session: str) -> None:
        for client in (*self.clients, *self._pending):
            if client.session == session:
                client.close_code = 1008
                client.wake.set()

    def _fan_event(self, event: dict[str, Any]) -> None:
        for client in self.clients:
            client.push_event(event)

    def _fan_frame(self, frame: PreviewFrame) -> None:
        for client in self.clients:
            if client.visible:
                client.push_frame(frame)

    # Connections

    @staticmethod
    def _session_revoked(websocket: WebSocket) -> bool:
        app = getattr(websocket, "app", None)
        state = getattr(app, "state", None) if app is not None else None
        auth = getattr(state, "auth", None) if state is not None else None
        if auth is not None and getattr(auth, "required", False):
            return auth.user_for(websocket, count_failures=False) is None
        return False

    async def serve(self, websocket: WebSocket) -> None:
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        from remi.server.auth import COOKIE_NAME

        client = Client(
            websocket,
            snapshot=self.snapshot,
            session=websocket.cookies.get(COOKIE_NAME),
            send_timeout=self.send_timeout,
        )
        # Track the handshake before accept() so a concurrent logout/disconnect()
        # during accept() marks client.close_code instead of missing it.
        self._pending.add(client)
        try:
            await websocket.accept()
            if client.close_code is not None or self._session_revoked(websocket):
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(websocket.close(code=client.close_code or 1008), timeout=min(self.send_timeout, 2.0))
                return
            client.push_event({"type": "hello", "features": list(SERVER_FEATURES), "version": self.version})
            try:
                for event in self.snapshot() if self.snapshot else []:
                    client.push_event(event)
            except Exception:
                log.exception("websocket snapshot")
            self._pending.discard(client)
            self.clients.add(client)
            sender = asyncio.create_task(client.send_loop())
            receiver = asyncio.create_task(self._receive(client))
            try:
                # until the client leaves or a send fails
                await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (sender, receiver):
                    task.cancel()
                for task in (sender, receiver):
                    with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, Exception):
                        await task
        finally:
            self._pending.discard(client)
            self.clients.discard(client)

    async def _receive(self, client: Client) -> None:
        while True:
            message = await client.websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            text = message.get("text")
            if not text:
                continue
            try:
                data = json.loads(text)
            except (ValueError, RecursionError):  # not JSON, or nested past the parser: ignored, as any bad message
                continue
            if not isinstance(data, dict):
                continue
            if data.get("type") == "hello" and isinstance(data.get("features"), list):
                client.features = {str(item) for item in data["features"]}
                if isinstance(data.get("visible"), bool):
                    client.visible = data["visible"]
            elif data.get("type") == "visibility" and isinstance(data.get("visible"), bool):
                # a page shown again gets the next preview (the encoder looks at the state at least once a second)
                client.visible = data["visible"]
                if not client.visible:
                    client.preview = None
            elif data.get("type") == "ping":
                client.push_event({"type": "pong"})

    def wants_frames(self) -> bool:
        """Whether a visible client accepts binary previews (any thread: the tuple copies the set in one step)."""

        return any(client.visible and client.accepts_previews() for client in tuple(self.clients))

    async def close_all(self) -> None:
        for client in [*self.clients, *self._pending]:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(client.websocket.close(code=1001), timeout=min(self.send_timeout, 2.0))
        self.clients.clear()
        self._pending.clear()
