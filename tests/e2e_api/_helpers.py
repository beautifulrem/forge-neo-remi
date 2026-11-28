"""Pure client-side helpers for opaque-box E2E tests (HTTP + WebSocket only)."""

from __future__ import annotations

import base64
import io
import json
import struct
import time
from collections.abc import Callable
from typing import Any

import httpx
from PIL import Image, PngImagePlugin

_PREVIEW_HEADER = struct.Struct("<BBII")
_PREVIEW_FORMATS = {1: "jpeg", 2: "webp"}


class OpaqueHelpers:
    """Stateless client-side builders, upload/job submitters, and WS decoders."""

    @staticmethod
    def make_png_bytes(
        size: tuple[int, int] = (32, 32),
        color: tuple[int, ...] = (120, 80, 40),
        mode: str = "RGB",
        text: str | None = None,
        itxt: dict[str, str] | None = None,
    ) -> bytes:
        buf = io.BytesIO()
        info = PngImagePlugin.PngInfo()
        if text is not None:
            info.add_text("parameters", text)
        for key, value in (itxt or {}).items():
            info.add_itxt(key, value)
        with Image.new(mode, size, color) as image:
            image.save(buf, format="PNG", pnginfo=info)
        return buf.getvalue()

    @staticmethod
    def make_jpeg_bytes(
        size: tuple[int, int] = (32, 32),
        color: tuple[int, int, int] = (120, 80, 40),
        quality: int = 85,
    ) -> bytes:
        buf = io.BytesIO()
        with Image.new("RGB", size, color) as image:
            image.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @classmethod
    def make_b64_png(
        cls,
        size: tuple[int, int] = (32, 32),
        color: tuple[int, ...] = (120, 80, 40),
        **kwargs: Any,
    ) -> str:
        return base64.b64encode(cls.make_png_bytes(size=size, color=color, **kwargs)).decode("ascii")

    @staticmethod
    def make_safetensors_bytes(metadata: dict[str, Any] | None = None) -> bytes:
        header_obj: dict[str, Any] = {"__metadata__": metadata or {"format": "pt"}}
        header = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
        return struct.pack("<Q", len(header)) + header

    @staticmethod
    def upload_bytes(
        server: Any,
        data: bytes,
        filename: str = "upload.png",
        content_type: str = "image/png",
        *,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        http = client or server.http
        response = http.post("/api/v2/uploads", files={"file": (filename, data, content_type)})
        assert response.status_code == 201, f"upload failed ({response.status_code}): {response.text}"
        return response.json()

    @classmethod
    def upload_png(
        cls,
        server: Any,
        *,
        size: tuple[int, int] = (32, 32),
        color: tuple[int, ...] = (120, 80, 40),
        filename: str = "input.png",
        text: str | None = None,
        itxt: dict[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        raw = cls.make_png_bytes(size=size, color=color, text=text, itxt=itxt)
        return cls.upload_bytes(server, raw, filename=filename, content_type="image/png", client=client)

    @staticmethod
    def create_job(
        server: Any,
        *,
        tab: str = "txt2img",
        core: dict[str, Any] | None = None,
        scripts: dict[str, Any] | None = None,
        priority: int = 0,
        repeat: int = 1,
        forever: bool = False,
        upscale_from: str | None = None,
        idempotency_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        http = client or server.http
        payload: dict[str, Any] = {
            "tab": tab,
            "core": dict(core or {}),
            "scripts": dict(scripts or {}),
            "priority": priority,
            "repeat": repeat,
            "forever": forever,
        }
        if upscale_from is not None:
            payload["upscale_from"] = upscale_from
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key is not None else None
        response = http.post("/api/v2/jobs", json=payload, headers=headers)
        assert response.status_code == 201, f"create_job failed ({response.status_code}): {response.text}"
        return response.json()

    @classmethod
    def submit_txt2img(
        cls,
        server: Any,
        prompt: str = "1girl, test",
        *,
        width: int = 64,
        height: int = 64,
        steps: int | None = None,
        seed: int | None = None,
        core: dict[str, Any] | None = None,
        scripts: dict[str, Any] | None = None,
        priority: int = 0,
        repeat: int = 1,
        forever: bool = False,
        upscale_from: str | None = None,
        idempotency_key: str | None = None,
        wait: bool = True,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        core_payload: dict[str, Any] = {"prompt": prompt, "width": width, "height": height, **(core or {})}
        scripts_payload: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (scripts or {}).items()}
        if steps is not None:
            sampler = dict(scripts_payload.get("sampler") or {})
            sampler.setdefault("steps", steps)
            scripts_payload["sampler"] = sampler
        if seed is not None:
            seed_cfg = dict(scripts_payload.get("seed") or {})
            seed_cfg.setdefault("seed", seed)
            scripts_payload["seed"] = seed_cfg
        created = cls.create_job(
            server,
            tab="txt2img",
            core=core_payload,
            scripts=scripts_payload,
            priority=priority,
            repeat=repeat,
            forever=forever,
            upscale_from=upscale_from,
            idempotency_key=idempotency_key,
            client=client,
        )
        if not wait:
            return created
        return server.wait_job(created["job_id"], timeout=timeout, client=client)

    @classmethod
    def submit_img2img(
        cls,
        server: Any,
        init_img_id: str,
        prompt: str = "1girl, refine",
        *,
        denoising_strength: float = 0.4,
        width: int = 64,
        height: int = 64,
        steps: int | None = None,
        seed: int | None = None,
        core: dict[str, Any] | None = None,
        scripts: dict[str, Any] | None = None,
        priority: int = 0,
        repeat: int = 1,
        forever: bool = False,
        idempotency_key: str | None = None,
        wait: bool = True,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        core_payload: dict[str, Any] = {
            "prompt": prompt,
            "init_img": init_img_id,
            "denoising_strength": denoising_strength,
            "width": width,
            "height": height,
            **(core or {}),
        }
        scripts_payload: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (scripts or {}).items()}
        if steps is not None:
            sampler = dict(scripts_payload.get("sampler") or {})
            sampler.setdefault("steps", steps)
            scripts_payload["sampler"] = sampler
        if seed is not None:
            seed_cfg = dict(scripts_payload.get("seed") or {})
            seed_cfg.setdefault("seed", seed)
            scripts_payload["seed"] = seed_cfg
        created = cls.create_job(
            server,
            tab="img2img",
            core=core_payload,
            scripts=scripts_payload,
            priority=priority,
            repeat=repeat,
            forever=forever,
            idempotency_key=idempotency_key,
            client=client,
        )
        if not wait:
            return created
        return server.wait_job(created["job_id"], timeout=timeout, client=client)

    @staticmethod
    def recv_ws_message(ws: Any, *, timeout: float = 5.0) -> dict[str, Any] | bytes:
        raw = ws.recv(timeout=timeout)
        if isinstance(raw, bytes):
            return raw
        return json.loads(raw)

    @classmethod
    def recv_ws_json(cls, ws: Any, *, timeout: float = 5.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"did not receive a JSON WebSocket message within {timeout:.2f}s")
            msg = cls.recv_ws_message(ws, timeout=remaining)
            if isinstance(msg, dict):
                return msg

    @classmethod
    def recv_ws_until(
        cls,
        ws: Any,
        predicate: Callable[[dict[str, Any] | bytes], bool],
        *,
        timeout: float = 5.0,
        collect_all: bool = False,
    ) -> Any:
        deadline = time.monotonic() + timeout
        seen: list[dict[str, Any] | bytes] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"predicate not matched within {timeout:.2f}s; seen={seen!r}")
            try:
                msg = cls.recv_ws_message(ws, timeout=remaining)
            except TimeoutError as exc:
                raise TimeoutError(f"predicate not matched within {timeout:.2f}s; seen={seen!r}") from exc
            seen.append(msg)
            if predicate(msg):
                return (msg, seen) if collect_all else msg

    @classmethod
    def drain_ws(
        cls,
        ws: Any,
        *,
        idle_timeout: float = 0.15,
        max_messages: int = 500,
    ) -> list[dict[str, Any] | bytes]:
        collected: list[dict[str, Any] | bytes] = []
        for _ in range(max_messages):
            try:
                collected.append(cls.recv_ws_message(ws, timeout=idle_timeout))
            except TimeoutError:
                break
        return collected

    @staticmethod
    def parse_preview_frame(data: bytes) -> dict[str, Any]:
        if len(data) < _PREVIEW_HEADER.size:
            raise ValueError(f"preview frame too short: {len(data)} bytes")
        kind, fmt, job_seq, step = _PREVIEW_HEADER.unpack_from(data)
        return {
            "type": kind,
            "fmt": fmt,
            "format": _PREVIEW_FORMATS.get(fmt, "unknown"),
            "job_seq": job_seq,
            "step": step,
            "payload": data[_PREVIEW_HEADER.size :],
        }

    @staticmethod
    def login(
        server: Any,
        username: str,
        password: str,
        *,
        client: httpx.Client | None = None,
    ) -> httpx.Response:
        http = client or server.http
        return http.post("/api/v2/auth/login", json={"username": username, "password": password})

    @staticmethod
    def basic_auth_header(username: str, password: str) -> dict[str, str]:
        token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        return {"Authorization": f"Basic {token}"}
