import base64
import io
import ipaddress
import math
import re
import socket
import time
from collections.abc import Mapping
from contextlib import closing, contextmanager
from contextvars import ContextVar
from io import BytesIO
from secrets import compare_digest
from threading import Lock
from types import UnionType
from typing import Any, Union, get_args, get_origin
from urllib.parse import unquote_to_bytes, urljoin, urlparse

import piexif
import piexif.helper
import requests
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Response
from fastapi.exceptions import HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import PngImagePlugin

import modules.shared as shared
from modules import errors, images, infotext_utils, postprocessing, restart, script_callbacks, scripts, sd_models, sd_samplers, sd_schedulers, shared_items
from modules.api import models
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, process_extra_images, process_images
from modules.progress import add_task_to_queue, create_task_id, current_task, finish_task, start_task
from modules.shared import opts
from modules.util import plaintext_to_html


def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e


def pasted_value(value, target_type):
    """An infotext value as a request field value of ``target_type``:
    infotext "False" is False, and "3.0" fills an int field."""

    if value is None or isinstance(value, target_type):
        return value
    if target_type is bool and value == "False":
        return False
    if target_type is int and isinstance(value, str):
        return int(float(value))
    return target_type(value)


INFOTEXT_API_FIELDS = {"batch_count": "n_iter"}
"""Core form params (remi/core/forms.py) whose request field has another name."""

INFOTEXT_SCRIPT_FIELDS = {
    "sampler": ("steps", "sampler_name", "scheduler"),
    "seed": ("seed", "subseed", "subseed_strength", "seed_resize_from_h", "seed_resize_from_w"),
}
"""Built-in scripts whose params the API takes as request fields."""


def get_base_type(annotation):
    """``X`` of ``Optional[X]`` / ``X | None``."""

    if get_origin(annotation) in (Union, UnionType):
        non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
    return annotation


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=400, detail="Sampler not found")

    return name


_PNGINFO_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _sanitize_pnginfo_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError:
            return f"<{len(data)} bytes>"
        if decoded.isprintable() or all(ch.isprintable() or ch in "\n\r\t" for ch in decoded):
            return decoded
        return f"<{len(data)} bytes>"
    if isinstance(value, str):
        return _PNGINFO_SURROGATE_RE.sub("\ufffd", value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _sanitize_pnginfo_items(items: Mapping[Any, Any] | None) -> dict[str, Any]:
    """Ensure ``items`` from ``read_info_from_image`` holds only UTF-8-safe,
    JSON/Pydantic-serializable values (never raw non-UTF-8 ``bytes``)."""

    if not isinstance(items, Mapping):
        return {}

    def _sanitize_key(key: Any) -> str:
        clean = _sanitize_pnginfo_scalar(key)
        return clean if isinstance(clean, str) else _PNGINFO_SURROGATE_RE.sub("\ufffd", str(clean))

    placeholder: dict[str, Any] = {}
    pending: list[tuple[Any, Any]] = [(items, placeholder)]
    while pending:
        src, dst = pending.pop()
        if isinstance(src, Mapping):
            for k, v in src.items():
                sk = _sanitize_key(k)
                if isinstance(v, Mapping):
                    child: Any = {}
                    dst[sk] = child
                    pending.append((v, child))
                elif isinstance(v, (list, tuple)):
                    child = []
                    dst[sk] = child
                    pending.append((v, child))
                else:
                    dst[sk] = _sanitize_pnginfo_scalar(v)
        else:
            for v in src:
                if isinstance(v, Mapping):
                    child = {}
                    dst.append(child)
                    pending.append((v, child))
                elif isinstance(v, (list, tuple)):
                    child = []
                    dst.append(child)
                    pending.append((v, child))
                else:
                    dst.append(_sanitize_pnginfo_scalar(v))
    return placeholder


def _split_scheduler_suffix(name: str) -> tuple[str, str | None]:
    if name in sd_samplers.all_samplers_map:
        return name, None
    options = [
        (name_option, scheduler.label)
        for scheduler in sd_schedulers.schedulers
        for name_option in (scheduler.label, scheduler.name, *(scheduler.aliases or ()))
        if name_option
    ]
    options.sort(key=lambda item: len(item[0]), reverse=True)
    for name_option, scheduler_label in options:
        suffix = " " + name_option
        if name.endswith(suffix):
            return name[: -len(suffix)], scheduler_label
    return name, None


def _strip_scheduler_suffix(name: str) -> str:
    return _split_scheduler_suffix(name)[0]


def _resolve_sampler_and_scheduler(sampler_name: str | None, sampler_index: str | None, scheduler_name: str | None) -> tuple[str, str]:
    requested = sampler_name or sampler_index
    if requested:
        stripped, matched_scheduler = _split_scheduler_suffix(requested)
        validate_sampler_name(stripped)
        sampler, scheduler = sd_samplers.get_sampler_and_scheduler(stripped, matched_scheduler or scheduler_name)
    else:
        sampler, scheduler = sd_samplers.get_sampler_and_scheduler(requested, scheduler_name)
    return validate_sampler_name(sampler), scheduler


def _require_state():
    state = shared.state
    if state is None:
        raise HTTPException(status_code=503, detail="Runtime state is not initialized")
    return state


def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict["extras_upscaler_1"] = reqDict.pop("upscaler_1", None)
    reqDict["extras_upscaler_2"] = reqDict.pop("upscaler_2", None)
    return reqDict


MAX_URL_IMAGE_BYTES = 64 * 1024 * 1024
MAX_URL_REDIRECTS = 5


_PINNED_DNS: ContextVar[tuple[frozenset[str], tuple[str, ...]] | None] = ContextVar("_remi_sdapi_pinned_dns", default=None)
_PIN_HOOK_LOCK = Lock()
_ORIG_CREATE_CONNECTION = None
_UNSET = object()


def _resolve_public_ips(url: str) -> tuple[frozenset[str], tuple[str, ...]] | None:
    """Resolve ``url``'s hostname once and verify every returned address is public.
    Returns ``(normalized_hostnames, unique_ips)`` or ``None`` if invalid/non-global."""

    try:
        host = urlparse(url).hostname
        if not host:
            return None
        raw_ips: list[str] = []
        seen: set[str] = set()
        for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
            ip = info[4][0].split("%", 1)[0]
            ip_addr = ipaddress.ip_address(ip)
            mapped = getattr(ip_addr, "ipv4_mapped", None)
            if not (mapped or ip_addr).is_global:
                return None
            if ip not in seen:
                seen.add(ip)
                raw_ips.append(ip)
        if not raw_ips:
            return None
        norm = host.strip("[]").lower()
        hosts = {norm}
        try:
            hosts.add(norm.encode("idna").decode("ascii").lower())
        except UnicodeError:
            pass
        return frozenset(hosts), tuple(raw_ips)
    except Exception:
        return None


def _pinned_create_connection(address, timeout=_UNSET, source_address=None, socket_options=None):
    from urllib3.util import connection as urllib3_connection

    if timeout is _UNSET:
        timeout = urllib3_connection._DEFAULT_TIMEOUT
    pinned = _PINNED_DNS.get()
    host, port = address
    norm_host = host.strip("[]").lower()
    if pinned is None or norm_host not in pinned[0]:
        assert _ORIG_CREATE_CONNECTION is not None
        return _ORIG_CREATE_CONNECTION(address, timeout=timeout, source_address=source_address, socket_options=socket_options)

    family = urllib3_connection.allowed_gai_family()
    err = None
    for ip_str in pinned[1]:
        ip_obj = ipaddress.ip_address(ip_str)
        af = socket.AF_INET6 if ip_obj.version == 6 else socket.AF_INET
        if family != socket.AF_UNSPEC and af != family:
            continue
        sa = (ip_str, port, 0, 0) if af == socket.AF_INET6 else (ip_str, port)
        sock = None
        try:
            sock = socket.socket(af, socket.SOCK_STREAM, 0)
            urllib3_connection._set_socket_options(sock, socket_options)
            if timeout is not urllib3_connection._DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            return sock
        except OSError as exc:
            err = exc
            if sock is not None:
                sock.close()
    if err is not None:
        raise err
    raise OSError("getaddrinfo returns an empty list")


def _ensure_dns_pin_hook() -> None:
    from urllib3.util import connection as urllib3_connection

    global _ORIG_CREATE_CONNECTION
    with _PIN_HOOK_LOCK:
        if urllib3_connection.create_connection is not _pinned_create_connection:
            _ORIG_CREATE_CONNECTION = urllib3_connection.create_connection
            urllib3_connection.create_connection = _pinned_create_connection


@contextmanager
def _pin_dns(pinned: tuple[frozenset[str], tuple[str, ...]] | None):
    if pinned is None:
        yield
        return
    _ensure_dns_pin_hook()
    token = _PINNED_DNS.set(pinned)
    try:
        yield
    finally:
        _PINNED_DNS.reset(token)


def verify_url(url):
    """Returns True if the url refers to a global resource: every address
    its host resolves to (IPv4 and IPv6) is public."""

    return _resolve_public_ips(url) is not None


def fetch_url_image_bytes(url: str) -> bytes:
    """GET an image URL for the API (``api_enable_requests``): http(s) only,
    redirects followed by hand and each hop checked against
    ``api_forbid_local_requests`` with the resolved public IP(s) pinned on
    the socket connection to prevent DNS-rebinding TOCTOU, the body capped
    at ``MAX_URL_IMAGE_BYTES``."""

    headers = {"user-agent": opts.api_useragent} if opts.api_useragent else {}
    current = url
    for _hop in range(MAX_URL_REDIRECTS + 1):
        if urlparse(current).scheme not in ("http", "https"):
            raise HTTPException(status_code=500, detail="Invalid image url")
        pinned = None
        request_kwargs: dict[str, Any] = {"timeout": 30, "headers": headers, "stream": True, "allow_redirects": False}
        if opts.api_forbid_local_requests:
            pinned = _resolve_public_ips(current)
            if pinned is None:
                raise HTTPException(status_code=500, detail="Request to local resource not allowed")
            request_kwargs["proxies"] = {"http": "", "https": ""}
        with _pin_dns(pinned), requests.get(current, **request_kwargs) as response:
            if response.is_redirect:
                current = urljoin(current, response.headers.get("location", ""))
                continue
            data = bytearray()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                data.extend(chunk)
                if len(data) > MAX_URL_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="The image at the url is too large")
            return bytes(data)
    raise HTTPException(status_code=500, detail="Too many redirects")


def _checked_image(data: bytes):
    """Decode-ready image, refusing decompression bombs whatever
    ``Image.MAX_IMAGE_PIXELS`` currently is."""

    from PIL import Image

    from remi.files import MAX_UPLOAD_PIXELS

    image = Image.open(BytesIO(data))  # reads the header only
    if image.width * image.height > MAX_UPLOAD_PIXELS:
        raise HTTPException(status_code=413, detail="The image has too many pixels")
    return images.fix_image(image)  # as images.read


def decode_base64_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")

        data = fetch_url_image_bytes(encoding)
        try:
            return _checked_image(data)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid image url") from e

    try:
        if encoding.startswith("data:image/"):
            header, sep, payload = encoding.partition(",")
            if not sep:
                raise ValueError("Missing data URI payload")
            params = [part.strip().lower() for part in header.split(";")[1:]]
            if "base64" in params:
                raw = base64.b64decode(payload)
            else:
                raw = unquote_to_bytes(payload)
                try:
                    return _checked_image(raw)
                except HTTPException:
                    raise
                except Exception:
                    raw = base64.b64decode(payload)
        else:
            raw = base64.b64decode(encoding)
        return _checked_image(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:
        if isinstance(image, str):
            return image
        if opts.samples_format.lower() == "png":
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            if image.mode in ("RGBA", "P") and opts.samples_format.lower() in ("jpg", "jpeg"):
                image = image.convert("RGB")
            parameters = image.info.get("parameters", None)
            exif_bytes = piexif.dump({"Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode")}})
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif=exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif=exif_bytes, quality=opts.jpeg_quality, lossless=opts.webp_lossless)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def refuse_deep_settings(values, what: str):
    """Remi: 422 for setting values nested deeper than ``MAX_NESTING`` levels. They reach
    ``opts.data``, and ``Options.save``'s recursive ``json.dump`` stopped half way on one about
    a thousand levels deep, cutting ``config.json`` short (remi.setting_values.MAX_NESTING)."""
    from remi.setting_values import MAX_NESTING, nested_deeper

    if isinstance(values, dict):
        deep = sorted(str(key) for key, value in values.items() if nested_deeper(value))
        if deep:
            problems = "; ".join(f"{what}.{key}: nested deeper than {MAX_NESTING} levels" for key in deep)
            raise HTTPException(status_code=422, detail=f"invalid {what}: {problems}"[:1000])


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                auth = auth.strip()
                if not auth:
                    continue
                user, sep, password = auth.partition(":")
                if sep and user:
                    self.credentials[user] = password

        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        # (no api_middleware: errors go through remi.server.app.handle_exception,
        # which never sends exception text or bodies to the client)
        self.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"], response_model=models.TextToImageResponse)
        self.add_api_route("/sdapi/v1/img2img", self.img2imgapi, methods=["POST"], response_model=models.ImageToImageResponse)
        self.add_api_route("/sdapi/v1/extra-single-image", self.extras_single_image_api, methods=["POST"], response_model=models.ExtrasSingleImageResponse)
        self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_api, methods=["POST"], response_model=models.ExtrasBatchImagesResponse)
        self.add_api_route("/sdapi/v1/png-info", self.pnginfoapi, methods=["POST"], response_model=models.PNGInfoResponse)
        self.add_api_route("/sdapi/v1/progress", self.progressapi, methods=["GET"], response_model=models.ProgressResponse)
        self.add_api_route("/sdapi/v1/interrupt", self.interruptapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/skip", self.skip, methods=["POST"])
        self.add_api_route("/sdapi/v1/options", self.get_config, methods=["GET"], response_model=models.OptionsModel)
        self.add_api_route("/sdapi/v1/options", self.set_config, methods=["POST"])
        self.add_api_route("/sdapi/v1/cmd-flags", self.get_cmd_flags, methods=["GET"], response_model=models.FlagsModel)
        self.add_api_route("/sdapi/v1/samplers", self.get_samplers, methods=["GET"], response_model=list[models.SamplerItem])
        self.add_api_route("/sdapi/v1/schedulers", self.get_schedulers, methods=["GET"], response_model=list[models.SchedulerItem])
        self.add_api_route("/sdapi/v1/upscalers", self.get_upscalers, methods=["GET"], response_model=list[models.UpscalerItem])
        self.add_api_route("/sdapi/v1/latent-upscale-modes", self.get_latent_upscale_modes, methods=["GET"], response_model=list[models.LatentUpscalerModeItem])
        self.add_api_route("/sdapi/v1/sd-models", self.get_sd_models, methods=["GET"], response_model=list[models.SDModelItem])
        self.add_api_route("/sdapi/v1/sd-modules", self.get_sd_vaes_and_text_encoders, methods=["GET"], response_model=list[models.SDModuleItem])
        self.add_api_route("/sdapi/v1/face-restorers", self.get_face_restorers, methods=["GET"], response_model=list[models.FaceRestorerItem])
        self.add_api_route("/sdapi/v1/prompt-styles", self.get_prompt_styles, methods=["GET"], response_model=list[models.PromptStyleItem])
        self.add_api_route("/sdapi/v1/embeddings", self.get_embeddings, methods=["GET"], response_model=models.EmbeddingsResponse)
        self.add_api_route("/sdapi/v1/refresh-embeddings", self.refresh_embeddings, methods=["POST"])
        self.add_api_route("/sdapi/v1/refresh-checkpoints", self.refresh_checkpoints, methods=["POST"])
        self.add_api_route("/sdapi/v1/refresh-vae", self.refresh_vae, methods=["POST"])
        self.add_api_route("/sdapi/v1/memory", self.get_memory, methods=["GET"], response_model=models.MemoryResponse)
        self.add_api_route("/sdapi/v1/unload-checkpoint", self.unloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/scripts", self.get_scripts_list, methods=["GET"], response_model=models.ScriptsList)
        self.add_api_route("/sdapi/v1/script-info", self.get_script_info, methods=["GET"], response_model=list[models.ScriptInfo])
        self.add_api_route("/sdapi/v1/extensions", self.get_extensions_list, methods=["GET"], response_model=list[models.ExtensionItem])

        if shared.cmd_opts.api_server_stop:
            self.add_api_route("/sdapi/v1/server-kill", self.kill_webui, methods=["POST"])
            self.add_api_route("/sdapi/v1/server-restart", self.restart_webui, methods=["POST"])
            self.add_api_route("/sdapi/v1/server-stop", self.stop_webui, methods=["POST"])

        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

        txt2img_script_runner = scripts.scripts_txt2img
        img2img_script_runner = scripts.scripts_img2img

        if not txt2img_script_runner.scripts:
            txt2img_script_runner.initialize_scripts(False)
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = self.init_default_script_args(txt2img_script_runner)

        if not img2img_script_runner.scripts:
            img2img_script_runner.initialize_scripts(True)
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = self.init_default_script_args(img2img_script_runner)

        from remi.networks import embedding_database

        self.embedding_db = embedding_database()  # shared with /api/v2/networks/embeddings

    def add_api_route(self, path: str, endpoint, **kwargs):
        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        if credentials.username in self.credentials:
            if compare_digest(credentials.password, self.credentials[credentials.username]):
                return True

        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def get_scripts_list(self):
        t2ilist = [script.name for script in scripts.scripts_txt2img.scripts if script.name is not None]
        i2ilist = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]

        return models.ScriptsList(txt2img=t2ilist, img2img=i2ilist)

    def get_script_info(self):
        res = []

        for runner in [scripts.scripts_txt2img, scripts.scripts_img2img]:
            for script in runner.scripts:
                if script.api_info is not None:
                    res.append(script.api_info)

        return res

    def get_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]

    def init_default_script_args(self, script_runner):
        """Every script's default arguments, from its parameter layout (slot
        0, the selected script, is 0: none)."""

        try:
            return list(script_runner.script_args_from_values(None, {}))
        except Exception:
            errors.report("Error reading the default script arguments", exc_info=True)
            script_args = [None] * max([1, *(script.args_to or 0 for script in script_runner.scripts)])
            script_args[0] = 0
            return script_args

    def init_script_args(self, request, default_script_args, selectable_scripts, selectable_idx, script_runner, *, input_script_args=None):
        script_args = default_script_args.copy()

        if input_script_args is not None:
            for index, value in input_script_args.items():
                script_args[index] = value

        # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
        if selectable_scripts:
            script_args[selectable_scripts.args_from : selectable_scripts.args_to] = request.script_args
            script_args[0] = selectable_idx + 1

        # Now check for always on scripts
        if request.alwayson_scripts:
            for alwayson_script_name in request.alwayson_scripts.keys():
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                if alwayson_script is None:
                    raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
                # Selectable script in always on script param check
                if alwayson_script.alwayson is False:
                    raise HTTPException(status_code=422, detail="Cannot have a selectable script in the always on scripts params")
                # always on script with no arg should always run so you don't really need to add them to the requests
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    # min between arg length in scriptrunner and arg length in the request
                    for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                        script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
        return script_args

    def apply_infotext(self, request, tabname, *, script_runner=None, mentioned_script_args=None):
        """Processes `infotext` field from the `request`, and sets other fields of the `request` according to what's in infotext.

        If request already has a field set, and that field is encountered in infotext too, the value from infotext is ignored.

        Additionally, fills `mentioned_script_args` dict with index: value pairs for script arguments read from infotext.
        """

        if not request.infotext:
            return {}

        from remi.core.infotext import parse_infotext
        from remi.ui import Context

        parsed = parse_infotext(request.infotext, tabname)
        params = parsed["params"]
        fields = type(request).model_fields
        set_fields = request.model_fields_set

        def put(field, value):
            if value is None or field not in fields or field in set_fields:
                return
            target_type = get_base_type(fields[field].annotation)
            if target_type is type(None):
                return
            try:
                setattr(request, field, pasted_value(value, target_type) if isinstance(target_type, type) else value)
            except (TypeError, ValueError, OverflowError):  # OverflowError: "Steps: 1e999"
                pass

        # Remi: the form's values are what the UI shows (a choice's label); a request field takes the
        # processing argument, which for a choice by index ("Inpaint area: Only masked") is its index
        from remi.core import forms

        core_nodes = forms.core_layout(tabname, script_runner).by_name
        arg_ctx = Context()

        def arg(name, value):
            node = core_nodes.get(name)
            try:
                return node.to_arg(value, arg_ctx) if node is not None else value
            except Exception:  # a value the node refuses: put() sees it as before
                return value

        for name, value in parsed["core"].items():
            if name != "override_settings":
                put(INFOTEXT_API_FIELDS.get(name, name), arg(name, value))
        for owner, names in INFOTEXT_SCRIPT_FIELDS.items():
            for name in names:
                if name in parsed["scripts"].get(owner, {}):
                    put(name, parsed["scripts"][owner][name])

        if request.override_settings is None:
            request.override_settings = {}

        overridden_settings = infotext_utils.get_override_settings(params)
        for _, setting_name, value in overridden_settings:
            if setting_name not in request.override_settings:
                request.override_settings[setting_name] = value

        if script_runner is not None and mentioned_script_args is not None:
            ctx = Context()
            for script in script_runner.scripts:
                values = parsed["scripts"].get(script.script_id)
                if not values or script.script_id in INFOTEXT_SCRIPT_FIELDS or script.args_from is None:
                    continue
                layout = script_runner.layout_for(script)
                try:
                    args = layout.values_to_args(values, ctx, unknown="ignore")
                except Exception:
                    errors.report(f"Error pasting infotext values of {script.filename}", exc_info=True)
                    continue
                for index, name in enumerate(layout.names):
                    if name in values:
                        mentioned_script_args[script.args_from + index] = args[index]

        return params

    def text2imgapi(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        refuse_deep_settings(txt2imgreq.override_settings, "override_settings")
        task_id = txt2imgreq.force_task_id or create_task_id("txt2img")

        script_runner = scripts.scripts_txt2img

        infotext_script_args = {}
        self.apply_infotext(txt2imgreq, "txt2img", script_runner=script_runner, mentioned_script_args=infotext_script_args)

        selectable_scripts, selectable_script_idx = self.get_selectable_script(txt2imgreq.script_name, script_runner)
        sampler, scheduler = _resolve_sampler_and_scheduler(txt2imgreq.sampler_name, txt2imgreq.sampler_index, txt2imgreq.scheduler)

        populate = txt2imgreq.copy(
            update={  # Override __init__ params
                "sampler_name": sampler,
                "do_not_save_samples": not txt2imgreq.save_images,
                "do_not_save_grid": not txt2imgreq.save_images,
            }
        )
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        if not populate.scheduler and scheduler != "Automatic":
            populate.scheduler = scheduler

        args = vars(populate)
        args.pop("script_name", None)
        args.pop("script_args", None)  # will refeed them to the pipeline directly after initializing them
        args.pop("alwayson_scripts", None)
        args.pop("infotext", None)

        script_args = self.init_script_args(txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner, input_script_args=infotext_script_args)

        send_images = args.pop("send_images", True)
        args.pop("save_images", None)

        state = _require_state()
        add_task_to_queue(task_id)

        with self.queue_lock:
            with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
                p.is_api = True
                p.scripts = script_runner
                p.outpath_grids = opts.outdir_txt2img_grids
                p.outpath_samples = opts.outdir_txt2img_samples

                try:
                    state.begin(job="scripts_txt2img")
                    start_task(task_id)
                    if selectable_scripts is not None:
                        p.script_args = script_args
                        processed = scripts.scripts_txt2img.run(p, *p.script_args)  # Need to pass args as list here
                    else:
                        p.script_args = tuple(script_args)  # Need to pass args as tuple here
                        processed = process_images(p)
                    process_extra_images(processed)
                    finish_task(task_id)
                finally:
                    state.end()
                    if shared.total_tqdm is not None:
                        shared.total_tqdm.clear()

        b64images = list(map(encode_pil_to_base64, processed.images + processed.extra_images)) if send_images else []

        return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def img2imgapi(self, img2imgreq: models.StableDiffusionImg2ImgProcessingAPI):
        refuse_deep_settings(img2imgreq.override_settings, "override_settings")
        task_id = img2imgreq.force_task_id or create_task_id("img2img")

        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)

        script_runner = scripts.scripts_img2img

        infotext_script_args = {}
        self.apply_infotext(img2imgreq, "img2img", script_runner=script_runner, mentioned_script_args=infotext_script_args)

        selectable_scripts, selectable_script_idx = self.get_selectable_script(img2imgreq.script_name, script_runner)
        sampler, scheduler = _resolve_sampler_and_scheduler(img2imgreq.sampler_name, img2imgreq.sampler_index, img2imgreq.scheduler)

        populate = img2imgreq.copy(
            update={  # Override __init__ params
                "sampler_name": sampler,
                "do_not_save_samples": not img2imgreq.save_images,
                "do_not_save_grid": not img2imgreq.save_images,
                "mask": mask,
            }
        )
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        if not populate.scheduler and scheduler != "Automatic":
            populate.scheduler = scheduler

        args = vars(populate)
        args.pop("include_init_images", None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        args.pop("script_name", None)
        args.pop("script_args", None)  # will refeed them to the pipeline directly after initializing them
        args.pop("alwayson_scripts", None)
        args.pop("infotext", None)

        script_args = self.init_script_args(img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner, input_script_args=infotext_script_args)

        send_images = args.pop("send_images", True)
        args.pop("save_images", None)

        state = _require_state()
        add_task_to_queue(task_id)

        with self.queue_lock:
            with closing(StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)) as p:
                p.init_images = [decode_base64_to_image(x) for x in init_images]
                p.is_api = True
                p.scripts = script_runner
                p.outpath_grids = opts.outdir_img2img_grids
                p.outpath_samples = opts.outdir_img2img_samples

                try:
                    state.begin(job="scripts_img2img")
                    start_task(task_id)
                    if selectable_scripts is not None:
                        p.script_args = script_args
                        processed = scripts.scripts_img2img.run(p, *p.script_args)  # Need to pass args as list here
                    else:
                        p.script_args = tuple(script_args)  # Need to pass args as tuple here
                        processed = process_images(p)
                    process_extra_images(processed)
                    finish_task(task_id)
                finally:
                    state.end()
                    if shared.total_tqdm is not None:
                        shared.total_tqdm.clear()

        b64images = list(map(encode_pil_to_base64, processed.images + processed.extra_images)) if send_images else []

        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return models.ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed.js())

    def extras_single_image_api(self, req: models.ExtrasSingleImageRequest):
        reqDict = setUpscalers(req)

        reqDict["image"] = decode_base64_to_image(reqDict["image"])

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)

        # A1111's response field is HTML: the infotext as an escaped paragraph
        return models.ExtrasSingleImageResponse(image=encode_pil_to_base64(result[0][0]), html_info=plaintext_to_html(result[1]))

    def extras_batch_images_api(self, req: models.ExtrasBatchImagesRequest):
        reqDict = setUpscalers(req)

        image_list = reqDict.pop("imageList", [])
        image_folder = [decode_base64_to_image(x.data) for x in image_list]

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)

        return models.ExtrasBatchImagesResponse(images=list(map(encode_pil_to_base64, result[0])), html_info=plaintext_to_html(result[1]))

    def pnginfoapi(self, req: models.PNGInfoRequest):
        image = decode_base64_to_image(req.image.strip())
        if image is None:
            return models.PNGInfoResponse(info="")

        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""
        elif not isinstance(geninfo, str):
            geninfo = str(_sanitize_pnginfo_scalar(geninfo))
        geninfo = _PNGINFO_SURROGATE_RE.sub("\ufffd", geninfo)
        sanitized_items = _sanitize_pnginfo_items(items)

        params = infotext_utils.parse_generation_parameters(geninfo)
        script_callbacks.infotext_pasted_callback(geninfo, params)

        return models.PNGInfoResponse(info=geninfo, items=sanitized_items, parameters=params)

    def progressapi(self, req: models.ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py
        state = _require_state()

        if state.job_count == 0:
            return models.ProgressResponse(progress=0, eta_relative=0, state=state.dict(), textinfo=state.textinfo)

        # avoid dividing zero
        progress = 0.01

        if state.job_count > 0:
            progress += state.job_no / state.job_count
        if state.sampling_steps > 0 and state.job_count > 0:
            progress += 1 / state.job_count * state.sampling_step / state.sampling_steps

        progress = max(0.0, min(progress, 1.0))

        time_since_start = (time.time() - state.time_start) if state.time_start is not None else 0.0
        if progress > 0:
            eta = time_since_start / progress
            eta_relative = eta - time_since_start
        else:
            eta_relative = 0.0

        current_image = None
        if state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(state.current_image)

        return models.ProgressResponse(progress=progress, eta_relative=eta_relative, state=state.dict(), current_image=current_image, textinfo=state.textinfo, current_task=current_task)

    def interruptapi(self):
        _require_state().interrupt()

        return {}

    def unloadapi(self):
        sd_models.unload_model_weights()

        return {}

    def skip(self):
        _require_state().skip()

    def get_config(self):
        from modules.sysinfo import get_config

        return get_config()

    def set_config(self, req: dict[str, Any]):
        from modules.sysinfo import set_config

        refuse_deep_settings(req, "settings")  # Remi: the model selection below skips the value check
        # A setting that does not exist is a client error (422, nothing set),
        # not a KeyError half-way through (500 and a traceback per request).
        # Unchanged values are skipped as before: a posted-back GET answer
        # may hold the settings of a removed extension.
        unknown = sorted(str(key) for key, value in req.items() if key not in shared.opts.data_labels and value != shared.opts.data.get(key))
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown settings: {', '.join(unknown)}"[:1000])
        # Remi: the model selection goes through remi.model_selection, held while a generation runs
        # (set here it switched the running job's model between its batches)
        from remi import model_selection

        from remi.setting_values import frozen_reason
        from remi.ui import ParamError, Sentence, ValidationError
        from remi.ui.settings import check_api_values

        order = list(req)
        req = dict(req)
        picks = {key: req.pop(key) for key in model_selection.MODEL_OPTIONS if key in req}
        problems: list[ParamError] = []
        # a frozen one (--freeze-settings*; or the UI preset's copy main_entry writes with it): refused with the rest, named, not
        # select's AssertionError (a 500 after the other settings were saved)
        frozen = {key: found for key, value in picks.items() if value is not None and (found := model_selection.frozen_pick(key, value)) is not None}
        # so is a changed setting a freeze or a hidden component keeps (Options.set's assertion, a 500 after the values before it
        # were set, or dropped with a 200); a help block and a restrict_api one stay ignored, as Options.set checks those first
        for key, value in list(req.items()):
            info = shared.opts.data_labels.get(key)
            if info is None or info.do_not_save or info.restrict_api or (reason := frozen_reason(shared.opts, key, shared.cmd_opts)) is None:
                continue
            if value == shared.opts.data.get(key, info.default):  # no change (one never stored: its default, as GET answers it)
                del req[key]
            else:
                frozen[key] = key, reason
        problems += [ParamError(reason, owner="settings", param=name) for name, reason in frozen.values()]
        # posted back unchanged, as the other picks: a stored checkpoint whose file was removed passes, as upstream skips it
        if "sd_model_checkpoint" not in frozen and (name := picks.get("sd_model_checkpoint")) is not None and name != shared.opts.data.get("sd_model_checkpoint") and (not isinstance(name, str) or name not in sd_models.checkpoint_aliases):
            problems.append(ParamError(Sentence("unknown checkpoint {{value}}", value=repr(name)), owner="settings", param="sd_model_checkpoint"))  # A1111's exact names
        for key in ("forge_additional_modules", "forge_unet_storage_dtype"):  # checked as PATCH /api/v2/options checks them
            if key in frozen:
                continue
            value, stored = picks.get(key), shared.opts.data.get(key)
            if isinstance(value, list) and isinstance(stored, list) and all(isinstance(item, str) for item in value + stored):
                value, stored = sorted(value), sorted(stored)  # the same modules in another order (modules_change sorts them)
            if value is not None and value != stored:  # posted back unchanged: as before
                try:
                    picks[key] = model_selection.check_value(key, picks[key])  # module file names, as PATCH stores them
                except ParamError as error:
                    problems.append(ParamError(error.message, owner="settings", param=key))
        # Remi: a changed value is checked as the settings page checks it (kind, range, choices); one
        # of the wrong kind reached upstream readers and broke them until the next start
        try:
            req = check_api_values(shared.opts, {key: value for key, value in req.items() if key not in frozen})
        except ValidationError as error:
            problems += error.errors
        if problems:  # every refused value, in the order sent
            problems.sort(key=lambda problem: order.index(problem.param) if problem.param in order else len(order))
            raise HTTPException(status_code=422, detail=f"invalid settings: {ValidationError(problems)}"[:1000])
        if shared.cmd_opts.freeze_settings:  # nothing left to change, and Options.save (also a pick's) refuses to save
            return
        before = {key: shared.opts.data.get(key) for key in req}
        set_config(req, is_api=True)  # restrict_api settings (URL fetching) stay out of the API's reach
        changed = [key for key in req if shared.opts.data.get(key) != before[key]]
        for key, value in picks.items():
            if value is not None and model_selection.select(key, value):  # applied now (and saved) when no generation runs
                changed.append(key)
        # Remi: announced as PATCH /api/v2/options announces a change, or open pages keep the old values
        if changed and (services := getattr(self.app.state, "services", None)) is not None:
            services.bus.publish("options.changed", keys=sorted(set(changed)))

    def get_cmd_flags(self):
        from modules import redact

        flags = dict(vars(shared.cmd_opts))
        for name in ("auth", "api_auth", "ngrok", "ngrok_options"):
            if flags.get(name):
                flags[name] = {redact.HIDDEN: redact.HIDDEN} if isinstance(flags[name], dict) else redact.HIDDEN
        return flags

    def get_samplers(self):
        return [{"name": sampler[0], "aliases": sampler[2], "options": sampler[3]} for sampler in sd_samplers.all_samplers]

    def get_schedulers(self):
        return [
            {
                "name": scheduler.name,
                "label": scheduler.label,
                "aliases": scheduler.aliases,
                "default_rho": scheduler.default_rho,
                "need_inner_model": scheduler.need_inner_model,
            }
            for scheduler in sd_schedulers.schedulers
        ]

    def get_upscalers(self):
        return [
            {
                "name": upscaler.name,
                "model_name": upscaler.scaler.model_name,
                "model_path": upscaler.data_path,
                "model_url": None,
                "scale": upscaler.scale,
            }
            for upscaler in shared.sd_upscalers
        ]

    def get_latent_upscale_modes(self):
        return [
            {
                "name": upscale_mode,
            }
            for upscale_mode in [*(shared.latent_upscale_modes or {})]
        ]

    def get_sd_models(self):
        import modules.sd_models as sd_models

        return [{"title": x.title, "model_name": x.model_name, "hash": x.shorthash, "sha256": x.sha256, "filename": x.filename, "config": getattr(x, "config", None)} for x in sd_models.checkpoints_list.values()]

    def get_sd_vaes_and_text_encoders(self):
        from modules_forge.main_entry import module_list

        return [{"model_name": x, "filename": module_list[x]} for x in module_list.keys()]

    def get_face_restorers(self):
        return [{"name": x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    def get_prompt_styles(self):
        styleList = []
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append({"name": style[0], "prompt": style[1], "negative_prompt": style[2]})

        return styleList

    def get_embeddings(self):
        def convert_embedding(embedding):
            return {
                "step": embedding.step,
                "sd_checkpoint": embedding.sd_checkpoint,
                "sd_checkpoint_name": embedding.sd_checkpoint_name,
                "shape": embedding.shape,
                "vectors": embedding.vectors,
            }

        def convert_embeddings(embeddings):
            return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}

        return {
            "loaded": convert_embeddings(self.embedding_db.word_embeddings),
            "skipped": convert_embeddings(self.embedding_db.skipped_embeddings),
        }

    def refresh_embeddings(self):
        with self.queue_lock:
            self.embedding_db.load_textual_inversion_embeddings(force_reload=True, sync_with_sd_model=False)

    def refresh_checkpoints(self):
        with self.queue_lock:
            shared.refresh_checkpoints()

    def refresh_vae(self):
        with self.queue_lock:
            shared_items.refresh_vae_list()

    def get_memory(self):
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            res = process.memory_info()  # only rss is cross-platform guaranteed so we dont rely on other values
            ram_total = 100 * res.rss / process.memory_percent()  # and total memory is calculated as actual value is not cross-platform safe
            ram = {"free": ram_total - res.rss, "used": res.rss, "total": ram_total}
        except Exception as err:
            ram = {"error": f"{err}"}
        try:
            import torch

            if torch.cuda.is_available():
                _backend = torch.cuda
            elif torch.xpu.is_available():
                _backend = torch.xpu
            else:
                _backend = None

            if _backend is not None:
                s = _backend.mem_get_info()
                system = {"free": s[0], "used": s[1] - s[0], "total": s[1]}
                s = dict(_backend.memory_stats(shared.device))
                allocated = {"current": s["allocated_bytes.all.current"], "peak": s["allocated_bytes.all.peak"]}
                reserved = {"current": s["reserved_bytes.all.current"], "peak": s["reserved_bytes.all.peak"]}
                active = {"current": s.get("active_bytes.all.current") or s.get("active.all.current") or 0, "peak": s.get("active_bytes.all.peak") or s.get("active.all.peak") or 0}
                inactive = {"current": s["inactive_split_bytes.all.current"], "peak": s["inactive_split_bytes.all.peak"]}
                warnings = {"retries": s["num_alloc_retries"], "oom": s["num_ooms"]}
                cuda = {
                    "device": str(shared.device),
                    "system": system,
                    "active": active,
                    "allocated": allocated,
                    "reserved": reserved,
                    "inactive": inactive,
                    "events": warnings,
                }
            else:
                cuda = {"error": "unavailable"}
        except Exception as err:
            cuda = {"error": f"{err}"}
        return models.MemoryResponse(ram=ram, cuda=cuda)

    def get_extensions_list(self):
        from modules import extensions, redact

        extensions.list_extensions()
        ext_list = []
        for ext in extensions.extensions:
            ext: extensions.Extension
            ext.read_info_from_repo()
            if ext.remote is not None:
                remote = redact.url(ext.remote)
                if "://" not in remote:
                    remote = re.sub(r"^([^:@/\s]+):[^@/\s]+@([^:/\s]+:)", rf"\1:{redact.HIDDEN}@\2", remote)
                ext_list.append(
                    {
                        "name": ext.name,
                        "remote": remote,
                        "branch": ext.branch,
                        "commit_hash": ext.commit_hash,
                        "commit_date": ext.commit_date,
                        "version": ext.version,
                        "enabled": ext.enabled,
                    }
                )
        return ext_list

    def launch(self, server_name, port, root_path):
        self.app.include_router(self.router)
        uvicorn.run(
            self.app,
            host=server_name,
            port=port,
            timeout_keep_alive=shared.cmd_opts.timeout_keep_alive,
            root_path=root_path,
            ssl_keyfile=shared.cmd_opts.tls_keyfile,
            ssl_certfile=shared.cmd_opts.tls_certfile,
        )

    def kill_webui(self):
        restart.stop_program()

    def restart_webui(self):
        if restart.is_restartable():
            services = getattr(getattr(self.app, "state", None), "services", None)
            if services is not None:
                queue = getattr(services, "queue", None)
                if queue is not None:
                    try:
                        queue.stop(timeout=30, interrupt=True)
                    except Exception:
                        pass
                if hasattr(services, "close"):
                    try:
                        services.close()
                    except Exception:
                        pass
                else:
                    try:
                        services.stop()
                    except Exception:
                        pass
                    workers = getattr(services, "workers", None)
                    if workers is not None:
                        try:
                            workers.shutdown()
                        except Exception:
                            pass
            restart.restart_program()
        return Response(status_code=501)

    def stop_webui(self):
        _require_state().server_command = "stop"
        return Response("Stopping.")
