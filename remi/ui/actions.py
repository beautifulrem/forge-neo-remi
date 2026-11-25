"""Server actions: ``@ui.action`` callables returning patches.

An action receives ``(self, ctx, values)``: the owner's current named values
(plus ``ctx.core``, the generation form) and returns a patch
(schema-format.md §5)::

    {"param": {"value": 3, "enabled": False}, "core.width": {"value": 1024},
     "__toast__": {"level": "info", "message": "..."}, "__outputs__": {...},
     "__settings__": {setting: value}, "__refresh__": {"keys": ["upscalers"]}}

``__settings__`` reports settings the action has already changed on the
server (the client refreshes its copy; it must not write them back).
``__refresh__`` names server-side lists the action rescanned (``"models"``,
``"upscalers"``, ...); the server announces them as ``options.changed`` so
every client refetches the schemas that offer them. Actions never reach the
server's services themselves.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from remi.ui.errors import ParamError, Refused, SchemaError, ValidationError
from remi.ui.sentence import Sentence
from remi.ui.types import Patch, PatchEntry

log = logging.getLogger(__name__)

PATCH_KEYS = frozenset({"value", "choices", "visible", "enabled", "label", "info", "hint", "minimum", "maximum", "step"})
SPECIAL_KEYS = frozenset({"__toast__", "__outputs__", "__settings__", "__refresh__", "__advice__", "__capabilities__"})
TOAST_LEVELS = ("info", "success", "warning", "error")


class ActionTimeoutError(TimeoutError):
    """Raised when an isolated ``@ui.action`` exceeds its timeout."""


@dataclass(frozen=True)
class ActionSpec:
    name: str
    func: Callable[..., Any]
    gpu: bool = False
    """Queue behind generation (runs under the generation lock)."""
    timeout: float | None = None
    """Timeout in seconds, run isolated (``None``: no time limit). Ignored with
    ``gpu``: the action holds the generation lock until it really ends."""


@dataclass
class ActionContext:
    """What an action gets besides its owner's values."""

    tab: str | None = None
    owner: str = ""
    core: dict[str, Any] = field(default_factory=dict)
    scripts: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Other owners' values, by owner id, when the caller sent them."""
    user: str | None = None
    request: Any = None
    """The HTTP request (``fastapi.Request``) when the server runs the action."""
    extra: dict[str, Any] = field(default_factory=dict)
    files: Any = None
    """Uploads and outputs by id (``remi.ui.files.ActionFiles``: ``get``,
    ``image``, ``image_size``, ``path``, ``store_image``, ``store_bytes``);
    filled by the server, ``None`` elsewhere."""
    generation_lock: Any = None
    """The lock generation runs under (filled by the server; ``gpu=True``
    page tasks hold it). ``None``: the WebUI's ``call_queue.queue_lock``."""


def action(name: str | None = None, *, gpu: bool = False, timeout: float | None = None):
    """Register a method (or function) as an action named ``name``
    (default: the function name)."""

    def decorate(func):
        func.__remi_action__ = ActionSpec(name or func.__name__, func, gpu, timeout)
        return func

    if callable(name):  # used bare: @ui.action
        func, name = name, None
        return decorate(func)
    return decorate


def actions_of(owner: Any) -> dict[str, ActionSpec]:
    """The actions declared on ``owner``'s class (or module/namespace)."""

    found: dict[str, ActionSpec] = {}
    source = owner if inspect.ismodule(owner) or isinstance(owner, type) else type(owner)
    for attribute in dir(source):
        try:
            member = inspect.getattr_static(source, attribute)
        except AttributeError:
            continue
        member = getattr(member, "__func__", member)
        spec = getattr(member, "__remi_action__", None)
        if isinstance(spec, ActionSpec):
            if spec.name in found:
                raise SchemaError(f"action {spec.name!r} is declared twice on {source!r}")
            found[spec.name] = spec
    return found


def call_action(owner: Any, name: str, ctx: ActionContext, values: Mapping[str, Any] | None) -> Patch:
    """Run ``owner``'s action ``name``; returns the validated patch."""

    spec = actions_of(owner).get(name)
    if spec is None:
        raise KeyError(f"no action {name!r}")
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValidationError([ParamError(Sentence("expected an object of named values, got {{type}}", type=type(values).__name__), owner=ctx.owner or None)])
    bound = spec.func if inspect.ismodule(owner) else spec.func.__get__(owner, type(owner))
    return check_patch(bound(ctx, dict(values)) or {})


def _answered(error: BaseException) -> bool:
    """An error Remi answers as the request's fault (a 4xx), not as the
    extension's: an ``HTTPException``, a rejected upload (``FileError``), a
    queue operation that cannot be done (``JobError``), another refused
    operation (``Refused``). A status on any other
    error is someone else's: an action whose call to a remote service got a 401
    or a 429 back (``urllib``'s ``HTTPError.status``, an SDK's
    ``status_code``) has failed."""

    from starlette.exceptions import HTTPException

    from remi.files import FileError
    from remi.jobs import JobError  # imported here: the job queue imports remi.ui

    if isinstance(error, HTTPException):
        status = error.status_code
    elif isinstance(error, (FileError, JobError, Refused)):
        status = error.status
    else:
        return False
    return 400 <= status < 500


def call_action_isolated(
    owner: Any,
    name: str,
    ctx: ActionContext,
    values: Mapping[str, Any] | None,
    *,
    timeout: float | None = None,
    extension: str | None = None,
    return_toast: bool = True,
) -> Patch:
    """Run ``owner``'s action ``name`` with timeout and error isolation.

    Client input errors (:class:`ValidationError`, or one of Remi's own errors
    with a 4xx status: see ``_answered``) and missing actions
    (:class:`KeyError`) propagate normally. Runtime failures and timeouts are
    recorded in :data:`remi.ui.manifest.diagnostics` and, when
    ``return_toast=True``, turned into a safe error toast patch so a buggy
    extension action never crashes the rest of the page or server.
    """

    from remi.ui.manifest import diagnostics

    spec = actions_of(owner).get(name)
    if spec is None:
        raise KeyError(f"no action {name!r}")
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValidationError([ParamError(Sentence("expected an object of named values, got {{type}}", type=type(values).__name__), owner=ctx.owner or None)])

    effective_timeout = timeout if timeout is not None else spec.timeout
    diag_owner = extension or ctx.owner or type(owner).__name__
    bound = spec.func if inspect.ismodule(owner) else spec.func.__get__(owner, type(owner))

    def _run() -> Patch:
        return check_patch(bound(ctx, dict(values)) or {})

    if effective_timeout is None or effective_timeout <= 0:
        try:
            return _run()
        except ValidationError:
            raise
        except Exception as error:
            if _answered(error):
                raise
            log.warning("isolated action %s.%s failed: %s", diag_owner, name, error, exc_info=True)
            diagnostics.record_action_error(diag_owner, name, f"{type(error).__name__}: {error}", timed_out=False)
            if not return_toast:
                raise
            return toast(message("Action {{name}} failed: {{error}}", name=name, error=str(error)), level="error")

    result_box: dict[str, Any] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            result_box["patch"] = _run()
        except BaseException as error:
            result_box["error"] = error
        finally:
            done.set()

    thread = threading.Thread(target=_worker, name=f"remi-action-{name}", daemon=True)
    thread.start()
    if not done.wait(effective_timeout):
        msg = f"timed out after {effective_timeout:.1f}s"
        log.warning("isolated action %s.%s %s", diag_owner, name, msg)
        diagnostics.record_action_error(diag_owner, name, msg, timed_out=True)
        if not return_toast:
            raise ActionTimeoutError(f"action {name!r} {msg}")
        return toast(message("Action {{name}} timed out after {{seconds}} s", name=name, seconds=f"{effective_timeout:.1f}"), level="error")

    error = result_box.get("error")
    if error is not None:
        if isinstance(error, ValidationError) or _answered(error):
            raise error
        log.warning("isolated action %s.%s failed: %s", diag_owner, name, error)
        diagnostics.record_action_error(diag_owner, name, f"{type(error).__name__}: {error}", timed_out=False)
        if not return_toast:
            raise error
        return toast(message("Action {{name}} failed: {{error}}", name=name, error=str(error)), level="error")
    return result_box["patch"]


def update(**props: Any) -> PatchEntry:
    """One param's patch: ``update(value=1, enabled=False)``."""

    unknown = set(props) - PATCH_KEYS
    if unknown:
        raise SchemaError(f"unknown patch keys {sorted(unknown)}")
    if "choices" in props:
        props["choices"] = normalize_choices(props["choices"])
    return props


def message(text: str, **params: Any) -> dict:
    """A translatable text for an ``output`` node (schema-format.md §5):
    ``text`` is the English catalog key with ``{{name}}`` placeholders the
    client fills from ``params`` after translating it."""

    if not isinstance(text, str):
        raise SchemaError("message text is a string")
    return {"text": text, "params": params} if params else {"text": text}


def toast(message: str | Mapping[str, Any], level: str = "info") -> dict:
    """A toast; the client translates ``message`` when a catalog has it.
    ``message`` may be a :func:`message` (a catalog key with ``{{name}}``
    placeholders and their ``params``) or a :class:`Sentence` (sent as its
    template and values) for texts with variable parts."""

    if level not in TOAST_LEVELS:
        raise SchemaError(f"toast level is one of {TOAST_LEVELS}")
    if isinstance(message, Sentence) and message.params:
        return {"__toast__": {"level": level, "message": message_of({"text": message.template, "params": message.params})}}
    if isinstance(message, Mapping):
        return {"__toast__": {"level": level, "message": message_of(message)}}
    return {"__toast__": {"level": level, "message": str(message)}}


def message_of(value: Mapping[str, Any]) -> dict:
    """A validated :func:`message` dict."""

    params = value.get("params")
    if not isinstance(value.get("text"), str) or not (params is None or isinstance(params, Mapping)):
        raise SchemaError("a message is {'text': str, 'params'?: dict}")
    return message(value["text"], **dict(params or {}))


def message_english(value: str | Mapping[str, Any]) -> str:
    """The English text of a string or :func:`message` (placeholders filled)."""

    if not isinstance(value, Mapping):
        return str(value)
    text = str(value.get("text", ""))
    for name, param in (value.get("params") or {}).items():
        text = text.replace("{{" + str(name) + "}}", str(param))
    return text


def normalize_choices(choices: Any) -> list[dict]:
    """``[{"value", "label"}]`` from values, ``(label, value)`` pairs or
    already normalized entries."""

    from remi.ui.nodes import _choice_pair

    normalized = []
    for item in choices:
        if isinstance(item, Mapping) and set(item) == {"value", "label"}:
            normalized.append(dict(item))
        else:
            label, value = _choice_pair(item)
            normalized.append({"value": value, "label": label})
    return normalized


def check_patch(patch: Mapping[str, Any]) -> Patch:
    """Validate a patch's shape; choices are normalized to ``{value, label}``."""

    if not isinstance(patch, Mapping):
        raise SchemaError(f"an action returns a patch dict, got {type(patch).__name__}")
    result: Patch = {}
    for key, entry in patch.items():
        if key in SPECIAL_KEYS:
            if key == "__advice__":
                if not isinstance(entry, list) or not all(isinstance(item, Mapping) for item in entry):
                    raise SchemaError("__advice__ takes a list of objects")
                result[key] = [dict(item) for item in entry]
                continue
            if not isinstance(entry, Mapping):
                raise SchemaError(f"{key} takes an object")
            if key == "__toast__" and entry.get("level", "info") not in TOAST_LEVELS:
                raise SchemaError(f"toast level is one of {TOAST_LEVELS}")
            if key == "__toast__" and "parts" in entry:
                parts = entry["parts"]
                if not isinstance(parts, list) or not all(isinstance(part, str | Mapping) for part in parts):
                    raise SchemaError("toast parts are a list of texts and messages")
                result[key] = {**entry, "parts": [part if isinstance(part, str) else message_of(part) for part in parts]}
                continue
            if key == "__refresh__":
                keys = entry.get("keys")
                if not isinstance(keys, list | tuple) or not all(isinstance(k, str) and k for k in keys):
                    raise SchemaError("__refresh__ takes {\"keys\": [name, ...]}")
                result[key] = {"keys": list(keys)}
                continue
            result[key] = dict(entry)
            continue
        if not isinstance(key, str) or key.startswith("__"):
            raise SchemaError(f"unknown patch entry {key!r}")
        if not isinstance(entry, Mapping):
            raise SchemaError(f"patch for {key!r} must be an object")
        result[key] = update(**entry)
    return result


def toast_parts(entry: Mapping[str, Any]) -> list[str | dict]:
    """The messages a toast is made of: its ``parts`` when it was merged from several, else its
    one ``message`` (a :class:`Sentence` as its template and values, as :func:`toast` sends it)."""

    if isinstance(entry.get("parts"), list):
        return list(entry["parts"])
    value = entry.get("message", "")
    if isinstance(value, Sentence) and value.params:
        return [message(value.template, **value.params)]
    if isinstance(value, Mapping):
        return [message_of(value)]
    return [str(value)]


def merge_patches(*patches: Mapping[str, Any]) -> Patch:
    """Later patches win per param property; toasts concatenate: the English ``message``
    joins them with "; " and ``parts`` lists each one's message (a catalog key or a
    :func:`message`), which the client translates one by one (schema-format.md §5)."""

    merged: dict[str, Any] = {}
    for patch in patches:
        for key, entry in patch.items():
            if key == "__toast__" and key in merged:
                previous = merged[key]
                parts = [*toast_parts(previous), *toast_parts(entry)]
                merged[key] = {"level": entry.get("level", previous.get("level", "info")), "message": "; ".join(message_english(part) for part in parts), "parts": parts}
            elif key == "__advice__":
                existing = list(merged.get(key) or [])
                seen_codes = {item.get("code") for item in existing if isinstance(item, Mapping) and item.get("code")}
                for item in entry if isinstance(entry, list) else []:
                    code = item.get("code") if isinstance(item, Mapping) else None
                    if not code or code not in seen_codes:
                        existing.append(dict(item) if isinstance(item, Mapping) else item)
                        if code:
                            seen_codes.add(code)
                merged[key] = existing
            else:
                merged[key] = {**merged.get(key, {}), **entry}
    return merged
