"""Extension manifest (``remi-extension.json``) schema and runtime diagnostics.

An extension may ship ``remi-extension.json`` at its root directory declaring
its required ``remi.ui`` API version, minimum Remi version, model capabilities,
custom widgets, license and homepage metadata.

This module also hosts the lightweight process-wide :class:`DiagnosticsCollector`
that tracks per-extension load times, hook warnings/errors, and isolated action
errors/timeouts without coupling ``remi.ui`` to ``remi.server``.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from remi.ui.version import REMI_API_VERSION, check_api_compatibility, parse_semver

MANIFEST_FILENAME = "remi-extension.json"
ManifestStatus = Literal["valid", "missing", "invalid", "incompatible"]


@dataclass(frozen=True)
class ExtensionManifest:
    name: str | None = None
    version: str | None = None
    api_version: str = REMI_API_VERSION
    min_remi_version: str | None = None
    requires_capabilities: tuple[str, ...] = ()
    requires_widgets: tuple[str, ...] = ()
    license: str | None = None
    homepage: str | None = None
    description: str | None = None
    author: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "api_version": self.api_version,
            "min_remi_version": self.min_remi_version,
            "requires_capabilities": list(self.requires_capabilities),
            "requires_widgets": list(self.requires_widgets),
            "license": self.license,
            "homepage": self.homepage,
            "description": self.description,
            "author": self.author,
        }


@dataclass(frozen=True)
class ManifestValidationResult:
    manifest: ExtensionManifest | None
    status: ManifestStatus
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    compatible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "compatible": self.compatible,
            "manifest": self.manifest.to_dict() if self.manifest is not None else None,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _optional_str(data: Mapping[str, Any], key: str, errors: list[str]) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str):
        errors.append(f"{key} must be a string")
        return None
    return value.strip() or None


def _string_tuple(data: Mapping[str, Any], key: str, errors: list[str]) -> tuple[str, ...]:
    if key not in data or data[key] is None:
        return ()
    raw = data[key]
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        errors.append(f"{key} must be a list of non-empty strings")
        return ()
    return tuple(dict.fromkeys(item.strip() for item in raw))


def validate_manifest_dict(
    data: Any,
    *,
    known_widgets: Iterable[str] | None = None,
    known_capabilities: Iterable[str] | None = None,
    api_version: str = REMI_API_VERSION,
) -> ManifestValidationResult:
    """Validate a parsed ``remi-extension.json`` mapping. Required widgets and
    capabilities missing from ``known_widgets`` / ``known_capabilities`` (when
    given) are warnings."""

    if not isinstance(data, Mapping):
        return ManifestValidationResult(None, "invalid", ("manifest root must be a JSON object",), (), False)

    errors: list[str] = []
    warnings: list[str] = []

    raw_api = data.get("api_version")
    if raw_api is None:
        errors.append("api_version is required")
        api_req = ""
    elif not isinstance(raw_api, str) or not raw_api.strip():
        errors.append("api_version must be a non-empty string")
        api_req = ""
    else:
        api_req = raw_api.strip()

    name = _optional_str(data, "name", errors)
    version = _optional_str(data, "version", errors)
    if version is not None:
        try:
            parse_semver(version)
        except ValueError:
            warnings.append(f"version {version!r} is not standard SemVer")

    min_remi_version = _optional_str(data, "min_remi_version", errors)
    if min_remi_version is not None:
        try:
            parse_semver(min_remi_version)
        except ValueError:
            errors.append(f"min_remi_version {min_remi_version!r} is not valid SemVer")

    requires_capabilities = _string_tuple(data, "requires_capabilities", errors)
    requires_widgets = _string_tuple(data, "requires_widgets", errors)
    license_id = _optional_str(data, "license", errors)
    homepage = _optional_str(data, "homepage", errors)
    description = _optional_str(data, "description", errors)
    author = _optional_str(data, "author", errors)

    if homepage is not None and not homepage.startswith(("https://", "http://")):
        errors.append("homepage must start with https:// or http://")

    if known_widgets is not None and requires_widgets:
        available = set(known_widgets)
        missing_widgets = [w for w in requires_widgets if w not in available]
        if missing_widgets:
            warnings.append(f"unknown required widgets: {', '.join(missing_widgets)}")
    if known_capabilities is not None and requires_capabilities:
        available = set(known_capabilities)
        missing_capabilities = [c for c in requires_capabilities if c not in available]
        if missing_capabilities:
            warnings.append(f"unknown required capabilities: {', '.join(missing_capabilities)}")

    if errors:
        return ManifestValidationResult(None, "invalid", tuple(errors), tuple(warnings), False)

    compat = check_api_compatibility(api_req, current=api_version)
    manifest = ExtensionManifest(
        name=name,
        version=version,
        api_version=api_req,
        min_remi_version=min_remi_version,
        requires_capabilities=requires_capabilities,
        requires_widgets=requires_widgets,
        license=license_id,
        homepage=homepage,
        description=description,
        author=author,
    )
    if not compat.compatible:
        return ManifestValidationResult(
            manifest,
            "incompatible",
            (compat.reason or f"incompatible api_version {api_req!r}",),
            tuple(warnings),
            False,
        )
    return ManifestValidationResult(manifest, "valid", (), tuple(warnings), True)


def load_manifest(
    extension_dir: str | Path,
    *,
    known_widgets: Iterable[str] | None = None,
    known_capabilities: Iterable[str] | None = None,
    api_version: str = REMI_API_VERSION,
) -> ManifestValidationResult:
    """Read and validate ``<extension_dir>/remi-extension.json`` if present."""

    path = Path(extension_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return ManifestValidationResult(None, "missing", (), (), True)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return ManifestValidationResult(None, "invalid", (f"cannot parse {MANIFEST_FILENAME}: {error}",), (), False)
    return validate_manifest_dict(raw, known_widgets=known_widgets, known_capabilities=known_capabilities, api_version=api_version)


@dataclass
class ExtensionDiagnosticsRecord:
    extension: str
    load_time_ms: float | None = None
    hook_warnings: int = 0
    hook_errors: int = 0
    action_errors: int = 0
    action_timeouts: int = 0
    last_error: str | None = None
    last_error_at: float | None = None
    recent_errors: list[dict[str, Any]] = field(default_factory=list)


MAX_RECENT_ERRORS = 10


class DiagnosticsCollector:
    """Thread-safe collector for per-extension runtime health metrics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, ExtensionDiagnosticsRecord] = {}

    def _entry(self, extension: str) -> ExtensionDiagnosticsRecord:
        record = self._records.get(extension)
        if record is None:
            record = self._records[extension] = ExtensionDiagnosticsRecord(extension=extension)
        return record

    def record_load_time(self, extension: str, duration_ms: float) -> None:
        with self._lock:
            self._entry(extension).load_time_ms = round(max(0.0, float(duration_ms)), 2)

    def record_hook_warning(self, extension: str, message: str) -> None:
        now = time.time()
        with self._lock:
            rec = self._entry(extension)
            rec.hook_warnings += 1
            rec.last_error = message[:500]
            rec.last_error_at = now
            self._push_error(rec, "hook_warning", message, now)

    def record_hook_error(self, extension: str, message: str) -> None:
        now = time.time()
        with self._lock:
            rec = self._entry(extension)
            rec.hook_errors += 1
            rec.last_error = message[:500]
            rec.last_error_at = now
            self._push_error(rec, "hook_error", message, now)

    def record_action_error(self, extension: str, action_name: str, message: str, *, timed_out: bool = False) -> None:
        now = time.time()
        detail = f"action {action_name}: {message}"[:500]
        with self._lock:
            rec = self._entry(extension)
            if timed_out:
                rec.action_timeouts += 1
            else:
                rec.action_errors += 1
            rec.last_error = detail
            rec.last_error_at = now
            self._push_error(rec, "action_timeout" if timed_out else "action_error", detail, now)

    @staticmethod
    def _push_error(rec: ExtensionDiagnosticsRecord, kind: str, message: str, timestamp: float) -> None:
        rec.recent_errors.append({"kind": kind, "message": message[:500], "timestamp": timestamp})
        if len(rec.recent_errors) > MAX_RECENT_ERRORS:
            del rec.recent_errors[:-MAX_RECENT_ERRORS]

    def get(self, extension: str) -> ExtensionDiagnosticsRecord | None:
        with self._lock:
            rec = self._records.get(extension)
            if rec is None:
                return None
            return ExtensionDiagnosticsRecord(
                extension=rec.extension,
                load_time_ms=rec.load_time_ms,
                hook_warnings=rec.hook_warnings,
                hook_errors=rec.hook_errors,
                action_errors=rec.action_errors,
                action_timeouts=rec.action_timeouts,
                last_error=rec.last_error,
                last_error_at=rec.last_error_at,
                recent_errors=list(rec.recent_errors),
            )

    def snapshot(self) -> dict[str, ExtensionDiagnosticsRecord]:
        with self._lock:
            result: dict[str, ExtensionDiagnosticsRecord] = {}
            for name in list(self._records):
                rec = self.get(name)
                if rec is not None:
                    result[name] = rec
            return result

    def reset(self, extension: str | None = None) -> None:
        with self._lock:
            if extension is None:
                self._records.clear()
            else:
                self._records.pop(extension, None)


diagnostics = DiagnosticsCollector()
"""Process-wide extension diagnostics collector."""


def extension_of(path: str | None) -> str | None:
    """The folder under ``extensions/`` or ``extensions-builtin/`` that ``path``
    lives in: the name Extensions → Diagnostics lists the extension by. None for
    a file outside both (the WebUI's own ``scripts/``)."""

    if not path:
        return None
    from modules.paths_internal import extensions_builtin_dir, extensions_dir
    from remi.files import path_inside

    # as written first: an extension folder may be a link to a checkout elsewhere
    for resolve in (os.path.abspath, os.path.realpath):
        where = resolve(path)
        for root in (extensions_dir, extensions_builtin_dir):
            if below := path_inside(where, resolve(root)):
                return below.split(os.sep)[0]
    return None


def extension_of_owner(owner: Any) -> str | None:
    """The extension an action's owner comes from: a script or an Extras
    operation by its ``filename`` (a page by the file that built it), a module
    or a class by its source file."""

    path = getattr(owner, "filename", None)
    if not isinstance(path, str) or not path:
        try:
            path = inspect.getfile(owner if inspect.ismodule(owner) else type(owner))
        except (TypeError, OSError):  # a builtin, or a class without a module file
            path = None
    return extension_of(path)



def not_ported() -> list[dict[str, Any]]:
    """What could not be loaded: script files that failed to import, scripts
    and Extras operations left out by their runner (no ``params()``, or it
    failed), with the reason and the extension they come from (bootstrap's
    ``not_ported``, Extensions → Diagnostics)."""

    from modules import scripts

    found: list[dict[str, Any]] = []
    for scriptfile, reason in getattr(scripts, "scripts_not_loaded", []):
        found.append({"kind": "script", "id": scriptfile.filename, "title": scriptfile.filename, "extension": extension_of(scriptfile.path), "status": "disabled", "reason": reason})
    for tab, runner in (("txt2img", scripts.scripts_txt2img), ("img2img", scripts.scripts_img2img)):
        for script, reason in getattr(runner, "disabled_scripts", None) or []:
            found.append({"kind": "script", "id": getattr(script, "script_id", "") or "", "title": _script_title(script), "tab": tab, "extension": extension_of(script.filename), "status": "disabled", "reason": reason})
    for operation, reason in getattr(scripts.scripts_postproc, "disabled_scripts", None) or []:
        found.append({"kind": "extras", "id": "", "title": operation.name or type(operation).__name__, "tab": "extras", "extension": extension_of(operation.filename), "status": "disabled", "reason": reason})
    return found


def _script_title(script: Any) -> str:
    try:
        return str(script.title())
    except Exception:
        return os.path.basename(script.filename or "script")

__all__ = [
    "MANIFEST_FILENAME",
    "DiagnosticsCollector",
    "ExtensionDiagnosticsRecord",
    "ExtensionManifest",
    "ManifestStatus",
    "ManifestValidationResult",
    "diagnostics",
    "load_manifest",
    "not_ported",
    "validate_manifest_dict",
]
