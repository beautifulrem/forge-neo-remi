"""Test support: a minimal initialised WebUI for the ``remi`` tests.

Default options (no config file), an empty style database and txt2img /
img2img script runners holding only the built-in processing scripts; no
model, extension or UI is loaded. Everything is restored on exit, so suites
that expect an uninitialised WebUI are unaffected.
"""

from __future__ import annotations

import ast
import importlib
import json
import warnings
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROCESSING_SCRIPTS = ("seed", "sampler", "refiner", "comments", "mahiro", "rescale_cfg")


def processing_script_classes(names=PROCESSING_SCRIPTS) -> list:
    from modules import scripts

    classes = []
    for name in names:
        module = importlib.import_module(f"modules.processing_scripts.{name}")
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, scripts.Script) and value.__module__ == module.__name__:
                classes.append(scripts.ScriptClassData(value, module.__file__, str(Path(module.__file__).parents[2]), module))
    return classes


@contextmanager
def webui_state(options: dict | None = None, *, script_classes=None, postprocessing=()):
    """Yield ``SimpleNamespace(opts, txt2img, img2img, postproc)`` with
    ``options`` overriding the defaults. ``script_classes``: extra
    ``ScriptClassData`` (or classes) to load after the processing scripts;
    ``postprocessing``: the Extras operation classes (none by default)."""

    from modules import options as options_module
    from modules import shared, shared_options

    saved = {name: getattr(shared, name) for name in ("opts", "options_templates", "prompt_styles")}
    shared.options_templates = shared_options.options_templates
    shared.opts = options_module.Options(shared_options.options_templates, shared_options.restricted_opts)
    try:
        for key, value in (options or {}).items():
            shared.opts.data[key] = value

        import modules.processing  # noqa: F401 - the WebUI's import order (sd_models <-> processing)
        from modules import scripts, scripts_postprocessing, styles

        shared.prompt_styles = styles.StyleDatabase([str(Path(shared.data_path) / "__remi_tests_no_styles__.csv")])
        saved_scripts = (scripts.scripts_txt2img, scripts.scripts_img2img, scripts.scripts_postproc, list(scripts.scripts_data))
        try:
            scripts.scripts_data[:] = processing_script_classes()
            for extra in script_classes or ():
                if isinstance(extra, type):
                    extra = scripts.ScriptClassData(extra, f"{extra.__name__}.py", "", None)
                scripts.scripts_data.append(extra)
            scripts.scripts_txt2img = scripts.ScriptRunner()
            scripts.scripts_img2img = scripts.ScriptRunner()
            scripts.scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()
            scripts.scripts_postproc.scripts = []
            for cls in postprocessing:
                operation = cls()
                operation.filename = f"{cls.__name__}.py"
                scripts.scripts_postproc.scripts.append(operation)
            scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
            scripts.scripts_img2img.initialize_scripts(is_img2img=True)
            yield SimpleNamespace(opts=shared.opts, txt2img=scripts.scripts_txt2img, img2img=scripts.scripts_img2img, postproc=scripts.scripts_postproc)
        finally:
            scripts.scripts_txt2img, scripts.scripts_img2img, scripts.scripts_postproc, data = saved_scripts
            scripts.scripts_data[:] = data
    finally:
        for name, value in saved.items():
            setattr(shared, name, value)


# region Public extension contract verification


@dataclass(frozen=True)
class ExtensionContractReport:
    """Result of :func:`verify_extension_contract`."""

    ok: bool
    extension_dir: str
    script_count: int = 0
    manifest_status: str = "missing"
    tabs_checked: tuple[str, ...] = ()
    param_counts: dict[str, int] = field(default_factory=dict)
    ui_strings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "extension_dir": self.extension_dir,
            "script_count": self.script_count,
            "manifest_status": self.manifest_status,
            "tabs_checked": list(self.tabs_checked),
            "param_counts": dict(self.param_counts),
            "ui_strings": list(self.ui_strings),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


_BANNED_IMPORT_ROOTS = ("gradio", "gradio_client", "gradio_rangeslider")
_TEXT_KEYS = ("label", "info", "title", "placeholder")


def _extract_imports(source: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def collect_ui_strings(schema: Any) -> set[str]:
    """Extract all translatable English strings from an exported ``remi.ui`` schema."""

    found: set[str] = set()
    if isinstance(schema, list):
        for item in schema:
            found |= collect_ui_strings(item)
        return found
    if not isinstance(schema, dict):
        return found
    for key in _TEXT_KEYS:
        value = schema.get(key)
        if isinstance(value, str) and value.strip():
            found.add(value)
    if schema.get("type") in ("markdown", "html") and isinstance(schema.get("text"), str) and schema["text"].strip():
        found.add(schema["text"])
    for choice in schema.get("choices") or ():
        label = choice.get("label") if isinstance(choice, dict) else None
        if isinstance(label, str) and label.strip() and not label.replace(".", "", 1).lstrip("-").isdigit():
            found.add(label)
    for key in ("enable", "children"):
        if key in schema:
            found |= collect_ui_strings(schema[key])
    return found


def verify_extension_contract(
    extension_dir: str | Path,
    *,
    script_classes: Iterable[type] | None = None,
    tabs: Sequence[str] = ("txt2img", "img2img"),
    check_manifest: bool = True,
    require_manifest: bool = False,
    check_no_gradio: bool = True,
    check_no_server_imports: bool = True,
    check_locales: Iterable[str] = (),
    known_widgets: Iterable[str] | None = None,
) -> ExtensionContractReport:
    """Public contract verifier for third-party and built-in Remi extensions.

    Verifies:
    1. Zero ``gradio`` imports across the extension's Python files.
    2. Zero ``remi.server`` imports (layering boundary).
    3. ``remi-extension.json`` manifest schema & SemVer compatibility (if present or required).
    4. Every supplied ``Script`` class declares ``params(is_img2img)`` producing a valid,
       strict-JSON-serializable :class:`remi.ui.ScriptLayout` on applicable tabs, with valid
       declared ``@ui.action`` specs.
    5. Optional locale JSON coverage for extracted UI strings.
    """

    from remi import ui
    from remi.ui.manifest import load_manifest
    from remi.ui.schema import dumps, to_schema

    root = Path(extension_dir).resolve()
    errors: list[str] = []
    warn_list: list[str] = []
    manifest_status = "missing"
    param_counts: dict[str, int] = {}
    tabs_checked: list[str] = []
    strings: set[str] = set()

    if not root.is_dir():
        return ExtensionContractReport(ok=False, extension_dir=str(root), errors=(f"extension directory does not exist: {root}",))

    if check_manifest:
        mres = load_manifest(root, known_widgets=known_widgets)
        manifest_status = mres.status
        errors.extend(f"manifest: {err}" for err in mres.errors)
        warn_list.extend(f"manifest: {w}" for w in mres.warnings)
        if require_manifest and mres.status == "missing":
            errors.append("manifest: remi-extension.json is required but missing")

    if check_no_gradio or check_no_server_imports:
        for py_file in sorted(root.rglob("*.py")):
            rel = py_file.relative_to(root).as_posix()
            if "/tests/" in f"/{rel}" or rel.startswith("tests/"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                imports = _extract_imports(source)
            except SyntaxError as exc:
                errors.append(f"syntax error in {rel}:{exc.lineno}: {exc.msg}")
                continue
            for lineno, mod in imports:
                top = mod.split(".", 1)[0]
                if check_no_gradio and top in _BANNED_IMPORT_ROOTS:
                    errors.append(f"banned Gradio import {mod!r} at {rel}:{lineno}")
                if check_no_server_imports and (mod == "remi.server" or mod.startswith("remi.server.")):
                    errors.append(f"forbidden remi.server import {mod!r} at {rel}:{lineno}")

    classes = list(script_classes or ())
    for cls in classes:
        cls_name = getattr(cls, "__name__", repr(cls))
        try:
            instance = cls()
        except Exception as exc:
            errors.append(f"script {cls_name}: failed to instantiate: {exc}")
            continue
        if not callable(getattr(instance, "params", None)):
            errors.append(f"script {cls_name}: missing params(is_img2img) method")
            continue
        try:
            ui.actions_of(instance)
        except Exception as exc:
            errors.append(f"script {cls_name}: invalid @ui.action declaration: {exc}")
        for tab in tabs:
            is_i2i = tab == "img2img"
            show = getattr(instance, "show", None)
            if callable(show):
                try:
                    if not show(is_i2i):
                        continue
                except Exception as exc:
                    errors.append(f"script {cls_name} ({tab}): show() failed: {exc}")
                    continue
            instance.tabname = tab
            instance.is_img2img = is_i2i
            instance.is_txt2img = not is_i2i
            try:
                tree = instance.params(is_i2i)
                layout = ui.ScriptLayout(tree, owner=cls_name.lower(), order=getattr(instance, "param_order", None))
                exported = to_schema(tree)
                dumps(exported)
                strings |= collect_ui_strings(exported)
                param_counts[f"{cls_name}:{tab}"] = len(layout.inputs)
                if tab not in tabs_checked:
                    tabs_checked.append(tab)
            except Exception as exc:
                errors.append(f"script {cls_name} ({tab}): invalid params() schema: {exc}")

    for locale in check_locales:
        loc_path = root / "localizations" / f"{locale}.json"
        if not loc_path.is_file():
            errors.append(f"locale {locale}: missing {loc_path.relative_to(root).as_posix()}")
            continue
        try:
            catalog = json.loads(loc_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"locale {locale}: invalid JSON: {exc}")
            continue
        if not isinstance(catalog, dict):
            errors.append(f"locale {locale}: root must be a JSON object")
            continue
        missing = sorted(text for text in strings if not isinstance(catalog.get(text), str) or not catalog[text].strip())
        if missing:
            errors.append(f"locale {locale}: missing translations for {missing[:10]}" + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""))

    return ExtensionContractReport(
        ok=not errors,
        extension_dir=str(root),
        script_count=len(classes),
        manifest_status=manifest_status,
        tabs_checked=tuple(tabs_checked),
        param_counts=param_counts,
        ui_strings=tuple(sorted(strings)),
        errors=tuple(errors),
        warnings=tuple(warn_list),
    )


def assert_extension_contract(extension_dir: str | Path, **kwargs: Any) -> ExtensionContractReport:
    """Assert that :func:`verify_extension_contract` succeeds and return the report."""

    report = verify_extension_contract(extension_dir, **kwargs)
    assert report.ok, "Extension contract violations:\n- " + "\n- ".join(report.errors)
    return report


# endregion


__all__ = [
    "PROCESSING_SCRIPTS",
    "ExtensionContractReport",
    "assert_extension_contract",
    "collect_ui_strings",
    "processing_script_classes",
    "verify_extension_contract",
    "webui_state",
]
