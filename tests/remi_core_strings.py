"""The English strings of the core UI the server catalog must translate
(``localizations/<locale>.json``): the generation forms (core params and the
built-in processing scripts), every setting (category, section, label, info,
notes, link labels, choice labels, tooltips, HTML help blocks keyed by their
exact markup), and the server's translatable messages
(action toasts and texts, page task messages, extension admin errors).

Not a test module: ``tests/test_remi_core_zh_cn.py`` checks the zh_CN
catalog with it, and the CLI lists what a catalog lacks::

    $PYTHON tests/remi_core_strings.py [--locale zh_CN] [--all] [--out list.json]

``$PYTHON``: an interpreter with requirements.txt (no Gradio), as ``tools/check.sh`` picks.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABS = ("txt2img", "img2img", "extras")

# Python sources whose toast / message literals are user-facing server texts
MESSAGE_SOURCES = ("remi", "modules/processing_scripts")
# Calls whose first string argument is a catalog key (ui.toast / ui.message,
# a page task's progress text, an extension admin error's template)
MESSAGE_CALLS = {"toast": 0, "message": 0, "progress": 1, "AdminError": 0, "GitError": 0}


def _text_strings(node, found: set[str]) -> None:
    """Strings of an exported layout / settings item the SPA translates
    (the choices of :data:`DATA_CHOICES` are data, not text)."""

    if isinstance(node, list):
        for item in node:
            _text_strings(item, found)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "html":
        # help markup: the SPA translates it by its exact text (markup included)
        if isinstance(node.get("text"), str) and node["text"].strip():
            found.add(node["text"])
    else:
        for key in ("label", "info", "hint", "title", "placeholder"):
            if isinstance(node.get(key), str) and node[key].strip():
                found.add(node[key])
        for hint in (node.get("choice_hints") or {}).values():
            if isinstance(hint, str) and hint.strip():
                found.add(hint)
        if node.get("type") == "markdown" and isinstance(node.get("text"), str):
            found.add(node["text"])
    if isinstance(node.get("html"), str) and node["html"].strip():  # a setting's A1111 help markup
        found.add(node["html"])
    for note in node.get("notes") or ():
        if isinstance(note, str) and note.strip():
            found.add(note)
    for link in node.get("links") or ():
        if isinstance(link, dict) and isinstance(link.get("label"), str):
            found.add(link["label"])
    name = node.get("name") or node.get("key")
    if name not in DATA_CHOICES and not str(name).startswith(DATA_CHOICE_PREFIXES):
        for choice in node.get("choices") or ():
            if isinstance(choice, dict) and isinstance(choice.get("label"), str) and choice["label"].strip():
                found.add(choice["label"])
    for key in ("children", "enable", "layout", "sections", "items", "categories", "scripts", "core"):
        if key in node:
            _text_strings(node[key], found)


# Params / settings whose choice labels are identifiers, listed one by one
# (a new lowercase label anywhere else is UI text and must be translated)
DATA_CHOICES = frozenset(
    {
        "samples_format",  # file formats
        "grid_format",
        "live_previews_image_format",
        "video_container",  # ffmpeg container, preset and profile names
        "video_preset",
        "video_profile",
        "svdq_attention",  # attention backend ids
        "infotext_skip_pasting",  # infotext keys
        "ui_reorder_list",  # the generation form's section ids
    }
)
DATA_CHOICE_PREFIXES = ("prioritized_callbacks_",)  # callback ids (file/callback/class)


def _dynamic_names() -> set[str]:
    """Names that are data, not UI text: samplers, schedulers, upscalers,
    locale ids, UI presets, model and file names offered as choices."""

    from modules import localization, sd_samplers, sd_schedulers, shared
    from modules_forge.presets import PresetArch

    names: set[str] = {"None"}
    for sampler in sd_samplers.all_samplers:
        names.add(sampler.name)
        names.update(sampler.aliases or ())
    for scheduler in sd_schedulers.schedulers:
        names.update((scheduler.name, scheduler.label))
    names.update(upscaler.name for upscaler in shared.sd_upscalers or ())
    names.update(localization.localizations)
    names.update(PresetArch.choices())
    names.update(shared.latent_upscale_modes)
    names.update(shared.opts.data_labels)  # setting keys (quick settings, …)
    names.update(face.name() for face in shared.face_restorers or ())
    return names


def form_strings() -> set[str]:
    """Labels, infos, titles, placeholders, choices and Markdown of the three
    core tab schemas (core form and the built-in processing scripts)."""

    from remi.core.tabs import tab_schema

    found: set[str] = set()
    for tab in TABS:
        schema = tab_schema(tab)
        _text_strings(schema["core"]["layout"], found)
        found.update(entry["message"] for entry in schema["core"].get("required", ()))  # shown inline and in the toast
        for script in schema["scripts"]:
            found.add(script["title"])
            _text_strings(script["layout"], found)
    return found


def settings_strings(opts) -> set[str]:
    """Every setting the server shows, the Callbacks section included
    (``remi.server.runtime.prepare`` adds it at startup)."""

    from modules import shared_items
    from modules.options import options_section
    from remi.server.settings import register_options
    from remi.ui.settings import settings_schema

    register_options(opts)
    callbacks = options_section(("callbacks", "Callbacks", "system"), shared_items.callbacks_order_settings())
    saved = opts.data_labels
    opts.data_labels = {**saved, **callbacks}
    try:
        found: set[str] = set()
        _text_strings(settings_schema(opts)["categories"], found)
    finally:
        opts.data_labels = saved
    return found - set(callbacks)  # setting keys offered as choices (quick settings) are data


def message_strings() -> set[str]:
    """String literals passed as catalog keys in the server's sources."""

    found: set[str] = set()
    for folder in MESSAGE_SOURCES:
        for path in sorted((REPO / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                index = MESSAGE_CALLS.get(name)
                if index is None:
                    continue
                args = list(node.args)
                if name == "progress":
                    args += [kw.value for kw in node.keywords if kw.arg == "text"]
                if len(args) > index and isinstance(args[index], ast.Constant) and isinstance(args[index].value, str) and args[index].value.strip():
                    found.add(args[index].value)
    return found


def job_strings() -> set[str]:
    """The fixed sentences a job or a model load ends with (shown in the
    history and the failure toasts)."""

    from remi.jobs.texts import JOB_TEXTS

    return set(JOB_TEXTS)


# Model families' workflow advice (the bar above Generate; the notes a job ends with) and why a
# script does not apply: folders or files
ADVICE_SOURCES = ("backend/nn", "modules/anima_presets.py", "remi/prompt_tools/pup.py")


def _advice_nodes():
    """``(path, node)`` for every AST node of the advice sources."""

    for source in ADVICE_SOURCES:
        base = REPO / source
        assert base.exists(), source
        for path in [base] if base.is_file() else sorted(base.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                yield path, node


def _constant(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip() else None


def _choices(node) -> list[str] | None:
    """The texts of a constant or of a conditional choosing between constants; None for anything else."""

    if isinstance(node, ast.IfExp):
        body, orelse = _choices(node.body), _choices(node.orelse)
        return None if body is None or orelse is None else body + orelse
    text = _constant(node)
    return None if text is None else [text]


def _advice_items():
    for path, node in _advice_nodes():
        if isinstance(node, ast.Call) and _call_name(node) in ("AdviceItem", "advice_cls"):
            yield path, node


def advice_strings() -> set[str]:
    """The model advice and capability sentences: ``Sentence`` templates, the
    constant ``message`` and ``action_label`` of an ``AdviceItem`` (or each
    constant a conditional label chooses between), and the constant
    sentences a ``*_workflow_advice`` function appends."""

    found: set[str] = set()
    for _path, node in _advice_nodes():
        if isinstance(node, ast.FunctionDef) and node.name.endswith("workflow_advice"):
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "append" and call.args and (text := _constant(call.args[0])):
                    found.add(text)
        if isinstance(node, ast.Call) and _call_name(node) == "Sentence" and node.args and (text := _constant(node.args[0])):
            found.add(text)
    for _path, node in _advice_items():
        for kw in node.keywords:
            if kw.arg == "message" and (text := _constant(kw.value)):
                found.add(text)
            elif kw.arg == "action_label":
                found.update(_choices(kw.value) or ())
    return found


def unlisted_action_labels() -> list[str]:
    """``path:line`` of each ``AdviceItem`` whose ``action_label`` is neither a
    constant nor a choice between constants (an f-string, say): the SPA looks
    the label up in the catalog, which cannot hold it."""

    return [f"{path.relative_to(REPO).as_posix()}:{kw.value.lineno}" for path, node in _advice_items() for kw in node.keywords if kw.arg == "action_label" and _choices(kw.value) is None]


# Calls that make a rejected value's error, with the index of their message
# argument: ParamError and the helpers that wrap one
PARAM_ERROR_CALLS = {"ParamError": 0, "_error": 1, "bad_input": 1}
# Remi's own sources that reject values, and modules/scripts.py, whose value checks
# (``selector_index``, ``script_args_from_values``) Remi wrote; the rest of the shared runtime
# is left as it is
PARAM_ERROR_SOURCES = ("remi", "modules/scripts.py")


def _call_name(node) -> str | None:
    return node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None


def _message_keys(arg, constants: dict[str, str]) -> list[str] | None:
    """The catalog keys a message argument can be, or None when it is formatted text."""

    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return [arg.value]
    if isinstance(arg, ast.Call) and _call_name(arg) == "Sentence" and arg.args and isinstance(arg.args[0], ast.Constant):
        return [arg.args[0].value]
    if isinstance(arg, ast.IfExp):
        body, orelse = _message_keys(arg.body, constants), _message_keys(arg.orelse, constants)
        return body + orelse if body is not None and orelse is not None else None
    if isinstance(arg, ast.Name) and arg.id in constants:
        return [constants[arg.id]]
    if isinstance(arg, ast.Attribute) and arg.attr == "message":
        return []
    return None


def param_error_messages(folders=PARAM_ERROR_SOURCES) -> tuple[set[str], list[tuple[str, str]]]:
    """``(texts, unchecked)``: the catalog keys a rejected value's message can
    be (a constant, a module constant, a ``Sentence`` template) and the calls
    whose message is none of them, as ``(path, source)``. A message passed on
    from a caught error (``error.message``) was checked where it was made."""

    texts: set[str] = set()
    unchecked: list[tuple[str, str]] = []
    for folder in folders:
        root = REPO / folder
        for path in sorted([root] if root.is_file() else root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            constants = {
                target.id: node.value.value
                for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                for target in node.targets
                if isinstance(target, ast.Name)
            }

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or (index := PARAM_ERROR_CALLS.get(_call_name(node))) is None or len(node.args) <= index:
                    continue
                found = _message_keys(node.args[index], constants)
                if found is None:
                    unchecked.append((path.relative_to(REPO).as_posix(), ast.get_source_segment(source, node.args[index]) or ""))
                else:
                    texts.update(found)
    return texts, unchecked


def param_error_strings() -> set[str]:
    """The messages of the values the server rejects (shown under their field)."""

    return param_error_messages()[0]


def core_strings(opts) -> set[str]:
    """Everything the core catalog must translate (data names left out)."""

    strings = form_strings() | settings_strings(opts) | message_strings() | job_strings() | advice_strings() | param_error_strings()
    dynamic = _dynamic_names()
    return {text for text in strings if text not in dynamic and not _is_data(text)}


def _is_data(text: str) -> bool:
    """Numbers, units and symbols need no translation."""

    return bool(re.fullmatch(r"[\d\s.,:%x×+\-/()#*]+|[A-Z0-9_]{1,6}", text.strip()))


def spa_catalog(locale: str = "zh-CN") -> dict[str, str]:
    """The SPA's own catalog (it wins over the server catalog at runtime)."""

    return json.loads((REPO / "webui" / "src" / "locales" / f"{locale}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--locale", default="zh_CN")
    parser.add_argument("--all", action="store_true", help="every string, not only the missing ones")
    parser.add_argument("--out", type=Path, help="write the JSON list here (the WebUI logs to stdout)")
    options = parser.parse_args()
    sys.argv = sys.argv[:1]
    sys.path[:0] = [str(REPO / "modules_forge" / "packages"), str(REPO)]
    from remi.testing import webui_state

    with webui_state() as state:
        strings = core_strings(state.opts)
    catalog = json.loads((REPO / "localizations" / f"{options.locale}.json").read_text(encoding="utf-8"))
    spa = spa_catalog(options.locale.replace("_", "-"))
    wanted = sorted(strings if options.all else (text for text in strings if not catalog.get(text) and text not in spa))
    text = json.dumps(wanted, ensure_ascii=False, indent=1)
    if options.out:
        options.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)  # noqa: T201 - command-line output
