"""Settings: widget kinds for ``OptionInfo(component=...)`` and the settings
schema exported from ``shared.opts.data_labels`` (schema-format.md §6).

``OptionInfo.component`` takes a ``remi.ui`` class (``ui.Slider``,
``ui.Choice``, ``ui.MultiChoice``, ``ui.Color``, ``ui.Text``, ...) with
A1111-style ``component_args`` (``minimum``/``maximum``/``step``,
``choices``, ``precision``, ``lines``, ``placeholder``, ``visible``,
``interactive``) plus ``style`` (``"radio"``, ``"checkboxes"``) and
``allow_custom``; any other component (or none) is picked from the default's
type.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from html.parser import HTMLParser
from typing import Any

from remi.setting_values import setting
from remi.ui import nodes
from remi.ui.context import Context, ensure_context
from remi.ui.errors import ParamError, ValidationError
from remi.ui.schema import SCHEMA_VERSION, jsonable
from remi.ui.sentence import Sentence

REMI_WIDGETS = (nodes.Toggle, nodes.Slider, nodes.Number, nodes.Text, nodes.Choice, nodes.MultiChoice, nodes.Color, nodes.State, nodes.Json, nodes.Markdown, nodes.Html, nodes.Separator, nodes.Row)


def is_remi_widget(component: Any) -> bool:
    return isinstance(component, type) and issubclass(component, REMI_WIDGETS)


def widget_kind(component: Any, args: Mapping[str, Any], default: Any) -> tuple[type, dict]:
    """``(remi.ui class, extra arguments)`` for an option's component."""

    if is_remi_widget(component):
        return component, {}
    if isinstance(default, bool):
        return nodes.Toggle, {}
    if isinstance(default, str):
        return nodes.Text, {}
    if isinstance(default, (int, float)):
        return nodes.Number, {}
    return nodes.Json, {}


def component_args(info) -> dict:
    args = info.component_args() if callable(info.component_args) else info.component_args
    return dict(args or {})


def frozen_reason(opts: Any, key: str, cmd_opts: Any) -> str | None:
    """Why the WebUI may not change ``key`` (None when it may): ``PATCH
    /options`` refuses it, the settings schema marks it ``locked`` with this
    reason, and other writers of a setting (extensions, UI presets) ask too
    instead of tripping the setter's assertion."""

    info = opts.data_labels.get(key)
    if info is None:
        return "unknown setting"
    if getattr(cmd_opts, "freeze_settings", False):
        return "changing settings is disabled (--freeze-settings)"
    if info.do_not_save:
        return "not a stored setting"
    # ``restrict_api`` guards the A1111 API (/sdapi/v1/options); /api/v2 is the
    # WebUI's own API, behind the same login as the Gradio settings page was.
    try:
        args = info.component_args() if callable(info.component_args) else info.component_args
    except Exception:  # an extension's arguments that fail to build: coerce_values refuses the value
        args = None
    if isinstance(args, dict) and args.get("visible", True) is False:
        return "this setting is hidden"
    sections = getattr(cmd_opts, "freeze_settings_in_sections", None)
    if sections and info.section and info.section[0] in [item.strip() for item in sections.split(",")]:
        return "the settings in this section are frozen"
    keys = getattr(cmd_opts, "freeze_specific_settings", None)
    if keys and key in [item.strip() for item in keys.split(",")]:
        return "this setting is frozen"
    if getattr(cmd_opts, "hide_ui_dir_config", False) and key in opts.restricted_opts:
        return "this setting is restricted (--hide-ui-dir-config)"
    return None


def setting_node(key: str, info, value: Any) -> nodes.Node:
    """The ``remi.ui`` node of one option, its default being ``value``."""

    args = component_args(info)
    kind, extra = widget_kind(info.component, args, info.default)
    extra.update({name: args[name] for name in ("style", "allow_custom") if name in args})
    if args.get("allow_custom_value"):
        extra["allow_custom"] = True

    if kind is nodes.Markdown:  # OptionMarkdown: Remi's help blocks
        return nodes.Markdown(str(info.default))
    if kind is nodes.Html:  # OptionHTML: an A1111 extension's help markup
        return nodes.Html(str(info.default))
    if kind in (nodes.Separator, nodes.Row):
        return kind()

    common: dict[str, Any] = {"label": info.label or ""}
    if args.get("visible") is False:
        common["hidden"] = True
    if args.get("interactive") is False:
        common["enabled_if"] = False
    if getattr(info, "infotext", None):
        common["infotext"] = info.infotext

    if kind is nodes.Slider:
        minimum, maximum, step = args.get("minimum", 0), args.get("maximum", 100), args.get("step", 1)
        # By the declared default: a stored value may have lost its int type
        integer = all(nodes._is_int(item) for item in (minimum, maximum, step, info.default))
        return nodes.Slider(key, minimum, maximum, value, step, integer=integer, **common)
    if kind is nodes.Number:
        precision = args.get("precision")
        if precision is None and isinstance(info.default, int) and not isinstance(info.default, bool):
            precision = 0
        return nodes.Number(key, value, args.get("minimum"), args.get("maximum"), args.get("step"), precision, **common)
    if kind is nodes.Text:
        return nodes.Text(key, value, lines=max(int(args.get("lines", 1) or 1), 1), max_lines=args.get("max_lines"), placeholder=args.get("placeholder"), **common)
    if kind in (nodes.Choice, nodes.MultiChoice):
        style = extra.get("style", "dropdown")
        hints = {str(label): str(text) for label, text in (getattr(info, "choice_hints", None) or {}).items()}
        return kind(key, args.get("choices") or [], value, style=style, allow_custom=bool(extra.get("allow_custom")), hints=hints or None, **common)
    if kind is nodes.Toggle:
        return nodes.Toggle(key, value, **common)
    if kind is nodes.Color:
        return nodes.Color(key, value, **common)
    if kind is nodes.State:
        return nodes.State(key, value, **common)
    return nodes.Json(key, value, **common)


_MARKUP_RE = re.compile(r"<[A-Za-z/!][^>]*>|&[A-Za-z#0-9]+;")


class _NoteText(HTMLParser):
    """The text of an A1111 ``.info()`` note written with inline markup
    (``<b>``, ``<code>``, ``<a href>``, entities), and its links."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._label: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self.parts.append("\n")
        elif tag == "a":
            self._href, self._label = dict(attrs).get("href"), []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            label = " ".join("".join(self._label).split())
            if label and self._href.startswith(("https://", "http://")):
                self.links.append((label, self._href))
            self._href = None

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
        if self._href is not None:
            self._label.append(data)


def plain_note(note: str) -> tuple[str, list[tuple[str, str]]]:
    """``(text, links)`` of a setting note: notes are plain text (the SPA
    translates them by their English text); markup is dropped and the
    targets of ``<a href>`` become links of the setting."""

    if not _MARKUP_RE.search(note):
        return note, []
    parser = _NoteText()
    parser.feed(note)
    parser.close()
    text = "\n".join(" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
    return text.strip(), parser.links


def _setting_extras(info) -> dict:
    data: dict[str, Any] = {}
    # ``notes`` are plain text (translated); links found in a note join ``links``
    notes: list[str] = []
    html_blocks = list(getattr(info, "html_blocks", ()))
    links = list(getattr(info, "links", ()))
    for note in getattr(info, "notes", ()):
        text, found = plain_note(str(note))
        if text:
            notes.append(text)
        links += [link for link in found if link not in links]
    if notes:
        data["notes"] = notes
    if links:
        data["links"] = [{"label": label, "url": url} for label, url in links]
    default_order = [str(name) for name in getattr(info, "default_order", None) or ()]
    if default_order:
        data["default_order"] = default_order
    markup = "".join(html_blocks)
    if markup.strip():
        data["html"] = markup.strip()
    if getattr(info, "requires_restart", False):
        data["needs_restart"] = True
    if getattr(info, "requires_reload_ui", False):
        data["needs_reload_ui"] = True
    if info.refresh is not None:
        data["refresh"] = True
    if info.onchange is not None:
        data["onchange"] = True
    if info.restrict_api:
        data["restricted"] = True
    return data


def settings_schema(opts, ctx: Context | None = None, *, categories=None, quicksettings: list[str] | None = None, locked: Callable[[str], str | None] | None = None, held: Mapping[str, Any] | None = None) -> dict:
    """The settings page: categories -> sections -> items, plus values.

    ``locked(key)``: why the server refuses changes to ``key`` (None: it
    accepts them); a locked setting is ``enabled_if: false`` with ``locked``
    set to that reason. ``held``: values picked but not applied yet (a model
    selection made while a job runs), shown instead of the stored ones."""

    held = held or {}

    from modules import options as options_module

    ctx = ensure_context(ctx)
    categories = categories if categories is not None else options_module.categories
    result_categories: list[dict] = []
    by_category: dict[str, dict] = {}
    by_section: dict[tuple, dict] = {}
    open_rows: dict[tuple, dict | None] = {}

    for key, info in opts.data_labels.items():
        section = info.section
        if not section or section[0] is None:
            continue
        category = categories.mapping.get(info.category_id)
        category_id, category_label = (category.id, category.label) if category is not None else ("extensions", "Extensions")
        if category_id not in by_category:
            by_category[category_id] = {"id": category_id, "label": category_label, "sections": []}
            result_categories.append(by_category[category_id])
        if section not in by_section:
            by_section[section] = {"id": section[0], "label": section[1], "items": []}
            by_category[category_id]["sections"].append(by_section[section])
            open_rows[section] = None
        items = by_section[section]["items"]

        # the stored value, else the default (also for a value of another kind: the field can set it right)
        value = held[key] if key in held else setting(key, opts=opts)
        try:
            node = setting_node(key, info, value)
        except Exception as error:  # an extension option this exporter cannot read
            node = nodes.Markdown(f"`{key}`: {error}")

        if isinstance(node, nodes.Row):  # OptionRow markers open and close a row
            if open_rows[section] is None:
                open_rows[section] = {"type": "row", "children": []}
                items.append(open_rows[section])
            else:
                open_rows[section] = None
            continue

        data = node.schema(ctx)
        if isinstance(node, nodes.Input):
            data["default"] = jsonable(value)
            data.update(_setting_extras(info))
            reason = locked(key) if locked is not None else None
            if reason:
                data["enabled_if"] = False
                data["locked"] = reason
        target = open_rows[section]["children"] if open_rows[section] is not None else items
        target.append(data)

    stored = {key: info for key, info in opts.data_labels.items() if not getattr(info, "do_not_save", False)}
    return {
        "version": SCHEMA_VERSION,
        "categories": result_categories,
        "quicksettings": list(quicksettings if quicksettings is not None else setting("quicksettings_list", [], opts=opts)),
        "values": {key: jsonable(held[key] if key in held else setting(key, opts=opts)) for key in stored},
        "defaults": {key: jsonable(info.default) for key, info in stored.items()},
    }


def check_api_values(opts, values: Mapping[str, Any]) -> dict[str, Any]:
    """``POST /sdapi/v1/options``: ``values`` with each one that changes a
    setting checked and coerced as the settings page does (its widget's kind,
    range and choices), or :class:`ValidationError` listing every refused one
    (nothing is set then). A value equal to the stored one passes as it is, as
    upstream's ``set_config`` skips it (a client posting back what GET
    answered). A setting without a widget that checks it (a ``Json`` node)
    takes only a list of its default's kinds where the default is a list (of
    names when it is empty: ``disabled_extensions: 5`` stopped the next
    start). A value nested deeper than ``MAX_NESTING`` levels is refused,
    whatever the setting.

    Left as upstream had them: a setting set elsewhere (``interactive=False``:
    ``sd_vae``, whose offered choices are not what a client sends), an
    extension option the settings page cannot read either, and a
    ``restrict_api`` one (``Options.set`` ignores it for the API). The route
    takes out the model selection and refuses unknown settings itself;
    ``--freeze-settings`` stays ``Options.set``'s call. Here rather than in
    ``remi.server.options``: ``modules/api`` does not import the server."""

    from remi.setting_values import MAX_NESTING, fit, nested_deeper

    ctx = Context()
    result = dict(values)
    problems: list[ParamError] = []
    for key, value in values.items():
        info = opts.data_labels.get(key)
        if info is None or getattr(info, "restrict_api", False):
            continue
        if nested_deeper(value):  # first: even one left unchecked is saved (see MAX_NESTING)
            problems.append(ParamError(Sentence("nested deeper than {{limit}} levels", limit=MAX_NESTING), owner="settings", param=key))
            continue
        if value == opts.data.get(key):
            continue
        try:
            if component_args(info).get("interactive") is False:
                continue
            node = setting_node(key, info, setting(key, opts=opts))
        except Exception:  # an extension option the settings page shows as an error: unchecked
            continue
        if not isinstance(node, nodes.Input):  # a help block: nothing to check
            continue
        try:
            coerced = node.coerce(value, ctx)
        except ParamError as error:
            problems.append(ParamError(error.message, owner="settings", param=key))
            continue
        except Exception:  # an extension's choices that fail to build: unchecked, as the page cannot show it either
            continue
        if isinstance(node, nodes.Json) and info.default is not None and fit(info.default, coerced) is info.default and coerced != info.default:
            problems.append(ParamError(Sentence("expected a value like {{default}}", default=repr(info.default)), owner="settings", param=key))
            continue
        result[key] = coerced
    if problems:
        raise ValidationError(problems)
    return result


def refresh_setting(opts, key: str) -> dict:
    """Run a setting's ``refresh`` and return its patch (new choices)."""

    info = opts.data_labels[key]
    if info.refresh is not None:
        info.refresh()
    args = component_args(info)
    patch: dict[str, Any] = {}
    if "choices" in args:
        from remi.ui.actions import normalize_choices

        patch["choices"] = normalize_choices(args["choices"])
    for name in ("minimum", "maximum", "step"):
        if name in args:
            patch[name] = args[name]
    return {key: patch}
