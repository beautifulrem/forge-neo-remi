"""Per-tab bindings: the runner, the core layout, the tab schema and the
mapping of named request values to processing arguments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remi.core import forms
from remi.ui import SCHEMA_VERSION, Context, ParamError, ScriptLayout, ValidationError, jsonable
from remi.ui.types import TabSchema


def runner_for(tab: str):
    """The script runner of ``tab`` (None for extras, whose operations run
    through ``scripts.scripts_postproc``)."""

    from modules import scripts

    if tab == "txt2img":
        return scripts.scripts_txt2img
    if tab == "img2img":
        return scripts.scripts_img2img
    if tab == "extras":
        return None
    raise KeyError(f"unknown tab {tab!r} (one of {forms.TABS})")


def core_layout(tab: str) -> ScriptLayout:
    return forms.core_layout(tab, runner_for(tab))


def conflict_params(tab: str, layouts: Mapping[str, ScriptLayout] | None = None) -> list[str]:
    """``owner.param`` references whose changes should run the core
    ``__conflicts__`` action: every param bound to a conflict role or an
    exclusive group. ``layouts``: the tab's owners, when already built."""

    from modules.anima_feature_conflicts import CONFLICT_ROLES
    from remi.core.conflicts import bindings

    roles, groups = bindings(tab, layouts)
    refs = [f"{owner}.{name}" for role, (owner, name) in roles.items() if role in CONFLICT_ROLES]
    refs += [f"{owner}.{name}" for members in groups.values() for owner, name in members.values()]
    return sorted(set(refs))


def tab_schema(tab: str, ctx: Context | None = None) -> TabSchema:
    """``GET /api/v2/schema/{tab}`` (schema-format.md §4)."""

    from modules import scripts
    from remi.core import extras
    from remi.core.capabilities import active_capability_summary, derive_advice

    ctx = ctx or Context()
    runner = runner_for(tab)
    layout = forms.core_layout(tab, runner)
    capabilities = active_capability_summary()
    if runner is None:
        script_entries = extras.scripts_schema(scripts.scripts_postproc, ctx)
    else:
        script_entries = runner.scripts_schema(ctx, capability_summary=capabilities)
    core_defaults = jsonable(layout.defaults(ctx))
    schema: TabSchema = {
        "version": SCHEMA_VERSION,
        "tab": tab,
        "core": {"owner": "core", "layout": layout.schema(ctx), "defaults": core_defaults, "required": forms.required_inputs(tab)},
        "scripts": script_entries,
        "capabilities": capabilities,
    }
    if runner is not None:
        layouts = {"core": layout, **{script.script_id: script.layout for script in runner.scripts if script.layout is not None}}
        schema["conflict_params"] = conflict_params(tab, layouts)
        initial_values = {"core": core_defaults, **{entry["id"]: entry.get("defaults") or {} for entry in script_entries}}
        schema["advice"] = derive_advice(tab, initial_values, preset=capabilities.get("preset"))
    return schema


def _selected_script(layout: ScriptLayout, core: Any, ctx: Context) -> Any:
    """The core ``script`` value on its own, so the selected script's values
    are still checked when other core values are rejected."""

    node = layout.by_name.get("script")
    if node is None:
        return None
    try:
        return node.coerce(core["script"], ctx) if isinstance(core, Mapping) and "script" in core else node.resolve_default(ctx)
    except ParamError:
        return None


def carried_values(tab: str, core: Any, scripts_values: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """``core`` and ``scripts_values`` as a job keeps them: without the values
    of ``submit=False`` params and tool inputs (``arg=False``), which are for
    actions only. The SPA never sends them; an API client's (or an imported
    task's) are dropped, checked or not: processing uses the defaults either
    way. A script's values given as a list are kept as they are."""

    from modules import scripts
    from remi.core import extras

    runner = runner_for(tab)
    owners: list[tuple[str, ScriptLayout]] = [("core", forms.core_layout(tab, runner))]
    if runner is not None:
        owners += [(script.script_id, runner.layout_for(script)) for script in runner.scripts]
    else:
        owners += [(layout.owner, layout) for _script, layout in extras.assign_args(scripts.scripts_postproc, assign=False)]
    held = {owner: {node.name for node in (*layout.inputs, *layout.tools) if not node.submit} for owner, layout in owners}

    def carried(owner: str, values: Any) -> Any:
        if not isinstance(values, Mapping) or not held.get(owner):
            return values
        return {name: value for name, value in values.items() if name not in held[owner]}

    return carried("core", dict(core or {})), {owner: carried(owner, values) for owner, values in (scripts_values or {}).items()}


def processing_args(tab: str, core: Any, scripts_values: Any, ctx: Context | None = None, *, assign_slots: bool = True) -> tuple[dict[str, Any], list]:
    """Validate a request: ``(core processing args by name, script args)``.

    Core values are coerced and converted (choice indices, uploads); script
    values become the full positional ``p.script_args`` (extras: the
    operations' positional args). Raises :class:`ValidationError` listing
    every rejected value (a body that is not an object of named values
    included). ``assign_slots=False``: a check only, which leaves the Extras
    operations' argument slots alone (:func:`remi.core.extras.assign_args`)."""

    from modules import scripts
    from remi.core import extras

    ctx = ctx or Context()
    runner = runner_for(tab)
    layout = forms.core_layout(tab, runner)
    problems: list = []
    core_args: dict[str, Any] = {}
    try:
        values = layout.coerce(core, ctx)
    except ValidationError as error:
        problems.append(error)
        values = {"script": _selected_script(layout, core, ctx)}
    else:
        problems += forms.check_core(tab, values)
        # a layer the mode does not read is not resolved: its upload may be gone
        values = {**values, **{name: layout.by_name[name].resolve_default(ctx) for name in forms.unread_uploads(tab, values)}}
        try:
            core_args = dict(zip(layout.names, layout.args_for(values, ctx), strict=True))
        except ValidationError as error:
            problems.append(error)
    try:
        if runner is None:
            script_args = extras.postprocessing_args(scripts.scripts_postproc, scripts_values, ctx, assign=assign_slots)
        else:
            script_args = runner.script_args_from_values(values.get("script"), scripts_values, ctx)
    except ValidationError as error:
        problems.append(error)
        script_args = []
    if problems:
        raise ValidationError.merge(problems)
    return core_args, script_args
