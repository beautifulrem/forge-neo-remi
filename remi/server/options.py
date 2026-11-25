"""``PATCH /api/v2/options``: validate every value against its setting's
widget, then apply them (running ``onchange`` side effects), save
``config.json`` and report which keys changed.

The model selection (checkpoint, VAE/text encoder modules, low-bits dtype)
goes through the model manager, as the model bar did: it is remembered for
the current UI preset and the next generation loads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remi.ui import Context, ParamError, Sentence, ValidationError

MODEL_KEYS = ("sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype")


def _error(key: str, message: str) -> ParamError:
    return ParamError(message, owner="settings", param=key)


def frozen_reason(opts: Any, key: str, cmd_opts: Any) -> str | None:
    """Why ``PATCH /options`` may not change ``key`` (None when it may); the
    settings schema marks such settings ``locked`` with this reason."""

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


def coerce_values(opts: Any, values: Any, cmd_opts: Any) -> dict[str, Any]:
    """The coerced values, or :class:`ValidationError` listing every
    rejected one (nothing is applied then)."""

    from remi.setting_values import MAX_NESTING, encodable, nested_deeper
    from remi.ui import nodes
    from remi.ui.settings import setting_node

    if not isinstance(values, Mapping):
        raise ValidationError([ParamError("expected an object of setting values", owner="settings")])
    ctx = Context()
    result: dict[str, Any] = {}
    problems: list[ParamError] = []
    for key, value in values.items():
        reason = frozen_reason(opts, key, cmd_opts)
        if reason is not None:
            problems.append(_error(key, reason))
            continue
        if nested_deeper(value):  # config.json's write would stop half way (see MAX_NESTING)
            problems.append(ParamError(Sentence("nested deeper than {{limit}} levels", limit=MAX_NESTING), owner="settings", param=key))
            continue
        if not encodable(value):  # config.json is written as UTF-8: the write would stop half way
            problems.append(_error(key, "the text has a broken character (an unpaired surrogate)"))
            continue
        if key in MODEL_KEYS:
            try:
                result[key] = _model_value(key, value)
            except ParamError as error:
                problems.append(_error(key, error.message))
            continue
        if key == "forge_preset":  # free text in the settings, but only an offered preset has its model settings
            from remi.core.presets import check_preset

            try:
                result[key] = check_preset(value)
            except ValidationError as error:
                problems.extend(_error(key, problem.message) for problem in error.errors)
            continue
        current = getattr(opts, key)
        try:
            node = setting_node(key, opts.data_labels[key], current)
        except Exception:  # an extension option the settings page shows as an error, not a field
            node = None
        if not isinstance(node, nodes.Input):
            problems.append(_error(key, "not a setting"))
            continue
        try:
            result[key] = node.coerce(value, ctx)
        except ParamError as error:
            if value == current:  # a stored value outside stale choices
                result[key] = value
            else:
                problems.append(_error(key, error.message))
    if problems:
        raise ValidationError(problems)
    return result


def _model_value(key: str, value: Any) -> Any:
    from remi import model_selection
    from remi.server.runtime import unet_dtypes

    return model_selection.check_value(key, value, dtypes=unet_dtypes)


def apply_values(opts: Any, values: Mapping[str, Any], config_filename: str | None) -> tuple[list[str], list[ParamError]]:
    """Apply coerced values: ``(changed keys, failures)``; saves the
    options when something changed."""

    from remi.server import runtime

    changed: list[str] = []
    failed: list[ParamError] = []
    for key, value in values.items():
        try:
            if key == "sd_model_checkpoint":
                done = runtime.select_checkpoint(value)
            elif key == "forge_additional_modules":
                done = runtime.select_modules(value)
            elif key == "forge_unet_storage_dtype":
                done = runtime.select_unet_dtype(value)
            else:
                before = opts.data.get(key)
                done = opts.set(key, value)
                if not done and value != before and opts.data.get(key) != value:
                    failed.append(_error(key, "the setting rejected the value"))
                    continue
        except Exception as error:  # an onchange or model manager failure
            failed.append(_error(key, f"{type(error).__name__}: {error}"))
            continue
        if done:
            changed.append(key)
    if changed and config_filename:
        opts.save(config_filename)
    return changed, failed
