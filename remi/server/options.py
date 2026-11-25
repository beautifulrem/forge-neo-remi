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

from remi.setting_values import frozen_reason
from remi.ui import Context, ParamError, Sentence, ValidationError

MODEL_KEYS = ("sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype")


def _error(key: str, message: str) -> ParamError:
    return ParamError(message, owner="settings", param=key)


def coerce_values(opts: Any, values: Any, cmd_opts: Any) -> dict[str, Any]:
    """The coerced values, or :class:`ValidationError` listing every
    rejected one (nothing is applied then)."""

    from remi import model_selection
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
            # the UI preset's copy the model manager writes with the pick, when a freeze keeps it (its assertion came after
            # the setting itself was set): refused, named
            if (found := model_selection.frozen_pick(key, result[key])) is not None:
                name, reason = found
                problems.append(_error(name, reason))
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
