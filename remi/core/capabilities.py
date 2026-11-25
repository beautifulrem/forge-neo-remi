"""Active model capability resolution, script applicability, and pre-submit
model advice bridging ``backend.nn.model_capabilities`` to ``remi.core``.

Layering: lives in ``remi.core`` (never imports ``remi.server``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from backend.nn.dit_registry import AdviceItem, derive_model_advice
from backend.nn.model_capabilities import (
    capability_summary_for_preset,
    check_script_applicability,
    ensure_ext_capabilities_loaded,
    normalize_preset_name,
)
from remi.setting_values import setting
from remi.ui.sentence import sentence_fields


def _resolve_opts(opts: Any = None) -> Any:
    if opts is not None:
        return opts
    try:
        from modules import shared

        return getattr(shared, "opts", None)
    except Exception:
        return None


def _resolve_sd_model(sd_model: Any = None) -> Any:
    if sd_model is not None:
        return sd_model
    try:
        from modules import sd_models

        return getattr(getattr(sd_models, "model_data", None), "sd_model", None)
    except Exception:
        return None


def active_preset(opts: Any = None) -> str:
    """The active UI preset name (default ``"sd"``)."""

    active_opts = _resolve_opts(opts)
    raw = setting("forge_preset", "sd", opts=active_opts) if active_opts is not None else None
    return normalize_preset_name(raw or "sd")


def active_capability_summary(
    opts: Any = None,
    *,
    preset: str | None = None,
    sd_model: Any = None,
) -> dict[str, Any]:
    """JSON capability summary for the currently selected preset/model."""

    active_opts = _resolve_opts(opts)
    target_preset = normalize_preset_name(preset) if preset else active_preset(active_opts)
    model = _resolve_sd_model(sd_model)
    return capability_summary_for_preset(target_preset, sd_model=model, opts=active_opts)


def active_capabilities(
    opts: Any = None,
    *,
    preset: str | None = None,
    sd_model: Any = None,
) -> frozenset[str]:
    """Capability token set for the currently selected preset/model."""

    summary = active_capability_summary(opts=opts, preset=preset, sd_model=sd_model)
    return frozenset(summary["capabilities"])


def script_applicability(
    script: Any,
    active_caps: Collection[str] | None = None,
    *,
    preset: str | None = None,
    opts: Any = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute ``requires_capabilities``, ``applicable``, and optional
    ``inapplicable_reason`` metadata for a ``scripts.Script`` instance."""

    raw_req = getattr(script, "requires_capabilities", ()) or ()
    required = [str(cap) for cap in raw_req if cap]
    if active_caps is None or summary is None:
        resolved_summary = summary if summary is not None else active_capability_summary(opts=opts, preset=preset)
        caps = active_caps if active_caps is not None else resolved_summary.get("capabilities", ())
        engine_label = resolved_summary.get("engine_label")
        resolved_preset = resolved_summary.get("preset")
    else:
        caps = active_caps
        engine_label = summary.get("engine_label")
        resolved_preset = preset or summary.get("preset")

    applicable, reason = check_script_applicability(
        required,
        caps,
        engine_label=str(engine_label) if engine_label else None,
        preset=str(resolved_preset) if resolved_preset else None,
    )
    meta: dict[str, Any] = {
        "requires_capabilities": required,
        "applicable": applicable,
    }
    if not applicable and reason:
        meta["inapplicable_reason"] = str(reason)
        meta.update(sentence_fields(reason, "inapplicable_"))  # the UI translates the template (the reason is English)
    return meta


def derive_advice(
    tab: str,
    values: Mapping[str, Mapping[str, Any]],
    *,
    preset: str | None = None,
    opts: Any = None,
) -> list[dict[str, Any]]:
    """Evaluate pre-submit structured model advice for ``tab`` and ``values``,
    returned as JSON-ready ``AdviceItem`` dicts."""

    ensure_ext_capabilities_loaded()
    active_opts = _resolve_opts(opts)
    target_preset = normalize_preset_name(preset) if preset else active_preset(active_opts)
    items: list[AdviceItem] = derive_model_advice(target_preset, tab, values, opts=active_opts)
    return [item.to_dict() for item in items]
