"""Full-stack ModelCapability bridging ``DiTCapability`` and ``PresetArch``.

Provides a single source of truth connecting backend DiT adapters
(``backend.nn.dit_registry``), UI presets (``modules_forge.presets.PresetArch``),
external DiT self-registration (``backend.nn.dit_capabilities_ext``), script
``requires_capabilities`` matching, and Model Bar metadata badges.

Kept lightweight and free of server/UI imports so both backend runtime and
``remi.core`` can share it without circular dependencies.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from backend.nn.dit_registry import (
    DiTCapability,
    all_dit_capabilities,
    capability_for_preset,
    get_dit_capability,
)

# Human-readable names for declarative capability tokens (used in inapplicable reasons & UI)
CAPABILITY_LABELS: dict[str, str] = {
    "anima_hooks": "Anima engine hooks",
    "multi_reference": "Multi-reference conditioning",
    "rgba_output": "Native RGBA output",
    "control_blocks": "DiT Control Blocks",
    "structured_layout": "Structured Layout",
    "distilled_cfg1": "Distilled CFG 1",
    "prefix_kv_cache": "Prefix KV Cache",
    "per_block_compile": "Per-block torch.compile",
    "spectrum": "Spectrum forecasting",
    "video": "Video generation",
}

_EXT_LOADED = False


def ensure_ext_capabilities_loaded(*, force: bool = False) -> bool:
    """Import ``backend.nn.dit_capabilities_ext`` so optional/external DiT
    families (e.g. Z-Image, ERNIE-Image, Bria Fibo 2.0) self-register into
    ``backend.nn.dit_registry``. Safe no-op when the module is absent."""

    global _EXT_LOADED
    if _EXT_LOADED and not force:
        return True
    try:
        mod = importlib.import_module("backend.nn.dit_capabilities_ext")
    except ModuleNotFoundError as exc:
        if exc.name in ("backend.nn.dit_capabilities_ext", "dit_capabilities_ext"):
            return False
        raise
    except ImportError:
        return False
    register_fn = getattr(mod, "register_extended_dit_capabilities", None)
    if callable(register_fn):
        register_fn()
    _EXT_LOADED = True
    return True


def _reset_ext_loaded_for_tests() -> None:
    global _EXT_LOADED
    _EXT_LOADED = False


@dataclass(frozen=True)
class _PresetBaseline:
    """Fallback declarative capability metadata for ``PresetArch`` presets that
    do not (yet) have a registered ``DiTCapability`` entry."""

    preset: str
    engine_label: str
    dit_alias: str | None = None
    capabilities: frozenset[str] = frozenset()
    max_references: int = 0
    reference_slot_format: str | None = None
    reference_slot_start: int = 1
    supports_rgba: bool = False
    supports_control_blocks: bool = False
    supports_cfg_guidance: bool = True
    supports_negative_prompt: bool = True
    attention_backends: tuple[str, ...] = ("sdpa", "flash", "sage")
    quant_formats: tuple[str, ...] = ("bf16", "fp8", "gguf")


# All 15 PresetArch presets in modules_forge/presets.py
_PRESET_BASELINES: dict[str, _PresetBaseline] = {
    "sd": _PresetBaseline(
        preset="sd",
        engine_label="SD 1.5",
        capabilities=frozenset(),
        attention_backends=("sdpa", "flash", "sage", "xformers"),
        quant_formats=("fp16", "bf16", "fp8", "gguf"),
    ),
    "xl": _PresetBaseline(
        preset="xl",
        engine_label="SDXL",
        capabilities=frozenset(),
        attention_backends=("sdpa", "flash", "sage", "xformers"),
        quant_formats=("fp16", "bf16", "fp8", "gguf"),
    ),
    "flux": _PresetBaseline(
        preset="flux",
        engine_label="FLUX.1",
        capabilities=frozenset({"multi_reference", "distilled_cfg1"}),
        max_references=1,
        quant_formats=("bf16", "fp8", "gguf", "svdq", "nvfp4"),
    ),
    "klein": _PresetBaseline(
        preset="klein",
        engine_label="FLUX.2 Klein",
        capabilities=frozenset({"multi_reference", "distilled_cfg1"}),
        max_references=4,
    ),
    "qwen": _PresetBaseline(
        preset="qwen",
        engine_label="Qwen-Image",
        capabilities=frozenset({"multi_reference", "distilled_cfg1"}),
        max_references=4,
        reference_slot_format="Picture {i}",
        reference_slot_start=0,
    ),
    "lumina": _PresetBaseline(
        preset="lumina",
        engine_label="Lumina-Image-2.0",
        capabilities=frozenset(),
    ),
    "zit": _PresetBaseline(
        preset="zit",
        engine_label="Z-Image",
        dit_alias="zimage",
        capabilities=frozenset({"multi_reference", "control_blocks", "distilled_cfg1", "per_block_compile", "spectrum"}),
        max_references=4,
        supports_control_blocks=True,
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
    ),
    "wan": _PresetBaseline(
        preset="wan",
        engine_label="Wan 2.2",
        capabilities=frozenset({"multi_reference", "video"}),
        max_references=1,
    ),
    "anima": _PresetBaseline(
        preset="anima",
        engine_label="Anima",
        dit_alias="anima",
        capabilities=frozenset({"anima_hooks", "multi_reference", "rgba_output", "control_blocks", "per_block_compile", "spectrum"}),
        max_references=4,
        supports_rgba=True,
        supports_control_blocks=True,
        attention_backends=("sdpa", "flash", "sage", "masked_sdpa"),
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
    ),
    "ernie": _PresetBaseline(
        preset="ernie",
        engine_label="ERNIE-Image",
        dit_alias="ernie",
        capabilities=frozenset({"distilled_cfg1", "per_block_compile", "spectrum"}),
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
    ),
    "pid": _PresetBaseline(
        preset="pid",
        engine_label="PiD",
        capabilities=frozenset({"distilled_cfg1"}),
        quant_formats=("bf16", "fp8"),
    ),
    "krea": _PresetBaseline(
        preset="krea",
        engine_label="Krea 2",
        dit_alias="krea2",
        capabilities=frozenset({"multi_reference", "distilled_cfg1", "per_block_compile", "spectrum"}),
        max_references=4,
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
    ),
    "qwen21": _PresetBaseline(
        preset="qwen21",
        engine_label="Qwen-Image-2.1",
        dit_alias="qwen21",
        capabilities=frozenset({"multi_reference", "rgba_output", "control_blocks", "distilled_cfg1", "prefix_kv_cache", "per_block_compile", "spectrum"}),
        max_references=10,
        reference_slot_format="<image_{i}>",
        reference_slot_start=1,
        supports_rgba=True,
        supports_control_blocks=True,
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
    ),
    "moss": _PresetBaseline(
        preset="moss",
        engine_label="MOSS-Image",
        dit_alias="moss",
        capabilities=frozenset({"multi_reference", "structured_layout", "distilled_cfg1", "per_block_compile", "spectrum"}),
        max_references=4,
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
    ),
    "flux3": _PresetBaseline(
        preset="flux3",
        engine_label="FLUX.3",
        dit_alias="flux3",
        capabilities=frozenset({"multi_reference", "structured_layout", "distilled_cfg1", "per_block_compile", "spectrum"}),
        max_references=10,
        reference_slot_format="ref_image_{i}",
        reference_slot_start=0,
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
    ),
}

_PRESET_ALIASES: dict[str, str] = {
    "krea2": "krea",
    "zimage": "zit",
    "z-image": "zit",
    "flux2": "klein",
    "sd15": "sd",
    "sd1": "sd",
    "sdxl": "xl",
    "qwen-2.1": "qwen21",
    "qwen2.1": "qwen21",
    "moss-image": "moss",
    "kromoss": "moss",
    "flux-3": "flux3",
    "flux.3": "flux3",
}

_GGUF_RE = re.compile(r"\.gguf\b|\bgguf\b", re.IGNORECASE)
_INT8_CONVROT_RE = re.compile(r"(?:convrot|int8[-_ ]?convrot|int8)", re.IGNORECASE)
_NVFP4_RE = re.compile(r"(?:nvfp4|\bfp4\b)", re.IGNORECASE)
_SVDQ_RE = re.compile(r"(?:svdq|nunchaku|w4a4)", re.IGNORECASE)
_FP8_RE = re.compile(r"(?:fp8|float8|e4m3|e5m2)", re.IGNORECASE)
_BF16_RE = re.compile(r"(?:\bbf16\b|bfloat16)", re.IGNORECASE)
_TURBO_RE = re.compile(r"(?:^|[-_ /])(?:turbo\d*|dmd2?|distill(?:ed)?|lightning|4[-_]?steps?|8[-_]?steps?)(?:$|[-_ /.])", re.IGNORECASE)
_DIFFUSERS_RE = re.compile(r"(?:diffusers|model_index\.json)", re.IGNORECASE)


def normalize_preset_name(preset: str | None) -> str:
    """Normalize a preset or DiT family alias to a canonical preset key."""

    key = str(preset or "sd").strip().lower() or "sd"
    return _PRESET_ALIASES.get(key, key)


def _slot_start_for_format(slot_format: str | None, fallback: int = 1) -> int:
    if not slot_format:
        return fallback
    lower = slot_format.strip().lower()
    if lower.startswith(("ref_image_", "ref_", "picture ")):
        return 0
    return 1


def _preset_defaults(preset: str) -> tuple[dict[str, Any], bool, bool, int]:
    """Read default sampling profile and flags from ``modules_forge.presets``."""

    from modules_forge import presets as forge_presets

    try:
        arch = forge_presets.PresetArch[preset]
    except KeyError:
        return (
            {"sampler": "Euler", "scheduler": "Simple", "steps": 20, "cfg_scale": 1.0},
            False,
            False,
            1,
        )
    defaults = {
        "sampler": forge_presets.SAMPLERS[arch],
        "scheduler": forge_presets.SCHEDULERS[arch],
        "steps": forge_presets.STEPS[arch],
        "cfg_scale": forge_presets.CFG[arch],
    }
    return (
        defaults,
        forge_presets.use_distill(preset),
        forge_presets.use_shift(preset),
        forge_presets.is_video(preset),
    )


def detect_model_badges(
    preset: str | None,
    *,
    checkpoint: str | None = None,
    unet_dtype: str | None = None,
    supports_rgba: bool = False,
    supports_control_blocks: bool = False,
    is_distilled: bool = False,
) -> tuple[str, ...]:
    """Derive compact Model Bar metadata badges from checkpoint, dtype, and capabilities."""

    badges: list[str] = []
    ckpt = str(checkpoint or "").strip()
    dtype = str(unet_dtype or "").strip()
    combined = f"{ckpt} {dtype}".strip()

    if ckpt and _DIFFUSERS_RE.search(ckpt):
        badges.append("Diffusers")

    if combined:
        if _GGUF_RE.search(combined):
            badges.append("GGUF")
        elif _NVFP4_RE.search(combined):
            badges.append("NVFP4")
        elif _SVDQ_RE.search(combined):
            badges.append("SVDQuant")
        elif _INT8_CONVROT_RE.search(combined):
            badges.append("INT8-ConvRot")
        elif _FP8_RE.search(combined):
            badges.append("FP8")
        elif _BF16_RE.search(combined):
            badges.append("BF16")

    if is_distilled or (ckpt and _TURBO_RE.search(ckpt)):
        badges.append("Turbo")
    if supports_rgba:
        badges.append("RGBA")
    if supports_control_blocks:
        badges.append("Control")

    return tuple(badges)


@dataclass(frozen=True)
class ModelCapabilitySummary:
    """Normalized, UI- and runtime-ready view of an architecture's capabilities."""

    preset: str
    dit_name: str | None
    engine_label: str
    capabilities: frozenset[str]
    max_references: int = 0
    reference_slot_format: str | None = None
    reference_slot_start: int = 1
    supports_rgba: bool = False
    supports_control_blocks: bool = False
    supports_cfg_guidance: bool = True
    supports_negative_prompt: bool = True
    attention_backends: tuple[str, ...] = ()
    quant_formats: tuple[str, ...] = ()
    compile_preset: str | None = None
    is_video: bool = False
    frames: int = 1
    use_distill: bool = False
    use_shift: bool = False
    defaults: dict[str, Any] | None = None
    badges: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "dit_name": self.dit_name,
            "engine_label": self.engine_label,
            "capabilities": sorted(self.capabilities),
            "max_references": int(self.max_references),
            "reference_slot_format": self.reference_slot_format,
            "reference_slot_start": int(self.reference_slot_start),
            "supports_rgba": bool(self.supports_rgba),
            "supports_control_blocks": bool(self.supports_control_blocks),
            "supports_cfg_guidance": bool(self.supports_cfg_guidance),
            "supports_negative_prompt": bool(self.supports_negative_prompt),
            "attention_backends": list(self.attention_backends),
            "quant_formats": list(self.quant_formats),
            "compile_preset": self.compile_preset,
            "is_video": bool(self.is_video),
            "frames": int(self.frames),
            "use_distill": bool(self.use_distill),
            "use_shift": bool(self.use_shift),
            "defaults": dict(self.defaults) if self.defaults else {},
            "badges": list(self.badges),
        }


def _find_registered_dit(preset: str, baseline: _PresetBaseline | None) -> DiTCapability | None:
    cap = capability_for_preset(preset)
    if cap is not None:
        return cap
    if baseline is not None and baseline.dit_alias:
        cap = get_dit_capability(baseline.dit_alias)
        if cap is not None:
            return cap
    return get_dit_capability(preset)


def capability_summary_for_preset(
    preset: str | None,
    *,
    checkpoint: str | None = None,
    unet_dtype: str | None = None,
    sd_model: Any = None,
    opts: Any = None,
) -> dict[str, Any]:
    """Build the unified JSON capability summary for ``preset`` (covering all
    15 ``PresetArch`` values plus any self-registered external DiT family).

    Merges live ``DiTCapability`` registrations over ``PresetArch`` defaults so
    runtime and UI share a single source of truth.
    """

    ensure_ext_capabilities_loaded()
    norm = normalize_preset_name(preset)
    baseline = _PRESET_BASELINES.get(norm)
    dit = _find_registered_dit(norm, baseline)

    defaults, use_distill, use_shift, frames = _preset_defaults(norm)
    is_video = frames > 1

    if checkpoint is None and opts is not None:
        checkpoint = getattr(opts, f"forge_checkpoint_{norm}", None) or getattr(opts, "sd_model_checkpoint", None)
    if checkpoint is None and sd_model is not None:
        checkpoint = getattr(getattr(sd_model, "sd_checkpoint_info", None), "name", None)
    if unet_dtype is None and opts is not None:
        unet_dtype = getattr(opts, f"forge_unet_storage_dtype_{norm}", None) or getattr(opts, "forge_unet_storage_dtype", None)

    if dit is not None:
        caps = set(dit.capabilities)
        if dit.compile_preset:
            caps.add("per_block_compile")
        if dit.spectrum_fast_forward is not None:
            caps.add("spectrum")
        if dit.supports_rgba:
            caps.add("rgba_output")
        if dit.supports_control_blocks:
            caps.add("control_blocks")
        if dit.max_references > 0:
            caps.add("multi_reference")
        if is_video:
            caps.add("video")

        engine_label = dit.engine_label
        dit_name: str | None = dit.name
        max_refs = dit.max_references if dit.max_references > 0 else (baseline.max_references if baseline else 0)
        slot_fmt = dit.reference_slot_format if dit.reference_slot_format is not None else (baseline.reference_slot_format if baseline else None)
        slot_start = _slot_start_for_format(slot_fmt, baseline.reference_slot_start if baseline else 1)
        supports_rgba = bool(dit.supports_rgba or (baseline.supports_rgba if baseline else False))
        supports_control_blocks = bool(dit.supports_control_blocks or (baseline.supports_control_blocks if baseline else False))
        attention_backends = dit.attention_backends or (baseline.attention_backends if baseline else ("sdpa", "flash", "sage"))
        quant_formats = dit.quant_formats or (baseline.quant_formats if baseline else ("bf16", "fp8", "gguf"))
        compile_preset = dit.compile_preset
        supports_cfg = baseline.supports_cfg_guidance if baseline else True
        supports_neg = baseline.supports_negative_prompt if baseline else True
    elif baseline is not None:
        caps = set(baseline.capabilities)
        engine_label = baseline.engine_label
        dit_name = None
        max_refs = baseline.max_references
        slot_fmt = baseline.reference_slot_format
        slot_start = baseline.reference_slot_start
        supports_rgba = baseline.supports_rgba
        supports_control_blocks = baseline.supports_control_blocks
        attention_backends = baseline.attention_backends
        quant_formats = baseline.quant_formats
        compile_preset = None
        supports_cfg = baseline.supports_cfg_guidance
        supports_neg = baseline.supports_negative_prompt
    else:
        caps = set()
        engine_label = norm.upper()
        dit_name = None
        max_refs = 0
        slot_fmt = None
        slot_start = 1
        supports_rgba = False
        supports_control_blocks = False
        attention_backends = ("sdpa",)
        quant_formats = ("bf16", "fp8", "gguf")
        compile_preset = None
        supports_cfg = True
        supports_neg = True

    is_distilled = bool(getattr(sd_model, "is_flux3_distilled", False))
    badges = detect_model_badges(
        norm,
        checkpoint=checkpoint,
        unet_dtype=unet_dtype,
        supports_rgba=supports_rgba,
        supports_control_blocks=supports_control_blocks,
        is_distilled=is_distilled,
    )

    summary = ModelCapabilitySummary(
        preset=norm,
        dit_name=dit_name,
        engine_label=engine_label,
        capabilities=frozenset(caps),
        max_references=max_refs,
        reference_slot_format=slot_fmt,
        reference_slot_start=slot_start,
        supports_rgba=supports_rgba,
        supports_control_blocks=supports_control_blocks,
        supports_cfg_guidance=supports_cfg,
        supports_negative_prompt=supports_neg,
        attention_backends=tuple(attention_backends),
        quant_formats=tuple(quant_formats),
        compile_preset=compile_preset,
        is_video=is_video,
        frames=frames,
        use_distill=use_distill,
        use_shift=use_shift,
        defaults=defaults,
        badges=badges,
    )
    return summary.to_dict()


def resolve_capabilities_for_preset(preset: str | None) -> frozenset[str]:
    """Return the capability token set for ``preset``."""

    summary = capability_summary_for_preset(preset)
    return frozenset(summary["capabilities"])


def all_preset_capability_summaries() -> dict[str, dict[str, Any]]:
    """Capability summaries for all 15 ``PresetArch`` values plus any extra registered DiTs."""

    from modules_forge.presets import PresetArch

    ensure_ext_capabilities_loaded()
    result: dict[str, dict[str, Any]] = {}
    for arch in PresetArch:
        result[arch.name] = capability_summary_for_preset(arch.name)
    for cap in all_dit_capabilities():
        key = cap.preset_arch or cap.name
        if key not in result:
            result[key] = capability_summary_for_preset(key)
    return result


def check_script_applicability(
    requires_capabilities: Collection[str] | None,
    active_capabilities: Collection[str],
    *,
    engine_label: str | None = None,
    preset: str | None = None,
) -> tuple[bool, str | None]:
    """Evaluate whether a script declaring ``requires_capabilities`` applies to
    a model with ``active_capabilities``. Returns ``(applicable, reason_or_none)``."""

    required = [str(cap) for cap in (requires_capabilities or ()) if cap]
    if not required:
        return True, None
    active = set(active_capabilities)
    missing = [cap for cap in required if cap not in active]
    if not missing:
        return True, None
    labels = ", ".join(CAPABILITY_LABELS.get(cap, cap) for cap in missing)
    active_desc = engine_label or preset
    if active_desc:
        return False, f"Requires {labels} (active model: {active_desc})"
    return False, f"Requires {labels}"
