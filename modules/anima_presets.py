"""Shared GUI preset registry for Remi's built-in Anima features.

The preset intentionally owns generation algorithms and their numeric tuning,
but not user-authored content such as prompts, LoRA names, or local model paths.
Scripts bind their params to preset fields declaratively (``preset=`` in
``params()``; ``ScriptRunner.preset_bindings()`` collects them), so the preset
panel sets them all in one click as named values (:func:`preset_named_values`)
without importing extension modules or coupling their runtimes together.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from remi.ui.sentence import Sentence


SAFE_BASE_AESTHETIC_LABEL = "Official Base/Aesthetic — Safe (Recommended)"

SAFE_BASE_AESTHETIC_PRESET: dict[str, object] = {
    # Native Forge generation controls.
    "native.sampler": "ER SDE",
    "native.scheduler": "Automatic",
    "native.steps": 36,
    "native.cfg": 4.5,
    "native.distilled_cfg": 3.0,
    "native.hires": False,
    "native.refiner": False,
    # Guidance and corrective samplers. Optional algorithms remain disabled,
    # while their inactive fields are reset for reproducible later opt-in.
    "guidance.enabled": False,
    "guidance.mode": "Standard / preserve existing",
    "guidance.smc_alpha": 0.2,
    "guidance.smc_lambda": 5.0,
    "guidance.skim_enabled": False,
    "guidance.skim_scale": 2.5,
    "guidance.skim_full_negative": False,
    "guidance.skim_disable_flip": False,
    "guidance.skim_start": 0.0,
    "guidance.skim_end": 1.0,
    "guidance.skim_flip": 0.0,
    "guidance.dcw_enabled": False,
    "guidance.dcw_lambda": -0.015,
    "guidance.dcw_schedule": "one_minus_sigma",
    "guidance.dcw_bands": "LL",
    "guidance.cns_strength": 1.0,
    "guidance.unipc_solver": "bh2",
    "guidance.unipc_disabled_correctors": 0,
    "guidance.unipc_thresholding": False,
    "guidance.unipc_threshold_ratio": 0.995,
    "guidance.unipc_threshold_max": 1.0,
    "guidance.pc3_gamma": 1.0,
    "guidance.pc3_tolerance": 0.005,
    "guidance.modulation_enabled": False,
    "guidance.modulation_weight": 3.0,
    "guidance.modulation_start": 0,
    "guidance.modulation_end": -1,
    "guidance.modulation_adapter_mode": "Auto-download pinned",
    "guidance.modulation_clip_mode": "Auto-download pinned",
    "guidance.nag_enabled": False,
    "guidance.nag_scale": 2.0,
    "guidance.nag_tau": 2.5,
    "guidance.nag_alpha": 0.5,
    "guidance.nag_start": 0.0,
    "guidance.nag_end": 0.5,
    "guidance.momentum_enabled": False,
    "guidance.momentum_strength": 0.5,
    "guidance.momentum_ema": 0.6,
    "guidance.fdg_detail": 12.0,
    "guidance.range_enabled": False,
    "guidance.range_start": 0.0,
    "guidance.range_end": 1.0,
    "guidance.cfg_zero_init_steps": 1,
    "guidance.apg_eta": 0.0,
    "guidance.apg_norm_threshold": 0.0,
    "guidance.apg_momentum": -0.5,
    # Seed diversity and prompt adherence block patches. Every block selection
    # (DAVE's band, Regional's and FreeFuse's full "0-27") is written in the
    # 28-block (2B) layout and mapped onto the 40/52-block models.
    "dave.enabled": False,
    "dave.strength": 0.3,
    "dave.tau": 0.1,
    "dave.blocks": "8-18",
    "xattn_boost.enabled": False,
    "xattn_boost.strength": 2.0,
    "xattn_boost.band": 0.85,
    "xattn_boost.renorm": 0.5,
    # Scheduler and cache acceleration.
    "dynamic_shift.enabled": False,
    "dynamic_shift.start": 5.0,
    "dynamic_shift.end": 1.5,
    "dynamic_shift.curve": "Cosine",
    "dynamic_shift.hires_mode": "Same as first pass",
    "dynamic_shift.hires_start": 3.0,
    "dynamic_shift.hires_end": 1.5,
    "spectrum.enabled": False,
    "spectrum.weight": 0.25,
    "spectrum.degree": 4,
    "spectrum.regularization": 0.1,
    "spectrum.window": 2,
    "spectrum.window_growth": 0.0,
    "spectrum.warmup": 6,
    "spectrum.stop": 0.9,
    "spectrum.tail_actual": 3,
    "spectrum.history": 10,
    "spectrum.schedule": "Window",
    "spectrum.refresh_ratio": 0.0,
    "spectrum.sea_beta": 2.0,
    "spectrum.policy": "Conservative",
    "spectrum.verbose": False,
    "compile.preset": "Automatic",
    # Spatial and post-processing stacks.
    "regional.enabled": False,
    "regional.blocks": "0-27",
    "regional.start": 0.0,
    "regional.end": 0.65,
    "regional.feather": 0.03,
    "regional.base_preserve": 0.15,
    "regional.region1_active": True,
    "regional.region1_x": 0.0,
    "regional.region1_y": 0.0,
    "regional.region1_width": 0.5,
    "regional.region1_height": 1.0,
    "regional.region1_strength": 1.0,
    "regional.region2_active": True,
    "regional.region2_x": 0.5,
    "regional.region2_y": 0.0,
    "regional.region2_width": 0.5,
    "regional.region2_height": 1.0,
    "regional.region2_strength": 1.0,
    "regional.region3_active": False,
    "regional.region3_x": 0.25,
    "regional.region3_y": 0.25,
    "regional.region3_width": 0.5,
    "regional.region3_height": 0.5,
    "regional.region3_strength": 1.0,
    "artist.enabled": False,
    "artist.global_strength": 0.7,
    "artist.optimization": "Balance",
    "artist.combine": "Output average",
    "artist.fusion": "Interpolate",
    "artist.cache": True,
    "artist.apply_uncond": False,
    "freefuse.enabled": False,
    "freefuse.collect_step": 3,
    "freefuse.collect_block": 18,
    "freefuse.top_k": 0.10,
    "freefuse.temperature": 300,
    "freefuse.background_scale": 0.95,
    "freefuse.balance_iterations": 15,
    "freefuse.feather": 1,
    "freefuse.routing_strength": 1.0,
    "freefuse.routing_end": 1.0,
    "freefuse.bias_scale": 6.0,
    "freefuse.positive_bias": 1.0,
    "freefuse.bias_blocks": "0-27",
    "pid.enabled": False,
    "pid.degrade_sigma": 0.0,
    "pid.color_correction": True,
    "hires_guard.enabled": False,
    "hires_guard.policy": "Clamp unsafe values",
    "hires_guard.preset": "Balanced",
    "hires_guard.align": True,
    "hires_guard.base_mp": 1.2,
    "hires_guard.hires_mp": 2.1,
    "hires_guard.max_upscale": 1.75,
    # Auxiliary opt-in helpers.
    "resolution.random_standard": False,
    "resolution.random_highres": False,
    "prompt_rescale.enabled": False,
    "prompt_rescale.source_steps": 35,
    "prompt_rescale.fractions": False,
    "lora_stage.enabled": False,
    "lora_layer.lora_enabled": False,
    "lora_layer.lokr_enabled": False,
}

TURBO_LABEL = "Official Turbo — Fast (CFG 1 · 10 steps)"

# Anima-Turbo (v1.0 / v1.1) is step- and CFG-distilled: the model card
# recommends CFG 1 and 8-12 steps, and Euler for the stable Turbo/Aesthetic
# versions. At CFG 1 Forge skips the unconditional pass, which is where most
# of Turbo's speed comes from, so every CFG-dependent stack stays off.
TURBO_OVERRIDES: dict[str, object] = {
    "native.sampler": "Euler",
    "native.steps": 10,
    "native.cfg": 1.0,
}

TURBO_NAG_LABEL = "Turbo + NAG negative prompt (CFG 1 · 10 steps)"

# CFG 1 drops the negative prompt. Normalized Attention Guidance (Chen et
# al., 2025) restores negative guidance for few-step distilled models, where
# CFG fails, by extrapolating in cross-attention space; it only touches
# cross-attention, and without an unconditional model pass.
TURBO_NAG_OVERRIDES: dict[str, object] = {
    **TURBO_OVERRIDES,
    "guidance.enabled": True,
    "guidance.nag_enabled": True,
}

ANIMA_2_9B_LABEL = "Anima-2.9B (Euler · SGM Uniform · 30 steps · CFG 4)"

# The depth-expanded models' cards: Anima-2.9B (Gazingstars123) recommends Euler with
# SGM Uniform, 28-50 steps and CFG 3.5-5; Anima-3.8B (lylogummy) Res Multistep with Beta,
# 28-50 steps and CFG 4-7, at about one megapixel.
ANIMA_2_9B_OVERRIDES: dict[str, object] = {
    "native.sampler": "Euler",
    "native.scheduler": "SGM Uniform",
    "native.steps": 30,
    "native.cfg": 4.0,
}

ANIMA_3_8B_LABEL = "Anima-3.8B (Res Multistep · Beta · 32 steps · CFG 5)"

ANIMA_3_8B_OVERRIDES: dict[str, object] = {
    "native.sampler": "Res Multistep",
    "native.scheduler": "Beta",
    "native.steps": 32,
    "native.cfg": 5.0,
}

# Label -> overrides applied on top of SAFE_BASE_AESTHETIC_PRESET, in menu
# order. Overrides only name fields of the base preset.
PRESETS: dict[str, dict[str, object]] = dict(
    (
        (SAFE_BASE_AESTHETIC_LABEL, {}),
        (TURBO_LABEL, TURBO_OVERRIDES),
        (TURBO_NAG_LABEL, TURBO_NAG_OVERRIDES),
        (ANIMA_2_9B_LABEL, ANIMA_2_9B_OVERRIDES),
        (ANIMA_3_8B_LABEL, ANIMA_3_8B_OVERRIDES),
    )
)

PRESET_DESCRIPTIONS: dict[str, str] = {
    SAFE_BASE_AESTHETIC_LABEL: (
        "**Base / Aesthetic:** ER SDE · 36 steps · CFG 4.5 · Automatic scheduler."
    ),
    TURBO_LABEL: (
        "**Turbo (distilled):** Euler · 10 steps · CFG 1 · Automatic scheduler. "
        "Use only with Anima-Turbo checkpoints; CFG above 1 or extra guidance "
        "over-cooks distilled models. CFG 1 ignores the negative prompt: enable "
        "NAG (Anima Guidance) to apply it without a second model pass."
    ),
    TURBO_NAG_LABEL: (
        "**Turbo + NAG:** the Turbo settings with Normalized Attention Guidance "
        "on (scale 2, first half of the schedule), so the negative prompt "
        "applies at CFG 1. Costs extra cross-attention only, not a second "
        "model pass. For Anima-Turbo checkpoints."
    ),
    ANIMA_2_9B_LABEL: (
        "**Anima-2.9B:** Euler · SGM Uniform · 30 steps · CFG 4, the 40-block model's "
        "recommended settings (28-50 steps, CFG 3.5-5)."
    ),
    ANIMA_3_8B_LABEL: (
        "**Anima-3.8B:** Res Multistep · Beta · 32 steps · CFG 5, the 52-block model's "
        "recommended settings (28-50 steps, CFG 4-7, about one megapixel). Needs "
        "qwen35_4b.safetensors among the text encoder modules."
    ),
}


_LORA_TAG_RE = re.compile(r"<(?:lora|lyco):([^:>]+)(?::([^:>]*))?", re.IGNORECASE)
_DISTILLED_LORA_RE = re.compile(r"(?:^|[^a-z0-9])(?:turbo|lightning|lcm|hyper|dmd2?)(?:[^a-z0-9]|$)", re.IGNORECASE)
_EDIT_HINT_RE = re.compile(r"(?:^|[^a-z0-9])edit(?:[^a-z0-9]|$)", re.IGNORECASE)


def _iter_prompt_texts(prompts: Iterable[str] | str | None) -> list[str]:
    if prompts is None:
        return []
    if isinstance(prompts, str):
        return [prompts]
    return [str(text) for text in prompts if text]


def _active_lora_names(prompts: Iterable[str] | str | None) -> list[str]:
    names: list[str] = []
    for prompt in _iter_prompt_texts(prompts):
        for match in _LORA_TAG_RE.finditer(prompt):
            name = match.group(1).strip()
            raw_weight = (match.group(2) or "").strip()
            if raw_weight:
                try:
                    if float(raw_weight) == 0.0:
                        continue
                except ValueError:
                    pass
            if name:
                names.append(name)
    return names


def has_distilled_lora(prompts: Iterable[str] | str | None) -> bool:
    """Whether any active LoRA tag in ``prompts`` names a step/CFG-distilled LoRA."""

    return any(_DISTILLED_LORA_RE.search(name) for name in _active_lora_names(prompts))


def _has_edit_hint(checkpoint: str, prompts: Iterable[str] | str | None) -> bool:
    return bool(_EDIT_HINT_RE.search(str(checkpoint or ""))) or any(_EDIT_HINT_RE.search(name) for name in _active_lora_names(prompts))


def _has_transparent_alpha(image: object) -> bool:
    getbands = getattr(image, "getbands", None)
    getchannel = getattr(image, "getchannel", None)
    if not callable(getbands) or not callable(getchannel):
        return False
    try:
        if "A" not in getbands():
            return False
        extrema = getchannel("A").getextrema()
    except Exception:
        return False
    return isinstance(extrema, tuple) and len(extrema) >= 1 and extrema[0] < 255


def _is_white_background(value: object) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"#fff", "#ffffff", "#ffffffff", "fff", "ffffff", "white"}:
            return True
        if text.startswith("rgb"):
            nums = [int(n) for n in re.findall(r"\d+", text)[:3]]
            return len(nums) == 3 and nums == [255, 255, 255]
        return False
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        try:
            return all(int(c) == 255 for c in value[:3])
        except (TypeError, ValueError):
            return False
    return False


def turbo_settings_advice(
    checkpoint: str,
    cfg_scale: float,
    steps: int,
    *,
    prompts: Iterable[str] | str | None = None,
) -> str | None:
    """A hint when the settings clearly do not match the Anima checkpoint
    (distilled Turbo vs Base/Aesthetic), judged by its file name and active LoRAs.
    The hints here are the advice bar's sentences (``backend/nn/dit_registry.py``),
    word for word: one catalog entry translates both, and a job that notes both
    shows the note once."""

    name = str(checkpoint or "").lower()
    is_turbo = "turbo" in name
    if is_turbo and (cfg_scale > 1.5 or steps > 16):
        return Sentence("Anima-Turbo checkpoint with CFG {{cfg}} / {{steps}} steps: it is distilled for CFG 1 and 8-12 steps (see the Official Turbo preset); higher values over-cook it.", cfg=f"{cfg_scale:g}", steps=int(steps))
    if not is_turbo and "anima" in name and cfg_scale <= 1.0 and steps <= 12:
        if has_distilled_lora(prompts):
            return None
        return Sentence("Non-Turbo Anima checkpoint with CFG {{cfg}} / {{steps}} steps: Base/Aesthetic need about 30-50 steps and CFG 4-5 (Turbo checkpoints are meant for these settings).", cfg=f"{cfg_scale:g}", steps=int(steps))
    return None


def anima_workflow_advice(p, *, opts=None) -> list[str]:
    """Collect actionable setup hints for an Anima generation pass (Turbo/Base
    mismatch, Anima Edit without Reference enabled, RGBA 4ch VAE transparent
    img2img with a non-white fill background)."""

    if opts is None:
        try:
            from modules import shared

            opts = getattr(shared, "opts", None)
        except Exception:
            opts = None

    sd_model = getattr(p, "sd_model", None)
    checkpoint = getattr(getattr(sd_model, "sd_checkpoint_info", None), "name", "")
    prompts: list[str] = []
    for attr in ("prompt", "hr_prompt"):
        if value := getattr(p, attr, None):
            prompts.append(str(value))
    for attr in ("prompts", "hr_prompts"):
        if values := getattr(p, attr, None):
            prompts.extend(str(v) for v in values if v)

    messages: list[str] = []
    cfg_scale = getattr(p, "cfg_scale", None)
    steps = getattr(p, "steps", None)
    if cfg_scale is not None and steps is not None:
        if advice := turbo_settings_advice(checkpoint, float(cfg_scale), int(steps), prompts=prompts):
            messages.append(advice)

    init_images = getattr(p, "init_images", None) or ()
    has_ref_latents = False
    try:
        from backend.args import dynamic_args

        has_ref_latents = bool(getattr(dynamic_args, "ref_latents", None))
    except Exception:
        has_ref_latents = False

    if (
        opts is not None
        and not bool(getattr(opts, "anima_do_reference", False))
        and _has_edit_hint(checkpoint, prompts)
        and (bool(init_images) or has_ref_latents)
    ):
        messages.append(
            "Anima Edit model/LoRA detected with an input or reference image while '[Anima] Enable Reference' "
            "(anima_do_reference) is off: enable it so Anima stacks the reference frame instead of running plain img2img."
        )

    vae = getattr(getattr(sd_model, "forge_objects", None), "vae", None)
    if getattr(vae, "output_channels", 3) == 4 and init_images and any(_has_transparent_alpha(img) for img in init_images):
        bg = getattr(opts, "img2img_background_color", "#808080") if opts is not None else "#808080"
        if not _is_white_background(bg):
            messages.append(Sentence("Anima RGBA (4-channel) VAE with a transparent img2img input and img2img_background_color={{color}}: set img2img_background_color to #ffffff (white) to avoid dark-fringe alpha artifacts.", color=repr(bg)))

    return messages


def preset_value(label: str, name: str, registered: object) -> object:
    """The value ``label`` gives the param bound with ``registered``
    (the base preset value, possibly localized by the extension)."""

    overrides = PRESETS.get(label, {})
    return overrides.get(name, registered)


def resolved_preset(label: str) -> dict[str, object]:
    """Every preset field's value under ``label`` (the base preset with the
    label's overrides; an unknown label gives the base preset)."""

    return {**SAFE_BASE_AESTHETIC_PRESET, **PRESETS.get(label, {})}


def preset_named_values(label: str, bindings) -> dict[str, dict[str, object]]:
    """Named values ``label`` gives the bound params: ``{owner: {param: value}}``.

    ``bindings`` is ``[(field, owner, param, own value or None), ...]``; an
    own value (e.g. a localized choice) replaces the base preset's, but not a
    label's override, exactly as :func:`preset_value`. Pure."""

    values: dict[str, dict[str, object]] = {}
    overrides = PRESETS.get(label, {})
    for name, owner, param, own in bindings:
        if name not in SAFE_BASE_AESTHETIC_PRESET:
            raise KeyError(f"Unknown Anima preset field: {name}")
        value = overrides[name] if name in overrides else (SAFE_BASE_AESTHETIC_PRESET[name] if own is None else own)
        values.setdefault(owner, {})[param] = value
    return values


# Only these controls are ever locked by Remi's mutual-exclusion GUI. The
# preset unlocks them after moving the whole set back to a compatible state;
# all other components keep their extension-defined interactivity.
PRESET_CONFLICT_CONTROLS = frozenset(
    {
        "native.sampler",
        "native.scheduler",
        "native.hires",
        "native.refiner",
        "guidance.mode",
        "guidance.momentum_enabled",
        "guidance.dcw_enabled",
        "dynamic_shift.enabled",
        "regional.enabled",
        "artist.enabled",
        "freefuse.enabled",
        "pid.enabled",
    }
)
