"""Declarative capability registry for DiT architectures (Anima, MOSS-Image,
Krea 2, FLUX.3, Qwen-Image-2.1, and future models).

Centralizes per-architecture metadata and hooks so shared infrastructure
(on-the-fly quantization, Diffusers/sharded checkpoint discovery and loading,
per-block ``torch.compile``, Spectrum pre-head forecasting, and workflow advice)
stays data-driven and upstream Forge Neo files keep minimal 1–3 line hooks.

Kept lightweight (stdlib + ``torch`` at import time; all model/loader imports
are deferred inside callables) so standalone extension tests import cleanly.
"""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from remi.ui.sentence import Sentence, sentence_fields

import torch
from torch import nn


@dataclass(frozen=True)
class AdviceItem:
    """Actionable structured workflow advice for pre-submit UI chips and runtime hints."""

    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    patch: dict[str, Any] | None = None
    action_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "patch": dict(self.patch) if self.patch is not None else None,
        }
        if self.action_label is not None:
            data["action_label"] = self.action_label
        data.update(sentence_fields(self.message))
        return data


@dataclass(frozen=True)
class DiTCapability:
    """Declarative specification of a DiT family's capabilities and hooks."""

    name: str
    dit_cls_name: str
    model_type_name: str
    engine_label: str
    # On-the-fly FP8 / scaled FP8 / INT8-ConvRot quantization
    sensitive_prefixes: tuple[str, ...] = ()
    sensitive_module_types: tuple[str, ...] = ()
    matches_state_dict: Callable[[dict[str, torch.Tensor]], bool] | None = None
    block_linear_filter: Callable[[str], bool] | None = None
    fp8_rowwise: bool = False
    fp8_full_precision_matmul_filter: Callable[[str], bool] | None = None
    int8_convrot_filter: Callable[[str], bool] | None = None
    # Diffusers / sharded checkpoint discovery & conversion
    diffusers_arch: str | None = None
    is_diffusers_dir: Callable[[os.PathLike | str], bool] | None = None
    diffusers_primary_file: Callable[[os.PathLike | str], str | None] | None = None
    load_diffusers_dir: Callable[[os.PathLike | str], tuple[dict[str, torch.Tensor], dict]] | None = None
    is_diffusers_single_file: Callable[[dict[str, torch.Tensor]], bool] | None = None
    convert_diffusers_single_file: Callable[[dict[str, torch.Tensor], str], dict[str, torch.Tensor]] | None = None
    # Per-block torch.compile
    block_attr: str | None = None
    compile_preset: str | None = None
    matches_engine: Callable[[object], bool] | None = None
    non_matching_engine_reason: str | None = None
    # Spectrum pre-head feature forecasting
    spectrum_required_attrs: tuple[str, ...] = ()
    spectrum_capture_attr: str | None = None
    spectrum_fast_forward: Callable[..., torch.Tensor] | None = None
    # Non-blocking workflow advice
    workflow_advice: Callable[..., list[str]] | None = None
    # Full-stack declarative capability metadata (UI visibility, reference slots, execution plan)
    capabilities: frozenset[str] = frozenset()
    preset_arch: str | None = None
    max_references: int = 0
    reference_slot_format: str | None = None
    supports_rgba: bool = False
    supports_control_blocks: bool = False
    attention_backends: tuple[str, ...] = ()
    quant_formats: tuple[str, ...] = ()
    # Structured advice (runtime + pre-submit form-level)
    structured_advice: Callable[..., list[AdviceItem]] | None = None
    form_advice: Callable[..., list[AdviceItem]] | None = None


_REGISTRY: dict[str, DiTCapability] = {}


def register_dit_capability(capability: DiTCapability) -> DiTCapability:
    """Register (or replace) a ``DiTCapability`` by its ``name``."""

    _REGISTRY[capability.name] = capability
    return capability


def unregister_dit_capability(name: str) -> DiTCapability | None:
    """Remove a registered ``DiTCapability`` by ``name`` (mainly for tests)."""

    return _REGISTRY.pop(name, None)


def all_dit_capabilities() -> tuple[DiTCapability, ...]:
    return tuple(_REGISTRY.values())


def get_dit_capability(name: str) -> DiTCapability | None:
    return _REGISTRY.get(name)


_PRESET_TO_DIT: dict[str, str] = {
    "krea": "krea2",
    "krea2": "krea2",
    "zit": "zimage",
    "zimage": "zimage",
}
_DIT_TO_PRESET: dict[str, str] = {
    "krea2": "krea",
    "zimage": "zit",
}


def capability_for_preset(preset: str | None) -> DiTCapability | None:
    """Lookup registered ``DiTCapability`` by ``PresetArch`` name or alias."""

    if not preset:
        return None
    key = str(preset).strip().lower()
    if not key:
        return None
    for cap in _REGISTRY.values():
        if cap.preset_arch == key or cap.name == key:
            return cap
    alias = _PRESET_TO_DIT.get(key)
    if alias:
        return _REGISTRY.get(alias)
    return None


def capability_for_dit_cls(cls_name: str) -> DiTCapability | None:
    """Lookup by loader diffusers class name (e.g. ``CosmosTransformer3DModel``)."""

    for cap in _REGISTRY.values():
        if cap.dit_cls_name == cls_name:
            return cap
    return None


def capability_for_model(model: nn.Module) -> DiTCapability | None:
    """Lookup by live ``nn.Module`` class name (e.g. ``Anima``, ``MOSSImage``, ``SingleStreamDiT``)."""

    cls_name = type(model).__name__
    for cap in _REGISTRY.values():
        if cap.model_type_name == cls_name or cap.dit_cls_name == cls_name:
            return cap
    return None


def capability_for_state_dict(state_dict: dict[str, torch.Tensor]) -> DiTCapability | None:
    """Identify the registered DiT family matching ``state_dict``."""

    for cap in _REGISTRY.values():
        if cap.matches_state_dict is not None and cap.matches_state_dict(state_dict):
            return cap
    # Fallback for minimal synthetic Anima-like dicts containing only ``blocks.*`` weights.
    if any(k.startswith("blocks.") for k in state_dict):
        return _REGISTRY.get("anima")
    return None


def default_sensitive_prefixes() -> tuple[str, ...]:
    """Fallback sensitive prefixes for synthetic/untyped DiT modules (Anima + MOSS)."""

    prefixes: list[str] = []
    for name in ("anima", "moss"):
        cap = _REGISTRY.get(name)
        if cap is not None:
            for prefix in cap.sensitive_prefixes:
                if prefix not in prefixes:
                    prefixes.append(prefix)
    return tuple(prefixes)


def default_sensitive_module_types() -> tuple[str, ...]:
    """Union of all registered precision-sensitive module class names."""

    names: list[str] = []
    for cap in _REGISTRY.values():
        for cls_name in cap.sensitive_module_types:
            if cls_name not in names:
                names.append(cls_name)
    return tuple(names)


# region Diffusers / sharded checkpoint helpers


def is_diffusers_checkpoint_dir(path: os.PathLike | str) -> bool:
    """True for any registered Diffusers or standalone sharded transformer directory."""

    return any(cap.is_diffusers_dir is not None and cap.is_diffusers_dir(path) for cap in _REGISTRY.values())


def diffusers_checkpoint_info(path: os.PathLike | str) -> tuple[str | None, str | None]:
    """Return ``(modelspec_architecture, primary_safetensors_file)`` for a checkpoint dir."""

    for cap in _REGISTRY.values():
        if cap.is_diffusers_dir is not None and cap.is_diffusers_dir(path):
            primary = cap.diffusers_primary_file(path) if cap.diffusers_primary_file is not None else None
            return cap.diffusers_arch, primary
    return None, None


def load_diffusers_checkpoint_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load the first registered Diffusers/sharded checkpoint directory matching ``path``."""

    for cap in _REGISTRY.values():
        if cap.is_diffusers_dir is not None and cap.load_diffusers_dir is not None and cap.is_diffusers_dir(path):
            return cap.load_diffusers_dir(path)
    raise FileNotFoundError(f"Unrecognized Diffusers checkpoint directory: {os.fspath(path)}")


def convert_registered_diffusers_mmdit(state_dict: dict[str, torch.Tensor], output_prefix: str = "") -> dict[str, torch.Tensor] | None:
    """Convert a standalone single-file Diffusers transformer state dict if a registered DiT matches."""

    for cap in _REGISTRY.values():
        if cap.is_diffusers_single_file is not None and cap.convert_diffusers_single_file is not None and cap.is_diffusers_single_file(state_dict):
            return cap.convert_diffusers_single_file(state_dict, output_prefix)
    return None


# endregion


# region Per-block torch.compile helpers


def registered_block_attrs() -> tuple[str, ...]:
    """Unique transformer block ``ModuleList`` attribute names across registered DiTs."""

    attrs: list[str] = []
    for cap in _REGISTRY.values():
        if cap.block_attr and cap.block_attr not in attrs:
            attrs.append(cap.block_attr)
    return tuple(attrs)


def per_block_compile_presets() -> tuple[str, ...]:
    """All registered per-block ``torch.compile`` preset names."""

    return tuple(cap.compile_preset for cap in _REGISTRY.values() if cap.compile_preset)


def capability_for_compile_preset(preset: str | None) -> DiTCapability | None:
    """Return the capability owning ``preset`` (e.g. ``"Anima per-block"``), or ``None``."""

    if not preset:
        return None
    for cap in _REGISTRY.values():
        if cap.compile_preset == preset:
            return cap
    return None


# endregion


# region Spectrum pre-head forecasting helpers


def fast_forward_anima(dit, predictor, sigma, input_x, feature):
    # The feature keeps the compute dtype final_layer saw in the real forward
    # (Anima casts to the context dtype); the parameters' storage dtype (bf16,
    # FP8, ...) is not what the manual-cast layers compute in.
    internal_timestep = predictor.timestep(sigma).float()
    if internal_timestep.ndim == 1:
        internal_timestep = internal_timestep.unsqueeze(1)
    sinusoidal = dit.t_embedder[0](internal_timestep)
    time_embedding, adaln = dit.t_embedder[1](sinusoidal.to(feature.dtype))
    time_embedding = dit.t_embedding_norm(time_embedding)
    model_output = dit.final_layer(feature, time_embedding, adaln_lora_B_T_3D=adaln)
    model_output = dit.unpatchify(model_output)
    spatial = input_x.shape[-3:]
    model_output = model_output[:, :, : spatial[-3], : spatial[-2], : spatial[-1]]
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


def fast_forward_moss(dit, predictor, sigma, input_x, feature):
    internal_timestep = predictor.timestep(sigma).float()
    model_output = dit.compute_video_head(feature, internal_timestep, tuple(input_x.shape))
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


def fast_forward_krea2(dit, predictor, sigma, input_x, feature):
    internal_timestep = predictor.timestep(sigma).float()
    model_output = dit.compute_last(feature, internal_timestep, tuple(input_x.shape))
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


def fast_forward_flux3(dit, predictor, sigma, input_x, feature):
    internal_timestep = predictor.timestep(sigma).float()
    model_output = dit.compute_final(feature, internal_timestep, tuple(input_x.shape))
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


def fast_forward_qwen21(dit, predictor, sigma, input_x, feature):
    internal_timestep = predictor.timestep(sigma).float()
    model_output = dit.compute_final(feature, internal_timestep, tuple(input_x.shape))
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


def match_spectrum_capability(dit) -> DiTCapability | None:
    """Return the capability whose Spectrum pre-head seam ``dit`` satisfies."""

    if dit is None:
        return None
    for cap in _REGISTRY.values():
        if cap.spectrum_required_attrs and cap.spectrum_capture_attr and all(hasattr(dit, attr) for attr in cap.spectrum_required_attrs):
            return cap
    return None


def resolve_spectrum_capture_module(target_module):
    """If ``target_module`` is a DiT exposing a registered Spectrum capture attr,
    return that submodule; otherwise return ``target_module`` unchanged."""

    cap = match_spectrum_capability(target_module)
    if cap is not None:
        return getattr(target_module, cap.spectrum_capture_attr)
    for cap in _REGISTRY.values():
        attr = cap.spectrum_capture_attr
        if attr and hasattr(target_module, attr):
            return getattr(target_module, attr)
    return target_module


# endregion


# region Workflow advice helpers

_MOSS_FEW_STEP_RE = re.compile(r"(?:^|[-_ /])(?:turbo|dmd2?|4[-_]?steps?|8[-_]?steps?|16[-_]?steps?)(?:$|[-_ /.])", re.IGNORECASE)
_MOSS_DMD2_8STEP_RE = re.compile(r"(?:^|[-_ /])(?:dmd2|8[-_]?steps?)(?:$|[-_ /.])", re.IGNORECASE)
_MOSS_4STEP_RE = re.compile(r"(?:^|[-_ /])4[-_]?steps?(?:$|[-_ /.])", re.IGNORECASE)
_MOSS_16STEP_RE = re.compile(r"(?:^|[-_ /])16[-_]?steps?(?:$|[-_ /.])", re.IGNORECASE)
_MOSS_KROMOSS_RE = re.compile(r"(?:^|[-_ /])kromoss(?:$|[-_ /.])", re.IGNORECASE)
_MOSS_EDIT_RE = re.compile(r"(?:^|[-_ /])(?:moss[-_ ]?edit|moss[-_ ]?image[-_ ]?edit)(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_TURBO_RE = re.compile(r"(?:^|[-_ /])turbo(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_RAW_RE = re.compile(r"(?:^|[-_ /])raw(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_EDIT_RE = re.compile(r"(?:^|[-_ /])(?:krea[-_ ]?2[-_ ]?edit|krea2edit)(?:$|[-_ /.])", re.IGNORECASE)
_ANIMA_DISTILLED_LORA_RE = re.compile(r"(?:^|[^a-z0-9])(?:turbo|lightning|lcm|hyper|dmd2?)(?:[^a-z0-9]|$)", re.IGNORECASE)
_ANIMA_EDIT_RE = re.compile(r"(?:^|[^a-z0-9])edit(?:[^a-z0-9]|$)", re.IGNORECASE)
_LORA_WITH_WEIGHT_RE = re.compile(r"<(?:lora|lyco):([^:>]+)(?::([^:>]*))?", re.IGNORECASE)
_RGBA_PROMPT_RE = re.compile(r"(?:\brgba\b|\btransparent\b|\balpha\s+channel\b|透明背景|透明图|透明通道)", re.IGNORECASE)


def _active_lora_names_from_prompts(prompts: list[str]) -> list[str]:
    names: list[str] = []
    for prompt_text in prompts:
        for match in _LORA_WITH_WEIGHT_RE.finditer(prompt_text):
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


def _prompt_and_checkpoint_names(p) -> tuple[str, list[str]]:
    sd_model = getattr(p, "sd_model", None)
    ckpt_name = str(getattr(getattr(sd_model, "sd_checkpoint_info", None), "name", "") or "")
    texts: list[str] = []
    for prompt_text in (getattr(p, "prompt", None), getattr(p, "hr_prompt", None), *(getattr(p, "all_prompts", None) or ()), *(getattr(p, "prompts", None) or ())):
        if isinstance(prompt_text, str) and prompt_text:
            texts.append(prompt_text)
    lora_names = _active_lora_names_from_prompts(texts)
    return ckpt_name, lora_names


def _resolve_opts(opts):
    if opts is not None:
        return opts
    try:
        from modules import shared

        return shared.opts
    except Exception:
        return None


def _is_white_color(value: object) -> bool:
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


def _has_transparent_alpha_image(image: object) -> bool:
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


def _moss_recommended_steps(combined: str) -> int:
    if _MOSS_4STEP_RE.search(combined):
        return 4
    if _MOSS_DMD2_8STEP_RE.search(combined):
        return 8
    if _MOSS_16STEP_RE.search(combined):
        return 16
    return 8


def _evaluate_anima_advice(
    *,
    ckpt_name: str,
    lora_names: list[str],
    prompt: str,
    steps: int | float | None,
    cfg: int | float | None,
    has_refs: bool,
    has_transparent_init: bool,
    opts: Any,
    vae_is_rgba: bool = False,
) -> list[AdviceItem]:
    items: list[AdviceItem] = []
    lower_ckpt = ckpt_name.lower()
    is_turbo = "turbo" in lower_ckpt
    has_distilled_lora = any(_ANIMA_DISTILLED_LORA_RE.search(n) for n in lora_names)

    if isinstance(steps, (int, float)) and isinstance(cfg, (int, float)):
        if is_turbo and (cfg > 1.5 or steps > 16):
            items.append(
                AdviceItem(
                    code="anima.turbo_mismatch",
                    severity="warning",
                    message=(
                        Sentence("Anima-Turbo checkpoint with CFG {{cfg}} / {{steps}} steps: it is distilled for CFG 1 and 8-12 steps (see the Official Turbo preset); higher values over-cook it.", cfg=f"{cfg:g}", steps=int(steps))
                    ),
                    patch={"sampler.steps": {"value": 10}, "cfg_scale": {"value": 1.0}, "sampler.sampler_name": {"value": "Euler"}},
                    action_label="Set Turbo (10 steps · CFG 1.0)",
                )
            )
        elif not is_turbo and "anima" in lower_ckpt and cfg <= 1.0 and steps <= 12 and not has_distilled_lora:
            items.append(
                AdviceItem(
                    code="anima.base_low_steps",
                    severity="warning",
                    message=(
                        Sentence("Non-Turbo Anima checkpoint with CFG {{cfg}} / {{steps}} steps: Base/Aesthetic need about 30-50 steps and CFG 4-5 (Turbo checkpoints are meant for these settings).", cfg=f"{cfg:g}", steps=int(steps))
                    ),
                    patch={"sampler.steps": {"value": 36}, "cfg_scale": {"value": 4.5}},
                    action_label="Set Base (36 steps · CFG 4.5)",
                )
            )

    has_edit_hint = bool(_ANIMA_EDIT_RE.search(ckpt_name)) or any(_ANIMA_EDIT_RE.search(n) for n in lora_names)
    do_ref = bool(getattr(opts, "anima_do_reference", False)) if opts is not None else False
    if opts is not None and not do_ref and has_edit_hint and has_refs:
        items.append(
            AdviceItem(
                code="anima.edit_ref_disabled",
                severity="warning",
                message=(
                    "Anima Edit model/LoRA detected with an input or reference image while '[Anima] Enable Reference' "
                    "(anima_do_reference) is off: enable it so Anima stacks the reference frame instead of running plain img2img."
                ),
                patch={"__options__": {"anima_do_reference": True}},
                action_label="Enable Anima Reference",
            )
        )

    if vae_is_rgba and has_transparent_init:
        bg = getattr(opts, "img2img_background_color", "#808080") if opts is not None else "#808080"
        if not _is_white_color(bg):
            items.append(
                AdviceItem(
                    code="anima.rgba_non_white_bg",
                    severity="warning",
                    message=(
                        Sentence("Anima RGBA (4-channel) VAE with a transparent img2img input and img2img_background_color={{color}}: set img2img_background_color to #ffffff (white) to avoid dark-fringe alpha artifacts.", color=repr(bg))
                    ),
                    patch={"__options__": {"img2img_background_color": "#ffffff"}},
                    action_label="Set White Background (#ffffff)",
                )
            )

    samples_format = str(getattr(opts, "samples_format", "png") or "png").lower() if opts is not None else "png"
    wants_rgba = vae_is_rgba or "rgba" in lower_ckpt or (isinstance(prompt, str) and bool(_RGBA_PROMPT_RE.search(prompt)))
    if wants_rgba and samples_format in ("jpg", "jpeg"):
        items.append(
            AdviceItem(
                code="anima.rgba_jpeg_format",
                severity="warning",
                message=(
                    Sentence("Anima RGBA transparent setup detected while samples_format={{format}}: JPEG drops the alpha channel; switch the output format to PNG or WebP to keep transparency.", format=repr(samples_format))
                ),
                patch={"__options__": {"samples_format": "png"}},
                action_label="Switch Output to PNG",
            )
        )
    return items


def anima_structured_advice(p, *, opts=None) -> list[AdviceItem]:
    """Structured workflow advice for Anima runtime passes."""

    from backend.args import dynamic_args

    sd_model = getattr(p, "sd_model", None)
    if not (dynamic_args.anima or getattr(sd_model, "is_anima_engine", False)):
        return []

    active_opts = _resolve_opts(opts)
    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    init_images = getattr(p, "init_images", None) or ()
    has_refs = bool(init_images) or bool(getattr(dynamic_args, "ref_latents", None)) or bool(getattr(sd_model, "ref_latents", None))
    vae = getattr(getattr(sd_model, "forge_objects", None), "vae", None)
    vae_is_rgba = getattr(vae, "output_channels", 3) == 4
    has_transparent_init = bool(init_images) and any(_has_transparent_alpha_image(img) for img in init_images)
    return _evaluate_anima_advice(
        ckpt_name=ckpt_name,
        lora_names=lora_names,
        prompt=str(getattr(p, "prompt", "") or ""),
        steps=getattr(p, "steps", None),
        cfg=getattr(p, "cfg_scale", None),
        has_refs=has_refs,
        has_transparent_init=has_transparent_init,
        opts=active_opts,
        vae_is_rgba=vae_is_rgba,
    )


def _evaluate_moss_advice(
    *,
    ckpt_name: str,
    lora_names: list[str],
    steps: int | float | None,
    cfg: int | float | None,
    has_refs: bool,
    opts: Any,
) -> list[AdviceItem]:
    items: list[AdviceItem] = []
    ref_mode = getattr(opts, "moss_do_reference", "Disable") if opts is not None else "Disable"
    if has_refs and ref_mode == "Disable":
        items.append(
            AdviceItem(
                code="moss.ref_disabled",
                severity="warning",
                message=(
                    "MOSS-Image: reference images are attached while [Moss-Image] Reference Method is 'Disable'; "
                    "switch it to 'Moss-Image-Edit' (for chinoll/Edit-HQ LoRAs) or 'ai-toolkit' (for vision+latent Edit LoRAs)."
                ),
                patch={"__options__": {"moss_do_reference": "Moss-Image-Edit"}},
                action_label="Enable Moss-Image-Edit",
            )
        )

    combined = " ".join([ckpt_name, *lora_names]).strip()
    if combined and _MOSS_FEW_STEP_RE.search(combined):
        is_dmd2_or_8step = bool(_MOSS_DMD2_8STEP_RE.search(combined) or _MOSS_4STEP_RE.search(combined))
        step_threshold = 20 if is_dmd2_or_8step else 30
        cfg_threshold = 1.5 if is_dmd2_or_8step else 2.0
        if (isinstance(steps, (int, float)) and steps >= step_threshold) or (isinstance(cfg, (int, float)) and cfg > cfg_threshold):
            target_steps = _moss_recommended_steps(combined)
            variant = "MOSS-Image / Kromoss Turbo/DMD2" if _MOSS_KROMOSS_RE.search(combined) or "dmd2" in combined.lower() else "MOSS-Image Turbo/DMD"
            items.append(
                AdviceItem(
                    code="moss.dmd_turbo_mismatch",
                    severity="warning",
                    message=(
                        Sentence("{{variant}} detected with steps={{steps}}, cfg={{cfg}}: distilled/Turbo variants typically work best at 4-16 steps and CFG 1.0 (with SLG automatically disabled at CFG 1).", variant=variant, steps=steps, cfg=cfg)
                    ),
                    patch={"sampler.steps": {"value": target_steps}, "cfg_scale": {"value": 1.0}},
                    action_label=f"Set {target_steps} steps · CFG 1.0",
                )
            )
    if combined and _MOSS_EDIT_RE.search(combined) and ref_mode == "Disable" and not has_refs:
        items.append(
            AdviceItem(
                code="moss.edit_lora_ref_disabled",
                severity="info",
                message=(
                    "MOSS-Image Edit LoRA detected while [Moss-Image] Reference Method is 'Disable'; "
                    "enable 'Moss-Image-Edit' or 'ai-toolkit' in Settings when supplying reference images."
                ),
                patch={"__options__": {"moss_do_reference": "Moss-Image-Edit"}},
                action_label="Enable Moss-Image-Edit",
            )
        )
    return items


def moss_structured_advice(p, *, opts=None) -> list[AdviceItem]:
    """Structured hints for common MOSS-Image / Kromoss DMD2 configuration pitfalls."""

    from backend.args import dynamic_args

    if not dynamic_args.moss:
        return []

    active_opts = _resolve_opts(opts)
    sd_model = getattr(p, "sd_model", None)
    has_refs = bool(getattr(sd_model, "ref_latents", None)) or getattr(sd_model, "ini_latent", None) is not None
    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    return _evaluate_moss_advice(
        ckpt_name=ckpt_name,
        lora_names=lora_names,
        steps=getattr(p, "steps", None),
        cfg=getattr(p, "cfg_scale", None),
        has_refs=has_refs,
        opts=active_opts,
    )


def moss_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking hints for common MOSS-Image configuration pitfalls."""

    return [item.message for item in moss_structured_advice(p, opts=opts)]


def _evaluate_krea2_advice(
    *,
    ckpt_name: str,
    lora_names: list[str],
    steps: int | float | None,
    cfg: int | float | None,
    has_refs: bool,
    opts: Any,
) -> list[AdviceItem]:
    items: list[AdviceItem] = []
    do_ref = bool(getattr(opts, "krea2_do_reference", False)) if opts is not None else False
    if has_refs and not do_ref:
        items.append(
            AdviceItem(
                code="krea2.ref_disabled",
                severity="warning",
                message=(
                    "Krea 2: reference images are attached while [Krea2] Enable Reference is off; "
                    "enable it in Settings when using a Krea 2 Edit LoRA."
                ),
                patch={"__options__": {"krea2_do_reference": True}},
                action_label="Enable Krea 2 Reference",
            )
        )

    combined = " ".join([ckpt_name, *lora_names]).strip()
    is_turbo = bool(combined and _KREA2_TURBO_RE.search(combined))
    is_raw_only = bool(ckpt_name and _KREA2_RAW_RE.search(ckpt_name)) and not is_turbo

    if is_turbo:
        if (isinstance(steps, (int, float)) and steps >= 20) or (isinstance(cfg, (int, float)) and cfg > 1.5):
            items.append(
                AdviceItem(
                    code="krea2.turbo_mismatch",
                    severity="warning",
                    message=(
                        Sentence("Krea-2-Turbo detected with steps={{steps}}, cfg={{cfg}}: Turbo is step-distilled and typically works best at ~8 steps with CFG 1.0.", steps=steps, cfg=cfg)
                    ),
                    patch={"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}},
                    action_label="Set Turbo (8 steps · CFG 1.0)",
                )
            )
    elif is_raw_only:
        if (isinstance(steps, (int, float)) and steps <= 12) or (isinstance(cfg, (int, float)) and cfg <= 1.0):
            items.append(
                AdviceItem(
                    code="krea2.raw_low_steps",
                    severity="warning",
                    message=(
                        Sentence("Krea-2-Raw detected with steps={{steps}}, cfg={{cfg}}: the undistilled Raw base typically needs ~28-52 steps and CFG 3.5-4.5 (or a Krea-2 Turbo LoRA for 8-step CFG 1.0).", steps=steps, cfg=cfg)
                    ),
                    patch={"sampler.steps": {"value": 32}, "cfg_scale": {"value": 4.0}},
                    action_label="Set Raw (32 steps · CFG 4.0)",
                )
            )
    if combined and _KREA2_EDIT_RE.search(combined) and not do_ref and not has_refs:
        items.append(
            AdviceItem(
                code="krea2.edit_lora_ref_disabled",
                severity="info",
                message=(
                    "Krea 2 Edit LoRA detected while [Krea2] Enable Reference is off; "
                    "enable [Krea2] Enable Reference in Settings when supplying reference images."
                ),
                patch={"__options__": {"krea2_do_reference": True}},
                action_label="Enable Krea 2 Reference",
            )
        )
    return items


def krea2_structured_advice(p, *, opts=None) -> list[AdviceItem]:
    """Structured hints for common Krea 2 Raw / Turbo / Edit configuration pitfalls."""

    from backend.args import dynamic_args

    if not dynamic_args.krea2:
        return []

    active_opts = _resolve_opts(opts)
    sd_model = getattr(p, "sd_model", None)
    has_refs = bool(getattr(sd_model, "ref_latents", None)) or getattr(sd_model, "ini_latent", None) is not None
    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    return _evaluate_krea2_advice(
        ckpt_name=ckpt_name,
        lora_names=lora_names,
        steps=getattr(p, "steps", None),
        cfg=getattr(p, "cfg_scale", None),
        has_refs=has_refs,
        opts=active_opts,
    )


def krea2_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking hints for common Krea 2 Raw / Turbo / Edit configuration pitfalls."""

    return [item.message for item in krea2_structured_advice(p, opts=opts)]


FLUX3_MAX_REFERENCES = 10


def _evaluate_flux3_advice(
    *,
    distilled: bool | None,
    prompt: str | None,
    steps: int | float | None,
    cfg: int | float | None,
    ref_count: int,
    opts: Any,
    require_non_empty_prompt_for_short: bool = False,
) -> list[AdviceItem]:
    items: list[AdviceItem] = []
    if distilled is True:
        if isinstance(cfg, (int, float)) and cfg > 1.0:
            items.append(
                AdviceItem(
                    code="flux3.distilled_cfg",
                    severity="warning",
                    message=(
                        Sentence("FLUX.3 Klein (step-distilled) with cfg={{cfg}}: the distilled flow-map model samples without classifier-free guidance; use CFG 1.0 (the negative prompt is then skipped) and about 16 steps.", cfg=cfg)
                    ),
                    patch={"cfg_scale": {"value": 1.0}, "sampler.steps": {"value": 16}},
                    action_label="Set Distilled (16 steps · CFG 1.0)",
                )
            )
    elif distilled is False:
        if (isinstance(steps, (int, float)) and steps < 28) or (isinstance(cfg, (int, float)) and cfg <= 1.0):
            items.append(
                AdviceItem(
                    code="flux3.base_low_steps",
                    severity="warning",
                    message=(
                        Sentence("FLUX.3 Klein Base with steps={{steps}}, cfg={{cfg}}: the undistilled base typically needs about 50 steps with CFG around 3.5 (use the distilled Klein checkpoint for 16-step CFG 1 sampling).", steps=steps, cfg=cfg)
                    ),
                    patch={"sampler.steps": {"value": 50}, "cfg_scale": {"value": 3.5}},
                    action_label="Set Base (50 steps · CFG 3.5)",
                )
            )

    if ref_count > 0 and not bool(getattr(opts, "flux3_do_reference", True) if opts is not None else True):
        items.append(
            AdviceItem(
                code="flux3.ref_disabled",
                severity="warning",
                message="FLUX.3: reference images are attached while [FLUX.3] Enable Reference is off; they are ignored (plain img2img).",
                patch={"__options__": {"flux3_do_reference": True}},
                action_label="Enable FLUX.3 Reference",
            )
        )
    if ref_count > FLUX3_MAX_REFERENCES:
        items.append(
            AdviceItem(
                code="flux3.too_many_refs",
                severity="warning",
                message=Sentence("FLUX.3 was trained with at most {{max}} reference images; {{count}} are attached.", max=FLUX3_MAX_REFERENCES, count=ref_count),
                patch=None,
            )
        )

    if isinstance(prompt, str) and (not require_non_empty_prompt_for_short or prompt.strip()):
        if len(re.findall(r"[\w']+", prompt)) < 8 and "[" not in prompt:
            items.append(
                AdviceItem(
                    code="flux3.short_prompt",
                    severity="info",
                    message=(
                        "FLUX.3 is trained on long, detailed captions (about 100-350 words, as written by its prompt upsampler); a short "
                        "prompt leaves the scene under-specified. Describe medium, scene, subjects and layout, or place elements with the "
                        "FLUX.3 Layout panel."
                    ),
                    patch=None,
                )
            )
    return items


def flux3_structured_advice(p, *, opts=None) -> list[AdviceItem]:
    """Structured hints for FLUX.3 Image Klein (Base vs step-distilled, references, captions)."""

    from backend.args import dynamic_args

    if not dynamic_args.flux3:
        return []

    active_opts = _resolve_opts(opts)
    sd_model = getattr(p, "sd_model", None)
    distilled = bool(getattr(sd_model, "is_flux3_distilled", False))
    refs = [*(getattr(sd_model, "ref_latents", None) or ())]
    if getattr(sd_model, "ini_latent", None) is not None:
        refs.append(sd_model.ini_latent)
    return _evaluate_flux3_advice(
        distilled=distilled,
        prompt=getattr(p, "prompt", None),
        steps=getattr(p, "steps", None),
        cfg=getattr(p, "cfg_scale", None),
        ref_count=len(refs),
        opts=active_opts,
        require_non_empty_prompt_for_short=False,
    )


def flux3_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking hints for FLUX.3 Image Klein (Base vs step-distilled, references, captions)."""

    return [item.message for item in flux3_structured_advice(p, opts=opts)]


QWEN21_MAX_REFERENCES = 10
_QWEN21_TURBO_RE = re.compile(r"(?:^|[-_ /])(?:turbo\d*|distill(?:ed)?|8[-_]?steps?)(?:$|[-_ /.])", re.IGNORECASE)
_QWEN21_RGBA_PROMPT_RE = _RGBA_PROMPT_RE


def _evaluate_qwen21_advice(
    *,
    is_turbo: bool | None,
    prompt: str | None,
    steps: int | float | None,
    cfg: int | float | None,
    ref_count: int,
    opts: Any,
) -> list[AdviceItem]:
    items: list[AdviceItem] = []
    if is_turbo is True:
        if (isinstance(steps, (int, float)) and steps >= 16) or (isinstance(cfg, (int, float)) and cfg > 1.5):
            items.append(
                AdviceItem(
                    code="qwen21.turbo_mismatch",
                    severity="warning",
                    message=(
                        Sentence("Qwen-Image-2.1 Turbo detected with steps={{steps}}, cfg={{cfg}}: the 8-step distilled Turbo checkpoint/LoRA works best at ~8 steps with CFG 1.0.", steps=steps, cfg=cfg)
                    ),
                    patch={"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}},
                    action_label="Set Turbo (8 steps · CFG 1.0)",
                )
            )
    elif is_turbo is False:
        if (isinstance(steps, (int, float)) and steps <= 12) and (isinstance(cfg, (int, float)) and cfg <= 1.0):
            items.append(
                AdviceItem(
                    code="qwen21.base_low_steps",
                    severity="warning",
                    message=(
                        Sentence("Qwen-Image-2.1 Base detected with steps={{steps}}, cfg={{cfg}}: the undistilled Base model typically needs ~32-40 steps and CFG 2.5-4.0 (use Qwen-Image-2.1-Turbo or a Turbo LoRA for 8-step CFG 1.0).", steps=steps, cfg=cfg)
                    ),
                    patch={"sampler.steps": {"value": 32}, "cfg_scale": {"value": 2.5}},
                    action_label="Set Base (32 steps · CFG 2.5)",
                )
            )

    if ref_count > 0 and not bool(getattr(opts, "qwen21_do_reference", True) if opts is not None else True):
        items.append(
            AdviceItem(
                code="qwen21.ref_disabled",
                severity="warning",
                message="Qwen-Image-2.1: reference images are attached while [Qwen 2.1] Enable Reference is off; they are ignored (plain img2img).",
                patch={"__options__": {"qwen21_do_reference": True}},
                action_label="Enable Qwen 2.1 Reference",
            )
        )
    if ref_count > QWEN21_MAX_REFERENCES:
        items.append(
            AdviceItem(
                code="qwen21.too_many_refs",
                severity="warning",
                message=Sentence("Qwen-Image-2.1 was trained with at most {{max}} reference images; {{count}} are attached.", max=QWEN21_MAX_REFERENCES, count=ref_count),
                patch=None,
            )
        )

    samples_format = str(getattr(opts, "samples_format", "png") or "png").lower() if opts is not None else "png"
    if isinstance(prompt, str) and _QWEN21_RGBA_PROMPT_RE.search(prompt) and samples_format in ("jpg", "jpeg"):
        items.append(
            AdviceItem(
                code="qwen21.rgba_jpeg_format",
                severity="warning",
                message=(
                    Sentence("Qwen-Image-2.1 RGBA transparent prompt detected while samples_format={{format}}: JPEG drops the alpha channel; switch the output format to PNG or WebP to keep transparency.", format=repr(samples_format))
                ),
                patch={"__options__": {"samples_format": "png"}},
                action_label="Switch Output to PNG",
            )
        )
    return items


def qwen21_structured_advice(p, *, opts=None) -> list[AdviceItem]:
    """Structured hints for Qwen-Image-2.1 (Base vs Turbo, references, RGBA output)."""

    from backend.args import dynamic_args

    if not dynamic_args.qwen21:
        return []

    active_opts = _resolve_opts(opts)
    sd_model = getattr(p, "sd_model", None)
    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    combined = " ".join([ckpt_name, *lora_names]).strip()
    is_turbo = bool(combined and _QWEN21_TURBO_RE.search(combined)) or not getattr(sd_model, "use_dynamic_shifting", True)

    refs = [*(getattr(sd_model, "ref_latents", None) or ())]
    if getattr(sd_model, "ini_latent", None) is not None:
        refs.append(sd_model.ini_latent)

    return _evaluate_qwen21_advice(
        is_turbo=is_turbo,
        prompt=getattr(p, "prompt", None),
        steps=getattr(p, "steps", None),
        cfg=getattr(p, "cfg_scale", None),
        ref_count=len(refs),
        opts=active_opts,
    )


def qwen21_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking hints for Qwen-Image-2.1 (Base vs Turbo, references, RGBA output)."""

    return [item.message for item in qwen21_structured_advice(p, opts=opts)]


def _call_advice_fn(fn: Callable[..., Any], *args: Any, opts: Any = None) -> list[Any]:
    if opts is not None and "opts" in inspect.signature(fn).parameters:
        return list(fn(*args, opts=opts))
    return list(fn(*args))


def collect_structured_advice(p, *, opts=None) -> list[AdviceItem]:
    """Collect structured ``AdviceItem`` hints across all registered DiT capabilities.

    Capabilities providing ``structured_advice`` yield rich items with patches;
    capabilities providing only legacy ``workflow_advice`` are wrapped cleanly.
    """

    items: list[AdviceItem] = []
    for cap in _REGISTRY.values():
        if cap.structured_advice is not None:
            items.extend(_call_advice_fn(cap.structured_advice, p, opts=opts))
        elif cap.workflow_advice is not None:
            for msg in _call_advice_fn(cap.workflow_advice, p, opts=opts):
                items.append(AdviceItem(code=f"{cap.name}.advice", severity="warning", message=msg if isinstance(msg, str) else str(msg)))
    return items


def collect_workflow_advice(p, *, opts=None) -> list[str]:
    """Collect non-blocking workflow hints across all registered DiT capabilities."""

    notes: list[str] = []
    for cap in _REGISTRY.values():
        fn = cap.workflow_advice
        if fn is None:
            continue
        notes.extend(_call_advice_fn(fn, p, opts=opts))
    return notes


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _extract_form_state(
    preset: str,
    tab: str,
    values: Mapping[str, Any] | None,
    opts: Any,
    *,
    preset_aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    raw = values if isinstance(values, Mapping) else {}
    core = raw.get("core") if isinstance(raw.get("core"), Mapping) else raw
    sampler = raw.get("sampler") if isinstance(raw.get("sampler"), Mapping) else {}

    steps = _to_number(sampler.get("steps") if "steps" in sampler else core.get("steps"))
    cfg = _to_number(core.get("cfg_scale"))
    prompt = str(core.get("prompt") or "")
    hr_prompt = str(core.get("hr_prompt") or "")
    prompt_texts = [t for t in (prompt, hr_prompt) if t]
    lora_names = _active_lora_names_from_prompts(prompt_texts)

    stitch_refs = 0
    for owner_key, owner_vals in raw.items():
        if not isinstance(owner_vals, Mapping):
            continue
        if owner_key in ("imagestitch_integrated", "image_stitch", "imagestitch") or (
            "references" in owner_vals and "enable" in owner_vals
        ):
            if owner_vals.get("enable"):
                refs_val = owner_vals.get("references")
                if isinstance(refs_val, (list, tuple)):
                    stitch_refs += len([r for r in refs_val if r])

    has_init_img = tab == "img2img" and bool(core.get("init_img"))
    ref_count = stitch_refs + (1 if has_init_img else 0)

    ckpt_name = ""
    if opts is not None:
        for key in (preset, *preset_aliases):
            if key and (found := getattr(opts, f"forge_checkpoint_{key}", None)):
                ckpt_name = str(found)
                break
        if not ckpt_name:
            ckpt_name = str(getattr(opts, "sd_model_checkpoint", None) or "")
    if not ckpt_name and isinstance(core.get("checkpoint"), str):
        ckpt_name = core["checkpoint"]

    return {
        "core": core,
        "sampler": sampler,
        "steps": int(steps) if steps is not None and steps.is_integer() else steps,
        "cfg": cfg,
        "prompt": prompt,
        "lora_names": lora_names,
        "ref_count": ref_count,
        "stitch_refs": stitch_refs,
        "has_init_img": has_init_img,
        "ckpt_name": ckpt_name,
    }


def derive_model_advice(
    preset: str | None,
    tab: str,
    values: Mapping[str, Any] | None,
    *,
    opts: Any = None,
) -> list[AdviceItem]:
    """Evaluate pre-submit form-level structured advice for ``preset``, ``tab``,
    and form ``values`` (``{"core": {...}, "sampler": {...}, ...}``).

    Used by ``/api/v2/rules/derive``, ``core.__conflicts__``, and
    ``core.__advice__`` to render actionable one-click fix chips before
    generation starts.
    """

    active_opts = _resolve_opts(opts)
    norm_preset = str(preset or "sd").strip().lower() or "sd"
    cap = capability_for_preset(norm_preset)
    canonical_preset = (cap.preset_arch if cap is not None and cap.preset_arch else None) or _DIT_TO_PRESET.get(norm_preset, norm_preset)
    dit_key = cap.name if cap is not None else _PRESET_TO_DIT.get(canonical_preset, canonical_preset)
    aliases = tuple(k for k in (norm_preset, dit_key) if k and k != canonical_preset)
    state = _extract_form_state(canonical_preset, tab, values, active_opts, preset_aliases=aliases)

    if cap is not None and cap.form_advice is not None:
        return list(_call_advice_fn(cap.form_advice, canonical_preset, tab, values, opts=active_opts))

    if dit_key == "anima":
        return _evaluate_anima_advice(
            ckpt_name=state["ckpt_name"],
            lora_names=state["lora_names"],
            prompt=state["prompt"],
            steps=state["steps"],
            cfg=state["cfg"],
            has_refs=state["ref_count"] > 0,
            has_transparent_init=False,
            opts=active_opts,
            vae_is_rgba="rgba" in state["ckpt_name"].lower(),
        )

    if dit_key == "moss":
        return _evaluate_moss_advice(
            ckpt_name=state["ckpt_name"],
            lora_names=state["lora_names"],
            steps=state["steps"],
            cfg=state["cfg"],
            has_refs=state["ref_count"] > 0,
            opts=active_opts,
        )

    if dit_key == "krea2":
        return _evaluate_krea2_advice(
            ckpt_name=state["ckpt_name"],
            lora_names=state["lora_names"],
            steps=state["steps"],
            cfg=state["cfg"],
            has_refs=state["ref_count"] > 0,
            opts=active_opts,
        )

    if dit_key == "flux3":
        lower_ckpt = state["ckpt_name"].lower()
        distilled: bool | None = None
        if lower_ckpt:
            if "base" in lower_ckpt:
                distilled = False
            elif any(tok in lower_ckpt for tok in ("distill", "turbo", "klein")):
                distilled = True
        return _evaluate_flux3_advice(
            distilled=distilled,
            prompt=state["prompt"],
            steps=state["steps"],
            cfg=state["cfg"],
            ref_count=state["ref_count"],
            opts=active_opts,
            require_non_empty_prompt_for_short=True,
        )

    if dit_key == "qwen21":
        combined = " ".join([state["ckpt_name"], *state["lora_names"]]).strip()
        is_turbo: bool | None = None
        if combined:
            if _QWEN21_TURBO_RE.search(combined):
                is_turbo = True
            elif state["ckpt_name"]:
                is_turbo = False
        return _evaluate_qwen21_advice(
            is_turbo=is_turbo,
            prompt=state["prompt"],
            steps=state["steps"],
            cfg=state["cfg"],
            ref_count=state["ref_count"],
            opts=active_opts,
        )

    return []


# endregion


# region Built-in DiT registrations (lazy imports keep startup & standalone tests fast)

_ANIMA_INT8_LINEAR_SUFFIXES = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.output_proj.weight",
    "mlp.layer1.weight",
)
_MOSS_INT8_LINEAR_SUFFIXES = (
    ".proj.q.weight",
    ".proj.k.weight",
    ".proj.v.weight",
    ".proj.qkv.weight",
    ".proj.o.weight",
    ".proj.ffn_up_gate.weight",
    ".proj.ffn_down.weight",
)
_KREA2_INT8_LINEAR_SUFFIXES = (
    ".attn.wq.weight",
    ".attn.wk.weight",
    ".attn.wv.weight",
    ".attn.gate.weight",
    ".attn.wo.weight",
    ".mlp.gate.weight",
    ".mlp.up.weight",
    ".mlp.down.weight",
)


def _is_anima_state_dict(sd: dict[str, torch.Tensor]) -> bool:
    return "x_embedder.proj.1.weight" in sd or any(
        k.startswith("blocks.") and (".self_attn." in k or ".mlp.layer1." in k or ".adaln_modulation_" in k)
        for k in sd
    )


def _is_moss_state_dict(sd: dict[str, torch.Tensor]) -> bool:
    return "shared_block_modulation.weight" in sd or any(k.startswith("layers.") and ".proj." in k for k in sd)


def _is_krea2_state_dict(sd: dict[str, torch.Tensor]) -> bool:
    return "txtfusion.projector.weight" in sd or "first.weight" in sd


def _is_anima_dir(path: os.PathLike | str) -> bool:
    from backend.nn.anima_diffusers import is_anima_diffusers_dir

    return is_anima_diffusers_dir(path)


def _anima_primary_file(path: os.PathLike | str) -> str | None:
    from backend.nn.anima_diffusers import anima_diffusers_primary_file

    return anima_diffusers_primary_file(path)


def _load_anima_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    from backend.nn.anima_diffusers import load_anima_diffusers_dir

    return load_anima_diffusers_dir(path)


def _is_anima_single_file(sd: dict[str, torch.Tensor]) -> bool:
    from backend.nn.anima_diffusers import is_diffusers_anima_transformer

    return is_diffusers_anima_transformer(sd)


def _convert_anima_single_file(sd: dict[str, torch.Tensor], output_prefix: str) -> dict[str, torch.Tensor]:
    from backend.nn.anima_diffusers import convert_diffusers_anima_transformer

    return convert_diffusers_anima_transformer(sd, output_prefix=output_prefix)


def _is_anima_engine(model) -> bool:
    from modules.anima_support import is_anima_engine

    return is_anima_engine(model)


def _is_moss_dir(path: os.PathLike | str) -> bool:
    from backend.nn.moss_diffusers import is_moss_diffusers_dir, is_sharded_moss_transformer_dir

    return is_moss_diffusers_dir(path) or is_sharded_moss_transformer_dir(path)


def _moss_primary_file(path: os.PathLike | str) -> str | None:
    from backend.nn.moss_diffusers import moss_diffusers_primary_file

    return moss_diffusers_primary_file(path)


def _load_moss_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    from backend.nn.moss_diffusers import load_moss_diffusers_dir

    return load_moss_diffusers_dir(path)


def _is_krea2_dir(path: os.PathLike | str) -> bool:
    from backend.nn.krea_diffusers import is_krea2_diffusers_dir, is_sharded_krea2_transformer_dir

    return is_krea2_diffusers_dir(path) or is_sharded_krea2_transformer_dir(path)


def _krea2_primary_file(path: os.PathLike | str) -> str | None:
    from backend.nn.krea_diffusers import krea2_diffusers_primary_file

    return krea2_diffusers_primary_file(path)


def _load_krea2_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    from backend.nn.krea_diffusers import load_krea2_diffusers_dir

    return load_krea2_diffusers_dir(path)


def _is_krea2_single_file(sd: dict[str, torch.Tensor]) -> bool:
    from backend.nn.krea_diffusers import is_diffusers_krea2_transformer

    return is_diffusers_krea2_transformer(sd)


def _convert_krea2_single_file(sd: dict[str, torch.Tensor], output_prefix: str) -> dict[str, torch.Tensor]:
    from backend.nn.krea_diffusers import convert_diffusers_krea2_transformer

    return convert_diffusers_krea2_transformer(sd, output_prefix=output_prefix)


register_dit_capability(
    DiTCapability(
        name="anima",
        dit_cls_name="CosmosTransformer3DModel",
        model_type_name="Anima",
        engine_label="Anima",
        sensitive_prefixes=(
            "x_embedder.",
            "t_embedder.",
            "t_embedding_norm.",
            "final_layer.",
            "blocks.0.",
            "blocks.1.adaln_modulation_",
            "llm_adapter.",
            "anima_v2_connector.",
        ),
        sensitive_module_types=("RMSNorm",),
        matches_state_dict=_is_anima_state_dict,
        block_linear_filter=lambda k: k.startswith("blocks.") and k.endswith(".weight"),
        fp8_rowwise=False,
        fp8_full_precision_matmul_filter=lambda k: k.endswith("mlp.layer2.weight") or ".adaln_modulation_" in k,
        int8_convrot_filter=lambda k: k.endswith(_ANIMA_INT8_LINEAR_SUFFIXES),
        diffusers_arch="anima/diffusers",
        is_diffusers_dir=_is_anima_dir,
        diffusers_primary_file=_anima_primary_file,
        load_diffusers_dir=_load_anima_dir,
        is_diffusers_single_file=_is_anima_single_file,
        convert_diffusers_single_file=_convert_anima_single_file,
        block_attr="blocks",
        compile_preset="Anima per-block",
        matches_engine=_is_anima_engine,
        non_matching_engine_reason="Anima per-block ignored: non-Anima model",
        spectrum_required_attrs=("final_layer", "t_embedder", "t_embedding_norm", "unpatchify"),
        spectrum_capture_attr="final_layer",
        spectrum_fast_forward=fast_forward_anima,
        capabilities=frozenset({"anima_hooks", "multi_reference", "rgba_output", "control_blocks", "per_block_compile", "spectrum"}),
        preset_arch="anima",
        max_references=4,
        reference_slot_format=None,
        supports_rgba=True,
        supports_control_blocks=True,
        attention_backends=("sdpa", "flash", "sage", "masked_sdpa"),
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
        structured_advice=anima_structured_advice,
    )
)

register_dit_capability(
    DiTCapability(
        name="moss",
        dit_cls_name="MOSSImageTransformer",
        model_type_name="MOSSImage",
        engine_label="MOSS-Image",
        sensitive_prefixes=(
            "video_embedder.",
            "text_embedder.",
            "time_embedding.",
            "shared_block_modulation.",
            "final_norm_video.",
            "final_linear_video.",
            "video_head_modulation.",
        ),
        sensitive_module_types=("RMSNorm", "SimpleModulation", "DoubleSharedModulation"),
        matches_state_dict=_is_moss_state_dict,
        block_linear_filter=lambda k: k.startswith("layers.") and ".proj." in k and k.endswith(".weight"),
        fp8_rowwise=True,
        int8_convrot_filter=lambda k: k.endswith(_MOSS_INT8_LINEAR_SUFFIXES),
        diffusers_arch="moss-image/diffusers",
        is_diffusers_dir=_is_moss_dir,
        diffusers_primary_file=_moss_primary_file,
        load_diffusers_dir=_load_moss_dir,
        block_attr="layers",
        compile_preset="MOSS-Image per-block",
        matches_engine=lambda model: bool(getattr(model, "is_moss_engine", False)),
        non_matching_engine_reason="MOSS-Image per-block ignored: non-MOSS-Image model",
        spectrum_required_attrs=("video_head_input", "final_norm_video", "final_linear_video", "video_head_modulation", "time_embedding", "compute_video_head"),
        spectrum_capture_attr="video_head_input",
        spectrum_fast_forward=fast_forward_moss,
        workflow_advice=moss_workflow_advice,
        capabilities=frozenset({"multi_reference", "structured_layout", "distilled_cfg1", "per_block_compile", "spectrum"}),
        preset_arch="moss",
        max_references=4,
        reference_slot_format=None,
        supports_rgba=False,
        supports_control_blocks=False,
        attention_backends=("sdpa", "flash", "sage"),
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
        structured_advice=moss_structured_advice,
    )
)

register_dit_capability(
    DiTCapability(
        name="krea2",
        dit_cls_name="Krea2Transformer2DModel",
        model_type_name="SingleStreamDiT",
        engine_label="Krea 2",
        sensitive_prefixes=(
            "first.",
            "tmlp.",
            "tproj.",
            "txtfusion.projector.",
            "txtmlp.",
            "last.",
        ),
        sensitive_module_types=("RMSNorm", "SimpleModulation", "DoubleSharedModulation"),
        matches_state_dict=_is_krea2_state_dict,
        block_linear_filter=lambda k: (k.startswith("blocks.") or k.startswith(("txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks."))) and k.endswith(".weight"),
        fp8_rowwise=True,
        int8_convrot_filter=lambda k: k.startswith("blocks.") and k.endswith(_KREA2_INT8_LINEAR_SUFFIXES),
        diffusers_arch="krea-2/diffusers",
        is_diffusers_dir=_is_krea2_dir,
        diffusers_primary_file=_krea2_primary_file,
        load_diffusers_dir=_load_krea2_dir,
        is_diffusers_single_file=_is_krea2_single_file,
        convert_diffusers_single_file=_convert_krea2_single_file,
        block_attr="blocks",
        compile_preset="Krea-2 per-block",
        matches_engine=lambda model: bool(getattr(model, "is_krea2_engine", False)),
        non_matching_engine_reason="Krea-2 per-block ignored: non-Krea-2 model",
        spectrum_required_attrs=("last_input", "last", "tmlp", "compute_last"),
        spectrum_capture_attr="last_input",
        spectrum_fast_forward=fast_forward_krea2,
        workflow_advice=krea2_workflow_advice,
        capabilities=frozenset({"multi_reference", "distilled_cfg1", "per_block_compile", "spectrum"}),
        preset_arch="krea",
        max_references=4,
        reference_slot_format=None,
        supports_rgba=False,
        supports_control_blocks=False,
        attention_backends=("sdpa", "flash", "sage"),
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
        structured_advice=krea2_structured_advice,
    )
)


_FLUX3_BLOCK_PREFIXES = ("content_mode_blocks.", "txt_mode_blocks.", "single_blocks.")
_FLUX3_INT8_LINEAR_SUFFIXES = (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".qkv.weight", ".attn_out.weight", ".mlp_in.weight", ".mlp_out.weight")


def _is_flux3_state_dict(sd: dict[str, torch.Tensor]) -> bool:
    return "single_stream_modulations.txt.lin.weight" in sd or "content_mode_blocks.image.0.attn_out.weight" in sd


def _is_flux3_dir(path: os.PathLike | str) -> bool:
    from backend.nn.flux3_diffusers import is_flux3_checkpoint_dir

    return is_flux3_checkpoint_dir(path)


def _flux3_primary_file(path: os.PathLike | str) -> str | None:
    from backend.nn.flux3_diffusers import flux3_diffusers_primary_file

    return flux3_diffusers_primary_file(path)


def _load_flux3_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    from backend.nn.flux3_diffusers import load_flux3_diffusers_dir

    return load_flux3_diffusers_dir(path)


def _is_flux3_single_file(sd: dict[str, torch.Tensor]) -> bool:
    from backend.nn.flux3_diffusers import is_diffusers_flux3_transformer

    return is_diffusers_flux3_transformer(sd)


def _convert_flux3_single_file(sd: dict[str, torch.Tensor], output_prefix: str) -> dict[str, torch.Tensor]:
    from backend.nn.flux3_diffusers import convert_diffusers_flux3_transformer

    return convert_diffusers_flux3_transformer(sd, output_prefix=output_prefix)


register_dit_capability(
    DiTCapability(
        name="flux3",
        dit_cls_name="Flux3Transformer2DModel",
        model_type_name="Flux3",
        engine_label="FLUX.3",
        # embedders, time MLPs, every per-mode / joint modulation head and the output layer
        sensitive_prefixes=("emb_in.", "txt_in.", "time_in.", "ref_time_in.", "early_stream_modulations.", "single_stream_modulations.", "final_layer."),
        sensitive_module_types=("RMSNorm",),
        matches_state_dict=_is_flux3_state_dict,
        block_linear_filter=lambda k: k.startswith(_FLUX3_BLOCK_PREFIXES) and k.endswith(".weight"),
        fp8_rowwise=True,
        int8_convrot_filter=lambda k: k.startswith(_FLUX3_BLOCK_PREFIXES) and k.endswith(_FLUX3_INT8_LINEAR_SUFFIXES),
        diffusers_arch="flux-3/diffusers",
        is_diffusers_dir=_is_flux3_dir,
        diffusers_primary_file=_flux3_primary_file,
        load_diffusers_dir=_load_flux3_dir,
        is_diffusers_single_file=_is_flux3_single_file,
        convert_diffusers_single_file=_convert_flux3_single_file,
        block_attr="single_blocks",
        compile_preset="FLUX.3 per-block",
        matches_engine=lambda model: bool(getattr(model, "is_flux3_engine", False)),
        non_matching_engine_reason="FLUX.3 per-block ignored: non-FLUX.3 model",
        spectrum_required_attrs=("final_input", "final_layer", "compute_final"),
        spectrum_capture_attr="final_input",
        spectrum_fast_forward=fast_forward_flux3,
        workflow_advice=flux3_workflow_advice,
        capabilities=frozenset({"multi_reference", "structured_layout", "distilled_cfg1", "per_block_compile", "spectrum"}),
        preset_arch="flux3",
        max_references=FLUX3_MAX_REFERENCES,
        reference_slot_format="ref_image_{i}",
        supports_rgba=False,
        supports_control_blocks=False,
        attention_backends=("sdpa", "flash", "sage"),
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
        structured_advice=flux3_structured_advice,
    )
)


_QWEN21_INT8_LINEAR_SUFFIXES = (
    ".attn.to_q.weight",
    ".attn.to_k.weight",
    ".attn.to_v.weight",
    ".attn.to_out.0.weight",
    ".img_mlp.gate_up.weight",
    ".img_mlp.gate_layer.weight",
    ".img_mlp.proj.weight",
    ".img_mlp.out.weight",
)


def _is_qwen21_state_dict(sd: dict[str, torch.Tensor]) -> bool:
    return "txt_in.text_norm.weight" in sd and "modulation.1.weight" in sd


def _is_qwen21_dir(path: os.PathLike | str) -> bool:
    from backend.nn.qwen21_diffusers import is_qwen21_checkpoint_dir

    return is_qwen21_checkpoint_dir(path)


def _qwen21_primary_file(path: os.PathLike | str) -> str | None:
    from backend.nn.qwen21_diffusers import qwen21_diffusers_primary_file

    return qwen21_diffusers_primary_file(path)


def _load_qwen21_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    from backend.nn.qwen21_diffusers import load_qwen21_diffusers_dir

    return load_qwen21_diffusers_dir(path)


register_dit_capability(
    DiTCapability(
        name="qwen21",
        dit_cls_name="QwenImage21Transformer2DModel",
        model_type_name="QwenImage21Transformer2DModel",
        engine_label="Qwen-Image-2.1",
        sensitive_prefixes=(
            "img_in.",
            "txt_in.",
            "time_text_embed.",
            "modulation.",
            "norm_out.",
            "proj_out.",
        ),
        sensitive_module_types=("RMSNorm", "ZeroCenteredRMSNorm"),
        matches_state_dict=_is_qwen21_state_dict,
        block_linear_filter=lambda k: k.startswith("transformer_blocks.") and k.endswith(".weight"),
        fp8_rowwise=True,
        int8_convrot_filter=lambda k: k.startswith("transformer_blocks.") and k.endswith(_QWEN21_INT8_LINEAR_SUFFIXES),
        diffusers_arch="qwen-image-2.1/diffusers",
        is_diffusers_dir=_is_qwen21_dir,
        diffusers_primary_file=_qwen21_primary_file,
        load_diffusers_dir=_load_qwen21_dir,
        block_attr="transformer_blocks",
        compile_preset="Qwen-2.1 per-block",
        matches_engine=lambda model: bool(getattr(model, "is_qwen21_engine", False)),
        non_matching_engine_reason="Qwen-2.1 per-block ignored: non-Qwen-Image-2.1 model",
        spectrum_required_attrs=("final_input", "norm_out", "proj_out", "time_text_embed", "compute_final"),
        spectrum_capture_attr="final_input",
        spectrum_fast_forward=fast_forward_qwen21,
        workflow_advice=qwen21_workflow_advice,
        capabilities=frozenset({"multi_reference", "rgba_output", "control_blocks", "distilled_cfg1", "prefix_kv_cache", "per_block_compile", "spectrum"}),
        preset_arch="qwen21",
        max_references=QWEN21_MAX_REFERENCES,
        reference_slot_format="<image_{i}>",
        supports_rgba=True,
        supports_control_blocks=True,
        attention_backends=("sdpa", "flash", "sage"),
        quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
        structured_advice=qwen21_structured_advice,
    )
)

# endregion
