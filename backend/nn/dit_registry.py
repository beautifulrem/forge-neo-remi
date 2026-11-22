"""Declarative capability registry for DiT architectures (Anima, MOSS-Image,
Krea 2, and future models such as Qwen-Image-2.1 / FLUX.3).

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
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


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

_MOSS_FEW_STEP_RE = re.compile(r"(?:^|[-_ /])(?:turbo|dmd|4[-_]?steps?|8[-_]?steps?|16[-_]?step)(?:$|[-_ /.])", re.IGNORECASE)
_MOSS_EDIT_RE = re.compile(r"(?:^|[-_ /])(?:moss[-_ ]?edit|moss[-_ ]?image[-_ ]?edit)(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_TURBO_RE = re.compile(r"(?:^|[-_ /])turbo(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_RAW_RE = re.compile(r"(?:^|[-_ /])raw(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_EDIT_RE = re.compile(r"(?:^|[-_ /])(?:krea[-_ ]?2[-_ ]?edit|krea2edit)(?:$|[-_ /.])", re.IGNORECASE)


def _prompt_and_checkpoint_names(p) -> tuple[str, list[str]]:
    sd_model = getattr(p, "sd_model", None)
    ckpt_name = str(getattr(getattr(sd_model, "sd_checkpoint_info", None), "name", "") or "")
    lora_names: list[str] = []
    for prompt_text in (getattr(p, "prompt", None), *(getattr(p, "all_prompts", None) or ())):
        if isinstance(prompt_text, str):
            lora_names.extend(re.findall(r"<lora:([^:>]+)", prompt_text))
    return ckpt_name, lora_names


def _resolve_opts(opts):
    if opts is not None:
        return opts
    from modules import shared

    return shared.opts


def moss_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking hints for common MOSS-Image configuration pitfalls."""

    from backend.args import dynamic_args

    if not dynamic_args.moss:
        return []

    active_opts = _resolve_opts(opts)
    advice: list[str] = []
    sd_model = getattr(p, "sd_model", None)
    ref_mode = getattr(active_opts, "moss_do_reference", "Disable")
    has_refs = bool(getattr(sd_model, "ref_latents", None)) or getattr(sd_model, "ini_latent", None) is not None
    if has_refs and ref_mode == "Disable":
        advice.append(
            "MOSS-Image: reference images are attached while [Moss-Image] Reference Method is 'Disable'; "
            "switch it to 'Moss-Image-Edit' (for chinoll/Edit-HQ LoRAs) or 'ai-toolkit' (for vision+latent Edit LoRAs)."
        )

    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    combined = " ".join([ckpt_name, *lora_names]).strip()

    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if combined and _MOSS_FEW_STEP_RE.search(combined):
        if (isinstance(steps, (int, float)) and steps >= 30) or (isinstance(cfg, (int, float)) and cfg > 2.0):
            advice.append(
                f"MOSS-Image Turbo/DMD detected with steps={steps}, cfg={cfg}: distilled/Turbo variants "
                "typically work best at 4-16 steps and CFG 1.0 (with SLG automatically disabled at CFG 1)."
            )
    if combined and _MOSS_EDIT_RE.search(combined) and ref_mode == "Disable" and not has_refs:
        advice.append(
            "MOSS-Image Edit LoRA detected while [Moss-Image] Reference Method is 'Disable'; "
            "enable 'Moss-Image-Edit' or 'ai-toolkit' in Settings when supplying reference images."
        )
    return advice


def krea2_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking hints for common Krea 2 Raw / Turbo / Edit configuration pitfalls."""

    from backend.args import dynamic_args

    if not dynamic_args.krea2:
        return []

    active_opts = _resolve_opts(opts)
    advice: list[str] = []
    sd_model = getattr(p, "sd_model", None)
    do_ref = bool(getattr(active_opts, "krea2_do_reference", False))
    has_refs = bool(getattr(sd_model, "ref_latents", None)) or getattr(sd_model, "ini_latent", None) is not None
    if has_refs and not do_ref:
        advice.append(
            "Krea 2: reference images are attached while [Krea2] Enable Reference is off; "
            "enable it in Settings when using a Krea 2 Edit LoRA."
        )

    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    combined = " ".join([ckpt_name, *lora_names]).strip()
    is_turbo = bool(combined and _KREA2_TURBO_RE.search(combined))
    is_raw_only = bool(ckpt_name and _KREA2_RAW_RE.search(ckpt_name)) and not is_turbo

    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if is_turbo:
        if (isinstance(steps, (int, float)) and steps >= 20) or (isinstance(cfg, (int, float)) and cfg > 1.5):
            advice.append(
                f"Krea-2-Turbo detected with steps={steps}, cfg={cfg}: Turbo is step-distilled and "
                "typically works best at ~8 steps with CFG 1.0."
            )
    elif is_raw_only:
        if (isinstance(steps, (int, float)) and steps <= 12) or (isinstance(cfg, (int, float)) and cfg <= 1.0):
            advice.append(
                f"Krea-2-Raw detected with steps={steps}, cfg={cfg}: the undistilled Raw base typically "
                "needs ~28-52 steps and CFG 3.5-4.5 (or a Krea-2 Turbo LoRA for 8-step CFG 1.0)."
            )
    if combined and _KREA2_EDIT_RE.search(combined) and not do_ref and not has_refs:
        advice.append(
            "Krea 2 Edit LoRA detected while [Krea2] Enable Reference is off; "
            "enable [Krea2] Enable Reference in Settings when supplying reference images."
        )
    return advice


FLUX3_MAX_REFERENCES = 10


def flux3_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking hints for FLUX.3 Image Klein (Base vs step-distilled, references, captions)."""

    from backend.args import dynamic_args

    if not dynamic_args.flux3:
        return []

    active_opts = _resolve_opts(opts)
    advice: list[str] = []
    sd_model = getattr(p, "sd_model", None)
    distilled = bool(getattr(sd_model, "is_flux3_distilled", False))
    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if distilled:
        if isinstance(cfg, (int, float)) and cfg > 1.0:
            advice.append(
                f"FLUX.3 Klein (step-distilled) with cfg={cfg}: the distilled flow-map model samples without classifier-free "
                "guidance; use CFG 1.0 (the negative prompt is then skipped) and about 16 steps."
            )
    elif (isinstance(steps, (int, float)) and steps < 28) or (isinstance(cfg, (int, float)) and cfg <= 1.0):
        advice.append(
            f"FLUX.3 Klein Base with steps={steps}, cfg={cfg}: the undistilled base typically needs about 50 steps with "
            "CFG around 3.5 (use the distilled Klein checkpoint for 16-step CFG 1 sampling)."
        )

    refs = [*(getattr(sd_model, "ref_latents", None) or ())]
    if getattr(sd_model, "ini_latent", None) is not None:
        refs.append(sd_model.ini_latent)
    if refs and not bool(getattr(active_opts, "flux3_do_reference", True)):
        advice.append("FLUX.3: reference images are attached while [FLUX.3] Enable Reference is off; they are ignored (plain img2img).")
    if len(refs) > FLUX3_MAX_REFERENCES:
        advice.append(f"FLUX.3 was trained with at most {FLUX3_MAX_REFERENCES} reference images; {len(refs)} are attached.")

    prompt = getattr(p, "prompt", None)
    if isinstance(prompt, str) and len(re.findall(r"[\w']+", prompt)) < 8 and "[" not in prompt:
        advice.append(
            "FLUX.3 is trained on long, detailed captions (about 100-350 words, as written by its prompt upsampler); a short "
            "prompt leaves the scene under-specified. Describe medium, scene, subjects and layout, or place elements with the "
            "FLUX.3 Layout panel."
        )
    return advice


def collect_workflow_advice(p, *, opts=None) -> list[str]:
    """Collect non-blocking workflow hints across all registered DiT capabilities."""

    notes: list[str] = []
    for cap in _REGISTRY.values():
        fn = cap.workflow_advice
        if fn is None:
            continue
        if opts is not None and "opts" in inspect.signature(fn).parameters:
            notes.extend(fn(p, opts=opts))
        else:
            notes.extend(fn(p))
    return notes


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
    )
)

# endregion
