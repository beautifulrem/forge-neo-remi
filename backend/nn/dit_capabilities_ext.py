"""Extended DiT Capability registrations (Z-Image, ERNIE-Image, Bria Fibo 2.0).

Self-registers into ``backend.nn.dit_registry`` via ``register_dit_capability()``
without modifying ``backend/nn/dit_registry.py``. Dynamically populates any
extended ``ModelCapability`` fields (``capabilities``, ``preset_arch``,
``max_references``, ``reference_slot_format``, ``supports_rgba``,
``supports_control_blocks``, ``attention_backends``, ``quant_formats``,
``structured_advice``, ``form_advice``) when present on ``DiTCapability``.

Hooked lazily by ``backend.nn.model_capabilities.ensure_ext_capabilities_loaded()``
and by each extended engine on initialization.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from backend.nn.dit_registry import DiTCapability, register_dit_capability
from backend.nn.ernie_diffusers import (
    ERNIE_INT8_LINEAR_SUFFIXES,
    ERNIE_SENSITIVE_PREFIXES,
    ernie_diffusers_primary_file,
    ernie_workflow_advice,
    fast_forward_ernie,
    is_ernie_checkpoint_dir,
    is_ernie_state_dict,
    load_ernie_diffusers_dir,
)
from backend.nn.fibo import (
    FIBO_INT8_LINEAR_SUFFIXES,
    FIBO_MAX_REFERENCES,
    FIBO_SENSITIVE_PREFIXES,
    fast_forward_fibo,
    fibo_diffusers_primary_file,
    fibo_workflow_advice,
    is_fibo_checkpoint_dir,
    is_fibo_state_dict,
    load_fibo_diffusers_dir,
)
from backend.nn.zimage_diffusers import (
    ZIMAGE_INT8_LINEAR_SUFFIXES,
    ZIMAGE_SENSITIVE_PREFIXES,
    convert_diffusers_zimage_transformer,
    fast_forward_zimage,
    is_diffusers_zimage_transformer,
    is_zimage_checkpoint_dir,
    is_zimage_state_dict,
    load_zimage_diffusers_dir,
    zimage_diffusers_primary_file,
    zimage_workflow_advice,
)

EXTENDED_DIT_NAMES: tuple[str, ...] = ("zimage", "ernie", "fibo")


def _build_capability(**kwargs: Any) -> DiTCapability:
    """Instantiate ``DiTCapability``, filtering optional extended fields to
    those currently declared on ``DiTCapability``."""

    valid_fields = set(DiTCapability.__dataclass_fields__.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return DiTCapability(**filtered)


def _zimage_structured_advice(p, *, opts=None) -> list:
    from backend.nn import dit_registry

    advice_cls = getattr(dit_registry, "AdviceItem", None)
    if advice_cls is None:
        return []
    items = []
    for msg in zimage_workflow_advice(p, opts=opts):
        if "Z-Image-Turbo" in msg:
            items.append(
                advice_cls(
                    code="zimage.turbo_mismatch",
                    severity="warning",
                    message=msg,
                    patch={"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}},
                )
            )
        elif "Z-Image Base" in msg:
            items.append(
                advice_cls(
                    code="zimage.base_low_steps",
                    severity="warning",
                    message=msg,
                    patch={"sampler.steps": {"value": 28}, "cfg_scale": {"value": 4.0}},
                )
            )
        else:
            items.append(advice_cls(code="zimage.info", severity="info", message=msg, patch=None))
    return items


def _ernie_structured_advice(p, *, opts=None) -> list:
    from backend.nn import dit_registry

    advice_cls = getattr(dit_registry, "AdviceItem", None)
    if advice_cls is None:
        return []
    items = []
    for msg in ernie_workflow_advice(p, opts=opts):
        if "Turbo" in msg:
            items.append(
                advice_cls(
                    code="ernie.turbo_mismatch",
                    severity="warning",
                    message=msg,
                    patch={"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}},
                )
            )
        else:
            items.append(
                advice_cls(
                    code="ernie.base_low_steps",
                    severity="warning",
                    message=msg,
                    patch={"sampler.steps": {"value": 32}, "cfg_scale": {"value": 4.0}},
                )
            )
    return items


def _fibo_structured_advice(p, *, opts=None) -> list:
    from backend.nn import dit_registry

    advice_cls = getattr(dit_registry, "AdviceItem", None)
    if advice_cls is None:
        return []
    items = []
    for msg in fibo_workflow_advice(p, opts=opts):
        if "Edit Turbo" in msg:
            items.append(
                advice_cls(
                    code="fibo.turbo_mismatch",
                    severity="warning",
                    message=msg,
                    patch={"sampler.steps": {"value": 4}, "cfg_scale": {"value": 1.0}},
                )
            )
        elif "Dev detected" in msg:
            items.append(
                advice_cls(
                    code="fibo.dev_low_steps",
                    severity="warning",
                    message=msg,
                    patch={"sampler.steps": {"value": 32}, "cfg_scale": {"value": 4.0}},
                )
            )
        else:
            items.append(advice_cls(code="fibo.vgl_or_ref", severity="warning", message=msg, patch=None))
    return items


def _is_zimage_dir(path: os.PathLike | str) -> bool:
    return is_zimage_checkpoint_dir(path)


def _load_zimage_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    return load_zimage_diffusers_dir(path)


def _is_ernie_dir(path: os.PathLike | str) -> bool:
    return is_ernie_checkpoint_dir(path)


def _load_ernie_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    return load_ernie_diffusers_dir(path)


def _is_fibo_dir(path: os.PathLike | str) -> bool:
    return is_fibo_checkpoint_dir(path)


def _load_fibo_dir(path: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    return load_fibo_diffusers_dir(path)


def register_extended_dit_capabilities() -> tuple[DiTCapability, ...]:
    """Idempotently register ``zimage``, ``ernie``, and ``fibo`` in ``dit_registry``."""

    zimage_cap = register_dit_capability(
        _build_capability(
            name="zimage",
            dit_cls_name="ZImageTransformer2DModel",
            model_type_name="NextDiT",
            engine_label="Z-Image",
            sensitive_prefixes=ZIMAGE_SENSITIVE_PREFIXES,
            sensitive_module_types=("RMSNorm",),
            matches_state_dict=is_zimage_state_dict,
            block_linear_filter=lambda k: k.startswith(("layers.", "noise_refiner.", "context_refiner.")) and k.endswith(".weight"),
            fp8_rowwise=True,
            int8_convrot_filter=lambda k: k.startswith(("layers.", "noise_refiner.", "context_refiner.")) and k.endswith(ZIMAGE_INT8_LINEAR_SUFFIXES),
            diffusers_arch="z-image/diffusers",
            is_diffusers_dir=_is_zimage_dir,
            diffusers_primary_file=zimage_diffusers_primary_file,
            load_diffusers_dir=_load_zimage_dir,
            is_diffusers_single_file=is_diffusers_zimage_transformer,
            convert_diffusers_single_file=convert_diffusers_zimage_transformer,
            block_attr="layers",
            compile_preset="Z-Image per-block",
            matches_engine=lambda model: bool(getattr(model, "is_zimage_engine", False) or type(model).__name__ == "ZImage"),
            non_matching_engine_reason="Z-Image per-block ignored: non-Z-Image model",
            spectrum_required_attrs=("zimage_final_input", "final_layer", "t_embedder", "compute_zimage_final"),
            spectrum_capture_attr="zimage_final_input",
            spectrum_fast_forward=fast_forward_zimage,
            workflow_advice=zimage_workflow_advice,
            capabilities=frozenset({"multi_reference", "control_blocks", "distilled_cfg1", "per_block_compile", "spectrum"}),
            preset_arch="zit",
            max_references=4,
            reference_slot_format=None,
            supports_rgba=False,
            supports_control_blocks=True,
            attention_backends=("sdpa", "flash", "sage"),
            quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "svdq_w4a4", "gguf"),
            structured_advice=_zimage_structured_advice,
        )
    )

    ernie_cap = register_dit_capability(
        _build_capability(
            name="ernie",
            dit_cls_name="ErnieImageTransformer2DModel",
            model_type_name="ErnieImageModel",
            engine_label="ERNIE-Image",
            sensitive_prefixes=ERNIE_SENSITIVE_PREFIXES,
            sensitive_module_types=("RMSNorm",),
            matches_state_dict=is_ernie_state_dict,
            block_linear_filter=lambda k: k.startswith("layers.") and k.endswith(".weight"),
            fp8_rowwise=True,
            int8_convrot_filter=lambda k: k.startswith("layers.") and k.endswith(ERNIE_INT8_LINEAR_SUFFIXES),
            diffusers_arch="ernie-image/diffusers",
            is_diffusers_dir=_is_ernie_dir,
            diffusers_primary_file=ernie_diffusers_primary_file,
            load_diffusers_dir=_load_ernie_dir,
            block_attr="layers",
            compile_preset="ERNIE-Image per-block",
            matches_engine=lambda model: bool(getattr(model, "is_ernie_engine", False) or type(model).__name__ == "ErnieImage"),
            non_matching_engine_reason="ERNIE-Image per-block ignored: non-ERNIE-Image model",
            spectrum_required_attrs=("final_norm", "final_linear", "time_proj", "time_embedding", "layers"),
            spectrum_capture_attr="final_norm",
            spectrum_fast_forward=fast_forward_ernie,
            workflow_advice=ernie_workflow_advice,
            capabilities=frozenset({"distilled_cfg1", "per_block_compile", "spectrum"}),
            preset_arch="ernie",
            max_references=0,
            reference_slot_format=None,
            supports_rgba=False,
            supports_control_blocks=False,
            attention_backends=("sdpa", "flash", "sage"),
            quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
            structured_advice=_ernie_structured_advice,
        )
    )

    fibo_cap = register_dit_capability(
        _build_capability(
            name="fibo",
            dit_cls_name="FiboTransformer2DModel",
            model_type_name="FiboTransformer2DModel",
            engine_label="Bria Fibo 2.0",
            sensitive_prefixes=FIBO_SENSITIVE_PREFIXES,
            sensitive_module_types=("RMSNorm",),
            matches_state_dict=is_fibo_state_dict,
            block_linear_filter=lambda k: k.startswith("transformer_blocks.") and k.endswith(".weight"),
            fp8_rowwise=True,
            int8_convrot_filter=lambda k: k.startswith("transformer_blocks.") and k.endswith(FIBO_INT8_LINEAR_SUFFIXES),
            diffusers_arch="fibo-2.0/diffusers",
            is_diffusers_dir=_is_fibo_dir,
            diffusers_primary_file=fibo_diffusers_primary_file,
            load_diffusers_dir=_load_fibo_dir,
            block_attr="transformer_blocks",
            compile_preset="Fibo-2.0 per-block",
            matches_engine=lambda model: bool(getattr(model, "is_fibo_engine", False) or type(model).__name__ == "Fibo2"),
            non_matching_engine_reason="Fibo-2.0 per-block ignored: non-Bria Fibo 2.0 model",
            spectrum_required_attrs=("fibo_final_input", "final_layer", "time_in", "compute_fibo_final"),
            spectrum_capture_attr="fibo_final_input",
            spectrum_fast_forward=fast_forward_fibo,
            workflow_advice=fibo_workflow_advice,
            capabilities=frozenset({"multi_reference", "structured_layout", "distilled_cfg1", "per_block_compile", "spectrum"}),
            preset_arch="fibo",
            max_references=FIBO_MAX_REFERENCES,
            reference_slot_format="ref_{i}",
            supports_rgba=False,
            supports_control_blocks=False,
            attention_backends=("sdpa", "flash", "sage"),
            quant_formats=("bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"),
            structured_advice=_fibo_structured_advice,
        )
    )

    return (zimage_cap, ernie_cap, fibo_cap)
