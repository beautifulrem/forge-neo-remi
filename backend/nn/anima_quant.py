"""Load-time storage of registered DiTs (Anima, MOSS-Image, Krea 2, ...) in low
bits: precision-sensitive parameters kept out of FP8 storage, and the
"Diffusion in Low Bits" modes that quantize a full-precision checkpoint while
loading (see backend/loader.py). Architecture-specific rules live in
``backend/nn/dit_registry.py``."""

import torch
from torch import nn

from backend.nn.dit_registry import (
    capability_for_dit_cls,
    capability_for_model,
    capability_for_state_dict,
    default_sensitive_module_types,
    default_sensitive_prefixes,
)

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def should_keep_sensitive_weights_precise(cls_name: str, storage_dtype, state_dict_dtype) -> bool:
    """Whether the loader applies ``keep_sensitive_weights_precise``: only to a
    registered DiT stored in FP8 on the fly, not to a checkpoint already FP8 or
    GGUF (its sensitive parts have no higher precision to keep)."""

    return capability_for_dit_cls(cls_name) is not None and storage_dtype in _FP8_DTYPES and state_dict_dtype not in (*_FP8_DTYPES, "gguf")


def keep_sensitive_weights_precise(model: nn.Module, dtype: torch.dtype) -> int:
    """With on-the-fly FP8 storage, store the precision-sensitive parameters
    declared by the model's ``DiTCapability`` in ``dtype`` instead. Call before
    loading the state dict, so they receive the checkpoint's original precision.
    Returns how many parameters were kept."""

    cap = capability_for_model(model)
    prefixes = cap.sensitive_prefixes if cap is not None else default_sensitive_prefixes()
    sensitive_types = cap.sensitive_module_types if cap is not None else default_sensitive_module_types()
    kept = 0
    for module_name, module in model.named_modules():
        name = module_name + "."
        sensitive = (
            (bool(prefixes) and name.startswith(prefixes))
            or isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm))
            or type(module).__name__ in sensitive_types
        )
        if not sensitive:
            continue
        converted = False
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param.dtype in _FP8_DTYPES:
                if param.device.type == "mps":
                    precise = param.detach().cpu().to(dtype).to(param.device)
                else:
                    precise = param.detach().to(dtype)
                setattr(module, param_name, nn.Parameter(precise, requires_grad=False))
                converted = True
                kept += 1
        if converted and hasattr(module, "parameters_manual_cast"):
            module.parameters_manual_cast = True  # cast to the input's dtype
    return kept


# "Diffusion in Low Bits" modes that quantize a full-precision DiT while
# loading (see backend/loader.py); the precision-sensitive parts stay as they are.
ANIMA_QUANT_SCALED_FP8 = "scaled_fp8"
ANIMA_QUANT_INT8_CONVROT = "int8_convrot"

_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_INT8_CONVROT_GROUP = 256


def _comfy_quant_conf(conf: dict) -> torch.Tensor:
    import json

    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _block_linear_weights(state_dict: dict[str, torch.Tensor], cap):
    """``(key, prefix, weight)`` for full-precision block Linear weights outside
    ``cap.sensitive_prefixes``. Lazily, over a fixed key list: callers replace
    each weight as they go, so the originals are freed one by one."""

    if cap is None or cap.block_linear_filter is None:
        return
    keys = [k for k in state_dict if cap.block_linear_filter(k)]
    sensitive = cap.sensitive_prefixes

    for key in keys:
        weight = state_dict[key]
        if weight.ndim != 2 or not weight.is_floating_point() or weight.dtype in _FP8_DTYPES or (sensitive and key.startswith(sensitive)):
            continue
        yield key, key[: -len("weight")], weight


def scale_fp8_state_dict(state_dict: dict[str, torch.Tensor], *, fp8_matmul: bool, device: torch.device = None) -> int:
    """Quantize the block Linear weights of a full-precision state dict to
    scaled ``float8_e4m3fn`` in the ``comfy_quant`` layout, in place, following
    the matched ``DiTCapability``'s scaling granularity (per-tensor or per-row)
    and full-precision matmul exemptions. Scaling runs on ``device`` (the load
    device) and results are stored on CPU. Returns how many weights were
    quantized."""

    cap = capability_for_state_dict(state_dict)
    if cap is None:
        return 0
    fp8_gemm = _comfy_quant_conf({"format": "float8_e4m3fn"})
    full_gemm = _comfy_quant_conf({"format": "float8_e4m3fn", "full_precision_matrix_mult": True})
    rowwise_fp8_gemm = _comfy_quant_conf({"format": "float8_e4m3fn_rowwise"})
    rowwise_full_gemm = _comfy_quant_conf({"format": "float8_e4m3fn_rowwise", "full_precision_matrix_mult": True})
    force_full_gemm = cap.fp8_full_precision_matmul_filter

    from backend import memory_management

    cast_on_cpu = device is not None and not memory_management.supports_cast(device, torch.float8_e4m3fn)
    quantized = 0
    for key, prefix, weight in _block_linear_weights(state_dict, cap):
        weight = weight.to(device=device, dtype=torch.float32)
        if cap.fp8_rowwise:
            scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / _FP8_E4M3_MAX
            q = (weight / scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
            state_dict[key] = q.cpu().to(torch.float8_e4m3fn) if cast_on_cpu else q.to(torch.float8_e4m3fn).cpu()
            state_dict[f"{prefix}weight_scale"] = scale.cpu()
            use_fp8_matmul = fp8_matmul and not (force_full_gemm is not None and force_full_gemm(key))
            state_dict[f"{prefix}comfy_quant"] = rowwise_fp8_gemm if use_fp8_matmul else rowwise_full_gemm
        else:
            scale = weight.abs().amax().clamp_min(1e-12) / _FP8_E4M3_MAX
            q = (weight / scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
            state_dict[key] = q.cpu().to(torch.float8_e4m3fn) if cast_on_cpu else q.to(torch.float8_e4m3fn).cpu()
            state_dict[f"{prefix}weight_scale"] = scale.cpu()
            use_fp8_matmul = fp8_matmul and not (force_full_gemm is not None and force_full_gemm(key))
            state_dict[f"{prefix}comfy_quant"] = fp8_gemm if use_fp8_matmul else full_gemm
        quantized += 1
    return quantized


def quantize_int8_convrot_state_dict(state_dict: dict[str, torch.Tensor], *, device: torch.device = None) -> int:
    """Quantize the block attention projections and FFN projections declared by
    ``cap.int8_convrot_filter`` to INT8 with ConvRot (group-wise Hadamard
    rotation, arXiv:2512.03673) and per-row scales, in the ``comfy_quant``
    ``int8_tensorwise`` layout, in place. Activations are quantized per row at
    run time by comfy-kitchen's ``int8_linear``.

    Rotating runs on ``device`` (the load device: much faster than the CPU);
    the results are stored on the CPU like the rest of the state dict. Returns
    how many weights were quantized."""

    cap = capability_for_state_dict(state_dict)
    if cap is None or cap.int8_convrot_filter is None:
        return 0

    from backend.quant_ops import TensorWiseINT8Layout

    conf = _comfy_quant_conf({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": _INT8_CONVROT_GROUP})
    quantized = 0
    for key, prefix, weight in _block_linear_weights(state_dict, cap):
        if not cap.int8_convrot_filter(key):
            continue
        if weight.shape[1] % _INT8_CONVROT_GROUP:
            continue
        qdata, params = TensorWiseINT8Layout.quantize(
            weight.to(device=device, dtype=torch.float32),
            is_weight=True,
            per_channel=True,
            convrot=True,
            convrot_groupsize=_INT8_CONVROT_GROUP,
        )
        state_dict[key] = qdata.cpu()
        state_dict[f"{prefix}weight_scale"] = params.scale.float().cpu()
        state_dict[f"{prefix}comfy_quant"] = conf
        quantized += 1
    return quantized


def should_quantize_on_load(mode, cls_name: str, quant_config, nunchaku: bool, state_dict_dtype) -> bool:
    """Whether the loader applies ``quantize_on_load`` (``mode`` is the chosen
    "Diffusion in Low Bits" mode, empty for none): only to a registered DiT
    stored at full precision, never to one already mixed-precision, Nunchaku,
    FP8 or GGUF."""

    return bool(mode) and capability_for_dit_cls(cls_name) is not None and quant_config is None and not nunchaku and state_dict_dtype in (torch.float16, torch.bfloat16, torch.float32)


def quantize_on_load(state_dict: dict[str, torch.Tensor], mode: str, load_device: torch.device) -> None:
    """Apply a "Diffusion in Low Bits" mode to a full-precision DiT state dict
    before the model is built; the mixed-precision loader takes it from there.
    Unsupported devices keep the option's plain storage dtype, with a warning."""

    import logging

    from backend import memory_management
    from backend.args import args

    logger = logging.getLogger("loader")
    cap = capability_for_state_dict(state_dict)
    label = cap.engine_label if cap is not None else "DiT"
    # The mixed-precision ops compute in bf16, or in fp32 where bf16 is unsupported
    if not memory_management.should_use_bf16(load_device):
        logger.warning(f"{label} {mode} needs a bf16-capable device ; loading without it")
        return
    is_mps = memory_management.is_device_mps(load_device)
    if mode == ANIMA_QUANT_SCALED_FP8:
        if not memory_management.supports_cast(load_device, torch.float8_e4m3fn):
            if is_mps and cap is not None and cap.int8_convrot_filter is not None:
                logger.warning(f"{label}: scaled FP8 is not supported on {load_device}; falling back to INT8 (ConvRot)")
                quantized = quantize_int8_convrot_state_dict(state_dict, device=memory_management.cpu)
                logger.info(f"{label}: stored {quantized} weights as INT8 (ConvRot, CPU-rotated fallback for MPS)")
            else:
                logger.warning(f"{label}: scaled FP8 is not supported on {load_device}; loading at full precision")
            return
        fp8_matmul = args.fast_fp8 and memory_management.supports_fp8_compute(load_device)
        quantized = scale_fp8_state_dict(state_dict, fp8_matmul=fp8_matmul, device=load_device)
        logger.info(f"{label}: stored {quantized} weights as scaled FP8" + (" (FP8 matmul)" if fp8_matmul else ""))
    elif mode == ANIMA_QUANT_INT8_CONVROT:
        if not int8_matmul_available(load_device) and not is_mps:
            logger.warning(f"{label} INT8 needs comfy-kitchen's INT8 GEMM (CUDA 13 build or --enable-triton-backend, SM 8.0+) ; loading at full precision")
            return
        # On MPS, CPU rotation is 3-40x faster and _MPSDequantCache handles dequantized matmul
        quant_device = memory_management.cpu if is_mps else load_device
        quantized = quantize_int8_convrot_state_dict(state_dict, device=quant_device)
        logger.info(f"{label}: stored {quantized} weights as INT8 (ConvRot)" + (" (CPU-rotated for MPS)" if is_mps else ""))
    else:
        logger.warning(f"Unknown {label} quantization mode {mode!r}")


def int8_matmul_available(device: torch.device) -> bool:
    """Whether comfy-kitchen has a real INT8 GEMM (CUDA / Triton, SM 8.0+) for
    ``device``; its eager fallback would make INT8 weights slower than bf16."""

    from backend.quant_ops import ck

    device = torch.device(device)
    if device.type != "cuda":
        return False
    try:
        return ck.registry.get_capable_backend(
            "int8_linear",
            {
                "x": torch.zeros(1, _INT8_CONVROT_GROUP, device=device, dtype=torch.bfloat16),
                "weight": torch.zeros(1, _INT8_CONVROT_GROUP, device=device, dtype=torch.int8),
                "weight_scale": torch.ones(1, 1, device=device),
            },
        ) != "eager"
    except Exception:
        return False
