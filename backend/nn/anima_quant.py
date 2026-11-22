"""Load-time storage of the Anima DiT in low bits: the precision-sensitive
parameters kept out of FP8 storage, and the "Diffusion in Low Bits" modes
that quantize a full-precision checkpoint while loading (see backend/loader.py).
Kept apart from the model file (upstream's backend/nn/anima.py)."""

import torch
from torch import nn


# Parts of the DiT that low-bit storage hurts most (the layout comfy-quants
# uses for its Anima FP8/NVFP4 checkpoints, and Unsloth Studio's skip list for
# MOSS-Image): patch/video/text/timestep embedders, shared/head modulations,
# the final projection, Anima block 0 & block 1's modulation, and every norm.
_PRECISION_SENSITIVE_PREFIXES = (
    "x_embedder.",
    "t_embedder.",
    "t_embedding_norm.",
    "final_layer.",
    "blocks.0.",
    "blocks.1.adaln_modulation_",
    # Anima-3.8B's own LLM adapter and Semantic Connector v2 (small, and run on every step)
    "llm_adapter.",
    "anima_v2_connector.",
    # MOSS-Image embedders, shared AdaLN modulation and output head
    "video_embedder.",
    "text_embedder.",
    "time_embedding.",
    "shared_block_modulation.",
    "final_norm_video.",
    "final_linear_video.",
    "video_head_modulation.",
)
# Krea 2 sensitive modules (matching Comfy-Org/Krea-2 & Unsloth skip lists:
# patch/time/text projections, text-fusion projector, output head, norms, mod.lin;
# blocks.0.* IS quantized on Krea 2, unlike Anima).
_KREA2_SENSITIVE_PREFIXES = (
    "first.",
    "tmlp.",
    "tproj.",
    "txtfusion.projector.",
    "txtmlp.",
    "last.",
)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_SUPPORTED_DIT_CLASSES = ("CosmosTransformer3DModel", "MOSSImageTransformer", "Krea2Transformer2DModel")


def should_keep_sensitive_weights_precise(cls_name: str, storage_dtype, state_dict_dtype) -> bool:
    """Whether the loader applies ``keep_sensitive_weights_precise``: only to an
    Anima, MOSS-Image, or Krea 2 DiT stored in FP8 on the fly, not to a
    checkpoint already FP8 or GGUF (its sensitive parts have no higher precision
    to keep)."""

    return cls_name in _SUPPORTED_DIT_CLASSES and storage_dtype in _FP8_DTYPES and state_dict_dtype not in (*_FP8_DTYPES, "gguf")


def keep_sensitive_weights_precise(model: nn.Module, dtype: torch.dtype) -> int:
    """With on-the-fly FP8 storage, store the precision-sensitive parameters
    (see above) in ``dtype`` instead. Call before loading the state dict, so
    they receive the checkpoint's original precision. Returns how many
    parameters were kept."""

    is_krea2 = type(model).__name__ == "SingleStreamDiT"
    prefixes = _KREA2_SENSITIVE_PREFIXES if is_krea2 else _PRECISION_SENSITIVE_PREFIXES
    kept = 0
    for module_name, module in model.named_modules():
        name = module_name + "."
        sensitive = (
            name.startswith(prefixes)
            or isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm))
            or type(module).__name__ in ("RMSNorm", "SimpleModulation", "DoubleSharedModulation")
        )
        if not sensitive:
            continue
        converted = False
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param.dtype in _FP8_DTYPES:
                setattr(module, param_name, nn.Parameter(param.detach().to(dtype), requires_grad=False))
                converted = True
                kept += 1
        if converted and hasattr(module, "parameters_manual_cast"):
            module.parameters_manual_cast = True  # cast to the input's dtype
    return kept


# "Diffusion in Low Bits" modes that quantize a full-precision Anima DiT while
# loading (see backend/loader.py); the precision-sensitive parts stay as they are.
ANIMA_QUANT_SCALED_FP8 = "scaled_fp8"
ANIMA_QUANT_INT8_CONVROT = "int8_convrot"

_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_INT8_CONVROT_GROUP = 256
# Block Linears whose GEMMs have enough rows for INT8 to pay off. The AdaLN
# modulation Linears only see one row per image, and the FFN down projection
# carries the outlier activations (Ideogram INT8 study, arXiv:2606.12280).
_INT8_LINEAR_SUFFIXES = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.output_proj.weight",
    "mlp.layer1.weight",
    # MOSS-Image block Linears (all 6 projections, matching Unsloth INT8-ConvRot)
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
_FFN_DOWN_SUFFIX = "mlp.layer2.weight"
_ADALN_MODULATION = ".adaln_modulation_"


def _is_krea2_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return "txtfusion.projector.weight" in state_dict or "first.weight" in state_dict


def _comfy_quant_conf(conf: dict) -> torch.Tensor:
    import json

    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _block_linear_weights(state_dict: dict[str, torch.Tensor]):
    """``(key, prefix, weight)`` for full-precision block Linear weights outside
    the precision-sensitive parts. Lazily, over a fixed key list: callers
    replace each weight as they go, so the originals are freed one by one."""

    if _is_krea2_state_dict(state_dict):
        keys = [
            k
            for k in state_dict
            if (k.startswith("blocks.") or k.startswith(("txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks.")))
            and k.endswith(".weight")
        ]
        sensitive = _KREA2_SENSITIVE_PREFIXES
    else:
        keys = [k for k in state_dict if (k.startswith("blocks.") or (k.startswith("layers.") and ".proj." in k)) and k.endswith(".weight")]
        sensitive = _PRECISION_SENSITIVE_PREFIXES

    for key in keys:
        weight = state_dict[key]
        if weight.ndim != 2 or not weight.is_floating_point() or weight.dtype in _FP8_DTYPES or key.startswith(sensitive):
            continue
        yield key, key[: -len("weight")], weight


def scale_fp8_state_dict(state_dict: dict[str, torch.Tensor], *, fp8_matmul: bool, device: torch.device = None) -> int:
    """Quantize the block Linear weights of a full-precision state dict to
    scaled ``float8_e4m3fn`` in the ``comfy_quant`` layout, in place.

    Anima uses per-tensor scaling; MOSS-Image (``layers.*.proj.*``) and Krea 2
    use per-row ``float8_e4m3fn_rowwise`` scaling (matching Unsloth's FP8 scheme
    with much higher fidelity than per-tensor scaling). Scaling runs on ``device``
    (the load device) and results are stored on CPU. Returns how many weights were
    quantized."""

    fp8_gemm = _comfy_quant_conf({"format": "float8_e4m3fn"})
    full_gemm = _comfy_quant_conf({"format": "float8_e4m3fn", "full_precision_matrix_mult": True})
    rowwise_fp8_gemm = _comfy_quant_conf({"format": "float8_e4m3fn_rowwise"})
    rowwise_full_gemm = _comfy_quant_conf({"format": "float8_e4m3fn_rowwise", "full_precision_matrix_mult": True})
    is_krea2 = _is_krea2_state_dict(state_dict)

    quantized = 0
    for key, prefix, weight in _block_linear_weights(state_dict):
        weight = weight.to(device=device, dtype=torch.float32)
        if is_krea2 or key.startswith("layers."):
            scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / _FP8_E4M3_MAX
            state_dict[key] = (weight / scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).cpu()
            state_dict[f"{prefix}weight_scale"] = scale.cpu()
            state_dict[f"{prefix}comfy_quant"] = rowwise_fp8_gemm if fp8_matmul else rowwise_full_gemm
        else:
            scale = weight.abs().amax().clamp_min(1e-12) / _FP8_E4M3_MAX
            state_dict[key] = (weight / scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).cpu()
            state_dict[f"{prefix}weight_scale"] = scale.cpu()
            state_dict[f"{prefix}comfy_quant"] = fp8_gemm if fp8_matmul and not key.endswith(_FFN_DOWN_SUFFIX) and _ADALN_MODULATION not in key else full_gemm
        quantized += 1
    return quantized


def quantize_int8_convrot_state_dict(state_dict: dict[str, torch.Tensor], *, device: torch.device = None) -> int:
    """Quantize the block attention projections and FFN up projections of a
    full-precision state dict to INT8 with ConvRot (group-wise Hadamard
    rotation, arXiv:2512.03673) and per-row scales, in the ``comfy_quant``
    ``int8_tensorwise`` layout, in place. Activations are quantized per row at
    run time by comfy-kitchen's ``int8_linear``.

    Rotating runs on ``device`` (the load device: much faster than the CPU);
    the results are stored on the CPU like the rest of the state dict. Returns
    how many weights were quantized."""

    from backend.quant_ops import TensorWiseINT8Layout

    conf = _comfy_quant_conf({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": _INT8_CONVROT_GROUP})
    is_krea2 = _is_krea2_state_dict(state_dict)
    quantized = 0
    for key, prefix, weight in _block_linear_weights(state_dict):
        if is_krea2:
            # Match Comfy-Org/Krea-2 INT8-ConvRot: quantize the 224 main blocks.*
            # projections and keep the smaller text-fusion transformer in full precision.
            if not key.startswith("blocks.") or not key.endswith(_KREA2_INT8_LINEAR_SUFFIXES):
                continue
        elif not key.endswith(_INT8_LINEAR_SUFFIXES):
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
    "Diffusion in Low Bits" mode, empty for none): only to an Anima or MOSS-Image
    DiT stored at full precision, never to one already mixed-precision, Nunchaku,
    FP8 or GGUF."""

    return bool(mode) and cls_name in _SUPPORTED_DIT_CLASSES and quant_config is None and not nunchaku and state_dict_dtype in (torch.float16, torch.bfloat16, torch.float32)


def quantize_on_load(state_dict: dict[str, torch.Tensor], mode: str, load_device: torch.device) -> None:
    """Apply a "Diffusion in Low Bits" Anima mode to a full-precision state dict
    before the model is built; the mixed-precision loader takes it from there.
    Unsupported devices keep the option's plain storage dtype, with a warning."""

    import logging

    from backend import memory_management
    from backend.args import args

    logger = logging.getLogger("loader")
    # The mixed-precision ops compute in bf16, or in fp32 where bf16 is unsupported
    if not memory_management.should_use_bf16(load_device):
        logger.warning(f"Anima {mode} needs a bf16-capable device ; loading without it")
        return
    if mode == ANIMA_QUANT_SCALED_FP8:
        fp8_matmul = args.fast_fp8 and memory_management.supports_fp8_compute(load_device)
        quantized = scale_fp8_state_dict(state_dict, fp8_matmul=fp8_matmul, device=load_device)
        logger.info(f"Anima: stored {quantized} weights as scaled FP8" + (" (FP8 matmul)" if fp8_matmul else ""))
    elif mode == ANIMA_QUANT_INT8_CONVROT:
        if not int8_matmul_available(load_device):
            logger.warning("Anima INT8 needs comfy-kitchen's INT8 GEMM (CUDA 13 build or --enable-triton-backend, SM 8.0+) ; loading at full precision")
            return
        quantized = quantize_int8_convrot_state_dict(state_dict, device=load_device)
        logger.info(f"Anima: stored {quantized} weights as INT8 (ConvRot)")
    else:
        logger.warning(f"Unknown Anima quantization mode {mode!r}")


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
