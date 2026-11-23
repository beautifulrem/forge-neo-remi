"""Unified quantization format detection, metadata & attention backend capability matrix.

Centralizes:
- Format identification across all modern low-bit checkpoint encodings used by
  Forge Neo Remi and ComfyUI: **INT8-ConvRot**, **INT8-TensorWise**, **FP8**
  (tensorwise & rowwise E4M3/E5M2), **NVFP4**, **SVDQuant W4A4 / NVFP4**,
  **GGUF**, and full/half precision (`bf16`/`fp16`/`fp32`);
- Per-DiT supported quantization format lookup;
- Explicit attention backend capability matrix (SageAttention 1/2/3,
  FlashAttention, Comfy-Kitchen INT8, xFormers, PyTorch SDPA, Basic) with
  deterministic safe fallback selection for arbitrary device, head_dim, mask,
  GQA, and ``torch.compile`` constraints.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger("quant_formats")


# region Quantization format detection & metadata


@dataclass(frozen=True)
class QuantFormatInfo:
    """Normalized quantization metadata for a loaded state dict or checkpoint."""

    format_id: str
    label: str
    bits_per_weight: float
    quantized_tensors: int = 0
    total_tensors: int = 0
    convrot_groupsize: int | None = None
    gguf_types: tuple[str, ...] = ()
    svdq_rank: int | None = None
    hardware_requirements: tuple[str, ...] = ()
    supported_devices: tuple[str, ...] = ("cuda", "mps", "cpu")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_id": self.format_id,
            "label": self.label,
            "bits_per_weight": self.bits_per_weight,
            "quantized_tensors": self.quantized_tensors,
            "total_tensors": self.total_tensors,
            "convrot_groupsize": self.convrot_groupsize,
            "gguf_types": list(self.gguf_types),
            "svdq_rank": self.svdq_rank,
            "hardware_requirements": list(self.hardware_requirements),
            "supported_devices": list(self.supported_devices),
        }


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _decode_comfy_quant(tensor: torch.Tensor) -> dict[str, Any]:
    try:
        raw = tensor.detach().cpu().numpy().tobytes()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _parse_unsloth_meta(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    raw = metadata.get("unsloth_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def is_svdq_state_dict(state_dict: Mapping[str, Any]) -> bool:
    """True when ``state_dict`` contains Nunchaku SVDQuant W4A4 / NVFP4 tensors."""

    has_qweight = False
    has_svdq_sidecar = False
    for key in state_dict:
        if key.endswith(".qweight"):
            has_qweight = True
        elif key.endswith((".wscales", ".proj_down", ".proj_up", ".wcscales")):
            has_svdq_sidecar = True
        if has_qweight and has_svdq_sidecar:
            return True
    return False


def detect_quant_format(
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any] | None = None,
) -> QuantFormatInfo:
    """Detect the dominant quantization format of ``state_dict`` + ``metadata``."""

    total_tensors = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
    meta = metadata or {}
    unsloth = _parse_unsloth_meta(meta)

    # 1. GGUF tensors (ParameterGGUF carrying .gguf_cls / .tensor_type)
    gguf_tensors = 0
    gguf_types: set[str] = set()
    for val in state_dict.values():
        if getattr(val, "gguf_cls", None) is not None or getattr(val, "tensor_type", None) is not None:
            gguf_tensors += 1
            tt = getattr(val, "tensor_type", None)
            if tt is not None:
                gguf_types.add(getattr(tt, "name", str(tt)))
    if gguf_tensors > 0:
        sorted_types = tuple(sorted(gguf_types))
        bits = 4.5 if any("Q4" in t for t in sorted_types) else (8.0 if any("Q8" in t for t in sorted_types) else 6.0)
        return QuantFormatInfo(
            format_id="gguf",
            label="GGUF",
            bits_per_weight=bits,
            quantized_tensors=gguf_tensors,
            total_tensors=total_tensors,
            gguf_types=sorted_types,
            supported_devices=("cuda", "mps", "cpu"),
        )

    # 2. SVDQuant W4A4 / NVFP4 (Nunchaku layout)
    if is_svdq_state_dict(state_dict):
        qweights = [k for k in state_dict if k.endswith(".qweight")]
        rank: int | None = None
        for k, v in state_dict.items():
            if k.endswith(".proj_down") and isinstance(v, torch.Tensor) and v.ndim == 2:
                rank = int(min(v.shape))
                break
        precision = str(meta.get("precision", "")).lower()
        is_fp4 = precision == "nvfp4" or any(k.endswith(".wtscale") for k in state_dict)
        return QuantFormatInfo(
            format_id="svdq_nvfp4" if is_fp4 else "svdq_w4a4",
            label="SVDQuant NVFP4" if is_fp4 else "SVDQuant W4A4",
            bits_per_weight=4.5,
            quantized_tensors=len(qweights),
            total_tensors=total_tensors,
            svdq_rank=rank,
            hardware_requirements=("CUDA SM 10.0+ (Blackwell)" if is_fp4 else "CUDA SM 8.0+", "nunchaku"),
            supported_devices=("cuda",),
        )

    # 3. comfy_quant per-layer descriptors (INT8-ConvRot, INT8, NVFP4, FP8)
    comfy_counts: dict[str, int] = {}
    convrot_layers = 0
    convrot_group: int | None = None
    for key, val in state_dict.items():
        if key.endswith(".comfy_quant") and isinstance(val, torch.Tensor):
            conf = _decode_comfy_quant(val)
            fmt = str(conf.get("format", "")).lower()
            if fmt:
                comfy_counts[fmt] = comfy_counts.get(fmt, 0) + 1
            if conf.get("convrot"):
                convrot_layers += 1
                if convrot_group is None:
                    convrot_group = int(conf.get("convrot_groupsize", 256))

    if comfy_counts:
        dominant = max(comfy_counts, key=comfy_counts.get)
        q_count = sum(comfy_counts.values())
        if "nvfp4" in dominant or "fp4" in dominant:
            return QuantFormatInfo(
                format_id="nvfp4",
                label="NVFP4",
                bits_per_weight=4.25,
                quantized_tensors=q_count,
                total_tensors=total_tensors,
                hardware_requirements=("CUDA SM 10.0+ (Blackwell)",),
                supported_devices=("cuda",),
            )
        if "int8" in dominant:
            if convrot_layers > 0:
                return QuantFormatInfo(
                    format_id="int8_convrot",
                    label="INT8-ConvRot",
                    bits_per_weight=8.0,
                    quantized_tensors=q_count,
                    total_tensors=total_tensors,
                    convrot_groupsize=convrot_group or 256,
                    hardware_requirements=("CUDA SM 8.0+ or Triton INT8 GEMM",),
                    supported_devices=("cuda", "mps", "cpu"),
                )
            return QuantFormatInfo(
                format_id="int8_tensorwise",
                label="INT8",
                bits_per_weight=8.0,
                quantized_tensors=q_count,
                total_tensors=total_tensors,
                supported_devices=("cuda", "mps", "cpu"),
            )
        if "float8_e4m3fn_rowwise" in dominant:
            return QuantFormatInfo(
                format_id="fp8_e4m3_rowwise",
                label="FP8",
                bits_per_weight=8.0,
                quantized_tensors=q_count,
                total_tensors=total_tensors,
                hardware_requirements=("CUDA SM 8.9+ for native FP8 matmul (dequant fallback elsewhere)",),
                supported_devices=("cuda", "mps", "cpu"),
            )
        if "float8_e5m2" in dominant:
            return QuantFormatInfo(
                format_id="fp8_e5m2",
                label="FP8",
                bits_per_weight=8.0,
                quantized_tensors=q_count,
                total_tensors=total_tensors,
                supported_devices=("cuda", "mps", "cpu"),
            )
        if "float8" in dominant or "fp8" in dominant:
            return QuantFormatInfo(
                format_id="fp8_e4m3",
                label="FP8",
                bits_per_weight=8.0,
                quantized_tensors=q_count,
                total_tensors=total_tensors,
                supported_devices=("cuda", "mps", "cpu"),
            )

    # 4. Raw TorchAO / Unsloth _weight_qdata tensors (prior to convert_quantization)
    qdata_tensors = [v for k, v in state_dict.items() if k.endswith("._weight_qdata") and isinstance(v, torch.Tensor)]
    if qdata_tensors:
        sample = qdata_tensors[0]
        act_rot = str(unsloth.get("activation_rotation") or "").lower()
        if sample.dtype == torch.int8:
            if act_rot.startswith("convrot"):
                return QuantFormatInfo(
                    format_id="int8_convrot",
                    label="INT8-ConvRot",
                    bits_per_weight=8.0,
                    quantized_tensors=len(qdata_tensors),
                    total_tensors=total_tensors,
                    convrot_groupsize=int(unsloth.get("activation_rotation_group") or 256),
                    hardware_requirements=("CUDA SM 8.0+ or Triton INT8 GEMM",),
                    supported_devices=("cuda", "mps", "cpu"),
                )
            return QuantFormatInfo(
                format_id="int8_tensorwise",
                label="INT8",
                bits_per_weight=8.0,
                quantized_tensors=len(qdata_tensors),
                total_tensors=total_tensors,
                supported_devices=("cuda", "mps", "cpu"),
            )
        if sample.dtype in _FP8_DTYPES:
            is_rowwise = str(unsloth.get("fp8_granularity", "")).lower() == "per_row"
            return QuantFormatInfo(
                format_id="fp8_e4m3_rowwise" if is_rowwise else "fp8_e4m3",
                label="FP8",
                bits_per_weight=8.0,
                quantized_tensors=len(qdata_tensors),
                total_tensors=total_tensors,
                supported_devices=("cuda", "mps", "cpu"),
            )
        if sample.dtype == torch.uint8 and "fp4" in str(unsloth.get("scheme", "")).lower():
            return QuantFormatInfo(
                format_id="nvfp4",
                label="NVFP4",
                bits_per_weight=4.25,
                quantized_tensors=len(qdata_tensors),
                total_tensors=total_tensors,
                hardware_requirements=("CUDA SM 10.0+ (Blackwell)",),
                supported_devices=("cuda",),
            )

    # 5. Native FP8 safetensors (or scaled_fp8 marker)
    fp8_count = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor) and v.dtype in _FP8_DTYPES)
    if fp8_count > 0:
        has_e5m2 = any(isinstance(v, torch.Tensor) and v.dtype == torch.float8_e5m2 for v in state_dict.values())
        return QuantFormatInfo(
            format_id="fp8_e5m2" if has_e5m2 else "fp8_e4m3",
            label="FP8",
            bits_per_weight=8.0,
            quantized_tensors=fp8_count,
            total_tensors=total_tensors,
            supported_devices=("cuda", "mps", "cpu"),
        )

    # 6. Full / half precision fallback
    dtype_counts: dict[torch.dtype, int] = {}
    for v in state_dict.values():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            dtype_counts[v.dtype] = dtype_counts.get(v.dtype, 0) + 1
    dominant_dtype = max(dtype_counts, key=dtype_counts.get) if dtype_counts else torch.bfloat16
    if dominant_dtype == torch.float16:
        return QuantFormatInfo("fp16", "FP16", 16.0, 0, total_tensors)
    if dominant_dtype == torch.float32:
        return QuantFormatInfo("fp32", "FP32", 32.0, 0, total_tensors)
    return QuantFormatInfo("bf16", "BF16", 16.0, 0, total_tensors)


_DIT_EXTRA_FORMATS: dict[str, tuple[str, ...]] = {
    "anima": ("bf16", "fp8", "int8_convrot", "gguf"),
    "moss": ("bf16", "fp8", "int8_convrot", "gguf"),
    "krea2": ("bf16", "fp8", "int8_convrot", "gguf"),
    "flux3": ("bf16", "fp8", "int8_convrot", "nvfp4", "gguf"),
    "qwen21": ("bf16", "fp8", "int8_convrot", "nvfp4", "gguf"),
    "zimage": ("bf16", "fp8", "int8_convrot", "svdq_w4a4", "nvfp4", "gguf"),
    "ernie": ("bf16", "fp8", "int8_convrot", "gguf"),
    "fibo": ("bf16", "fp8", "int8_convrot", "gguf"),
    "flux": ("bf16", "fp8", "svdq_w4a4", "nvfp4", "gguf"),
    "qwen": ("bf16", "fp8", "svdq_w4a4", "gguf"),
}


def quant_formats_for_dit(dit_name: str) -> tuple[str, ...]:
    """Return supported quantization format IDs for a registered DiT family."""

    key = (dit_name or "").strip().lower()
    if key in _DIT_EXTRA_FORMATS:
        return _DIT_EXTRA_FORMATS[key]
    from backend.nn.dit_registry import get_dit_capability

    cap = get_dit_capability(key)
    if cap is None:
        return ("bf16", "fp8", "gguf")
    formats = ["bf16", "fp8"]
    if cap.int8_convrot_filter is not None:
        formats.append("int8_convrot")
    formats.append("gguf")
    return tuple(formats)


# endregion


# region Attention backend capability matrix & safe fallback


@dataclass(frozen=True)
class AttentionBackendSpec:
    """Declarative capability specification for an attention backend."""

    name: str
    supports_mask: bool
    supports_4d_bool_mask: bool
    supports_additive_mask: bool
    supports_causal: bool
    supports_gqa: bool
    max_head_dim: int | None
    supported_head_dims: tuple[int, ...] | None
    supported_devices: tuple[str, ...]
    torch_compile_safe: bool


ATTENTION_CAPABILITY_MATRIX: dict[str, AttentionBackendSpec] = {
    "sage3": AttentionBackendSpec(
        name="sage3",
        supports_mask=True,
        supports_4d_bool_mask=True,
        supports_additive_mask=False,
        supports_causal=True,
        supports_gqa=False,
        max_head_dim=128,
        supported_head_dims=(64, 128),
        supported_devices=("cuda",),
        torch_compile_safe=False,
    ),
    "sage2": AttentionBackendSpec(
        name="sage2",
        supports_mask=True,
        supports_4d_bool_mask=True,
        supports_additive_mask=True,
        supports_causal=True,
        supports_gqa=False,
        max_head_dim=128,
        supported_head_dims=None,
        supported_devices=("cuda",),
        torch_compile_safe=False,
    ),
    "sage1": AttentionBackendSpec(
        name="sage1",
        supports_mask=False,
        supports_4d_bool_mask=False,
        supports_additive_mask=False,
        supports_causal=True,
        supports_gqa=False,
        max_head_dim=128,
        supported_head_dims=(64, 96, 128),
        supported_devices=("cuda",),
        torch_compile_safe=False,
    ),
    "flash": AttentionBackendSpec(
        name="flash",
        supports_mask=False,
        supports_4d_bool_mask=False,
        supports_additive_mask=False,
        supports_causal=True,
        supports_gqa=True,
        max_head_dim=256,
        supported_head_dims=None,
        supported_devices=("cuda",),
        torch_compile_safe=False,
    ),
    "ck_int8": AttentionBackendSpec(
        name="ck_int8",
        supports_mask=True,
        supports_4d_bool_mask=True,
        supports_additive_mask=True,
        supports_causal=False,
        supports_gqa=True,
        max_head_dim=128,
        supported_head_dims=None,
        supported_devices=("cuda",),
        torch_compile_safe=False,
    ),
    "xformers": AttentionBackendSpec(
        name="xformers",
        supports_mask=True,
        supports_4d_bool_mask=False,
        supports_additive_mask=True,
        supports_causal=False,
        supports_gqa=False,
        max_head_dim=256,
        supported_head_dims=None,
        supported_devices=("cuda",),
        torch_compile_safe=False,
    ),
    "sdpa": AttentionBackendSpec(
        name="sdpa",
        supports_mask=True,
        supports_4d_bool_mask=True,
        supports_additive_mask=True,
        supports_causal=True,
        supports_gqa=True,
        max_head_dim=None,
        supported_head_dims=None,
        supported_devices=("cuda", "mps", "cpu"),
        torch_compile_safe=True,
    ),
    "basic": AttentionBackendSpec(
        name="basic",
        supports_mask=True,
        supports_4d_bool_mask=True,
        supports_additive_mask=True,
        supports_causal=True,
        supports_gqa=True,
        max_head_dim=None,
        supported_head_dims=None,
        supported_devices=("cuda", "mps", "cpu"),
        torch_compile_safe=True,
    ),
}


def backend_incompatibility_reason(
    backend: str,
    *,
    device: str | torch.device = "cpu",
    head_dim: int = 128,
    has_mask: bool = False,
    mask_is_bool: bool = False,
    mask_is_causal: bool = False,
    needs_gqa: bool = False,
    compiled: bool = False,
) -> str | None:
    """Return a human-readable incompatibility reason or ``None`` if supported."""

    spec = ATTENTION_CAPABILITY_MATRIX.get(backend)
    if spec is None:
        return f"unknown attention backend {backend!r}"
    dev_type = torch.device(device).type
    if dev_type not in spec.supported_devices:
        return f"{backend} does not support device '{dev_type}'"
    if spec.max_head_dim is not None and head_dim > spec.max_head_dim:
        return f"{backend} max head_dim is {spec.max_head_dim} (got {head_dim})"
    if spec.supported_head_dims is not None and head_dim not in spec.supported_head_dims:
        return f"{backend} supports head_dims {spec.supported_head_dims} (got {head_dim})"
    if has_mask:
        if mask_is_causal and spec.supports_causal and not spec.supports_mask:
            pass
        elif not spec.supports_mask:
            return f"{backend} does not support arbitrary attention masks"
        elif mask_is_bool and not spec.supports_4d_bool_mask:
            return f"{backend} does not support boolean attention masks"
        elif not mask_is_bool and not spec.supports_additive_mask:
            return f"{backend} does not support additive float attention masks"
    if needs_gqa and not spec.supports_gqa:
        return f"{backend} does not support grouped-query attention (GQA)"
    if compiled and not spec.torch_compile_safe:
        return f"{backend} is not torch.compile graph-safe"
    return None


def select_attention_backend(
    *,
    preferred: str | None = None,
    device: str | torch.device = "cpu",
    head_dim: int = 128,
    has_mask: bool = False,
    mask_is_bool: bool = False,
    mask_is_causal: bool = False,
    needs_gqa: bool = False,
    compiled: bool = False,
    available_backends: Sequence[str] | None = None,
) -> tuple[str, str | None]:
    """Select the fastest compatible attention backend with safe fallback.

    Returns ``(selected_backend, fallback_reason_or_none)``.
    """

    candidates = list(available_backends) if available_backends is not None else ["sdpa", "basic"]
    if preferred:
        reason = backend_incompatibility_reason(
            preferred,
            device=device,
            head_dim=head_dim,
            has_mask=has_mask,
            mask_is_bool=mask_is_bool,
            mask_is_causal=mask_is_causal,
            needs_gqa=needs_gqa,
            compiled=compiled,
        )
        if reason is None and (available_backends is None or preferred in available_backends):
            return preferred, None
        fallback_reason = reason or f"{preferred} is not available in this runtime"
    else:
        fallback_reason = None

    for candidate in (*candidates, "sdpa", "basic"):
        if candidate not in ATTENTION_CAPABILITY_MATRIX:
            continue
        if (
            backend_incompatibility_reason(
                candidate,
                device=device,
                head_dim=head_dim,
                has_mask=has_mask,
                mask_is_bool=mask_is_bool,
                mask_is_causal=mask_is_causal,
                needs_gqa=needs_gqa,
                compiled=compiled,
            )
            is None
        ):
            return candidate, fallback_reason

    return "basic", fallback_reason or "fallback to basic attention"


def safe_attention_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    preferred: str | None = None,
    enable_gqa: bool = False,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    transformer_options: dict | None = None,
    **kwargs,
) -> torch.Tensor:
    """Invoke the best compatible attention backend, falling back safely to
    PyTorch SDPA / Basic attention on mask, head_dim, device, or kernel errors."""

    from backend import attention as attn_mod

    head_dim = int(q.shape[-1]) if skip_reshape else int(q.shape[-1]) // int(heads)
    backend_map = {
        "sage3": getattr(attn_mod, "attention_sage", None),
        "sage2": getattr(attn_mod, "attention_sage", None),
        "sage1": getattr(attn_mod, "attention_sage", None),
        "flash": getattr(attn_mod, "attention_flash", None),
        "ck_int8": getattr(attn_mod, "attention_comfy_kitchen_int8", None),
        "xformers": getattr(attn_mod, "attention_xformers", None),
        "sdpa": attn_mod.attention_pytorch,
        "basic": attn_mod.attention_basic,
    }
    available = [name for name, fn in backend_map.items() if fn is not None]
    selected, _reason = select_attention_backend(
        preferred=preferred,
        device=q.device,
        head_dim=head_dim,
        has_mask=mask is not None,
        mask_is_bool=bool(mask is not None and mask.dtype == torch.bool),
        mask_is_causal=causal,
        needs_gqa=enable_gqa,
        compiled=bool(getattr(torch.compiler, "is_compiling", lambda: False)()),
        available_backends=available,
    )
    fn = backend_map.get(selected) or attn_mod.attention_pytorch
    call_kwargs = dict(
        mask=mask,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        enable_gqa=enable_gqa,
        transformer_options=transformer_options or {},
        **kwargs,
    )
    if causal:
        call_kwargs["causal"] = True
    try:
        return fn(q, k, v, heads, **call_kwargs)
    except Exception as err:
        logger.warning("Attention backend %s failed (%s); falling back to SDPA/Basic.", selected, err)
        call_kwargs.pop("causal", None)
        try:
            return attn_mod.attention_pytorch(q, k, v, heads, **call_kwargs)
        except Exception:
            return attn_mod.attention_basic(q, k, v, heads, **call_kwargs)


# endregion
