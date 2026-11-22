# https://github.com/Comfy-Org/ComfyUI/blob/v0.33.1/comfy/quant_ops.py

from dataclasses import dataclass

import comfy_kitchen as ck
import torch
from comfy_kitchen.scaled_mm_v2 import ScalingType, has_scaled_mm_v2
from comfy_kitchen.tensor import (  # noqa
    AsymW4A8Int8Layout,
    QuantizedTensor,
    TensorCoreConvRotW4A4Layout,
    TensorCoreFP8Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
    get_layout_class,
    register_layout_class,
)
from comfy_kitchen.tensor.base import (
    BaseLayoutParams,
    QuantizedLayout,
    dequantize_args,
    register_layout_op,
)

if torch.version.cuda is None:
    ck.registry.disable("cuda")
else:
    cuda_version = tuple(map(int, str(torch.version.cuda).split(".")))
    if cuda_version < (13,):
        ck.registry.disable("cuda")

from backend.args import args

if args.enable_triton_backend:
    try:
        import triton  # noqa
    except ImportError:
        ck.registry.disable("triton")
else:
    ck.registry.disable("triton")

import importlib.metadata

ver = importlib.metadata.version("comfy-kitchen")

print(f"Comfy-Kitchen {ver}:", {k: v["available"] and not v["disabled"] for k, v in ck.list_backends().items()})

from . import float

# region Layouts


class TensorCoreFP8LayoutBase(TensorCoreFP8Layout):
    FP8_DTYPE = None

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        assert cls.FP8_DTYPE is not None

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:
                tensor_info = torch.finfo(tensor.dtype)
                scale = 1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max)

        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = float.stochastic_rounding(tensor, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        params = cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params


class TensorCoreMXFP8Layout_(TensorCoreMXFP8Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        assert tensor.dim() == 2

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = float.stochastic_round_quantize_mxfp8_by_block(tensor, pad_32x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)

        params = cls.Params(
            scale=block_scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params


class TensorCoreNVFP4Layout_(TensorCoreNVFP4Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        assert tensor.dim() == 2

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scale = torch.amax(tensor.abs()) / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = float.stochastic_round_quantize_nvfp4_by_block(tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_scale=block_scale,
        )
        return qdata, params


class TensorCoreFP8E4M3Layout(TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


class RowWiseFP8LayoutBase(QuantizedLayout):
    """Per-row (channel-wise) FP8 weight quantization (e.g. TorchAO / Unsloth
    ``Float8Tensor`` with ``block_size=[1, K]`` and ``scale`` shape ``[N, 1]``).

    Preserves 1-byte FP8 storage and exact per-row scales (``qdata * scale``).
    Uses dynamic per-token FP8 ``scaled_mm(ScalingType.RowWise)`` on SM >= 8.9
    CUDA GPUs when shapes satisfy hardware alignment, falling back seamlessly
    to per-row dequantized matmul elsewhere.
    """

    MIN_SM_VERSION = (8, 9)
    FP8_DTYPE: torch.dtype | None = None

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        transposed: bool = False

        def _validate_tensor_fields(self):
            if isinstance(self.scale, torch.Tensor):
                object.__setattr__(self, "scale", self.scale.to(dtype=torch.float32, non_blocking=True))

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False, **kwargs):
        assert cls.FP8_DTYPE is not None
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        fmax = torch.finfo(cls.FP8_DTYPE).max

        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scale = tensor.abs().amax(dim=-1, keepdim=True).to(dtype=torch.float32).clamp_min(1e-12) / fmax
        elif not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)
        else:
            scale = scale.to(device=tensor.device, dtype=torch.float32)

        scaled = tensor.to(dtype=torch.float32) / scale
        if stochastic_rounding and stochastic_rounding > 0:
            qdata = float.stochastic_rounding(scaled, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = scaled.clamp(-fmax, fmax).to(dtype=cls.FP8_DTYPE)

        params = cls.Params(scale=scale, orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        out = qdata.to(dtype=torch.float32) * params.scale.to(device=qdata.device, dtype=torch.float32)
        if getattr(params, "transposed", False):
            out = out.t()
        return out.to(dtype=params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        return {"": qdata, "_scale": params.scale}


class RowWiseFP8E4M3Layout(RowWiseFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class RowWiseFP8E5M2Layout(RowWiseFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


def _rowwise_fp8_linear(input_tensor: torch.Tensor, weight: QuantizedTensor, bias: torch.Tensor | None = None, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if getattr(weight._params, "transposed", False):
        return torch.nn.functional.linear(input_tensor, weight.dequantize().to(dtype=input_tensor.dtype), bias)

    w_qdata, w_scale = RowWiseFP8LayoutBase.get_plain_tensors(weight)
    target_dtype = out_dtype or input_tensor.dtype

    if (
        input_tensor.is_cuda
        and w_qdata.is_cuda
        and getattr(torch.version, "hip", None) is None
        and has_scaled_mm_v2()
        and hasattr(ScalingType, "RowWise")
        and w_qdata.dim() == 2
        and w_scale.dim() == 2
        and w_scale.shape[1] == 1
        and target_dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        orig_shape = input_tensor.shape
        x2d = input_tensor.reshape(-1, orig_shape[-1])
        m, k = x2d.shape
        n = w_qdata.shape[0]
        if m > 0 and (m % 16 == 0) and (k % 16 == 0) and (n % 16 == 0):
            try:
                fmax = torch.finfo(w_qdata.dtype).max
                scale_a = x2d.abs().amax(dim=-1, keepdim=True).to(dtype=torch.float32).clamp_min(1e-12) / fmax
                x_q = (x2d.to(dtype=torch.float32) / scale_a).clamp(-fmax, fmax).to(dtype=w_qdata.dtype).contiguous()
                mm_bias = bias.to(device=w_qdata.device, dtype=target_dtype) if bias is not None else None
                out = torch.nn.functional.scaled_mm(
                    x_q,
                    w_qdata.t(),
                    scale_a=scale_a,
                    scale_recipe_a=ScalingType.RowWise,
                    scale_b=w_scale.to(device=w_qdata.device, dtype=torch.float32).t(),
                    scale_recipe_b=ScalingType.RowWise,
                    bias=mm_bias,
                    output_dtype=target_dtype,
                    use_fast_accum=True,
                )
                return out.reshape((*orig_shape[:-1], n))
            except (RuntimeError, TypeError):
                pass

    w_deq = (w_qdata.to(dtype=torch.float32) * w_scale.to(device=w_qdata.device, dtype=torch.float32)).to(dtype=input_tensor.dtype)
    return torch.nn.functional.linear(input_tensor, w_deq, bias)


@register_layout_op(torch.ops.aten.linear.default, RowWiseFP8LayoutBase)
def _handle_rowwise_fp8_linear(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    return _rowwise_fp8_linear(input_tensor, weight, bias, out_dtype=kwargs.get("out_dtype"))


@register_layout_op(torch.ops.aten.mm.default, RowWiseFP8LayoutBase)
def _handle_rowwise_fp8_mm(qt, args, kwargs):
    a, b = args[0], args[1]
    if isinstance(b, QuantizedTensor) and getattr(b._params, "transposed", False):
        untransposed = QuantizedTensor(
            b._qdata,
            b._layout_cls,
            RowWiseFP8LayoutBase.Params(scale=b._params.scale, orig_dtype=b._params.orig_dtype, orig_shape=tuple(b._qdata.shape), transposed=False),
        )
        return _rowwise_fp8_linear(a, untransposed, None, out_dtype=kwargs.get("out_dtype"))
    return torch.mm(*dequantize_args(args))


@register_layout_op(torch.ops.aten.addmm.default, RowWiseFP8LayoutBase)
def _handle_rowwise_fp8_addmm(qt, args, kwargs):
    bias, a, b = args[0], args[1], args[2]
    if isinstance(b, QuantizedTensor) and getattr(b._params, "transposed", False):
        untransposed = QuantizedTensor(
            b._qdata,
            b._layout_cls,
            RowWiseFP8LayoutBase.Params(scale=b._params.scale, orig_dtype=b._params.orig_dtype, orig_shape=tuple(b._qdata.shape), transposed=False),
        )
        return _rowwise_fp8_linear(a, untransposed, bias, out_dtype=kwargs.get("out_dtype"))
    return torch.addmm(*dequantize_args(args))


@register_layout_op(torch.ops.aten.t.default, RowWiseFP8LayoutBase)
def _handle_rowwise_fp8_t(qt, args, kwargs):
    t = args[0]
    if not isinstance(t, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)
    old = t._params
    new_params = RowWiseFP8LayoutBase.Params(
        scale=old.scale,
        orig_dtype=old.orig_dtype,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not getattr(old, "transposed", False),
    )
    return QuantizedTensor(t._qdata, t._layout_cls, new_params)


# region Registry


register_layout_class("TensorCoreFP8Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("RowWiseFP8E4M3Layout", RowWiseFP8E4M3Layout)
register_layout_class("RowWiseFP8E5M2Layout", RowWiseFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout_)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout_)
register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
register_layout_class("TensorCoreConvRotW4A4Layout", TensorCoreConvRotW4A4Layout)
register_layout_class("AsymW4A8Int8Layout", AsymW4A8Int8Layout)


QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "float8_e4m3fn_rowwise": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "RowWiseFP8E4M3Layout",
        "quantize_input": False,
    },
    "float8_e5m2_rowwise": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "RowWiseFP8E5M2Layout",
        "quantize_input": False,
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
    "mxfp8": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreMXFP8Layout",
        "group_size": 32,
    },
    "int8_tensorwise": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorWiseINT8Layout",
        "quantize_input": False,
    },
    "convrot_w4a4": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorCoreConvRotW4A4Layout",
        "quantize_input": False,
    },
    "asym_w4a8_int8": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "AsymW4A8Int8Layout",
        "quantize_input": False,
    },
    "w6a8_int8": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "AsymW4A8Int8Layout",
        "quantize_input": False,
    },
}
