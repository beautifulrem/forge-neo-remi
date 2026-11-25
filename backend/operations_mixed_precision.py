# https://github.com/Comfy-Org/ComfyUI/blob/v0.27.0/comfy/ops.py#L1163

import json
import os
import threading
import weakref
from collections import OrderedDict

import torch

from backend.args import args
from backend.memory_management import (
    cast_to_device,
    get_free_memory,
    is_device_mps,
    logger,
    minimum_inference_memory,
    register_cache_clear_hook,
)

from .operations import (
    ForgeOperations,
    ForgeWeights,
    main_stream_worker,
    weights_manual_cast,
)
from .quant_ops import (  # noqa
    QUANT_ALGOS,
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorWiseINT8Layout,
    ck,
    get_layout_class,
)

_GROUPED_INT8_FORMATS = {"asym_w4a8_int8": 4, "w6a8_int8": 6}

_COMPILE_UNSAFE_LAYOUTS = {"TensorCoreConvRotW4A4Layout", "AsymW4A8Int8Layout"}


_MPS_DEQUANT_CACHE_SAFETY_MARGIN = 512 * 1024 * 1024
_MPS_DEQUANT_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024
_mps_cache_override_warned = False


def _warn_mps_cache_override(value: str, consequence: str) -> None:
    """Once per process: the budget is read for every cached weight."""
    global _mps_cache_override_warned
    if not _mps_cache_override_warned:
        _mps_cache_override_warned = True
        logger.warning("REMI_MPS_DEQUANT_CACHE_MB=%r %s", value, consequence)


def _mps_dequant_cache_budget_bytes() -> int:
    """Safe byte budget for caching dequantized resident weights on Apple Silicon MPS.

    Respects ``REMI_MPS_DEQUANT_CACHE_MB`` (``0`` disables caching; a value that
    is not a whole number is ignored and a negative one means ``0``, each with a
    warning, once). Otherwise
    uses at most 25% of unified-memory surplus above ``minimum_inference_memory()``
    plus a 512 MiB safety floor, capped at 2 GiB.
    """
    override = os.environ.get("REMI_MPS_DEQUANT_CACHE_MB")
    if override is not None:
        try:
            mb = int(override)
        except ValueError:
            _warn_mps_cache_override(override, "is not a whole number of MB: the automatic budget is used")
        else:
            if mb < 0:
                _warn_mps_cache_override(override, "is negative: the cache is off")
            return max(0, mb) * 1024 * 1024
    try:
        free_bytes = int(get_free_memory(torch.device("mps")))
        floor = int(minimum_inference_memory()) + _MPS_DEQUANT_CACHE_SAFETY_MARGIN
    except Exception:
        return 0
    surplus = free_bytes - floor
    if surplus <= 0:
        return 0
    return min(int(surplus * 0.25), _MPS_DEQUANT_CACHE_MAX_BYTES)


class _MPSDequantCache:
    """Thread-safe, headroom-bounded LRU cache for resident MPS dequantized weights."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._finalizers: dict[int, weakref.finalize] = {}
        self._current_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    @staticmethod
    def _make_key(module: torch.nn.Module, weight: QuantizedTensor, target_dtype: torch.dtype) -> tuple:
        qdata = weight._qdata
        params = weight._params
        scale = getattr(params, "scale", None)
        scale_ptr = scale.data_ptr() if isinstance(scale, torch.Tensor) else None
        block_scale = getattr(params, "block_scale", None)
        block_scale_ptr = block_scale.data_ptr() if isinstance(block_scale, torch.Tensor) else None
        transposed = bool(getattr(params, "transposed", False))
        return (id(module), id(qdata), qdata.data_ptr(), scale_ptr, block_scale_ptr, transposed, target_dtype)

    def _drop_module_id(self, module_id: int) -> int:
        freed = 0
        with self._lock:
            stale = [k for k in self._entries if k[0] == module_id]
            for k in stale:
                tensor = self._entries.pop(k)
                nbytes = tensor.nbytes
                self._current_bytes = max(0, self._current_bytes - nbytes)
                freed += nbytes
            self._finalizers.pop(module_id, None)
        return freed

    def invalidate_module(self, module: torch.nn.Module) -> int:
        return self._drop_module_id(id(module))

    def clear(self) -> int:
        with self._lock:
            freed = self._current_bytes
            self._entries.clear()
            for fin in self._finalizers.values():
                fin.detach()
            self._finalizers.clear()
            self._current_bytes = 0
        return freed

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._current_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }

    def get_or_dequantize(self, module: torch.nn.Module, weight: QuantizedTensor, target_dtype: torch.dtype) -> torch.Tensor | None:
        key = self._make_key(module, weight, target_dtype)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return cached

        budget = _mps_dequant_cache_budget_bytes()
        entry_bytes = weight.numel() * target_dtype.itemsize
        if budget <= 0 or entry_bytes > budget:
            return None

        dequant = weight.dequantize()
        if dequant.dtype != target_dtype:
            dequant = dequant.to(dtype=target_dtype)
        actual_bytes = dequant.nbytes
        if actual_bytes > budget:
            return dequant

        module_id = id(module)
        with self._lock:
            while self._entries and (self._current_bytes + actual_bytes) > budget:
                _, evicted = self._entries.popitem(last=False)
                self._current_bytes = max(0, self._current_bytes - evicted.nbytes)
                self._evictions += 1
            if key in self._entries:
                old = self._entries.pop(key)
                self._current_bytes = max(0, self._current_bytes - old.nbytes)
            self._entries[key] = dequant
            self._current_bytes += actual_bytes
            self._misses += 1
            if module_id not in self._finalizers:
                try:
                    fin = weakref.finalize(module, self._drop_module_id, module_id)
                    fin.atexit = False
                    self._finalizers[module_id] = fin
                except TypeError:
                    pass
        return dequant


_mps_dequant_cache = _MPSDequantCache()
register_cache_clear_hook(_mps_dequant_cache.clear)


def clear_mps_dequant_cache() -> int:
    """Drop all cached MPS dequantized weights and return freed bytes."""
    return _mps_dequant_cache.clear()


def mps_dequant_cache_stats() -> dict[str, int]:
    """Return hit/miss/eviction and byte-usage counters for the MPS dequantization cache."""
    return _mps_dequant_cache.stats()


def _convrot_rotate_input(module: torch.nn.Linear, x: torch.Tensor) -> bool:
    weight = module.weight
    return module._full_precision_mm and isinstance(weight, QuantizedTensor) and weight._layout_cls == "TensorWiseINT8Layout" and getattr(weight._params, "convrot", False) and not getattr(weight._params, "transposed", False) and not isinstance(x, QuantizedTensor) and x.numel() // max(x.shape[-1], 1) < module.out_features and len(module.weight_function) == 0 and not getattr(module, "forge_force_cast_weights", False)


def _convrot_linear(x: torch.Tensor, weight: QuantizedTensor, bias: torch.Tensor) -> torch.Tensor:
    qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    group_size = weight._params.convrot_groupsize
    x = ck.tensor.int8_utils._rotate_activation(x, ck.tensor.int8_utils._build_hadamard(group_size, device=x.device, dtype=x.dtype), group_size)
    scale = scale.to(device=x.device, dtype=x.dtype).reshape(-1)
    if x.dtype == torch.float16:
        w = qdata.to(dtype=x.dtype)
        w.mul_(scale.reshape(-1, 1))
        return torch.nn.functional.linear(x, w, bias)
    out = torch.nn.functional.linear(x, qdata.to(dtype=x.dtype)) * scale
    return out if bias is None else out + bias


@torch.compiler.disable
def _run_eager(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _quantized_apply(module: torch.nn.Module, fn, recurse=True):
    _mps_dequant_cache.invalidate_module(module)
    if recurse:
        for child in module.children():
            child._apply(fn)
    for key, param in module._parameters.items():
        if param is None:
            continue
        p: torch.Tensor = fn(param)
        try:
            module.register_parameter(key, torch.nn.Parameter(p, requires_grad=False))
        except RuntimeError:
            module.register_parameter(key, torch.nn.Parameter(p.clone(), requires_grad=False))
    for key, buf in module._buffers.items():
        if buf is not None:
            module._buffers[key] = fn(buf)
    return module


def _load_quantized_module(module: torch.nn.Module, super_load, state_dict: dict[str, torch.Tensor], prefix: str, local_metadata, strict, missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    disabled_formats = module._disabled_formats
    layer_name = prefix.rstrip(".")

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
    if layer_conf is not None:
        layer_conf = json.loads(layer_conf.numpy().tobytes())

    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
    else:
        module.quant_format = layer_conf.get("format", None)
        module._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
        if not module._full_precision_mm:
            module._full_precision_mm = module._full_precision_mm_config
        if module.quant_format in disabled_formats:
            module._full_precision_mm = True
        if module.quant_format is None:
            raise ValueError(f"Unknown quantization format for layer {layer_name}")

        qconfig = QUANT_ALGOS[module.quant_format]
        module.layout_type = qconfig["comfy_tensor_layout"]
        layout_cls = get_layout_class(module.layout_type)

        if module.quant_format in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fn_rowwise", "float8_e5m2_rowwise"):
            scale = pop_scale("weight_scale")
            if module.quant_format.endswith("_rowwise"):
                if scale is None:
                    raise ValueError(f"Missing rowwise FP8 weight scale for layer {layer_name}")
                module._per_row = scale.dim() == 2 and scale.shape[1] == 1
            scales = {"scale": scale}
        elif module.quant_format == "mxfp8":
            bs = pop_scale("weight_scale", torch.float8_e8m0fnu)
            if bs is None:
                raise ValueError(f"Missing MXFP8 block scales for layer {layer_name}")
            scales = {"scale": bs}
        elif module.quant_format == "nvfp4":
            ts = pop_scale("weight_scale_2")
            bs = pop_scale("weight_scale", torch.float8_e4m3fn)
            if ts is None or bs is None:
                raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")
            scales = {"scale": ts, "block_scale": bs}
        elif module.quant_format == "int8_tensorwise":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")
            module._per_row = scale.dim() == 2 and scale.shape[1] == 1
            scales = {"scale": scale}
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            if layer_conf.get("convrot", params_conf.get("convrot", False)):
                scales["convrot"] = True
                scales["convrot_groupsize"] = int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
        elif module.quant_format == "convrot_w4a4":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing ConvRot W4A4 weight scale for layer {layer_name}")
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "convrot_groupsize": int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
                "quant_group_size": 64,
                "linear_dtype": "int8" if args.force_w4a8 else layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
            }
        elif module.quant_format in _GROUPED_INT8_FORMATS:
            bits = _GROUPED_INT8_FORMATS[module.quant_format]
            if weight.shape[1] * 8 != module._orig_shape[1] * bits:
                raise ValueError(f'Invalid Layer "{layer_name}" ([{module.quant_format}] packed weight width {weight.shape[1]} does not match K={module._orig_shape[1]} at {bits} bits)')
            scale = pop_scale("weight_s_rel")
            if scale is None:
                raise ValueError(f"Missing {module.quant_format} group scale (weight_s_rel) for layer {layer_name}")
            if scale.dtype == torch.uint8:
                scale = scale.view(torch.float8_e4m3fn)
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "s_channel": pop_scale("weight_s_channel"),
                "codebook": pop_scale("weight_codebook"),
                "group_size": int(layer_conf.get("group_size", params_conf.get("group_size", 16))),
                "convrot_groupsize": int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
            }
        else:
            raise ValueError(f"Unsupported quantization format: {module.quant_format}")

        params = layout_cls.Params(**scales, orig_dtype=compute_dtype, orig_shape=module._orig_shape)
        module.weight = torch.nn.Parameter(
            QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), module.layout_type, params),
            requires_grad=False,
        )

        if load_extra_params:
            for param_name in qconfig["parameters"]:
                if param_name in {"weight_scale", "weight_scale_2"}:
                    continue
                param_key = f"{prefix}{param_name}"
                _v = state_dict.pop(param_key, None)
                if _v is None:
                    continue
                module.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                manually_loaded_keys.append(param_key)

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)


def _quantized_weight_state_dict(module: torch.nn.Module, sd: dict[str, torch.Tensor], prefix: str, extra_quant_conf: dict = None, extra_quant_params: tuple[str] = ()):
    if not hasattr(module, "weight"):
        logger.warning(f"uninitialized op {prefix}")
        return sd
    bias = getattr(module, "bias", None)
    if bias is not None:
        sd[f"{prefix}bias"] = bias
    if module.weight is None:
        return sd

    if not isinstance(module.weight, QuantizedTensor):
        sd[f"{prefix}weight"] = module.weight
    else:
        sd.update(module.weight.state_dict(f"{prefix}weight"))
        quant_conf = {"format": module.quant_format}
        if getattr(module, "_full_precision_mm_config", False):
            quant_conf["full_precision_matrix_mult"] = True
        params = getattr(module.weight, "_params", None)
        if module.quant_format == "int8_tensorwise" and getattr(params, "convrot", False):
            quant_conf["convrot"] = True
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
        elif module.quant_format == "convrot_w4a4":
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
            linear_dtype = getattr(params, "linear_dtype", "int4")
            if linear_dtype != "int4":
                quant_conf["linear_dtype"] = linear_dtype
        elif module.quant_format in _GROUPED_INT8_FORMATS:
            quant_conf["group_size"] = getattr(params, "group_size", 16)
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
        if extra_quant_conf:
            quant_conf.update(extra_quant_conf)
        sd[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8)
        for name in extra_quant_params:
            value = getattr(module, name, None)
            if value is not None:
                sd[f"{prefix}{name}"] = value

    return sd


def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
    class MixedPrecisionOps(ForgeOperations):
        _quant_config = quant_config
        _compute_dtype = compute_dtype
        _full_precision_mm = full_precision_mm
        _disabled = disabled

        class Linear(torch.nn.Module, ForgeWeights):
            _disabled_formats = disabled

            def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
                super().__init__()

                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}

                self.in_features = in_features
                self.out_features = out_features
                self._orig_shape = (out_features, in_features)
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False

            def reset_parameters(self):
                return None

            def _load_from_state_dict(self, *args):
                _mps_dequant_cache.invalidate_module(self)
                _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=True)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale",))

            def forward(self, input, *args, **kwargs):
                if getattr(self, "layout_type", None) in _COMPILE_UNSAFE_LAYOUTS:
                    return _run_eager(self._forward, input, *args, **kwargs)
                return self._forward(input, *args, **kwargs)

            def _forward(self, input, *args, **kwargs):
                assert not input.requires_grad

                weight_param = self.weight
                if (
                    self._full_precision_mm
                    and isinstance(weight_param, QuantizedTensor)
                    and not isinstance(input, QuantizedTensor)
                    and is_device_mps(weight_param.device)
                    and is_device_mps(input.device)
                    and len(self.weight_function) == 0
                    and len(self.bias_function) == 0
                    and not getattr(self, "forge_force_cast_weights", False)
                ):
                    bias = self.bias
                    if bias is not None:
                        bias = cast_to_device(bias, input.device, input.dtype)
                    cached_weight = _mps_dequant_cache.get_or_dequantize(self, weight_param, input.dtype)
                    if cached_weight is not None:
                        return torch.nn.functional.linear(input, cached_weight, bias)
                    if _convrot_rotate_input(self, input):
                        return _convrot_linear(input, weight_param, bias)
                    dequant = weight_param.dequantize()
                    if dequant.dtype != input.dtype:
                        dequant = dequant.to(dtype=input.dtype)
                    return torch.nn.functional.linear(input, dequant, bias)

                input_shape = input.shape
                reshaped_nd = False

                _use_quantized = (
                    getattr(self, "layout_type", None) is not None
                    and not isinstance(input, QuantizedTensor)
                    and not self._full_precision_mm
                    and not getattr(self, "forge_force_cast_weights", False)
                    and len(self.weight_function) == 0
                    and len(self.bias_function) == 0
                )
                quantize_input = QUANT_ALGOS.get(getattr(self, "quant_format", None), {}).get("quantize_input", True) if _use_quantized else False

                if _use_quantized and quantize_input:
                    input_reshaped = input.reshape(-1, input_shape[-1]) if input.ndim >= 3 else input

                    if input_reshaped.ndim == 2:
                        reshaped_nd = input.ndim >= 3
                        scale = getattr(self, "input_scale", None)
                        if scale is not None:
                            scale = cast_to_device(scale, input.device, None)
                        input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)

                weight_only_quant = _use_quantized and not quantize_input and isinstance(weight_param, QuantizedTensor)

                if not weight_only_quant and _convrot_rotate_input(self, input):
                    weight, bias, signal = weights_manual_cast(self, x=input, dtype=weight_param.dtype)
                    with main_stream_worker(weight, bias, signal):
                        if isinstance(weight, QuantizedTensor):
                            return _convrot_linear(input, weight, bias)
                        return torch.nn.functional.linear(input, weight.to(dtype=input.dtype), bias)

                if weight_only_quant:
                    weight, bias, signal = weights_manual_cast(self, x=None, dtype=weight_param.dtype, device=input.device, bias_dtype=input.dtype)
                    weight = weight.to(dtype=input.dtype)
                else:
                    weight, bias, signal = weights_manual_cast(self, x=input)

                with main_stream_worker(weight, bias, signal):
                    if self._full_precision_mm and isinstance(weight, QuantizedTensor):
                        weight = weight.dequantize()
                    output = torch.nn.functional.linear(input, weight, bias)

                if reshaped_nd:
                    output = output.reshape((*input_shape[:-1], self.weight.shape[0]))

                return output

            def convert_weight(self, weight, inplace=False, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                _mps_dequant_cache.invalidate_module(self)
                if getattr(self, "layout_type", None) is not None:
                    weight = self.weight.requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed, inplace_ops=True).to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def _apply(self, fn, recurse=True):
                return _quantized_apply(self, fn, recurse)

        class Embedding(ForgeOperations.Embedding):
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = f"{prefix}weight"
                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                quant_format = layer_conf.get("format") if layer_conf is not None else None
                manually_loaded_keys = []

                if quant_format in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fn_rowwise", "float8_e5m2_rowwise", "int8_tensorwise") and weight_key in state_dict:
                    self.quant_format = quant_format
                    qconfig = QUANT_ALGOS[quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)
                    weight = state_dict.pop(weight_key)
                    manually_loaded_keys.append(weight_key)

                    scale_key = f"{prefix}weight_scale"
                    scale = state_dict.pop(scale_key, None)
                    if scale is not None:
                        scale = scale.float()
                        manually_loaded_keys.append(scale_key)
                    self._scale_is_unit = bool(scale is None or (scale.numel() == 1 and scale.device.type == "cpu" and scale.item() == 1.0))

                    extra = {}
                    if quant_format == "int8_tensorwise" and layer_conf.get("convrot", False):
                        extra["convrot"] = True
                        extra["convrot_groupsize"] = int(layer_conf.get("convrot_groupsize", 256))

                    params = layout_cls.Params(
                        scale=scale if scale is not None else torch.ones((), dtype=torch.float32),
                        orig_dtype=MixedPrecisionOps._compute_dtype,
                        orig_shape=(self.num_embeddings, self.embedding_dim),
                        **extra,
                    )
                    self.weight = torch.nn.Parameter(QuantizedTensor(weight.to(dtype=qconfig["storage_t"]), qconfig["comfy_tensor_layout"], params), requires_grad=False)
                elif layer_conf is not None:
                    state_dict[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(layer_conf).encode("utf-8")), dtype=torch.uint8)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                for k in manually_loaded_keys:
                    if k in missing_keys:
                        missing_keys.remove(k)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix)

            def forward(self, input):
                if getattr(self, "layout_type", None) in _COMPILE_UNSAFE_LAYOUTS:
                    return _run_eager(self._forward, input)
                return self._forward(input)

            def _forward(self, input):
                weight = self.weight

                if isinstance(weight, QuantizedTensor) and len(self.weight_function) == 0:
                    qdata, _, signal = weights_manual_cast(self, device=input.device, dtype=weight.dtype)
                    if isinstance(qdata, QuantizedTensor):
                        params = qdata._params
                        scale = params.scale
                        qdata = qdata._qdata
                    else:
                        params = weight._params
                        scale = None

                    with main_stream_worker(qdata, None, signal):
                        if self.quant_format == "int8_tensorwise":
                            return get_layout_class(self.layout_type).dequantize_embedding(qdata, params, input)

                        x = torch.nn.functional.embedding(input, qdata, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

                    target_dtype = weight._params.orig_dtype
                    x = x.to(dtype=target_dtype)
                    if scale is not None:
                        if scale.dim() >= 2 and scale.shape[0] == self.num_embeddings:
                            x = x * torch.nn.functional.embedding(input, scale.to(device=x.device, dtype=target_dtype))
                        elif not (scale.numel() == 1 and (getattr(self, "_scale_is_unit", False) or (scale.device.type == "cpu" and scale.item() == 1.0))):
                            x = x * scale.to(device=x.device, dtype=target_dtype)

                    return x

                return super().forward(input)

    return MixedPrecisionOps


# region ops

import torch.nn.functional as F


def _swiglu_eager(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def _situ_eager(x: torch.Tensor) -> torch.Tensor:
    hidden, gate = x.chunk(2, dim=-1)
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


INPUT_ACT_EAGER = {
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "swiglu": _swiglu_eager,
    "situ": _situ_eager,
}


def _eager_input_act(x: torch.Tensor, input_act: str, act_weight: torch.Tensor = None, act_eps: float = 0.0) -> torch.Tensor:
    if input_act is None:
        return x
    if input_act == "rms_norm":
        return F.rms_norm(x, act_weight.shape, cast_to_device(act_weight, x.device, x.dtype), act_eps)
    return INPUT_ACT_EAGER[input_act](x)


def _fp16_linear_wanted(x: torch.Tensor) -> bool:
    return x.dtype == torch.float16 and x.is_cuda and getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)


@torch.compiler.disable
def linear_input_act(linear: torch.nn.Linear, x: torch.Tensor, input_act: str, act_weight: torch.Tensor = None, act_eps: float = 0.0, residual: torch.Tensor = None, residual_scale: float = None) -> torch.Tensor:
    """https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ops.py#L976"""

    def _residual_out(out):
        if residual is None:
            return out
        return torch.addcmul(residual, out, residual_scale)

    weight = linear.weight
    full_precision_mm = getattr(linear, "_full_precision_mm", False)

    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout" or getattr(weight._params, "transposed", False) or full_precision_mm:
        if not isinstance(weight, QuantizedTensor) and not full_precision_mm and _fp16_linear_wanted(x) and getattr(weight, "gguf_cls", None) is None:
            weight, bias, offload_stream = weights_manual_cast(linear, x)
            with main_stream_worker(weight, bias, offload_stream):
                return ck.fp16_linear(_eager_input_act(x, input_act, act_weight, act_eps), weight, bias, residual=residual, residual_scale=residual_scale)

        return _residual_out(linear(_eager_input_act(x, input_act, act_weight, act_eps)))

    weight, bias, offload_stream = weights_manual_cast(linear, x)
    with main_stream_worker(weight, bias, offload_stream):
        if not isinstance(weight, QuantizedTensor):
            return _residual_out(torch.nn.functional.linear(_eager_input_act(x, input_act, act_weight, act_eps), weight, bias))

        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        return ck.int8_linear(
            x,
            qdata,
            scale,
            bias,
            x.dtype,
            convrot=getattr(weight._params, "convrot", False),
            convrot_groupsize=getattr(weight._params, "convrot_groupsize", 256),
            input_act=input_act,
            input_act_weight=act_weight,
            input_act_eps=act_eps,
            residual=residual,
            residual_scale=residual_scale,
        )
