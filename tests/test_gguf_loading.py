"""GGUF weights load through `backend.state_dict.load_state_dict`.

Upstream (neo-2.33) keeps `ParameterGGUF` tensors quantized when they land in
`ForgeOperationsGGUF` layers — including on a second load — and dequantizes
them for ordinary torch layers. These tests pin that behaviour on CPU with
real Q8_0 `ParameterGGUF` weights: no CUDA, no full checkpoints.
"""
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, quants

from backend import memory_management, state_dict
from backend.operations import ForgeOperationsGGUF, using_forge_operations
from backend.operations_gguf import ParameterGGUF

CPU = torch.device("cpu")


def gguf_weight(shape, seed=0):
    weight = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    raw = torch.from_numpy(quants.quantize(weight.reshape(shape[0], -1), GGMLQuantizationType.Q8_0))
    param = ParameterGGUF(raw, tensor_type=GGMLQuantizationType.Q8_0, tensor_shape=torch.Size(shape))
    param.gguf_cls.bake(param)  # what the loader does before the weight is used
    return param, torch.from_numpy(weight)


def tiny_model(layer_factory):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer_factory()

    return Model()


def load(model, sd):
    state_dict.load_state_dict(model, dict(sd))


@pytest.fixture(autouse=True)
def no_nvfp4(monkeypatch):
    # the plain dequantize path, not the TensorCore nvfp4 repack (CUDA only)
    monkeypatch.setattr(memory_management, "supports_nvfp4_compute", lambda device=None: False)
    monkeypatch.setattr(memory_management, "get_torch_device", lambda: CPU)


def make_layer(kind, inside):
    if inside:
        with using_forge_operations(device=CPU, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype="gguf"):
            if kind == "linear":
                return torch.nn.Linear(32, 4)
            if kind == "conv2d":
                return torch.nn.Conv2d(32, 4, 1)
            return torch.nn.Embedding(16, 32)
    if kind == "linear":
        return torch.nn.Linear(32, 4)
    if kind == "conv2d":
        return torch.nn.Conv2d(32, 4, 1)
    return torch.nn.Embedding(16, 32)


LAYER_SD = {
    "linear": lambda: {"layer.weight": gguf_weight((4, 32))[0], "layer.bias": torch.zeros(4, dtype=torch.bfloat16)},
    "conv2d": lambda: {"layer.weight": gguf_weight((4, 32, 1, 1))[0], "layer.bias": torch.zeros(4, dtype=torch.bfloat16)},
    "embedding": lambda: {"layer.weight": gguf_weight((16, 32))[0]},
}

GGUF_CLASS = {"linear": ForgeOperationsGGUF.Linear, "conv2d": ForgeOperationsGGUF.Conv2d, "embedding": ForgeOperationsGGUF.Embedding}


@pytest.mark.parametrize("kind", ["linear", "conv2d", "embedding"])
def test_gguf_layer_keeps_the_quantized_parameter_on_first_and_second_load(kind):
    model = tiny_model(lambda: make_layer(kind, inside=True))
    assert isinstance(model.layer, GGUF_CLASS[kind])

    load(model, LAYER_SD[kind]())
    weight = model.layer.weight
    assert isinstance(weight, ParameterGGUF) and weight.gguf_cls.__name__ == "Q8_0"
    assert weight.computation_dtype is torch.bfloat16
    if kind != "embedding":
        assert isinstance(model.layer.bias, torch.nn.Parameter) and model.layer.bias.computation_dtype is torch.bfloat16

    # weight (and bias) are registered parameters of the layer, none trainable
    names = {name for name, _ in model.layer.named_parameters(recurse=False)}
    assert names == ({"weight", "bias"} if kind != "embedding" else {"weight"})
    assert all(not p.requires_grad for p in model.parameters())

    # a second load into the same GGUF-native layer stays quantized — and
    # carries the new weights, not the old ones
    other, _ = gguf_weight(tuple(model.layer.weight.shape), seed=1)
    sd = {"layer.weight": other}
    if kind != "embedding":
        sd["layer.bias"] = torch.ones(model.layer.bias.shape, dtype=torch.bfloat16)
    load(model, sd)
    weight = model.layer.weight
    assert isinstance(weight, ParameterGGUF) and weight.gguf_cls.__name__ == "Q8_0"
    assert tuple(weight.shape) == tuple(other.shape) and weight.computation_dtype is torch.bfloat16
    assert torch.equal(weight.as_subclass(torch.Tensor), other.as_subclass(torch.Tensor))
    if kind != "embedding":
        assert torch.equal(model.layer.bias, torch.ones_like(model.layer.bias))


@pytest.mark.parametrize("kind", ["linear", "conv2d", "embedding"])
def test_plain_layer_dequantizes_a_gguf_entry(kind):
    model = tiny_model(lambda: make_layer(kind, inside=False))
    param, dense = gguf_weight(tuple(model.layer.weight.shape))
    sd = {"layer.weight": param}
    if kind != "embedding":
        sd["layer.bias"] = torch.zeros(model.layer.bias.shape, dtype=torch.float32)

    load(model, sd)
    weight = model.layer.weight
    assert not isinstance(weight, ParameterGGUF) and getattr(weight, "gguf_cls", None) is None
    assert torch.allclose(weight.float(), dense, atol=0.05)


def test_meta_linear_gets_the_dequantized_weight_at_the_param_dtype():
    with torch.device("meta"):
        model = tiny_model(lambda: torch.nn.Linear(32, 4, dtype=torch.bfloat16))
    param, dense = gguf_weight((4, 32))
    load(model, {"layer.weight": param, "layer.bias": torch.zeros(4)})
    weight = model.layer.weight
    assert not weight.is_meta and weight.dtype is torch.bfloat16
    assert torch.allclose(weight.float(), dense, atol=0.05)
