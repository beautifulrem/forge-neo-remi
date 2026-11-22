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


def gguf_weight(shape, seed=0, *, baked=True):
    weight = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    raw = torch.from_numpy(quants.quantize(weight.reshape(shape[0], -1), GGMLQuantizationType.Q8_0))
    param = ParameterGGUF(raw, tensor_type=GGMLQuantizationType.Q8_0, tensor_shape=torch.Size(shape))
    if baked:
        param.gguf_cls.bake(param)
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


LAYER_SHAPE = {"linear": (4, 32), "conv2d": (4, 32, 1, 1), "embedding": (16, 32)}


def _forward_and_reference(kind: str, layer: torch.nn.Module, dense: torch.Tensor, bias: torch.Tensor | None, device: torch.device):
    gen = torch.Generator().manual_seed(42)
    with torch.inference_mode():
        if kind == "linear":
            x = torch.randn(2, 32, generator=gen, dtype=torch.bfloat16).to(device)
            actual = layer(x).float().cpu()
            expected = torch.nn.functional.linear(x.float().cpu(), dense.float(), None if bias is None else bias.float().cpu())
        elif kind == "conv2d":
            x = torch.randn(2, 32, 3, 3, generator=gen, dtype=torch.bfloat16).to(device)
            actual = layer(x).float().cpu()
            expected = torch.nn.functional.conv2d(x.float().cpu(), dense.float(), None if bias is None else bias.float().cpu())
        else:
            idx = torch.tensor([[0, 3, 7, 15]], dtype=torch.long, device=device)
            actual = layer(idx).float().cpu()
            expected = torch.nn.functional.embedding(idx.cpu(), dense.float())
    return actual, expected


@pytest.mark.parametrize("kind", ["linear", "conv2d", "embedding"])
def test_gguf_layer_forward_matches_dense_across_first_and_second_load(kind):
    """Real Q8_0 forward pass through the real loader lifecycle (unbaked load ->
    ``bake_gguf_model`` -> forward -> second unbaked load -> re-bake -> forward)
    matches dense weights and reflects the newly loaded tensors."""
    model = tiny_model(lambda: make_layer(kind, inside=True))
    shape = LAYER_SHAPE[kind]

    param1, dense1 = gguf_weight(shape, seed=10, baked=False)
    bias1 = None if kind == "embedding" else torch.linspace(-0.2, 0.2, shape[0], dtype=torch.bfloat16)
    sd1 = {"layer.weight": param1} if bias1 is None else {"layer.weight": param1, "layer.bias": bias1}
    load(model, sd1)
    memory_management.bake_gguf_model(model)

    out1, ref1 = _forward_and_reference(kind, model.layer, dense1, bias1, CPU)
    assert torch.allclose(out1, ref1, atol=0.15, rtol=0.05)

    param2, dense2 = gguf_weight(shape, seed=20, baked=False)
    bias2 = None if kind == "embedding" else torch.linspace(0.5, 1.0, shape[0], dtype=torch.bfloat16)
    sd2 = {"layer.weight": param2} if bias2 is None else {"layer.weight": param2, "layer.bias": bias2}
    load(model, sd2)
    model.gguf_baked = False
    memory_management.bake_gguf_model(model)

    out2, ref2 = _forward_and_reference(kind, model.layer, dense2, bias2, CPU)
    assert torch.allclose(out2, ref2, atol=0.15, rtol=0.05)
    assert not torch.allclose(out1, out2, atol=1e-2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("kind", ["linear", "conv2d", "embedding"])
def test_gguf_layer_forward_on_mps_across_first_and_second_load(kind, monkeypatch):
    """Same Q8_0 first-load + second-load forward verification on Apple Silicon MPS."""
    mps = torch.device("mps")
    monkeypatch.setattr(memory_management, "get_torch_device", lambda: mps)

    with using_forge_operations(device=mps, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype="gguf"):
        if kind == "linear":
            layer = torch.nn.Linear(32, 4)
        elif kind == "conv2d":
            layer = torch.nn.Conv2d(32, 4, 1)
        else:
            layer = torch.nn.Embedding(16, 32)
    model = tiny_model(lambda: layer)
    shape = LAYER_SHAPE[kind]

    param1, dense1 = gguf_weight(shape, seed=30, baked=False)
    bias1 = None if kind == "embedding" else torch.zeros(shape[0], dtype=torch.bfloat16)
    load(model, {"layer.weight": param1} if bias1 is None else {"layer.weight": param1, "layer.bias": bias1})
    memory_management.bake_gguf_model(model)
    model.to(mps)
    out1, ref1 = _forward_and_reference(kind, model.layer, dense1, bias1, mps)
    assert torch.allclose(out1, ref1, atol=0.15, rtol=0.05)

    param2, dense2 = gguf_weight(shape, seed=40, baked=False)
    bias2 = None if kind == "embedding" else torch.ones(shape[0], dtype=torch.bfloat16)
    load(model, {"layer.weight": param2} if bias2 is None else {"layer.weight": param2, "layer.bias": bias2})
    model.gguf_baked = False
    memory_management.bake_gguf_model(model)
    model.to(mps)
    out2, ref2 = _forward_and_reference(kind, model.layer, dense2, bias2, mps)
    assert torch.allclose(out2, ref2, atol=0.15, rtol=0.05)
    assert not torch.allclose(out1, out2, atol=1e-2)
