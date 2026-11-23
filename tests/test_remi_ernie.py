"""Unit tests for ERNIE-Image (8B Apache-2.0) capability registration,
Diffusers loading, quantization, LoRA key mapping, Spectrum pre-head seam,
per-block compile, and engine caching.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from backend import memory_management, operations
from backend.diffusion_engine.ernie import ErnieImage
from backend.nn import dit_capabilities_ext
from backend.nn.anima_quant import (
    keep_sensitive_weights_precise,
    quantize_int8_convrot_state_dict,
    scale_fp8_state_dict,
    should_keep_sensitive_weights_precise,
    should_quantize_on_load,
)
from backend.nn.dit_registry import (
    capability_for_compile_preset,
    capability_for_dit_cls,
    capability_for_state_dict,
    get_dit_capability,
    match_spectrum_capability,
    resolve_spectrum_capture_module,
    unregister_dit_capability,
)
from backend.nn.ernie import ErnieImageModel
from backend.nn.ernie_diffusers import (
    ERNIE_LICENSE,
    ernie_lora_key_map,
    ernie_workflow_advice,
    fast_forward_ernie,
    is_ernie_diffusers_dir,
    is_ernie_state_dict,
    is_sharded_ernie_transformer_dir,
    load_ernie_diffusers_dir,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _register_ext_caps():
    preexisting = {name for name in dit_capabilities_ext.EXTENDED_DIT_NAMES if get_dit_capability(name) is not None}
    dit_capabilities_ext.register_extended_dit_capabilities()
    try:
        yield
    finally:
        for name in dit_capabilities_ext.EXTENDED_DIT_NAMES:
            if name not in preexisting:
                unregister_dit_capability(name)


def _tiny_ernie(*, hidden_size: int = 32, num_heads: int = 2, num_layers: int = 2) -> ErnieImageModel:
    torch.manual_seed(0)
    head_dim = hidden_size // num_heads
    a0 = head_dim // 4
    a1 = (head_dim - a0) // 2
    a2 = head_dim - a0 - a1
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        m = ErnieImageModel(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            ffn_hidden_size=hidden_size * 2,
            in_channels=8,
            out_channels=8,
            patch_size=1,
            text_in_dim=16,
            rope_theta=256,
            rope_axes_dim=(a0, a1, a2),
        ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.03)
    return m


def test_ernie_capability_quantization_and_sensitive_prefixes():
    dit_capabilities_ext.register_extended_dit_capabilities()
    cap = get_dit_capability("ernie")
    assert cap is not None
    assert capability_for_dit_cls("ErnieImageTransformer2DModel") is cap
    assert capability_for_compile_preset("ERNIE-Image per-block") is cap
    assert should_keep_sensitive_weights_precise("ErnieImageTransformer2DModel", torch.float8_e4m3fn, torch.bfloat16) is True
    assert should_quantize_on_load("scaled_fp8", "ErnieImageTransformer2DModel", None, False, torch.bfloat16) is True
    assert should_quantize_on_load("int8_convrot", "ErnieImageTransformer2DModel", None, False, torch.bfloat16) is True

    wide = _tiny_ernie(hidden_size=256, num_heads=4, num_layers=1)
    fp8_model = _tiny_ernie(hidden_size=256, num_heads=4, num_layers=1).to(torch.float8_e4m3fn)
    kept = keep_sensitive_weights_precise(fp8_model, torch.bfloat16)
    assert kept > 0
    assert fp8_model.x_embedder.proj.weight.dtype == torch.bfloat16
    assert fp8_model.text_proj.weight.dtype == torch.bfloat16
    assert fp8_model.adaLN_modulation[1].weight.dtype == torch.bfloat16
    assert fp8_model.final_linear.weight.dtype == torch.bfloat16
    assert fp8_model.layers[0].self_attention.to_q.weight.dtype == torch.float8_e4m3fn

    bf16_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in wide.state_dict().items()}
    assert is_ernie_state_dict(bf16_sd) is True
    assert capability_for_state_dict(bf16_sd) is cap

    fp8_sd = dict(bf16_sd)
    assert scale_fp8_state_dict(fp8_sd, fp8_matmul=False) == 7
    assert fp8_sd["layers.0.self_attention.to_q.weight"].dtype == torch.float8_e4m3fn
    assert fp8_sd["layers.0.self_attention.to_q.weight_scale"].shape == (256, 1)
    assert "final_linear.comfy_quant" not in fp8_sd

    int8_sd = dict(bf16_sd)
    assert quantize_int8_convrot_state_dict(int8_sd) == 7
    assert int8_sd["layers.0.mlp.linear_fc2.weight"].dtype == torch.int8


def test_ernie_diffusers_dir_and_sharded_loading(tmp_path):
    model = _tiny_ernie(hidden_size=32, num_heads=2, num_layers=1)
    t_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in model.state_dict().items()}
    keys = sorted(t_sd.keys())
    half = len(keys) // 2

    pipe_dir = tmp_path / "ERNIE-Image"
    (pipe_dir / "transformer").mkdir(parents=True)
    (pipe_dir / "text_encoder").mkdir(parents=True)
    (pipe_dir / "vae").mkdir(parents=True)
    (pipe_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "ErnieImagePipeline", "transformer": ["diffusers", "ErnieImageTransformer2DModel"]}),
        encoding="utf-8",
    )
    s1 = "diffusion_pytorch_model-00001-of-00002.safetensors"
    s2 = "diffusion_pytorch_model-00002-of-00002.safetensors"
    save_file({k: t_sd[k] for k in keys[:half]}, str(pipe_dir / "transformer" / s1))
    save_file({k: t_sd[k] for k in keys[half:]}, str(pipe_dir / "transformer" / s2))
    (pipe_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {**dict.fromkeys(keys[:half], s1), **dict.fromkeys(keys[half:], s2)}}),
        encoding="utf-8",
    )
    save_file({"language_model.embed_tokens.weight": torch.zeros(8, 3072)}, str(pipe_dir / "text_encoder" / "model.safetensors"))
    save_file({"decoder.post_quant_conv.weight": torch.zeros(32, 32, 1, 1)}, str(pipe_dir / "vae" / "diffusion_pytorch_model.safetensors"))

    assert is_ernie_diffusers_dir(pipe_dir) is True
    assert is_sharded_ernie_transformer_dir(pipe_dir / "transformer") is True

    sd, meta = load_ernie_diffusers_dir(pipe_dir)
    assert meta["license"] == ERNIE_LICENSE
    assert "model.diffusion_model.layers.0.adaLN_sa_ln.weight" in sd
    assert "text_encoders.ministral3_3b.transformer.embed_tokens.weight" in sd
    assert "vae.post_quant_conv.weight" in sd


def test_ernie_spectrum_pre_head_compile_lora_and_engine_cache(monkeypatch):
    dit_capabilities_ext.register_extended_dit_capabilities()
    model = _tiny_ernie(hidden_size=32, num_heads=2, num_layers=2)

    assert match_spectrum_capability(model) is get_dit_capability("ernie")
    assert resolve_spectrum_capture_module(model) is model.final_norm

    captured = {}

    def hook(_mod, args):
        captured["feat"] = args[0].detach().clone()

    h = model.final_norm.register_forward_pre_hook(hook)
    try:
        x = torch.randn(1, 8, 4, 4)
        sigma = torch.tensor([500.0])
        ctx = torch.randn(1, 5, 16)
        with torch.inference_mode():
            full = model(x, sigma, ctx)
            predictor = SimpleNamespace(timestep=lambda s: s, calculate_denoised=lambda s, pred, inp: pred)
            fast = fast_forward_ernie(model, predictor, sigma, x, captured["feat"])
        torch.testing.assert_close(fast, full, atol=1e-5, rtol=1e-5)
    finally:
        h.remove()

    # Per-block compile resolves model.layers
    compile_root = str(REPO / "extensions-builtin" / "sd_forge_compile")
    if compile_root not in sys.path:
        sys.path.insert(0, compile_root)
    from anima_block_compile import _block_list

    assert _block_list(model) is model.layers

    # LoRA key map
    km = ernie_lora_key_map({f"diffusion_model.{k}": v for k, v in model.state_dict().items()})
    assert km["transformer.layers.0.self_attention.to_q"] == "diffusion_model.layers.0.self_attention.to_q.weight"
    assert km["lycoris_layers_0_mlp_linear_fc2"] == "diffusion_model.layers.0.mlp.linear_fc2.weight"

    # Engine prompt cache
    monkeypatch.setattr(memory_management, "load_model_gpu", lambda *_a, **_kw: None)
    eng = ErnieImage.__new__(ErnieImage)
    eng.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=object()))
    calls = []
    eng.text_processing_engine_ministral = lambda p: calls.append(list(p)) or [torch.ones(1, 4, 16)]
    eng._cond_cache = {}
    from collections import OrderedDict

    eng._cond_cache = OrderedDict()
    eng.ref_latents = []
    eng.ini_latent = None
    c1 = eng.get_learned_conditioning(["hello ernie"])
    c2 = eng.get_learned_conditioning(["hello ernie"])
    assert len(calls) == 1 and len(c1) == len(c2) == 1
    eng.clear_references()
    assert len(eng._cond_cache) == 0

    # Workflow advice
    p_turbo = SimpleNamespace(
        sd_model=SimpleNamespace(is_ernie_engine=True, sd_checkpoint_info=SimpleNamespace(name="ERNIE-Image-Turbo.safetensors")),
        prompt="a city skyline",
        steps=32,
        cfg_scale=4.0,
    )
    assert any("Turbo" in msg for msg in ernie_workflow_advice(p_turbo))
