"""Unit tests for Z-Image family upgrades (Base, Turbo, Omni/Edit) and
MOSS-Image / Kromoss 8-step DMD2 helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from backend import operations
from backend.args import dynamic_args
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
from backend.nn.lumina import NextDiT
from backend.nn.moss_diffusers import (
    is_dmd2_lora,
    is_kromoss_variant,
    kromoss_dmd2_workflow_advice,
    moss_lora_key_map,
)
from backend.nn.zimage_diffusers import (
    convert_diffusers_zimage_transformer,
    fast_forward_zimage,
    is_diffusers_zimage_transformer,
    is_zimage_diffusers_dir,
    is_zimage_state_dict,
    load_zimage_diffusers_dir,
    zimage_lora_key_map,
    zimage_workflow_advice,
)
from backend.state_dict import convert_quantization, detect_quantization

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


def _tiny_zimage(*, dim: int = 32, n_heads: int = 2, n_layers: int = 2) -> NextDiT:
    torch.manual_seed(0)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        m = NextDiT(
            patch_size=2,
            in_channels=4,
            dim=dim,
            n_layers=n_layers,
            n_refiner_layers=1,
            n_heads=n_heads,
            multiple_of=8,
            ffn_dim_multiplier=2.0,
            qk_norm=True,
            cap_feat_dim=16,
            axes_dims=(dim // n_heads // 4, (dim // n_heads) // 2 - (dim // n_heads // 4), (dim // n_heads) // 2),
            z_image_modulation=True,
        ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.03)
    return m


def test_zimage_capability_registration_quantization_and_sensitive_prefixes():
    dit_capabilities_ext.register_extended_dit_capabilities()
    cap = get_dit_capability("zimage")
    assert cap is not None
    assert capability_for_dit_cls("ZImageTransformer2DModel") is cap
    assert capability_for_compile_preset("Z-Image per-block") is cap
    assert should_keep_sensitive_weights_precise("ZImageTransformer2DModel", torch.float8_e4m3fn, torch.bfloat16) is True
    assert should_quantize_on_load("scaled_fp8", "ZImageTransformer2DModel", None, False, torch.bfloat16) is True
    assert should_quantize_on_load("int8_convrot", "ZImageTransformer2DModel", None, False, torch.bfloat16) is True

    # On-the-fly FP8 sensitive layer protection
    fp8_model = _tiny_zimage(dim=256, n_heads=4, n_layers=1).to(torch.float8_e4m3fn)
    kept = keep_sensitive_weights_precise(fp8_model, torch.bfloat16)
    assert kept > 0
    assert fp8_model.x_embedder.weight.dtype == torch.bfloat16
    assert fp8_model.cap_embedder[1].weight.dtype == torch.bfloat16
    assert fp8_model.final_layer.linear.weight.dtype == torch.bfloat16
    assert fp8_model.layers[0].attention.qkv.weight.dtype == torch.float8_e4m3fn

    # Scaled FP8 (rowwise) and INT8-ConvRot (dim=256 divisible by 256)
    wide = _tiny_zimage(dim=256, n_heads=4, n_layers=1)
    bf16_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in wide.state_dict().items()}
    assert is_zimage_state_dict(bf16_sd) is True
    assert capability_for_state_dict(bf16_sd) is cap

    fp8_sd = dict(bf16_sd)
    n_fp8 = scale_fp8_state_dict(fp8_sd, fp8_matmul=False)
    assert n_fp8 > 0
    assert fp8_sd["layers.0.attention.qkv.weight"].dtype == torch.float8_e4m3fn
    assert fp8_sd["layers.0.attention.qkv.weight_scale"].shape == (768, 1)
    assert "x_embedder.comfy_quant" not in fp8_sd

    int8_sd = dict(bf16_sd)
    n_int8 = quantize_int8_convrot_state_dict(int8_sd)
    assert n_int8 > 0
    assert int8_sd["layers.0.attention.qkv.weight"].dtype == torch.int8
    assert int8_sd["layers.0.feed_forward.w1.weight"].dtype == torch.int8
    assert "final_layer.linear.comfy_quant" not in int8_sd


def test_zimage_diffusers_unsloth_fp8_int8_and_single_file_conversion(tmp_path):
    dit_capabilities_ext.register_extended_dit_capabilities()
    model = _tiny_zimage(dim=32, n_heads=2, n_layers=1)
    native_sd = model.state_dict()

    # Convert native -> Diffusers layout (all_x_embedder.2-1, split to_q/to_k/to_v, to_out.0, norm_q/norm_k)
    diff_sd: dict[str, torch.Tensor] = {}
    for k, v in native_sd.items():
        if k.startswith("x_embedder."):
            diff_sd[k.replace("x_embedder.", "all_x_embedder.2-1.")] = v
        elif k.startswith("final_layer."):
            diff_sd[k.replace("final_layer.", "all_final_layer.2-1.")] = v
        elif ".attention.qkv." in k:
            q, kw, vw = v.chunk(3, dim=0)
            diff_sd[k.replace(".attention.qkv.", ".attention.to_q.")] = q
            diff_sd[k.replace(".attention.qkv.", ".attention.to_k.")] = kw
            diff_sd[k.replace(".attention.qkv.", ".attention.to_v.")] = vw
        elif ".attention.out." in k:
            diff_sd[k.replace(".attention.out.", ".attention.to_out.0.")] = v
        elif ".attention.q_norm." in k:
            diff_sd[k.replace(".attention.q_norm.", ".attention.norm_q.")] = v
        elif ".attention.k_norm." in k:
            diff_sd[k.replace(".attention.k_norm.", ".attention.norm_k.")] = v
        else:
            diff_sd[k] = v

    assert is_diffusers_zimage_transformer(diff_sd) is True
    roundtrip = convert_diffusers_zimage_transformer(diff_sd)
    assert set(roundtrip.keys()) == set(native_sd.keys())
    for k in native_sd:
        torch.testing.assert_close(roundtrip[k], native_sd[k])

    # Unsloth TorchAO INT8-ConvRot state dict conversion preserves per-row scale & convrot
    wide = _tiny_zimage(dim=256, n_heads=4, n_layers=1)
    raw_int8: dict[str, torch.Tensor] = {}
    fqns = []
    for k, v in wide.state_dict().items():
        if k.startswith("layers.") and k.endswith(".weight") and v.ndim == 2:
            stem = k[: -len(".weight")]
            fqns.append(stem)
            scale = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0
            raw_int8[f"{stem}._weight_qdata"] = (v / scale).round().clamp(-127, 127).to(torch.int8)
            raw_int8[f"{stem}._weight_scale"] = scale.float()
            raw_int8[f"{stem}._weight_zero_point"] = torch.zeros(v.shape[0], 1, dtype=torch.int8)
        else:
            raw_int8[k] = v.to(torch.bfloat16)

    converted, _ = convert_quantization(
        raw_int8,
        {
            "unsloth_metadata": json.dumps(
                {
                    "family": "z-image",
                    "scheme": "int8",
                    "activation_rotation": "convrot_hadamard_v1",
                    "activation_rotation_group": 256,
                    "activation_rotation_fqns": fqns,
                }
            )
        },
    )
    assert detect_quantization(converted, is_unet=True)["layout"] == "TensorWiseINT8Layout"

    # Full Z-Image-Turbo Diffusers directory discovery & load
    pipe_dir = tmp_path / "Z-Image-Turbo"
    (pipe_dir / "transformer").mkdir(parents=True)
    (pipe_dir / "text_encoder").mkdir(parents=True)
    (pipe_dir / "vae").mkdir(parents=True)
    (pipe_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "ZImageTurboPipeline", "transformer": ["diffusers", "ZImageTransformer2DModel"]}),
        encoding="utf-8",
    )
    save_file(diff_sd, str(pipe_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    save_file({"embed_tokens.weight": torch.zeros(8, 2560)}, str(pipe_dir / "text_encoder" / "model.safetensors"))
    save_file({"decoder.conv_in.weight": torch.zeros(16, 16, 3, 3)}, str(pipe_dir / "vae" / "diffusion_pytorch_model.safetensors"))

    assert is_zimage_diffusers_dir(pipe_dir) is True
    loaded_sd, meta = load_zimage_diffusers_dir(pipe_dir)
    assert meta.get("zimage_variant") == "turbo"
    assert "model.diffusion_model.x_embedder.weight" in loaded_sd
    assert "model.diffusion_model.layers.0.attention.qkv.weight" in loaded_sd
    assert "text_encoders.qwen3_4b.transformer.model.embed_tokens.weight" in loaded_sd
    assert "vae.decoder.conv_in.weight" in loaded_sd


def test_zimage_cross_step_cache_omni_reference_conditioning_and_spectrum():
    dit_capabilities_ext.register_extended_dit_capabilities()
    model = _tiny_zimage(dim=32, n_heads=2, n_layers=2)

    cap_calls = 0
    orig_cap_fwd = model.cap_embedder.forward

    def counted_cap(*args, **kwargs):
        nonlocal cap_calls
        cap_calls += 1
        return orig_cap_fwd(*args, **kwargs)

    model.cap_embedder.forward = counted_cap

    x = torch.randn(1, 4, 8, 8)
    ctx = torch.randn(1, 5, 16)
    with torch.inference_mode():
        y1 = model(x, torch.tensor([0.9]), ctx)
        y2 = model(x, torch.tensor([0.3]), ctx)
    assert cap_calls == 1  # text + context_refiner cached across denoising steps
    assert (y1 - y2).abs().max().item() > 1e-5

    # Mutating cap_embedder weight invalidates immediately
    with torch.no_grad():
        model.cap_embedder[1].weight.add_(0.2)
    with torch.inference_mode():
        y3 = model(x, torch.tensor([0.9]), ctx)
    assert cap_calls == 2
    assert not torch.allclose(y1, y3)

    # Z-Image-Omni / Edit reference conditioning path
    ref = torch.randn(1, 4, 8, 8)
    with torch.inference_mode():
        y_ref = model(x, torch.tensor([0.5]), ctx, ref_latents=[ref])
        y_noref = model(x, torch.tensor([0.5]), ctx, ref_latents=[])
    assert y_ref.shape == x.shape
    assert (y_ref - y_noref).abs().max().item() > 1e-5

    # Spectrum pre-head capture seam & bit-exact fast_forward_zimage
    assert match_spectrum_capability(model) is get_dit_capability("zimage")
    assert resolve_spectrum_capture_module(model) is model.zimage_final_input
    captured = {}

    def hook(_mod, args):
        captured["feat"] = args[0].detach().clone()

    h = model.zimage_final_input.register_forward_pre_hook(hook)
    try:
        sigma = torch.tensor([0.6])
        with torch.inference_mode():
            full = model(x, sigma, ctx)
            predictor = SimpleNamespace(timestep=lambda s: s, calculate_denoised=lambda s, pred, inp: pred)
            fast = fast_forward_zimage(model, predictor, sigma, x, captured["feat"])
        torch.testing.assert_close(fast, full, atol=1e-5, rtol=1e-5)
    finally:
        h.remove()


def test_zimage_bidirectional_lora_compile_and_workflow_advice():
    from modules_forge.packages.comfy.lora import SplitTargets, load_lora

    dit_capabilities_ext.register_extended_dit_capabilities()
    model = _tiny_zimage(dim=32, n_heads=2, n_layers=1)
    fused_sd = {f"diffusion_model.{k}": v for k, v in model.state_dict().items()}
    km = zimage_lora_key_map(fused_sd)

    # Split Diffusers to_q/to_k/to_v LoRA onto fused attention.qkv
    assert km["transformer.layers.0.attention.to_q"] == ("diffusion_model.layers.0.attention.qkv.weight", (0, 0, 32))
    assert km["lycoris_layers_0_attention_to_v"] == ("diffusion_model.layers.0.attention.qkv.weight", (0, 64, 32))
    assert km["transformer.all_x_embedder.2-1"] == "diffusion_model.x_embedder.weight"

    # Split checkpoint accepts fused qkv LoRA via SplitTargets
    split_sd = {
        "diffusion_model.layers.0.attention.to_q.weight": torch.zeros(32, 32),
        "diffusion_model.layers.0.attention.to_k.weight": torch.zeros(32, 32),
        "diffusion_model.layers.0.attention.to_v.weight": torch.zeros(32, 32),
    }
    km_split = zimage_lora_key_map(split_sd)
    assert isinstance(km_split["lora_unet_layers_0_attention_qkv"], SplitTargets)
    patches, rem = load_lora(
        {
            "lora_unet_layers_0_attention_qkv.lora_down.weight": torch.randn(4, 32),
            "lora_unet_layers_0_attention_qkv.lora_up.weight": torch.randn(96, 4),
            "lora_unet_layers_0_attention_qkv.alpha": torch.tensor(4.0),
        },
        km_split,
    )
    assert not rem and len(patches) == 3

    # Per-block compile picks model.layers
    compile_root = str(REPO / "extensions-builtin" / "sd_forge_compile")
    if compile_root not in sys.path:
        sys.path.insert(0, compile_root)
    from anima_block_compile import _block_list

    assert _block_list(model) is model.layers

    # Workflow advice
    p_turbo = SimpleNamespace(
        sd_model=SimpleNamespace(is_zimage_engine=True, is_zimage_turbo=True, sd_checkpoint_info=SimpleNamespace(name="Z-Image-Turbo.safetensors")),
        prompt="a castle",
        steps=28,
        cfg_scale=4.0,
    )
    notes = zimage_workflow_advice(p_turbo)
    assert any("Z-Image-Turbo" in n for n in notes)


def test_moss_kromoss_dmd2_detection_lora_mapping_and_advice():
    assert is_kromoss_variant("lodestones/Kromoss-fp8.safetensors") is True
    assert is_kromoss_variant("MOSS-Image-SFT-8-steps-dmd2") is True
    assert is_kromoss_variant("MOSS-Image-SFT.safetensors") is False
    assert is_dmd2_lora("moss_8_steps_dmd2_lora.safetensors") is True

    # Bidirectional fused qkv + split ffn_up/ffn_gate onto fused ffn_up_gate
    sd = {
        "diffusion_model.layers.0.proj.qkv.weight": torch.zeros(96, 32),
        "diffusion_model.layers.0.proj.ffn_up_gate.weight": torch.zeros(128, 32),
        "diffusion_model.layers.0.proj.o.weight": torch.zeros(32, 32),
    }
    km = moss_lora_key_map(sd)
    assert km["base_model.model.transformer.layers.0.proj.q"] == ("diffusion_model.layers.0.proj.qkv.weight", (0, 0, 32))
    assert km["transformer.layers.0.proj.ffn_up"] == ("diffusion_model.layers.0.proj.ffn_up_gate.weight", (0, 0, 64))
    assert km["lycoris_layers_0_proj_ffn_gate"] == ("diffusion_model.layers.0.proj.ffn_up_gate.weight", (0, 64, 64))

    p_kromoss = SimpleNamespace(
        sd_model=SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name="Kromoss-BF16.safetensors")),
        prompt="poster design <lora:moss_dmd2_8steps:1.0>",
        steps=30,
        cfg_scale=4.0,
    )
    notes = kromoss_dmd2_workflow_advice(p_kromoss)
    assert any("8 steps" in n and "CFG 1.0" in n for n in notes)
    assert dynamic_args is not None
