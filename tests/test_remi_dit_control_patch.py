"""Unit tests for Unified DiT ControlNet-Union (``backend/nn/dit_control_patch.py``),
quantization format detection & attention backend matrix (``backend/nn/quant_formats.py``),
and ControlNet UI family filtering hooks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from backend import operations
from backend.args import dynamic_args
from backend.nn.dit_control_patch import (
    FUN_UNION_CONTROL_TYPES,
    DiTControlNetUnionPatcher,
    DiTControlPatch,
    FunControlNetUnionPatch,
    Qwen21FunControlNet,
    ZImageFunControlNet,
    active_control_family,
    detect_dit_control_family,
    is_fun_controlnet_union,
    normalize_zimage_control_state_dict,
)
from backend.nn.lumina import NextDiT
from backend.nn.quant_formats import (
    ATTENTION_CAPABILITY_MATRIX,
    detect_quant_format,
    quant_formats_for_dit,
    safe_attention_call,
    select_attention_backend,
)
from backend.nn.qwen21 import QwenImage21Transformer2DModel
from backend.nn.svdq import svdq_quant_metadata, svdq_supported_architectures

REPO = Path(__file__).resolve().parents[1]


def _tiny_qwen21(num_layers: int = 4) -> QwenImage21Transformer2DModel:
    torch.manual_seed(0)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        m = QwenImage21Transformer2DModel(
            in_channels=8,
            out_channels=8,
            num_layers=num_layers,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=16,
            mlp_ratio=2,
            axes_dims_rope=(4, 6, 6),
            fused_mlp=False,
        ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.04)
    return m


def _tiny_nextdit(n_layers: int = 4) -> NextDiT:
    torch.manual_seed(1)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        m = NextDiT(
            patch_size=2,
            in_channels=4,
            dim=32,
            n_layers=n_layers,
            n_refiner_layers=1,
            n_heads=2,
            multiple_of=8,
            ffn_dim_multiplier=2.0,
            qk_norm=True,
            cap_feat_dim=16,
            axes_dims=(4, 6, 6),
            z_image_modulation=True,
        ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.04)
    return m


def test_qwen21_fun_controlnet_union_zero_gate_129ch_and_prefix_kv_invalidation():
    assert len(FUN_UNION_CONTROL_TYPES) == 8
    dit = _tiny_qwen21(num_layers=4)

    # 129-ch equivalent in tiny space: 8 control + 1 mask + 8 masked_latent = 17
    ctrl = Qwen21FunControlNet(
        inner_dim=32,
        num_attention_heads=2,
        attention_head_dim=16,
        mlp_ratio=2,
        fused_mlp=False,
        control_in_dim=17,
        latent_channels=8,
        num_base_layers=4,
        control_layers=[0, 2],  # even layers
        control_context_scale=1.0,
    ).eval()
    with torch.no_grad():
        for p in ctrl.parameters():
            p.normal_(0.0, 0.03)
    ctrl.init_zero_weights()

    # Auto-detection + from_state_dict round-trip (also test 16 even layers on 32-layer config)
    sd = ctrl.state_dict()
    assert detect_dit_control_family(sd) == "qwen21"
    assert is_fun_controlnet_union(sd) is True
    assert detect_dit_control_family({f"control_model.{k}": v for k, v in sd.items()}) == "qwen21"

    reloaded = Qwen21FunControlNet.from_state_dict(sd, metadata={"control_num_base_layers": "4"})
    assert reloaded.control_layers == [0, 2]
    assert reloaded.control_in_dim == 17

    # Verify default 16-even-layer inference on 32 layers + control_in_dim=129
    mock_129_sd = {
        "control_img_in.weight": torch.zeros(64, 129),
        **{f"control_blocks.{i}.block.attn.norm_q.weight": torch.ones(32) for i in range(16)},
        "control_blocks.0.block.img_mlp.proj.weight": torch.zeros(192, 64),
        "control_blocks.0.before_proj.weight": torch.zeros(64, 64),
        "control_blocks.0.after_proj.weight": torch.zeros(64, 64),
    }
    cfg_129 = Qwen21FunControlNet.infer_config(mock_129_sd)
    assert cfg_129["control_in_dim"] == 129
    assert cfg_129["latent_channels"] == 64
    assert cfg_129["control_layers"] == list(range(0, 32, 2))
    assert len(cfg_129["control_layers"]) == 16

    # Build 129-ch (tiny 17-ch) control input = control + mask + masked_latent
    x = torch.randn(1, 8, 4, 4)
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 5, 16)
    ref = [torch.randn(1, 8, 2, 2)]
    ctrl_lat = torch.randn(1, 8, 4, 4)
    mask = torch.ones(1, 1, 4, 4) * 0.5
    masked_lat = torch.randn(1, 8, 4, 4)

    assembled = reloaded.build_control_input(ctrl_lat, mask=mask, masked_latent=masked_lat)
    assert assembled.shape == (1, 17, 4, 4)

    # Warm up Prefix KV cache first
    with torch.inference_mode():
        base_out = dit(x, t, ctx, ref_latents=ref, image_slots=[2])
    assert len(dit._prefix_kv_cache) == 1

    # Attach zero-gated FunControlNetUnionPatch -> MUST invalidate Prefix KV cache and match base output identically
    patch = FunControlNetUnionPatch(reloaded, ctrl_lat, strength=1.0, mask=mask, masked_latent=masked_lat)
    assert isinstance(patch, DiTControlPatch)
    patch.attach(dit)
    assert len(dit._prefix_kv_cache) == 0

    with torch.inference_mode():
        zero_out = dit(x, t, ctx, ref_latents=ref, image_slots=[2])
    # Prefix KV cache must remain bypassed while control patch is attached
    assert len(dit._prefix_kv_cache) == 0
    torch.testing.assert_close(zero_out, base_out, atol=1e-5, rtol=1e-5)

    # Non-zero after_proj weights (non-constant across channels, outside LayerNorm's null space)
    # must visibly modify the output proportional to control_context_scale
    torch.manual_seed(7)
    with torch.no_grad():
        for cb in reloaded.control_blocks:
            cb.after_proj.weight.normal_(0.0, 0.2)
    with torch.inference_mode():
        active_out = dit(x, t, ctx, ref_latents=ref, image_slots=[2])
    assert (active_out - base_out).abs().max().item() > 1e-3

    # Detach restores clean unpatched behavior and re-enables Prefix KV caching
    patch.detach()
    with torch.inference_mode():
        restored_out = dit(x, t, ctx, ref_latents=ref, image_slots=[2])
    torch.testing.assert_close(restored_out, base_out, atol=1e-5, rtol=1e-5)
    assert len(dit._prefix_kv_cache) == 1


def test_zimage_fun_controlnet_union_zero_gate_and_diffusers_key_normalization():
    dit = _tiny_nextdit(n_layers=4)
    # patch_size=2, in_channels=4 -> control_in_dim = 2*2*(4+1+4) = 36 (control + mask + masked_latent)
    ctrl = ZImageFunControlNet(
        dim=32,
        n_heads=2,
        multiple_of=8,
        ffn_dim_multiplier=2.0,
        qk_norm=True,
        patch_size=2,
        control_in_dim=36,
        latent_channels=4,
        num_base_layers=4,
        control_layers=[0, 2],
        control_context_scale=0.8,
    ).eval()
    with torch.no_grad():
        for p in ctrl.parameters():
            p.normal_(0.0, 0.03)
    ctrl.init_zero_weights()

    sd = ctrl.state_dict()
    assert detect_dit_control_family(sd) == "zimage"
    assert is_fun_controlnet_union(sd) is True

    # Verify Diffusers-style split to_q/to_k/to_v + control_all_x_embedder.2-1 normalization
    diffusers_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("control_img_in."):
            diffusers_sd[k.replace("control_img_in.", "control_all_x_embedder.2-1.")] = v
        elif ".attention.qkv." in k:
            q, k_w, v_w = v.chunk(3, dim=0)
            diffusers_sd[k.replace(".block.attention.qkv.", ".attention.to_q.")] = q
            diffusers_sd[k.replace(".block.attention.qkv.", ".attention.to_k.")] = k_w
            diffusers_sd[k.replace(".block.attention.qkv.", ".attention.to_v.")] = v_w
        elif ".block.attention.out." in k:
            diffusers_sd[k.replace(".block.attention.out.", ".attention.to_out.0.")] = v
        elif ".block.attention.q_norm." in k:
            diffusers_sd[k.replace(".block.attention.q_norm.", ".attention.norm_q.")] = v
        elif ".block.attention.k_norm." in k:
            diffusers_sd[k.replace(".block.attention.k_norm.", ".attention.norm_k.")] = v
        else:
            diffusers_sd[k.replace(".block.", ".")] = v

    assert detect_dit_control_family(diffusers_sd) == "zimage"
    normalized = normalize_zimage_control_state_dict(diffusers_sd)
    assert set(normalized.keys()) == set(sd.keys())

    reloaded = ZImageFunControlNet.from_state_dict(diffusers_sd, metadata={"control_num_base_layers": "4"})
    assert reloaded.control_layers == [0, 2]

    x = torch.randn(1, 4, 8, 8)
    t = torch.tensor([0.6])
    ctx = torch.randn(1, 6, 16)
    ctrl_lat = torch.randn(1, 4, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    masked_lat = torch.randn(1, 4, 8, 8)

    with torch.inference_mode():
        base_out = dit(x, t, ctx)

    patch = FunControlNetUnionPatch(reloaded, ctrl_lat, strength=1.0, mask=mask, masked_latent=masked_lat)
    patch.attach(dit)
    with torch.inference_mode():
        zero_out = dit(x, t, ctx)
    torch.testing.assert_close(zero_out, base_out, atol=1e-5, rtol=1e-5)

    torch.manual_seed(9)
    with torch.no_grad():
        for cb in reloaded.control_blocks:
            cb.after_proj.weight.normal_(0.0, 0.2)
    with torch.inference_mode():
        active_out = dit(x, t, ctx)
    assert (active_out - base_out).abs().max().item() > 1e-3

    patch.detach()
    with torch.inference_mode():
        after_detach = dit(x, t, ctx)
    torch.testing.assert_close(after_detach, base_out, atol=1e-5, rtol=1e-5)


def test_patcher_builder_and_global_state_family_filtering(tmp_path, monkeypatch):
    ctrl = Qwen21FunControlNet(
        inner_dim=32,
        num_attention_heads=2,
        attention_head_dim=16,
        mlp_ratio=2,
        fused_mlp=False,
        control_in_dim=17,
        latent_channels=8,
        num_base_layers=4,
        control_layers=[0, 2],
    )
    patcher = DiTControlNetUnionPatcher.try_build_from_state_dict(ctrl.state_dict(), None)
    assert isinstance(patcher, DiTControlNetUnionPatcher)
    assert patcher.family == "qwen21"
    assert DiTControlNetUnionPatcher.try_build_from_state_dict({"zero_convs.0.0.weight": torch.zeros(4, 4)}, None) is None

    # Save safetensors fixtures and verify global_state family filtering
    qwen_file = tmp_path / "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"
    zimg_file = tmp_path / "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"
    sdxl_file = tmp_path / "controlnet_union_sdxl.safetensors"
    save_file(ctrl.state_dict(), str(qwen_file))
    z_ctrl = ZImageFunControlNet(dim=32, n_heads=2, multiple_of=8, ffn_dim_multiplier=2.0, control_in_dim=16, latent_channels=4, num_base_layers=4, control_layers=[0, 2])
    save_file(z_ctrl.state_dict(), str(zimg_file))
    save_file({"zero_convs.0.0.weight": torch.zeros(4, 4)}, str(sdxl_file))

    cn_root = str(REPO / "extensions-builtin" / "sd_forge_controlnet")
    if cn_root not in sys.path:
        sys.path.insert(0, cn_root)
    from lib_controlnet import global_state

    assert global_state.control_model_family(str(qwen_file)) == "qwen21"
    assert global_state.control_model_family(str(zimg_file)) == "zimage"
    assert global_state.control_model_family(str(sdxl_file)) == "unet"

    monkeypatch.setattr(dynamic_args, "anima", False)
    monkeypatch.setattr(dynamic_args, "qwen21", True)
    monkeypatch.setattr(dynamic_args, "zimage", False, raising=False)
    assert active_control_family() == "qwen21"

    monkeypatch.setattr(dynamic_args, "qwen21", False)
    monkeypatch.setattr(dynamic_args, "zimage", True, raising=False)
    assert active_control_family() == "zimage"


def test_quant_format_detection_and_attention_capability_matrix():
    # 1. INT8-ConvRot (comfy_quant)
    int8_sd = {
        "layers.0.proj.weight": torch.zeros(16, 256, dtype=torch.int8),
        "layers.0.proj.weight_scale": torch.ones(16, 1),
        "layers.0.proj.comfy_quant": torch.tensor(list(b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'), dtype=torch.uint8),
    }
    info_int8 = detect_quant_format(int8_sd)
    assert info_int8.format_id == "int8_convrot" and info_int8.label == "INT8-ConvRot" and info_int8.convrot_groupsize == 256

    # 2. TorchAO FP8 per-row
    fp8_sd = {
        "layers.0.q._weight_qdata": torch.zeros(16, 16, dtype=torch.float8_e4m3fn),
        "layers.0.q._weight_scale": torch.ones(16, 1),
    }
    info_fp8 = detect_quant_format(fp8_sd, {"unsloth_metadata": json.dumps({"scheme": "fp8", "fp8_granularity": "per_row"})})
    assert info_fp8.format_id == "fp8_e4m3_rowwise" and info_fp8.label == "FP8"

    # 3. NVFP4
    nvfp4_sd = {
        "blocks.0.w.weight": torch.zeros(16, 8, dtype=torch.uint8),
        "blocks.0.w.comfy_quant": torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8),
    }
    info_nvfp4 = detect_quant_format(nvfp4_sd)
    assert info_nvfp4.format_id == "nvfp4" and info_nvfp4.label == "NVFP4"

    # 4. SVDQuant W4A4 + svdq.py helper
    svdq_sd = {
        "layers.0.qkv.qweight": torch.zeros(16, 8, dtype=torch.int8),
        "layers.0.qkv.wscales": torch.ones(16, 1),
        "layers.0.qkv.proj_down": torch.zeros(32, 16),
        "layers.0.qkv.proj_up": torch.zeros(16, 32),
    }
    info_svdq = detect_quant_format(svdq_sd)
    assert info_svdq.format_id == "svdq_w4a4" and info_svdq.svdq_rank == 16
    assert svdq_quant_metadata(svdq_sd) == info_svdq
    assert "zimage" in svdq_supported_architectures()

    # 5. GGUF synthetic tensor
    fake_gguf = torch.zeros(16, dtype=torch.uint8)
    fake_gguf.gguf_cls = object()
    fake_gguf.tensor_type = SimpleNamespace(name="Q4_K_M")
    info_gguf = detect_quant_format({"blocks.0.weight": fake_gguf})
    assert info_gguf.format_id == "gguf" and "Q4_K_M" in info_gguf.gguf_types

    # 6. Per-DiT format matrix
    assert {"bf16", "fp8", "int8_convrot", "gguf"} <= set(quant_formats_for_dit("zimage"))

    # 7. Attention backend capability matrix & safe fallbacks
    assert set(ATTENTION_CAPABILITY_MATRIX) >= {"sage3", "sage2", "sage1", "flash", "sdpa", "basic"}
    sel, reason = select_attention_backend(preferred="flash", device="cuda", head_dim=128, has_mask=True, mask_is_causal=False, available_backends=["flash", "sdpa", "basic"])
    assert sel == "sdpa" and reason is not None and "mask" in reason

    sel_mps, reason_mps = select_attention_backend(preferred="sage2", device="mps", head_dim=128, available_backends=["sage2", "sdpa"])
    assert sel_mps == "sdpa" and reason_mps is not None and "mps" in reason_mps

    sel_head, reason_head = select_attention_backend(preferred="sage1", device="cuda", head_dim=80, available_backends=["sage1", "sdpa"])
    assert sel_head == "sdpa" and reason_head is not None

    # Real CPU safe_attention_call with 4D bool mask falls back cleanly to SDPA/basic
    q = torch.randn(1, 4, 32)
    k = torch.randn(1, 4, 32)
    v = torch.randn(1, 4, 32)
    bool_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    out = safe_attention_call(q, k, v, heads=2, mask=bool_mask, preferred="flash")
    assert out.shape == (1, 4, 32) and torch.isfinite(out).all()
