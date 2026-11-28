"""Regression and pinning tests for ML runtime, quantization, and attention fixes."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from backend import memory_management, operations
from backend import operations_mixed_precision as omp
from backend.attention import attention_basic, attention_pytorch
from backend.nn.anima import Anima
from backend.nn.anima_quant import (
    ANIMA_QUANT_INT8_CONVROT,
    ANIMA_QUANT_SCALED_FP8,
    int8_matmul_available,
    keep_sensitive_weights_precise,
    quantize_on_load,
    scale_fp8_state_dict,
)
from backend.patcher.vae import VAE
from backend.quant_ops import (
    RowWiseFP8E4M3Layout,
    TensorCoreFP8E4M3Layout,
    TensorCoreMXFP8Layout_,
    TensorCoreNVFP4Layout_,
)
from backend.state_dict import detect_quantization

ROOT = Path(__file__).resolve().parents[1]
HAS_MPS = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
_ANIMA_CFG = {
    "in_channels": 4,
    "out_channels": 4,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "model_channels": 256,
    "crossattn_emb_channels": 256,
    "adaln_lora_dim": 32,
    "num_blocks": 3,
    "num_heads": 2,
}


def _tiny_anima_state_dict() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(13)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        shapes = {k: v.shape for k, v in Anima(**_ANIMA_CFG).state_dict().items()}
    return {k: (torch.randn(shape, generator=gen) * 0.02).to(torch.bfloat16) for k, shape in shapes.items()}


# ---------------------------------------------------------------------------
# MPS INT8-ConvRot enablement, FP8 fallback, CPU dequant
# ---------------------------------------------------------------------------


def test_quantize_on_load_mps_int8_and_scaled_fp8_fallback(monkeypatch):
    mps = torch.device("mps")
    monkeypatch.setattr(memory_management, "should_use_bf16", lambda dev: True)
    monkeypatch.setattr(memory_management, "supports_cast", lambda dev, dt: getattr(dev, "type", None) != "mps")

    base = _tiny_anima_state_dict()

    # INT8-ConvRot on MPS runs rotation on CPU while keeping supports_int8_compute("mps") == False
    int8_calls: list[torch.device | None] = []
    from backend.nn import anima_quant

    orig_convrot = anima_quant.quantize_int8_convrot_state_dict

    def spy_convrot(sd, *, device=None):
        int8_calls.append(device)
        return orig_convrot(sd, device=device)

    monkeypatch.setattr(anima_quant, "quantize_int8_convrot_state_dict", spy_convrot)

    sd_int8 = {k: v.clone() for k, v in base.items()}
    quantize_on_load(sd_int8, ANIMA_QUANT_INT8_CONVROT, mps)
    assert int8_calls == [memory_management.cpu]
    assert any(v.dtype == torch.int8 for v in sd_int8.values())
    assert int8_matmul_available(mps) is False
    assert memory_management.supports_int8_compute(mps) is False

    # scaled_fp8 on MPS falls back cleanly to INT8-ConvRot (0 float8 tensors)
    sd_fp8 = {k: v.clone() for k, v in base.items()}
    quantize_on_load(sd_fp8, ANIMA_QUANT_SCALED_FP8, mps)
    assert not any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd_fp8.values())
    assert any(v.dtype == torch.int8 for v in sd_fp8.values())
    assert detect_quantization(sd_fp8, is_unet=True) is not None


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
def test_mps_real_quantize_on_load_and_scale_fp8_and_loader_fallback(monkeypatch):
    import backend.args
    from backend import loader

    mps = torch.device("mps")
    base = _tiny_anima_state_dict()

    sd_fp8 = {k: v.clone() for k, v in base.items()}
    quantize_on_load(sd_fp8, ANIMA_QUANT_SCALED_FP8, mps)
    assert not any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd_fp8.values())

    # Direct scale_fp8_state_dict with device=mps must not raise Undefined type Float8_e4m3fn
    sd_direct = {k: v.clone() for k, v in base.items()}
    n = scale_fp8_state_dict(sd_direct, fp8_matmul=False, device=mps)
    assert n > 0
    assert all(v.device.type == "cpu" for v in sd_direct.values())

    # On-the-fly FP8 storage override targeting MPS falls back cleanly to computation_dtype (BF16)
    monkeypatch.setattr(memory_management, "get_torch_device", lambda: mps)
    monkeypatch.setattr(memory_management, "unet_offload_device", lambda: memory_management.cpu)
    monkeypatch.setattr(memory_management, "unet_initial_load_device", lambda parameters, dtype: mps)
    monkeypatch.setattr(backend.args.dynamic_args, "anima_quant", None)
    monkeypatch.setattr(backend.args.dynamic_args, "forge_unet_storage_dtype", torch.float8_e4m3fn)
    monkeypatch.setattr(backend.args.dynamic_args, "ops", None)

    guess = SimpleNamespace(unet_config=dict(_ANIMA_CFG), nunchaku=False, supported_inference_dtypes=[torch.bfloat16, torch.float32])
    model = loader.load_huggingface_component(guess, "transformer", "diffusers", "CosmosTransformer3DModel", "/unused", {k: v.clone() for k, v in base.items()})
    assert model.storage_dtype == torch.bfloat16
    assert model.blocks[2].mlp.layer1.weight.device.type == "mps"
    assert model.blocks[2].mlp.layer1.weight.dtype == torch.bfloat16


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
@pytest.mark.parametrize(
    ("fmt", "layout_cls"),
    [
        ("float8_e4m3fn", TensorCoreFP8E4M3Layout),
        ("float8_e4m3fn_rowwise", RowWiseFP8E4M3Layout),
        ("nvfp4", TensorCoreNVFP4Layout_),
        ("mxfp8", TensorCoreMXFP8Layout_),
    ],
)
def test_prequantized_fp8_nvfp4_dequants_on_cpu_for_mps_linear(fmt, layout_cls):
    mps = torch.device("mps")
    ops = omp.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=True, disabled=[])
    w = torch.randn(128, 256, dtype=torch.bfloat16)
    qd, params = layout_cls.quantize(w, scale="recalculate")
    sd = {
        "weight": qd,
        "comfy_quant": torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8),
    }
    if "input_scale" in omp.QUANT_ALGOS[fmt]["parameters"]:
        sd["input_scale"] = torch.ones((), dtype=torch.float32)
    if fmt == "nvfp4":
        sd["weight_scale_2"] = params.scale
        sd["weight_scale"] = params.block_scale
    else:
        sd["weight_scale"] = params.scale

    layer = ops.Linear(256, 128, bias=False, device=mps, dtype=torch.bfloat16)
    layer.load_state_dict(sd)
    assert not isinstance(layer.weight, omp.QuantizedTensor)
    assert layer.weight.device.type == "mps"
    assert layer.weight.dtype == torch.bfloat16

    x = torch.randn(2, 8, 256, device=mps, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = layer(x)
    assert out.device.type == "mps"
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
def test_fp8_embedding_and_keep_sensitive_and_cast_to_on_mps():
    mps = torch.device("mps")
    ops = omp.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=True, disabled=[])
    emb = ops.Embedding(16, 8)
    qd, _ = TensorCoreFP8E4M3Layout.quantize(torch.randn(16, 8, dtype=torch.bfloat16), scale=1.0)
    emb.load_state_dict(
        {
            "weight": qd,
            "weight_scale": torch.ones((), dtype=torch.float32),
            "comfy_quant": torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8),
        }
    )
    out = emb(torch.tensor([[0, 2, 5]], device=mps))
    assert out.device.type == "mps"
    assert torch.isfinite(out).all()

    # cast_to on MPS FP8 tensor falls back via CPU without Float8_e4m3fn error
    mps_fp8 = qd.to(device=mps)
    casted = memory_management.cast_to(mps_fp8, dtype=torch.bfloat16, device=mps)
    assert casted.device.type == "mps" and casted.dtype == torch.bfloat16

    # keep_sensitive_weights_precise handles MPS FP8 parameter cleanly
    mod = torch.nn.Module()
    mod.norm = torch.nn.LayerNorm(8)
    mod.norm.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.float8_e4m3fn).to(device=mps), requires_grad=False)
    assert keep_sensitive_weights_precise(mod, torch.bfloat16) >= 1
    assert mod.norm.weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Opt-in REMI_MPS_SDPA=1 + 2D (B, K) padding mask fix
# ---------------------------------------------------------------------------


def test_mps_sdpa_opt_in_policy_default_off_and_env_opt_in():
    assert memory_management._should_enable_mps_sdpa(True, {}) is False
    assert memory_management._should_enable_mps_sdpa(True, {"REMI_MPS_SDPA": "0"}) is False
    assert memory_management._should_enable_mps_sdpa(False, {"REMI_MPS_SDPA": "1"}) is False
    assert memory_management._should_enable_mps_sdpa(True, {"REMI_MPS_SDPA": "1"}) is True

    if HAS_MPS:
        env_off = {k: v for k, v in os.environ.items() if k != "REMI_MPS_SDPA"}
        res_off = subprocess.run(
            [sys.executable, "-c", "from backend import memory_management; print(memory_management.pytorch_attention_enabled())"],
            cwd=ROOT,
            env=env_off,
            check=True,
            capture_output=True,
            text=True,
        )
        assert res_off.stdout.strip().endswith("False")

        env_on = {**env_off, "REMI_MPS_SDPA": "1"}
        res_on = subprocess.run(
            [sys.executable, "-c", "from backend import memory_management; print(memory_management.pytorch_attention_enabled())"],
            cwd=ROOT,
            env=env_on,
            check=True,
            capture_output=True,
            text=True,
        )
        assert res_on.stdout.strip().endswith("True")


@pytest.mark.parametrize("device_name", ["cpu", *(["mps"] if HAS_MPS else [])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_pytorch_2d_bk_and_qk_masks_match_basic(device_name, dtype):
    dev = torch.device(device_name)
    gen = torch.Generator().manual_seed(21)
    b, q_len, k_len, heads, dim_head = 2, 32, 16, 4, 32
    q = torch.randn(b, q_len, heads * dim_head, generator=gen).to(device=dev, dtype=dtype)
    k = torch.randn(b, k_len, heads * dim_head, generator=gen).to(device=dev, dtype=dtype)
    v = torch.randn(b, k_len, heads * dim_head, generator=gen).to(device=dev, dtype=dtype)

    # 2D (B, K) boolean key-padding mask (previously crashed in attention_pytorch)
    mask_bk = torch.ones(b, k_len, dtype=torch.bool, device=dev)
    mask_bk[0, 10:] = False
    mask_bk[1, 6:] = False
    out_sdpa = attention_pytorch(q, k, v, heads, mask=mask_bk)
    out_basic = attention_basic(q, k, v, heads, mask=mask_bk)
    assert torch.isfinite(out_sdpa).all()
    torch.testing.assert_close(out_sdpa.float(), out_basic.float(), atol=2e-2, rtol=2e-2)

    # 2D (Q, K) boolean mask
    mask_qk = torch.ones(q_len, k_len, dtype=torch.bool, device=dev)
    mask_qk[:, 8:] = False
    out_sdpa_qk = attention_pytorch(q, k, v, heads, mask=mask_qk)
    out_basic_qk = attention_basic(q, k, v, heads, mask=mask_qk)
    torch.testing.assert_close(out_sdpa_qk.float(), out_basic_qk.float(), atol=2e-2, rtol=2e-2)

    # 2D (B, K) when B == Q (edge case: must treat as (B, 1, 1, K) matching attention_basic)
    q_sq = q[:, :b, :]
    out_sdpa_sq = attention_pytorch(q_sq, k, v, heads, mask=mask_bk)
    out_basic_sq = attention_basic(q_sq, k, v, heads, mask=mask_bk)
    torch.testing.assert_close(out_sdpa_sq.float(), out_basic_sq.float(), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# denoising_strength == 0 + img2img_fix_steps & scheduler n <= 0
# ---------------------------------------------------------------------------


def test_setup_img2img_steps_and_schedulers_guard_zero_denoise(monkeypatch):
    from remi.testing import webui_state

    with webui_state():
        from modules import shared
        from modules.sd_samplers_common import setup_img2img_steps
        from modules.sd_schedulers import ddim_scheduler, kl_optimal, linear_quadratic, simple_scheduler, turbo_scheduler

        monkeypatch.setattr(shared.opts, "img2img_fix_steps", True, raising=False)
        steps, t_enc = setup_img2img_steps(SimpleNamespace(denoising_strength=0.0, steps=20))
        assert steps >= 1
        assert 0 <= t_enc < steps

        steps_neg, t_enc_neg = setup_img2img_steps(SimpleNamespace(denoising_strength=-0.5, steps=10))
        assert steps_neg >= 1 and t_enc_neg == 0

        inner = SimpleNamespace(
            sigmas=torch.linspace(0.0, 1.0, 1000),
            inner_model=SimpleNamespace(
                forge_objects=SimpleNamespace(
                    unet=SimpleNamespace(
                        model=SimpleNamespace(predictor=SimpleNamespace(sigmas=torch.linspace(0.0, 1.0, 1000)))
                    )
                )
            ),
        )
        for sched_fn in (simple_scheduler, turbo_scheduler, ddim_scheduler):
            out = sched_fn(0, 0.0, 1.0, inner, "cpu")
            assert out.numel() >= 1 and torch.isfinite(out).all()
        assert kl_optimal(0, 0.01, 14.0, "cpu").numel() >= 1
        assert linear_quadratic(0, 0.0, 1.0, "cpu").numel() >= 1


# ---------------------------------------------------------------------------
# Txt2Img.close() preserves 3-slot cached_hr_c / cached_hr_uc
# ---------------------------------------------------------------------------


def test_txt2img_close_preserves_three_slot_hr_cond_cache(monkeypatch):
    from remi.testing import webui_state

    with webui_state():
        from modules import shared
        from modules.processing import StableDiffusionProcessingTxt2Img as Txt2Img

        orig_c = list(Txt2Img.cached_hr_c)
        orig_uc = list(Txt2Img.cached_hr_uc)
        try:
            monkeypatch.setattr(shared.opts, "persistent_cond_cache", False, raising=False)
            p = Txt2Img.__new__(Txt2Img)
            p.c = p.uc = p.hr_c = p.hr_uc = p.sd_model = None
            p.close()
            assert Txt2Img.cached_hr_c == [None, None, None]
            assert Txt2Img.cached_hr_uc == [None, None, None]
        finally:
            Txt2Img.cached_hr_c = orig_c
            Txt2Img.cached_hr_uc = orig_uc


# ---------------------------------------------------------------------------
# VAE.clone() copies geometry & dim attributes
# ---------------------------------------------------------------------------


def test_vae_clone_copies_geometry_and_dim_attrs():
    v = VAE(no_init=True)
    v.patcher = SimpleNamespace(clone=lambda: "cloned_patcher")
    v.output_channels = 4
    v.memory_used_encode = lambda *_a: 1
    v.memory_used_decode = lambda *_a: 1
    v.downscale_ratio = (lambda a: a, 8, 8)
    v.upscale_ratio = (lambda a: a, 8, 8)
    v.upscale_index_formula = (4, 8, 8)
    v.downscale_index_formula = (4, 8, 8)
    v.latent_dim = 3
    v.latent_channels = 16
    v.first_stage_model = object()
    v.device = torch.device("cpu")
    v.vae_dtype = torch.float32
    v.output_device = torch.device("cpu")
    v.is_wan = True
    v.pad_channel_value = 1.0

    vc = v.clone()
    assert vc.upscale_ratio is v.upscale_ratio
    assert vc.upscale_index_formula == (4, 8, 8)
    assert vc.downscale_index_formula == (4, 8, 8)
    assert vc.latent_dim == 3
    assert vc.pad_channel_value == 1.0


# ---------------------------------------------------------------------------
# QwenImage.get_learned_conditioning_with_image preserves batch > 1
# ---------------------------------------------------------------------------


def test_qwen_image_preserves_per_item_prompt_batch():
    from backend.args import dynamic_args
    from backend.diffusion_engine.qwen import QwenImage

    engine = object.__new__(QwenImage)
    recorded: list[tuple[list[str], int]] = []
    engine.encode_vision = lambda img, idx: (img, img, f"Picture {idx}: <vision>")
    engine.text_processing_engine_qwen = lambda prompts, images=None: (
        recorded.append((list(prompts), len(images or []))),
        [torch.ones(1, 2, 4) for _ in prompts],
    )[1]

    orig_refs = list(dynamic_args.ref_latents)
    try:
        ref = torch.zeros(1, 16, 16, 3)
        out = engine.get_learned_conditioning_with_image(["first prompt", "second prompt"], [ref])
        assert len(out) == 2
        assert recorded == [(["Picture 0: <vision>\nfirst prompt", "Picture 0: <vision>\nsecond prompt"], 1)]
    finally:
        dynamic_args.ref_latents = orig_refs


# ---------------------------------------------------------------------------
# ZImage.clear_references() resets dynamic_args.zimage
# ---------------------------------------------------------------------------


def test_zimage_clear_references_resets_dynamic_args_flag():
    from backend.args import dynamic_args
    from backend.diffusion_engine.zimage import ZImage

    engine = object.__new__(ZImage)
    engine.ref_latents = [torch.zeros(1, 16, 4, 4)]
    engine.ini_latent = torch.zeros(1, 16, 4, 4)
    engine.zimage_do_reference = True
    engine.forge_objects = SimpleNamespace(
        clip=SimpleNamespace(patcher=None),
        unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(clear_caches=lambda: None))),
    )
    engine.text_processing_engine_qwen = lambda prompt: prompt

    orig_zimage = getattr(dynamic_args, "zimage", False)
    orig_refs = list(dynamic_args.ref_latents)
    try:
        # Mock load_model_gpu via memory_management attribute not needed if we test get_learned_conditioning + clear_references
        dynamic_args.zimage = True
        dynamic_args.ref_latents = [torch.zeros(1, 16, 4, 4)]
        engine.clear_references()
        assert dynamic_args.zimage is False
        assert dynamic_args.ref_latents == []
    finally:
        dynamic_args.zimage = orig_zimage
        dynamic_args.ref_latents = orig_refs


# ---------------------------------------------------------------------------
# Content-safe & ref-lifetime-safe vision / DiT cache keys
# ---------------------------------------------------------------------------


def test_krea2_vision_cache_rejects_gc_id_reuse_and_in_place_mutation():
    from backend.args import dynamic_args
    from backend.diffusion_engine.krea import Krea2

    encode_calls = 0

    class _FakeKrea(Krea2):
        def encode_vision(self, image: torch.Tensor):
            nonlocal encode_calls
            encode_calls += 1
            return image.clone(), torch.full((1, 16, 1, 2, 2), float(encode_calls))

    engine = object.__new__(_FakeKrea)
    engine._vision_cache = None
    engine.text_processing_engine_qwen = lambda prompt, images=None: list(prompt)

    orig_refs = list(dynamic_args.ref_latents)
    try:
        img1 = torch.zeros(1, 8, 8, 3)
        engine.get_learned_conditioning_with_image(["p1"], [img1])
        assert encode_calls == 1

        # Identical pixels hit cache
        engine.get_learned_conditioning_with_image(["p2"], [img1.clone()])
        assert encode_calls == 1

        # In-place mutation must invalidate cache
        img1.fill_(1.0)
        engine.get_learned_conditioning_with_image(["p3"], [img1])
        assert encode_calls == 2

        # Force GC id() reuse with different content -> must NOT hit stale cache
        old_id = id(img1)
        del img1
        gc.collect()
        img2 = torch.full((1, 8, 8, 3), 2.0)
        # Regardless of whether Python reused old_id, content digest guarantees cache miss
        _ = old_id
        engine.get_learned_conditioning_with_image(["p4"], [img2])
        assert encode_calls == 3
    finally:
        dynamic_args.ref_latents = orig_refs


def test_krea_nn_and_qwen21_nn_caches_guard_weight_weakrefs():
    from backend.nn.krea import SingleStreamDiT
    from backend.nn.qwen21 import QwenImage21Transformer2DModel

    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=False):
        krea_dit = SingleStreamDiT(features=256, tdim=64, txtdim=160, heads=8, kvheads=4, layers=1, patch=2, channels=4, txtlayers=2, txtheads=4, txtkvheads=4).eval()
        qwen_dit = QwenImage21Transformer2DModel(
            in_channels=16,
            out_channels=16,
            num_layers=1,
            attention_head_dim=32,
            num_attention_heads=4,
            context_in_dim=64,
            axes_dims_rope=(8, 12, 12),
            fused_mlp=False,
        ).eval()
    with torch.no_grad():
        for p in krea_dit.parameters():
            p.normal_(0.0, 0.02)
        for p in qwen_dit.parameters():
            p.normal_(0.0, 0.02)

    ctx = torch.randn(1, 4, 2, 160)
    with torch.inference_mode():
        out1 = krea_dit._encode_context_cached(ctx, torch.float32, {})
        assert len(krea_dit._txt_cache) == 1
        # Replacing the cached weight parameter breaks its weakref -> cache miss
        krea_dit.txtfusion.projector.weight = torch.nn.Parameter(torch.randn_like(krea_dit.txtfusion.projector.weight))
        gc.collect()
        out2 = krea_dit._encode_context_cached(ctx, torch.float32, {})
        assert out2 is not out1

    q_ctx = torch.randn(1, 4, 64)
    with torch.inference_mode():
        res1 = qwen_dit._get_prefix_kv(1, (0, 0), torch.device("cpu"), torch.float32, q_ctx, [], [], {})
        assert res1 is not None
        assert len(qwen_dit._prefix_kv_cache) == 1
        entry = next(iter(qwen_dit._prefix_kv_cache.values()))
        assert "weight_weakrefs" in entry and all(wr() is not None for wr in entry["weight_weakrefs"])
        qwen_dit.txt_in.in_layer.weight = torch.nn.Parameter(torch.randn_like(qwen_dit.txt_in.in_layer.weight))
        gc.collect()
        res2 = qwen_dit._get_prefix_kv(1, (0, 0), torch.device("cpu"), torch.float32, q_ctx, [], [], {})
        assert res2 is not None and res2[0] is not res1[0]


def test_qwen21_vision_cache_rejects_gc_id_reuse_and_in_place_mutation(monkeypatch):
    from types import SimpleNamespace

    from backend.args import dynamic_args
    from backend.diffusion_engine.qwen21 import QwenImage21
    from modules import shared

    encode_calls = 0

    class _FakeQwen21(QwenImage21):
        def encode_vision(self, image: torch.Tensor):
            nonlocal encode_calls
            encode_calls += 1
            return image.clone(), torch.full((1, 64, 1, 2, 2), float(encode_calls))

    engine = object.__new__(_FakeQwen21)
    engine._vision_cache = None
    engine.text_processing_engine_qwen = lambda prompt, images=None: list(prompt)

    if getattr(shared, "opts", None) is None:
        monkeypatch.setattr(shared, "opts", SimpleNamespace(reference_max_side=1024), raising=False)
    else:
        monkeypatch.setattr(shared.opts, "reference_max_side", 1024, raising=False)
    orig_refs = list(dynamic_args.ref_latents)
    try:
        img1 = torch.zeros(1, 8, 8, 4)
        engine.get_learned_conditioning_with_image(["p1"], [img1])
        assert encode_calls == 1

        engine.get_learned_conditioning_with_image(["p2"], [img1.clone()])
        assert encode_calls == 1

        img1.fill_(1.0)
        engine.get_learned_conditioning_with_image(["p3"], [img1])
        assert encode_calls == 2

        del img1
        gc.collect()
        img2 = torch.full((1, 8, 8, 4), 2.0)
        engine.get_learned_conditioning_with_image(["p4"], [img2])
        assert encode_calls == 3

        monkeypatch.setattr(shared.opts, "reference_max_side", 512, raising=False)
        engine.get_learned_conditioning_with_image(["p5"], [img2])
        assert encode_calls == 4
    finally:
        dynamic_args.ref_latents = orig_refs


def test_using_forge_operations_restores_global_state():
    from backend.nn.qwen21_vae import Wan22VAE

    prev = (operations.current_device, operations.current_dtype, operations.current_manual_cast_enabled)
    try:
        with pytest.raises(RuntimeError, match="boom"), operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float8_e4m3fn, manual_cast_enabled=True):
            assert operations.current_dtype is torch.float8_e4m3fn
            raise RuntimeError("boom")
        assert (operations.current_device, operations.current_dtype, operations.current_manual_cast_enabled) == prev
        vae = Wan22VAE(dim=8, dec_dim=8, z_dim=8, dim_mult=[1, 2, 2], num_res_blocks=1, attn_scales=[], temporal_downsample=[False, False], dropout=0.0, image_channels=4, patch_size=1, temporal_kernel=1)
        assert vae.encoder.conv1.weight.dtype == torch.float32
    finally:
        operations.current_device, operations.current_dtype, operations.current_manual_cast_enabled = prev


def test_cast_to_mps_unsupported_target_dtype_stages_via_cpu(monkeypatch):
    src = torch.ones((2, 2), dtype=torch.bfloat16)
    staged_devices: list[str] = []

    class _FakeTensor:
        def __init__(self, t: torch.Tensor, dev_type: str):
            self._t = t
            self.device = SimpleNamespace(type=dev_type)
            self.dtype = t.dtype

        def cpu(self):
            staged_devices.append("cpu")
            return _FakeTensor(self._t.cpu(), "cpu")

        def to(self, *args, dtype=None, device=None, copy=False, **kwargs):
            if self.device.type == "mps" and dtype == torch.float8_e4m3fn:
                raise RuntimeError("Undefined type Float8_e4m3fn")
            out_t = self._t.to(dtype=dtype) if dtype is not None else self._t
            out_dev = getattr(device, "type", self.device.type) if device is not None else self.device.type
            return _FakeTensor(out_t, out_dev)

    monkeypatch.setattr(memory_management, "is_device_mps", lambda dev: getattr(dev, "type", None) == "mps")
    monkeypatch.setattr(memory_management, "supports_cast", lambda dev, dt: not (getattr(dev, "type", None) == "mps" and dt in (torch.float8_e4m3fn, torch.float8_e5m2)))

    fake_mps = _FakeTensor(src, "mps")
    res = memory_management.cast_to(fake_mps, dtype=torch.float8_e4m3fn)
    assert staged_devices == ["cpu"]
    assert res.device.type == "mps"
    assert res.dtype == torch.float8_e4m3fn

    res_same = memory_management.cast_to(fake_mps, dtype=torch.float8_e4m3fn, device=SimpleNamespace(type="mps"))
    assert res_same.device.type == "mps"
    assert res_same.dtype == torch.float8_e4m3fn


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
def test_cast_to_mps_fp8_roundtrip_no_undefined_type_error():
    mps = torch.device("mps")
    w_bf16 = torch.randn(4, 4, device=mps, dtype=torch.bfloat16)

    w_fp8_none = memory_management.cast_to(w_bf16, dtype=torch.float8_e4m3fn, device=None)
    assert w_fp8_none.device.type == "mps"
    assert w_fp8_none.dtype == torch.float8_e4m3fn

    w_fp8_mps = memory_management.cast_to(w_bf16, dtype=torch.float8_e4m3fn, device=mps)
    assert w_fp8_mps.device.type == "mps"
    assert w_fp8_mps.dtype == torch.float8_e4m3fn

    w_back = memory_management.cast_to(w_fp8_mps, dtype=torch.bfloat16, device=mps)
    assert w_back.device.type == "mps"
    assert w_back.dtype == torch.bfloat16


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
@pytest.mark.parametrize(
    ("fmt", "layout_cls"),
    [
        ("float8_e4m3fn", TensorCoreFP8E4M3Layout),
        ("float8_e4m3fn_rowwise", RowWiseFP8E4M3Layout),
        ("nvfp4", TensorCoreNVFP4Layout_),
        ("mxfp8", TensorCoreMXFP8Layout_),
    ],
)
def test_mps_string_device_and_cpu_to_mps_linear_no_fp8_error(fmt, layout_cls):
    ops_fp = omp.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=True, disabled=[])
    ops_quant = omp.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[])
    w = torch.randn(128, 256, dtype=torch.bfloat16)
    qd, params = layout_cls.quantize(w, scale="recalculate")
    sd_base = {
        "weight": qd,
        "comfy_quant": torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8),
    }
    if "input_scale" in omp.QUANT_ALGOS[fmt]["parameters"]:
        sd_base["input_scale"] = torch.ones((), dtype=torch.float32)
    if fmt == "nvfp4":
        sd_base["weight_scale_2"] = params.scale
        sd_base["weight_scale"] = params.block_scale
    else:
        sd_base["weight_scale"] = params.scale

    # 1. Loading directly with device="mps" string must dequantize on CPU without RuntimeError
    layer_str = ops_fp.Linear(256, 128, bias=False, device="mps", dtype=torch.bfloat16)
    layer_str.load_state_dict({k: v.clone() for k, v in sd_base.items()})
    assert not isinstance(layer_str.weight, omp.QuantizedTensor)
    assert layer_str.weight.device.type == "mps"

    x = torch.randn(2, 8, 256, device="mps", dtype=torch.bfloat16)
    with torch.inference_mode():
        out_str = layer_str(x)
    assert out_str.device.type == "mps" and torch.isfinite(out_str).all()

    # 2. Loading quantized on CPU (full_precision_mm=False) then moving .to("mps") must work cleanly
    layer_cpu = ops_quant.Linear(256, 128, bias=True, device="cpu", dtype=torch.bfloat16)
    sd_with_bias = {k: v.clone() for k, v in sd_base.items()}
    sd_with_bias["bias"] = torch.zeros(128, dtype=torch.bfloat16)
    layer_cpu.load_state_dict(sd_with_bias)
    assert isinstance(layer_cpu.weight, omp.QuantizedTensor)
    layer_cpu.to("mps")
    with torch.inference_mode():
        out_moved = layer_cpu(x)
    assert out_moved.device.type == "mps" and torch.isfinite(out_moved).all()

    # 3. Also verify weight_function (e.g. LoRA) path with MPS-moved FP8 QuantizedTensor
    layer_cpu.weight_function = [lambda wt: wt * 1.0]
    with torch.inference_mode():
        out_wf = layer_cpu(x)
    assert out_wf.device.type == "mps" and torch.isfinite(out_wf).all()


# ---------------------------------------------------------------------------
# MPS Embedding FP8 + weight_function & Linear int8_tensorwise (full_precision_mm=False)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
@pytest.mark.parametrize("manual_cast", [False, True])
@pytest.mark.parametrize("move_to_mps", [False, True])
def test_mps_fp8_embedding_with_weight_function(manual_cast, move_to_mps):
    mps = torch.device("mps")
    ops = omp.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[])
    emb = ops.Embedding(16, 8)
    emb.parameters_manual_cast = manual_cast
    qd, _ = TensorCoreFP8E4M3Layout.quantize(torch.randn(16, 8, dtype=torch.bfloat16), scale=1.0)
    emb.load_state_dict(
        {
            "weight": qd,
            "weight_scale": torch.ones((), dtype=torch.float32),
            "comfy_quant": torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8),
        }
    )
    if move_to_mps:
        emb.to(mps)

    idx = torch.tensor([[0, 2, 5]], device=mps)
    with torch.inference_mode():
        base_out = emb(idx)

    emb.weight_function = [lambda w: w * 2.0]
    with torch.inference_mode():
        wf_out = emb(idx)

    assert wf_out.device.type == "mps"
    assert wf_out.dtype == torch.bfloat16
    assert torch.isfinite(wf_out).all()
    torch.testing.assert_close(wf_out.float(), base_out.float() * 2.0, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
@pytest.mark.parametrize("convrot", [False, True])
def test_mps_int8_tensorwise_linear_full_precision_false(monkeypatch, convrot):
    from backend.quant_ops import TensorWiseINT8Layout

    mps = torch.device("mps")
    ops = omp.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[])
    w = torch.randn(128, 256, dtype=torch.bfloat16)
    qd, params = TensorWiseINT8Layout.quantize(w, is_weight=True, per_channel=True, convrot=convrot, convrot_groupsize=256)
    conf = {"format": "int8_tensorwise"}
    if convrot:
        conf["convrot"] = True
        conf["convrot_groupsize"] = 256
    sd_base = {
        "weight": qd,
        "weight_scale": params.scale.float(),
        "bias": torch.zeros(128, dtype=torch.bfloat16),
        "comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8),
    }

    x = torch.randn(1, 8, 256, device=mps, dtype=torch.bfloat16)

    # 1. Direct device="mps" load with full_precision_mm=False, disabled=[]
    layer_mps = ops.Linear(256, 128, bias=True, device=mps, dtype=torch.bfloat16)
    layer_mps.load_state_dict({k: v.clone() for k, v in sd_base.items()})
    assert omp._mps_unsupported_quant(layer_mps.weight, mps) is True

    with torch.inference_mode():
        out_mps = layer_mps(x)
    assert out_mps.device.type == "mps" and torch.isfinite(out_mps).all()

    # 2. CPU load -> .to("mps")
    layer_cpu = ops.Linear(256, 128, bias=True, device="cpu", dtype=torch.bfloat16)
    layer_cpu.load_state_dict({k: v.clone() for k, v in sd_base.items()})
    layer_cpu.to(mps)
    with torch.inference_mode():
        out_moved = layer_cpu(x)
    assert out_moved.device.type == "mps" and torch.isfinite(out_moved).all()

    # 3. Uncached path (REMI_MPS_DEQUANT_CACHE_MB=0) exercises _convrot_linear / _dequantize_quantized_weight directly
    monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", "0")
    with torch.inference_mode():
        out_uncached = layer_mps(x)
    assert out_uncached.device.type == "mps" and torch.isfinite(out_uncached).all()

    # 4. With weight_function (LoRA) attached
    layer_mps.weight_function = [lambda wt: wt * 1.0]
    with torch.inference_mode():
        out_wf = layer_mps(x)
    assert out_wf.device.type == "mps" and torch.isfinite(out_wf).all()


# ---------------------------------------------------------------------------
# MPS CPU-offload (lowvram / parameters_manual_cast=True) Linear
# ---------------------------------------------------------------------------


def _make_cpu_offload_quant_linear_sd(fmt: str, weight: torch.Tensor, *, convrot: bool = False) -> dict[str, torch.Tensor]:
    from backend.quant_ops import (
        AsymW4A8Int8Layout,
        TensorCoreFP8E5M2Layout,
        TensorWiseINT8Layout,
    )

    bias = torch.zeros(weight.shape[0], dtype=weight.dtype)
    if fmt == "float8_e4m3fn":
        qd, params = TensorCoreFP8E4M3Layout.quantize(weight, scale="recalculate")
        return {"weight": qd, "weight_scale": params.scale, "bias": bias, "comfy_quant": torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8)}
    if fmt == "float8_e5m2":
        qd, params = TensorCoreFP8E5M2Layout.quantize(weight, scale="recalculate")
        return {"weight": qd, "weight_scale": params.scale, "bias": bias, "comfy_quant": torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8)}
    if fmt == "nvfp4":
        qd, params = TensorCoreNVFP4Layout_.quantize(weight, scale="recalculate")
        return {"weight": qd, "weight_scale_2": params.scale, "weight_scale": params.block_scale, "bias": bias, "comfy_quant": torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8)}
    if fmt == "mxfp8":
        qd, params = TensorCoreMXFP8Layout_.quantize(weight, scale="recalculate")
        return {"weight": qd, "weight_scale": params.scale, "bias": bias, "comfy_quant": torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8)}
    if fmt == "int8_tensorwise":
        qd, params = TensorWiseINT8Layout.quantize(weight, is_weight=True, per_channel=True, convrot=convrot, convrot_groupsize=256)
        conf = {"format": fmt}
        if convrot:
            conf.update({"convrot": True, "convrot_groupsize": 256})
        return {"weight": qd, "weight_scale": params.scale.float(), "bias": bias, "comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)}
    if fmt == "asym_w4a8_int8":
        qd, params = AsymW4A8Int8Layout.quantize(weight, group_size=16, convrot_groupsize=256)
        sd = {
            "weight": qd,
            "weight_s_rel": params.scale,
            "weight_s_channel": params.s_channel,
            "bias": bias,
            "comfy_quant": torch.tensor(list(json.dumps({"format": fmt, "group_size": 16, "convrot_groupsize": 256}).encode()), dtype=torch.uint8),
        }
        if params.codebook is not None:
            sd["weight_codebook"] = params.codebook
        return sd
    raise ValueError(fmt)


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
@pytest.mark.parametrize(
    ("fmt", "convrot"),
    [
        ("float8_e4m3fn", False),
        ("float8_e5m2", False),
        ("nvfp4", False),
        ("mxfp8", False),
        ("int8_tensorwise", False),
        ("int8_tensorwise", True),
        ("asym_w4a8_int8", False),
    ],
)
def test_mps_cpu_offload_lowvram_linear_quantized_no_cache_device_mismatch(monkeypatch, fmt, convrot):
    mps = torch.device("mps")
    omp.clear_mps_dequant_cache()
    monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", "64")

    gen = torch.Generator().manual_seed(7)
    w = (torch.randn(128, 256, generator=gen) * 0.05).to(torch.bfloat16)
    sd = _make_cpu_offload_quant_linear_sd(fmt, w, convrot=convrot)

    ops = omp.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[])
    layer = ops.Linear(256, 128, bias=True, device="cpu", dtype=torch.bfloat16)
    layer.parameters_manual_cast = True
    layer.load_state_dict({k: v.clone() for k, v in sd.items()})
    assert isinstance(layer.weight, omp.QuantizedTensor)
    assert layer.weight.device.type == "cpu"

    x_cpu = torch.randn(1, 8, 256, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    x_mps = x_cpu.to(mps)

    entries_before = omp.mps_dequant_cache_stats()["entries"]
    with torch.inference_mode():
        ref = torch.nn.functional.linear(
            x_cpu,
            layer.weight.dequantize().to(dtype=x_cpu.dtype),
            layer.bias.to(dtype=x_cpu.dtype),
        )
        out = layer(x_mps)
    entries_after = omp.mps_dequant_cache_stats()["entries"]

    assert out.device.type == "mps"
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 8, 128)
    assert torch.isfinite(out).all()
    assert entries_after == entries_before
    torch.testing.assert_close(out.cpu().float(), ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon MPS")
@pytest.mark.parametrize(("fmt", "convrot"), [("float8_e4m3fn", False), ("int8_tensorwise", True)])
def test_mps_cpu_offload_lowvram_linear_full_precision_zero_cache_and_lora(monkeypatch, fmt, convrot):
    mps = torch.device("mps")
    omp.clear_mps_dequant_cache()
    gen = torch.Generator().manual_seed(11)
    w = (torch.randn(128, 256, generator=gen) * 0.05).to(torch.bfloat16)
    sd = _make_cpu_offload_quant_linear_sd(fmt, w, convrot=convrot)

    x_small = torch.randn(1, 8, 256, device=mps, dtype=torch.bfloat16)
    x_large = torch.randn(1, 128, 256, device=mps, dtype=torch.bfloat16)

    for full_precision_mm in (True, False):
        for cache_mb in ("64", "0"):
            monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", cache_mb)
            ops = omp.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=full_precision_mm, disabled=[])
            layer = ops.Linear(256, 128, bias=True, device="cpu", dtype=torch.bfloat16)
            layer.parameters_manual_cast = True
            layer.load_state_dict({k: v.clone() for k, v in sd.items()})
            assert layer.weight.device.type == "cpu"

            entries_before = omp.mps_dequant_cache_stats()["entries"]
            assert omp._mps_dequant_cache.get_or_dequantize(layer, layer.weight, torch.bfloat16) is None
            with torch.inference_mode():
                out_small = layer(x_small)
                out_large = layer(x_large)
            assert out_small.device.type == "mps" and torch.isfinite(out_small).all()
            assert out_large.device.type == "mps" and torch.isfinite(out_large).all()
            assert omp.mps_dequant_cache_stats()["entries"] == entries_before

            # Attached weight_function (LoRA) with CPU-resident weight & MPS input
            layer.weight_function = [lambda wt: wt * 1.0]
            with torch.inference_mode():
                out_wf = layer(x_small)
            assert out_wf.device.type == "mps" and torch.isfinite(out_wf).all()
            assert omp.mps_dequant_cache_stats()["entries"] == entries_before
    omp.clear_mps_dequant_cache()


# ---------------------------------------------------------------------------
# Neo-decoupled runtime fixes: sd_vae CPU offload & loaded_vae_file + kdiffusion zero-denoise
# ---------------------------------------------------------------------------


def test_sd_vae_store_reload_restore_cpu_offload_and_double_override_safety(monkeypatch, tmp_path):
    from remi.testing import webui_state

    with webui_state():
        from modules import sd_vae, shared

        dev = torch.device("mps") if HAS_MPS else torch.device("cpu")
        vae_mod = torch.nn.Linear(4, 4, bias=True).to(device=dev, dtype=torch.float32)
        orig_weight = vae_mod.weight.detach().cpu().clone()
        ckpt_info = SimpleNamespace(filename="base.safetensors")
        fake_model = SimpleNamespace(first_stage_model=vae_mod, sd_checkpoint_info=ckpt_info)

        monkeypatch.setattr(shared, "sd_model", fake_model)
        monkeypatch.setattr(sd_vae, "base_vae", None)
        monkeypatch.setattr(sd_vae, "loaded_vae_file", None)
        monkeypatch.setattr(sd_vae, "checkpoint_info", None)

        vae_path_1 = tmp_path / "custom_a.safetensors"
        vae_path_2 = tmp_path / "custom_b.safetensors"
        vae_path_1.write_bytes(b"vae-a")
        vae_path_2.write_bytes(b"vae-b")

        override_1 = {
            "weight": torch.full((4, 4), 2.5, dtype=torch.float32),
            "bias": torch.zeros(4, dtype=torch.float32),
            "loss.unused": torch.ones(1),
        }
        override_2 = {
            "weight": torch.full((4, 4), -3.0, dtype=torch.float32),
            "bias": torch.ones(4, dtype=torch.float32),
        }
        sd_by_path = {str(vae_path_1): override_1, str(vae_path_2): override_2}
        monkeypatch.setattr(sd_vae.utils, "load_torch_file", lambda p: sd_by_path[str(p)])

        try:
            assert sd_vae.reload_vae_weights(str(vae_path_1)) is True
            assert sd_vae.get_loaded_vae_name() == "custom_a.safetensors"
            assert sd_vae.base_vae is not None
            # Stored base_vae tensors are cloned onto CPU rather than retained on GPU/MPS
            assert all(t.device.type == "cpu" for t in sd_vae.base_vae.values() if isinstance(t, torch.Tensor))
            torch.testing.assert_close(vae_mod.weight.detach().cpu(), override_1["weight"])

            # A second override before restore must NOT overwrite base_vae with override_1
            assert sd_vae.reload_vae_weights(str(vae_path_2)) is True
            assert sd_vae.get_loaded_vae_name() == "custom_b.safetensors"
            torch.testing.assert_close(vae_mod.weight.detach().cpu(), override_2["weight"])
            torch.testing.assert_close(sd_vae.base_vae["weight"], orig_weight)

            # Restoring puts original base weights back on the model's device and clears loaded_vae_file
            sd_vae.restore_vae_weights()
            assert sd_vae.loaded_vae_file is None
            assert sd_vae.get_loaded_vae_name() is None
            assert sd_vae.base_vae is None
            assert vae_mod.weight.device.type == dev.type
            torch.testing.assert_close(vae_mod.weight.detach().cpu(), orig_weight)
        finally:
            sd_vae.delete_base_vae()
            sd_vae.loaded_vae_file = None


def test_kdiffusion_sample_img2img_zero_denoise_and_short_sigma_sched(monkeypatch):
    from remi.testing import webui_state

    with webui_state():
        from modules import sd_samplers_kdiffusion, shared

        monkeypatch.setattr(sd_samplers_kdiffusion, "sampling_prepare", lambda unet, x: None)

        called = []

        def fake_func(model, x, sigmas, extra_args=None, **kwargs):
            called.append((x, sigmas))
            return x + 1.0

        sampler = object.__new__(sd_samplers_kdiffusion.KDiffusionSampler)
        sampler.func = fake_func
        sampler.config = SimpleNamespace(options={"discard_next_to_last_sigma": True})
        sampler.model_wrap = SimpleNamespace(
            inner_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=object())),
            predictor=SimpleNamespace(noise_scaling=lambda sigma, noise, x, max_denoise=False: x + noise),
        )
        sampler.get_sigmas = lambda p, n: torch.tensor([0.0], dtype=torch.float32)
        sampler.initialize = lambda p: {}

        x = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(x)

        # 1. denoising_strength == 0.0 returns x untouched without calling self.func
        p_zero = SimpleNamespace(denoising_strength=0.0, steps=20, extra_generation_params={})
        out_zero = sampler.sample_img2img(p_zero, x, noise, None, None)
        assert torch.equal(out_zero, x)
        assert called == []

        # 2. Short 1-element sigma_sched (e.g. discard_next_to_last_sigma at 1 step) returns x instead of IndexError on sigma_sched[-2]
        def fake_sigma_min_func(model, x, sigma_min, sigma_max, n, **kwargs):
            called.append((sigma_min, sigma_max, n))
            return x + 1.0

        sampler.func = fake_sigma_min_func
        monkeypatch.setattr(shared.opts, "img2img_fix_steps", False, raising=False)
        p_short = SimpleNamespace(denoising_strength=0.5, steps=1, extra_generation_params={})
        out_short = sampler.sample_img2img(p_short, x, noise, None, None)
        assert torch.equal(out_short, x)
        assert called == []
