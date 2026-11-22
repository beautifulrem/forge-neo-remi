"""Comprehensive tests for Qwen-Image-2.1 (7B) deep support:

- Diffusers pipeline & sharded transformer directory discovery/loading (Base & Turbo);
- Exact 238-tensor Diffusers ``AutoencoderKLQwenImage21`` VAE conversion;
- Unsloth TorchAO FP8 / INT8 / INT8-ConvRot DiT & text encoder loading;
- GGUF Q8_0 forward parity;
- Lossless cross-step block-causal prefix KV cache (txt2img, single/multi-ref,
  batched & split CFG, exact invalidation, VRAM budget guard);
- On-the-fly FP8 / INT8-ConvRot quantization with sensitive-layer protection;
- Bidirectional fused/unfused MLP LoRA key mapping across all trainers;
- ``Qwen-2.1 per-block`` ``torch.compile`` preset;
- Spectrum pre-head feature forecasting;
- Workflow advice (Base vs Turbo, references, >10 refs, RGBA + JPEG);
- End-to-end 4-channel RGBA transparent generation, reference encoding, and
  mask editing (Whole picture & Only masked, standard & precise mask).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from backend import operations
from backend.args import dynamic_args
from backend.diffusion_engine.qwen21 import QwenImage21
from backend.nn.anima_diffusers import is_diffusers_wan_vae
from backend.nn.anima_quant import (
    keep_sensitive_weights_precise,
    quantize_int8_convrot_state_dict,
    scale_fp8_state_dict,
    should_keep_sensitive_weights_precise,
    should_quantize_on_load,
)
from backend.nn.dit_registry import (
    capability_for_dit_cls,
    capability_for_model,
    capability_for_state_dict,
    collect_workflow_advice,
    diffusers_checkpoint_info,
    fast_forward_qwen21,
    get_dit_capability,
    is_diffusers_checkpoint_dir,
    load_diffusers_checkpoint_dir,
    match_spectrum_capability,
    qwen21_workflow_advice,
    resolve_spectrum_capture_module,
)
from backend.nn.qwen21 import QwenImage21Transformer2DModel
from backend.nn.qwen21_diffusers import (
    convert_diffusers_qwen21_vae,
    is_diffusers_qwen21_vae,
    is_qwen21_diffusers_dir,
    is_sharded_qwen21_transformer_dir,
    qwen21_lora_key_map,
)
from backend.nn.qwen21_vae import Wan22VAE
from backend.state_dict import convert_quantization, detect_quantization

REPO = pathlib.Path(__file__).resolve().parents[1]


def _tiny_qwen21(*, num_layers: int = 2, fused_mlp: bool = False, seed: int = 0) -> QwenImage21Transformer2DModel:
    torch.manual_seed(seed)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        model = QwenImage21Transformer2DModel(
            in_channels=8,
            out_channels=8,
            num_layers=num_layers,
            attention_head_dim=32,
            num_attention_heads=2,
            context_in_dim=32,
            mlp_ratio=2,
            axes_dims_rope=(8, 12, 12),
            fused_mlp=fused_mlp,
        ).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.05)
    return model


def _wide_qwen21(*, fused_mlp: bool = False) -> QwenImage21Transformer2DModel:
    """Block linears have in_features divisible by 256 (for INT8-ConvRot)."""
    torch.manual_seed(1)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        model = QwenImage21Transformer2DModel(
            in_channels=16,
            out_channels=16,
            num_layers=2,
            attention_head_dim=64,
            num_attention_heads=4,
            context_in_dim=64,
            mlp_ratio=2,
            axes_dims_rope=(16, 24, 24),
            fused_mlp=fused_mlp,
        ).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
    return model


def _native_to_diffusers_qwen21_vae(native_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Invert ``convert_diffusers_qwen21_vae`` to build a realistic 238-tensor Diffusers VAE dict."""

    reverse_resnet = {
        "residual.0.": "norm1.",
        "residual.2.": "conv1.",
        "residual.3.": "norm2.",
        "residual.6.": "conv2.",
        "shortcut.": "conv_shortcut.",
    }

    def unmap_resnet(rest: str) -> str:
        for src, dst in reverse_resnet.items():
            if rest.startswith(src):
                return dst + rest[len(src) :]
        return rest

    out: dict[str, torch.Tensor] = {}
    for k, v in native_sd.items():
        val = v.squeeze(2) if v.ndim == 5 and v.shape[2] == 1 else v
        if k.startswith("conv1."):
            dk = "quant_conv." + k[len("conv1.") :]
        elif k.startswith("conv2."):
            dk = "post_quant_conv." + k[len("conv2.") :]
        elif k.startswith(("encoder.", "decoder.")):
            side, rest = k.split(".", 1)
            if rest.startswith("conv1."):
                dk = f"{side}.conv_in.{rest[len('conv1.') :]}"
            elif rest.startswith("head.0."):
                dk = f"{side}.norm_out.{rest[len('head.0.') :]}"
            elif rest.startswith("head.2."):
                dk = f"{side}.conv_out.{rest[len('head.2.') :]}"
            elif rest.startswith("middle.0."):
                dk = f"{side}.mid_block.resnets.0.{unmap_resnet(rest[len('middle.0.') :])}"
            elif rest.startswith("middle.1."):
                dk = f"{side}.mid_block.attentions.0.{rest[len('middle.1.') :]}"
            elif rest.startswith("middle.2."):
                dk = f"{side}.mid_block.resnets.1.{unmap_resnet(rest[len('middle.2.') :])}"
            elif side == "encoder" and rest.startswith("downsamples."):
                parts = rest.split(".", 4)
                stage, slot, tail = parts[1], int(parts[3]), parts[4]
                if tail.startswith(("resample.", "time_conv.")):
                    dk = f"encoder.down_blocks.{stage}.downsampler.{tail}"
                else:
                    dk = f"encoder.down_blocks.{stage}.resnets.{slot}.{unmap_resnet(tail)}"
            elif side == "decoder" and rest.startswith("upsamples."):
                parts = rest.split(".", 4)
                stage, slot, tail = parts[1], int(parts[3]), parts[4]
                if tail.startswith(("resample.", "time_conv.")):
                    dk = f"decoder.up_blocks.{stage}.upsampler.{tail}"
                else:
                    dk = f"decoder.up_blocks.{stage}.resnets.{slot}.{unmap_resnet(tail)}"
            else:
                dk = k
        else:
            dk = k
        out[dk] = val
    return out


# region 1. Detection, Diffusers VAE, Diffusers & Sharded Directories, Unsloth TorchAO, GGUF


def test_diffusers_qwen21_vae_exact_238_tensor_bijection_and_rgba_forward():
    native_vae = Wan22VAE(
        dim=96,
        dec_dim=144,
        z_dim=64,
        dim_mult=[1, 2, 4, 8, 8],
        num_res_blocks=2,
        attn_scales=[],
        temporal_downsample=[False, True, True, True],
        dropout=0.0,
        image_channels=4,
        patch_size=1,
        temporal_kernel=1,
    )
    native_sd = native_vae.state_dict()
    assert len(native_sd) == 238

    diffusers_sd = _native_to_diffusers_qwen21_vae(native_sd)
    assert len(diffusers_sd) == 238
    assert diffusers_sd["decoder.conv_in.weight"].ndim == 4
    assert diffusers_sd["encoder.down_blocks.0.resnets.0.conv1.weight"].ndim == 4
    assert is_diffusers_qwen21_vae(diffusers_sd) is True
    # Must NOT collide with Wan 2.1's flat AutoencoderKLWan converter.
    assert is_diffusers_wan_vae(diffusers_sd) is False

    converted = convert_diffusers_qwen21_vae(diffusers_sd)
    assert set(converted.keys()) == set(native_sd.keys())
    for k, expected in native_sd.items():
        assert converted[k].shape == expected.shape, f"shape mismatch for {k}: {converted[k].shape} vs {expected.shape}"

    # Verify real 4-channel RGBA encode + decode on a compact Wan22VAE loaded from converted Diffusers weights.
    small_vae = Wan22VAE(
        dim=8,
        dec_dim=8,
        z_dim=8,
        dim_mult=[1, 2, 2],
        num_res_blocks=1,
        attn_scales=[],
        temporal_downsample=[False, False],
        dropout=0.0,
        image_channels=4,
        patch_size=1,
        temporal_kernel=1,
    ).eval()
    with torch.no_grad():
        for p in small_vae.parameters():
            p.normal_(0.0, 0.02)
        small_diff = _native_to_diffusers_qwen21_vae(small_vae.state_dict())
        assert is_diffusers_qwen21_vae(small_diff) is True
        small_conv = convert_diffusers_qwen21_vae(small_diff)
        reloaded = Wan22VAE(
            dim=8,
            dec_dim=8,
            z_dim=8,
            dim_mult=[1, 2, 2],
            num_res_blocks=1,
            attn_scales=[],
            temporal_downsample=[False, False],
            dropout=0.0,
            image_channels=4,
            patch_size=1,
            temporal_kernel=1,
        ).eval()
        missing, unexpected = reloaded.load_state_dict(small_conv, strict=True)
        assert not missing and not unexpected
        rgba = torch.randn(1, 4, 16, 16)
        lat = reloaded.encode(rgba)
        rec = reloaded.decode(lat)
        assert lat.shape == (1, 8, 4, 4)
        assert rec.shape == (1, 4, 16, 16)
        torch.testing.assert_close(rec, small_vae.decode(small_vae.encode(rgba)), atol=1e-6, rtol=1e-6)


def test_diffusers_pipeline_and_sharded_transformer_dir_loading(tmp_path):
    from huggingface_guess import detection, model_list

    from backend.loader import _load_diffusers_dir, replace_state_dict

    tiny = _tiny_qwen21(fused_mlp=False)
    t_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in tiny.state_dict().items()}
    keys = sorted(t_sd.keys())
    half = len(keys) // 2

    # 1. Full Qwen-Image-2.1-Turbo Diffusers directory with sharded transformer + TE + Diffusers RGBA VAE
    pipe_dir = tmp_path / "Qwen-Image-2.1-Turbo"
    (pipe_dir / "transformer").mkdir(parents=True)
    (pipe_dir / "text_encoder").mkdir(parents=True)
    (pipe_dir / "vae").mkdir(parents=True)
    (pipe_dir / "scheduler").mkdir(parents=True)

    (pipe_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImage21Pipeline",
                "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
                "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                "vae": ["diffusers", "AutoencoderKLQwenImage21"],
                "sample_sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568],
            }
        ),
        encoding="utf-8",
    )
    (pipe_dir / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler", "use_dynamic_shifting": False}),
        encoding="utf-8",
    )
    (pipe_dir / "transformer" / "config.json").write_text(json.dumps({"_class_name": "QwenImage21Transformer2DModel"}), encoding="utf-8")

    s1 = "diffusion_pytorch_model-00001-of-00002.safetensors"
    s2 = "diffusion_pytorch_model-00002-of-00002.safetensors"
    save_file({k: t_sd[k] for k in keys[:half]}, str(pipe_dir / "transformer" / s1))
    save_file({k: t_sd[k] for k in keys[half:]}, str(pipe_dir / "transformer" / s2))
    (pipe_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {**dict.fromkeys(keys[:half], s1), **dict.fromkeys(keys[half:], s2)}}),
        encoding="utf-8",
    )

    # Bundled Qwen3-VL-8B text encoder (Unprefixed language_model.* + visual.*)
    te_sd = {
        "language_model.embed_tokens.weight": torch.zeros(16, 4096, dtype=torch.bfloat16),
        "visual.deepstack_merger_list.0.norm.weight": torch.ones(1152, dtype=torch.bfloat16),
        "visual.merger.linear_fc2.weight": torch.zeros(4096, 4608, dtype=torch.bfloat16),
    }
    save_file(te_sd, str(pipe_dir / "text_encoder" / "model.safetensors"))

    # Bundled Diffusers 4ch RGBA VAE
    small_vae = Wan22VAE(dim=8, dec_dim=8, z_dim=8, dim_mult=[1, 2, 2], num_res_blocks=1, attn_scales=[], temporal_downsample=[False, False], image_channels=4, patch_size=1, temporal_kernel=1)
    diff_vae_sd = _native_to_diffusers_qwen21_vae(small_vae.state_dict())
    save_file(diff_vae_sd, str(pipe_dir / "vae" / "diffusion_pytorch_model.safetensors"))

    assert is_qwen21_diffusers_dir(pipe_dir) is True
    assert is_diffusers_checkpoint_dir(pipe_dir) is True
    arch, primary = diffusers_checkpoint_info(pipe_dir)
    assert arch == "qwen-image-2.1/diffusers" and primary.endswith(s1)

    loaded_sd, meta, guess = _load_diffusers_dir(pipe_dir)
    assert isinstance(guess, model_list.QwenImage21)
    assert guess.unet_config["fused_mlp"] is False
    assert guess.unet_config["num_layers"] == 2
    assert guess.sampling_settings["use_dynamic_shifting"] is False
    assert "qwen21_sample_sigmas" in meta
    assert "model.diffusion_model.img_in.weight" in loaded_sd
    assert "text_encoders.qwen3vl_8b.transformer.model.language_model.embed_tokens.weight" in loaded_sd
    assert "vae.decoder.upsamples.0.upsamples.0.residual.2.weight" in loaded_sd
    assert loaded_sd["vae.decoder.conv1.weight"].ndim == 5

    # Ensure class-level sampling_settings was NOT mutated for Base checkpoints.
    assert model_list.QwenImage21.sampling_settings.get("use_dynamic_shifting") is None

    # 2. Standalone sharded transformer folder
    sharded_dir = tmp_path / "qwen21_transformer_only"
    sharded_dir.mkdir()
    save_file({k: t_sd[k] for k in keys[:half]}, str(sharded_dir / s1))
    save_file({k: t_sd[k] for k in keys[half:]}, str(sharded_dir / s2))
    (sharded_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {**dict.fromkeys(keys[:half], s1), **dict.fromkeys(keys[half:], s2)}}),
        encoding="utf-8",
    )
    assert is_sharded_qwen21_transformer_dir(sharded_dir) is True
    assert is_diffusers_checkpoint_dir(sharded_dir) is True
    only_sd, _ = load_diffusers_checkpoint_dir(sharded_dir)
    assert "model.diffusion_model.txt_in.text_norm.weight" in only_sd
    assert not any(k.startswith("vae.") for k in only_sd)

    # 3. Standalone Diffusers Qwen-Image-2.1 VAE loaded via replace_state_dict
    base_sd = {f"model.diffusion_model.{k}": v for k, v in t_sd.items()}
    base_guess = detection.model_config_from_unet(base_sd, "model.diffusion_model.")
    replaced = replace_state_dict(base_sd, dict(diff_vae_sd), {}, base_guess, tmp_path / "diffusion_pytorch_model.safetensors")
    assert "vae.decoder.middle.0.residual.0.gamma" in replaced
    assert replaced["vae.encoder.conv1.weight"].ndim == 5


def test_unsloth_torchao_fp8_int8_convrot_dit_and_text_encoder():
    from huggingface_guess import detection, model_list

    from backend.loader import replace_state_dict

    # 1. Unsloth FP8 TorchAO DiT (_weight_qdata F8_E4M3 + _weight_scale [out, 1])
    model = _wide_qwen21(fused_mlp=False)
    fp8_raw: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.startswith("transformer_blocks.") and v.ndim == 2:
            stem = k[: -len(".weight")]
            scale = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
            fp8_raw[f"{stem}._weight_qdata"] = (v / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            fp8_raw[f"{stem}._weight_scale"] = scale.float()
        else:
            fp8_raw[k] = v.to(torch.bfloat16)

    converted_fp8, _ = convert_quantization(fp8_raw, {"unsloth_format": "unsloth_prequant_transformer_state_dict_v1"})
    assert detect_quantization(converted_fp8, is_unet=True) == {"mixed_ops": True, "TE": False, "layout": "RowWiseFP8E4M3Layout"}
    prefixed = {f"model.diffusion_model.{k}": v for k, v in converted_fp8.items()}
    guess = detection.model_config_from_unet(prefixed, "model.diffusion_model.")
    assert isinstance(guess, model_list.QwenImage21) and guess.unet_config["fused_mlp"] is False

    # 2. Unsloth INT8-ConvRot TorchAO DiT (_weight_qdata I8 + _weight_scale + zero _weight_zero_point + activation_rotation_fqns)
    fqns = [k[: -len(".weight")] for k, v in model.state_dict().items() if k.startswith("transformer_blocks.") and v.ndim == 2]
    int8_raw: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.startswith("transformer_blocks.") and v.ndim == 2:
            stem = k[: -len(".weight")]
            scale = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0
            int8_raw[f"{stem}._weight_qdata"] = (v / scale).round().clamp(-127, 127).to(torch.int8)
            int8_raw[f"{stem}._weight_scale"] = scale.float()
            int8_raw[f"{stem}._weight_zero_point"] = torch.zeros(v.shape[0], 1, dtype=torch.int8)
        else:
            int8_raw[k] = v.to(torch.bfloat16)

    meta = {
        "unsloth_format": "unsloth_prequant_transformer_state_dict_v2",
        "unsloth_metadata": json.dumps(
            {
                "family": "qwen-image-2.1",
                "scheme": "int8",
                "activation_rotation": "convrot_hadamard_v1",
                "activation_rotation_group": 256,
                "activation_rotation_fqns": fqns,
            }
        ),
    }
    converted_int8, _ = convert_quantization(int8_raw, meta)
    assert detect_quantization(converted_int8, is_unet=True) == {"mixed_ops": True, "TE": False, "layout": "TensorWiseINT8Layout"}
    conf = json.loads(converted_int8["transformer_blocks.0.attn.to_q.comfy_quant"].numpy().tobytes())
    assert conf == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    assert "img_in.comfy_quant" not in converted_int8
    assert "modulation.1.comfy_quant" not in converted_int8

    # 3. Unsloth Qwen-Image-2.1 text_encoder INT8-ConvRot & FP8 routed to qwen3vl_8b via replace_state_dict
    te_int8 = {
        "model.language_model.embed_tokens.weight": torch.zeros(8, 4096, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(4096, 4096, dtype=torch.int8),
        "model.language_model.layers.0.self_attn.q_proj.weight_scale": torch.ones(4096, 1),
        "model.language_model.layers.0.self_attn.q_proj.comfy_quant": torch.tensor(list(b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'), dtype=torch.uint8),
        "model.visual.deepstack_merger_list.0.norm.weight": torch.ones(1152, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc2.weight": torch.zeros(4096, 4608, dtype=torch.bfloat16),
    }
    out_sd = replace_state_dict({}, te_int8, {}, guess, pathlib.Path("Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors"))
    assert "text_encoders.qwen3vl_8b.transformer.model.language_model.layers.0.self_attn.q_proj.comfy_quant" in out_sd
    assert detect_quantization(out_sd, is_unet=False) == {"mixed_ops": True, "TE": True, "layout": "TensorWiseINT8Layout"}


def test_gguf_q8_0_forward_parity():
    import gguf

    from backend.operations import ForgeOperationsGGUF, using_forge_operations
    from backend.operations_gguf import ParameterGGUF
    from backend.state_dict import load_state_dict

    ref = _tiny_qwen21(num_layers=2, fused_mlp=True)
    x = torch.randn(1, 8, 4, 4)
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 6, 32)
    with torch.inference_mode():
        ref_out = ref(x, t, ctx)

    q_sd = {}
    for k, v in ref.state_dict().items():
        if v.ndim == 2 and v.shape[-1] % 32 == 0 and k.startswith("transformer_blocks."):
            q_np = gguf.quants.quantize(v.detach().cpu().float().numpy(), gguf.GGMLQuantizationType.Q8_0)
            param = ParameterGGUF(torch.from_numpy(q_np.copy()), tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=v.shape)
            param.gguf_cls.bake(param)
            q_sd[k] = param
        else:
            q_sd[k] = v.detach().clone()

    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True, extra_dtype="gguf"):
        q_model = QwenImage21Transformer2DModel(
            in_channels=8,
            out_channels=8,
            num_layers=2,
            attention_head_dim=32,
            num_attention_heads=2,
            context_in_dim=32,
            mlp_ratio=2,
            axes_dims_rope=(8, 12, 12),
            fused_mlp=True,
        ).eval()
    load_state_dict(q_model, q_sd)
    assert isinstance(q_model.transformer_blocks[0].attn.to_q, ForgeOperationsGGUF.Linear)

    with torch.inference_mode():
        q_out = q_model(x, t, ctx)
    cos = torch.nn.functional.cosine_similarity(ref_out.flatten(), q_out.flatten(), dim=0).item()
    assert cos > 0.999, f"GGUF Q8_0 cosine similarity too low: {cos}"


# endregion


# region 2. Lossless Cross-Step Prefix KV Cache


@pytest.mark.parametrize("fused_mlp", [False, True])
def test_prefix_kv_cache_bit_exact_parity_across_modes(fused_mlp):
    """Cached prefix KV matches full uncached block-causal forward across
    txt2img, single ref (even & odd target parity), multi-ref with image_slots,
    batched CFG (fresh torch.cat tensors every step), and split CFG."""

    model = _tiny_qwen21(num_layers=3, fused_mlp=fused_mlp, seed=42)

    def uncached_forward(x, t, ctx, refs=(), slots=()):
        # Passing a no-op single_block patch forces the uncached reference path.
        opts = {"patches": {"single_block": [lambda a: a]}}
        with torch.inference_mode():
            return model(x, t, ctx, ref_latents=list(refs), image_slots=list(slots), transformer_options=opts)

    prefill_calls = 0
    orig_prefill = model.preprocess_prefix

    def counted_prefill(*args, **kwargs):
        nonlocal prefill_calls
        prefill_calls += 1
        return orig_prefill(*args, **kwargs)

    model.preprocess_prefix = counted_prefill

    # Case A: txt2img across 3 denoising steps with fresh torch.cat context tensors each step (like cond_cat)
    base_ctx = torch.randn(1, 7, 32)
    for step_t in (0.9, 0.5, 0.1):
        x = torch.randn(1, 8, 4, 4)
        t = torch.tensor([step_t])
        fresh_ctx = torch.cat([base_ctx])  # new data_ptr every step!
        with torch.inference_mode():
            cached = model(x, t, fresh_ctx, ref_latents=[], image_slots=[])
        expected = uncached_forward(x, t, fresh_ctx)
        torch.testing.assert_close(cached, expected, atol=1e-5, rtol=1e-5)
    assert prefill_calls == 1

    # Case B: multi-reference (3 refs, mixed shapes, interleaved image_slots) + odd target (3x5) vs even target (4x4)
    refs = [torch.randn(1, 8, 3, 4), torch.randn(1, 8, 2, 2), torch.randn(1, 8, 4, 3)]
    slots = [2, 4, 6]
    ctx_multi = torch.randn(1, 9, 32)
    prefill_calls = 0
    for step_t in (0.8, 0.3):
        x_odd = torch.randn(1, 8, 3, 5)
        with torch.inference_mode():
            cached = model(x_odd, torch.tensor([step_t]), ctx_multi.clone(), ref_latents=refs, image_slots=slots)
        expected = uncached_forward(x_odd, torch.tensor([step_t]), ctx_multi, refs, slots)
        torch.testing.assert_close(cached, expected, atol=1e-5, rtol=1e-5)
    assert prefill_calls == 1

    # Changing target parity (3,5) -> (4,4) MUST invalidate because ref RoPE has 0.5*(h%2 - H%2)!
    x_even = torch.randn(1, 8, 4, 4)
    with torch.inference_mode():
        cached_even = model(x_even, torch.tensor([0.5]), ctx_multi.clone(), ref_latents=refs, image_slots=slots)
    expected_even = uncached_forward(x_even, torch.tensor([0.5]), ctx_multi, refs, slots)
    torch.testing.assert_close(cached_even, expected_even, atol=1e-5, rtol=1e-5)
    assert prefill_calls == 2

    # Case C: Batched CFG (B=2, fresh torch.cat([c, uc]) every step)
    model.clear_caches()
    prefill_calls = 0
    c = torch.randn(1, 5, 32)
    uc = torch.randn(1, 5, 32)
    for step_t in (0.9, 0.4, 0.1):
        x2 = torch.randn(2, 8, 4, 4)
        t2 = torch.tensor([step_t, step_t])
        ctx2 = torch.cat([c, uc], dim=0)
        with torch.inference_mode():
            cached2 = model(x2, t2, ctx2, ref_latents=[refs[0]], image_slots=[2])
        expected2 = uncached_forward(x2, t2, ctx2, [refs[0]], [2])
        torch.testing.assert_close(cached2, expected2, atol=1e-5, rtol=1e-5)
    assert prefill_calls == 1

    # Case D: Split CFG (cond B=1 and uncond B=1 alternating) — both entries stay warm in LRU(2)
    model.clear_caches()
    prefill_calls = 0
    for step_t in (0.9, 0.5, 0.2):
        for branch_ctx in (c, uc):
            x1 = torch.randn(1, 8, 4, 4)
            with torch.inference_mode():
                out = model(x1, torch.tensor([step_t]), branch_ctx.clone(), ref_latents=[refs[0]], image_slots=[2])
            exp = uncached_forward(x1, torch.tensor([step_t]), branch_ctx, [refs[0]], [2])
            torch.testing.assert_close(out, exp, atol=1e-5, rtol=1e-5)
    assert prefill_calls == 2  # 1 for cond + 1 for uncond across all 3 steps!


def test_prefix_kv_cache_invalidation_and_vram_budget_fallback(monkeypatch):
    model = _tiny_qwen21(num_layers=2, fused_mlp=False, seed=7)
    prefill_calls = 0
    orig_prefill = model.preprocess_prefix

    def counted(*args, **kwargs):
        nonlocal prefill_calls
        prefill_calls += 1
        return orig_prefill(*args, **kwargs)

    model.preprocess_prefix = counted

    x = torch.randn(1, 8, 4, 4)
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 5, 32)
    with torch.inference_mode():
        y1 = model(x, t, ctx)
        y2 = model(x, t, ctx)
    assert prefill_calls == 1
    torch.testing.assert_close(y1, y2)

    # 1. Prompt change invalidates
    ctx2 = ctx + 0.5
    with torch.inference_mode():
        y_changed = model(x, t, ctx2)
    assert prefill_calls == 2
    assert not torch.allclose(y1, y_changed)

    # 2. Weight mutation (LoRA patch) invalidates
    with torch.no_grad():
        model.transformer_blocks[0].attn.to_k.weight.add_(0.1)
    with torch.inference_mode():
        y_lora = model(x, t, ctx2)
    assert prefill_calls == 3
    assert not torch.allclose(y_changed, y_lora)

    # 3. Online weight_function hook (and grad mode) bypasses cache cleanly
    assert model._prefix_cacheable({}) is False  # grad enabled outside inference_mode
    with torch.inference_mode():
        model.txt_in.in_layer.weight_function = [lambda w: w]
        assert model._prefix_cacheable({}) is False
        model.txt_in.in_layer.weight_function = []
        assert model._prefix_cacheable({}) is True

    # 4. SLG patches_replace bypasses cache cleanly
    assert model._prefix_cacheable({"patches_replace": {"dit": {("single_block", 0): lambda a, _e: a}}}) is False

    # 5. _apply() clears cache
    assert len(model._prefix_kv_cache) > 0
    model.to(dtype=torch.float32)
    assert len(model._prefix_kv_cache) == 0

    # 6. Low-VRAM budget guard gracefully falls back to uncached path without OOM
    monkeypatch.setattr(model, "_prefix_kv_fits_budget", lambda *a, **kw: False)
    prefill_calls = 0
    with torch.inference_mode():
        y_fallback = model(x, t, ctx2)
    assert prefill_calls == 0
    torch.testing.assert_close(y_fallback, y_lora, atol=1e-5, rtol=1e-5)


# endregion


# region 3. On-the-Fly Quantization, Bidirectional LoRA, Per-Block Compile, Spectrum, Workflow Advice


@pytest.mark.parametrize("fused_mlp", [False, True])
def test_on_the_fly_quantization_and_sensitive_layer_protection(fused_mlp):
    assert capability_for_dit_cls("QwenImage21Transformer2DModel") is get_dit_capability("qwen21")
    assert should_keep_sensitive_weights_precise("QwenImage21Transformer2DModel", torch.float8_e4m3fn, torch.bfloat16) is True
    assert should_keep_sensitive_weights_precise("QwenImage21Transformer2DModel", torch.float8_e4m3fn, torch.float8_e4m3fn) is False
    assert should_quantize_on_load("scaled_fp8", "QwenImage21Transformer2DModel", None, False, torch.bfloat16) is True
    assert should_quantize_on_load("int8_convrot", "QwenImage21Transformer2DModel", None, False, torch.bfloat16) is True

    model = _wide_qwen21(fused_mlp=fused_mlp)
    assert capability_for_model(model) is get_dit_capability("qwen21")

    # On-the-fly FP8 storage: sensitive prefixes + norms stay BF16, block linears stay FP8
    fp8_model = _wide_qwen21(fused_mlp=fused_mlp).to(torch.float8_e4m3fn)
    kept = keep_sensitive_weights_precise(fp8_model, torch.bfloat16)
    assert kept > 0
    assert fp8_model.img_in.weight.dtype == torch.bfloat16
    assert fp8_model.txt_in.text_norm.weight.dtype == torch.bfloat16
    assert fp8_model.txt_in.in_layer.weight.dtype == torch.bfloat16
    assert fp8_model.txt_in.out_layer.weight.dtype == torch.bfloat16
    assert fp8_model.time_text_embed.timestep_embedder.linear_1.weight.dtype == torch.bfloat16
    assert fp8_model.modulation[1].weight.dtype == torch.bfloat16
    assert fp8_model.norm_out.linear.weight.dtype == torch.bfloat16
    assert fp8_model.proj_out.weight.dtype == torch.bfloat16
    assert fp8_model.transformer_blocks[0].attn.norm_q.weight.dtype == torch.bfloat16
    assert fp8_model.transformer_blocks[0].attn.norm_k.weight.dtype == torch.bfloat16
    assert fp8_model.transformer_blocks[0].attn.to_q.weight.dtype == torch.float8_e4m3fn

    # Scaled FP8 (rowwise)
    bf16_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in model.state_dict().items()}
    assert capability_for_state_dict(bf16_sd) is get_dit_capability("qwen21")
    fp8_sd = dict(bf16_sd)
    expected_block_linears = 2 * (6 if fused_mlp else 7)
    assert scale_fp8_state_dict(fp8_sd, fp8_matmul=False) == expected_block_linears
    assert fp8_sd["transformer_blocks.0.attn.to_q.weight"].dtype == torch.float8_e4m3fn
    assert fp8_sd["transformer_blocks.0.attn.to_q.weight_scale"].shape == (256, 1)
    for sensitive_prefix in ("img_in.", "txt_in.in_layer.", "modulation.1.", "norm_out.linear.", "proj_out."):
        assert f"{sensitive_prefix}comfy_quant" not in fp8_sd

    # INT8-ConvRot
    int8_sd = dict(bf16_sd)
    assert quantize_int8_convrot_state_dict(int8_sd) == expected_block_linears
    assert int8_sd["transformer_blocks.0.attn.to_q.weight"].dtype == torch.int8
    assert int8_sd["transformer_blocks.0.img_mlp.out.weight"].dtype == torch.int8
    assert "modulation.1.comfy_quant" not in int8_sd


def test_bidirectional_lora_key_map_and_load_lora():
    from modules_forge.packages.comfy.lora import SplitTargets, load_lora, model_lora_keys_unet
    from modules_forge.packages.comfy.weight_adapter.base import SlicedWeightAdapter

    # 1. Fused ComfyUI checkpoint accepts BOTH fused gate_up AND unfused gate_layer/proj LoRAs
    fused_model = _tiny_qwen21(num_layers=1, fused_mlp=True)
    fused_sd = {f"diffusion_model.{k}": v for k, v in fused_model.state_dict().items()}
    fake_kmodel_fused = SimpleNamespace(
        state_dict=lambda: fused_sd,
        diffusion_model=SimpleNamespace(config={}),
        config=SimpleNamespace(huggingface_repo="Qwen/Qwen-Image-2.1"),
    )
    km_fused = model_lora_keys_unet(fake_kmodel_fused)
    assert km_fused["transformer.transformer_blocks.0.img_mlp.gate_up"] == "diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight"
    assert km_fused["base_model.model.transformer_blocks.0.img_mlp.gate_up"] == "diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight"
    assert km_fused["lora_transformer_transformer_blocks_0_img_mlp_gate_up"] == "diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight"
    assert km_fused["transformer.transformer_blocks.0.img_mlp.gate_layer"] == ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight", (0, 0, 128))
    assert km_fused["lycoris_transformer_blocks_0_img_mlp_proj"] == ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight", (0, 128, 128))

    # 2. Unfused Diffusers/Unsloth checkpoint accepts fused ComfyUI/ai-toolkit gate_up LoRA via SplitTargets
    unfused_model = _tiny_qwen21(num_layers=1, fused_mlp=False)
    unfused_sd = {f"diffusion_model.{k}": v for k, v in unfused_model.state_dict().items()}
    km_unfused = qwen21_lora_key_map(unfused_sd)
    target = km_unfused["lora_unet_transformer_blocks_0_img_mlp_gate_up"]
    assert isinstance(target, SplitTargets)
    assert target.full_dim == 256

    fused_lora_sd = {
        "lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_down.weight": torch.randn(4, 64),
        "lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_up.weight": torch.randn(256, 4),
        "lora_unet_transformer_blocks_0_img_mlp_gate_up.alpha": torch.tensor(4.0),
        "base_model.model.transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(4, 64),
        "base_model.model.transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(64, 4),
    }
    patches, remaining = load_lora(fused_lora_sd, km_unfused)
    assert not remaining
    assert isinstance(patches["diffusion_model.transformer_blocks.0.img_mlp.gate_layer.weight"], SlicedWeightAdapter)
    assert isinstance(patches["diffusion_model.transformer_blocks.0.img_mlp.proj.weight"], SlicedWeightAdapter)
    assert "diffusion_model.transformer_blocks.0.attn.to_q.weight" in patches


def test_qwen21_per_block_compile_keeps_prefix_prefill_eager(monkeypatch):
    compile_root = REPO / "extensions-builtin" / "sd_forge_compile"
    if str(compile_root) not in sys.path:
        sys.path.insert(0, str(compile_root))
    from anima_block_compile import AnimaBlockCompileManager, _block_list

    model = _tiny_qwen21(num_layers=2, fused_mlp=False)
    assert _block_list(model) is model.transformer_blocks

    compiled_calls: list[int] = []

    def fake_compile(block, **_kwargs):
        class _CompiledWrapper(torch.nn.Module):
            def __init__(self, orig):
                super().__init__()
                self._orig_mod = orig

            def forward(self, *args, **kwargs):
                # Prefix prefill uses eager_block (_orig_mod), so compiled blocks
                # must only see target-only calls (kv_out is None, prefix_kv is set).
                assert kwargs.get("kv_out") is None
                assert kwargs.get("prefix_kv") is not None
                compiled_calls.append(1)
                return self._orig_mod(*args, **kwargs)

        return _CompiledWrapper(block)

    monkeypatch.setattr(torch, "compile", fake_compile)

    x = torch.randn(1, 8, 4, 4)
    ctx = torch.randn(1, 5, 32)
    kmodel = SimpleNamespace(
        diffusion_model=model,
        apply_model=lambda inp, t, c_crossattn: model(inp, t, c_crossattn),
    )
    AnimaBlockCompileManager.install(kmodel, {"backend": "inductor"})
    try:
        with torch.inference_mode():
            out1 = kmodel.apply_model(x, torch.tensor([0.8]), ctx)
            out2 = kmodel.apply_model(x, torch.tensor([0.4]), ctx)
        assert out1.shape == x.shape and out2.shape == x.shape
        assert len(compiled_calls) == 4  # 2 blocks x 2 target steps; prefix prefill stayed eager!
    finally:
        AnimaBlockCompileManager.remove(kmodel)


def test_qwen21_spectrum_pre_head_forecasting(tmp_path):
    spectrum_root = REPO / "extensions-builtin" / "sd_forge_spectrum"
    if str(spectrum_root) not in sys.path:
        sys.path.insert(0, str(spectrum_root))
    from lib_spectrum.forecaster import SpectrumNode

    model = _tiny_qwen21(num_layers=2, fused_mlp=False)
    cap = match_spectrum_capability(model)
    assert cap is get_dit_capability("qwen21")
    assert resolve_spectrum_capture_module(model) is model.final_input

    # Verify compute_final matches exact full forward when fed the captured pre-head feature
    captured = {}

    def capture_hook(_mod, args):
        captured["feat"] = args[0].detach().clone()

    handle = model.final_input.register_forward_pre_hook(capture_hook)
    try:
        x = torch.randn(1, 8, 4, 4)
        sigma = torch.tensor([0.6])
        ctx = torch.randn(1, 5, 32)
        with torch.inference_mode():
            real_out = model(x, sigma, ctx)
            recomputed = model.compute_final(captured["feat"], sigma, tuple(x.shape))
        torch.testing.assert_close(recomputed, real_out, atol=1e-6, rtol=1e-6)
    finally:
        handle.remove()

    # Verify SpectrumNode.patch captures and runs fast_forward_qwen21 on cached steps
    class _FakePatcher:
        def __init__(self, kmodel):
            self.model = kmodel
            self.model_options = {}

        def clone(self):
            cloned = _FakePatcher(self.model)
            cloned.model_options = dict(self.model_options)
            return cloned

        def set_model_unet_function_wrapper(self, fn):
            self.model_options["model_function_wrapper"] = fn

    predictor = SimpleNamespace(
        timestep=lambda s: s,
        calculate_denoised=lambda s, pred, inp: inp - pred * s.view(-1, 1, 1, 1),
    )
    fast_calls = 0
    orig_compute_final = model.compute_final

    def counted_compute_final(*args, **kwargs):
        nonlocal fast_calls
        fast_calls += 1
        return orig_compute_final(*args, **kwargs)

    model.compute_final = counted_compute_final
    patcher = _FakePatcher(SimpleNamespace(diffusion_model=model, predictor=predictor))
    patched = SpectrumNode.patch(
        patcher,
        steps=8,
        weight=0.5,
        degree=1,
        lam=0.1,
        window_size=2,
        flex_window=0.0,
        warmup_steps=3,
        stop_caching_step=1.0,
        tail_actual_steps=1,
        history_size=4,
        schedule="Window",
        refresh_ratio=0.5,
        sea_beta=0.9,
        compat_policy="Conservative",
        verbose=False,
        sea_cache_dir=str(tmp_path),
        sea_cache_context={},
    )
    wrapper = patched.model_options["model_function_wrapper"]
    with torch.inference_mode():
        for step in range(6):
            s = torch.tensor([1.0 - step * 0.12])
            out = wrapper(
                lambda inp, ts, **c: predictor.calculate_denoised(ts, model(inp, ts, c["c_crossattn"]), inp),
                {"input": x, "timestep": s, "c": {"c_crossattn": ctx, "transformer_options": {}}, "cond_or_uncond": [0]},
            )
            assert out.shape == x.shape
    assert fast_calls >= 1

    # Direct fast_forward_qwen21 check
    with torch.inference_mode():
        ff = fast_forward_qwen21(model, predictor, torch.tensor([0.5]), x, captured["feat"])
    assert ff.shape == x.shape


def test_qwen21_workflow_advice(monkeypatch):
    monkeypatch.setattr(dynamic_args, "qwen21", True)
    opts_png = SimpleNamespace(qwen21_do_reference=True, samples_format="png")

    # 1. Turbo checkpoint with too many steps / high CFG
    p_turbo_bad = SimpleNamespace(
        sd_model=SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name="Qwen-Image-2.1-Turbo-FP8.safetensors"), ref_latents=[], ini_latent=None, use_dynamic_shifting=False),
        prompt="a castle",
        steps=32,
        cfg_scale=3.5,
    )
    notes = collect_workflow_advice(p_turbo_bad, opts=opts_png)
    assert any("Turbo" in n and "8 steps" in n for n in notes)

    # 2. Base checkpoint with Turbo settings (and no Turbo LoRA)
    p_base_low = SimpleNamespace(
        sd_model=SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name="Qwen-Image-2.1-FP8.safetensors"), ref_latents=[], ini_latent=None, use_dynamic_shifting=True),
        prompt="a castle",
        steps=8,
        cfg_scale=1.0,
    )
    notes_base = qwen21_workflow_advice(p_base_low, opts=opts_png)
    assert any("Base" in n and "32-40" in n for n in notes_base)

    # Base checkpoint + Turbo8 LoRA at 8 steps / CFG 1 -> clean (no warning!)
    p_base_lora_ok = SimpleNamespace(
        sd_model=p_base_low.sd_model,
        prompt="a castle <lora:Turbo8-LoRA-Qwen-Image-2.1:1.0>",
        steps=8,
        cfg_scale=1.0,
    )
    assert qwen21_workflow_advice(p_base_lora_ok, opts=opts_png) == []

    # 3. References attached while qwen21_do_reference=False + >10 refs + RGBA prompt with JPEG format
    p_rgba_jpg = SimpleNamespace(
        sd_model=SimpleNamespace(
            sd_checkpoint_info=SimpleNamespace(name="Qwen-Image-2.1.safetensors"),
            ref_latents=[torch.zeros(1, 4, 4, 4) for _ in range(11)],
            ini_latent=None,
            use_dynamic_shifting=True,
        ),
        prompt="This is an RGBA image with transparency. A cute dragon sticker.",
        steps=32,
        cfg_scale=2.5,
    )
    notes_warn = qwen21_workflow_advice(p_rgba_jpg, opts=SimpleNamespace(qwen21_do_reference=False, samples_format="jpg"))
    assert any("Enable Reference is off" in n for n in notes_warn)
    assert any("at most 10" in n for n in notes_warn)
    assert any("JPEG" in n and "alpha" in n for n in notes_warn)


# endregion


# region 4. End-to-End RGBA Transparent Output, Reference Preservation & Mask Editing


def test_qwen21_engine_rgba_vision_cache_and_shift_control(monkeypatch):
    from remi.testing import webui_state

    vae_calls = []

    class _FakeFirstStage:
        def process_in(self, x):
            return x * 0.5

    class _FakeVAE:
        output_channels = 4
        first_stage_model = _FakeFirstStage()

        def encode(self, pixel_samples):
            vae_calls.append(pixel_samples.detach().clone())
            b, h, w, c = pixel_samples.shape
            assert c == 4, f"Wan22VAE expects 4-channel RGBA, got {c}"
            return torch.zeros(b, 64, h // 16, w // 16)

    seen_vl: list[list[torch.Tensor]] = []
    shift_updates: list[float] = []

    engine = object.__new__(QwenImage21)
    engine.forge_objects = SimpleNamespace(
        vae=_FakeVAE(),
        clip=SimpleNamespace(patcher=object()),
        unet=SimpleNamespace(
            model=SimpleNamespace(
                diffusion_model=SimpleNamespace(clear_caches=lambda: shift_updates.append(-1.0)),
                predictor=SimpleNamespace(set_parameters=lambda *, shift: shift_updates.append(shift)),
            )
        ),
    )
    engine.text_processing_engine_qwen = lambda prompt, images=None: (seen_vl.append(images or []), [torch.zeros(1, 4, 32)])[1]
    engine.ref_latents = []
    engine.ini_latent = None
    engine.is_qwen21_engine = True
    engine.use_dynamic_shifting = True
    engine._vision_cache = None

    monkeypatch.setattr("backend.diffusion_engine.qwen21.memory_management.load_model_gpu", lambda _p: None)

    with webui_state({"qwen21_do_reference": True, "reference_max_side": 64}):
        # Create a real 4-channel RGBA image: left half opaque red (1,0,0,1), right half fully transparent black (0,0,0,0)
        rgba = torch.zeros(1, 4, 64, 64)
        rgba[:, 0, :, :32] = 1.0  # R
        rgba[:, 3, :, :32] = 1.0  # A
        x_norm = rgba * 2.0 - 1.0  # [-1, 1] as passed by images_tensor_to_samples

        monkeypatch.setattr(dynamic_args, "is_referencing", True)
        engine.encode_first_stage(x_norm)
        monkeypatch.setattr(dynamic_args, "is_referencing", False)
        assert len(engine.ref_latents) == 1
        assert engine.ref_latents[0].shape == (1, 64, 64, 4)

        # Call negative prompt then positive prompt (CFG > 1): VAE encode MUST only run ONCE thanks to _vision_cache!
        neg_prompt = SimpleNamespace(is_negative_prompt=True)
        pos_prompt = SimpleNamespace(is_negative_prompt=False)
        engine.get_learned_conditioning(neg_prompt)
        engine.get_learned_conditioning(pos_prompt)
        assert len(vae_calls) == 1, f"Expected 1 cached VAE encode across uc+c, got {len(vae_calls)}"

        # Verify Qwen3-VL received RGB composited over WHITE (right half transparent -> white 1,1,1),
        # while VAE received TRUE 4-channel RGBA (right half alpha == 0.0)!
        vl_img = seen_vl[0][0]
        assert vl_img.shape[-1] == 3
        torch.testing.assert_close(vl_img[0, 16, 16], torch.tensor([1.0, 0.0, 0.0]), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(vl_img[0, 16, 48], torch.tensor([1.0, 1.0, 1.0]), atol=1e-4, rtol=1e-4)
        assert vae_calls[0][0, 16, 16, 3].item() == pytest.approx(1.0, abs=1e-4)
        assert vae_calls[0][0, 16, 48, 3].item() == pytest.approx(0.0, abs=1e-4)

        # Dynamic shifting enabled vs disabled (Turbo)
        engine.set_shift(0.0, 1024, 1024)
        assert len(shift_updates) == 1
        engine.use_dynamic_shifting = False
        engine.set_shift(0.0, 2048, 2048)
        assert len(shift_updates) == 1  # untouched for Turbo

        # clear_references clears vision cache, dynamic_args, and DiT caches
        engine.clear_references()
        assert engine._vision_cache is None
        assert dynamic_args.ref_latents == []
        assert -1.0 in shift_updates


def test_image_stitch_preserves_rgba_references_for_qwen21(monkeypatch):
    from remi.testing import webui_state

    with webui_state():
        stitch_path = REPO / "extensions-builtin" / "sd_forge_image_stitch" / "scripts" / "image_stitch.py"
        spec = importlib.util.spec_from_file_location("image_stitch_rgba_test", stitch_path)
        stitch_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stitch_mod)

        captured: list[torch.Tensor] = []
        monkeypatch.setattr(stitch_mod.dynamic_args, "can_reference", lambda: True, raising=False)
        monkeypatch.setattr(stitch_mod.dynamic_args, "wan", False, raising=False)
        monkeypatch.setattr(stitch_mod.dynamic_args, "qwen21", True, raising=False)
        monkeypatch.setattr(stitch_mod, "images_tensor_to_samples", lambda img, _a, _m: captured.append(img.cpu()))
        monkeypatch.setattr(stitch_mod.ImageStitch, "reset_references", staticmethod(lambda _p: None))
        monkeypatch.setattr(stitch_mod.ImageStitch, "cached_parameters", None)
        monkeypatch.setattr(stitch_mod.ImageStitch, "cached_model", None)

        ref_a = Image.new("RGBA", (32, 32), (200, 50, 10, 0))
        ref_b = Image.new("RGBA", (32, 32), (200, 50, 10, 255))
        # Alpha difference MUST produce a different hash so ImageStitch re-encodes!
        assert stitch_mod.ImageStitch.hash_image(ref_a) != stitch_mod.ImageStitch.hash_image(ref_b)

        class _FakeSDModel:
            pass

        p = SimpleNamespace(sd_model=_FakeSDModel(), width=64, height=64, batch_size=1)
        script = stitch_mod.ImageStitch()
        script.process(p, True, [ref_a])
        assert len(captured) == 1
        assert captured[0].shape == (1, 4, 32, 32)
        assert captured[0][0, 3].max().item() == 0.0  # true transparent alpha preserved!

        # Same RGB, different alpha -> cache miss, re-encoded with alpha=1.0
        script.process(p, True, [ref_b])
        assert len(captured) == 2
        assert captured[1][0, 3].min().item() == 1.0

        # Non-Qwen2.1 models still flatten RGBA -> 3-channel RGB as before
        monkeypatch.setattr(stitch_mod.dynamic_args, "qwen21", False, raising=False)
        monkeypatch.setattr(stitch_mod.ImageStitch, "cached_parameters", None)
        script.process(p, True, [ref_a])
        assert captured[-1].shape == (1, 3, 32, 32)


def test_rgba_transparent_png_output_and_mask_editing_end_to_end(tmp_path, monkeypatch):
    """End-to-end verification of:
    1. RGBA transparent PNG save vs opaque RGB collapse (``images.save_image_with_geninfo``);
    2. RGB input -> mask background -> Qwen-Image-2.1 RGBA transparent output (Whole & Only masked);
    3. Transparent RGBA sticker input -> mask subject -> Qwen-Image-2.1 selective mask edit
       preserving transparent background outside mask (standard & precise mask, fill mode, color correction);
    4. Plain RGB input + RGB output stays RGB (no regression for 3-channel models).
    """

    from remi.testing import webui_state

    with webui_state(
        {
            "alpha_threshold": 0,
            "enable_pnginfo": False,
            "img2img_background_color": "#808080",
            "save_init_img": False,
            "img2img_color_correction": False,
            "sd_vae_encode_method": "Full",
        }
    ) as state:
        from modules import images, processing, shared

        monkeypatch.setattr(processing.sd_samplers, "create_sampler", lambda *_a, **_kw: SimpleNamespace())

        # 1. PNG save preserves transparency on RGBA, collapses fully opaque RGBA -> RGB
        trans_img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        for y in range(8, 24):
            for x in range(8, 24):
                trans_img.putpixel((x, y), (220, 40, 60, 255))
        trans_path = tmp_path / "transparent.png"
        images.save_image_with_geninfo(trans_img, "", str(trans_path))
        with Image.open(trans_path) as reloaded:
            assert reloaded.mode == "RGBA"
            assert reloaded.getpixel((0, 0))[3] == 0
            assert reloaded.getpixel((16, 16)) == (220, 40, 60, 255)

        opaque_rgba = Image.new("RGBA", (16, 16), (100, 150, 200, 255))
        opaque_path = tmp_path / "opaque.png"
        images.save_image_with_geninfo(opaque_rgba, "", str(opaque_path))
        with Image.open(opaque_path) as reloaded_opaque:
            assert reloaded_opaque.mode == "RGB"

        # 2. Color correction preserves alpha on RGBA images
        cc_target = processing.setup_color_correction(trans_img)
        cc_out = processing.apply_color_correction(cc_target, trans_img)
        assert cc_out.mode == "RGBA"
        assert cc_out.getpixel((0, 0))[3] == 0
        assert cc_out.getpixel((16, 16))[3] == 255

        # 3. End-to-end mask editing through StableDiffusionProcessingImg2Img.init + apply_overlay
        #    Test BOTH standard overlay (img2img_inpaint_precise_mask=False) AND precise mask (True),
        #    AND BOTH Whole picture (inpaint_full_res=False) AND Only masked (inpaint_full_res=True).
        for precise_mask in (False, True):
            for only_masked in (False, True):
                state.opts.data["img2img_inpaint_precise_mask"] = precise_mask
                monkeypatch.setattr(dynamic_args, "qwen21", True)

                # Case 3a: Transparent RGBA sticker input (transparent outer border, opaque red center),
                # mask ONLY the inner 16x16 core to turn it blue (opaque), keep outer transparent border!
                src_rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                for y in range(16, 48):
                    for x in range(16, 48):
                        src_rgba.putpixel((x, y), (220, 30, 30, 255))
                mask = Image.new("L", (64, 64), 0)
                for y in range(24, 40):
                    for x in range(24, 40):
                        mask.putpixel((x, y), 255)

                encoded_inputs: list[torch.Tensor] = []

                def fake_encode(img_tensor, _approx, _model, _sink=encoded_inputs):
                    _sink.append(img_tensor.detach().cpu())
                    return torch.zeros(img_tensor.shape[0], 64, img_tensor.shape[2] // 16, img_tensor.shape[3] // 16)

                monkeypatch.setattr(processing, "images_tensor_to_samples", fake_encode)

                fake_model = SimpleNamespace(is_sdxl=False, is_inpaint=False, comments=[])
                monkeypatch.setattr(shared, "sd_model", fake_model)
                p = processing.StableDiffusionProcessingImg2Img(
                    sd_model=fake_model,
                    init_images=[src_rgba],
                    mask=mask,
                    width=64,
                    height=64,
                    resize_mode=0,
                    mask_blur=0,
                    inpainting_fill=0,  # also exercises masking.fill alpha preservation!
                    inpaint_full_res=only_masked,
                    inpaint_full_res_padding=8,
                    batch_size=1,
                )
                p.init(["turn center blue"], [42], [42])

                # Input reaching Qwen-Image-2.1 VAE MUST retain all 4 RGBA channels!
                assert encoded_inputs[-1].shape[1] == 4
                if not only_masked:
                    assert encoded_inputs[-1][0, 3, 0, 0].item() == pytest.approx(0.0, abs=1e-4)
                    assert encoded_inputs[-1][0, 3, 32, 32].item() == pytest.approx(1.0, abs=1e-4)

                # Simulate Qwen-Image-2.1 RGBA decoder outputting a blue patch (opaque)
                denoised_rgba = Image.new("RGBA", (64, 64), (20, 60, 240, 255))
                composited, _ = processing.apply_overlay(denoised_rgba, p.paste_to, p.overlay_images[0])
                assert composited.mode == "RGBA"
                # Unmasked transparent corner MUST stay transparent (alpha == 0)!
                assert composited.getpixel((2, 2))[3] == 0
                # Unmasked opaque red ring MUST stay red & opaque!
                assert composited.getpixel((18, 18))[0] > 200 and composited.getpixel((18, 18))[3] == 255
                # Masked center MUST take the newly denoised blue pixel!
                assert composited.getpixel((32, 32))[2] > 200 and composited.getpixel((32, 32))[3] == 255

                # Case 3b: Opaque RGB input -> mask background -> Qwen-Image-2.1 removes background (alpha=0)
                src_rgb = Image.new("RGB", (64, 64), (200, 180, 40))
                bg_mask = Image.new("L", (64, 64), 255)
                for y in range(20, 44):
                    for x in range(20, 44):
                        bg_mask.putpixel((x, y), 0)  # keep center subject unmasked
                p_bg = processing.StableDiffusionProcessingImg2Img(
                    sd_model=fake_model,
                    init_images=[src_rgb],
                    mask=bg_mask,
                    width=64,
                    height=64,
                    resize_mode=0,
                    mask_blur=0,
                    inpainting_fill=1,
                    inpaint_full_res=only_masked,
                    inpaint_full_res_padding=0,
                    batch_size=1,
                )
                p_bg.init(["transparent background, RGBA"], [42], [42])
                denoised_cutout = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                cutout, _ = processing.apply_overlay(denoised_cutout, p_bg.paste_to, p_bg.overlay_images[0])
                assert cutout.mode == "RGBA"
                # Masked background becomes transparent (alpha == 0)!
                assert cutout.getpixel((2, 2))[3] == 0
                # Unmasked center subject stays opaque original color!
                assert cutout.getpixel((32, 32)) == (200, 180, 40, 255)

        # 4. Regression check: non-RGBA RGB input + RGB denoised output stays RGB
        state.opts.data["img2img_inpaint_precise_mask"] = False
        monkeypatch.setattr(dynamic_args, "qwen21", False)
        rgb_img = Image.new("RGB", (64, 64), (10, 20, 30))
        rgb_mask = Image.new("L", (64, 64), 0)
        for y in range(16, 48):
            for x in range(16, 48):
                rgb_mask.putpixel((x, y), 255)
        fake_rgb_model = SimpleNamespace(is_sdxl=False, is_inpaint=False, comments=[])
        monkeypatch.setattr(shared, "sd_model", fake_rgb_model)
        p_rgb = processing.StableDiffusionProcessingImg2Img(
            sd_model=fake_rgb_model,
            init_images=[rgb_img],
            mask=rgb_mask,
            width=64,
            height=64,
            resize_mode=0,
            mask_blur=0,
            inpainting_fill=1,
            inpaint_full_res=True,
            inpaint_full_res_padding=8,
            batch_size=1,
        )
        p_rgb.init(["rgb edit"], [1], [1])
        out_rgb, _ = processing.apply_overlay(Image.new("RGB", (64, 64), (200, 100, 50)), p_rgb.paste_to, p_rgb.overlay_images[0])
        assert out_rgb.mode == "RGB"
        assert out_rgb.getpixel((2, 2)) == (10, 20, 30)
        assert out_rgb.getpixel((32, 32)) == (200, 100, 50)


# endregion
