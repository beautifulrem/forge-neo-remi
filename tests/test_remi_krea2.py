"""Tests for Krea 2 Diffusers/Unsloth formats, Qwen3-VL text encoder loading,
step caches, Edit-mode fixes, LoRA trainer formats, on-the-fly quantization,
SLG and workflow advice."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import safetensors.torch
import torch

from backend.args import dynamic_args


def _tiny_krea2_dit(
    *,
    features: int = 256,
    tdim: int = 64,
    txtdim: int = 128,
    heads: int = 2,
    kvheads: int = 2,
    multiplier: int = 2,
    layers: int = 2,
    patch: int = 2,
    channels: int = 16,
    txtlayers: int = 12,
    txtheads: int = 2,
    txtkvheads: int = 2,
    seed: int = 0,
):
    from backend.nn.krea import SingleStreamDiT

    gen = torch.Generator().manual_seed(seed)
    dit = SingleStreamDiT(
        features=features,
        tdim=tdim,
        txtdim=txtdim,
        heads=heads,
        kvheads=kvheads,
        multiplier=multiplier,
        layers=layers,
        patch=patch,
        channels=channels,
        txtlayers=txtlayers,
        txtheads=txtheads,
        txtkvheads=txtkvheads,
    ).eval()
    sd = {k: torch.randn(v.shape, generator=gen, dtype=torch.float32) * 0.05 for k, v in dit.state_dict().items()}
    dit.load_state_dict(sd)
    return dit


def _native_to_diffusers_krea2(native_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Invert ``convert_diffusers_krea2_transformer`` to synthesize an exact
    Diffusers ``Krea2Transformer2DModel`` state dict (matching ``krea/Krea-2-*``
    and ``unsloth/Krea-2-Turbo*``)."""

    inv_block = (
        ("attn.wq.", "attn.to_q."),
        ("attn.wk.", "attn.to_k."),
        ("attn.wv.", "attn.to_v."),
        ("attn.gate.", "attn.to_gate."),
        ("attn.wo.", "attn.to_out.0."),
        ("attn.qknorm.qnorm.scale", "attn.norm_q.weight"),
        ("attn.qknorm.knorm.scale", "attn.norm_k.weight"),
        ("mlp.gate.", "ff.gate."),
        ("mlp.up.", "ff.up."),
        ("mlp.down.", "ff.down."),
        ("prenorm.scale", "norm1.weight"),
        ("postnorm.scale", "norm2.weight"),
    )
    inv_top = (
        ("first.", "img_in."),
        ("tmlp.0.", "time_embed.linear_1."),
        ("tmlp.2.", "time_embed.linear_2."),
        ("tproj.1.", "time_mod_proj."),
        ("txtmlp.0.scale", "txt_in.norm.weight"),
        ("txtmlp.1.", "txt_in.linear_1."),
        ("txtmlp.3.", "txt_in.linear_2."),
        ("txtfusion.projector.", "text_fusion.projector."),
        ("last.norm.scale", "final_layer.norm.weight"),
        ("last.modulation.lin", "final_layer.scale_shift_table"),
        ("last.linear.", "final_layer.linear."),
    )

    def _inv_rest(rest: str) -> str:
        for src, dst in inv_block:
            if src.endswith("."):
                if rest.startswith(src):
                    return dst + rest[len(src) :]
            elif rest == src:
                return dst
        return rest

    out: dict[str, torch.Tensor] = {}
    for k, v in native_sd.items():
        if k.startswith("blocks."):
            _, idx, rest = k.split(".", 2)
            if rest == "mod.lin":
                out[f"transformer_blocks.{idx}.scale_shift_table"] = v.reshape(6, -1)
            else:
                out[f"transformer_blocks.{idx}.{_inv_rest(rest)}"] = v
        elif k.startswith(("txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks.")):
            _, sub, idx, rest = k.split(".", 3)
            out[f"text_fusion.{sub}.{idx}.{_inv_rest(rest)}"] = v
        else:
            mapped = None
            for src, dst in inv_top:
                if src.endswith("."):
                    if k.startswith(src):
                        mapped = dst + k[len(src) :]
                        break
                elif k == src:
                    mapped = dst
                    break
            out[mapped or k] = v
    return out


def test_krea2_diffusers_dir_sharded_transformer_and_qwen3vl_variants(tmp_path, monkeypatch):
    from backend import loader, memory_management
    from backend.nn.krea import SingleStreamDiT
    from backend.nn.krea_diffusers import (
        convert_diffusers_krea2_transformer,
        is_diffusers_krea2_transformer,
        is_krea2_diffusers_dir,
        is_sharded_krea2_transformer_dir,
        load_krea2_diffusers_dir,
    )
    from backend.nn.moss_diffusers import diffusers_checkpoint_info, is_diffusers_checkpoint_dir

    # Use realistic head_dim=128 (features=256, heads=2, kvheads=2)
    ref_dit = _tiny_krea2_dit(features=256, heads=2, kvheads=2, layers=2)
    native_sd = {k: v.detach().clone() for k, v in ref_dit.state_dict().items()}
    diff_sd = _native_to_diffusers_krea2(native_sd)

    assert is_diffusers_krea2_transformer(diff_sd)
    converted = convert_diffusers_krea2_transformer(diff_sd)
    assert set(converted.keys()) == set(native_sd.keys())
    for k in native_sd:
        assert torch.equal(converted[k], native_sd[k]), f"mismatch on {k}"

    # Bit-exact forward parity between converted Diffusers weights and native SingleStreamDiT
    reloaded = SingleStreamDiT(features=256, tdim=64, txtdim=128, heads=2, kvheads=2, multiplier=2, layers=2, patch=2, channels=16, txtlayers=12, txtheads=2, txtkvheads=2).eval()
    reloaded.load_state_dict(converted)
    x = torch.randn(1, 16, 1, 4, 4)
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 5, 12, 128)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(x, t, ctx), ref_dit(x, t, ctx), rtol=0, atol=0)

    # Full krea/Krea-2-Turbo Diffusers directory (sharded transformer + Diffusers Qwen3-VL + Diffusers Qwen-Image VAE)
    pipe_dir = tmp_path / "Krea-2-Turbo"
    tr_dir = pipe_dir / "transformer"
    te_dir = pipe_dir / "text_encoder"
    vae_dir = pipe_dir / "vae"
    for d in (tr_dir, te_dir, vae_dir):
        d.mkdir(parents=True)

    (pipe_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "Krea2Pipeline", "transformer": ["diffusers", "Krea2Transformer2DModel"], "is_distilled": True}),
        encoding="utf-8",
    )
    items = list(diff_sd.items())
    mid = len(items) // 2
    s1, s2 = dict(items[:mid]), dict(items[mid:])
    safetensors.torch.save_file(s1, str(tr_dir / "diffusion_pytorch_model-00001-of-00002.safetensors"))
    safetensors.torch.save_file(s2, str(tr_dir / "diffusion_pytorch_model-00002-of-00002.safetensors"))
    wmap = dict.fromkeys(s1, "diffusion_pytorch_model-00001-of-00002.safetensors")
    wmap.update(dict.fromkeys(s2, "diffusion_pytorch_model-00002-of-00002.safetensors"))
    (tr_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}), encoding="utf-8")

    # Official krea/Krea-2-*/text_encoder/model.safetensors uses language_model.* + visual.* (no model. prefix)
    te_sd = {
        "language_model.embed_tokens.weight": torch.randn(16, 2560, dtype=torch.bfloat16),
        "language_model.layers.0.input_layernorm.weight": torch.ones(2560, dtype=torch.bfloat16),
        "visual.deepstack_merger_list.0.norm.weight": torch.ones(4096, dtype=torch.bfloat16),
        "visual.merger.linear_fc2.weight": torch.randn(2560, 4096, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(te_sd, str(te_dir / "model.safetensors"))

    vae_sd = {
        "decoder.mid_block.resnets.0.norm1.gamma": torch.ones(8, 1, 1, 1),
        "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3, 3),
        "decoder.norm_out.gamma": torch.ones(8, 1, 1, 1),
        "post_quant_conv.weight": torch.randn(16, 16, 1, 1, 1),
    }
    safetensors.torch.save_file(vae_sd, str(vae_dir / "diffusion_pytorch_model.safetensors"))

    assert is_krea2_diffusers_dir(pipe_dir)
    assert is_diffusers_checkpoint_dir(pipe_dir)
    arch, primary = diffusers_checkpoint_info(pipe_dir)
    assert arch == "krea-2/diffusers" and primary is not None and primary.endswith("00001-of-00002.safetensors")

    assert is_sharded_krea2_transformer_dir(tr_dir)
    assert is_diffusers_checkpoint_dir(tr_dir)
    assert diffusers_checkpoint_info(tr_dir)[0] == "krea-2/diffusers"

    loaded_sd, loaded_meta = load_krea2_diffusers_dir(pipe_dir)
    assert loaded_meta.get("krea2_is_distilled") == "true"
    assert "model.diffusion_model.txtfusion.projector.weight" in loaded_sd
    assert "model.diffusion_model.blocks.0.mod.lin" in loaded_sd
    assert loaded_sd["model.diffusion_model.blocks.0.mod.lin"].ndim == 1
    assert "text_encoders.qwen3vl_4b.transformer.model.language_model.embed_tokens.weight" in loaded_sd
    assert "text_encoders.qwen3vl_4b.transformer.model.visual.deepstack_merger_list.0.norm.weight" in loaded_sd
    assert "vae.conv2.weight" in loaded_sd and "vae.decoder.head.0.gamma" in loaded_sd

    # Additional TE loader with official/Unsloth Qwen3-VL (language_model.* + visual.*) routes to qwen3vl_4b, NOT qwen35_9b
    override_te = tmp_path / "Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors"
    override_sd = {
        "language_model.embed_tokens.weight": torch.full((16, 2560), 2.0, dtype=torch.bfloat16),
        "language_model.layers.0.input_layernorm.weight": torch.ones(2560, dtype=torch.bfloat16),
        "visual.deepstack_merger_list.0.norm.weight": torch.ones(4096, dtype=torch.bfloat16),
        "visual.merger.linear_fc2.weight": torch.randn(2560, 4096, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(override_sd, str(override_te))
    fake_guess = SimpleNamespace(vae_key_prefix=["vae."], text_encoder_key_prefix=["text_encoders."])
    sd_with_override = loader.replace_state_dict(dict(loaded_sd), override_sd, {}, fake_guess, str(override_te))
    assert not any("qwen35_9b" in k for k in sd_with_override)
    assert torch.all(sd_with_override["text_encoders.qwen3vl_4b.transformer.model.language_model.embed_tokens.weight"] == 2.0)

    # Plain FP8 Qwen3-VL text encoder (e.g. unsloth Krea-2-Turbo-text_encoder-FP8.safetensors) keeps embeddings/norms/Conv3d/bias in BF16
    cpu = torch.device("cpu")
    monkeypatch.setattr(memory_management, "text_encoder_dtype", lambda *_a, **_kw: torch.bfloat16)
    monkeypatch.setattr(memory_management, "cpu", cpu)

    class _TinyQwen3VL(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(16, 64)
            self.model.norm = torch.nn.RMSNorm(64)
            self.model.proj = torch.nn.Linear(64, 64, bias=True)
            self.visual = torch.nn.Module()
            self.visual.patch_embed = torch.nn.Conv3d(3, 8, kernel_size=2)
            self.visual.fc = torch.nn.Linear(8, 8, bias=True)

    import backend.nn.llm.llama as llama_mod

    monkeypatch.setattr(llama_mod, "Qwen3VL", lambda cfg, model_type="qwen3vl_4b": _TinyQwen3VL(cfg))
    fp8_te_sd = {
        "language_model.embed_tokens.weight": torch.randn(16, 64, dtype=torch.bfloat16),
        "language_model.norm.weight": torch.ones(64, dtype=torch.bfloat16),
        "language_model.proj.weight": torch.randn(64, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        "language_model.proj.bias": torch.randn(64, dtype=torch.bfloat16),
        "visual.patch_embed.weight": torch.randn(8, 3, 2, 2, 2, dtype=torch.bfloat16),
        "visual.patch_embed.bias": torch.randn(8, dtype=torch.bfloat16),
        "visual.fc.weight": torch.randn(8, 8, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        "visual.fc.bias": torch.randn(8, dtype=torch.bfloat16),
        # Pad >16 tensors to satisfy loader assertion
        **{f"lm_head.pad_{i}": torch.zeros(1, dtype=torch.float8_e4m3fn) for i in range(12)},
    }
    repo_path = os.path.join(loader.HF, "krea", "Krea-2-Raw")
    loaded_te = loader.load_huggingface_component(SimpleNamespace(unet_config={}), "text_encoder", "transformers", "Qwen3VLModel", repo_path, fp8_te_sd)
    assert loaded_te.model.embed_tokens.weight.dtype in (torch.bfloat16, torch.float32)
    assert loaded_te.model.norm.weight.dtype in (torch.bfloat16, torch.float32)
    assert loaded_te.model.proj.weight.dtype == torch.float8_e4m3fn
    assert loaded_te.model.proj.bias.dtype == torch.bfloat16
    assert loaded_te.visual.patch_embed.weight.dtype == torch.bfloat16
    assert loaded_te.visual.fc.weight.dtype == torch.float8_e4m3fn
    assert loaded_te.visual.fc.bias.dtype == torch.bfloat16


def test_krea2_unsloth_torchao_diffusers_fp8_and_int8_load_and_run(monkeypatch):
    import huggingface_guess

    from backend import loader, memory_management
    from backend.quant_ops import QuantizedTensor, RowWiseFP8E4M3Layout, TensorWiseINT8Layout
    from backend.state_dict import convert_quantization, detect_quantization
    from modules_forge.packages.comfy.utils import convert_diffusers_mmdit

    cpu = torch.device("cpu")
    monkeypatch.setattr(memory_management, "get_torch_device", lambda: cpu)
    monkeypatch.setattr(memory_management, "unet_offload_device", lambda: cpu)
    monkeypatch.setattr(memory_management, "unet_initial_load_device", lambda *_a, **_kw: cpu)
    monkeypatch.setattr(memory_management, "should_use_bf16", lambda *_a, **_kw: True)

    ref_dit = _tiny_krea2_dit(features=256, tdim=64, txtdim=128, heads=2, kvheads=2, multiplier=2, layers=1, patch=2, channels=16, txtlayers=12, txtheads=2, txtkvheads=2)
    ref_native = {k: v.detach().clone().to(torch.bfloat16) for k, v in ref_dit.state_dict().items()}
    ref_diff = _native_to_diffusers_krea2(ref_native)

    x = torch.randn(1, 16, 1, 4, 4, dtype=torch.bfloat16)
    t = torch.tensor([0.5], dtype=torch.bfloat16)
    ctx = torch.randn(1, 4, 12, 128, dtype=torch.bfloat16)
    with torch.inference_mode():
        ref_out = ref_dit(x.float(), t.float(), ctx.float())

    quant_keys = [k for k in ref_diff if k.startswith(("transformer_blocks.0.attn.to_", "transformer_blocks.0.ff.")) and k.endswith(".weight")]
    assert len(quant_keys) == 8

    # 1. Unsloth Krea-2-Turbo-FP8.safetensors (Diffusers keys + TorchAO Float8Tensor per-row [N, 1])
    fp8_sd = dict(ref_diff)
    for k in quant_keys:
        w = fp8_sd.pop(k)
        stem = k[: -len(".weight")]
        qdata, params = RowWiseFP8E4M3Layout.quantize(w)
        fp8_sd[f"{stem}._weight_qdata"] = qdata
        fp8_sd[f"{stem}._weight_scale"] = params.scale
    fp8_meta = {"unsloth_format": "unsloth_prequant_transformer_state_dict_v1", "unsloth_metadata": json.dumps({"family": "krea-2", "scheme": "fp8"})}

    fp8_sd, _ = convert_quantization(fp8_sd, fp8_meta)
    fp8_native = convert_diffusers_mmdit(fp8_sd, "")
    assert fp8_native is not None
    guess = huggingface_guess.guess({f"model.diffusion_model.{k}": v for k, v in fp8_native.items()})
    assert "Krea-2" in guess.huggingface_repo
    assert detect_quantization(fp8_native, is_unet=True)["layout"] == "RowWiseFP8E4M3Layout"

    guess.unet_config.update({"tdim": 64, "multiplier": 2, "txtheads": 2, "txtkvheads": 2})
    fp8_model = loader.load_huggingface_component(guess, "transformer", "diffusers", "Krea2Transformer2DModel", "/unused", fp8_native).eval()
    wq = fp8_model.blocks[0].attn.wq
    assert isinstance(wq.weight, QuantizedTensor) and wq.weight._qdata.dtype == torch.float8_e4m3fn
    assert wq.weight._params.scale.shape == (256, 1)
    assert not isinstance(fp8_model.first.weight, QuantizedTensor)
    with torch.inference_mode():
        fp8_out = fp8_model(x, t, ctx).float()
    assert torch.nn.functional.cosine_similarity(fp8_out.flatten(), ref_out.flatten(), dim=0) > 0.99

    # LoRA merge + requantize round-trip preserves [256, 1] per-row scale
    deq_w = wq.convert_weight(wq.weight)
    wq.set_weight(deq_w + 0.01)
    assert isinstance(wq.weight, QuantizedTensor) and wq.weight._params.scale.shape == (256, 1)

    # 2. Unsloth Krea-2-Turbo-INT8.safetensors (Diffusers keys + TorchAO Int8Tensor per-row [N, 1])
    int8_sd = dict(ref_diff)
    for k in quant_keys:
        w = int8_sd.pop(k)
        stem = k[: -len(".weight")]
        qdata, params = TensorWiseINT8Layout.quantize(w.float(), is_weight=True, per_channel=True, convrot=False)
        int8_sd[f"{stem}._weight_qdata"] = qdata
        int8_sd[f"{stem}._weight_scale"] = params.scale.float()
    int8_meta = {"unsloth_format": "unsloth_prequant_transformer_state_dict_v1", "unsloth_metadata": json.dumps({"family": "krea-2", "scheme": "int8"})}

    int8_sd, _ = convert_quantization(int8_sd, int8_meta)
    int8_native = convert_diffusers_mmdit(int8_sd, "")
    assert int8_native is not None
    assert detect_quantization(int8_native, is_unet=True)["layout"] == "TensorWiseINT8Layout"
    int8_model = loader.load_huggingface_component(guess, "transformer", "diffusers", "Krea2Transformer2DModel", "/unused", int8_native).eval()
    assert isinstance(int8_model.blocks[0].attn.wq.weight, QuantizedTensor)
    with torch.inference_mode():
        int8_out = int8_model(x, t, ctx).float()
    assert torch.nn.functional.cosine_similarity(int8_out.flatten(), ref_out.flatten(), dim=0) > 0.99


def test_krea2_step_caches_edit_batch_broadcast_and_slg_hook(monkeypatch):
    from backend.sampling.sampling_function import sampling_cleanup

    dit = _tiny_krea2_dit(features=256, tdim=64, txtdim=128, heads=2, kvheads=2, multiplier=2, layers=3)
    fusion_calls = 0
    pe_calls = 0
    orig_fusion = dit.txtfusion.forward
    orig_pe = dit.pe_embedder.forward

    def counted_fusion(*a, **kw):
        nonlocal fusion_calls
        fusion_calls += 1
        return orig_fusion(*a, **kw)

    def counted_pe(*a, **kw):
        nonlocal pe_calls
        pe_calls += 1
        return orig_pe(*a, **kw)

    monkeypatch.setattr(dit.txtfusion, "forward", counted_fusion)
    monkeypatch.setattr(dit.pe_embedder, "forward", counted_pe)
    monkeypatch.setattr(dynamic_args, "ref_latents", [])

    x = torch.randn(1, 16, 1, 4, 4)
    ctx_cond = torch.randn(1, 5, 12, 128)
    ctx_cond_orig = ctx_cond.clone()
    ctx_uncond = torch.randn(1, 3, 12, 128)

    with torch.inference_mode():
        # Separate cond + uncond calls stay cached across steps without evicting each other
        # and NEVER mutate the caller's cached conditioning tensor in-place.
        out_c1 = dit(x, torch.tensor([0.9]), ctx_cond)
        assert torch.equal(ctx_cond, ctx_cond_orig)
        out_u1 = dit(x, torch.tensor([0.9]), ctx_uncond)
        assert fusion_calls == 2 and pe_calls == 2

        out_c2 = dit(x, torch.tensor([0.9]), ctx_cond)
        out_u2 = dit(x, torch.tensor([0.9]), ctx_uncond)
        assert fusion_calls == 2 and pe_calls == 2
        torch.testing.assert_close(out_c1, out_c2, rtol=0, atol=0)
        torch.testing.assert_close(out_u1, out_u2, rtol=0, atol=0)

        # sampling_cleanup releases caches and forces recomputation on next run
        fake_unet = SimpleNamespace(
            list_controlnets=list,
            model=SimpleNamespace(diffusion_model=dit),
        )
        sampling_cleanup(fake_unet)
        out_c3 = dit(x, torch.tensor([0.9]), ctx_cond)
        assert fusion_calls == 3 and pe_calls == 3
        torch.testing.assert_close(out_c1, out_c3, rtol=0, atol=0)

        # Edit mode: bs=3 and bs=4 with single-batch ref + odd latent dims (5x5 -> padded 6x6)
        ref = torch.randn(1, 16, 5, 5)
        monkeypatch.setattr(dynamic_args, "ref_latents", [ref])
        for bs in (3, 4):
            xb = torch.randn(bs, 16, 1, 5, 5)
            tb = torch.full((bs,), 0.5)
            cb = torch.randn(bs, 4, 12, 128)
            yb = dit(xb, tb, cb)
            assert yb.shape == xb.shape and torch.isfinite(yb).all()

        # SLG patches_replace["dit"][("single_block", 1)] skips block 1 cleanly
        monkeypatch.setattr(dynamic_args, "ref_latents", [])
        visited: list[int] = []
        for idx, blk in enumerate(dit.blocks):
            orig_fwd = blk.forward
            monkeypatch.setattr(blk, "forward", (lambda *a, _i=idx, _f=orig_fwd, **kw: (visited.append(_i), _f(*a, **kw))[1]))
        slg_opts = {"patches_replace": {"dit": {("single_block", 1): lambda args, _extra: args}}}
        out_slg = dit(x, torch.tensor([0.5]), ctx_cond, transformer_options=slg_opts)
        assert out_slg.shape == x.shape
        assert visited == [0, 2]


def test_krea2_lora_all_trainer_formats_and_diff_b():
    from modules_forge.packages.comfy.lora import load_lora, model_lora_keys_unet

    dit = _tiny_krea2_dit(features=256, tdim=64, txtdim=128, heads=2, kvheads=2, multiplier=2, layers=2)
    dit.config = {"layers": 2}
    fake_model = SimpleNamespace(
        diffusion_model=dit,
        config=SimpleNamespace(huggingface_repo="krea/Krea-2-Raw"),
        state_dict=lambda: {f"diffusion_model.{k}": v for k, v in dit.state_dict().items()},
    )
    key_map = model_lora_keys_unet(fake_model, {})

    def _pair(prefix: str, out_d: int, in_d: int, style: str = "peft") -> dict[str, torch.Tensor]:
        if style == "kohya":
            return {f"{prefix}.lora_down.weight": torch.randn(4, in_d), f"{prefix}.lora_up.weight": torch.randn(out_d, 4)}
        return {f"{prefix}.lora_A.weight": torch.randn(4, in_d), f"{prefix}.lora_B.weight": torch.randn(out_d, 4)}

    # 1. Comfy-Org Diffusers style LoRA (krea2_darkbrush.safetensors)
    darkbrush = {
        **_pair("transformer.transformer_blocks.0.attn.to_q", 256, 256),
        **_pair("transformer.transformer_blocks.1.attn.to_out.0", 256, 256),
        **_pair("transformer.text_fusion.layerwise_blocks.0.attn.to_gate", 128, 128),
        **_pair("transformer.img_in", 256, 64),
        **_pair("transformer.final_layer.linear", 64, 256),
    }
    patches, rem = load_lora(darkbrush, key_map)
    assert rem == {} and "diffusion_model.blocks.0.attn.wq.weight" in patches and "diffusion_model.last.linear.weight" in patches

    # 2. Comfy-Org Turbo rank-64 LoRA with .diff_b bias diffs + ai-toolkit native
    turbo_lora = {
        **_pair("diffusion_model.blocks.0.attn.wq", 256, 256, style="kohya"),
        **_pair("diffusion_model.first", 256, 64, style="kohya"),
        "diffusion_model.first.diff_b": torch.randn(256),
        "diffusion_model.tmlp.0.diff_b": torch.randn(256),
        "diffusion_model.last.linear.diff_b": torch.randn(64),
    }
    patches, rem = load_lora(turbo_lora, key_map)
    assert rem == {}
    assert "diffusion_model.first.bias" in patches and "diffusion_model.tmlp.0.bias" in patches and "diffusion_model.last.linear.bias" in patches

    # 3. fal.ai / PEFT / musubi-tuner / OneTrainer / LyCORIS / bare formats
    community_lora = {
        **_pair("base_model.model.blocks.0.attn.wk", 256, 256),
        **_pair("base_model.model.diffusion_model.blocks.0.attn.wv", 256, 256),
        **_pair("base_model.model.transformer_blocks.0.ff.up", 512, 256),
        **_pair("base_model.model.transformer.transformer_blocks.1.ff.down", 256, 512),
        **_pair("lora_transformer_blocks_1_attn_wo", 256, 256, style="kohya"),
        **_pair("lycoris_blocks_1_mlp_gate", 512, 256, style="kohya"),
        **_pair("lora_unet_transformer_blocks_1_attn_to_gate", 256, 256, style="kohya"),
        **_pair("blocks.0.mlp.down", 256, 512),
    }
    patches, rem = load_lora(community_lora, key_map)
    assert rem == {}, f"unmatched keys: {list(rem)}"
    assert len(patches) == 8


def test_krea2_on_the_fly_quant_and_grounded_negative_and_grayscale_ref(monkeypatch):
    from backend.diffusion_engine.krea import Krea2 as Krea2Engine
    from backend.nn.anima_quant import (
        keep_sensitive_weights_precise,
        quantize_int8_convrot_state_dict,
        scale_fp8_state_dict,
        should_keep_sensitive_weights_precise,
        should_quantize_on_load,
    )
    from backend.operations import using_forge_operations
    from modules import shared

    # 1. On-the-fly FP8 / scaled_fp8 / int8_convrot for Krea2Transformer2DModel
    assert should_keep_sensitive_weights_precise("Krea2Transformer2DModel", torch.float8_e4m3fn, torch.bfloat16) is True
    assert should_quantize_on_load("scaled_fp8", "Krea2Transformer2DModel", None, False, torch.bfloat16) is True
    assert should_quantize_on_load("int8_convrot", "Krea2Transformer2DModel", None, False, torch.bfloat16) is True

    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float8_e4m3fn, manual_cast_enabled=True):
        dit = _tiny_krea2_dit(features=256, tdim=64, txtdim=256, heads=2, kvheads=2, multiplier=2, layers=1)
    kept = keep_sensitive_weights_precise(dit, torch.bfloat16)
    assert kept > 0
    assert dit.first.weight.dtype == torch.bfloat16
    assert dit.tmlp[0].weight.dtype == torch.bfloat16
    assert dit.tproj[1].weight.dtype == torch.bfloat16
    assert dit.txtfusion.projector.weight.dtype == torch.bfloat16
    assert dit.txtmlp[1].weight.dtype == torch.bfloat16
    assert dit.last.linear.weight.dtype == torch.bfloat16
    assert dit.blocks[0].attn.wq.weight.dtype == torch.float8_e4m3fn

    # Scaled FP8 uses per-row float8_e4m3fn_rowwise on blocks + txtfusion while keeping sensitive layers BF16
    # multiplier=3 gives mlpdim=512 (divisible by ConvRot group_size=256, matching real Krea 2's 16384)
    bf16_dit = _tiny_krea2_dit(features=256, tdim=64, txtdim=256, heads=2, kvheads=2, multiplier=3, layers=1)
    fp8_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in bf16_dit.state_dict().items()}
    n_fp8 = scale_fp8_state_dict(fp8_sd, fp8_matmul=False, device=torch.device("cpu"))
    assert n_fp8 == 8 + 4 * 8  # 1 block * 8 + 4 txtfusion blocks * 8
    assert json.loads(bytes(fp8_sd["blocks.0.attn.wq.comfy_quant"].tolist()))["format"] == "float8_e4m3fn_rowwise"
    assert fp8_sd["blocks.0.attn.wq.weight_scale"].shape == (256, 1)
    assert "first.comfy_quant" not in fp8_sd and "txtfusion.projector.comfy_quant" not in fp8_sd

    # INT8-ConvRot quantizes ONLY the main blocks.* Linears (matching Comfy-Org/Krea-2)
    cr_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in bf16_dit.state_dict().items()}
    n_cr = quantize_int8_convrot_state_dict(cr_sd, device=torch.device("cpu"))
    assert n_cr == 8  # 1 block * 8 (all in_features divisible by 256)
    assert json.loads(bytes(cr_sd["blocks.0.attn.wq.comfy_quant"].tolist()))["convrot"] is True
    assert "txtfusion.layerwise_blocks.0.attn.wq.comfy_quant" not in cr_sd

    # 2. Grounded negative conditioning + vision cache dedup + grayscale expansion + clear_references
    import backend.diffusion_engine.krea as krea_engine_mod

    fake_opts = SimpleNamespace(reference_max_side=1024, krea2_do_reference=True)
    monkeypatch.setattr(shared, "opts", fake_opts)
    monkeypatch.setattr(krea_engine_mod, "opts", fake_opts)
    vae_calls = 0
    seen_te_calls: list[tuple[list[str], int]] = []

    class _FakeVAE:
        def __init__(self):
            self.first_stage_model = SimpleNamespace(process_in=lambda t: t)

        def encode(self, img):
            nonlocal vae_calls
            vae_calls += 1
            assert img.shape[-1] == 3
            return torch.zeros(1, 16, 1, 2, 2)

    engine = Krea2Engine.__new__(Krea2Engine)
    engine.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=None), vae=_FakeVAE())
    engine.ref_latents = []
    engine.ini_latent = torch.rand(1, 32, 32, 1)  # 1-channel grayscale reference!
    engine._vision_cache = None
    engine.text_processing_engine_qwen = lambda prompt, images=None: (seen_te_calls.append((list(prompt), len(images or []))), [torch.zeros(4, 12, 128)])[1]

    from backend import memory_management

    monkeypatch.setattr(memory_management, "load_model_gpu", lambda *_a, **_kw: None)

    class _Cond(list):
        def __init__(self, items, *, is_negative_prompt: bool):
            super().__init__(items)
            self.is_negative_prompt = is_negative_prompt

    # Uncond runs first in Forge; must still receive grounded reference image and keep ini_latent for cond
    engine.get_learned_conditioning(_Cond([""], is_negative_prompt=True))
    assert engine.ini_latent is not None
    engine.get_learned_conditioning(_Cond(["make it snowy"], is_negative_prompt=False))
    assert engine.ini_latent is None
    assert vae_calls == 1  # deduplicated across uc + c!
    assert seen_te_calls == [([""], 1), (["make it snowy"], 1)]
    assert len(dynamic_args.ref_latents) == 1

    engine.clear_references()
    assert dynamic_args.ref_latents == []
    assert engine._vision_cache is None


def test_krea2_slg_and_workflow_advice(monkeypatch):
    from modules import shared
    from modules.processing_scripts.slg import SkipLayerGuidance, krea2_workflow_advice

    monkeypatch.setattr(dynamic_args, "moss", False)
    monkeypatch.setattr(dynamic_args, "krea2", True)

    # Advice for Turbo with high steps/CFG
    monkeypatch.setattr(shared, "opts", SimpleNamespace(krea2_do_reference=False, slg_enable=True, slg_single_layers="5, 10, 99", slg_start_percent=0.0, slg_end_percent=0.5))
    p_turbo = SimpleNamespace(
        steps=32,
        cfg_scale=4.0,
        prompt="a fox",
        all_prompts=["a fox"],
        sd_model=SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name="krea2_turbo_fp8.safetensors"), ref_latents=[], ini_latent=None),
    )
    notes = krea2_workflow_advice(p_turbo)
    assert any("Krea-2-Turbo" in n for n in notes)

    # Advice for Raw with CFG 1.0 / 8 steps (without Turbo LoRA)
    p_raw = SimpleNamespace(
        steps=8,
        cfg_scale=1.0,
        prompt="a fox",
        all_prompts=["a fox"],
        sd_model=SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name="krea2_raw_bf16.safetensors"), ref_latents=[], ini_latent=None),
    )
    notes_raw = krea2_workflow_advice(p_raw)
    assert any("Krea-2-Raw" in n for n in notes_raw)

    # Raw + Turbo LoRA at 8 steps CFG 1.0 produces NO Raw warning
    p_raw_turbo_lora = SimpleNamespace(
        steps=8,
        cfg_scale=1.0,
        prompt="a fox <lora:krea2_turbo_lora_rank_64_bf16:1>",
        all_prompts=["a fox <lora:krea2_turbo_lora_rank_64_bf16:1>"],
        sd_model=SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name="krea2_raw_bf16.safetensors"), ref_latents=[], ini_latent=None),
    )
    assert krea2_workflow_advice(p_raw_turbo_lora) == []

    # Edit ref attached while krea2_do_reference=False emits hint
    p_ref = SimpleNamespace(
        steps=8,
        cfg_scale=1.0,
        prompt="edit",
        all_prompts=["edit"],
        sd_model=SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name="krea2_turbo_fp8.safetensors"), ref_latents=[torch.zeros(1)], ini_latent=None),
    )
    assert any("Enable Reference" in n for n in krea2_workflow_advice(p_ref))

    # SkipLayerGuidance emits Krea 2 workflow advice without silently applying MOSS's default layer-22 SLG to Krea-2-Raw
    captured_fn = {}

    class _FakeUnet:
        def __init__(self):
            self.model = SimpleNamespace(
                diffusion_model=SimpleNamespace(blocks=[object() for _ in range(28)]),
                predictor=SimpleNamespace(percent_to_sigma=lambda pct: 1.0 - pct),
            )

        def clone(self):
            return self

        def set_model_sampler_calc_cond_batch_function(self, fn):
            captured_fn["fn"] = fn

    slg = SkipLayerGuidance()
    comments: list[str] = []
    p_slg = SimpleNamespace(
        steps=8,
        cfg_scale=1.0,
        prompt="a fox",
        all_prompts=["a fox"],
        is_hr_pass=False,
        comment=comments.append,
        extra_generation_params={},
        sd_model=SimpleNamespace(
            sd_checkpoint_info=SimpleNamespace(name="krea2_raw_bf16.safetensors"),
            ref_latents=[],
            ini_latent=None,
            forge_objects=SimpleNamespace(unet=_FakeUnet()),
        ),
    )
    slg.process_before_every_sampling(p_slg)
    assert "fn" not in captured_fn
    assert p_slg.extra_generation_params == {}
    assert any("Krea-2-Raw" in c for c in comments)

    p_slg.steps = 28
    p_slg.cfg_scale = 4.0
    slg.process_before_every_sampling(p_slg)
    assert "fn" not in captured_fn and p_slg.extra_generation_params == {}
