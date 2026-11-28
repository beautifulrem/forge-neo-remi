"""Targeted behavioral tests covering edge cases in Remi's strict modules."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from backend.args import dynamic_args


def test_anima_presets_prompt_alpha_and_workflow_advice(monkeypatch):
    from backend.args import dynamic_args as da
    from modules import anima_presets

    assert anima_presets._iter_prompt_texts(None) == []
    assert anima_presets._iter_prompt_texts("hello") == ["hello"]
    assert anima_presets._active_lora_names("<lora:zero:0.0> <lora:bad:nanx> <lora:turbo_fast:0.8>") == ["bad", "turbo_fast"]

    class BrokenImg:
        def getbands(self):
            return ("R", "G", "B", "A")

        def getchannel(self, _name):
            raise RuntimeError("broken")

    assert anima_presets._has_transparent_alpha(object()) is False
    assert anima_presets._has_transparent_alpha(Image.new("RGB", (2, 2))) is False
    assert anima_presets._has_transparent_alpha(Image.new("RGBA", (2, 2), (10, 20, 30, 128))) is True
    assert anima_presets._has_transparent_alpha(BrokenImg()) is False

    assert anima_presets._is_white_background("#ffffff") is True
    assert anima_presets._is_white_background("rgb(255, 255, 255)") is True
    assert anima_presets._is_white_background("black") is False
    assert anima_presets._is_white_background((255, 255, 255)) is True
    assert anima_presets._is_white_background(("x", "y", "z")) is False
    assert anima_presets._is_white_background(123) is False

    assert anima_presets.turbo_settings_advice("anima-base.safetensors", 1.0, 8, prompts="<lora:my_turbo:1.0>") is None
    base_warn = anima_presets.turbo_settings_advice("anima-base.safetensors", 1.0, 8, prompts="")
    assert base_warn is not None and "Non-Turbo Anima" in base_warn

    p = SimpleNamespace(
        sd_model=SimpleNamespace(
            sd_checkpoint_info=SimpleNamespace(name="anima-edit.safetensors"),
            forge_objects=SimpleNamespace(vae=SimpleNamespace(output_channels=4)),
        ),
        prompt="cat",
        hr_prompt="cat",
        prompts=["cat"],
        hr_prompts=["cat"],
        cfg_scale=4.0,
        steps=20,
        init_images=[Image.new("RGBA", (4, 4), (0, 0, 0, 128))],
    )
    monkeypatch.setattr(da, "ref_latents", [], raising=False)
    advice = anima_presets.anima_workflow_advice(p, opts=SimpleNamespace(anima_do_reference=False, img2img_background_color="#808080"))
    assert len(advice) == 2


def test_flux3_and_fibo_diffusion_engines_lifecycle(monkeypatch):
    from backend.diffusion_engine import fibo as fibo_engine_mod
    from backend.diffusion_engine import flux3 as flux3_engine_mod
    from backend.diffusion_engine.base import ForgeDiffusionEngine
    from backend.nn.dit_capabilities_ext import register_extended_dit_capabilities

    fake_clip = SimpleNamespace(
        cond_stage_model=SimpleNamespace(qwen3vl_4b=object()),
        tokenizer=SimpleNamespace(qwen3vl_4b=object()),
        patcher=object(),
    )
    fake_vae = SimpleNamespace()
    fake_unet = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(clear_caches=lambda: cleared.append(True))))
    cleared: list[bool] = []
    encode_calls: list[int] = []

    class FakeTextEngine:
        def __init__(self, text_encoder=None, tokenizer=None):
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer

        def __call__(self, prompt):
            return list(prompt)

        def tokenize(self, text):
            return text.split()

    for mod in (flux3_engine_mod, fibo_engine_mod):
        monkeypatch.setattr(mod, "CLIP", lambda **kw: fake_clip)
        monkeypatch.setattr(mod, "VAE", lambda **kw: fake_vae)
        monkeypatch.setattr(mod.UnetPatcher, "from_model", lambda **kw: fake_unet)
        monkeypatch.setattr(mod, "Flux3TextEngine", FakeTextEngine)
        monkeypatch.setattr(mod.memory_management, "load_model_gpu", lambda _m: None)
    monkeypatch.setattr(ForgeDiffusionEngine, "encode_first_stage", lambda self, x, multiple=0: (encode_calls.append(multiple), x)[1])

    cfg = SimpleNamespace(latent_format=SimpleNamespace(), inpaint_model=lambda: False, model_type=SimpleNamespace(name="FLOW"), sampling_settings={})
    comps = {"vae": SimpleNamespace(), "text_encoder": object(), "tokenizer": object(), "transformer": SimpleNamespace(flow_map=True, is_edit_turbo=True)}

    orig_refs = list(dynamic_args.ref_latents)
    try:
        monkeypatch.setattr(flux3_engine_mod, "opts", SimpleNamespace(flux3_do_reference=True), raising=False)
        f3 = flux3_engine_mod.Flux3(cfg, comps)
        f3.ini_latent = torch.zeros(1, 4, 2, 2)
        f3.ref_latents = [torch.ones(1, 4, 2, 2)]
        assert len(f3.references()) == 2
        assert f3.get_prompt_lengths_on_ui("hello world") == (2, 999)
        assert f3.encode_first_stage(torch.zeros(1, 3, 4, 4)).shape == (1, 3, 4, 4)

        register_extended_dit_capabilities()
        fibo = fibo_engine_mod.Fibo2(cfg, comps)
        fibo.ini_latent = torch.zeros(1, 4, 2, 2)
        fibo.ref_latents = [torch.ones(1, 4, 2, 2)]
        assert len(fibo.references()) == 2
        cond = fibo.get_learned_conditioning(["a prompt"])
        assert len(cond) == 1
        fibo.fibo_do_reference = False
        fibo.get_learned_conditioning(["another"])
        assert dynamic_args.ref_latents == []
        fibo.clear_references()
        assert cleared == [True]
        assert fibo.get_prompt_lengths_on_ui("a b c")[1] == 999
        assert fibo.encode_first_stage(torch.zeros(1, 3, 4, 4)).shape == (1, 3, 4, 4)
        assert encode_calls == [16, 0]
    finally:
        dynamic_args.ref_latents = orig_refs


def test_dit_capabilities_ext_structured_advice_and_dir_loaders(tmp_path, monkeypatch):
    from backend.nn import dit_capabilities_ext as ext
    from backend.nn import dit_registry

    monkeypatch.setattr(ext, "zimage_workflow_advice", lambda p, opts=None: ["Z-Image-Turbo mismatch", "Z-Image Base low", "Z-Image info"])
    monkeypatch.setattr(ext, "ernie_workflow_advice", lambda p, opts=None: ["ERNIE-Image Turbo mismatch", "ERNIE-Image Base low"])
    monkeypatch.setattr(ext, "fibo_workflow_advice", lambda p, opts=None: ["Bria Fibo 2.0 Edit Turbo", "Bria Fibo 2.0 Dev detected", "Bria Fibo 2.0 VGL note"])

    assert [i.code for i in ext._zimage_structured_advice(None)] == ["zimage.turbo_mismatch", "zimage.base_low_steps", "zimage.info"]
    assert [i.code for i in ext._ernie_structured_advice(None)] == ["ernie.turbo_mismatch", "ernie.base_low_steps"]
    assert [i.code for i in ext._fibo_structured_advice(None)] == ["fibo.turbo_mismatch", "fibo.dev_low_steps", "fibo.vgl_or_ref"]

    monkeypatch.setattr(dit_registry, "AdviceItem", None)
    assert ext._zimage_structured_advice(None) == []
    assert ext._ernie_structured_advice(None) == []
    assert ext._fibo_structured_advice(None) == []

    for key, cls_name, loader in (
        ("z", "ZImageTransformer2DModel", (ext._is_zimage_dir, ext._load_zimage_dir)),
        ("e", "ErnieImageTransformer2DModel", (ext._is_ernie_dir, ext._load_ernie_dir)),
        ("f", "FiboTransformer2DModel", (ext._is_fibo_dir, ext._load_fibo_dir)),
    ):
        d = tmp_path / f"dir_{key}"
        (d / "transformer").mkdir(parents=True)
        (d / "transformer" / "config.json").write_text(json.dumps({"_class_name": cls_name}))
        save_file({"img_in.weight": torch.zeros(2, 2)}, str(d / "transformer" / "diffusion_pytorch_model.safetensors"))
        assert loader[0](d) is True
        sd, _ = loader[1](d)
        assert "model.diffusion_model.img_in.weight" in sd


def test_flux3_layout_validation_reconcile_and_reference_size():
    from backend.nn import flux3_layout as fl

    assert fl.boxes_to_canvas([{"id": "a", "src_bbox": None, "tgt_bbox": [0.1, 0.2, 0.5, 0.6]}])[0]["src_bbox"] is None
    with pytest.raises(ValueError):
        fl.boxes_to_canvas([{"bbox": [0.1, 0.2]}])
    with pytest.raises(ValueError):
        fl.boxes_to_canvas([{"id": "empty"}])

    assert fl.split_caption("caption [bad json") == ("caption [bad json", None)
    assert fl.valid_canvas_box([0, 0, 1, 1]) is None
    assert fl.valid_canvas_box([float("nan"), 0, 100, 100]) is None
    assert fl.valid_added_row("not-dict", "<id_1>") is None
    assert fl.valid_added_row({"id": "obj_1", "from": "target", "src_bbox": [10, 10, 50, 50], "desc": "ok"}, "<obj_1>") is None

    reconciled = fl.reconcile_boxes(
        'Look at <new_1> [{"id": "new_1", "bbox": [100, 100, 400, 400], "desc": "cat"}]',
        [{"id": "obj_1", "bbox": [100, 100, 400, 400], "desc": "old"}],
    )
    assert "<obj_1>" in reconciled and '"desc": "cat"' in reconciled

    probs = fl.layout_problems(
        'only <a1> [{"bad": 1}, {"id": "a1", "bbox": [0, 0, 1, 1]}, {"id": "a1", "bbox": [10, 10, 50, 50]}]'
    )
    assert len(probs) >= 3
    assert fl.reference_size(32, 100000, 3)[0] == 32
    assert fl.reference_size(4096, 4096, 3) == (512, 512)


def test_moss_ernie_zimage_and_fibo_diffusers_helpers(tmp_path):
    from backend import operations
    from backend.nn import ernie_diffusers as ed
    from backend.nn import fibo as fibo_nn
    from backend.nn import moss_diffusers as md
    from backend.nn import zimage_diffusers as zd

    # MOSS
    empty_dir = tmp_path / "empty_moss"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        md.load_moss_diffusers_dir(empty_dir)

    moss_dir = tmp_path / "moss_pipe"
    for sub in ("transformer", "text_encoder", "vae"):
        (moss_dir / sub).mkdir(parents=True)
    save_file({"layers.0.proj.qkv.weight": torch.zeros(6, 2)}, str(moss_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    save_file({"model.embed.weight": torch.zeros(2, 2)}, str(moss_dir / "text_encoder" / "model.safetensors"))
    save_file(
        {"decoder.post_quant_conv.weight": torch.zeros(2, 2), "encoder.quant_conv.weight": torch.zeros(2, 2)},
        str(moss_dir / "vae" / "diffusion_pytorch_model.safetensors"),
    )
    msd, _ = md.load_moss_diffusers_dir(moss_dir)
    assert "vae.post_quant_conv.weight" in msd and "vae.quant_conv.weight" in msd

    assert md.is_kromoss_variant("", metadata={"title": "Kromoss 8-step"}) is True
    assert md.is_dmd2_lora("moss_dmd2.safetensors", state_dict={"x": torch.zeros(1)}) is False
    assert md.is_dmd2_lora("moss_dmd2.safetensors", state_dict={"layers.0.proj.q.weight": torch.zeros(1)}) is True

    split_km = md.moss_lora_key_map(
        {
            "other.weight": torch.zeros(2, 2),
            "diffusion_model.layers.0.proj.q.weight": torch.zeros(2, 2),
            "diffusion_model.layers.0.proj.k.weight": torch.zeros(2, 2),
            "diffusion_model.layers.0.proj.v.weight": torch.zeros(2, 2),
        }
    )
    assert "layers.0.proj.qkv" in split_km
    assert md.kromoss_dmd2_workflow_advice(SimpleNamespace(sd_model=None, prompt="a")) == []

    # ERNIE
    with pytest.raises(FileNotFoundError):
        ed.load_ernie_diffusers_dir(empty_dir)
    ernie_dir = tmp_path / "ernie_pipe"
    for sub in ("transformer", "vae"):
        (ernie_dir / sub).mkdir(parents=True)
    (ernie_dir / "model_index.json").write_text(json.dumps({"transformer": ["ernie", ed.DIFFUSERS_CLASS]}))
    save_file({"layers.0.adaLN_sa_ln.weight": torch.zeros(2)}, str(ernie_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    save_file(
        {"decoder.post_quant_conv.weight": torch.zeros(2, 2), "encoder.quant_conv.weight": torch.zeros(2, 2)},
        str(ernie_dir / "vae" / "diffusion_pytorch_model.safetensors"),
    )
    assert ed.ernie_diffusers_primary_file(ernie_dir) is not None
    assert ed.ernie_diffusers_primary_file(ernie_dir / "transformer") is not None
    assert ed.is_ernie_diffusers_dir(ernie_dir) is True
    esd, _ = ed.load_ernie_diffusers_dir(ernie_dir)
    assert "vae.post_quant_conv.weight" in esd and "vae.quant_conv.weight" in esd

    ernie_shard = tmp_path / "ernie_shard"
    ernie_shard.mkdir()
    save_file({"layers.0.adaLN_sa_ln.weight": torch.zeros(2), "layers.0.mlp.linear_fc2.weight": torch.zeros(2)}, str(ernie_shard / "model.safetensors"))
    (ernie_shard / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.adaLN_sa_ln.weight": "model.safetensors", "layers.0.mlp.linear_fc2.weight": "model.safetensors"}})
    )
    assert ed.is_sharded_ernie_transformer_dir(ernie_shard) is True
    assert "model.diffusion_model.layers.0.adaLN_sa_ln.weight" in ed.load_ernie_diffusers_dir(ernie_shard)[0]
    assert ed.is_sharded_ernie_transformer_dir(empty_dir) is False

    class FakeErnieDiT:
        patch_size = 2
        out_channels = 4

        def time_proj(self, t):
            return t.unsqueeze(-1).repeat(1, 4)

        def time_embedding(self, x):
            return x

        def final_norm(self, f, _c):
            return f

        def final_linear(self, h):
            return torch.zeros(h.shape[0], h.shape[1], 16)

    pred = SimpleNamespace(timestep=lambda s: s[0], calculate_denoised=lambda _s, out, _x: out)
    ff = ed.fast_forward_ernie(FakeErnieDiT(), pred, torch.tensor([0.5]), torch.zeros(2, 4, 4, 4), torch.zeros(2, 6, 4))
    assert ff.shape == (2, 4, 4, 4)
    assert ed.ernie_workflow_advice(SimpleNamespace(sd_model=SimpleNamespace(is_ernie_engine=True), steps=4, cfg_scale=1.0, prompt="x"))

    # Z-Image
    assert zd._strip_prefix("diffusion_model.x") == "x"
    assert zd.is_zimage_state_dict({"cap_embedder.1.weight": torch.zeros(3840, 16), "layers.0.attention.qkv.weight": torch.zeros(2, 2)}) is True
    z_conv = zd.convert_diffusers_zimage_transformer(
        {
            "layers.0.attention.to_q.comfy_quant": torch.zeros(4),
            "layers.0.attention.to_k.comfy_quant": torch.zeros(4),
            "layers.0.attention.to_v.comfy_quant": torch.zeros(4),
            "layers.0.attention.to_q.weight_scale": torch.tensor(1.0),
            "layers.0.attention.to_k.weight_scale": torch.tensor(2.0),
            "layers.0.attention.to_v.weight_scale": torch.tensor(3.0),
        }
    )
    assert float(z_conv["layers.0.attention.qkv.weight_scale"]) == 3.0

    z_dir = tmp_path / "z_pipe"
    (z_dir / "transformer").mkdir(parents=True)
    (z_dir / "model_index.json").write_text(json.dumps({"_class_name": "ZImageOmniPipeline", "transformer": ["d", zd.DIFFUSERS_CLASS]}))
    save_file({"x": torch.zeros(2)}, str(z_dir / "transformer" / "model.safetensors"))
    assert zd.zimage_diffusers_primary_file(z_dir) is not None
    assert zd.zimage_diffusers_primary_file(z_dir / "transformer") is not None
    assert zd.is_zimage_diffusers_dir(z_dir) is True
    _, zmeta = zd.load_zimage_diffusers_dir(z_dir)
    assert zmeta["zimage_variant"] == "omni"
    z_shard = tmp_path / "z_shard"
    z_shard.mkdir()
    save_file({"x": torch.zeros(2)}, str(z_shard / "model.safetensors"))
    (z_shard / "config.json").write_text(json.dumps({"_class_name": zd.DIFFUSERS_CLASS}))
    assert zd.is_sharded_zimage_transformer_dir(z_shard) is True
    z_advice = zd.zimage_workflow_advice(
        SimpleNamespace(sd_model=SimpleNamespace(is_zimage_engine=True, zimage_do_reference=False, ini_latent=torch.zeros(1), ref_latents=[]), steps=4, cfg_scale=1.0, prompt="x")
    )
    assert len(z_advice) == 2

    # FIBO
    with pytest.raises(FileNotFoundError):
        fibo_nn.load_fibo_diffusers_dir(empty_dir)
    fibo_dir = tmp_path / "fibo_pipe"
    for sub in ("transformer", "vae"):
        (fibo_dir / sub).mkdir(parents=True)
    (fibo_dir / "model_index.json").write_text(json.dumps({"transformer": ["f", fibo_nn.DIFFUSERS_CLASS], "is_edit_turbo": True}))
    save_file({"img_in.weight": torch.zeros(2)}, str(fibo_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    save_file(
        {"decoder.post_quant_conv.weight": torch.zeros(2, 2), "encoder.quant_conv.weight": torch.zeros(2, 2)},
        str(fibo_dir / "vae" / "diffusion_pytorch_model.safetensors"),
    )
    assert fibo_nn.fibo_diffusers_primary_file(fibo_dir) is not None
    assert fibo_nn.fibo_diffusers_primary_file(fibo_dir / "transformer") is not None
    assert fibo_nn.is_fibo_diffusers_dir(fibo_dir) is True
    fsd, fmeta = fibo_nn.load_fibo_diffusers_dir(fibo_dir)
    assert fmeta["fibo_variant"] == "edit_turbo" and "vae.post_quant_conv.weight" in fsd
    fibo_shard = tmp_path / "fibo_shard"
    fibo_shard.mkdir()
    save_file({"transformer_blocks.0.adaln.1.weight": torch.zeros(2), "final_layer.adaln.1.weight": torch.zeros(2)}, str(fibo_shard / "model.safetensors"))
    (fibo_shard / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"transformer_blocks.0.adaln.1.weight": "model.safetensors", "final_layer.adaln.1.weight": "model.safetensors"}})
    )
    assert fibo_nn.is_sharded_fibo_transformer_dir(fibo_shard) is True
    assert "model.diffusion_model.final_layer.adaln.1.weight" in fibo_nn.load_fibo_diffusers_dir(fibo_shard)[0]

    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        tiny_fibo = fibo_nn.FiboTransformer2DModel(
            in_channels=4,
            out_channels=4,
            hidden_size=16,
            num_attention_heads=2,
            num_layers=1,
            text_in_dim=8,
            axes_dims_rope=(2, 2, 4),
            guidance_embeds=False,
        )
    for idx in range(fibo_nn._ROPE_CACHE_MAX + 2):
        tiny_fibo._cached_rope(2, 2 + idx, 2, (), torch.device("cpu"))
    assert len(tiny_fibo._rope_cache) == fibo_nn._ROPE_CACHE_MAX
    kept_ctx = [torch.full((1, 2, 8), float(idx)) for idx in range(fibo_nn._TEXT_CACHE_MAX + 2)]
    with torch.no_grad():
        for ctx_item in kept_ctx:
            tiny_fibo._embed_text(ctx_item)
        assert len(tiny_fibo._text_cache) == fibo_nn._TEXT_CACHE_MAX
        tiny_fibo._embed_text(kept_ctx[-1])  # cache hit move_to_end
    x = torch.zeros(2, 4, 2, 2)
    t = torch.tensor([0.5])
    ctx = torch.zeros(1, 2, 8)
    refs = [torch.zeros(1, 4, 2, 2)]
    replace_called: list[bool] = []
    out = tiny_fibo(
        x,
        t,
        ctx,
        ref_latents=refs,
        transformer_options={"patches_replace": {"dit": {("single_block", 0): lambda args, extra: (replace_called.append(True), extra["original_block"](args))[1]}}},
    )
    assert out.shape == (2, 4, 2, 2) and replace_called == [True]
    assert tiny_fibo.compute_fibo_final(torch.zeros(2, 4, 16), t, (2, 4, 2, 2)).shape == (2, 4, 2, 2)
    with torch.enable_grad():
        assert tiny_fibo._embed_text(torch.zeros(1, 2, 8)).shape == (1, 2, 16)

    f_advice = fibo_nn.fibo_workflow_advice(
        SimpleNamespace(
            sd_model=SimpleNamespace(
                is_fibo_engine=True,
                is_fibo_turbo=False,
                ini_latent=torch.zeros(1),
                ref_latents=[torch.zeros(1)] * fibo_nn.FIBO_MAX_REFERENCES,
            ),
            steps=4,
            cfg_scale=1.0,
            prompt="plain",
        )
    )
    assert len(f_advice) == 2


def test_quant_formats_and_attention_selection(monkeypatch):
    from backend import attention as attn_mod
    from backend.nn import dit_registry
    from backend.nn import quant_formats as qf

    def comfy_meta(fmt: str, extra: dict | None = None) -> torch.Tensor:
        payload = {"format": fmt, **(extra or {})}
        return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)

    info_int8 = qf.detect_quant_format({"a.weight": torch.zeros(2, 2, dtype=torch.int8), "a.comfy_quant": comfy_meta("int8_tensorwise")})
    assert info_int8.to_dict()["format_id"] == "int8_tensorwise"
    assert qf.detect_quant_format({"a.weight": torch.zeros(2, 2), "a.comfy_quant": comfy_meta("float8_e4m3fn_rowwise")}).format_id == "fp8_e4m3_rowwise"
    assert qf.detect_quant_format({"a.weight": torch.zeros(2, 2), "a.comfy_quant": comfy_meta("float8_e5m2")}).format_id == "fp8_e5m2"
    assert qf.detect_quant_format({"a.weight": torch.zeros(2, 2), "a.comfy_quant": comfy_meta("fp8")}).format_id == "fp8_e4m3"

    assert qf._decode_comfy_quant(torch.tensor([255], dtype=torch.uint8)) == {}
    assert qf._parse_unsloth_meta({"unsloth_metadata": {"a": 1}}) == {"a": 1}
    assert qf._parse_unsloth_meta({"unsloth_metadata": "{bad"}) == {}

    assert qf.detect_quant_format({"a._weight_qdata": torch.zeros(2, 2, dtype=torch.int8)}, {"unsloth_metadata": {"activation_rotation": "convrot"}}).format_id == "int8_convrot"
    assert qf.detect_quant_format({"a._weight_qdata": torch.zeros(2, 2, dtype=torch.int8)}).format_id == "int8_tensorwise"
    assert qf.detect_quant_format({"a._weight_qdata": torch.zeros(2, 2, dtype=torch.float8_e4m3fn)}, {"unsloth_metadata": {"fp8_granularity": "per_row"}}).format_id == "fp8_e4m3_rowwise"
    assert qf.detect_quant_format({"a._weight_qdata": torch.zeros(2, 2, dtype=torch.uint8)}, {"unsloth_metadata": {"scheme": "nvfp4"}}).format_id == "nvfp4"

    assert qf.detect_quant_format({"a.weight": torch.zeros(2, 2, dtype=torch.float8_e5m2)}).format_id == "fp8_e5m2"
    assert qf.detect_quant_format({"a.weight": torch.zeros(2, 2, dtype=torch.float16)}).format_id == "fp16"
    assert qf.detect_quant_format({"a.weight": torch.zeros(2, 2, dtype=torch.float32)}).format_id == "fp32"
    assert qf.detect_quant_format({"a.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}).format_id == "bf16"

    monkeypatch.setattr(dit_registry, "get_dit_capability", lambda key: SimpleNamespace(int8_convrot_filter=lambda _: True) if key == "custom_dit" else None)
    assert qf.quant_formats_for_dit("unknown_dit") == ("bf16", "fp8", "gguf")
    assert qf.quant_formats_for_dit("custom_dit") == ("bf16", "fp8", "int8_convrot", "gguf")

    assert "unknown" in (qf.backend_incompatibility_reason("nope") or "")
    assert "max head_dim" in (qf.backend_incompatibility_reason("flash", device="cuda", head_dim=512) or "")
    assert "head_dims" in (qf.backend_incompatibility_reason("sage3", device="cuda", head_dim=96) or "")
    assert qf.backend_incompatibility_reason("flash", device="cuda", head_dim=128, has_mask=True, mask_is_causal=True) is None
    assert "boolean" in (qf.backend_incompatibility_reason("xformers", device="cuda", head_dim=128, has_mask=True, mask_is_bool=True) or "")
    assert "additive" in (qf.backend_incompatibility_reason("sage3", device="cuda", head_dim=128, has_mask=True, mask_is_bool=False) or "")
    assert "GQA" in (qf.backend_incompatibility_reason("sage2", device="cuda", head_dim=128, needs_gqa=True) or "")
    assert "graph-safe" in (qf.backend_incompatibility_reason("xformers", device="cuda", head_dim=128, compiled=True) or "")

    assert qf.select_attention_backend(preferred="sdpa", device="cpu", available_backends=["sdpa"])[0] == "sdpa"
    assert qf.select_attention_backend(device="meta", available_backends=["unknown"]) == ("basic", "fallback to basic attention")

    monkeypatch.setattr(attn_mod, "attention_pytorch", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("sdpa fail")))
    monkeypatch.setattr(attn_mod, "attention_basic", lambda q, k, v, heads, **kw: q + 1.0)
    out = qf.safe_attention_call(torch.zeros(1, 4, 8), torch.zeros(1, 4, 8), torch.zeros(1, 4, 8), 2, causal=True)
    assert float(out.sum()) == 32.0


def test_dit_control_patch_edge_cases_and_union_patcher(tmp_path, monkeypatch):
    from backend.nn import dit_control_patch as dcp

    assert dcp.strip_control_prefix({"control_model.control_img_in.weight": torch.zeros(4, 64)}).keys() == {"control_img_in.weight"}
    assert dcp.detect_dit_control_family({"control_blocks.0.x": torch.zeros(1), "control_img_in.weight": torch.zeros(4, 129)}) == "qwen21"
    assert dcp.detect_dit_control_family({"control_blocks.0.x": torch.zeros(1), "control_img_in.weight": torch.zeros(4, 33)}) == "zimage"
    assert dcp.detect_dit_control_family({"control_blocks.0.x": torch.zeros(1), "control_img_in.weight": torch.zeros(4, 99)}) is None

    monkeypatch.setattr(dynamic_args, "anima", True, raising=False)
    assert dcp.active_control_family() == "anima"
    monkeypatch.setattr(dynamic_args, "anima", False, raising=False)
    monkeypatch.setattr(dynamic_args, "qwen21", True, raising=False)
    assert dcp.active_control_family() == "qwen21"
    monkeypatch.setattr(dynamic_args, "qwen21", False, raising=False)
    monkeypatch.setattr(dynamic_args, "zimage", True, raising=False)
    assert dcp.active_control_family() == "zimage"
    monkeypatch.setattr(dynamic_args, "zimage", False, raising=False)
    monkeypatch.setitem(sys.modules, "modules.sd_models", SimpleNamespace(model_data=SimpleNamespace(sd_model=SimpleNamespace(is_qwen21_engine=True))))
    assert dcp.active_control_family() == "qwen21"

    with pytest.raises(ValueError):
        dcp.Qwen21FunControlNet.infer_config({})
    with pytest.raises(ValueError):
        dcp.Qwen21FunControlNet.infer_config({"control_img_in.weight": torch.zeros(16, 129)})

    q_cfg = dcp.Qwen21FunControlNet.infer_config(
        {
            "control_img_in.weight": torch.zeros(16, 129),
            "control_blocks.0.block.img_mlp.proj.weight": torch.zeros(32, 16),
        },
        metadata={"control_layers": "0, 2"},
    )
    assert q_cfg["control_layers"] == [0, 2]
    q_model = dcp.Qwen21FunControlNet(
        inner_dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        mlp_ratio=2,
        fused_mlp=False,
        control_in_dim=17,
        latent_channels=8,
        num_base_layers=4,
        control_layers=[0, 5],
        _dtype=torch.float32,
    )
    assert q_model.build_control_input(torch.zeros(1, 17, 2, 2)).shape == (1, 17, 2, 2)
    built_q = q_model.build_control_input(torch.zeros(2, 8, 2, 2), mask=torch.ones(4, 4), masked_latent=torch.zeros(1, 8, 4, 4))
    assert built_q.shape == (2, 17, 2, 2)
    with pytest.raises(ValueError):
        q_model.build_control_input(torch.zeros(1, 3, 2, 2))
    assert q_model.target_layers_for_dit(4) == [0, 2]

    with pytest.raises(ValueError):
        dcp.ZImageFunControlNet.infer_config({})
    with pytest.raises(ValueError):
        dcp.ZImageFunControlNet.infer_config({"control_img_in.weight": torch.zeros(16, 64)})
    z_cfg = dcp.ZImageFunControlNet.infer_config(
        {"control_img_in.weight": torch.zeros(16, 64), "control_blocks.0.x": torch.zeros(1)},
        metadata={"control_layers": "0,1"},
    )
    assert z_cfg["control_layers"] == [0, 1] and z_cfg["qk_norm"] is False
    z_model = dcp.ZImageFunControlNet(
        dim=16,
        n_heads=2,
        qk_norm=False,
        patch_size=2,
        control_in_dim=36,  # 2 * 2 * (2*4 + 1)
        latent_channels=4,
        num_base_layers=4,
        control_layers=[0, 5],
        _dtype=torch.float32,
    )
    assert z_model.build_control_input(torch.zeros(1, 9, 2, 2)).shape == (1, 1, 36)
    assert z_model.build_control_input(torch.zeros(2, 4, 2, 2), mask=torch.ones(1, 4, 4), masked_latent=torch.zeros(1, 4, 4, 4)).shape == (2, 1, 36)
    with pytest.raises(ValueError):
        z_model.build_control_input(torch.zeros(1, 3, 2, 2))
    assert z_model.target_layers_for_dit(4) == [0, 2]

    # Step window + UnionPatch edge cases
    p_step = SimpleNamespace(is_hr_pass=False, sampler=SimpleNamespace(model_wrap_cfg=SimpleNamespace(step=1, total_steps=10)))
    win = dcp._step_window(p_step, 0.2, 0.8)
    assert win is not None and win() is False
    p_step.sampler.model_wrap_cfg.step = 3
    assert win() is True
    p_step.is_hr_pass = True
    assert win() is False

    patch_z = dcp.FunControlNetUnionPatch(z_model, torch.zeros(1, 4, 2, 2), strength=0.0)
    with pytest.raises(ValueError):
        patch_z._block_list(torch.nn.Module())
    assert patch_z._pre_block0(None, (torch.zeros(1, 2, 16),), {}) is None
    assert patch_z._make_post_hook(0)(None, (torch.zeros(1, 2, 16),), {}, torch.zeros(1, 2, 16)).shape == (1, 2, 16)

    patch_q = dcp.FunControlNetUnionPatch(q_model, torch.zeros(1, 8, 2, 2))
    with pytest.raises(ValueError):
        patch_q._block_list(torch.nn.Module())
    mod_row = torch.zeros(2, 2, 16)
    mod_tuple = ((mod_row, mod_row), (mod_row, mod_row), (mod_row, mod_row), (mod_row, mod_row), torch.zeros(2, 16))
    pe_q = torch.zeros(1, 4, 4, 2, 2)
    # Mismatch tokens -> returns None (warns once)
    assert patch_q._compute_qwen21_hints((torch.zeros(1, 1, 16), mod_tuple, pe_q[:, :1], None, 0), {}) is None
    # Batch repeat 1 -> 2, full span output + hint (mock forward output to isolate FunControlNetUnionPatch dispatch)
    monkeypatch.setattr(q_model, "forward", lambda *a, **kw: {0: torch.ones(2, 4, 16)})
    q_hints = patch_q._compute_qwen21_hints((torch.zeros(2, 4, 16), mod_tuple, pe_q, None, 0), {})
    assert q_hints is not None and set(q_hints) == {0} and q_hints[0].shape == (2, 4, 16)
    assert patch_q._apply_hint_to_output(torch.zeros(2, 4, 16), torch.ones(2, 4, 16)).sum() == 128

    # ZImage patch mismatch + batch repeat + trailing start fallback
    patch_z2 = dcp.FunControlNetUnionPatch(z_model, torch.zeros(1, 4, 4, 4))
    assert patch_z2._compute_zimage_hints((torch.zeros(1, 1, 16), None, torch.zeros(1, 1, 4, 2), torch.zeros(1, 16)), {}) is None
    patch_z3 = dcp.FunControlNetUnionPatch(z_model, torch.zeros(1, 4, 2, 2))
    monkeypatch.setattr(z_model, "forward", lambda *a, **kw: {0: torch.ones(2, 1, 16)})
    z_hints = patch_z3._compute_zimage_hints((torch.zeros(2, 3, 16), None, torch.zeros(1, 3, 4, 2), torch.zeros(2, 16)), {})
    assert z_hints is not None and set(z_hints) == {0} and z_hints[0].shape == (2, 1, 16)

    # DiTControlNetUnionPatcher lifecycle
    assert dcp.DiTControlNetUnionPatcher.try_build_from_state_dict({"x": torch.zeros(1)}) is None
    sf = tmp_path / "ctrl.safetensors"
    save_file({"control_img_in.weight": torch.zeros(16, 17), "control_blocks.0.block.img_mlp.proj.weight": torch.zeros(32, 16)}, str(sf))
    monkeypatch.setattr(dcp, "load_torch_file", lambda _p, return_metadata=False: ({}, {"control_layers": "0"}))
    # try_build_from_state_dict with safetensors metadata error & success
    patcher = dcp.DiTControlNetUnionPatcher(q_model)
    assert patcher.process_after_running_preprocessors(None, None) is None
    assert patcher._patch is None
    assert patcher.process_before_conditioning(None) is False
    patcher.start_percent = 0.2
    patcher.end_percent = 0.8

    fake_dit = torch.nn.Module()
    fake_dit.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])
    fake_unet = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=fake_dit, computation_dtype=torch.float32, predictor=SimpleNamespace(percent_to_sigma=lambda p: p)),
        load_device=torch.device("cpu"),
        model_options={},
        clone=lambda: fake_unet,
    )
    fake_vae = SimpleNamespace(encode=lambda x: torch.zeros(x.shape[0], 8, 2, 2))
    proc = SimpleNamespace(
        sd_model=SimpleNamespace(
            forge_objects=SimpleNamespace(unet=fake_unet, vae=fake_vae),
            model_config=SimpleNamespace(latent_format=SimpleNamespace(process_in=lambda z: z)),
        ),
        inpainting_mask_invert=True,
        is_hr_pass=False,
        sampler=None,
    )
    patcher.process_before_every_sampling(proc, torch.zeros(1, 3, 4, 4), torch.ones(1, 1, 2, 2))
    first_patch = patcher._patch
    assert first_patch is not None
    assert first_patch._dit is fake_dit
    assert len(first_patch._handles) > 0
    assert len(fake_dit._forward_pre_hooks) == 1
    assert first_patch.masked_latent is not None
    assert "forge_spectrum_force_actual" in proc.sd_model.forge_objects.unet.model_options.get("transformer_options", {})

    patcher.process_before_every_sampling(proc, torch.zeros(1, 3, 4, 4), torch.ones(1, 1, 2, 2))
    second_patch = patcher._patch
    assert second_patch is not None and second_patch is not first_patch
    assert first_patch._dit is None and len(first_patch._handles) == 0
    assert second_patch._dit is fake_dit and len(fake_dit._forward_pre_hooks) == 1

    patcher.process_after_every_sampling()
    assert patcher._patch is None
    assert second_patch._dit is None and len(second_patch._handles) == 0
    assert len(fake_dit._forward_pre_hooks) == 0
    assert len(fake_dit.transformer_blocks[0]._forward_pre_hooks) == 0

    monkeypatch.setattr(dcp.memory_management, "unet_offload_device", lambda: torch.device("cpu"))
    patcher.process_after_every_sampling()
    assert patcher._patch is None
    assert next(patcher.control_model.parameters()).device.type == "cpu"
