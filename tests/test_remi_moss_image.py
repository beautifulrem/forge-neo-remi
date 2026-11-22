"""Moss-Image and Moss-Image-Edit (upstream be6fa5d4 "Moss-Image", c6c2b43d
"Moss-Image-Edit") as Remi sees them without loading the 9B text encoder or
the DiT: detection and the loader's engine list, the converted
``moss_do_reference`` setting, the engine's reference routing and its prompt
token count (the SPA token counter calls ``get_prompt_lengths_on_ui``)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "backend" / "huggingface" / "OpenMOSS-Team" / "MOSS-Image-SFT"
IMAGE_PAD = 248056  # "<|image_pad|>", replaced by the reference images
PAD = 248044  # "<|endoftext|>"


def _dit_state_dict(dim=256, ffn=512, layers=3, prefix="model.diffusion_model."):
    sd = {f"{prefix}shared_block_modulation.weight": torch.empty(6 * dim, 4)}
    for index in range(layers):
        sd[f"{prefix}layers.{index}.proj.ffn_up_gate.weight"] = torch.empty(2 * ffn, 4)
        sd[f"{prefix}layers.{index}.proj.qkv.weight"] = torch.empty(3 * dim, 4)
    return sd


def test_a_moss_dit_is_detected_and_has_an_engine():
    from huggingface_guess import guess, model_list

    from backend.diffusion_engine.moss_image import MOSSImage
    from backend.loader import possible_models

    config = guess(_dit_state_dict())
    assert isinstance(config, model_list.MOSSImage) and config.huggingface_repo == "OpenMOSS-Team/MOSS-Image-SFT"
    assert config.unet_config == {"dim": 256, "ffn_dim": 512, "num_heads": 2, "num_layers": 3, "fused_qkv": True}
    assert MOSSImage in possible_models and model_list.MOSSImage in MOSSImage.matched_guesses
    # The diffusers layout the loader builds the components from ships with Remi.
    for component in ("model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json", "transformer/config.json", "vae/config.json", "tokenizer/tokenizer_config.json"):
        assert (REPO / component).is_file(), component


def test_the_reference_setting_offers_every_method_the_engine_knows():
    from backend.diffusion_engine.moss_image import REFERENCE_METHODS
    from modules import shared_options
    from remi import ui

    info = shared_options.options_templates["moss_do_reference"]
    args = info.component_args() if callable(info.component_args) else info.component_args
    assert info.component is ui.Choice and info.default == "Disable"
    assert tuple(args["choices"]) == ("Disable", *REFERENCE_METHODS)


class _TextEngine:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, images=()):
        self.calls.append((list(prompt), list(images)))
        return ["cond"]


@pytest.fixture
def engine(monkeypatch):
    from backend import memory_management
    from backend.args import dynamic_args
    from backend.diffusion_engine import moss_image

    monkeypatch.setattr(memory_management, "load_model_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dynamic_args, "ref_latents", [])
    monkeypatch.setattr(dynamic_args, "moss_reference_method", None)
    monkeypatch.setattr(dynamic_args, "moss_ref_kv", None)
    model = moss_image.MOSSImage.__new__(moss_image.MOSSImage)
    model.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=object()))
    model.text_processing_engine_qwen = _TextEngine()
    model.ref_latents = []
    model.ini_latent = None
    # (vision input, reference latent) per image, without the VAE
    model.encode_vision = lambda image: (("vision", image), ("latent", image))
    return model


def _prompt(*texts, negative=False):
    from modules.prompt_parser import SdConditioning

    return SdConditioning(list(texts), is_negative_prompt=negative)


@pytest.mark.parametrize(
    "mode,method,vision",
    [("ai-toolkit", "index_timestep_zero_isolated", True), ("Moss-Image-Edit", "index_timestep_zero_text_start", False)],
)
def test_reference_methods_route_the_img2img_image_and_stitched_references(engine, monkeypatch, mode, method, vision):
    from backend.args import dynamic_args
    from backend.diffusion_engine import moss_image

    monkeypatch.setattr(moss_image, "opts", SimpleNamespace(moss_do_reference=mode))
    engine.ini_latent, engine.ref_latents = "init", ["stitched"]

    engine.get_learned_conditioning(_prompt("make it blue"))
    (prompt, images), = engine.text_processing_engine_qwen.calls
    assert prompt == ["make it blue"] and images == ([("vision", "init"), ("vision", "stitched")] if vision else [])
    assert dynamic_args.moss_reference_method == method and dynamic_args.ref_latents == [("latent", "init"), ("latent", "stitched")]
    assert engine.ini_latent is None  # consumed by the positive prompt

    engine.get_learned_conditioning(_prompt("", negative=True))  # the negative prompt keeps them
    assert engine.text_processing_engine_qwen.calls[-1] == ([""], [])
    assert dynamic_args.moss_reference_method == method and len(dynamic_args.ref_latents) == 2


def test_disabled_references_clear_the_previous_ones(engine, monkeypatch):
    from backend.args import dynamic_args
    from backend.diffusion_engine import moss_image

    monkeypatch.setattr(moss_image, "opts", SimpleNamespace(moss_do_reference="Disable"))
    dynamic_args.ref_latents.append("stale")
    dynamic_args.moss_reference_method, dynamic_args.moss_ref_kv = "index_timestep_zero_isolated", "kv"
    engine.ref_latents = ["stitched"]

    engine.get_learned_conditioning(_prompt("a cat"))
    assert engine.text_processing_engine_qwen.calls == [(["a cat"], [])]
    assert dynamic_args.ref_latents == [] and dynamic_args.moss_reference_method is None and dynamic_args.moss_ref_kv is None


@pytest.fixture(scope="module")
def text_engine(tmp_path_factory):
    import shutil

    from transformers import Qwen2TokenizerFast

    from backend.huggingface import Token, compare_sha256, decompress, sha256
    from backend.text_processing.moss_image_engine import Qwen359BEngine

    tokenizer_dir = REPO / "tokenizer"
    if not (tokenizer_dir / "tokenizer.json").is_file():
        # Not decompressed yet (the launcher does it): the shipped archive, in a
        # scratch folder rather than the checkout.
        scratch = tmp_path_factory.mktemp("moss-tokenizer")
        shutil.copy(tokenizer_dir / "tokenizer_config.json", scratch)
        decompress(Token.moss_compress, str(scratch / "tokenizer.json"))
        compare_sha256(str(scratch / "tokenizer.json"), sha256.moss)
        tokenizer_dir = scratch
    encoder = torch.nn.Module()
    encoder.num_layers = 32
    return Qwen359BEngine(text_encoder=encoder, tokenizer=Qwen2TokenizerFast.from_pretrained(str(tokenizer_dir)))


def test_prompt_lengths_count_the_chat_template(text_engine):
    from backend.diffusion_engine.moss_image import MOSSImage

    model = MOSSImage.__new__(MOSSImage)
    model.text_processing_engine_qwen = text_engine
    empty, _ = model.get_prompt_lengths_on_ui("")
    count, limit = model.get_prompt_lengths_on_ui("a red bicycle by the sea")
    assert empty > 0 and count > empty and limit == 999  # no 75-token chunks: one sequence
    long = " ".join(["cat"] * 1200)
    assert model.get_prompt_lengths_on_ui(long)[1] == model.get_prompt_lengths_on_ui(long)[0] > 999


def test_reference_images_fill_the_image_pad_tokens(text_engine):
    images = [torch.zeros(1, 8, 8, 3), torch.ones(1, 8, 8, 3)]
    (tokens,) = text_engine._tokenize_with_weights("swap the colours", images)
    filled = [token for token, *_ in tokens if isinstance(token, dict)]
    assert [item["data"] for item in filled] == images and all(item["type"] == "image" for item in filled)
    assert IMAGE_PAD not in [token for token, *_ in tokens]
    (plain,) = text_engine._tokenize_with_weights("swap the colours", [])
    assert not any(isinstance(token, dict) for token, *_ in plain) and PAD not in [token for token, *_ in plain]


def _tiny_moss_dit(*, fused_qkv: bool = False, num_layers: int = 2, dim: int = 32, ffn_dim: int = 64, num_heads: int = 2, text_dim: int = 16, latent_channels: int = 8):
    from backend.nn.moss_image import MOSSImage as MOSSDiT
    from backend.operations import using_forge_operations

    torch.manual_seed(0)
    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        dit = MOSSDiT(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, num_layers=num_layers, text_dim=text_dim, latent_channels=latent_channels, fused_qkv=fused_qkv).eval()
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return dit


def test_text_embedder_and_rope_caches_match_uncached_and_invalidate(monkeypatch):
    from backend.args import dynamic_args
    from backend.sampling.sampling_function import sampling_cleanup

    monkeypatch.setattr(dynamic_args, "ref_latents", [])
    monkeypatch.setattr(dynamic_args, "moss_reference_method", None)
    monkeypatch.setattr(dynamic_args, "moss_ref_kv", None)

    dit = _tiny_moss_dit()
    with torch.inference_mode():
        x = torch.randn(1, 8, 4, 4)
        ctx = torch.randn(1, 5, 16)

        # Baseline with cache disabled via an online hook on text_embedder
        dit.text_embedder.proj.weight_function = [lambda w: w]
        try:
            ref_t1 = dit(x, torch.tensor([0.9]), ctx)
            ref_t2 = dit(x, torch.tensor([0.2]), ctx)
        finally:
            dit.text_embedder.proj.weight_function = []

        dit.clear_caches()
        text_calls = 0
        rope_calls = 0
        orig_text_fwd = dit.text_embedder.forward
        orig_rope = dit.rope

        def count_text(*args, **kwargs):
            nonlocal text_calls
            text_calls += 1
            return orig_text_fwd(*args, **kwargs)

        def count_rope(*args, **kwargs):
            nonlocal rope_calls
            rope_calls += 1
            return orig_rope(*args, **kwargs)

        dit.text_embedder.forward = count_text
        dit.rope = count_rope
        try:
            out_t1 = dit(x, torch.tensor([0.9]), ctx)
            # SLG patches_replace["dit"] must NOT defeat TextEmbedder / RoPE caching
            slg_opts = {"patches_replace": {"dit": {("single_block", 1): lambda a, _e: a}}}
            out_t2 = dit(x, torch.tensor([0.2]), ctx)
            _ = dit(x, torch.tensor([0.5]), ctx, transformer_options=slg_opts)

            assert text_calls == 1 and rope_calls == 1
            assert torch.allclose(out_t1, ref_t1, atol=1e-6)
            assert torch.allclose(out_t2, ref_t2, atol=1e-6)

            # Mutating a text_embedder weight invalidates the cache immediately
            with torch.inference_mode(False), torch.no_grad():
                dit.text_embedder.proj.weight.add_(0.5)
            out_mut = dit(x, torch.tensor([0.9]), ctx)
            assert text_calls == 2
            assert not torch.equal(out_mut, out_t1)

            # sampling_cleanup clears step caches and moss_ref_kv
            dynamic_args.moss_ref_kv = torch.zeros(1)
            fake_unet = SimpleNamespace(list_controlnets=list, model=SimpleNamespace(diffusion_model=dit))
            sampling_cleanup(fake_unet)
            assert dynamic_args.moss_ref_kv is None
            assert len(dit._text_embed_cache) == 0 and len(dit._rope_cache) == 0
        finally:
            dit.text_embedder.forward = orig_text_fwd
            dit.rope = orig_rope


def test_moss_ref_kv_validates_weight_changes_and_frees_on_cleanup(monkeypatch):
    from backend.args import dynamic_args
    from backend.sampling.sampling_function import sampling_cleanup

    dit = _tiny_moss_dit()
    with torch.inference_mode():
        ref1 = torch.randn(1, 8, 2, 2)
        monkeypatch.setattr(dynamic_args, "ref_latents", [ref1])
        monkeypatch.setattr(dynamic_args, "moss_reference_method", "index_timestep_zero_text_start")
        monkeypatch.setattr(dynamic_args, "moss_ref_kv", None)

        prep_calls = 0
        orig_prep = dit.preprocess_refs

        def count_prep(*args, **kwargs):
            nonlocal prep_calls
            prep_calls += 1
            return orig_prep(*args, **kwargs)

        dit.preprocess_refs = count_prep
        x = torch.randn(1, 8, 4, 4)
        ctx = torch.randn(1, 5, 16)
        try:
            out1 = dit(x, torch.tensor([0.8]), ctx)
            out2 = dit(x, torch.tensor([0.4]), ctx)
            assert prep_calls == 1 and isinstance(dynamic_args.moss_ref_kv, torch.Tensor)
            assert (out1 - out2).abs().max() > 1e-5

            # Mutating a DiT block weight (e.g. LoRA switch under cached prompt) invalidates moss_ref_kv
            with torch.inference_mode(False), torch.no_grad():
                dit.layers[0].proj["q"].weight.add_(0.2)
            _ = dit(x, torch.tensor([0.8]), ctx)
            assert prep_calls == 2

            # Swapping reference tensor invalidates moss_ref_kv
            monkeypatch.setattr(dynamic_args, "ref_latents", [torch.randn(1, 8, 2, 2)])
            _ = dit(x, torch.tensor([0.8]), ctx)
            assert prep_calls == 3

            # Cleanup releases GPU ref KV immediately
            fake_unet = SimpleNamespace(list_controlnets=list, model=SimpleNamespace(diffusion_model=dit))
            sampling_cleanup(fake_unet)
            assert dynamic_args.moss_ref_kv is None
        finally:
            dit.preprocess_refs = orig_prep


def test_ai_toolkit_reference_mode_guards_missing_vision_weights(engine, monkeypatch):
    from backend.diffusion_engine import moss_image

    monkeypatch.setattr(moss_image, "opts", SimpleNamespace(moss_do_reference="ai-toolkit"))
    engine.forge_objects.clip = SimpleNamespace(patcher=object(), cond_stage_model=SimpleNamespace(qwen35_9b=SimpleNamespace(has_vision=False)))
    engine.ref_latents = ["img"]

    with pytest.raises(ValueError, match="vision weights"):
        engine.get_learned_conditioning(_prompt("edit the hat"))

    # Moss-Image-Edit does not need vision weights and succeeds cleanly
    monkeypatch.setattr(moss_image, "opts", SimpleNamespace(moss_do_reference="Moss-Image-Edit"))
    engine.get_learned_conditioning(_prompt("edit the hat"))
    assert len(engine.text_processing_engine_qwen.calls) == 1

    # clear_references also clears dynamic_args.moss_ref_kv
    from backend.args import dynamic_args

    dynamic_args.moss_ref_kv = "cached"
    engine.clear_references()
    assert dynamic_args.moss_ref_kv is None and dynamic_args.moss_reference_method is None


def test_moss_diffusers_dir_sharded_transformer_and_language_model_prefix_qwen35_9b(tmp_path, monkeypatch):
    import hashlib
    import json
    import os

    import safetensors.torch

    from backend.loader import replace_state_dict, split_state_dict
    from backend.nn.moss_diffusers import (
        diffusers_checkpoint_info,
        is_moss_diffusers_dir,
        is_sharded_moss_transformer_dir,
    )
    from remi import network_metadata as nm

    # Build a lightweight 2-shard OpenMOSS-Team/MOSS-Image-SFT Diffusers directory
    moss_dir = tmp_path / "Stable-diffusion" / "MOSS-Image-SFT"
    (moss_dir / "transformer").mkdir(parents=True)
    (moss_dir / "text_encoder").mkdir(parents=True)
    (moss_dir / "vae").mkdir(parents=True)
    (moss_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MOSSImagePipeline",
                "transformer": ["diffusers", "MOSSImageTransformer"],
                "text_encoder": ["transformers", "Qwen3_5Model"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            }
        ),
        encoding="utf-8",
    )
    (moss_dir / "transformer" / "config.json").write_text(json.dumps({"_class_name": "MOSSImageTransformer"}), encoding="utf-8")
    t_s1 = moss_dir / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    t_s2 = moss_dir / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors"
    safetensors.torch.save_file({"shared_block_modulation.weight": torch.zeros(6 * 256, 4)}, str(t_s1))
    safetensors.torch.save_file(
        {
            "layers.0.proj.ffn_up_gate.weight": torch.zeros(2 * 512, 4),
            "layers.0.proj.q.weight": torch.zeros(256, 256),
            "layers.0.proj.k.weight": torch.zeros(256, 256),
            "layers.0.proj.v.weight": torch.zeros(256, 256),
            "layers.0.proj.o.weight": torch.zeros(256, 256),
        },
        str(t_s2),
    )
    (moss_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "shared_block_modulation.weight": t_s1.name,
                    "layers.0.proj.ffn_up_gate.weight": t_s2.name,
                    "layers.0.proj.q.weight": t_s2.name,
                    "layers.0.proj.k.weight": t_s2.name,
                    "layers.0.proj.v.weight": t_s2.name,
                    "layers.0.proj.o.weight": t_s2.name,
                }
            }
        ),
        encoding="utf-8",
    )

    # Sharded OpenMOSS/Unsloth Qwen3.5-9B using "language_model.*" + "visual.*" (no "model." prefix)
    te_s1 = moss_dir / "text_encoder" / "model-00001-of-00002.safetensors"
    te_s2 = moss_dir / "text_encoder" / "model-00002-of-00002.safetensors"
    safetensors.torch.save_file(
        {
            "language_model.embed_tokens.weight": torch.zeros(8, 4096),
            "language_model.layers.0.linear_attn.A_log": torch.zeros(32),
            "language_model.layers.0.input_layernorm.weight": torch.zeros(4096),
        },
        str(te_s1),
    )
    safetensors.torch.save_file({"visual.merger.linear_fc2.weight": torch.zeros(4096, 16)}, str(te_s2))
    (moss_dir / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.embed_tokens.weight": te_s1.name,
                    "language_model.layers.0.linear_attn.A_log": te_s1.name,
                    "language_model.layers.0.input_layernorm.weight": te_s1.name,
                    "visual.merger.linear_fc2.weight": te_s2.name,
                }
            }
        ),
        encoding="utf-8",
    )

    safetensors.torch.save_file({"decoder.conv_in.weight": torch.zeros(384, 32, 3, 3), "post_quant_conv.weight": torch.zeros(32, 32, 1, 1)}, str(moss_dir / "vae" / "diffusion_pytorch_model.safetensors"))

    assert is_moss_diffusers_dir(moss_dir)
    assert is_sharded_moss_transformer_dir(moss_dir / "transformer")
    arch, primary = diffusers_checkpoint_info(moss_dir)
    assert arch == "moss-image/diffusers" and primary == str(t_s1)

    state_dicts, guess = split_state_dict(str(moss_dir))
    assert "MOSS-Image" in guess.huggingface_repo
    assert guess.unet_config["fused_qkv"] is False
    assert "shared_block_modulation.weight" in state_dicts["transformer"]
    assert "model.language_model.layers.0.linear_attn.A_log" in state_dicts["text_encoder"]
    assert "model.visual.merger.linear_fc2.weight" in state_dicts["text_encoder"]
    assert "decoder.conv_in.weight" in state_dicts["vae"]

    # Standalone sharded transformer folder (e.g. lodestones 8-step DMD) also loads directly
    sharded_sds, sharded_guess = split_state_dict(str(moss_dir / "transformer"))
    assert "MOSS-Image" in sharded_guess.huggingface_repo
    assert "shared_block_modulation.weight" in sharded_sds["transformer"]
    assert sharded_sds["text_encoder"] == {} and sharded_sds["vae"] == {}

    # Standalone Unsloth/OpenMOSS Qwen3.5-9B ("language_model.*" + "visual.*") via replace_state_dict
    sd_override: dict[str, torch.Tensor] = {"text_encoders.qwen35_9b.transformer.old": torch.zeros(1)}
    unsloth_te = {
        "language_model.layers.0.linear_attn.A_log": torch.ones(32),
        "language_model.layers.0.input_layernorm.weight": torch.ones(4096),
        "visual.merger.linear_fc2.weight": torch.ones(4096, 16),
    }
    replace_state_dict(sd_override, unsloth_te, {}, guess, "MOSS-Image-SFT-text_encoder-INT8-ConvRot.safetensors")
    assert "text_encoders.qwen35_9b.transformer.old" not in sd_override
    assert set(sd_override) == {
        "text_encoders.qwen35_9b.transformer.model.language_model.layers.0.linear_attn.A_log",
        "text_encoders.qwen35_9b.transformer.model.language_model.layers.0.input_layernorm.weight",
        "text_encoders.qwen35_9b.transformer.model.visual.merger.linear_fc2.weight",
    }

    # sd_models.list_models lists the Diffusers dir once and classifies it as "moss"
    from remi.testing import webui_state

    with webui_state():
        from modules import sd_models
        from modules.shared import cmd_opts

        ckpt_root = tmp_path / "Stable-diffusion"
        monkeypatch.setattr(sd_models, "model_path", str(ckpt_root))
        monkeypatch.setattr(cmd_opts, "ckpt_dirs", [], raising=False)
        monkeypatch.setattr(sd_models, "checkpoints_list", sd_models.OrderedDict())
        monkeypatch.setattr(sd_models, "checkpoint_aliases", sd_models.OrderedDict())
        monkeypatch.setattr(sd_models.cache, "cached_data_for_file", lambda _sub, _title, _file, fn: fn())
        monkeypatch.setattr(sd_models.hashes, "sha256_from_cache", lambda *_a, **_kw: None)
        monkeypatch.setattr(sd_models.hashes, "sha256", lambda filename, *_a, **_kw: hashlib.sha256(Path(filename).read_bytes()).hexdigest())

        sd_models.list_models()
        listed = {info.name: info for info in sd_models.checkpoints_list.values()}
        assert set(listed) == {"MOSS-Image-SFT"}
        info = listed["MOSS-Image-SFT"]
        assert info.is_diffusers_dir and info.filename == os.path.abspath(str(moss_dir))
        assert nm.detect_arch(info.metadata) == "moss"
        assert info.calculate_shorthash() is not None


@torch.inference_mode()
def test_moss_lora_and_lokr_bidirectional_qkv_and_prefixes():
    from backend.patcher.lora import merge_lora_to_weight
    from modules_forge.packages.comfy.lora import load_lora, model_lora_keys_unet

    def fake_model(fused_qkv: bool):
        dim = 12
        sd = {}
        if fused_qkv:
            sd["diffusion_model.layers.0.proj.qkv.weight"] = torch.zeros(3 * dim, dim)
        else:
            for p in ("q", "k", "v"):
                sd[f"diffusion_model.layers.0.proj.{p}.weight"] = torch.zeros(dim, dim)
        sd["diffusion_model.layers.0.proj.o.weight"] = torch.zeros(dim, dim)
        return SimpleNamespace(
            state_dict=lambda: sd,
            config=SimpleNamespace(huggingface_repo="OpenMOSS-Team/MOSS-Image-SFT"),
            diffusion_model=SimpleNamespace(config={}),
        )

    # 1. Fused proj.qkv LoKr (pmczip / ai-toolkit) onto a SPLIT q/k/v checkpoint via SlicedWeightAdapter
    torch.manual_seed(3)
    w1 = torch.randn(3, 3)
    w2 = torch.randn(12, 4)  # kron(w1, w2) -> [36, 12] = [3*12, 12]
    lokr_sd = {
        "diffusion_model.layers.0.proj.qkv.lokr_w1": w1,
        "diffusion_model.layers.0.proj.qkv.lokr_w2": w2,
        "diffusion_model.layers.0.proj.qkv.alpha": torch.tensor(1.0),
        "transformer.layers.0.proj.o.lora_A.weight": torch.randn(4, 12),
        "transformer.layers.0.proj.o.lora_B.weight": torch.randn(12, 4),
    }
    split_keys = model_lora_keys_unet(fake_model(fused_qkv=False), {})
    patches, remaining = load_lora(lokr_sd, split_keys)
    assert remaining == {}
    expected_full = torch.kron(w1, w2)
    for i, proj in enumerate(("q", "k", "v")):
        key = f"diffusion_model.layers.0.proj.{proj}.weight"
        merged = merge_lora_to_weight([(0.5, patches[key], 1.0, None, None)], torch.zeros(12, 12), key)
        assert torch.allclose(merged, 0.5 * expected_full[i * 12 : (i + 1) * 12], atol=1e-5)

    # 2. Split q/k/v LoRA (bare / base_model.model. / lycoris_ / lora_transformer_) onto a FUSED qkv checkpoint
    a_q = torch.randn(8, 12)
    b_q = torch.randn(12, 8)
    edit_sd = {
        "layers.0.proj.q.lora_A": a_q,
        "layers.0.proj.q.lora_B": b_q,
        "layers.0.proj.q.alpha": torch.tensor(4.0),
        "base_model.model.layers.0.proj.o.lora_A.weight": torch.randn(4, 12),
        "base_model.model.layers.0.proj.o.lora_B.weight": torch.randn(12, 4),
    }
    fused_keys = model_lora_keys_unet(fake_model(fused_qkv=True), {})
    assert "lycoris_layers_0_proj_q" in fused_keys and "lora_transformer_layers_0_proj_o" in fused_keys
    fused_patches, fused_rem = load_lora(edit_sd, fused_keys)
    assert fused_rem == {}
    target_key, offset = next(iter(k for k in fused_patches if isinstance(k, tuple)))
    assert target_key == "diffusion_model.layers.0.proj.qkv.weight" and offset == (0, 0, 12)
    fused_w = merge_lora_to_weight([(1.0, fused_patches[(target_key, offset)], 1.0, offset, None)], torch.zeros(36, 12), target_key)
    assert torch.allclose(fused_w[:12], (4.0 / 8.0) * (b_q @ a_q), atol=1e-5)
    assert not fused_w[12:].any()


def test_unsloth_torchao_qdata_fp8_int8_and_int8_convrot_load_and_run(monkeypatch):
    import json

    import huggingface_guess

    from backend import loader, memory_management, operations
    from backend.args import dynamic_args
    from backend.quant_ops import QuantizedTensor, RowWiseFP8E4M3Layout, TensorWiseINT8Layout
    from backend.state_dict import convert_quantization, detect_quantization

    monkeypatch.setattr(dynamic_args, "ref_latents", [])
    monkeypatch.setattr(dynamic_args, "moss_reference_method", None)
    monkeypatch.setattr(dynamic_args, "moss_ref_kv", None)
    monkeypatch.setattr(dynamic_args, "anima_quant", None)
    monkeypatch.setattr(dynamic_args, "forge_unet_storage_dtype", None)
    monkeypatch.setattr(dynamic_args, "ops", None)

    cpu = torch.device("cpu")
    monkeypatch.setattr(memory_management, "get_torch_device", lambda: cpu)
    monkeypatch.setattr(memory_management, "unet_offload_device", lambda: cpu)
    monkeypatch.setattr(memory_management, "unet_initial_load_device", lambda *_a, **_kw: cpu)
    monkeypatch.setattr(memory_management, "should_use_bf16", lambda *_a, **_kw: True)

    # dim=256 so ConvRot group_size=256 divides every block Linear's in_features cleanly
    ref_dit = _tiny_moss_dit(fused_qkv=False, num_layers=1, dim=256, ffn_dim=256, num_heads=2, text_dim=32, latent_channels=8)
    ref_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in ref_dit.state_dict().items()}
    block_linear_keys = [k for k in ref_sd if k.startswith("layers.0.proj.") and k.endswith(".weight")]
    assert len(block_linear_keys) == 6

    x = torch.randn(1, 8, 2, 2, dtype=torch.bfloat16)
    t = torch.tensor([0.5], dtype=torch.bfloat16)
    ctx = torch.randn(1, 4, 32, dtype=torch.bfloat16)
    with torch.inference_mode():
        ref_out = ref_dit(x.float(), t.float(), ctx.float())

    # 1. Unsloth MOSS-Image-SFT-FP8 (TorchAO Float8Tensor, per-row scale [N, 1])
    fp8_sd = dict(ref_sd)
    for k in block_linear_keys:
        w = fp8_sd.pop(k)
        stem = k[: -len(".weight")]
        qdata, params = RowWiseFP8E4M3Layout.quantize(w)
        fp8_sd[f"{stem}._weight_qdata"] = qdata
        fp8_sd[f"{stem}._weight_scale"] = params.scale
    fp8_meta = {"unsloth_format": "unsloth_prequant_transformer_state_dict_v1", "unsloth_metadata": json.dumps({"scheme": "fp8", "fp8_granularity": "per_row"})}

    fp8_conv, _ = convert_quantization(fp8_sd, fp8_meta)
    fp8_prefixed = {f"model.diffusion_model.{k}": v for k, v in fp8_conv.items()}
    guess = huggingface_guess.guess(fp8_prefixed)
    assert "MOSS-Image" in guess.huggingface_repo and guess.unet_config["fused_qkv"] is False
    assert detect_quantization(fp8_conv, is_unet=True)["layout"] == "RowWiseFP8E4M3Layout"

    guess.unet_config.update({"text_dim": 32, "latent_channels": 8})
    fp8_model = loader.load_huggingface_component(guess, "transformer", "diffusers", "MOSSImageTransformer", "/unused", fp8_conv).eval()
    q_proj = fp8_model.layers[0].proj["q"]
    assert isinstance(q_proj.weight, QuantizedTensor) and q_proj.weight._qdata.dtype == torch.float8_e4m3fn
    assert q_proj.weight._params.scale.shape == (256, 1)
    assert not isinstance(fp8_model.shared_block_modulation.weight, QuantizedTensor)
    with torch.inference_mode():
        fp8_out = fp8_model(x, t, ctx).float()
    assert torch.nn.functional.cosine_similarity(fp8_out.flatten(), ref_out.flatten(), dim=0) > 0.99

    # LoRA merge + requantize round-trip preserves per-row [256, 1] scaling
    deq_w = q_proj.convert_weight(q_proj.weight)
    q_proj.set_weight(deq_w + 0.01)
    assert isinstance(q_proj.weight, QuantizedTensor) and q_proj.weight._params.scale.shape == (256, 1)

    # 2. Unsloth MOSS-Image-SFT-INT8-ConvRot (TorchAO Int8Tensor + Hadamard group-256)
    cr_sd = dict(ref_sd)
    fqns = []
    for k in block_linear_keys:
        w = cr_sd.pop(k)
        stem = k[: -len(".weight")]
        fqns.append(stem)
        qdata, params = TensorWiseINT8Layout.quantize(w.float(), is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
        cr_sd[f"{stem}._weight_qdata"] = qdata
        cr_sd[f"{stem}._weight_scale"] = params.scale.float()
        cr_sd[f"{stem}._weight_zero_point"] = torch.zeros(w.shape[0], 1, dtype=torch.int8)
    cr_meta = {
        "unsloth_format": "unsloth_prequant_transformer_state_dict_v2",
        "unsloth_metadata": json.dumps(
            {
                "scheme": "int8",
                "activation_rotation": "convrot_hadamard_v1",
                "activation_rotation_group": 256,
                "activation_rotation_fqns": fqns,
            }
        ),
    }
    cr_conv, _ = convert_quantization(cr_sd, cr_meta)
    assert detect_quantization(cr_conv, is_unet=True)["layout"] == "TensorWiseINT8Layout"
    cr_model = loader.load_huggingface_component(guess, "transformer", "diffusers", "MOSSImageTransformer", "/unused", cr_conv).eval()
    cr_q = cr_model.layers[0].proj["q"]
    assert isinstance(cr_q.weight, QuantizedTensor) and cr_q.weight._params.convrot is True and cr_q.weight._params.convrot_groupsize == 256
    with torch.inference_mode():
        cr_out = cr_model(x, t, ctx).float()
    assert torch.nn.functional.cosine_similarity(cr_out.flatten(), ref_out.flatten(), dim=0) > 0.99
    # Verify Hadamard rotation is actually active: dequantizing without convrot destroys weight alignment
    w_rot_deq = TensorWiseINT8Layout.dequantize(cr_q.weight._qdata, cr_q.weight._params).float()
    w_unrot_deq = (cr_q.weight._qdata.float() * cr_q.weight._params.scale.float())
    w_ref = ref_sd["layers.0.proj.q.weight"].float()
    assert torch.nn.functional.cosine_similarity(w_rot_deq.flatten(), w_ref.flatten(), dim=0) > 0.999
    assert torch.nn.functional.cosine_similarity(w_unrot_deq.flatten(), w_ref.flatten(), dim=0) < 0.25

    # 3. Asymmetric TorchAO INT8 (non-zero zero_point) safely dequantizes instead of dropping zp
    asym_sd = {
        "layers.0.proj.q._weight_qdata": torch.tensor([[10, -2]], dtype=torch.int8),
        "layers.0.proj.q._weight_scale": torch.tensor([[0.5]], dtype=torch.float32),
        "layers.0.proj.q._weight_zero_point": torch.tensor([[2]], dtype=torch.int8),
    }
    asym_conv, _ = convert_quantization(asym_sd, {})
    assert "layers.0.proj.q.comfy_quant" not in asym_conv
    assert torch.allclose(asym_conv["layers.0.proj.q.weight"].float(), torch.tensor([[4.0, -2.0]]), atol=1e-2)
    del operations


def test_moss_on_the_fly_quant_and_grayscale_reference_expansion(monkeypatch):
    from backend.diffusion_engine.moss_image import MOSSImage as MOSSImageEngine
    from backend.nn.anima_quant import (
        keep_sensitive_weights_precise,
        scale_fp8_state_dict,
        should_keep_sensitive_weights_precise,
        should_quantize_on_load,
    )
    from backend.operations import using_forge_operations
    from modules import shared

    monkeypatch.setattr(shared, "opts", SimpleNamespace(reference_max_side=1024))

    # 1. On-the-fly FP8 keeps MOSS-Image embedders, shared AdaLN, head & norms precise
    assert should_keep_sensitive_weights_precise("MOSSImageTransformer", torch.float8_e4m3fn, torch.bfloat16) is True
    assert should_quantize_on_load("scaled_fp8", "MOSSImageTransformer", None, False, torch.bfloat16) is True
    assert should_quantize_on_load("int8_convrot", "MOSSImageTransformer", None, False, torch.bfloat16) is True

    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float8_e4m3fn, manual_cast_enabled=True):
        from backend.nn.moss_image import MOSSImage as MOSSDiT

        fp8_dit = MOSSDiT(dim=32, ffn_dim=64, num_heads=2, num_layers=1, text_dim=16, latent_channels=8)
    kept = keep_sensitive_weights_precise(fp8_dit, torch.bfloat16)
    assert kept > 0
    assert fp8_dit.video_embedder.weight.dtype == torch.bfloat16
    assert fp8_dit.text_embedder.proj.weight.dtype == torch.bfloat16
    assert fp8_dit.shared_block_modulation.weight.dtype == torch.bfloat16
    assert fp8_dit.final_linear_video.weight.dtype == torch.bfloat16
    assert fp8_dit.layers[0].proj["q"].weight.dtype == torch.float8_e4m3fn

    # 2. On-the-fly scaled FP8 uses per-row float8_e4m3fn_rowwise on MOSS-Image layers
    bf16_dit = _tiny_moss_dit(num_layers=1, dim=32, ffn_dim=64, num_heads=2, text_dim=16, latent_channels=8)
    sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in bf16_dit.state_dict().items()}
    count = scale_fp8_state_dict(sd, fp8_matmul=False, device=torch.device("cpu"))
    assert count == 6
    assert sd["layers.0.proj.q.weight"].dtype == torch.float8_e4m3fn
    assert sd["layers.0.proj.q.weight_scale"].shape == (32, 1)
    assert "shared_block_modulation.comfy_quant" not in sd

    # 3. encode_vision safely expands 1-channel grayscale reference images to 3 channels
    seen_vae_shapes = []
    eng = MOSSImageEngine.__new__(MOSSImageEngine)
    eng.forge_objects = SimpleNamespace(
        vae=SimpleNamespace(
            encode=lambda img: seen_vae_shapes.append(tuple(img.shape)) or torch.zeros(1, 32, 2, 2),
            first_stage_model=SimpleNamespace(process_in=lambda x: x),
        )
    )
    eng.model_config = SimpleNamespace(latent_format=SimpleNamespace(latent_channels=128))
    vision, latent = eng.encode_vision(torch.zeros(1, 32, 32, 1))
    assert vision.shape[-1] == 3 and seen_vae_shapes[0][-1] == 3 and latent.shape[1] == 128
