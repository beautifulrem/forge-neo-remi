"""Anima Diffusers format (``circlestone-labs/Anima-Base-v1.0-Diffusers``) and
sharded HuggingFace safetensors loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

import safetensors.torch
import torch

from backend.loader import _additional_key_filter, replace_state_dict, split_state_dict
from backend.nn.anima import Anima
from backend.nn.anima_diffusers import (
    anima_diffusers_primary_file,
    convert_diffusers_anima_transformer,
    convert_diffusers_wan_vae,
    find_sharded_index,
    is_anima_diffusers_dir,
    is_diffusers_anima_text_conditioner,
    is_diffusers_anima_transformer,
    is_diffusers_wan_vae,
    is_sharded_hf_dir,
    load_sharded_safetensors,
)
from backend.nn.wan_vae import WanVAE
from backend.operations import using_forge_operations
from backend.utils import load_torch_file
from modules_forge.packages.comfy.utils import convert_diffusers_mmdit
from remi import network_metadata as nm

REPO = Path(__file__).resolve().parents[1]
ANIMA_VAE_CONFIG = REPO / "backend" / "huggingface" / "circlestone-labs" / "Anima" / "vae"


def _native_to_diffusers_anima_key(key: str) -> str:
    basic = (
        ("x_embedder.proj.1.", "patch_embed.proj."),
        ("t_embedder.1.linear_1.", "time_embed.t_embedder.linear_1."),
        ("t_embedder.1.linear_2.", "time_embed.t_embedder.linear_2."),
        ("t_embedding_norm.", "time_embed.norm."),
        ("final_layer.adaln_modulation.1.", "norm_out.linear_1."),
        ("final_layer.adaln_modulation.2.", "norm_out.linear_2."),
        ("final_layer.linear.", "proj_out."),
    )
    block = (
        ("adaln_modulation_self_attn.1.", "norm1.linear_1."),
        ("adaln_modulation_self_attn.2.", "norm1.linear_2."),
        ("adaln_modulation_cross_attn.1.", "norm2.linear_1."),
        ("adaln_modulation_cross_attn.2.", "norm2.linear_2."),
        ("adaln_modulation_mlp.1.", "norm3.linear_1."),
        ("adaln_modulation_mlp.2.", "norm3.linear_2."),
        ("self_attn.q_proj.", "attn1.to_q."),
        ("self_attn.k_proj.", "attn1.to_k."),
        ("self_attn.v_proj.", "attn1.to_v."),
        ("self_attn.output_proj.", "attn1.to_out.0."),
        ("self_attn.q_norm.", "attn1.norm_q."),
        ("self_attn.k_norm.", "attn1.norm_k."),
        ("cross_attn.q_proj.", "attn2.to_q."),
        ("cross_attn.k_proj.", "attn2.to_k."),
        ("cross_attn.v_proj.", "attn2.to_v."),
        ("cross_attn.output_proj.", "attn2.to_out.0."),
        ("cross_attn.q_norm.", "attn2.norm_q."),
        ("cross_attn.k_norm.", "attn2.norm_k."),
        ("mlp.layer1.", "ff.net.0.proj."),
        ("mlp.layer2.", "ff.net.2."),
    )
    if key.startswith("blocks."):
        _, idx, rest = key.split(".", 2)
        for src, dst in block:
            if rest.startswith(src):
                return f"transformer_blocks.{idx}.{dst}{rest[len(src) :]}"
    for src, dst in basic:
        if key.startswith(src):
            return f"{dst}{key[len(src) :]}"
    return key


def _native_to_diffusers_wan_vae_key(key: str) -> str:
    resnet = (
        ("residual.0.", "norm1."),
        ("residual.2.", "conv1."),
        ("residual.3.", "norm2."),
        ("residual.6.", "conv2."),
        ("shortcut.", "conv_shortcut."),
    )

    def map_res(rest: str) -> str:
        for src, dst in resnet:
            if rest.startswith(src):
                return dst + rest[len(src) :]
        return rest

    if key.startswith("conv1."):
        return "quant_conv." + key[len("conv1.") :]
    if key.startswith("conv2."):
        return "post_quant_conv." + key[len("conv2.") :]
    side, rest = key.split(".", 1)
    if rest.startswith("conv1."):
        return f"{side}.conv_in.{rest[len('conv1.') :]}"
    if rest.startswith("head.0."):
        return f"{side}.norm_out.{rest[len('head.0.') :]}"
    if rest.startswith("head.2."):
        return f"{side}.conv_out.{rest[len('head.2.') :]}"
    if rest.startswith("middle.0."):
        return f"{side}.mid_block.resnets.0.{map_res(rest[len('middle.0.') :])}"
    if rest.startswith("middle.1."):
        return f"{side}.mid_block.attentions.0.{rest[len('middle.1.') :]}"
    if rest.startswith("middle.2."):
        return f"{side}.mid_block.resnets.1.{map_res(rest[len('middle.2.') :])}"
    if side == "encoder" and rest.startswith("downsamples."):
        _, idx, sub = rest.split(".", 2)
        return f"encoder.down_blocks.{idx}.{map_res(sub)}"
    if side == "decoder" and rest.startswith("upsamples."):
        _, idx_str, sub = rest.split(".", 2)
        idx = int(idx_str)
        stage, pos = divmod(idx, 4)
        if pos == 3:
            return f"decoder.up_blocks.{stage}.upsamplers.0.{sub}"
        return f"decoder.up_blocks.{stage}.resnets.{pos}.{map_res(sub)}"
    return key


def test_diffusers_anima_transformer_matches_native_28_block_dit_exactly():
    with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
        native = Anima(
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=2048,
            num_blocks=28,
            num_heads=16,
            crossattn_emb_channels=1024,
            adaln_lora_dim=256,
            concat_padding_mask=True,
        )
    native_sd = {k: v for k, v in native.state_dict().items() if not k.startswith("llm_adapter.")}
    assert len(native_sd) == 567

    diffusers_sd = {_native_to_diffusers_anima_key(k): v for k, v in native_sd.items()}
    assert is_diffusers_anima_transformer(diffusers_sd)
    assert not is_diffusers_anima_transformer(native_sd)

    converted = convert_diffusers_anima_transformer(diffusers_sd)
    assert {k: list(v.shape) for k, v in converted.items()} == {k: list(v.shape) for k, v in native_sd.items()}

    # Also wired through convert_diffusers_mmdit fallback
    via_mmdit = convert_diffusers_mmdit(diffusers_sd, output_prefix="model.diffusion_model.")
    assert via_mmdit is not None
    assert set(via_mmdit) == {f"model.diffusion_model.{k}" for k in native_sd}


def test_diffusers_wan_vae_bijects_all_194_keys_with_native_wan_vae():
    config = WanVAE.load_config(str(ANIMA_VAE_CONFIG))
    config["image_channels"] = 3
    config["conv_out_channels"] = 3
    with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
        native_vae = WanVAE.from_config(config)
    native_sd = native_vae.state_dict()
    assert len(native_sd) == 194

    diffusers_vae = {_native_to_diffusers_wan_vae_key(k): v for k, v in native_sd.items()}
    assert len(diffusers_vae) == 194
    assert is_diffusers_wan_vae(diffusers_vae)
    assert not is_diffusers_wan_vae(native_sd)

    converted = convert_diffusers_wan_vae(diffusers_vae)
    assert {k: list(v.shape) for k, v in converted.items()} == {k: list(v.shape) for k, v in native_sd.items()}


def _build_fake_anima_diffusers_dir(root: Path) -> Path:
    """Create a lightweight on-disk Anima-Base-v1.0-Diffusers directory."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "AnimaModularPipeline",
                "transformer": ["diffusers", "CosmosTransformer3DModel"],
                "text_conditioner": ["diffusers", "AnimaTextConditioner"],
                "text_encoder": ["transformers", "Qwen3Model"],
                "vae": ["diffusers", "AutoencoderKLWan"],
            }
        ),
        encoding="utf-8",
    )

    # Minimal 1-block Anima Diffusers transformer with real Anima detection shapes
    transformer_dir = root / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(json.dumps({"_class_name": "CosmosTransformer3DModel"}), encoding="utf-8")
    transformer_tensors = {
        "patch_embed.proj.weight": torch.zeros(2048, 68, dtype=torch.bfloat16),
        "time_embed.t_embedder.linear_1.weight": torch.zeros(2048, 2048, dtype=torch.bfloat16),
        "time_embed.t_embedder.linear_2.weight": torch.zeros(6144, 2048, dtype=torch.bfloat16),
        "time_embed.norm.weight": torch.zeros(2048, dtype=torch.bfloat16),
        "norm_out.linear_1.weight": torch.zeros(256, 2048, dtype=torch.bfloat16),
        "norm_out.linear_2.weight": torch.zeros(4096, 256, dtype=torch.bfloat16),
        "proj_out.weight": torch.zeros(64, 2048, dtype=torch.bfloat16),
        "transformer_blocks.0.norm1.linear_1.weight": torch.zeros(256, 2048, dtype=torch.bfloat16),
        "transformer_blocks.0.norm1.linear_2.weight": torch.zeros(6144, 256, dtype=torch.bfloat16),
        "transformer_blocks.0.attn1.to_q.weight": torch.zeros(2048, 2048, dtype=torch.bfloat16),
        "transformer_blocks.0.ff.net.0.proj.weight": torch.zeros(8192, 2048, dtype=torch.bfloat16),
        "transformer_blocks.0.ff.net.2.weight": torch.zeros(2048, 8192, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(transformer_tensors, str(transformer_dir / "diffusion_pytorch_model.safetensors"))

    # Diffusers AnimaTextConditioner (exact LLMAdapter layout)
    cond_dir = root / "text_conditioner"
    cond_dir.mkdir()
    cond_tensors = {
        "embed.weight": torch.zeros(4, 1024, dtype=torch.bfloat16),
        "out_proj.weight": torch.zeros(1024, 1024, dtype=torch.bfloat16),
        "blocks.0.cross_attn.q_proj.weight": torch.zeros(1024, 1024, dtype=torch.bfloat16),
        "blocks.0.self_attn.q_proj.weight": torch.zeros(1024, 1024, dtype=torch.bfloat16),
    }
    assert is_diffusers_anima_text_conditioner(cond_tensors)
    safetensors.torch.save_file(cond_tensors, str(cond_dir / "diffusion_pytorch_model.safetensors"))

    # Prefixless Diffusers Qwen3-0.6B text_encoder
    te_dir = root / "text_encoder"
    te_dir.mkdir()
    te_tensors = {
        "embed_tokens.weight": torch.zeros(8, 1024, dtype=torch.bfloat16),
        "layers.0.post_attention_layernorm.weight": torch.zeros(1024, dtype=torch.bfloat16),
        "layers.0.self_attn.q_norm.weight": torch.zeros(128, dtype=torch.bfloat16),
        "norm.weight": torch.zeros(1024, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(te_tensors, str(te_dir / "model.safetensors"))

    # Diffusers Wan 3D VAE
    vae_dir = root / "vae"
    vae_dir.mkdir()
    vae_tensors = {
        "quant_conv.weight": torch.zeros(32, 32, 1, 1, 1, dtype=torch.bfloat16),
        "post_quant_conv.weight": torch.zeros(16, 16, 1, 1, 1, dtype=torch.bfloat16),
        "encoder.conv_in.weight": torch.zeros(96, 3, 3, 3, 3, dtype=torch.bfloat16),
        "decoder.mid_block.resnets.0.norm1.gamma": torch.zeros(384, 1, 1, 1, dtype=torch.bfloat16),
        "decoder.up_blocks.0.resnets.0.conv1.weight": torch.zeros(384, 384, 3, 3, 3, dtype=torch.bfloat16),
        "decoder.norm_out.gamma": torch.zeros(96, 1, 1, 1, dtype=torch.bfloat16),
        "decoder.conv_out.weight": torch.zeros(3, 96, 3, 3, 3, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(vae_tensors, str(vae_dir / "diffusion_pytorch_model.safetensors"))
    return root


def test_split_state_dict_loads_full_diffusers_dir_and_standalone_transformer(tmp_path):
    diffusers_dir = _build_fake_anima_diffusers_dir(tmp_path / "Anima-Base-v1.0-Diffusers")
    assert is_anima_diffusers_dir(diffusers_dir)
    assert anima_diffusers_primary_file(diffusers_dir) == str(diffusers_dir / "transformer" / "diffusion_pytorch_model.safetensors")

    # 1. Complete Diffusers folder
    state_dicts, guess = split_state_dict(str(diffusers_dir))
    assert "Anima" in guess.huggingface_repo
    assert "x_embedder.proj.1.weight" in state_dicts["transformer"]
    assert "blocks.0.mlp.layer1.weight" in state_dicts["transformer"]
    # process_anima moved llm_adapter from transformer -> text_encoder
    assert not any(k.startswith("llm_adapter.") for k in state_dicts["transformer"])
    assert "llm_adapter.embed.weight" in state_dicts["text_encoder"]
    assert "model.embed_tokens.weight" in state_dicts["text_encoder"]
    assert "decoder.middle.0.residual.0.gamma" in state_dicts["vae"]
    assert "conv1.weight" in state_dicts["vae"]

    # 2. Override VAE via additional_state_dicts replaces bundled VAE
    override_vae = tmp_path / "override_vae.safetensors"
    safetensors.torch.save_file(
        {
            "decoder.middle.0.residual.0.gamma": torch.ones(384, 1, 1, 1, dtype=torch.bfloat16),
            "decoder.head.2.weight": torch.ones(3, 96, 3, 3, 3, dtype=torch.bfloat16),
        },
        str(override_vae),
    )
    state_dicts_override, _ = split_state_dict(str(diffusers_dir), additional_state_dicts=[str(override_vae)])
    assert float(state_dicts_override["vae"]["decoder.middle.0.residual.0.gamma"].sum()) > 0
    assert "conv1.weight" not in state_dicts_override["vae"]

    # 3. Standalone Diffusers transformer file auto-discovers sibling text_conditioner
    standalone = diffusers_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    state_dicts_standalone, guess_standalone = split_state_dict(
        str(standalone),
        additional_state_dicts=[
            str(diffusers_dir / "text_encoder" / "model.safetensors"),
            str(diffusers_dir / "vae" / "diffusion_pytorch_model.safetensors"),
        ],
    )
    assert "Anima" in guess_standalone.huggingface_repo
    assert "llm_adapter.embed.weight" in state_dicts_standalone["text_encoder"]
    assert "model.embed_tokens.weight" in state_dicts_standalone["text_encoder"]
    assert "decoder.middle.0.residual.0.gamma" in state_dicts_standalone["vae"]

    # 4. Standalone text_conditioner passed via replace_state_dict populates llm_adapter
    sd_target: dict[str, torch.Tensor] = {}
    cond_sd = load_torch_file(str(diffusers_dir / "text_conditioner" / "diffusion_pytorch_model.safetensors"))
    replace_state_dict(sd_target, cond_sd, {}, guess, str(diffusers_dir / "text_conditioner" / "diffusion_pytorch_model.safetensors"))
    assert "model.diffusion_model.llm_adapter.embed.weight" in sd_target


def test_sd_models_lists_diffusers_dir_once_and_hashes_cleanly(tmp_path, monkeypatch):
    import hashlib

    from remi.testing import webui_state

    with webui_state():
        from modules import sd_models
        from modules.shared import cmd_opts

        ckpt_root = tmp_path / "Stable-diffusion"
        ckpt_root.mkdir()
        diffusers_dir = _build_fake_anima_diffusers_dir(ckpt_root / "Anima-Base-v1.0-Diffusers")
        regular = ckpt_root / "plain_ckpt.safetensors"
        safetensors.torch.save_file({"weight": torch.zeros(2)}, str(regular))

        monkeypatch.setattr(sd_models, "model_path", str(ckpt_root))
        monkeypatch.setattr(cmd_opts, "ckpt_dirs", [], raising=False)
        monkeypatch.setattr(sd_models, "checkpoints_list", sd_models.OrderedDict())
        monkeypatch.setattr(sd_models, "checkpoint_aliases", sd_models.OrderedDict())
        monkeypatch.setattr(sd_models.cache, "cached_data_for_file", lambda _sub, _title, _file, fn: fn())
        monkeypatch.setattr(sd_models.hashes, "sha256_from_cache", lambda *_a, **_kw: None)
        monkeypatch.setattr(sd_models.hashes, "sha256", lambda filename, *_a, **_kw: hashlib.sha256(Path(filename).read_bytes()).hexdigest())

        sd_models.list_models()
        listed = {info.name: info for info in sd_models.checkpoints_list.values()}
        assert set(listed) == {"Anima-Base-v1.0-Diffusers", "plain_ckpt.safetensors"}

        info = listed["Anima-Base-v1.0-Diffusers"]
        assert info.is_diffusers_dir is True
        assert info.filename == os.path.abspath(str(diffusers_dir))
        assert info.name_for_extra == "Anima-Base-v1.0-Diffusers"
        assert info.metadata.get("modelspec.architecture") == "anima/diffusers"
        assert nm.detect_arch(info.metadata) == "anima"
        assert info.hash != "NOFILE"
        assert info.calculate_shorthash() is not None


def test_sharded_safetensors_loading_skips_filtered_out_shards(tmp_path):
    from backend.nn.anima_qwen35 import qwen35_4b_needed_key

    model_dir = tmp_path / "Qwen3.5-4B"
    model_dir.mkdir()
    shard1 = model_dir / "model-00001-of-00003.safetensors"
    shard2 = model_dir / "model-00002-of-00003.safetensors"
    # Intentionally do NOT create shard3 (vision + MTP only) to prove key_filter never opens it!
    safetensors.torch.save_file(
        {
            "model.language_model.embed_tokens.weight": torch.ones(4, 8),
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4),
        },
        str(shard1),
        metadata={"format": "pt"},
    )
    safetensors.torch.save_file(
        {
            "model.language_model.layers.31.self_attn.q_proj.weight": torch.ones(8, 8),
            "model.language_model.layers.31.mlp.gate_proj.weight": torch.ones(16, 8),
            "model.language_model.norm.weight": torch.ones(8),
        },
        str(shard2),
    )
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.embed_tokens.weight": shard1.name,
                    "model.language_model.layers.0.linear_attn.A_log": shard1.name,
                    "model.language_model.layers.31.self_attn.q_proj.weight": shard2.name,
                    "model.language_model.layers.31.mlp.gate_proj.weight": shard2.name,
                    "model.language_model.norm.weight": shard2.name,
                    "model.visual.blocks.0.attn.qkv.weight": "model-00003-of-00003.safetensors",
                    "mtp.layers.0.mlp.gate_proj.weight": "model-00003-of-00003.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert is_sharded_hf_dir(model_dir)
    assert find_sharded_index(model_dir) == str(index_path)
    assert find_sharded_index(shard1) == str(index_path)

    # Filtered load succeeds without touching missing shard3 and drops layer-31 MLP / norm / visual / mtp
    sd, meta = load_sharded_safetensors(index_path, key_filter=qwen35_4b_needed_key)
    assert set(sd) == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.31.self_attn.q_proj.weight",
    }
    assert meta == {"format": "pt"}

    # load_torch_file accepts folder, index file, or shard 1 directly
    sd_via_dir = load_torch_file(str(model_dir), key_filter=qwen35_4b_needed_key)
    assert set(sd_via_dir) == set(sd)

    # _additional_key_filter activates for Anima only, leaving MOSS-Image's vision shards intact
    anima_guess = type("G", (), {"huggingface_repo": "circlestone-labs/Anima"})()
    moss_guess = type("G", (), {"huggingface_repo": "OpenMOSS-Team/MOSS-Image"})()
    assert _additional_key_filter(str(model_dir), anima_guess) is qwen35_4b_needed_key
    assert _additional_key_filter(str(model_dir), moss_guess) is None
