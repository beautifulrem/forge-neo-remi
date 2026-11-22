"""HuggingFace Diffusers, Unsloth, VAE, and LoRA support for Qwen-Image-2.1.

Handles:
- complete ``Qwen/Qwen-Image-2.1``, ``Qwen/Qwen-Image-2.1-Turbo``, and
  ``unsloth/Qwen-Image-2.1-Turbo-FP8`` Diffusers directories
  (``model_index.json`` + sharded ``transformer/`` + sharded ``text_encoder/``
  + ``vae/``);
- standalone sharded Qwen-Image-2.1 ``transformer/`` folders;
- exact conversion of Diffusers ``AutoencoderKLQwenImage21`` (238 tensors,
  4-channel RGBA, 4D spatial conv weights) to Forge's native ``Wan22VAE``
  (5D ``CausalConv3d`` weights with ``temporal_kernel=1``);
- bidirectional LoRA key mapping across fused (``img_mlp.gate_up``) and
  unfused (``img_mlp.gate_layer`` + ``img_mlp.proj``) checkpoints for all
  common trainers (ComfyUI, Diffusers, PEFT, ai-toolkit, kohya, LyCORIS,
  OneTrainer).
"""

from __future__ import annotations

import json
import os

import torch

from backend.nn.anima_diffusers import (
    _DIFFUSERS_INDEX_FILES,
    _SHARD_INDEX_FILES,
    _WAN_RESNET_MAP,
    _load_component_sd,
    _read_json,
)

DIFFUSERS_CLASS = "QwenImage21Transformer2DModel"
DIFFUSERS_PIPELINE_CLASS = "QwenImage21Pipeline"

_CAUSAL_CONV_SUFFIXES = (
    ".conv1.weight",
    ".conv2.weight",
    ".conv_shortcut.weight",
    ".time_conv.weight",
    "quant_conv.weight",
    "post_quant_conv.weight",
    "conv_in.weight",
    "conv_out.weight",
)


def _dir_primary(directory: str) -> str | None:
    from backend.nn.moss_diffusers import _dir_primary_safetensors

    return _dir_primary_safetensors(directory)


# region Diffusers AutoencoderKLQwenImage21 VAE conversion


def is_diffusers_qwen21_vae(state_dict: dict) -> bool:
    """True for Diffusers ``AutoencoderKLQwenImage21`` (4-channel RGBA, 64-latent)
    state dicts (``Qwen/Qwen-Image-2.1/vae`` and ``unsloth/Qwen-Image-2.1-*/vae``).

    Distinguished from Wan 2.1 ``AutoencoderKLWan`` by nested
    ``encoder.down_blocks.0.resnets.*`` / singular ``upsampler.*`` containers.
    """

    if not ("decoder.mid_block.resnets.0.norm1.gamma" in state_dict and "decoder.norm_out.gamma" in state_dict):
        return False
    return (
        "encoder.down_blocks.0.resnets.0.conv1.weight" in state_dict
        or "decoder.up_blocks.0.upsampler.resample.1.weight" in state_dict
    )


def _map_qwen21_resnet(rest: str) -> str:
    for src, dst in _WAN_RESNET_MAP:
        if rest.startswith(src):
            return dst + rest[len(src) :]
    return rest


def _maybe_unsqueeze_causal_conv(raw_key: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4 and raw_key.endswith(_CAUSAL_CONV_SUFFIXES):
        return value.unsqueeze(2)
    return value


def convert_diffusers_qwen21_vae(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert Diffusers ``AutoencoderKLQwenImage21`` keys (238 tensors) into
    Forge's native ``Wan22VAE`` layout (bit-exact)."""

    # Count resnet blocks per stage so the trailing downsampler / upsampler
    # lands at the exact Sequential slot regardless of num_res_blocks.
    enc_resnets: dict[int, int] = {}
    dec_resnets: dict[int, int] = {}
    for key in state_dict:
        if key.startswith("encoder.down_blocks.") and ".resnets." in key:
            parts = key.split(".")
            if len(parts) >= 5 and parts[2].isdigit() and parts[4].isdigit():
                stage, r_idx = int(parts[2]), int(parts[4])
                enc_resnets[stage] = max(enc_resnets.get(stage, 0), r_idx + 1)
        elif key.startswith("decoder.up_blocks.") and ".resnets." in key:
            parts = key.split(".")
            if len(parts) >= 5 and parts[2].isdigit() and parts[4].isdigit():
                stage, r_idx = int(parts[2]), int(parts[4])
                dec_resnets[stage] = max(dec_resnets.get(stage, 0), r_idx + 1)

    out: dict[str, torch.Tensor] = {}
    for key, raw_value in state_dict.items():
        value = _maybe_unsqueeze_causal_conv(key, raw_value)
        new_key = key

        if key.startswith("quant_conv."):
            new_key = "conv1." + key[len("quant_conv.") :]
        elif key.startswith("post_quant_conv."):
            new_key = "conv2." + key[len("post_quant_conv.") :]
        elif key.startswith(("encoder.", "decoder.")):
            side, rest = key.split(".", 1)
            if rest.startswith("conv_in."):
                new_key = f"{side}.conv1.{rest[len('conv_in.') :]}"
            elif rest.startswith("norm_out."):
                new_key = f"{side}.head.0.{rest[len('norm_out.') :]}"
            elif rest.startswith("conv_out."):
                new_key = f"{side}.head.2.{rest[len('conv_out.') :]}"
            elif rest.startswith("mid_block."):
                mid = rest[len("mid_block.") :]
                if mid.startswith("resnets.0."):
                    new_key = f"{side}.middle.0.{_map_qwen21_resnet(mid[len('resnets.0.') :])}"
                elif mid.startswith("attentions.0."):
                    new_key = f"{side}.middle.1.{mid[len('attentions.0.') :]}"
                elif mid.startswith("resnets.1."):
                    new_key = f"{side}.middle.2.{_map_qwen21_resnet(mid[len('resnets.1.') :])}"
            elif side == "encoder" and rest.startswith("down_blocks."):
                parts = rest.split(".", 3)
                if len(parts) == 4 and parts[1].isdigit():
                    stage = int(parts[1])
                    kind, tail = parts[2], parts[3]
                    if kind == "resnets" and "." in tail:
                        r_idx, sub = tail.split(".", 1)
                        if r_idx.isdigit():
                            new_key = f"encoder.downsamples.{stage}.downsamples.{r_idx}.{_map_qwen21_resnet(sub)}"
                    elif kind == "downsampler":
                        slot = enc_resnets.get(stage, 2)
                        new_key = f"encoder.downsamples.{stage}.downsamples.{slot}.{tail}"
            elif side == "decoder" and rest.startswith("up_blocks."):
                parts = rest.split(".", 3)
                if len(parts) == 4 and parts[1].isdigit():
                    stage = int(parts[1])
                    kind, tail = parts[2], parts[3]
                    if kind == "resnets" and "." in tail:
                        r_idx, sub = tail.split(".", 1)
                        if r_idx.isdigit():
                            new_key = f"decoder.upsamples.{stage}.upsamples.{r_idx}.{_map_qwen21_resnet(sub)}"
                    elif kind == "upsampler":
                        slot = dec_resnets.get(stage, 3)
                        new_key = f"decoder.upsamples.{stage}.upsamples.{slot}.{tail}"

        out[new_key] = value
    return out


# endregion


# region Diffusers & sharded directories


def qwen21_diffusers_primary_file(directory: os.PathLike | str) -> str | None:
    """A concrete safetensors file inside a Qwen-Image-2.1 Diffusers or sharded folder."""

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    if os.path.isdir(transformer_dir) and (primary := _dir_primary(transformer_dir)):
        return primary
    return _dir_primary(root)


def is_qwen21_diffusers_dir(path: os.PathLike | str) -> bool:
    """True for a local Diffusers pipeline folder for Qwen-Image-2.1 (Base or Turbo)."""

    root = os.fspath(path)
    transformer_dir = os.path.join(root, "transformer")
    if not os.path.isdir(root) or not os.path.isdir(transformer_dir) or _dir_primary(transformer_dir) is None:
        return False
    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(root, index_name))
        if not index:
            continue
        cls_name = str(index.get("_class_name", ""))
        entry = index.get("transformer")
        if DIFFUSERS_PIPELINE_CLASS in cls_name:
            return True
        if isinstance(entry, list) and DIFFUSERS_CLASS in entry:
            return True
    config = _read_json(os.path.join(transformer_dir, "config.json"))
    return bool(config and config.get("_class_name") == DIFFUSERS_CLASS)


def is_sharded_qwen21_transformer_dir(path: os.PathLike | str) -> bool:
    """True for a standalone (sharded) Qwen-Image-2.1 transformer folder."""

    root = os.fspath(path)
    if not os.path.isdir(root) or os.path.isdir(os.path.join(root, "transformer")) or _dir_primary(root) is None:
        return False
    config = _read_json(os.path.join(root, "config.json"))
    if config and config.get("_class_name") == DIFFUSERS_CLASS:
        return True
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(root, name))
        weight_map = index.get("weight_map") if index else None
        if not isinstance(weight_map, dict):
            continue
        if (
            "txt_in.text_norm.weight" in weight_map
            and "modulation.1.weight" in weight_map
            and ("transformer_blocks.0.img_mlp.proj.weight" in weight_map or "transformer_blocks.0.img_mlp.gate_up.weight" in weight_map)
        ):
            return True
    return False


def is_qwen21_checkpoint_dir(path: os.PathLike | str) -> bool:
    return is_qwen21_diffusers_dir(path) or is_sharded_qwen21_transformer_dir(path)


def load_qwen21_diffusers_dir(directory: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a ``Qwen/Qwen-Image-2.1*`` / ``unsloth/Qwen-Image-2.1*`` Diffusers
    folder or standalone sharded transformer folder into Forge's unified
    checkpoint state dict."""

    from backend.nn.krea_diffusers import normalize_qwen3vl_key

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    is_pipeline = os.path.isdir(transformer_dir)
    if not is_pipeline:
        transformer_dir = root

    raw, metadata = _load_component_sd(transformer_dir)
    if not raw:
        raise FileNotFoundError(f"Qwen-Image-2.1 Diffusers directory has no transformer weights: {transformer_dir}")

    sd: dict[str, torch.Tensor] = {
        (k if k.startswith("model.diffusion_model.") else f"model.diffusion_model.{k}"): v
        for k, v in raw.items()
    }
    del raw

    if not is_pipeline:
        return sd, metadata

    metadata = dict(metadata or {})
    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(root, index_name))
        if index and isinstance(index.get("sample_sigmas"), list) and index["sample_sigmas"]:
            metadata["qwen21_sample_sigmas"] = json.dumps(index["sample_sigmas"])
            break

    sched_cfg = _read_json(os.path.join(root, "scheduler", "scheduler_config.json"))
    if sched_cfg and "use_dynamic_shifting" in sched_cfg:
        metadata["qwen21_dynamic_shifting"] = str(bool(sched_cfg["use_dynamic_shifting"])).lower()

    te_dir = os.path.join(root, "text_encoder")
    if os.path.isdir(te_dir):
        te_sd, _ = _load_component_sd(te_dir)
        for key, value in te_sd.items():
            sd[f"text_encoders.qwen3vl_8b.transformer.{normalize_qwen3vl_key(key)}"] = value
        del te_sd

    vae_dir = os.path.join(root, "vae")
    if os.path.isdir(vae_dir):
        vae_sd, _ = _load_component_sd(vae_dir)
        if vae_sd:
            if is_diffusers_qwen21_vae(vae_sd):
                vae_sd = convert_diffusers_qwen21_vae(vae_sd)
            for key, value in vae_sd.items():
                sd[f"vae.{key}"] = value
        del vae_sd

    return sd, metadata


# endregion


# region LoRA key mapping


def qwen21_lora_key_map(sd: dict[str, torch.Tensor]) -> dict[str, object]:
    """Map Qwen-Image-2.1 LoRA keys from every common trainer (ComfyUI,
    Diffusers, PEFT, ai-toolkit, kohya, LyCORIS, OneTrainer) onto the loaded
    ``diffusion_model.*`` weights — bidirectionally bridging fused
    ``img_mlp.gate_up`` and unfused ``img_mlp.gate_layer`` + ``img_mlp.proj``."""

    from modules_forge.packages.comfy.lora import SplitTargets

    key_map: dict[str, object] = {}

    def register(bare: str, target) -> None:
        flat = bare.replace(".", "_")
        for prefix in ("", "diffusion_model.", "transformer.", "base_model.model.", "base_model.model.diffusion_model.", "base_model.model.transformer."):
            key_map[f"{prefix}{bare}"] = target
        key_map[f"lora_unet_{flat}"] = target
        key_map[f"lycoris_{flat}"] = target
        key_map[f"lora_transformer_{flat}"] = target

    for k in sd:
        if not (k.startswith("diffusion_model.") and k.endswith(".weight")):
            continue
        bare = k[len("diffusion_model.") : -len(".weight")]
        register(bare, k)

        if bare.endswith(".img_mlp.gate_up"):
            # Fused checkpoint: accept both fused ``gate_up`` LoRAs (registered
            # above) and unfused Diffusers ``gate_layer`` / ``proj`` LoRAs.
            half = int(sd[k].shape[0]) // 2
            stem = bare[: -len(".gate_up")]
            register(f"{stem}.gate_layer", (k, (0, 0, half)))
            register(f"{stem}.proj", (k, (0, half, half)))
        elif bare.endswith(".img_mlp.gate_layer"):
            # Unfused Diffusers / Unsloth checkpoint: accept fused ComfyUI /
            # ai-toolkit ``img_mlp.gate_up`` LoRAs via SplitTargets.
            proj_key = k[: -len(".gate_layer.weight")] + ".proj.weight"
            if proj_key in sd:
                g_dim = int(sd[k].shape[0])
                p_dim = int(sd[proj_key].shape[0])
                stem = bare[: -len(".gate_layer")]
                register(
                    f"{stem}.gate_up",
                    SplitTargets(
                        targets=((k, 0, g_dim), (proj_key, g_dim, p_dim)),
                        slice_dim=0,
                        full_dim=g_dim + p_dim,
                    ),
                )

    return key_map


# endregion
