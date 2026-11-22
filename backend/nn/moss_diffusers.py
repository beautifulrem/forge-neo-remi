"""HuggingFace Diffusers directory & sharded checkpoint support for MOSS-Image.

Handles:
- complete ``OpenMOSS-Team/MOSS-Image-SFT`` / ``OpenMOSS-Team/MOSS-Image-Base``
  Diffusers directories (``model_index.json`` + sharded ``transformer/`` +
  sharded ``text_encoder/`` + ``vae/``);
- standalone sharded MOSS-Image transformer folders (e.g.
  ``lodestones/MOSS-Image-SFT-8-steps-dmd/transformer``);
- unified checkpoint-directory discovery shared by ``modules.sd_models`` for
  both Anima and MOSS-Image Diffusers folders.
"""

from __future__ import annotations

import os

import torch

from backend.nn.anima_diffusers import (
    _DIFFUSERS_INDEX_FILES,
    _SHARD_INDEX_FILES,
    _SINGLE_WEIGHT_FILES,
    _load_component_sd,
    _read_json,
    anima_diffusers_primary_file,
    is_anima_diffusers_dir,
)


def _dir_primary_safetensors(directory: str) -> str | None:
    if not os.path.isdir(directory):
        return None
    for name in _SINGLE_WEIGHT_FILES:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(directory, name))
        weight_map = index.get("weight_map") if index else None
        if isinstance(weight_map, dict) and weight_map:
            first_shard = sorted(set(weight_map.values()))[0]
            candidate = os.path.join(directory, first_shard)
            if os.path.isfile(candidate):
                return candidate
    return None


def moss_diffusers_primary_file(directory: os.PathLike | str) -> str | None:
    """A concrete safetensors file inside a MOSS-Image Diffusers or sharded folder."""

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    if os.path.isdir(transformer_dir):
        if primary := _dir_primary_safetensors(transformer_dir):
            return primary
    return _dir_primary_safetensors(root)


def is_moss_diffusers_dir(path: os.PathLike | str) -> bool:
    """True when ``path`` is a local Diffusers folder for a MOSS-Image pipeline."""

    root = os.fspath(path)
    if not os.path.isdir(root):
        return False
    transformer_dir = os.path.join(root, "transformer")
    if not os.path.isdir(transformer_dir) or _dir_primary_safetensors(transformer_dir) is None:
        return False
    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(root, index_name))
        if not index:
            continue
        cls_name = str(index.get("_class_name", ""))
        transformer_entry = index.get("transformer")
        if "MOSSImage" in cls_name:
            return True
        if isinstance(transformer_entry, list) and len(transformer_entry) >= 2 and transformer_entry[1] == "MOSSImageTransformer":
            return True
    t_cfg = _read_json(os.path.join(transformer_dir, "config.json"))
    return bool(t_cfg and t_cfg.get("_class_name") == "MOSSImageTransformer")


def is_sharded_moss_transformer_dir(path: os.PathLike | str) -> bool:
    """True when ``path`` is a standalone sharded MOSS-Image transformer folder
    (e.g. ``lodestones/MOSS-Image-SFT-8-steps-dmd/transformer``)."""

    root = os.fspath(path)
    if not os.path.isdir(root) or os.path.isdir(os.path.join(root, "transformer")):
        return False
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(root, name))
        weight_map = index.get("weight_map") if index else None
        if not isinstance(weight_map, dict):
            continue
        has_mod = any(k.endswith("shared_block_modulation.weight") for k in weight_map)
        has_ffn = any(k.endswith("layers.0.proj.ffn_up_gate.weight") for k in weight_map)
        if has_mod and has_ffn and _dir_primary_safetensors(root) is not None:
            return True
    return False


def is_diffusers_checkpoint_dir(path: os.PathLike | str) -> bool:
    """True for any Diffusers/sharded checkpoint directory recognised by ``sd_models``."""

    from backend.nn.krea_diffusers import is_krea2_diffusers_dir, is_sharded_krea2_transformer_dir

    return (
        is_anima_diffusers_dir(path)
        or is_moss_diffusers_dir(path)
        or is_sharded_moss_transformer_dir(path)
        or is_krea2_diffusers_dir(path)
        or is_sharded_krea2_transformer_dir(path)
    )


def diffusers_checkpoint_info(path: os.PathLike | str) -> tuple[str | None, str | None]:
    """Return ``(modelspec_architecture, primary_safetensors_file)`` for a checkpoint dir."""

    from backend.nn.krea_diffusers import is_krea2_diffusers_dir, is_sharded_krea2_transformer_dir, krea2_diffusers_primary_file

    if is_anima_diffusers_dir(path):
        return "anima/diffusers", anima_diffusers_primary_file(path)
    if is_moss_diffusers_dir(path) or is_sharded_moss_transformer_dir(path):
        return "moss-image/diffusers", moss_diffusers_primary_file(path)
    if is_krea2_diffusers_dir(path) or is_sharded_krea2_transformer_dir(path):
        return "krea-2/diffusers", krea2_diffusers_primary_file(path)
    return None, None


def normalize_qwen35_9b_key(key: str) -> str:
    """Normalize ``OpenMOSS-Team`` / ``unsloth`` ``language_model.*`` / ``visual.*``
    keys into the ``model.language_model.*`` / ``model.visual.*`` layout expected by
    Forge's ``Qwen3_5Model`` loader."""

    if key.startswith("language_model."):
        return f"model.{key}"
    if key.startswith("visual."):
        return f"model.{key}"
    return key


def load_moss_diffusers_dir(directory: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load an ``OpenMOSS-Team/MOSS-Image-*`` Diffusers folder or standalone sharded
    transformer folder into Forge's unified checkpoint state dict."""

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    is_pipeline_dir = os.path.isdir(transformer_dir)
    if not is_pipeline_dir:
        transformer_dir = root
    raw_transformer, metadata = _load_component_sd(transformer_dir)
    if not raw_transformer:
        raise FileNotFoundError(f"MOSS-Image Diffusers directory has no transformer weights: {transformer_dir}")

    sd: dict[str, torch.Tensor] = {}
    for key, value in raw_transformer.items():
        target = key if key.startswith("model.diffusion_model.") else f"model.diffusion_model.{key}"
        sd[target] = value
    del raw_transformer

    if not is_pipeline_dir:
        return sd, metadata

    te_dir = os.path.join(root, "text_encoder")
    if os.path.isdir(te_dir):
        te_sd, _ = _load_component_sd(te_dir)
        if te_sd:
            for key, value in te_sd.items():
                sd[f"text_encoders.qwen35_9b.transformer.{normalize_qwen35_9b_key(key)}"] = value
        del te_sd

    vae_dir = os.path.join(root, "vae")
    if os.path.isdir(vae_dir):
        vae_sd, _ = _load_component_sd(vae_dir)
        if vae_sd:
            for key, value in vae_sd.items():
                if key.startswith("decoder.post_quant_conv."):
                    key = "post_quant_conv." + key[len("decoder.post_quant_conv.") :]
                elif key.startswith("encoder.quant_conv."):
                    key = "quant_conv." + key[len("encoder.quant_conv.") :]
                sd[f"vae.{key}"] = value
        del vae_sd

    return sd, metadata
