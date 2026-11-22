"""HuggingFace Diffusers & Unsloth format support for Krea 2.

Handles:
- complete ``krea/Krea-2-Raw``, ``krea/Krea-2-Turbo``, and ``unsloth/Krea-2-Turbo``
  Diffusers directories (``model_index.json`` + sharded ``transformer/`` +
  ``text_encoder/`` + ``vae/``);
- standalone sharded Krea 2 Diffusers ``transformer/`` directories;
- standalone Diffusers-keyed checkpoints (including ``unsloth/Krea-2-Turbo-FP8``'s
  ``Krea-2-Turbo-FP8.safetensors`` and ``Krea-2-Turbo-INT8.safetensors`` after
  TorchAO ``_weight_qdata`` conversion);
- ``Qwen3VL`` text encoder key normalization (``language_model.*`` / ``visual.*``).
"""

from __future__ import annotations

import os

import torch

from backend.nn.anima_diffusers import (
    _DIFFUSERS_INDEX_FILES,
    _SHARD_INDEX_FILES,
    _load_component_sd,
    _read_json,
    convert_diffusers_wan_vae,
    is_diffusers_wan_vae,
)

_BLOCK_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("attn.to_q.", "attn.wq."),
    ("attn.to_k.", "attn.wk."),
    ("attn.to_v.", "attn.wv."),
    ("attn.to_gate.", "attn.gate."),
    ("attn.to_out.0.", "attn.wo."),
    ("attn.to_out.", "attn.wo."),
    ("attn.norm_q.weight", "attn.qknorm.qnorm.scale"),
    ("attn.norm_k.weight", "attn.qknorm.knorm.scale"),
    ("ff.gate.", "mlp.gate."),
    ("ff.up.", "mlp.up."),
    ("ff.down.", "mlp.down."),
    ("norm1.weight", "prenorm.scale"),
    ("norm2.weight", "postnorm.scale"),
)

_TOP_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("img_in.", "first."),
    ("time_embed.linear_1.", "tmlp.0."),
    ("time_embed.linear_2.", "tmlp.2."),
    ("time_mod_proj.", "tproj.1."),
    ("txt_in.norm.weight", "txtmlp.0.scale"),
    ("txt_in.linear_1.", "txtmlp.1."),
    ("txt_in.linear_2.", "txtmlp.3."),
    ("text_fusion.projector.", "txtfusion.projector."),
    ("final_layer.norm.weight", "last.norm.scale"),
    ("final_layer.scale_shift_table", "last.modulation.lin"),
    ("final_layer.linear.", "last.linear."),
)


def _strip_diffusers_prefix(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_diffusers_krea2_transformer(state_dict: dict) -> bool:
    """True for Diffusers ``Krea2Transformer2DModel`` state dicts (including
    Unsloth FP8/INT8 checkpoints converted via ``convert_quantization``)."""

    has_projector = any(
        _strip_diffusers_prefix(k) == "text_fusion.projector.weight"
        for k in state_dict
    )
    if not has_projector:
        return False
    return any(
        _strip_diffusers_prefix(k).startswith(
            ("transformer_blocks.0.attn.to_q.", "transformer_blocks.0.scale_shift_table")
        )
        for k in state_dict
    )


def _map_block_rest(rest: str) -> str:
    for src, dst in _BLOCK_KEY_MAP:
        if src.endswith("."):
            if rest.startswith(src):
                return f"{dst}{rest[len(src) :]}"
        elif rest == src:
            return dst
    return rest


def convert_diffusers_krea2_transformer(
    state_dict: dict[str, torch.Tensor],
    output_prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Convert Diffusers ``Krea2Transformer2DModel`` keys (plus any quantized
    sidecar tensors such as ``.weight_scale`` and ``.comfy_quant``) into Forge's
    native ``SingleStreamDiT`` state dict layout (bit-exact)."""

    out: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = _strip_diffusers_prefix(raw_key)
        mapped: str | None = None

        if key.startswith("transformer_blocks."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                idx, rest = parts[1], parts[2]
                if rest == "scale_shift_table":
                    mapped = f"blocks.{idx}.mod.lin"
                    if value.ndim == 2:
                        value = value.reshape(-1)
                else:
                    mapped = f"blocks.{idx}.{_map_block_rest(rest)}"
        elif key.startswith(("text_fusion.layerwise_blocks.", "text_fusion.refiner_blocks.")):
            parts = key.split(".", 3)
            if len(parts) == 4 and parts[2].isdigit():
                sub, idx, rest = parts[1], parts[2], parts[3]
                mapped = f"txtfusion.{sub}.{idx}.{_map_block_rest(rest)}"
        else:
            for src, dst in _TOP_KEY_MAP:
                if src.endswith("."):
                    if key.startswith(src):
                        mapped = f"{dst}{key[len(src) :]}"
                        break
                elif key == src:
                    mapped = dst
                    break

        out[f"{output_prefix}{mapped or key}"] = value
    return out


def normalize_qwen3vl_key(key: str) -> str:
    """Normalize Diffusers/Unsloth ``Qwen3VLModel`` keys (``language_model.*``,
    ``visual.*``) into ``model.language_model.*`` / ``model.visual.*`` while
    preserving Comfy-Org/HF ``model.*`` keys untouched."""

    if key.startswith("language_model."):
        return f"model.{key}"
    if key.startswith("visual."):
        return f"model.{key}"
    return key


def krea2_diffusers_primary_file(directory: os.PathLike | str) -> str | None:
    """A concrete safetensors file inside a Krea 2 Diffusers or sharded folder."""

    from backend.nn.moss_diffusers import _dir_primary_safetensors

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    if os.path.isdir(transformer_dir):
        if primary := _dir_primary_safetensors(transformer_dir):
            return primary
    return _dir_primary_safetensors(root)


def is_krea2_diffusers_dir(path: os.PathLike | str) -> bool:
    """True when ``path`` is a local Diffusers folder for a Krea 2 pipeline
    (``krea/Krea-2-Raw``, ``krea/Krea-2-Turbo``, ``unsloth/Krea-2-Turbo``)."""

    from backend.nn.moss_diffusers import _dir_primary_safetensors

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
        if "Krea2" in cls_name:
            return True
        if isinstance(transformer_entry, list) and len(transformer_entry) >= 2 and transformer_entry[1] == "Krea2Transformer2DModel":
            return True
    t_cfg = _read_json(os.path.join(transformer_dir, "config.json"))
    return bool(t_cfg and t_cfg.get("_class_name") == "Krea2Transformer2DModel")


def is_sharded_krea2_transformer_dir(path: os.PathLike | str) -> bool:
    """True when ``path`` is a standalone sharded Krea 2 transformer folder."""

    from backend.nn.moss_diffusers import _dir_primary_safetensors

    root = os.fspath(path)
    if not os.path.isdir(root) or os.path.isdir(os.path.join(root, "transformer")):
        return False
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(root, name))
        weight_map = index.get("weight_map") if index else None
        if not isinstance(weight_map, dict):
            continue
        has_diffusers = "text_fusion.projector.weight" in weight_map and "transformer_blocks.0.scale_shift_table" in weight_map
        has_native = "txtfusion.projector.weight" in weight_map and "blocks.0.mod.lin" in weight_map
        if (has_diffusers or has_native) and _dir_primary_safetensors(root) is not None:
            return True
    return False


def load_krea2_diffusers_dir(directory: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a ``krea/Krea-2-*`` / ``unsloth/Krea-2-*`` Diffusers folder or
    standalone sharded transformer folder into Forge's unified checkpoint state dict."""

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    is_pipeline_dir = os.path.isdir(transformer_dir)
    if not is_pipeline_dir:
        transformer_dir = root

    raw_transformer, metadata = _load_component_sd(transformer_dir)
    if not raw_transformer:
        raise FileNotFoundError(f"Krea 2 Diffusers directory has no transformer weights: {transformer_dir}")

    if is_diffusers_krea2_transformer(raw_transformer):
        sd = convert_diffusers_krea2_transformer(raw_transformer, output_prefix="model.diffusion_model.")
    else:
        sd = {
            (k if k.startswith("model.diffusion_model.") else f"model.diffusion_model.{k}"): v
            for k, v in raw_transformer.items()
        }
    del raw_transformer

    if not is_pipeline_dir:
        return sd, metadata

    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(root, index_name))
        if index and "is_distilled" in index:
            metadata = dict(metadata or {})
            metadata["krea2_is_distilled"] = str(bool(index["is_distilled"])).lower()
            break

    te_dir = os.path.join(root, "text_encoder")
    if os.path.isdir(te_dir):
        te_sd, _ = _load_component_sd(te_dir)
        if te_sd:
            for key, value in te_sd.items():
                sd[f"text_encoders.qwen3vl_4b.transformer.{normalize_qwen3vl_key(key)}"] = value
        del te_sd

    vae_dir = os.path.join(root, "vae")
    if os.path.isdir(vae_dir):
        vae_sd, _ = _load_component_sd(vae_dir)
        if vae_sd:
            if is_diffusers_wan_vae(vae_sd):
                vae_sd = convert_diffusers_wan_vae(vae_sd)
            for key, value in vae_sd.items():
                sd[f"vae.{key}"] = value
        del vae_sd

    return sd, metadata
