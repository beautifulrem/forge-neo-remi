"""HuggingFace Diffusers format & sharded safetensors support for Anima.

Handles ``circlestone-labs/Anima-Base-v1.0-Diffusers`` (both as a complete
Diffusers directory containing ``modular_model_index.json`` / ``model_index.json``
and as standalone Diffusers component files: ``transformer``, ``text_conditioner``,
``text_encoder``, ``vae``), plus sharded HuggingFace safetensors checkpoints
(e.g. ``Qwen/Qwen3.5-4B`` with ``model.safetensors.index.json``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

import safetensors
import torch

_SHARD_FILE_RE = re.compile(r"^(.*)-\d{5}-of-\d{5}\.safetensors$", re.IGNORECASE)
_DIFFUSERS_INDEX_FILES = ("modular_model_index.json", "model_index.json")
_SHARD_INDEX_FILES = ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
_SINGLE_WEIGHT_FILES = ("diffusion_pytorch_model.safetensors", "model.safetensors")

_TRANSFORMER_BASIC_MAP: tuple[tuple[str, str], ...] = (
    ("patch_embed.proj.", "x_embedder.proj.1."),
    ("time_embed.t_embedder.linear_1.", "t_embedder.1.linear_1."),
    ("time_embed.t_embedder.linear_2.", "t_embedder.1.linear_2."),
    ("time_embed.norm.", "t_embedding_norm."),
    ("norm_out.linear_1.", "final_layer.adaln_modulation.1."),
    ("norm_out.linear_2.", "final_layer.adaln_modulation.2."),
    ("proj_out.", "final_layer.linear."),
)

_TRANSFORMER_BLOCK_MAP: tuple[tuple[str, str], ...] = (
    ("norm1.linear_1.", "adaln_modulation_self_attn.1."),
    ("norm1.linear_2.", "adaln_modulation_self_attn.2."),
    ("norm2.linear_1.", "adaln_modulation_cross_attn.1."),
    ("norm2.linear_2.", "adaln_modulation_cross_attn.2."),
    ("norm3.linear_1.", "adaln_modulation_mlp.1."),
    ("norm3.linear_2.", "adaln_modulation_mlp.2."),
    ("attn1.to_q.", "self_attn.q_proj."),
    ("attn1.to_k.", "self_attn.k_proj."),
    ("attn1.to_v.", "self_attn.v_proj."),
    ("attn1.to_out.0.", "self_attn.output_proj."),
    ("attn1.norm_q.", "self_attn.q_norm."),
    ("attn1.norm_k.", "self_attn.k_norm."),
    ("attn2.to_q.", "cross_attn.q_proj."),
    ("attn2.to_k.", "cross_attn.k_proj."),
    ("attn2.to_v.", "cross_attn.v_proj."),
    ("attn2.to_out.0.", "cross_attn.output_proj."),
    ("attn2.norm_q.", "cross_attn.q_norm."),
    ("attn2.norm_k.", "cross_attn.k_norm."),
    ("ff.net.0.proj.", "mlp.layer1."),
    ("ff.net.2.", "mlp.layer2."),
)

_WAN_RESNET_MAP: tuple[tuple[str, str], ...] = (
    ("norm1.", "residual.0."),
    ("conv1.", "residual.2."),
    ("norm2.", "residual.3."),
    ("conv2.", "residual.6."),
    ("conv_shortcut.", "shortcut."),
)


def is_diffusers_anima_transformer(state_dict: dict) -> bool:
    """True for Diffusers ``CosmosTransformer3DModel`` (Anima) state dicts."""

    return (
        "patch_embed.proj.weight" in state_dict
        and "time_embed.t_embedder.linear_1.weight" in state_dict
        and "transformer_blocks.0.attn1.to_q.weight" in state_dict
        and "transformer_blocks.0.norm1.linear_1.weight" in state_dict
    )


def convert_diffusers_anima_transformer(state_dict: dict[str, torch.Tensor], output_prefix: str = "") -> dict[str, torch.Tensor]:
    """Convert Diffusers ``CosmosTransformer3DModel`` keys to native Anima DiT keys."""

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        mapped = None
        if key.startswith("transformer_blocks."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                idx, rest = parts[1], parts[2]
                for src, dst in _TRANSFORMER_BLOCK_MAP:
                    if rest.startswith(src):
                        mapped = f"blocks.{idx}.{dst}{rest[len(src) :]}"
                        break
        else:
            for src, dst in _TRANSFORMER_BASIC_MAP:
                if key.startswith(src):
                    mapped = f"{dst}{key[len(src) :]}"
                    break
        out[f"{output_prefix}{mapped or key}"] = value
    return out


def is_diffusers_anima_text_conditioner(state_dict: dict) -> bool:
    """True for Diffusers ``AnimaTextConditioner`` (``LLMAdapter``) weights."""

    return (
        "embed.weight" in state_dict
        and "out_proj.weight" in state_dict
        and "blocks.0.cross_attn.q_proj.weight" in state_dict
        and "blocks.0.self_attn.q_proj.weight" in state_dict
    )


def is_diffusers_wan_vae(state_dict: dict) -> bool:
    """True for Diffusers ``AutoencoderKLWan`` / ``AutoencoderKLQwenImage`` weights
    (excludes ``AutoencoderKLQwenImage21``, which uses nested ``down_blocks.*.resnets``
    and singular ``upsampler`` containers)."""

    return (
        "decoder.mid_block.resnets.0.norm1.gamma" in state_dict
        and "decoder.up_blocks.0.resnets.0.conv1.weight" in state_dict
        and "decoder.norm_out.gamma" in state_dict
        and "encoder.down_blocks.0.resnets.0.conv1.weight" not in state_dict
        and "decoder.up_blocks.0.upsampler.resample.1.weight" not in state_dict
    )


def _map_wan_resnet(rest: str) -> str:
    for src, dst in _WAN_RESNET_MAP:
        if rest.startswith(src):
            return dst + rest[len(src) :]
    return rest


def convert_diffusers_wan_vae(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert Diffusers 3D Wan/QwenImage VAE keys (194 keys) to native ``WanVAE`` keys."""

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
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
                    new_key = f"{side}.middle.0.{_map_wan_resnet(mid[len('resnets.0.') :])}"
                elif mid.startswith("attentions.0."):
                    new_key = f"{side}.middle.1.{mid[len('attentions.0.') :]}"
                elif mid.startswith("resnets.1."):
                    new_key = f"{side}.middle.2.{_map_wan_resnet(mid[len('resnets.1.') :])}"
            elif side == "encoder" and rest.startswith("down_blocks."):
                parts = rest.split(".", 2)
                if len(parts) == 3 and parts[1].isdigit():
                    new_key = f"encoder.downsamples.{parts[1]}.{_map_wan_resnet(parts[2])}"
            elif side == "decoder" and rest.startswith("up_blocks."):
                parts = rest.split(".", 3)
                if len(parts) == 4 and parts[1].isdigit():
                    stage = int(parts[1])
                    kind, tail = parts[2], parts[3]
                    if kind == "resnets" and "." in tail:
                        r_idx, sub = tail.split(".", 1)
                        if r_idx.isdigit():
                            new_key = f"decoder.upsamples.{stage * 4 + int(r_idx)}.{_map_wan_resnet(sub)}"
                    elif kind == "upsamplers" and "." in tail:
                        _, sub = tail.split(".", 1)
                        new_key = f"decoder.upsamples.{stage * 4 + 3}.{sub}"
        out[new_key] = value
    return out


def _read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def is_anima_diffusers_dir(path: os.PathLike | str) -> bool:
    """True when ``path`` is a local Diffusers directory for an Anima pipeline."""

    path = os.fspath(path)
    if not os.path.isdir(path):
        return False
    transformer_dir = os.path.join(path, "transformer")
    if not os.path.isdir(transformer_dir):
        return False
    if anima_diffusers_primary_file(path) is None:
        return False
    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(path, index_name))
        if not index:
            continue
        cls_name = str(index.get("_class_name", ""))
        transformer_entry = index.get("transformer")
        if "Anima" in cls_name:
            return True
        if isinstance(transformer_entry, list) and len(transformer_entry) >= 2 and transformer_entry[1] == "CosmosTransformer3DModel" and "text_conditioner" in index:
            return True
    t_cfg = _read_json(os.path.join(transformer_dir, "config.json"))
    return bool(t_cfg and t_cfg.get("_class_name") == "CosmosTransformer3DModel" and os.path.isdir(os.path.join(path, "text_conditioner")))


def _component_weight_source(component_dir: str) -> str | None:
    for name in _SHARD_INDEX_FILES:
        candidate = os.path.join(component_dir, name)
        if os.path.isfile(candidate):
            return candidate
    for name in _SINGLE_WEIGHT_FILES:
        candidate = os.path.join(component_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def anima_diffusers_primary_file(directory: os.PathLike | str) -> str | None:
    """A concrete safetensors file inside an Anima Diffusers folder (for hashing/metadata)."""

    transformer_dir = os.path.join(os.fspath(directory), "transformer")
    if not os.path.isdir(transformer_dir):
        return None
    for name in _SINGLE_WEIGHT_FILES:
        candidate = os.path.join(transformer_dir, name)
        if os.path.isfile(candidate):
            return candidate
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(transformer_dir, name))
        weight_map = index.get("weight_map") if index else None
        if isinstance(weight_map, dict) and weight_map:
            first_shard = sorted(set(weight_map.values()))[0]
            candidate = os.path.join(transformer_dir, first_shard)
            if os.path.isfile(candidate):
                return candidate
    return None


def is_sharded_hf_dir(path: os.PathLike | str) -> bool:
    """True when ``path`` is a HuggingFace model folder with a safetensors shard index
    (e.g. ``Qwen/Qwen3.5-4B`` downloaded via ``huggingface-cli download``)."""

    path = os.fspath(path)
    if not os.path.isdir(path):
        return False
    return any(os.path.isfile(os.path.join(path, name)) for name in _SHARD_INDEX_FILES)


def find_sharded_index(path: os.PathLike | str) -> str | None:
    """Resolve ``path`` (a folder, an ``*.safetensors.index.json``, or a shard file
    ``*-00001-of-0000N.safetensors``) to its safetensors shard index JSON, if any."""

    text = os.fspath(path)
    if os.path.isdir(text):
        for name in _SHARD_INDEX_FILES:
            candidate = os.path.join(text, name)
            if os.path.isfile(candidate):
                return candidate
        return None
    if text.lower().endswith(".safetensors.index.json") and os.path.isfile(text):
        return text
    if match := _SHARD_FILE_RE.match(os.path.basename(text)):
        directory = os.path.dirname(text)
        stem = match.group(1)
        for candidate_name in (f"{stem}.index.json", f"{stem}.safetensors.index.json", *_SHARD_INDEX_FILES):
            candidate = os.path.join(directory, candidate_name)
            if os.path.isfile(candidate):
                return candidate
    return None


def load_sharded_safetensors(
    index_path: os.PathLike | str,
    *,
    device: torch.device | str | None = None,
    disable_mmap: bool = False,
    key_filter: Callable[[str], bool] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load tensors referenced by a HuggingFace ``*.safetensors.index.json``.

    When ``key_filter`` is supplied, shards whose keys are all rejected are not
    even opened (avoiding loading unused vision/MTP weights on ``Qwen3.5-4B``).
    """

    index_path = os.fspath(index_path)
    index = _read_json(index_path)
    if not index or not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"Invalid safetensors index: {index_path}")

    base_dir = os.path.dirname(os.path.abspath(index_path))
    shard_keys: dict[str, list[str]] = {}
    for key, shard_name in index["weight_map"].items():
        if key_filter is not None and not key_filter(key):
            continue
        shard_keys.setdefault(str(shard_name), []).append(str(key))

    dev = torch.device(device or "cpu")
    sd: dict[str, torch.Tensor] = {}
    metadata: dict = {}
    for shard_name in sorted(shard_keys):
        shard_path = os.path.join(base_dir, shard_name)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Missing safetensors shard {shard_name!r} referenced by {index_path}")
        with safetensors.safe_open(shard_path, framework="pt", device=str(dev)) as handle:
            if not metadata and handle.metadata():
                metadata = dict(handle.metadata())
            for key in shard_keys[shard_name]:
                tensor = handle.get_tensor(key)
                if disable_mmap:
                    tensor = tensor.to(device=dev, copy=True)
                sd[key] = tensor
    return sd, metadata


def _load_component_sd(component_dir: str, *, device=None) -> tuple[dict[str, torch.Tensor], dict]:
    from backend.utils import DISABLE_MMAP, load_torch_file

    source = _component_weight_source(component_dir)
    if source is None:
        return {}, {}
    if source.lower().endswith(".safetensors.index.json"):
        return load_sharded_safetensors(source, device=device, disable_mmap=DISABLE_MMAP)
    sd, meta = load_torch_file(source, return_metadata=True, device=device)
    return sd, dict(meta or {})


def load_anima_diffusers_dir(directory: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load an Anima Diffusers folder into Forge's unified checkpoint state dict.

    Produces ``model.diffusion_model.*`` (transformer + ``llm_adapter``), plus
    bundled ``text_encoders.qwen3_06b.transformer.*`` and ``vae.*`` when present
    (any of which may still be overridden by user-selected additional modules).
    """

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    raw_transformer, metadata = _load_component_sd(transformer_dir)
    if not raw_transformer:
        raise FileNotFoundError(f"Anima Diffusers directory has no transformer weights: {transformer_dir}")

    if is_diffusers_anima_transformer(raw_transformer):
        sd = convert_diffusers_anima_transformer(raw_transformer, output_prefix="model.diffusion_model.")
    else:
        sd = {f"model.diffusion_model.{k}": v for k, v in raw_transformer.items()}
    del raw_transformer

    conditioner_dir = os.path.join(root, "text_conditioner")
    if os.path.isdir(conditioner_dir):
        cond_sd, _ = _load_component_sd(conditioner_dir)
        for key, value in cond_sd.items():
            target = key if key.startswith("llm_adapter.") else f"llm_adapter.{key}"
            sd[f"model.diffusion_model.{target}"] = value
        del cond_sd

    te_dir = os.path.join(root, "text_encoder")
    if os.path.isdir(te_dir):
        te_sd, _ = _load_component_sd(te_dir)
        if te_sd:
            has_model_prefix = any(k.startswith("model.") for k in te_sd)
            for key, value in te_sd.items():
                target = key if has_model_prefix or key.startswith("lm_head.") else f"model.{key}"
                sd[f"text_encoders.qwen3_06b.transformer.{target}"] = value
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
