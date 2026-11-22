"""Diffusers / sharded checkpoint and LoRA support for FLUX.3 Image Klein.

Handles:
- complete ``black-forest-labs/flux-3-image-klein[-base]`` Diffusers folders
  (``model_index.json`` / ``modular_model_index.json`` + sharded ``transformer/``
  + ``text_encoder/`` (Qwen3-VL-4B) + ``vae/`` (FLUX.2 VAE));
- standalone sharded ``transformer/`` folders and single-file Diffusers
  transformers (including TorchAO ``_weight_qdata`` FP8/INT8 packages once
  ``convert_quantization`` has run);
- LoRA key maps for native (BFL / ComfyUI) and Diffusers / PEFT trainers,
  including split q/k/v LoRAs on a fused-``qkv`` checkpoint.

Native keys follow the BFL layout used by ``backend/nn/flux3.py``.
"""

from __future__ import annotations

import os

import torch

from backend.nn.anima_diffusers import _DIFFUSERS_INDEX_FILES, _SHARD_INDEX_FILES, _load_component_sd, _read_json

DIFFUSERS_CLASS = "Flux3Transformer2DModel"

_EXACT = {
    "timestep_embedder.linear_1": "time_in.in_layer",
    "timestep_embedder.linear_2": "time_in.out_layer",
    "target_timestep_embedder.linear_1": "ref_time_in.in_layer",
    "target_timestep_embedder.linear_2": "ref_time_in.out_layer",
    "proj_out": "final_layer.image.linear",
    "norm_out.linear": "final_layer.image.adaLN_modulation.1",
}
_PREFIX = (
    ("x_embedders.", "emb_in."),
    ("context_embedder.", "txt_in."),
    ("mode_blocks.", "content_mode_blocks."),
    ("single_transformer_blocks.", "single_blocks."),
)
_BLOCK = (
    ("attn.to_qkv", "qkv"),
    ("attn.to_q", "q_proj"),
    ("attn.to_k", "k_proj"),
    ("attn.to_v", "v_proj"),
    ("attn.to_out.0", "attn_out"),
    ("attn.norm_q", "norm.query_norm"),
    ("attn.norm_k", "norm.key_norm"),
    ("ff.linear_in", "mlp_in"),
    ("ff.linear_out", "mlp_out"),
)
_PARAMS = ("weight", "bias", "weight_scale", "input_scale", "comfy_quant", "scale")


def _strip_prefix(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_diffusers_flux3_transformer(state_dict: dict) -> bool:
    """True for Diffusers ``Flux3Transformer2DModel`` state dicts."""

    keys = {_strip_prefix(k) for k in state_dict}
    return "early_stream_modulations.image.linear.weight" in keys and ("x_embedders.image.weight" in keys or any(k.startswith("single_transformer_blocks.0.") for k in keys))


def is_native_flux3_transformer(state_dict: dict) -> bool:
    keys = {_strip_prefix(k) for k in state_dict}
    return "content_mode_blocks.image.0.attn_out.weight" in keys and "single_stream_modulations.txt.lin.weight" in keys


def _split_param(key: str) -> tuple[str, str]:
    module, _, param = key.rpartition(".")
    if param in _PARAMS and module:
        return module, param
    return key, ""


def diffusers_to_native_module(module: str) -> str:
    """A Diffusers module path (no parameter name) as its native FLUX.3 path."""

    if module in _EXACT:
        return _EXACT[module]
    for src, dst in _PREFIX:
        if (module + ".").startswith(src):
            module = (dst + (module + ".")[len(src) :])[:-1]
            break
    if module.startswith(("early_stream_modulations.", "single_stream_modulations.")) and module.endswith(".linear"):
        return module[: -len(".linear")] + ".lin"
    for src, dst in _BLOCK:
        if module.endswith("." + src):
            return module[: -len(src)] + dst
    return module


def convert_diffusers_flux3_transformer(state_dict: dict[str, torch.Tensor], output_prefix: str = "") -> dict[str, torch.Tensor]:
    """Diffusers ``Flux3Transformer2DModel`` keys (and quantized sidecars such as
    ``.weight_scale`` / ``.comfy_quant``) as native FLUX.3 keys. The output
    adaLN produces [scale, shift] in Diffusers and [shift, scale] natively."""

    out: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = _strip_prefix(raw_key)
        if key.startswith("repa_projectors."):  # training-only heads
            continue
        module, param = _split_param(key)
        native = diffusers_to_native_module(module)
        if native == "final_layer.image.adaLN_modulation.1" and param == "weight" and value.ndim >= 1 and value.is_floating_point():
            scale, shift = value.chunk(2, dim=0)
            value = torch.cat([shift, scale], dim=0)
        out[f"{output_prefix}{native}.{param}" if param else f"{output_prefix}{native}"] = value
    return out


# region Diffusers folders


def _dir_primary(directory: str) -> str | None:
    from backend.nn.moss_diffusers import _dir_primary_safetensors

    return _dir_primary_safetensors(directory)


def flux3_diffusers_primary_file(directory: os.PathLike | str) -> str | None:
    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    if os.path.isdir(transformer_dir) and (primary := _dir_primary(transformer_dir)):
        return primary
    return _dir_primary(root)


def is_flux3_diffusers_dir(path: os.PathLike | str) -> bool:
    """True for a local FLUX.3 Diffusers pipeline folder."""

    root = os.fspath(path)
    transformer_dir = os.path.join(root, "transformer")
    if not os.path.isdir(root) or not os.path.isdir(transformer_dir) or _dir_primary(transformer_dir) is None:
        return False
    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(root, index_name))
        if not index:
            continue
        entry = index.get("transformer")
        if "Flux3" in str(index.get("_class_name", "")):
            return True
        if isinstance(entry, list) and DIFFUSERS_CLASS in entry:
            return True
    config = _read_json(os.path.join(transformer_dir, "config.json"))
    return bool(config and config.get("_class_name") == DIFFUSERS_CLASS)


def is_sharded_flux3_transformer_dir(path: os.PathLike | str) -> bool:
    """True for a standalone (sharded) FLUX.3 transformer folder."""

    root = os.fspath(path)
    if not os.path.isdir(root) or os.path.isdir(os.path.join(root, "transformer")) or _dir_primary(root) is None:
        return False
    config = _read_json(os.path.join(root, "config.json"))
    if config and config.get("_class_name") == DIFFUSERS_CLASS:
        return True
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(root, name))
        weight_map = index.get("weight_map") if index else None
        if isinstance(weight_map, dict) and any(k in weight_map for k in ("early_stream_modulations.image.linear.weight", "early_stream_modulations.image.lin.weight")):
            return True
    return False


def is_flux3_checkpoint_dir(path: os.PathLike | str) -> bool:
    return is_flux3_diffusers_dir(path) or is_sharded_flux3_transformer_dir(path)


def load_flux3_diffusers_dir(directory: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """A FLUX.3 Diffusers folder (or standalone transformer folder) as Forge's
    unified checkpoint state dict."""

    from backend.nn.krea_diffusers import normalize_qwen3vl_key

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    is_pipeline = os.path.isdir(transformer_dir)
    if not is_pipeline:
        transformer_dir = root

    raw, metadata = _load_component_sd(transformer_dir)
    if not raw:
        raise FileNotFoundError(f"FLUX.3 Diffusers directory has no transformer weights: {transformer_dir}")
    if is_diffusers_flux3_transformer(raw):
        sd = convert_diffusers_flux3_transformer(raw, output_prefix="model.diffusion_model.")
    else:
        sd = {(k if k.startswith("model.diffusion_model.") else f"model.diffusion_model.{k}"): v for k, v in raw.items()}
    del raw
    if not is_pipeline:
        return sd, metadata

    te_dir = os.path.join(root, "text_encoder")
    if os.path.isdir(te_dir):
        te_sd, _ = _load_component_sd(te_dir)
        for key, value in te_sd.items():
            sd[f"text_encoders.qwen3vl_4b.transformer.{normalize_qwen3vl_key(key)}"] = value
        del te_sd

    vae_dir = os.path.join(root, "vae")
    if os.path.isdir(vae_dir):
        vae_sd, _ = _load_component_sd(vae_dir)
        if "decoder.up_blocks.0.resnets.0.norm1.weight" in vae_sd:
            from modules_forge.packages.huggingface_guess.diffusers_convert import convert_vae_state_dict

            vae_sd = convert_vae_state_dict(vae_sd)
        for key, value in vae_sd.items():
            if key.startswith("decoder.post_quant_conv."):
                key = "post_quant_conv." + key[len("decoder.post_quant_conv.") :]
            elif key.startswith("encoder.quant_conv."):
                key = "quant_conv." + key[len("encoder.quant_conv.") :]
            sd[f"vae.{key}"] = value
        del vae_sd
    return sd, metadata


# endregion

# region LoRA


def _native_to_diffusers_modules() -> list[tuple[str, str]]:
    """(native suffix, Diffusers suffix) pairs for block internals."""

    return [(dst, src) for src, dst in _BLOCK if src != "attn.to_qkv"]


def flux3_lora_key_map(sd: dict[str, torch.Tensor]) -> dict[str, object]:
    """LoRA keys of every common trainer naming onto the loaded FLUX.3 model's
    ``diffusion_model.*`` weights (or slices of a fused ``qkv``)."""

    key_map: dict[str, object] = {}

    def register(bare: str, target) -> None:
        flat = bare.replace(".", "_")
        for prefix in ("", "diffusion_model.", "transformer.", "base_model.model.", "base_model.model.diffusion_model.", "base_model.model.transformer."):
            key_map[f"{prefix}{bare}"] = target
        key_map[f"lora_unet_{flat}"] = target
        key_map[f"lycoris_{flat}"] = target
        key_map[f"lora_transformer_{flat}"] = target

    native_to_diff_prefix = [(dst, src) for src, dst in _PREFIX]
    native_to_diff_exact = {dst: src for src, dst in _EXACT.items()}

    def diffusers_name(native: str) -> str:
        if native in native_to_diff_exact:
            return native_to_diff_exact[native]
        for src, dst in native_to_diff_prefix:
            if (native + ".").startswith(src):
                native = (dst + (native + ".")[len(src) :])[:-1]
                break
        if native.startswith(("early_stream_modulations.", "single_stream_modulations.")) and native.endswith(".lin"):
            return native[: -len(".lin")] + ".linear"
        for src, dst in _native_to_diffusers_modules():
            if native.endswith("." + src):
                return native[: -len(src)] + dst
        return native

    for k in sd:
        if not (k.startswith("diffusion_model.") and k.endswith(".weight")):
            continue
        native = k[len("diffusion_model.") : -len(".weight")]
        register(native, k)
        register(diffusers_name(native), k)
        if native.endswith(".qkv"):  # fused checkpoint: split q/k/v LoRAs address thirds
            third = sd[k].shape[0] // 3
            stem = native[: -len(".qkv")]
            diff_stem = diffusers_name(f"{stem}.attn_out")[: -len("attn.to_out.0")]
            for i, (nat, diff) in enumerate((("q_proj", "attn.to_q"), ("k_proj", "attn.to_k"), ("v_proj", "attn.to_v"))):
                target = (k, (0, i * third, third))
                register(f"{stem}.{nat}", target)
                register(f"{diff_stem}{diff}", target)
    return key_map


# endregion
