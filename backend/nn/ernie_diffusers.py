"""HuggingFace Diffusers, quantization, LoRA & Spectrum helpers for ERNIE-Image (8B Apache-2.0).

Supports:
- ``baidu/ERNIE-Image`` and ``baidu/ERNIE-Image-Turbo`` Diffusers pipeline
  directories (``model_index.json`` + sharded ``transformer/`` +
  ``text_encoder/`` Ministral-3 + ``vae/`` FLUX.2 VAE);
- Standalone sharded ERNIE-Image ``transformer/`` directories;
- Sensitive-layer protection and INT8-ConvRot / FP8 rowwise quantization filters;
- Multi-trainer LoRA key mapping;
- Spectrum pre-head forecasting directly on ``ErnieImageModel.final_norm``
  (zero edits to upstream ``backend/nn/ernie.py``);
- Non-blocking workflow advice.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

import torch

from backend.nn.anima_diffusers import (
    _DIFFUSERS_INDEX_FILES,
    _SHARD_INDEX_FILES,
    _load_component_sd,
    _read_json,
)

DIFFUSERS_CLASS = "ErnieImageTransformer2DModel"
DIFFUSERS_PIPELINE_CLASS = "ErnieImagePipeline"
ERNIE_LICENSE = "Apache-2.0"

ERNIE_SENSITIVE_PREFIXES: tuple[str, ...] = (
    "x_embedder.",
    "text_proj.",
    "time_proj.",
    "time_embedding.",
    "adaLN_modulation.",
    "final_norm.",
    "final_linear.",
)

ERNIE_INT8_LINEAR_SUFFIXES: tuple[str, ...] = (
    ".self_attention.to_q.weight",
    ".self_attention.to_k.weight",
    ".self_attention.to_v.weight",
    ".self_attention.to_out.0.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.linear_fc2.weight",
)

_ERNIE_TURBO_RE = re.compile(r"(?:^|[-_ /])(?:turbo|distill(?:ed)?|8[-_]?steps?)(?:$|[-_ /.])", re.IGNORECASE)


def _dir_primary(directory: str) -> str | None:
    from backend.nn.moss_diffusers import _dir_primary_safetensors

    return _dir_primary_safetensors(directory)


def _strip_prefix(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_ernie_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """True when ``state_dict`` matches an ``ErnieImageModel`` checkpoint."""

    keys = {_strip_prefix(k) for k in state_dict}
    return "layers.0.adaLN_sa_ln.weight" in keys and (
        "layers.0.mlp.linear_fc2.weight" in keys or "final_norm.linear.weight" in keys
    )


# region Diffusers & sharded directories


def ernie_diffusers_primary_file(directory: os.PathLike | str) -> str | None:
    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    if os.path.isdir(transformer_dir) and (primary := _dir_primary(transformer_dir)):
        return primary
    return _dir_primary(root)


def is_ernie_diffusers_dir(path: os.PathLike | str) -> bool:
    """True for a local ``baidu/ERNIE-Image*`` Diffusers pipeline folder."""

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
        if DIFFUSERS_PIPELINE_CLASS in cls_name or "ErnieImage" in cls_name:
            return True
        if isinstance(entry, list) and DIFFUSERS_CLASS in entry:
            return True
    cfg = _read_json(os.path.join(transformer_dir, "config.json"))
    return bool(cfg and cfg.get("_class_name") == DIFFUSERS_CLASS)


def is_sharded_ernie_transformer_dir(path: os.PathLike | str) -> bool:
    """True for a standalone sharded ERNIE-Image transformer folder."""

    root = os.fspath(path)
    if not os.path.isdir(root) or os.path.isdir(os.path.join(root, "transformer")) or _dir_primary(root) is None:
        return False
    cfg = _read_json(os.path.join(root, "config.json"))
    if cfg and cfg.get("_class_name") == DIFFUSERS_CLASS:
        return True
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(root, name))
        weight_map = index.get("weight_map") if index else None
        if isinstance(weight_map, dict) and "layers.0.adaLN_sa_ln.weight" in weight_map and "layers.0.mlp.linear_fc2.weight" in weight_map:
            return True
    return False


def is_ernie_checkpoint_dir(path: os.PathLike | str) -> bool:
    return is_ernie_diffusers_dir(path) or is_sharded_ernie_transformer_dir(path)


def load_ernie_diffusers_dir(directory: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a ``baidu/ERNIE-Image*`` Diffusers folder or standalone sharded
    transformer folder into Forge's unified checkpoint state dict."""

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    is_pipeline = os.path.isdir(transformer_dir)
    if not is_pipeline:
        transformer_dir = root

    raw, metadata = _load_component_sd(transformer_dir)
    if not raw:
        raise FileNotFoundError(f"ERNIE-Image Diffusers directory has no transformer weights: {transformer_dir}")

    sd: dict[str, torch.Tensor] = {
        (k if k.startswith("model.diffusion_model.") else f"model.diffusion_model.{k}"): v for k, v in raw.items()
    }
    del raw

    if not is_pipeline:
        return sd, metadata

    metadata = dict(metadata or {})
    metadata.setdefault("license", ERNIE_LICENSE)

    te_dir = os.path.join(root, "text_encoder")
    if os.path.isdir(te_dir):
        te_sd, _ = _load_component_sd(te_dir)
        for key, value in te_sd.items():
            norm_k = key[len("language_model.") :] if key.startswith("language_model.") else key
            sd[f"text_encoders.ministral3_3b.transformer.{norm_k}"] = value
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


# region LoRA, Spectrum fast-forward & workflow advice


def ernie_lora_key_map(sd: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """Map ERNIE-Image LoRA keys from all common trainers onto ``diffusion_model.*``."""

    key_map: dict[str, object] = {}

    def register(bare: str, target: str) -> None:
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
    return key_map


def fast_forward_ernie(dit, predictor, sigma, input_x, feature):
    """Spectrum pre-head fast-forward for ``ErnieImageModel``.

    Uses ``dit.final_norm`` as the zero-intrusion pre-head capture seam:
    recomputes timestep embedding ``c``, applies ``final_norm`` + ``final_linear``,
    slices image tokens ``:Hp*Wp``, unpatchifies, and converts via ``predictor``.
    """

    B, _, H, W = input_x.shape
    p = int(dit.patch_size)
    Hp, Wp = H // p, W // p
    N_img = Hp * Wp

    internal_timestep = predictor.timestep(sigma).float()
    if internal_timestep.ndim == 0:
        internal_timestep = internal_timestep.unsqueeze(0)
    if internal_timestep.shape[0] == 1 and B > 1:
        internal_timestep = internal_timestep.expand(B)

    sample = dit.time_proj(internal_timestep).to(feature.dtype)
    c = dit.time_embedding(sample)
    hidden = dit.final_norm(feature, c).type_as(feature)
    patches = dit.final_linear(hidden)[:, :N_img, :]
    model_output = patches.view(B, Hp, Wp, p, p, dit.out_channels).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, dit.out_channels, H, W)
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


def ernie_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking workflow hints for ERNIE-Image (Base vs Turbo)."""

    sd_model = getattr(p, "sd_model", None)
    if sd_model is None or not (getattr(sd_model, "is_ernie_engine", False) or type(sd_model).__name__ == "ErnieImage"):
        return []

    from backend.nn.dit_registry import _prompt_and_checkpoint_names
    from remi.ui.sentence import Sentence

    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    combined = " ".join([ckpt_name, *lora_names]).strip()
    is_turbo = bool(combined and _ERNIE_TURBO_RE.search(combined))

    advice: list[str] = []
    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if is_turbo:
        if (isinstance(steps, (int, float)) and steps >= 16) or (isinstance(cfg, (int, float)) and cfg > 1.5):
            advice.append(
                Sentence("ERNIE-Image Turbo detected with steps={{steps}}, cfg={{cfg}}: the distilled Turbo variant works best at ~8 steps with CFG 1.0.", steps=steps, cfg=cfg)
            )
    elif isinstance(steps, (int, float)) and steps <= 12 and isinstance(cfg, (int, float)) and cfg <= 1.2:
        advice.append(
            Sentence("ERNIE-Image Base (8B Apache-2.0) detected with steps={{steps}}, cfg={{cfg}}: undistilled Base typically needs ~28-50 steps with CFG 3.5-4.5.", steps=steps, cfg=cfg)
        )
    return advice


# endregion
