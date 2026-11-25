"""HuggingFace Diffusers, Unsloth FP8/INT8-ConvRot, LoRA & Spectrum helpers for Z-Image.

Covers the full Z-Image family (6B Apache-2.0):
- ``Tongyi-MAI/Z-Image`` (Base), ``Tongyi-MAI/Z-Image-Turbo`` (8-step distilled),
  ``Tongyi-MAI/Z-Image-Omni`` / ``Z-Image-Edit`` (reference conditioning);
- ``unsloth/Z-Image-Turbo-FP8`` and ``unsloth/Z-Image-Turbo-INT8-ConvRot``
  Diffusers / TorchAO checkpoints;
- Single-file Diffusers ``ZImageTransformer2DModel`` conversion into Forge's
  native ``NextDiT`` layout (preserving quantized sidecars ``.weight_scale`` /
  ``.comfy_quant``);
- Bidirectional LoRA key mapping across fused ``attention.qkv`` and split
  ``attention.to_q`` / ``to_k`` / ``to_v`` (plus ``all_x_embedder.2-1`` /
  ``all_final_layer.2-1`` and every common trainer prefix);
- Spectrum pre-head fast-forward & non-blocking workflow advice.
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

DIFFUSERS_CLASS = "ZImageTransformer2DModel"
DIFFUSERS_PIPELINE_CLASSES = ("ZImagePipeline", "ZImageTurboPipeline", "ZImageOmniPipeline", "ZImageEditPipeline")

ZIMAGE_SENSITIVE_PREFIXES: tuple[str, ...] = (
    "x_embedder.",
    "t_embedder.",
    "cap_embedder.",
    "final_layer.",
    "clip_text_pooled_proj.",
    "time_text_embed.",
    "x_pad_token",
    "cap_pad_token",
)

ZIMAGE_INT8_LINEAR_SUFFIXES: tuple[str, ...] = (
    ".attention.qkv.weight",
    ".attention.out.weight",
    ".feed_forward.w1.weight",
    ".feed_forward.w2.weight",
    ".feed_forward.w3.weight",
)

_ZIMAGE_TURBO_RE = re.compile(r"(?:^|[-_ /])(?:turbo|distill(?:ed)?|8[-_]?steps?|4[-_]?steps?)(?:$|[-_ /.])", re.IGNORECASE)
_ZIMAGE_EDIT_RE = re.compile(r"(?:^|[-_ /])(?:omni|edit)(?:$|[-_ /.])", re.IGNORECASE)


def _dir_primary(directory: str) -> str | None:
    from backend.nn.moss_diffusers import _dir_primary_safetensors

    return _dir_primary_safetensors(directory)


def _strip_prefix(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_zimage_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """True when ``state_dict`` matches a Z-Image ``NextDiT`` checkpoint (native
    or Diffusers keys, distinguished from Lumina2 by ``z_image_modulation``
    256-dim AdaLN or 3840 hidden dim)."""

    keys = {_strip_prefix(k): v for k, v in state_dict.items()}
    has_nextdit = (
        ("cap_embedder.1.weight" in keys or "cap_embedder.0.weight" in keys)
        and ("noise_refiner.0.attention.qkv.weight" in keys or "noise_refiner.0.attention.to_q.weight" in keys or "layers.0.attention.qkv.weight" in keys or "layers.0.attention.to_q.weight" in keys)
    )
    if not has_nextdit:
        return False
    # Z-Image uses 256-dim adaLN_modulation.0 (vs Lumina2's Sequential(SiLU, Linear(1024, ...)) at .1)
    if "layers.0.adaLN_modulation.0.weight" in keys or "all_x_embedder.2-1.weight" in keys:
        return True
    cap_w = keys.get("cap_embedder.1.weight")
    if isinstance(cap_w, torch.Tensor) and cap_w.ndim == 2 and int(cap_w.shape[0]) == 3840:
        return True
    return False


def is_diffusers_zimage_transformer(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """True for Diffusers ``ZImageTransformer2DModel`` single-file state dicts."""

    keys = {_strip_prefix(k) for k in state_dict}
    has_split_attn = any(k.startswith("layers.0.attention.to_q.") for k in keys) or any(k.startswith("noise_refiner.0.attention.to_q.") for k in keys)
    has_diffusers_embed = "all_x_embedder.2-1.weight" in keys or "all_final_layer.2-1.linear.weight" in keys
    return has_split_attn or has_diffusers_embed


def convert_diffusers_zimage_transformer(state_dict: Mapping[str, torch.Tensor], output_prefix: str = "") -> dict[str, torch.Tensor]:
    """Convert Diffusers / Unsloth ``ZImageTransformer2DModel`` keys (including
    quantized sidecars such as ``.weight_scale`` and ``.comfy_quant``) into
    Forge's native ``NextDiT`` state dict layout."""

    raw = {_strip_prefix(k): v for k, v in state_dict.items()}
    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    for key, val in raw.items():
        if key in consumed:
            continue

        if ".attention.to_q." in key:
            suffix = key.split(".attention.to_q.", 1)[1]
            k_key = key.replace(".attention.to_q.", ".attention.to_k.")
            v_key = key.replace(".attention.to_q.", ".attention.to_v.")
            if k_key in raw and v_key in raw:
                consumed.add(k_key)
                consumed.add(v_key)
                dst = key.replace(".attention.to_q.", ".attention.qkv.")
                if suffix == "comfy_quant":
                    out[f"{output_prefix}{dst}"] = val
                elif val.ndim >= 1 and val.shape[1:] == raw[k_key].shape[1:] == raw[v_key].shape[1:]:
                    out[f"{output_prefix}{dst}"] = torch.cat([val, raw[k_key], raw[v_key]], dim=0)
                else:
                    # Per-tensor scalar scales: keep max scale safely or preserve tensor
                    out[f"{output_prefix}{dst}"] = torch.maximum(torch.maximum(val, raw[k_key]), raw[v_key])
                continue
        if ".attention.to_k." in key or ".attention.to_v." in key:
            q_key = key.replace(".attention.to_k.", ".attention.to_q.").replace(".attention.to_v.", ".attention.to_q.")
            if q_key in raw:
                continue

        mapped = key
        if mapped.startswith("all_x_embedder.2-1."):
            mapped = "x_embedder." + mapped[len("all_x_embedder.2-1.") :]
        elif mapped.startswith("all_final_layer.2-1."):
            mapped = "final_layer." + mapped[len("all_final_layer.2-1.") :]
        elif ".attention.to_out.0." in mapped:
            mapped = mapped.replace(".attention.to_out.0.", ".attention.out.")
        elif ".attention.norm_q." in mapped:
            mapped = mapped.replace(".attention.norm_q.", ".attention.q_norm.")
        elif ".attention.norm_k." in mapped:
            mapped = mapped.replace(".attention.norm_k.", ".attention.k_norm.")

        out[f"{output_prefix}{mapped}"] = val

    return out


# region Diffusers & sharded directory loaders


def zimage_diffusers_primary_file(directory: os.PathLike | str) -> str | None:
    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    if os.path.isdir(transformer_dir) and (primary := _dir_primary(transformer_dir)):
        return primary
    return _dir_primary(root)


def is_zimage_diffusers_dir(path: os.PathLike | str) -> bool:
    """True for a local Z-Image / Z-Image-Turbo / Z-Image-Omni Diffusers folder."""

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
        if any(p in cls_name for p in DIFFUSERS_PIPELINE_CLASSES) or "ZImage" in cls_name:
            return True
        if isinstance(entry, list) and DIFFUSERS_CLASS in entry:
            return True
    cfg = _read_json(os.path.join(transformer_dir, "config.json"))
    return bool(cfg and cfg.get("_class_name") == DIFFUSERS_CLASS)


def is_sharded_zimage_transformer_dir(path: os.PathLike | str) -> bool:
    """True for a standalone sharded Z-Image transformer directory."""

    root = os.fspath(path)
    if not os.path.isdir(root) or os.path.isdir(os.path.join(root, "transformer")) or _dir_primary(root) is None:
        return False
    cfg = _read_json(os.path.join(root, "config.json"))
    if cfg and cfg.get("_class_name") == DIFFUSERS_CLASS:
        return True
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(root, name))
        weight_map = index.get("weight_map") if index else None
        if not isinstance(weight_map, dict):
            continue
        if "all_x_embedder.2-1.weight" in weight_map or ("cap_embedder.1.weight" in weight_map and "layers.0.adaLN_modulation.0.weight" in weight_map):
            return True
    return False


def is_zimage_checkpoint_dir(path: os.PathLike | str) -> bool:
    return is_zimage_diffusers_dir(path) or is_sharded_zimage_transformer_dir(path)


def load_zimage_diffusers_dir(directory: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a Z-Image Diffusers pipeline directory or standalone sharded
    transformer directory into Forge's unified checkpoint state dict."""

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    is_pipeline = os.path.isdir(transformer_dir)
    if not is_pipeline:
        transformer_dir = root

    raw, metadata = _load_component_sd(transformer_dir)
    if not raw:
        raise FileNotFoundError(f"Z-Image Diffusers directory has no transformer weights: {transformer_dir}")

    if is_diffusers_zimage_transformer(raw):
        sd = convert_diffusers_zimage_transformer(raw, output_prefix="model.diffusion_model.")
    else:
        sd = {(k if k.startswith("model.diffusion_model.") else f"model.diffusion_model.{k}"): v for k, v in raw.items()}
    del raw

    if not is_pipeline:
        return sd, metadata

    metadata = dict(metadata or {})
    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(root, index_name))
        if index:
            cls_name = str(index.get("_class_name", ""))
            if "Turbo" in cls_name:
                metadata["zimage_variant"] = "turbo"
            elif "Omni" in cls_name or "Edit" in cls_name:
                metadata["zimage_variant"] = "omni"
            break

    te_dir = os.path.join(root, "text_encoder")
    if os.path.isdir(te_dir):
        te_sd, _ = _load_component_sd(te_dir)
        for key, value in te_sd.items():
            norm_k = key if key.startswith("model.") else f"model.{key}"
            sd[f"text_encoders.qwen3_4b.transformer.{norm_k}"] = value
        del te_sd

    vae_dir = os.path.join(root, "vae")
    if os.path.isdir(vae_dir):
        vae_sd, _ = _load_component_sd(vae_dir)
        if "decoder.up_blocks.0.resnets.0.norm1.weight" in vae_sd:
            from modules_forge.packages.huggingface_guess.diffusers_convert import convert_vae_state_dict

            vae_sd = convert_vae_state_dict(vae_sd)
        for key, value in vae_sd.items():
            sd[f"vae.{key}"] = value
        del vae_sd

    return sd, metadata


# endregion


# region Bidirectional LoRA key mapping


def zimage_lora_key_map(sd: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """Bidirectional LoRA key map for Z-Image across ComfyUI, Diffusers, PEFT,
    ai-toolkit, kohya, LyCORIS, and OneTrainer — bridging fused ``attention.qkv``
    and split ``attention.to_q`` / ``to_k`` / ``to_v`` as well as
    ``all_x_embedder.2-1`` / ``all_final_layer.2-1`` aliases."""

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
        native = k[len("diffusion_model.") : -len(".weight")]
        register(native, k)

        if native == "x_embedder":
            register("all_x_embedder.2-1", k)
        elif native.startswith("final_layer."):
            register("all_final_layer.2-1." + native[len("final_layer.") :], k)
        elif native.endswith(".attention.out"):
            register(native[: -len(".out")] + ".to_out.0", k)
        elif native.endswith(".attention.q_norm"):
            register(native[: -len(".q_norm")] + ".norm_q", k)
        elif native.endswith(".attention.k_norm"):
            register(native[: -len(".k_norm")] + ".norm_k", k)
        elif native.endswith(".attention.qkv"):
            # Fused NextDiT checkpoint: map split Diffusers to_q/to_k/to_v LoRAs onto thirds
            third = int(sd[k].shape[0]) // 3
            stem = native[: -len(".qkv")]
            for i, name in enumerate(("to_q", "to_k", "to_v")):
                register(f"{stem}.{name}", (k, (0, i * third, third)))
        elif native.endswith(".attention.to_q"):
            # Split checkpoint: map fused qkv LoRAs via SplitTargets
            k_key = k[: -len(".to_q.weight")] + ".to_k.weight"
            v_key = k[: -len(".to_q.weight")] + ".to_v.weight"
            if k_key in sd and v_key in sd:
                qd, kd, vd = int(sd[k].shape[0]), int(sd[k_key].shape[0]), int(sd[v_key].shape[0])
                stem = native[: -len(".to_q")]
                register(
                    f"{stem}.qkv",
                    SplitTargets(
                        targets=((k, 0, qd), (k_key, qd, kd), (v_key, qd + kd, vd)),
                        slice_dim=0,
                        full_dim=qd + kd + vd,
                    ),
                )

    return key_map


# endregion


# region Spectrum fast-forward & workflow advice


def fast_forward_zimage(dit, predictor, sigma, input_x, feature):
    """Spectrum pre-head fast-forward for Z-Image (``NextDiT``)."""

    internal_timestep = predictor.timestep(sigma).float()
    model_output = dit.compute_zimage_final(feature, internal_timestep, tuple(input_x.shape))
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


def zimage_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking workflow advice for Z-Image (Base vs Turbo vs Omni/Edit)."""

    sd_model = getattr(p, "sd_model", None)
    if sd_model is None or not (getattr(sd_model, "is_zimage_engine", False) or type(sd_model).__name__ == "ZImage"):
        return []

    from backend.nn.dit_registry import _prompt_and_checkpoint_names
    from remi.ui.sentence import Sentence

    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    combined = " ".join([ckpt_name, *lora_names]).strip()
    is_turbo = bool(getattr(sd_model, "is_zimage_turbo", False)) or bool(combined and _ZIMAGE_TURBO_RE.search(combined))
    is_omni = bool(getattr(sd_model, "is_zimage_omni", False)) or bool(combined and _ZIMAGE_EDIT_RE.search(combined))

    advice: list[str] = []
    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if is_turbo:
        if (isinstance(steps, (int, float)) and steps >= 16) or (isinstance(cfg, (int, float)) and cfg > 1.5):
            advice.append(
                Sentence("Z-Image-Turbo detected with steps={{steps}}, cfg={{cfg}}: the 8-step distilled Turbo variant works best at ~8 steps with CFG 1.0.", steps=steps, cfg=cfg)
            )
    else:
        if isinstance(steps, (int, float)) and steps <= 10 and isinstance(cfg, (int, float)) and cfg <= 1.2:
            advice.append(
                Sentence("Z-Image Base detected with steps={{steps}}, cfg={{cfg}}: undistilled Z-Image Base typically needs ~25-40 steps and CFG 3.5-5.0 (or use Z-Image-Turbo for 8-step CFG 1.0).", steps=steps, cfg=cfg)
            )

    refs = [*(getattr(sd_model, "ref_latents", None) or ())]
    if getattr(sd_model, "ini_latent", None) is not None:
        refs.append(sd_model.ini_latent)
    if refs and not is_omni and not bool(getattr(sd_model, "zimage_do_reference", True)):
        advice.append("Z-Image: reference images are attached while Z-Image Omni/Edit reference mode is disabled.")
    return advice


# endregion
