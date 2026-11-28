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
import re
from collections.abc import Mapping

import torch

from backend.nn.anima_diffusers import (
    _DIFFUSERS_INDEX_FILES,
    _SHARD_INDEX_FILES,
    _SINGLE_WEIGHT_FILES,
    _load_component_sd,
    _read_json,
)
from backend.nn.dit_registry import (  # noqa: F401 -- backward-compat re-export
    diffusers_checkpoint_info,
    is_diffusers_checkpoint_dir,
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
    if os.path.isdir(transformer_dir) and (primary := _dir_primary_safetensors(transformer_dir)):
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


# region Kromoss & 8-step DMD/DMD2 detection and LoRA key mapping

_KROMOSS_RE = re.compile(r"(?:^|[-_ /])(?:kromoss|moss[-_ ]?image[-_ ]?sft[-_ ]?8[-_ ]?steps?[-_ ]?dmd2?)(?:$|[-_ /.])", re.IGNORECASE)
_DMD2_RE = re.compile(r"(?:^|[-_ /])(?:dmd2?|8[-_]?steps?[-_ ]?dmd2?|turbo)(?:$|[-_ /.])", re.IGNORECASE)


def is_kromoss_variant(name: str = "", metadata: Mapping | None = None) -> bool:
    """True when a checkpoint name or safetensors metadata identifies a
    ``lodestones/Kromoss`` or 8-step MOSS-Image DMD/DMD2 variant."""

    if name and _KROMOSS_RE.search(name):
        return True
    if metadata:
        joined = " ".join(str(v) for v in metadata.values() if isinstance(v, (str, int, float)))
        if joined and _KROMOSS_RE.search(joined):
            return True
    return False


def is_dmd2_lora(name: str = "", state_dict: Mapping | None = None) -> bool:
    """True when a LoRA filename or state dict matches an 8-step MOSS/Kromoss
    DMD/DMD2 distillation adapter."""

    if name and _DMD2_RE.search(name):
        if state_dict is None:
            return True
        # Confirm MOSS-Image layer structure if tensors are supplied
        if any(".proj." in k or "shared_block_modulation" in k for k in state_dict):
            return True
    return False


def moss_lora_key_map(sd: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """Comprehensive bidirectional LoRA/LoKr key map for MOSS-Image & Kromoss
    (including 8-step DMD/DMD2 LoRAs from ComfyUI, Diffusers, PEFT, ai-toolkit,
    kohya, LyCORIS, and OneTrainer).

    Bidirectionally bridges:
    - fused ``layers.{i}.proj.qkv`` <-> split ``layers.{i}.proj.{q,k,v}``;
    - fused ``layers.{i}.proj.ffn_up_gate`` <-> split ``ffn_up`` / ``ffn_gate``
      (emitted by some DMD2 / Diffusers trainers).
    """

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

        if bare.endswith(".proj.qkv"):
            third = int(sd[k].shape[0]) // 3
            stem = bare[: -len(".qkv")]
            for i, proj in enumerate(("q", "k", "v")):
                register(f"{stem}.{proj}", (k, (0, i * third, third)))
        elif bare.endswith(".proj.q"):
            k_key = k[: -len(".q.weight")] + ".k.weight"
            v_key = k[: -len(".q.weight")] + ".v.weight"
            if k_key in sd and v_key in sd:
                qd, kd, vd = int(sd[k].shape[0]), int(sd[k_key].shape[0]), int(sd[v_key].shape[0])
                stem = bare[: -len(".q")]
                register(
                    f"{stem}.qkv",
                    SplitTargets(
                        targets=((k, 0, qd), (k_key, qd, kd), (v_key, qd + kd, vd)),
                        slice_dim=0,
                        full_dim=qd + kd + vd,
                    ),
                )
        elif bare.endswith(".proj.ffn_up_gate"):
            half = int(sd[k].shape[0]) // 2
            stem = bare[: -len(".ffn_up_gate")]
            register(f"{stem}.ffn_up", (k, (0, 0, half)))
            register(f"{stem}.ffn_gate", (k, (0, half, half)))

    return key_map


def kromoss_dmd2_workflow_advice(p, *, opts=None) -> list[str]:
    """Targeted workflow hints for Kromoss / MOSS-Image 8-step DMD/DMD2 runs."""

    from backend.nn.dit_registry import _prompt_and_checkpoint_names

    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    is_kromoss = is_kromoss_variant(ckpt_name)
    has_dmd2 = any(is_dmd2_lora(name) for name in lora_names) or bool(ckpt_name and _DMD2_RE.search(ckpt_name))
    if not (is_kromoss or has_dmd2):
        return []

    advice: list[str] = []
    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if (isinstance(steps, (int, float)) and steps > 12) or (isinstance(cfg, (int, float)) and cfg > 1.5):
        label = "Kromoss" if is_kromoss else "MOSS-Image DMD2"
        advice.append(
            f"{label} 8-step DMD/DMD2 detected with steps={steps}, cfg={cfg}: "
            "switch to 8 steps and CFG 1.0 for optimal speed and typography fidelity."
        )
    return advice


# endregion
