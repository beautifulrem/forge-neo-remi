"""Unified DiT ControlNet-Union ModelPatch abstraction.

Inspired by ``anima_vace.py`` and ComfyUI's ``control_blocks`` ModelPatches,
this module provides:

- :class:`DiTControlPatch` — abstract forward-hook control injector for DiT
  block stacks with per-call hint isolation (keyed by ``id(transformer_options)``)
  and safe Prefix KV Cache invalidation/bypass;
- :class:`Qwen21FunControlNet` — ``Qwen-Image-2.1-Fun-Controlnet-Union``
  (16 even-layer zero-gated ``control_blocks`` on 32 base layers,
  ``control_in_dim=129`` = 64 control + 1 mask + 64 masked latent, plus
  64-channel structural-only variant);
- :class:`ZImageFunControlNet` — ``Z-Image-Turbo-Fun-Controlnet-Union``
  (spaced zero-gated ``control_blocks`` on ``NextDiT.layers``, supporting both
  native ``NextDiT`` and Diffusers weight naming, 16-ch structural and 33-ch
  control+mask+masked-latent inpaint inputs);
- :class:`FunControlNetUnionPatch` — unified :class:`DiTControlPatch` runtime
  for both Qwen-Image-2.1 and Z-Image Fun ControlNet-Union adapters;
- :class:`DiTControlNetUnionPatcher` — Forge ``ControlModelPatcher`` bridge
  auto-detecting ``control_blocks.*`` + ``control_img_in.*`` weights;
- :func:`detect_dit_control_family` / :func:`active_control_family` — key-only
  family classifier used by ControlNet UI model filtering.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import memory_management
from backend.args import dynamic_args
from backend.nn.lumina import JointTransformerBlock
from backend.nn.qwen21 import QwenImage21TransformerBlock
from backend.operations import using_forge_operations
from backend.utils import load_torch_file, pad_to_patch_size

logger = logging.getLogger("ControlNet")

_CONTROL_PREFIXES = ("control_model.", "diffusion_model.", "model.", "net.")

FUN_UNION_CONTROL_TYPES: tuple[str, ...] = (
    "canny",
    "hed",
    "depth",
    "pose",
    "mlsd",
    "scribble",
    "inpaint",
    "tile",
)


# region Prefix stripping & family auto-detection


def strip_control_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Strip an optional wrapper prefix (``control_model.`` / ``diffusion_model.``)."""

    if any(k.startswith(("control_blocks.", "control_img_in.", "control_all_x_embedder.")) for k in state_dict):
        return dict(state_dict)
    for prefix in _CONTROL_PREFIXES:
        if any(k.startswith((f"{prefix}control_blocks.", f"{prefix}control_img_in.", f"{prefix}control_all_x_embedder.")) for k in state_dict):
            return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return dict(state_dict)


def _strip_key_prefix(key: str) -> str:
    for prefix in _CONTROL_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def detect_dit_control_family(state_dict_or_keys: Mapping[str, Any] | Iterable[str]) -> str | None:
    """Classify a control checkpoint by tensor keys (without loading weights).

    Returns ``"qwen21"``, ``"zimage"``, ``"anima"``, or ``None`` (standard UNet).
    """

    if isinstance(state_dict_or_keys, Mapping):
        raw_keys = list(state_dict_or_keys.keys())
        tensors = state_dict_or_keys
    else:
        raw_keys = list(state_dict_or_keys)
        tensors = None

    keys = [_strip_key_prefix(k) for k in raw_keys]

    # Anima LLLite / Control-LoRA / VACE
    if any(
        k.startswith("lllite_dit")
        or k.endswith("control_embedder.proj.weight")
        or k == "control_embedder.proj.1.weight"
        or k.startswith("control_blocks.0.block.self_attn.")
        for k in keys
    ):
        return "anima"

    has_control_blocks = any(k.startswith("control_blocks.") for k in keys)
    if not has_control_blocks:
        return None

    # Qwen-Image-2.1 Fun ControlNet-Union signatures
    has_qwen_block = any(
        ".img_mlp." in k
        or ".img_norm1." in k
        or (".attn.to_q." in k and k.startswith("control_blocks."))
        or (".attn.norm_q." in k and k.startswith("control_blocks."))
        for k in keys
    )
    if has_qwen_block:
        return "qwen21"

    # Z-Image-Turbo Fun ControlNet-Union signatures (native NextDiT or Diffusers)
    has_zimage_block = any(
        ".feed_forward." in k
        or ".attention.qkv." in k
        or ".attention.to_q." in k
        or ".attention_norm1." in k
        or ".adaLN_modulation." in k
        or k.startswith(("control_all_x_embedder.", "control_x_embedder."))
        for k in keys
    )
    if has_zimage_block:
        return "zimage"

    # Fallback on control_img_in shape when tensors are available
    if tensors is not None:
        sd = strip_control_prefix(tensors)
        w = sd.get("control_img_in.weight")
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            if int(w.shape[1]) in (64, 129):
                return "qwen21"
            if int(w.shape[1]) in (16, 33, 64, 132):
                return "zimage"

    return None


def is_fun_controlnet_union(state_dict: Mapping[str, Any]) -> bool:
    """True for Qwen-Image-2.1 or Z-Image Fun ControlNet-Union checkpoints."""

    return detect_dit_control_family(state_dict) in ("qwen21", "zimage")


def active_control_family() -> str:
    """Return the active loaded model's control family (``"anima"``, ``"qwen21"``,
    ``"zimage"``, or ``"unet"``)."""

    if bool(getattr(dynamic_args, "anima", False)):
        return "anima"
    if bool(getattr(dynamic_args, "qwen21", False)):
        return "qwen21"
    if bool(getattr(dynamic_args, "zimage", False)):
        return "zimage"
    try:
        import sys

        sd_models = sys.modules.get("modules.sd_models")
        sd_model = getattr(getattr(sd_models, "model_data", None), "sd_model", None) if sd_models else None
        if sd_model is not None:
            if getattr(sd_model, "is_anima_engine", False):
                return "anima"
            if getattr(sd_model, "is_qwen21_engine", False):
                return "qwen21"
            if getattr(sd_model, "is_zimage_engine", False) or type(sd_model).__name__ == "ZImage":
                return "zimage"
    except Exception:
        pass
    return "unet"


# endregion


# region Zero-gated control blocks & Fun ControlNet-Union modules


class ZeroGatedControlBlock(nn.Module):
    """Wraps a DiT block with optional zero-initialized ``before_proj`` (block 0)
    and zero-initialized ``after_proj`` residual output gate."""

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        block_id: int = 0,
        *,
        use_before_proj: bool = True,
        use_after_proj: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.block = block
        self.block_id = int(block_id)
        self.dim = int(dim)
        self.use_before_proj = bool(use_before_proj and block_id == 0)
        self.use_after_proj = bool(use_after_proj)
        if self.use_before_proj:
            self.before_proj = nn.Linear(dim, dim, bias=bias)
        if self.use_after_proj:
            self.after_proj = nn.Linear(dim, dim, bias=bias)

    def init_zero_weights(self) -> None:
        with torch.no_grad():
            if self.use_before_proj:
                self.before_proj.weight.zero_()
                if self.before_proj.bias is not None:
                    self.before_proj.bias.zero_()
            if self.use_after_proj:
                self.after_proj.weight.zero_()
                if self.after_proj.bias is not None:
                    self.after_proj.bias.zero_()

    def forward(self, c: torch.Tensor, x_base: torch.Tensor, *block_args, **block_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_before_proj:
            c = self.before_proj(c) + x_base
        c = self.block(c, *block_args, **block_kwargs)
        hint = self.after_proj(c) if self.use_after_proj else c
        return hint, c


def _control_block_indices(sd: Mapping[str, Any]) -> list[int]:
    return sorted({int(k.split(".")[1]) for k in sd if k.startswith("control_blocks.") and len(k.split(".")) > 2 and k.split(".")[1].isdigit()})


def _normalize_control_block_sd(sd: Mapping[str, torch.Tensor], *, nested_block_key: str) -> dict[str, torch.Tensor]:
    """Accept both ``control_blocks.{i}.block.<sub...>`` and flat ``control_blocks.{i}.<sub...>``."""

    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith("control_blocks."):
            out[k] = v
            continue
        parts = k.split(".", 2)
        if len(parts) < 3 or not parts[1].isdigit():
            out[k] = v
            continue
        idx, tail = parts[1], parts[2]
        if tail.startswith(("before_proj.", "after_proj.", "block.")):
            out[k] = v
        elif tail.startswith(nested_block_key):
            out[f"control_blocks.{idx}.block.{tail}"] = v
        else:
            out[f"control_blocks.{idx}.block.{tail}"] = v
    return out


class Qwen21FunControlNet(nn.Module):
    """``Qwen-Image-2.1-Fun-Controlnet-Union`` branch.

    Default architecture: 16 even-layer zero-gated ``control_blocks`` targeting
    layers ``[0, 2, 4, ..., 30]`` of 32-layer Qwen-Image-2.1, with
    ``control_in_dim=129`` (64 control latent + 1 mask + 64 masked image latent).
    Also supports 64-channel structural-only checkpoints and custom layer counts.
    """

    family = "qwen21"

    def __init__(
        self,
        *,
        inner_dim: int = 4096,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        mlp_ratio: int = 3,
        fused_mlp: bool = False,
        control_in_dim: int = 129,
        latent_channels: int = 64,
        num_base_layers: int = 32,
        control_layers: list[int] | None = None,
        use_before_proj: bool = True,
        use_after_proj: bool = True,
        proj_bias: bool = False,
        control_context_scale: float = 1.0,
        eps: float = 1e-6,
        _device: torch.device | None = None,
        _dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.inner_dim = int(inner_dim)
        self.num_attention_heads = int(num_attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.control_in_dim = int(control_in_dim)
        self.latent_channels = int(latent_channels)
        self.num_base_layers = int(num_base_layers)
        self.control_context_scale = float(control_context_scale)
        if control_layers is None:
            self.control_layers = list(range(0, self.num_base_layers, 2))
        else:
            self.control_layers = [int(i) for i in control_layers]

        with using_forge_operations(
            device=_device if _device is not None else torch.device("cpu"),
            dtype=_dtype if _dtype is not None else torch.float32,
            manual_cast_enabled=True,
        ):
            self.control_img_in = nn.Linear(self.control_in_dim, self.inner_dim, bias=False)
            self.control_blocks = nn.ModuleList(
                [
                    ZeroGatedControlBlock(
                        QwenImage21TransformerBlock(
                            self.inner_dim,
                            self.num_attention_heads,
                            self.attention_head_dim,
                            mlp_ratio=mlp_ratio,
                            eps=eps,
                            fused_mlp=fused_mlp,
                        ),
                        dim=self.inner_dim,
                        block_id=i,
                        use_before_proj=use_before_proj,
                        use_after_proj=use_after_proj,
                        bias=proj_bias,
                    )
                    for i in range(len(self.control_layers))
                ]
            )

    def init_zero_weights(self) -> None:
        for cb in self.control_blocks:
            cb.init_zero_weights()

    @staticmethod
    def infer_config(state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        sd = _normalize_control_block_sd(strip_control_prefix(state_dict), nested_block_key="attn.")
        meta = metadata or {}

        img_in_w = sd.get("control_img_in.weight")
        if img_in_w is None:
            raise ValueError("Qwen21FunControlNet checkpoint is missing control_img_in.weight")
        inner_dim = int(img_in_w.shape[0])
        control_in_dim = int(img_in_w.shape[1])

        norm_q = sd.get("control_blocks.0.block.attn.norm_q.weight")
        head_dim = int(norm_q.shape[0]) if norm_q is not None else 128
        num_heads = max(1, inner_dim // head_dim)

        fused_mlp = "control_blocks.0.block.img_mlp.gate_up.weight" in sd
        if fused_mlp:
            mlp_ratio = max(1, int(sd["control_blocks.0.block.img_mlp.gate_up.weight"].shape[0]) // (2 * inner_dim))
        elif (proj_w := sd.get("control_blocks.0.block.img_mlp.proj.weight")) is not None:
            mlp_ratio = max(1, int(proj_w.shape[0]) // inner_dim)
        else:
            mlp_ratio = 3

        ctrl_indices = _control_block_indices(sd)
        if not ctrl_indices:
            raise ValueError("Qwen21FunControlNet checkpoint contains no control_blocks")
        n_ctrl = len(ctrl_indices)
        num_base_layers = int(meta.get("control_num_base_layers", max(32, n_ctrl * 2) if n_ctrl == 16 else n_ctrl * 2))
        if "control_layers" in meta:
            raw_layers = meta["control_layers"]
            if isinstance(raw_layers, str):
                control_layers = [int(x.strip()) for x in raw_layers.strip("[]").split(",") if x.strip()]
            else:
                control_layers = [int(x) for x in raw_layers]
        else:
            # Even layers [0, 2, 4, ..., 30] on 32-layer Qwen-Image-2.1
            step = max(1, num_base_layers // n_ctrl)
            control_layers = [i * step for i in range(n_ctrl)]

        latent_channels = (control_in_dim - 1) // 2 if control_in_dim % 2 == 1 and control_in_dim > 1 else control_in_dim
        use_before_proj = "control_blocks.0.before_proj.weight" in sd
        use_after_proj = "control_blocks.0.after_proj.weight" in sd
        proj_bias = "control_blocks.0.after_proj.bias" in sd or "control_blocks.0.before_proj.bias" in sd
        scale = float(meta.get("control_context_scale", 1.0))

        return {
            "inner_dim": inner_dim,
            "num_attention_heads": num_heads,
            "attention_head_dim": head_dim,
            "mlp_ratio": mlp_ratio,
            "fused_mlp": fused_mlp,
            "control_in_dim": control_in_dim,
            "latent_channels": latent_channels,
            "num_base_layers": num_base_layers,
            "control_layers": control_layers,
            "use_before_proj": use_before_proj,
            "use_after_proj": use_after_proj,
            "proj_bias": proj_bias,
            "control_context_scale": scale,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any] | None = None) -> "Qwen21FunControlNet":
        sd = _normalize_control_block_sd(strip_control_prefix(state_dict), nested_block_key="attn.")
        cfg = cls.infer_config(sd, metadata)
        dtype = next((v.dtype for v in sd.values() if isinstance(v, torch.Tensor) and v.is_floating_point()), torch.float32)
        model = cls(**cfg, _device=torch.device("meta"), _dtype=dtype)
        model.load_state_dict(sd, strict=True, assign=True)
        return model.eval().requires_grad_(False)

    def build_control_input(
        self,
        control_latent: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        masked_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the ``(B, control_in_dim, H, W)`` control tensor.

        For ``control_in_dim == 129`` (``2 * latent_channels + 1``), concatenates
        ``[control_latent (64), mask (1), masked_latent (64)]``.
        """

        b, c, h, w = control_latent.shape
        if c == self.control_in_dim:
            return control_latent
        if self.control_in_dim == 2 * self.latent_channels + 1:
            if c != self.latent_channels:
                raise ValueError(f"Expected control_latent with {self.latent_channels} channels, got {c}")
            if mask is None:
                m = torch.zeros((b, 1, h, w), device=control_latent.device, dtype=control_latent.dtype)
            else:
                m = mask
                if m.ndim == 2:
                    m = m.unsqueeze(0).unsqueeze(0)
                elif m.ndim == 3:
                    m = m.unsqueeze(1)
                if m.shape[-2:] != (h, w):
                    m = F.interpolate(m.float(), size=(h, w), mode="bilinear", align_corners=False)
                if m.shape[0] != b:
                    m = m[:1].expand(b, -1, -1, -1)
                m = m.to(device=control_latent.device, dtype=control_latent.dtype)
            if masked_latent is None:
                ml = torch.zeros_like(control_latent)
            else:
                ml = masked_latent
                if ml.shape[-2:] != (h, w):
                    ml = F.interpolate(ml.float(), size=(h, w), mode="bilinear", align_corners=False)
                if ml.shape[0] != b:
                    ml = ml[:1].expand(b, -1, -1, -1)
                ml = ml.to(device=control_latent.device, dtype=control_latent.dtype)
            return torch.cat([control_latent, m, ml], dim=1)
        raise ValueError(f"Control latent channels {c} incompatible with control_in_dim={self.control_in_dim}")

    def target_layers_for_dit(self, num_dit_layers: int) -> list[int]:
        if all(0 <= idx < num_dit_layers for idx in self.control_layers):
            return list(self.control_layers)
        n_ctrl = len(self.control_layers)
        step = max(1, num_dit_layers // max(1, n_ctrl))
        return [min(num_dit_layers - 1, i * step) for i in range(n_ctrl)]

    def forward(
        self,
        control_input: torch.Tensor,
        x_target_tokens: torch.Tensor,
        mod: tuple,
        pe_target: torch.Tensor,
        *,
        target_layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run the 16 even-layer control blocks and return ``{base_layer_idx: hint}``."""

        param = next(self.parameters())
        device, dtype = param.device, param.dtype

        ctrl_tokens = self.control_img_in(control_input.flatten(2).transpose(1, 2).to(device=device, dtype=dtype))
        x_base = x_target_tokens.to(device=device, dtype=dtype)
        pe = pe_target.to(device=device)

        # Normalize modulation rows to target-only (prefix_len=0 inside control branch)
        scale1, gate1, scale2, gate2, zero = mod
        t_mod = (
            (scale1[1].to(device=device, dtype=dtype), scale1[1].to(device=device, dtype=dtype)),
            (gate1[1].to(device=device, dtype=dtype), gate1[1].to(device=device, dtype=dtype)),
            (scale2[1].to(device=device, dtype=dtype), scale2[1].to(device=device, dtype=dtype)),
            (gate2[1].to(device=device, dtype=dtype), gate2[1].to(device=device, dtype=dtype)),
            zero.to(device=device, dtype=dtype),
        )

        layers = self.control_layers if target_layers is None else target_layers
        hints: dict[int, torch.Tensor] = {}
        c = ctrl_tokens
        for ctrl_idx, cb in enumerate(self.control_blocks):
            hint, c = cb(c, x_base, t_mod, pe, None, 0, {})
            base_idx = layers[ctrl_idx]
            if base_idx >= 0:
                hints[base_idx] = hint
        return hints


def normalize_zimage_control_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize Diffusers/Fun ``Z-Image-Turbo-Fun-Controlnet-Union`` keys to
    :class:`ZImageFunControlNet` layout (including fusing ``to_q/to_k/to_v`` -> ``qkv``)."""

    raw = strip_control_prefix(state_dict)
    out: dict[str, torch.Tensor] = {}
    skip: set[str] = set()

    for k, v in raw.items():
        if k in skip:
            continue
        nk = k
        if nk.startswith("control_all_x_embedder.2-1."):
            nk = "control_img_in." + nk[len("control_all_x_embedder.2-1.") :]
        elif nk.startswith("control_x_embedder."):
            nk = "control_img_in." + nk[len("control_x_embedder.") :]

        if ".attention.to_q." in nk:
            k_key = k.replace(".attention.to_q.", ".attention.to_k.")
            v_key = k.replace(".attention.to_q.", ".attention.to_v.")
            if k_key in raw and v_key in raw:
                skip.add(k_key)
                skip.add(v_key)
                nk = nk.replace(".attention.to_q.", ".attention.qkv.")
                v = torch.cat([v, raw[k_key], raw[v_key]], dim=0)
        elif ".attention.to_k." in nk or ".attention.to_v." in nk:
            q_key = k.replace(".attention.to_k.", ".attention.to_q.").replace(".attention.to_v.", ".attention.to_q.")
            if q_key in raw:
                continue
        elif ".attention.to_out.0." in nk:
            nk = nk.replace(".attention.to_out.0.", ".attention.out.")
        elif ".attention.norm_q." in nk:
            nk = nk.replace(".attention.norm_q.", ".attention.q_norm.")
        elif ".attention.norm_k." in nk:
            nk = nk.replace(".attention.norm_k.", ".attention.k_norm.")

        out[nk] = v

    return _normalize_control_block_sd(out, nested_block_key="attention.")


class ZImageFunControlNet(nn.Module):
    """``Z-Image-Turbo-Fun-Controlnet-Union`` branch for ``NextDiT`` (Z-Image)."""

    family = "zimage"

    def __init__(
        self,
        *,
        dim: int = 3840,
        n_heads: int = 30,
        n_kv_heads: int | None = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 8.0 / 3.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        patch_size: int = 2,
        control_in_dim: int = 64,
        latent_channels: int = 16,
        num_base_layers: int = 30,
        control_layers: list[int] | None = None,
        use_before_proj: bool = True,
        use_after_proj: bool = True,
        proj_bias: bool = True,
        control_context_scale: float = 1.0,
        _device: torch.device | None = None,
        _dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.patch_size = int(patch_size)
        self.control_in_dim = int(control_in_dim)
        self.latent_channels = int(latent_channels)
        self.num_base_layers = int(num_base_layers)
        self.control_context_scale = float(control_context_scale)
        if control_layers is None:
            self.control_layers = list(range(0, self.num_base_layers, 2))
        else:
            self.control_layers = [int(i) for i in control_layers]

        with using_forge_operations(
            device=_device if _device is not None else torch.device("cpu"),
            dtype=_dtype if _dtype is not None else torch.float32,
            manual_cast_enabled=True,
        ):
            self.control_img_in = nn.Linear(self.control_in_dim, self.dim, bias=True)
            self.control_blocks = nn.ModuleList(
                [
                    ZeroGatedControlBlock(
                        JointTransformerBlock(
                            layer_id=i,
                            dim=self.dim,
                            n_heads=self.n_heads,
                            n_kv_heads=n_kv_heads,
                            multiple_of=multiple_of,
                            ffn_dim_multiplier=ffn_dim_multiplier,
                            norm_eps=norm_eps,
                            qk_norm=qk_norm,
                            modulation=True,
                            z_image_modulation=True,
                        ),
                        dim=self.dim,
                        block_id=i,
                        use_before_proj=use_before_proj,
                        use_after_proj=use_after_proj,
                        bias=proj_bias,
                    )
                    for i in range(len(self.control_layers))
                ]
            )

    def init_zero_weights(self) -> None:
        for cb in self.control_blocks:
            cb.init_zero_weights()

    @staticmethod
    def infer_config(state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        sd = normalize_zimage_control_state_dict(state_dict)
        meta = metadata or {}

        img_in_w = sd.get("control_img_in.weight")
        if img_in_w is None:
            raise ValueError("ZImageFunControlNet checkpoint is missing control_img_in.weight")
        dim = int(img_in_w.shape[0])
        control_in_dim = int(img_in_w.shape[1])

        qk_norm = "control_blocks.0.block.attention.q_norm.weight" in sd
        if qk_norm:
            head_dim = int(sd["control_blocks.0.block.attention.q_norm.weight"].shape[0])
            n_heads = max(1, dim // head_dim)
        else:
            n_heads = max(1, dim // 128)

        w1 = sd.get("control_blocks.0.block.feed_forward.w1.weight")
        if w1 is not None:
            ffn_hidden = int(w1.shape[0])
            ffn_dim_multiplier = float(ffn_hidden) / float(dim)
            multiple_of = 1
        else:
            ffn_dim_multiplier = 8.0 / 3.0
            multiple_of = 256

        patch_size = int(meta.get("control_patch_size", 2))
        per_patch = control_in_dim // max(1, patch_size * patch_size)
        latent_channels = (per_patch - 1) // 2 if per_patch % 2 == 1 and per_patch > 1 else per_patch

        ctrl_indices = _control_block_indices(sd)
        if not ctrl_indices:
            raise ValueError("ZImageFunControlNet checkpoint contains no control_blocks")
        n_ctrl = len(ctrl_indices)
        num_base_layers = int(meta.get("control_num_base_layers", max(30, n_ctrl * 2) if n_ctrl == 15 else n_ctrl * 2))
        if "control_layers" in meta:
            raw_layers = meta["control_layers"]
            if isinstance(raw_layers, str):
                control_layers = [int(x.strip()) for x in raw_layers.strip("[]").split(",") if x.strip()]
            else:
                control_layers = [int(x) for x in raw_layers]
        else:
            step = max(1, num_base_layers // n_ctrl)
            control_layers = [i * step for i in range(n_ctrl)]

        use_before_proj = "control_blocks.0.before_proj.weight" in sd
        use_after_proj = "control_blocks.0.after_proj.weight" in sd
        proj_bias = "control_blocks.0.after_proj.bias" in sd or "control_blocks.0.before_proj.bias" in sd

        return {
            "dim": dim,
            "n_heads": n_heads,
            "multiple_of": multiple_of,
            "ffn_dim_multiplier": ffn_dim_multiplier,
            "qk_norm": qk_norm,
            "patch_size": patch_size,
            "control_in_dim": control_in_dim,
            "latent_channels": latent_channels,
            "num_base_layers": num_base_layers,
            "control_layers": control_layers,
            "use_before_proj": use_before_proj,
            "use_after_proj": use_after_proj,
            "proj_bias": proj_bias,
            "control_context_scale": float(meta.get("control_context_scale", 1.0)),
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any] | None = None) -> "ZImageFunControlNet":
        sd = normalize_zimage_control_state_dict(state_dict)
        cfg = cls.infer_config(sd, metadata)
        dtype = next((v.dtype for v in sd.values() if isinstance(v, torch.Tensor) and v.is_floating_point()), torch.float32)
        model = cls(**cfg, _device=torch.device("meta"), _dtype=dtype)
        model.load_state_dict(sd, strict=True, assign=True)
        return model.eval().requires_grad_(False)

    def build_control_input(
        self,
        control_latent: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        masked_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble & patchify ``(B, C, H, W)`` into ``(B, H_tok*W_tok, control_in_dim)``."""

        p = self.patch_size
        ctrl = pad_to_patch_size(control_latent, (p, p))
        b, c, h, w = ctrl.shape
        per_patch_expected = self.control_in_dim // (p * p)
        if c == per_patch_expected:
            combined = ctrl
        elif per_patch_expected == 2 * self.latent_channels + 1 and c == self.latent_channels:
            if mask is None:
                m = torch.zeros((b, 1, h, w), device=ctrl.device, dtype=ctrl.dtype)
            else:
                m = mask
                if m.ndim == 2:
                    m = m.unsqueeze(0).unsqueeze(0)
                elif m.ndim == 3:
                    m = m.unsqueeze(1)
                if m.shape[-2:] != (h, w):
                    m = F.interpolate(m.float(), size=(h, w), mode="bilinear", align_corners=False)
                if m.shape[0] != b:
                    m = m[:1].expand(b, -1, -1, -1)
                m = m.to(device=ctrl.device, dtype=ctrl.dtype)
            if masked_latent is None:
                ml = torch.zeros_like(ctrl)
            else:
                ml = pad_to_patch_size(masked_latent, (p, p))
                if ml.shape[-2:] != (h, w):
                    ml = F.interpolate(ml.float(), size=(h, w), mode="bilinear", align_corners=False)
                if ml.shape[0] != b:
                    ml = ml[:1].expand(b, -1, -1, -1)
                ml = ml.to(device=ctrl.device, dtype=ctrl.dtype)
            combined = torch.cat([ctrl, m, ml], dim=1)
        else:
            raise ValueError(f"Z-Image control channels {c} incompatible with control_in_dim={self.control_in_dim} (patch_size={p})")

        return combined.view(b, combined.shape[1], h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)

    def target_layers_for_dit(self, num_dit_layers: int) -> list[int]:
        if all(0 <= idx < num_dit_layers for idx in self.control_layers):
            return list(self.control_layers)
        n_ctrl = len(self.control_layers)
        step = max(1, num_dit_layers // max(1, n_ctrl))
        return [min(num_dit_layers - 1, i * step) for i in range(n_ctrl)]

    def forward(
        self,
        control_patches: torch.Tensor,
        x_image_tokens: torch.Tensor,
        freqs_cis_img: torch.Tensor,
        adaln_input: torch.Tensor,
        *,
        target_layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        param = next(self.parameters())
        device, dtype = param.device, param.dtype

        c = self.control_img_in(control_patches.to(device=device, dtype=dtype))
        x_base = x_image_tokens.to(device=device, dtype=dtype)
        freqs = freqs_cis_img.to(device=device, dtype=dtype)
        adaln = adaln_input.to(device=device, dtype=dtype)

        layers = self.control_layers if target_layers is None else target_layers
        hints: dict[int, torch.Tensor] = {}
        for ctrl_idx, cb in enumerate(self.control_blocks):
            hint, c = cb(c, x_base, None, freqs, adaln, transformer_options={})
            base_idx = layers[ctrl_idx]
            if base_idx >= 0:
                hints[base_idx] = hint
        return hints


# endregion


# region Abstract DiTControlPatch & unified FunControlNetUnionPatch


class DiTControlPatch(ABC):
    """Abstract forward-hook ModelPatch for DiT ``control_blocks`` injection.

    Subclasses implement :meth:`_block_list`, :meth:`_extract_target_and_compute`,
    and :meth:`_inject_hint`. Handles:
    - Safe Prefix KV Cache invalidation & bypass on DiTs that expose
      ``clear_caches`` / ``_prefix_cacheable`` (e.g. ``QwenImage21Transformer2DModel``);
    - Per-call hint buckets keyed by ``id(transformer_options)`` so stacked
      patches never clobber one another or leak hints across steps;
    - Clean ``attach(dit)`` / ``detach()`` lifecycle.
    """

    def __init__(
        self,
        *,
        family: str,
        target_layers: list[int],
        strength: float = 1.0,
        control_context_scale: float = 1.0,
        mask: torch.Tensor | None = None,
        active: Callable[[], bool] | None = None,
    ) -> None:
        self.family = family
        self.target_layers = list(target_layers)
        self.strength = float(strength)
        self.control_context_scale = float(control_context_scale)
        self.mask = mask
        self.active = active
        self._handles: list = []
        self._hints: dict[int, dict[int, torch.Tensor]] = {}
        self._dit: nn.Module | None = None
        self._warned = False

    @abstractmethod
    def _block_list(self, dit: nn.Module) -> nn.ModuleList:
        """Return the base DiT block list to hook."""

    @abstractmethod
    def _compute_call_hints(self, block_args: tuple, block_kwargs: dict) -> dict[int, torch.Tensor] | None:
        """Compute per-layer raw residual hints on entry to block 0."""

    @abstractmethod
    def _apply_hint_to_output(self, output: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        """Add ``hint`` to the target token slice of a base block's ``output``."""

    @staticmethod
    def _call_key(args: tuple, kwargs: dict) -> int:
        opts = kwargs.get("transformer_options")
        if isinstance(opts, dict):
            return id(opts)
        for item in args:
            if isinstance(item, dict) and ("block_index" in item or "total_blocks" in item or "patches" in item):
                return id(item)
        return 0

    def _on_dit_pre_forward(self, module: nn.Module, args: tuple, kwargs: dict):
        """Invalidate Prefix KV Cache and pass a non-mutated copy of
        ``transformer_options`` carrying a control-patch marker so
        ``QwenImage21Transformer2DModel._prefix_cacheable`` cleanly bypasses
        prefix KV caching while control is active."""

        if callable(clear := getattr(module, "clear_caches", None)):
            clear()
        orig_opts = kwargs.get("transformer_options")
        new_opts = dict(orig_opts) if isinstance(orig_opts, dict) else {}
        patches = dict(new_opts.get("patches") or {}) if isinstance(new_opts.get("patches"), dict) else {}
        patches["_dit_control_patch"] = [self.family]
        new_opts["patches"] = patches
        new_kwargs = dict(kwargs)
        new_kwargs["transformer_options"] = new_opts
        return args, new_kwargs

    def attach(self, dit: nn.Module) -> None:
        self.detach()
        self._dit = dit
        if callable(clear := getattr(dit, "clear_caches", None)):
            clear()
        blocks = self._block_list(dit)
        self._handles.append(dit.register_forward_pre_hook(self._on_dit_pre_forward, with_kwargs=True))
        self._handles.append(blocks[0].register_forward_pre_hook(self._pre_block0, with_kwargs=True))
        for base_idx in self.target_layers:
            if 0 <= base_idx < len(blocks):
                self._handles.append(blocks[base_idx].register_forward_hook(self._make_post_hook(base_idx), with_kwargs=True))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hints.clear()
        if self._dit is not None and callable(clear := getattr(self._dit, "clear_caches", None)):
            clear()
        self._dit = None

    def _pre_block0(self, _module, args: tuple, kwargs: dict):
        key = self._call_key(args, kwargs)
        self._hints.pop(key, None)
        if self.strength == 0.0 or (self.active is not None and not self.active()):
            return None
        hints = self._compute_call_hints(args, kwargs)
        if not hints:
            return None
        scale = self.strength * self.control_context_scale
        self._hints[key] = {idx: h * scale for idx, h in hints.items()}
        return None

    def _make_post_hook(self, base_idx: int):
        def _post_hook(_module, args: tuple, kwargs: dict, output: torch.Tensor) -> torch.Tensor:
            key = self._call_key(args, kwargs)
            bucket = self._hints.get(key)
            if not bucket:
                return output
            hint = bucket.pop(base_idx, None)
            if not bucket:
                self._hints.pop(key, None)
            if hint is None:
                return output
            return self._apply_hint_to_output(output, hint)

        return _post_hook


class FunControlNetUnionPatch(DiTControlPatch):
    """Unified ``DiTControlPatch`` for ``Qwen-Image-2.1-Fun-Controlnet-Union``
    and ``Z-Image-Turbo-Fun-Controlnet-Union``."""

    def __init__(
        self,
        control_model: Qwen21FunControlNet | ZImageFunControlNet,
        control_latent: torch.Tensor,
        *,
        strength: float = 1.0,
        control_context_scale: float | None = None,
        mask: torch.Tensor | None = None,
        masked_latent: torch.Tensor | None = None,
        target_layers: list[int] | None = None,
        active: Callable[[], bool] | None = None,
    ) -> None:
        self.control_model = control_model
        self.control_latent = control_latent
        self.masked_latent = masked_latent
        self._last_span: tuple[int, int] = (0, 0)  # (start_tok, num_tok)
        scale = control_model.control_context_scale if control_context_scale is None else float(control_context_scale)
        layers = list(control_model.control_layers) if target_layers is None else list(target_layers)
        super().__init__(
            family=control_model.family,
            target_layers=layers,
            strength=strength,
            control_context_scale=scale,
            mask=mask,
            active=active,
        )

    def _block_list(self, dit: nn.Module) -> nn.ModuleList:
        if self.family == "qwen21":
            blocks = getattr(dit, "transformer_blocks", None)
            if blocks is None:
                raise ValueError("Qwen-Image-2.1 Fun ControlNet-Union requires a QwenImage21Transformer2DModel")
            return blocks
        blocks = getattr(dit, "layers", None)
        if blocks is None:
            raise ValueError("Z-Image Fun ControlNet-Union requires a NextDiT model with .layers")
        return blocks

    def _compute_call_hints(self, block_args: tuple, block_kwargs: dict) -> dict[int, torch.Tensor] | None:
        if self.family == "qwen21":
            return self._compute_qwen21_hints(block_args, block_kwargs)
        return self._compute_zimage_hints(block_args, block_kwargs)

    def _compute_qwen21_hints(self, args: tuple, kwargs: dict) -> dict[int, torch.Tensor] | None:
        # QwenImage21TransformerBlock.forward(x, mod, pe, attn_fn, prefix_len, transformer_options=...)
        x, mod, pe = args[0], args[1], args[2]
        prefix_len = int(args[4]) if len(args) > 4 else int(kwargs.get("prefix_len", 0))
        b, seq_len, _ = x.shape
        target_tokens = seq_len - prefix_len

        ctrl_in = self.control_model.build_control_input(self.control_latent, mask=self.mask, masked_latent=self.masked_latent)
        ctrl_tok = int(ctrl_in.shape[-2] * ctrl_in.shape[-1])
        if ctrl_tok != target_tokens or b % ctrl_in.shape[0] != 0:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "Qwen-Image-2.1 Fun ControlNet-Union skipped: control tokens (%d) != target tokens (%d)",
                    ctrl_tok,
                    target_tokens,
                )
            return None
        if ctrl_in.shape[0] != b:
            ctrl_in = ctrl_in.repeat(b // ctrl_in.shape[0], 1, 1, 1)

        self._last_span = (prefix_len, target_tokens)
        x_target = x[:, prefix_len:]
        pe_target = pe[:, :, prefix_len:] if pe.ndim >= 4 and pe.shape[2] == seq_len else pe[:, prefix_len:]
        return self.control_model(ctrl_in, x_target, mod, pe_target, target_layers=self.target_layers)

    def _compute_zimage_hints(self, args: tuple, kwargs: dict) -> dict[int, torch.Tensor] | None:
        # NextDiT JointTransformerBlock.forward(x, x_mask, freqs_cis, adaln_input, transformer_options=...)
        x = args[0]
        freqs_cis = args[2] if len(args) > 2 else kwargs.get("freqs_cis")
        adaln_input = args[3] if len(args) > 3 else kwargs.get("adaln_input")
        b, seq_len, _ = x.shape

        ctrl_patches = self.control_model.build_control_input(self.control_latent, mask=self.mask, masked_latent=self.masked_latent)
        ctrl_tok = int(ctrl_patches.shape[1])
        if ctrl_tok > seq_len or b % ctrl_patches.shape[0] != 0:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "Z-Image Fun ControlNet-Union skipped: control tokens (%d) exceed sequence (%d)",
                    ctrl_tok,
                    seq_len,
                )
            return None
        if ctrl_patches.shape[0] != b:
            ctrl_patches = ctrl_patches.repeat(b // ctrl_patches.shape[0], 1, 1)

        # In NextDiT, sequence is [cap_tokens, image_tokens (, ref_tokens...)];
        # locate target image tokens from dit._last_cap_len when set, else trailing ctrl_tok.
        cap_len = getattr(self._dit, "_last_cap_len", None)
        if isinstance(cap_len, int) and cap_len + ctrl_tok <= seq_len:
            start = cap_len
        else:
            start = seq_len - ctrl_tok

        self._last_span = (start, ctrl_tok)
        x_img = x[:, start : start + ctrl_tok]
        freqs_img = freqs_cis[:, start : start + ctrl_tok] if freqs_cis.shape[1] == seq_len else freqs_cis[:, :, start : start + ctrl_tok]
        return self.control_model(ctrl_patches, x_img, freqs_img, adaln_input, target_layers=self.target_layers)

    def _apply_hint_to_output(self, output: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        start, length = self._last_span
        hint = hint.to(device=output.device, dtype=output.dtype)
        if start == 0 and length == output.shape[1]:
            return output + hint
        out = output.clone()
        out[:, start : start + length] = out[:, start : start + length] + hint
        return out


# endregion


# region Forge ControlModelPatcher registration


class DiTControlNetUnionPatcher:
    """Forge ``ControlModelPatcher`` subclass for Qwen-Image-2.1 and Z-Image
    Fun ControlNet-Union models (registered via ``add_supported_control_model``)."""

    anima_compatible: bool = False
    supported_dit_families: tuple[str, ...] = ("qwen21", "zimage")

    def __init__(self, control_model: Qwen21FunControlNet | ZImageFunControlNet) -> None:
        self.model_patcher = None
        self.control_model = control_model
        self.family = control_model.family
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0
        self.positive_advanced_weighting = None
        self.negative_advanced_weighting = None
        self.advanced_frame_weighting = None
        self.advanced_sigma_weighting = None
        self.advanced_mask_weighting = None
        self._patch: FunControlNetUnionPatch | None = None

    @staticmethod
    def try_build_from_state_dict(state_dict: dict[str, torch.Tensor], ckpt_path: str | None = None):
        family = detect_dit_control_family(state_dict)
        if family not in ("qwen21", "zimage"):
            return None
        metadata: dict[str, Any] = {}
        if ckpt_path and os.path.isfile(ckpt_path) and str(ckpt_path).lower().endswith((".safetensors", ".sft")):
            try:
                _, raw_meta = load_torch_file(ckpt_path, return_metadata=True)
                metadata = dict(raw_meta or {})
            except Exception:
                metadata = {}
        if family == "qwen21":
            model = Qwen21FunControlNet.from_state_dict(state_dict, metadata=metadata)
        else:
            model = ZImageFunControlNet.from_state_dict(state_dict, metadata=metadata)
        return DiTControlNetUnionPatcher(model)

    def process_after_running_preprocessors(self, process, params, *args, **kwargs):
        return

    def process_before_conditioning(self, process) -> bool:
        return False

    def process_before_every_sampling(self, process, cond: torch.Tensor, mask: torch.Tensor | None, *args, **kwargs):
        from modules import spectrum_force

        objects = process.sd_model.forge_objects
        unet = objects.unet
        dit = unet.model.diffusion_model

        if self.family == "qwen21" and not hasattr(dit, "transformer_blocks"):
            raise ValueError("Qwen-Image-2.1 Fun ControlNet-Union requires an active Qwen-Image-2.1 checkpoint")
        if self.family == "zimage" and not hasattr(dit, "layers"):
            raise ValueError("Z-Image Fun ControlNet-Union requires an active Z-Image checkpoint")

        blocks = dit.transformer_blocks if self.family == "qwen21" else dit.layers
        target_layers = self.control_model.target_layers_for_dit(len(blocks))

        if (self.start_percent > 0.0 or self.end_percent < 1.0) and hasattr(unet, "model_options"):
            unet = unet.clone() if hasattr(unet, "clone") else unet
            predictor = getattr(getattr(unet, "model", None), "predictor", None)
            if predictor is not None and hasattr(predictor, "percent_to_sigma"):
                reason = spectrum_force.sigma_window(predictor.percent_to_sigma(self.start_percent), predictor.percent_to_sigma(self.end_percent))
            else:
                reason = f"dit_control_union_{self.family}"
            spectrum_force.request(unet, reason)
            objects.unet = unet

        if mask is not None and getattr(process, "inpainting_mask_invert", False):
            mask = 1.0 - mask

        # Encode control image into active DiT latent space
        vae_in = cond.movedim(1, -1)
        if self.family == "qwen21" and vae_in.shape[-1] == 3:
            vae_in = torch.cat([vae_in, torch.ones_like(vae_in[..., :1])], dim=-1)
        latent = objects.vae.encode(vae_in)
        latent_format = getattr(getattr(process.sd_model, "model_config", None), "latent_format", None)
        if latent_format is not None and hasattr(latent_format, "process_in"):
            latent = latent_format.process_in(latent)

        masked_latent = None
        if mask is not None and self.control_model.control_in_dim == 2 * self.control_model.latent_channels + 1:
            m_img = F.interpolate(mask.float(), size=cond.shape[-2:], mode="bilinear", align_corners=False) if mask.shape[-2:] != cond.shape[-2:] else mask.float()
            masked_cond = cond * (1.0 - m_img.to(device=cond.device, dtype=cond.dtype))
            m_in = masked_cond.movedim(1, -1)
            if self.family == "qwen21" and m_in.shape[-1] == 3:
                m_in = torch.cat([m_in, torch.ones_like(m_in[..., :1])], dim=-1)
            masked_latent = objects.vae.encode(m_in)
            if latent_format is not None and hasattr(latent_format, "process_in"):
                masked_latent = latent_format.process_in(masked_latent)

        device, dtype = unet.load_device, unet.model.computation_dtype
        self.control_model.to(device=device, dtype=dtype)
        latent = latent.to(device=device, dtype=dtype)
        if masked_latent is not None:
            masked_latent = masked_latent.to(device=device, dtype=dtype)

        if self._patch is not None:
            self._patch.detach()
        self._patch = FunControlNetUnionPatch(
            self.control_model,
            latent,
            strength=self.strength,
            mask=mask,
            masked_latent=masked_latent,
            target_layers=target_layers,
        )
        self._patch.attach(dit)

    def process_after_every_sampling(self, *args, **kwargs):
        if self._patch is not None:
            self._patch.detach()
            self._patch = None
        self.control_model.to(device=memory_management.unet_offload_device())


# endregion
