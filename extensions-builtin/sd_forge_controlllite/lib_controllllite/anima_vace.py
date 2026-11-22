"""Anima VACE ControlNet (spaced-block duplication + zero-conv injection).

Supports ``TaihoC/Anima-ControlNet-VACE-Depth`` and compatible community VACE
adapters (Canny, etc.) trained with ``kohya-ss/sd-scripts``
(``networks/control_net_vace_anima.py``).

Architecture:
- ``control_embedder``: an Anima ``PatchEmbed`` over the Wan-VAE-encoded,
  ``Wan21.process_in``-normalised control latent (with the optional zero
  padding-mask channel, 16 + 1 = 17 channels -> 68 patch features -> 2048).
- ``control_blocks``: ``N`` copies of Anima's DiT ``Block`` wrapped with
  zero-initialised projections (``before_proj`` on block 0 to fuse the base
  DiT's patch tokens ``x_base``; ``after_proj`` on each block to emit per-block
  residual hints injected into the matching base DiT blocks).

Depth compatibility:
A VACE adapter trained on 28-block Anima 2B maps its target layers
(``[0, 7, 14, 21]``) onto the original blocks of depth-expanded Anima-2.9B (40)
and Anima-3.8B (52) via ``backend.nn.anima_layout`` so no offline weight
remapping is needed.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.args import dynamic_args
from backend.nn import anima_layout
from backend.nn.anima import Block as AnimaBlock
from backend.nn.anima import PatchEmbed
from backend.operations import using_forge_operations
from backend.utils import pad_to_patch_size

logger = logging.getLogger("ControlNet")

_VACE_PREFIXES = ("control_model.", "diffusion_model.", "model.", "net.")


def strip_vace_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip an optional wrapper prefix (``control_model.`` / ``diffusion_model.``)."""

    if any(k.startswith("control_blocks.") for k in state_dict):
        return state_dict
    for prefix in _VACE_PREFIXES:
        if any(k.startswith(f"{prefix}control_blocks.") for k in state_dict):
            return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


def is_anima_vace(state_dict: dict) -> bool:
    """True for Anima VACE ControlNet state dicts (spaced blocks + zero-conv)."""

    sd = strip_vace_prefix(state_dict)
    has_block = any(k.startswith("control_blocks.0.block.") for k in sd)
    has_vace = "control_embedder.proj.1.weight" in sd or any(k.startswith(("control_blocks.0.after_proj.", "control_blocks.0.before_proj.")) for k in sd)
    return has_block and has_vace


def infer_vace_config(state_dict: dict[str, torch.Tensor], metadata: dict | None = None) -> dict:
    """Infer ``VACEControlNetAnima`` constructor kwargs from weights and metadata."""

    sd = strip_vace_prefix(state_dict)
    meta = metadata or {}

    def _meta_int(key: str, default: int) -> int:
        raw = meta.get(key)
        if raw is None:
            return default
        try:
            return int(str(raw).strip())
        except ValueError:
            return default

    def _meta_bool(key: str, default: bool) -> bool:
        raw = meta.get(key)
        if raw is None:
            return default
        return str(raw).strip().lower() in ("1", "true", "yes")

    def _meta_float(key: str, default: float) -> float:
        raw = meta.get(key)
        if raw is None:
            return default
        try:
            return float(str(raw).strip())
        except ValueError:
            return default

    embed_w = sd.get("control_embedder.proj.1.weight")
    if embed_w is not None:
        model_channels = int(embed_w.shape[0])
        in_per_patch = int(embed_w.shape[1]) // 4
    elif (after_w := sd.get("control_blocks.0.after_proj.weight")) is not None:
        model_channels = int(after_w.shape[0])
        in_per_patch = 17
    else:
        model_channels = _meta_int("vace.model_channels", 2048)
        in_per_patch = 17

    latent_channels = _meta_int("vace.in_channels", 16)
    if "vace.concat_padding_mask" in meta:
        concat_padding_mask = _meta_bool("vace.concat_padding_mask", True)
    else:
        concat_padding_mask = in_per_patch == latent_channels + 1
        if not concat_padding_mask and in_per_patch > 0:
            latent_channels = in_per_patch

    if (k_proj := sd.get("control_blocks.0.block.cross_attn.k_proj.weight")) is not None:
        crossattn_emb_channels = int(k_proj.shape[1])
    else:
        crossattn_emb_channels = 1024

    if (adaln := sd.get("control_blocks.0.block.adaln_modulation_self_attn.1.weight")) is not None:
        adaln_lora_dim = int(adaln.shape[0])
    else:
        adaln_lora_dim = 256

    if (q_norm := sd.get("control_blocks.0.block.self_attn.q_norm.weight")) is not None and q_norm.shape[0] > 0:
        num_heads = model_channels // int(q_norm.shape[0])
    else:
        num_heads = 16

    if (mlp1 := sd.get("control_blocks.0.block.mlp.layer1.weight")) is not None:
        mlp_ratio = float(mlp1.shape[0]) / float(model_channels)
    else:
        mlp_ratio = 4.0

    ctrl_indices = sorted({int(k.split(".")[1]) for k in sd if k.startswith("control_blocks.") and k.split(".")[1].isdigit()})
    num_control_blocks = len(ctrl_indices)
    if num_control_blocks == 0:
        raise ValueError("Anima VACE checkpoint contains no control_blocks")

    num_blocks = _meta_int("vace.num_blocks", 28)
    condition_strategy = str(meta.get("vace.condition_strategy", "spaced")).strip() or "spaced"
    vace_block_every_n = _meta_int("vace.block_every_n", 7)
    if condition_strategy == "spaced" and num_control_blocks > 0:
        # Match reference loader: derive spacing from actual control block count in the file.
        computed = num_blocks // num_control_blocks
        if computed > 0 and len(range(0, num_blocks, computed)) == num_control_blocks:
            vace_block_every_n = computed
        elif len(range(0, num_blocks, vace_block_every_n)) != num_control_blocks:
            vace_block_every_n = max(1, (num_blocks + num_control_blocks - 1) // num_control_blocks)

    use_after_proj = "control_blocks.0.after_proj.weight" in sd if "vace.use_after_proj" not in meta else _meta_bool("vace.use_after_proj", True)
    control_context_scale = _meta_float("vace.control_context_scale", 1.0)

    return {
        "model_channels": model_channels,
        "num_heads": num_heads,
        "num_blocks": num_blocks,
        "crossattn_emb_channels": crossattn_emb_channels,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "latent_channels": latent_channels,
        "vace_block_every_n": vace_block_every_n,
        "condition_strategy": condition_strategy,
        "num_control_blocks": num_control_blocks,
        "use_after_proj": use_after_proj,
        "adaln_lora_dim": adaln_lora_dim,
        "concat_padding_mask": concat_padding_mask,
        "mlp_ratio": mlp_ratio,
        "control_context_scale": control_context_scale,
    }


class VACEControlBlock(nn.Module):
    """One Anima DiT block wrapped with VACE zero-conv projections."""

    def __init__(self, block: nn.Module, x_dim: int, block_id: int = 0, use_after_proj: bool = True):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.x_dim = x_dim
        self.use_after_proj = use_after_proj
        if block_id == 0:
            self.before_proj = nn.Linear(x_dim, x_dim)
        if use_after_proj:
            self.after_proj = nn.Linear(x_dim, x_dim)

    def init_weights(self):
        if self.block_id == 0:
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        if self.use_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)

    def forward(self, c: torch.Tensor, x_base: torch.Tensor, emb_B_T_D: torch.Tensor, crossattn_emb: torch.Tensor, **block_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if self.block_id == 0:
            c = self.before_proj(c) + x_base
        c = self.block(c, emb_B_T_D, crossattn_emb, **block_kwargs)
        c_skip = self.after_proj(c) if self.use_after_proj else c
        return c_skip, c


class VACEControlNetAnima(nn.Module):
    """Standalone VACE control branch for Anima (2B / 2.9B / 3.8B)."""

    def __init__(
        self,
        model_channels: int = 2048,
        num_heads: int = 16,
        num_blocks: int = 28,
        crossattn_emb_channels: int = 1024,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        latent_channels: int = 16,
        vace_block_every_n: int = 7,
        condition_strategy: str = "spaced",
        num_control_blocks: Optional[int] = None,
        use_after_proj: bool = True,
        adaln_lora_dim: int = 256,
        concat_padding_mask: bool = True,
        mlp_ratio: float = 4.0,
        control_context_scale: float = 1.0,
        _device: Optional[torch.device] = None,
        _dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.latent_channels = latent_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.vace_block_every_n = vace_block_every_n
        self.condition_strategy = condition_strategy
        self.concat_padding_mask = concat_padding_mask
        self.control_context_scale = float(control_context_scale)

        if condition_strategy == "spaced":
            layers = list(range(0, num_blocks, max(1, vace_block_every_n)))
            if num_control_blocks is not None:
                layers = layers[:num_control_blocks]
        else:
            n_ctrl = num_control_blocks if num_control_blocks is not None else max(1, num_blocks // max(1, vace_block_every_n))
            layers = list(range(0, n_ctrl))
        if not layers or layers[0] != 0:
            raise ValueError(f"Anima VACE first control block must map to base block 0, got {layers}")
        self.control_layers: list[int] = layers

        embed_in_channels = latent_channels + (1 if concat_padding_mask else 0)
        with using_forge_operations(
            device=_device if _device is not None else torch.device("cpu"),
            dtype=_dtype if _dtype is not None else torch.float32,
            manual_cast_enabled=True,
        ):
            self.control_embedder = PatchEmbed(
                spatial_patch_size=patch_spatial,
                temporal_patch_size=patch_temporal,
                in_channels=embed_in_channels,
                out_channels=model_channels,
            )
            self.control_blocks = nn.ModuleList(
                [
                    VACEControlBlock(
                        block=AnimaBlock(
                            x_dim=model_channels,
                            context_dim=crossattn_emb_channels,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            adaln_lora_dim=adaln_lora_dim,
                        ),
                        x_dim=model_channels,
                        block_id=i,
                        use_after_proj=use_after_proj,
                    )
                    for i in range(len(self.control_layers))
                ]
            )

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], metadata: dict | None = None) -> "VACEControlNetAnima":
        sd = strip_vace_prefix(state_dict)
        config = infer_vace_config(sd, metadata)
        weight_dtype = next((v.dtype for v in sd.values() if isinstance(v, torch.Tensor) and v.is_floating_point()), torch.float32)
        model = cls(**config, _device=torch.device("meta"), _dtype=weight_dtype)
        model.load_state_dict(sd, strict=True, assign=True)
        return model.eval().requires_grad_(False)

    def target_layers_for_dit(self, dit_num_blocks: int) -> list[int]:
        """Map trained control layer indices onto a target DiT depth (2B/2.9B/3.8B)."""

        if dit_num_blocks == self.num_blocks:
            return list(self.control_layers)
        if self.num_blocks < dit_num_blocks:
            originals = anima_layout.original_blocks(self.num_blocks, dit_num_blocks)
            if originals is None:
                raise ValueError(f"Anima VACE trained for {self.num_blocks} blocks cannot map onto a {dit_num_blocks}-block DiT")
            return [int(originals[idx]) for idx in self.control_layers]
        mapped = anima_layout.block_map(self.num_blocks, dit_num_blocks)
        if mapped is None:
            raise ValueError(f"Anima VACE trained for {self.num_blocks} blocks cannot map onto a {dit_num_blocks}-block DiT")
        targets, _ = mapped
        return [int(targets[idx][0]) if idx in targets else -1 for idx in self.control_layers]

    def forward(
        self,
        control_latent: torch.Tensor,
        x_base_tokens: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = None,
        silu_emb_B_T_D: Optional[torch.Tensor] = None,
        target_layers: Optional[list[int]] = None,
    ) -> dict[int, torch.Tensor]:
        """Run the VACE branch and return ``{base_block_idx: hint_tensor}``."""

        param = next(self.parameters())
        device, cdtype = param.device, param.dtype

        control_latent = control_latent.to(device=device, dtype=cdtype)
        if control_latent.ndim == 4:
            control_latent = control_latent.unsqueeze(2)
        control_latent = pad_to_patch_size(control_latent, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
        if self.concat_padding_mask and control_latent.shape[1] == self.latent_channels:
            zeros_mask = torch.zeros(
                control_latent.shape[0],
                1,
                *control_latent.shape[2:],
                device=device,
                dtype=cdtype,
            )
            control_latent = torch.cat([control_latent, zeros_mask], dim=1)

        x_base_tokens = x_base_tokens.to(device=device, dtype=cdtype)
        emb_B_T_D = emb_B_T_D.to(device=device, dtype=cdtype)
        crossattn_emb = crossattn_emb.to(device=device, dtype=cdtype)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(device=device, dtype=cdtype)
        if extra_per_block_pos_emb is not None:
            extra_per_block_pos_emb = extra_per_block_pos_emb.to(device=device, dtype=cdtype)
        if silu_emb_B_T_D is not None:
            silu_emb_B_T_D = silu_emb_B_T_D.to(device=device, dtype=cdtype)
        else:
            silu_emb_B_T_D = F.silu(emb_B_T_D)

        rope_for_blocks = rope_emb_L_1_1_D
        if rope_for_blocks is not None and rope_for_blocks.ndim == 4:
            rope_for_blocks = rope_for_blocks.unsqueeze(1).unsqueeze(0)

        c = self.control_embedder(control_latent)
        layers = self.control_layers if target_layers is None else target_layers
        # Keep VACE blocks isolated from base-DiT attention modifiers / block indices.
        block_kwargs = dict(
            rope_emb_L_1_1_D=rope_for_blocks,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            extra_per_block_pos_emb=extra_per_block_pos_emb,
            transformer_options={},
            silu_emb_B_T_D=silu_emb_B_T_D,
        )

        hints: dict[int, torch.Tensor] = {}
        for ctrl_idx, cb in enumerate(self.control_blocks):
            c_skip, c = cb(c, x_base_tokens, emb_B_T_D, crossattn_emb, **block_kwargs)
            base_idx = layers[ctrl_idx]
            if base_idx >= 0:
                hints[base_idx] = c_skip
        return hints


class VACEInjection:
    """Runs ``VACEControlNetAnima`` before DiT block 0 and injects its scaled hints
    after each target DiT block via PyTorch forward hooks.

    Hints are stored per forward call (keyed by ``id(transformer_options)``) and
    popped as each target block completes, so multiple VACE units stack cleanly
    and stale hints can never leak across steps or failed runs.
    """

    def __init__(
        self,
        vace: VACEControlNetAnima,
        control_latent: torch.Tensor,
        strength: float,
        target_layers: list[int],
        *,
        mask: Optional[torch.Tensor] = None,
        active: Optional[Callable[[], bool]] = None,
    ):
        self.vace = vace
        self.control_latent = control_latent
        self.strength = float(strength)
        self.target_layers = list(target_layers)
        self.mask = mask
        self.active = active
        self._handles: list = []
        self._hints: dict[int, dict[int, torch.Tensor]] = {}
        self._warned = False

    def attach(self, dit) -> None:
        self.detach()
        blocks = dit.blocks
        self._handles.append(blocks[0].register_forward_pre_hook(self._pre_block0, with_kwargs=True))
        for base_idx in self.target_layers:
            if 0 <= base_idx < len(blocks):
                self._handles.append(blocks[base_idx].register_forward_hook(self._make_post_hook(base_idx), with_kwargs=True))
        instances: set["VACEInjection"] = getattr(dynamic_args, "ACTIVE_ANIMA_VACE", set())
        instances.add(self)
        setattr(dynamic_args, "ACTIVE_ANIMA_VACE", instances)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hints.clear()
        instances = getattr(dynamic_args, "ACTIVE_ANIMA_VACE", None)
        if instances is not None:
            instances.discard(self)

    def _call_key(self, kwargs: dict) -> int:
        options = kwargs.get("transformer_options")
        return id(options) if isinstance(options, dict) else 0

    def _build_hint_mask(self, sample: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mask is None:
            return None
        b, _t, h_tok, w_tok, _d = sample.shape
        m = self.mask
        if m.ndim == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.ndim == 3:
            m = m.unsqueeze(1)
        m = F.interpolate(m.float(), size=(h_tok, w_tok), mode="bilinear", align_corners=False)
        if m.shape[0] != b:
            if b % m.shape[0] == 0:
                m = m.repeat(b // m.shape[0], 1, 1, 1)
            else:
                m = m[:1].expand(b, -1, -1, -1)
        return m.to(device=sample.device, dtype=sample.dtype).reshape(b, 1, h_tok, w_tok, 1)

    def _pre_block0(self, _module, args: tuple, kwargs: dict):
        key = self._call_key(kwargs)
        self._hints.pop(key, None)
        if self.strength == 0.0 or (self.active is not None and not self.active()):
            return None

        x_B_T_H_W_D, emb_B_T_D, crossattn_emb = args[0], args[1], args[2]
        b, t, h_tok, w_tok, _d = x_B_T_H_W_D.shape

        ctrl = self.control_latent
        if ctrl.ndim == 4:
            ctrl = ctrl.unsqueeze(2)
        ctrl_padded = pad_to_patch_size(ctrl, (self.vace.patch_temporal, self.vace.patch_spatial, self.vace.patch_spatial))
        ctrl_h_tok = ctrl_padded.shape[-2] // self.vace.patch_spatial
        ctrl_w_tok = ctrl_padded.shape[-1] // self.vace.patch_spatial
        if (ctrl_h_tok, ctrl_w_tok) != (h_tok, w_tok) or b % ctrl_padded.shape[0] != 0:
            if not self._warned:
                self._warned = True
                logger.warning(
                    f"Anima VACE ControlNet skipped: control token grid {(ctrl_padded.shape[0], ctrl_h_tok, ctrl_w_tok)} "
                    f"does not match image {(b, h_tok, w_tok)} (tiled or resized pass?)"
                )
            return None

        if ctrl_padded.shape[0] != b:
            ctrl_padded = ctrl_padded.repeat(b // ctrl_padded.shape[0], 1, 1, 1, 1)

        # Reference frames (Anima Edit / ImageStitch) sit after frame 0; control only applies to frame 0.
        x_base = x_B_T_H_W_D[:, :1] if t > 1 else x_B_T_H_W_D
        rope = kwargs.get("rope_emb_L_1_1_D")
        if t > 1 and rope is not None:
            tokens_per_frame = h_tok * w_tok
            if rope.ndim == 6 and rope.shape[1] >= tokens_per_frame:
                rope = rope[:, :tokens_per_frame]
            elif rope.ndim == 4 and rope.shape[0] >= tokens_per_frame:
                rope = rope[:tokens_per_frame]

        hints = self.vace(
            ctrl_padded,
            x_base,
            emb_B_T_D,
            crossattn_emb,
            rope_emb_L_1_1_D=rope,
            adaln_lora_B_T_3D=kwargs.get("adaln_lora_B_T_3D"),
            extra_per_block_pos_emb=kwargs.get("extra_per_block_pos_emb"),
            silu_emb_B_T_D=kwargs.get("silu_emb_B_T_D"),
            target_layers=self.target_layers,
        )

        scale = self.strength * self.vace.control_context_scale
        hint_mask = self._build_hint_mask(next(iter(hints.values()))) if hints else None
        scaled: dict[int, torch.Tensor] = {}
        for base_idx, hint in hints.items():
            if hint_mask is not None:
                hint = hint * hint_mask
            hint = hint * scale
            if t > 1 and hint.shape[1] != t:
                hint = F.pad(hint, (0, 0, 0, 0, 0, 0, 0, t - hint.shape[1]))
            scaled[base_idx] = hint
        self._hints[key] = scaled
        return None

    def _make_post_hook(self, base_idx: int):
        def _post_hook(_module, _args, kwargs: dict, output: torch.Tensor) -> torch.Tensor:
            key = self._call_key(kwargs)
            bucket = self._hints.get(key)
            if not bucket:
                return output
            hint = bucket.pop(base_idx, None)
            if not bucket:
                self._hints.pop(key, None)
            if hint is None:
                return output
            return output + hint.to(device=output.device, dtype=output.dtype)

        return _post_hook
