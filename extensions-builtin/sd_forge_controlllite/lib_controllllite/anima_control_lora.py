"""Anima control-LoRA: a control image added to the DiT's patch embedding, plus a LoRA.

The format of Claquasse/Anima-Control-Pose (``anima_pose_preview2.safetensors``): one file holds

- ``diffusion_model.control_embedder.proj.{weight,bias}``: a zero-initialised Linear over 2x2
  patches of the control image's VAE latent (16 x 2 x 2 -> the DiT width), whose output is added
  to the output of the DiT's own patch embedding (``x_embedder``);
- a LoRA (``diffusion_model.*.lora_A / lora_B``) on the DiT blocks and the LLM adapter, trained
  with the embedder.

As in the reference ComfyUI node (``comfyui/anima_control_lora`` of that repository): the control
latent is the VAE's own latent of the control image (not the sampler's normalised one), the
embedder's output is scaled by the strength (0: no control), and the LoRA is applied at 1.0.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from backend.utils import pad_to_patch_size

logger = logging.getLogger("ControlNet")

EMBEDDER_KEY = "control_embedder."
PATCH = 2  # the DiT's spatial patch size (temporal: 1)


def is_control_lora(state_dict: dict) -> bool:
    return any(key.endswith(EMBEDDER_KEY + "proj.weight") for key in state_dict)


def split_control_lora(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(the embedder's tensors, without their prefix; the LoRA's)"""

    embedder = {key.split(EMBEDDER_KEY, 1)[1]: value for key, value in state_dict.items() if EMBEDDER_KEY in key}
    lora = {key: value for key, value in state_dict.items() if EMBEDDER_KEY not in key}
    return embedder, lora


class ControlEmbedder(nn.Module):
    """The control latent [B, C, T, H, W] -> tokens [B, T, H/2, W/2, width] (the DiT's patch layout)."""

    def __init__(self, in_channels: int = 16, model_channels: int = 2048):
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels * PATCH * PATCH, model_channels)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "ControlEmbedder":
        weight = state_dict.get("proj.weight")
        if weight is None or weight.ndim != 2 or weight.shape[1] % (PATCH * PATCH):
            raise ValueError(f"not an Anima control embedder: {sorted(state_dict)[:4]}")
        embedder = cls(weight.shape[1] // (PATCH * PATCH), weight.shape[0])
        embedder.load_state_dict(state_dict, strict=True)
        return embedder.eval().requires_grad_(False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        b, c, t, h, w = latent.shape
        if c != self.in_channels:
            raise ValueError(f"the control latent has {c} channels, the embedder takes {self.in_channels}")
        # pad like the DiT's input, then "b c t (h m) (w n) -> b t h w (c m n)"
        latent = pad_to_patch_size(latent, (1, PATCH, PATCH))
        h, w = latent.shape[-2] // PATCH, latent.shape[-1] // PATCH
        patches = latent.reshape(b, c, t, h, PATCH, w, PATCH).permute(0, 2, 3, 5, 1, 4, 6).reshape(b, t, h, w, c * PATCH * PATCH)
        return self.proj(patches)


class ControlInjection:
    """Adds ``strength * tokens`` to the output of ``dit.x_embedder`` while ``active()`` says so.

    The output holds the generated frame first, then any reference frames (which get nothing),
    and the cond/uncond copies batched together (the tokens are repeated for them)."""

    def __init__(self, tokens: torch.Tensor, strength: float, active: Optional[Callable[[], bool]] = None):
        self.tokens = tokens
        self.strength = float(strength)
        self.active = active
        self.handle = None
        self._warned = False

    def attach(self, dit) -> None:
        self.detach()
        self.handle = dit.x_embedder.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, _inputs, output: torch.Tensor) -> torch.Tensor:
        if self.strength == 0 or (self.active is not None and not self.active()):
            return output
        tokens = self.tokens
        if tokens.shape[2:] != output.shape[2:] or output.shape[0] % tokens.shape[0]:
            if not self._warned:
                self._warned = True
                logger.warning(f"Anima control-LoRA skipped: the control is {tuple(tokens.shape[:4])}, the image {tuple(output.shape[:4])} (tiled or resized pass?)")
            return output
        tokens = tokens.to(device=output.device, dtype=output.dtype) * self.strength
        if output.shape[0] != tokens.shape[0]:
            tokens = tokens.repeat(output.shape[0] // tokens.shape[0], 1, 1, 1, 1)
        if output.shape[1] != tokens.shape[1]:  # reference frames follow the generated one
            tokens = nn.functional.pad(tokens, (0, 0, 0, 0, 0, 0, 0, output.shape[1] - tokens.shape[1]))
        return output + tokens
