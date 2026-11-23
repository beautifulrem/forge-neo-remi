"""Bria Fibo 2.0 (8.1B Dev & 4-step Edit Turbo) diffusion engine.

Reuses Forge's existing ``Qwen3-VL-4B`` text encoder (``Flux3TextEngine``) and
``FLUX.2`` 128-channel VAE stack, adds deterministic VGL 2.0 JSON structured
prompt canonicalization, multi-reference routing, and explicit CC BY-NC 4.0
non-commercial license metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.nn.fibo import (
    FIBO_COMMERCIAL_USE,
    FIBO_LICENSE,
    FIBO_LICENSE_NOTICE,
    FIBO_MAX_REFERENCES,
    format_vgl_for_encoder,
)
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.flux3_engine import Flux3TextEngine


class Fibo2(ForgeDiffusionEngine):
    """Bria Fibo 2.0 (8.1B Dev & 4-step Edit Turbo).

    License: CC BY-NC 4.0 (non-commercial use without a commercial license from Bria).
    """

    matched_guesses: list = []
    is_fibo_engine = True
    license = FIBO_LICENSE
    commercial_use = FIBO_COMMERCIAL_USE
    license_notice = FIBO_LICENSE_NOTICE

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        from backend.nn.dit_capabilities_ext import register_extended_dit_capabilities

        register_extended_dit_capabilities()

        clip = CLIP(
            model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]},
            tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]},
        )
        vae = VAE(model=huggingface_components["vae"], is_flux2=True)
        k_predictor = self._get_predictor()
        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        self.text_processing_engine_qwen = Flux3TextEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_4b,
            tokenizer=clip.tokenizer.qwen3vl_4b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_fibo_turbo = bool(getattr(huggingface_components["transformer"], "is_edit_turbo", False))
        self.fibo_do_reference = True

    def references(self) -> list[torch.Tensor]:
        refs = [*self.ref_latents]
        if self.ini_latent is not None:
            refs.insert(0, self.ini_latent)
        return refs[:FIBO_MAX_REFERENCES]

    def clear_references(self):
        dynamic_args.ref_latents.clear()
        dit = getattr(getattr(getattr(self, "forge_objects", None), "unet", None), "model", None)
        dit = getattr(dit, "diffusion_model", None)
        if callable(clear := getattr(dit, "clear_caches", None)):
            clear()
        super().clear_references()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        is_neg = bool(getattr(prompt, "is_negative_prompt", False))
        if not is_neg:
            if not getattr(self, "fibo_do_reference", True):
                dynamic_args.ref_latents.clear()
            else:
                dynamic_args.ref_latents = self.references()
                self.ini_latent = None

        canonical = [format_vgl_for_encoder(line) for line in prompt]
        return self.text_processing_engine_qwen(canonical)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(format_vgl_for_encoder(prompt)))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        return super().encode_first_stage(x, multiple=16 if getattr(self, "fibo_do_reference", True) else 0)
