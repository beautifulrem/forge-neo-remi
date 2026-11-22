from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.flux3_engine import Flux3TextEngine
from modules.shared import opts


class Flux3(ForgeDiffusionEngine):
    """FLUX.3 Image Klein: Qwen3-VL-4B text, FLUX.2 VAE, multi-reference editing.

    The base checkpoint samples with classifier-free guidance (~50 steps, CFG
    ~3.5); the step-distilled flow-map checkpoint (``is_flux3_distilled``)
    samples at CFG 1 in ~16 steps and embeds each step's target time."""

    matched_guesses = [model_list.Flux3Klein]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})
        vae = VAE(model=huggingface_components["vae"], is_flux2=True)
        k_predictor = self._get_predictor()
        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Flux3TextEngine(text_encoder=clip.cond_stage_model.qwen3vl_4b, tokenizer=clip.tokenizer.qwen3vl_4b)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_flux3_engine = True
        self.is_flux3_distilled = bool(getattr(huggingface_components["transformer"], "flow_map", False))

    def references(self) -> list[torch.Tensor]:
        """The img2img input (first) and the ImageStitch images, as reference latents."""

        refs = [*self.ref_latents]
        if self.ini_latent is not None:
            refs.insert(0, self.ini_latent)
        return refs

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            if not opts.flux3_do_reference:
                dynamic_args.ref_latents.clear()
            else:
                dynamic_args.ref_latents = self.references()
                self.ini_latent = None

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        return super().encode_first_stage(x, multiple=16 if opts.flux3_do_reference else 0)
