from collections import OrderedDict

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.ernie_engine import Ministral3Engine

_COND_CACHE_MAX = 8


class ErnieImage(ForgeDiffusionEngine):
    """ERNIE-Image (8B Apache-2.0) engine on ``ErnieImageModel`` + Ministral-3 + FLUX.2 VAE."""

    matched_guesses = [model_list.ErnieImage]
    is_ernie_engine = True
    license = "Apache-2.0"

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        from backend.nn.dit_capabilities_ext import register_extended_dit_capabilities

        register_extended_dit_capabilities()

        clip = CLIP(model_dict={"ministral3": huggingface_components["text_encoder"]}, tokenizer_dict={"ministral3": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_ministral = Ministral3Engine(
            text_encoder=clip.cond_stage_model.ministral3,
            tokenizer=clip.tokenizer.ministral3,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True
        self._cond_cache: OrderedDict[tuple[str, ...], list[torch.Tensor]] = OrderedDict()

    def clear_references(self):
        self._cond_cache.clear()
        super().clear_references()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        key = tuple(str(line) for line in prompt)
        cached = self._cond_cache.get(key)
        if cached is not None:
            self._cond_cache.move_to_end(key)
            return list(cached)
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        result = self.text_processing_engine_ministral(prompt)
        self._cond_cache[key] = list(result)
        while len(self._cond_cache) > _COND_CACHE_MAX:
            self._cond_cache.popitem(last=False)
        return result

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_ministral.tokenize(prompt))
        return token_count, max(999, token_count)
