from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.misc.image_resize import resize_reference
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.krea2_engine import Qwen3VL4BEngine
from modules.shared import opts


class Krea2(ForgeDiffusionEngine):
    matched_guesses = [model_list.Krea2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen3VL4BEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_4b,
            tokenizer=clip.tokenizer.qwen3vl_4b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True
        self.is_krea2_engine = True
        self._vision_cache: tuple[tuple[int, ...], list[torch.Tensor], list[torch.Tensor]] | None = None

    def clear_references(self):
        self._vision_cache = None
        dynamic_args.ref_latents.clear()
        super().clear_references()

    def _get_references(self) -> list[torch.Tensor]:
        _references = [*self.ref_latents]
        if self.ini_latent is not None:
            _references.insert(0, self.ini_latent)
        return _references

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        _references = self._get_references()
        if opts.krea2_do_reference and bool(_references):
            # Grounded conditioning for both positive and negative prompts (matches comfyui-krea2edit
            # and QwenImage21); clear ini_latent only once the positive prompt consumes it.
            if not prompt.is_negative_prompt:
                self.ini_latent = None
            return self.get_learned_conditioning_with_image(prompt, _references)

        if not prompt.is_negative_prompt:
            self.ini_latent = None
            self._vision_cache = None
            dynamic_args.ref_latents.clear()

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], images: list[torch.Tensor]):
        cache_key = tuple(id(img) for img in images)
        if self._vision_cache is not None and self._vision_cache[0] == cache_key:
            images_vl, ref_latents = self._vision_cache[1], self._vision_cache[2]
        else:
            images_vl, ref_latents = [], []
            for image in images:
                v, r = self.encode_vision(image)
                images_vl.append(v)
                ref_latents.append(r.squeeze(2))
            self._vision_cache = (cache_key, images_vl, ref_latents)

        dynamic_args.ref_latents = ref_latents.copy()
        return self.text_processing_engine_qwen(prompt, images=images_vl)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = image.movedim(-1, 1)  # b, c, h, w
        if samples.shape[1] == 1:
            samples = samples.expand(-1, 3, -1, -1)

        _vision = resize_reference(samples, 1, 768).movedim(1, -1)[:, :, :, :3]

        s = resize_reference(samples)
        sample = self.forge_objects.vae.encode(s.movedim(1, -1)[:, :, :, :3])
        _latent = self.forge_objects.vae.first_stage_model.process_in(sample)

        return (_vision, _latent)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if opts.krea2_do_reference:
            self.store_reference(x[:1].movedim(1, -1).mul(0.5).add(0.5).cpu())

        if not dynamic_args.is_referencing:
            return super().encode_first_stage(x)
