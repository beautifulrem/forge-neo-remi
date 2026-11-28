import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.z_image_engine import Qwen34BEngine

ZIMAGE_MAX_REFERENCES = 4


class ZImage(ForgeDiffusionEngine):
    """Z-Image family engine (Base, Turbo, Omni / Edit) on ``NextDiT`` + Qwen3-4B + FLUX VAE."""

    matched_guesses = [model_list.ZImage]
    is_zimage_engine = True

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        from backend.nn.dit_capabilities_ext import register_extended_dit_capabilities

        register_extended_dit_capabilities()

        clip = CLIP(model_dict={"qwen3": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"])

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen34BEngine(
            text_encoder=clip.cond_stage_model.qwen3,
            tokenizer=clip.tokenizer.qwen3,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True
        repo = str(getattr(estimated_config, "huggingface_repo", "") or "").lower()
        self.is_zimage_turbo = "turbo" in repo
        self.is_zimage_omni = "omni" in repo or "edit" in repo
        self.zimage_do_reference = True

    def references(self) -> list[torch.Tensor]:
        """Return attached Z-Image-Omni / Edit reference latents (img2img first, then ImageStitch)."""

        refs = [*self.ref_latents]
        if self.ini_latent is not None:
            refs.insert(0, self.ini_latent)
        return refs[:ZIMAGE_MAX_REFERENCES]

    def clear_references(self):
        dynamic_args.zimage = False
        dynamic_args.ref_latents.clear()
        dit = getattr(getattr(getattr(self, "forge_objects", None), "unet", None), "model", None)
        dit = getattr(dit, "diffusion_model", None)
        if callable(clear := getattr(dit, "clear_caches", None)):
            clear()
        super().clear_references()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        is_neg = bool(getattr(prompt, "is_negative_prompt", False))
        if not is_neg:
            refs = self.references()
            if refs and getattr(self, "zimage_do_reference", True):
                dynamic_args.zimage = True
                dynamic_args.ref_latents = refs
                self.ini_latent = None
            else:
                dynamic_args.zimage = False
                dynamic_args.ref_latents.clear()
        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if getattr(self, "zimage_do_reference", False) and (getattr(self, "is_zimage_omni", False) or bool(self.ref_latents)):
            encoded = super().encode_first_stage(x)
            if encoded is not None:
                self.store_reference(encoded)
            if not dynamic_args.is_referencing:
                return encoded
            return None
        return super().encode_first_stage(x)
