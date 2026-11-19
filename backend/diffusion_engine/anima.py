import logging
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
from backend.text_processing.anima_engine import Qwen06Engine
from modules.shared import opts

logger = logging.getLogger(__name__)


def qwen35_tokenizer():
    """Qwen3.5's tokenizer (Qwen/Qwen3.5-4B), for Anima-3.8B's second text encoder."""

    from tokenizers import Tokenizer

    from backend.huggingface import Token

    return Tokenizer.from_file(Token.qwen35)


class Anima(ForgeDiffusionEngine):
    is_anima_engine = True
    matched_guesses = [model_list.Anima]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        transformer = huggingface_components["transformer"]
        semantic_layers = transformer.config.get("semantic_connector", {}).get("layers") if getattr(transformer, "anima_v2_connector", None) is not None else None
        qwen35 = huggingface_components.get("text_encoder_2")
        if semantic_layers and qwen35 is None:
            raise ValueError("Anima-3.8B needs its Qwen3.5 4B text encoder: add qwen35_4b.safetensors (from lylogummy/Anima-3.8B, in models/text_encoder) to the VAE / Text Encoder modules")
        if qwen35 is not None and not semantic_layers:
            logger.warning("Qwen3.5 4B is only used by Anima-3.8B: ignored for this model")
            qwen35 = None

        models = {"qwen3_06b": huggingface_components["text_encoder"]}
        if qwen35 is not None:
            models["qwen35_4b"] = qwen35
        clip = CLIP(model_dict=models, tokenizer_dict={"qwen3_06b": huggingface_components["tokenizer"], "t5xxl": huggingface_components["tokenizer_2"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_anima = Qwen06Engine(
            text_encoder=clip.cond_stage_model.qwen3_06b,
            qwen_tokenizer=clip.tokenizer.qwen3_06b,
            t5_tokenizer=clip.tokenizer.t5xxl,
            semantic_encoder=getattr(clip.cond_stage_model, "qwen35_4b", None),
            semantic_tokenizer=qwen35_tokenizer() if qwen35 is not None else None,
            semantic_layers=semantic_layers,
        )
        self.is_anima_semantic = qwen35 is not None

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True
        self.use_shift = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            if not opts.anima_do_reference:
                dynamic_args.ref_latents.clear()
            else:
                _references = [*self.ref_latents]
                if self.ini_latent is not None:
                    _references.insert(0, self.ini_latent)
                    self.ini_latent = None
                dynamic_args.ref_latents = _references.copy()

        engine = self.text_processing_engine_anima
        # modules.anima_support.native_text_encoding: extra prompts (Regional, Artist Mixer)
        semantic = engine.uses_semantic(prompt.is_negative_prompt) and not getattr(self, "anima_native_text_only", False)
        if semantic and prompt.is_negative_prompt:
            dynamic_args.last_extra_generation_params["Anima negative connector"] = True
        return engine(prompt, semantic=semantic)

    # Upstream Neo renamed the attribute; NegPiP, FreeFuse and the 3.8B
    # extension read text_processing_engine_anima, so both names share one engine
    @property
    def text_processing_engine_qwen(self) -> Qwen06Engine:
        return self.text_processing_engine_anima

    @text_processing_engine_qwen.setter
    def text_processing_engine_qwen(self, engine: Qwen06Engine):
        self.text_processing_engine_anima = engine

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_anima.tokenize([prompt])[0][0])
        return token_count, max(512, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor, *, multiple: int = -1):
        # ``multiple``: Anima's own, from the anima_do_reference setting
        if opts.anima_do_reference and not dynamic_args.is_referencing:
            # Remi: the img2img image is the first reference. Anima stacks the
            # references along the frames, so they need the generation's size:
            # keep the image's own latent instead of one resized to
            # reference_max_side (upstream caaa2cca), which would not stack
            samples = self._encode_first_stage(x)
            self.store_reference(samples[:1].clone().cpu())
            return samples
        return super().encode_first_stage(x, multiple=16 if opts.anima_do_reference else 0)
