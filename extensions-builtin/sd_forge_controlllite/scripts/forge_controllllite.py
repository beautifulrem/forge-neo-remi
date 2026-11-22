import logging
import os

import torch
import torch.nn.functional as F
from lib_controllllite.anima_control_lora import ControlEmbedder, ControlInjection, is_control_lora, split_control_lora
from lib_controllllite.anima_vace import VACEControlNetAnima, VACEInjection, is_anima_vace
from lib_controllllite.lib_controllllite import LLLiteLoader, sampler_progress, step_active
from lib_controllllite.lib_controllllite_anima import (
    ControlNetLLLiteDiT,
    infer_anima_config,
    load_lllite_weights_from_dict,
    map_blocks,
)

from backend import memory_management
from backend.args import dynamic_args
from backend.utils import load_torch_file
from modules import spectrum_force
from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher

logger = logging.getLogger("ControlNet")


def _request_spectrum_window(unet, start_percent: float, end_percent: float, label: str, *, clone: bool = False):
    """Tell Spectrum to run actual forwards across a windowed control's active steps
    so it does not extrapolate across the control's on/off switch boundaries."""

    if start_percent <= 0.0 and end_percent >= 1.0:
        return unet
    if not hasattr(unet, "model_options"):
        return unet
    if clone and hasattr(unet, "clone"):
        unet = unet.clone()
    predictor = getattr(getattr(unet, "model", None), "predictor", None)
    if predictor is not None and hasattr(predictor, "percent_to_sigma"):
        reason = spectrum_force.sigma_window(predictor.percent_to_sigma(start_percent), predictor.percent_to_sigma(end_percent))
    else:
        reason = label
    spectrum_force.request(unet, reason)
    return unet


class ControlLLLiteAnimaPatcher(ControlModelPatcher):
    anima_compatible = True

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any(k.startswith("lllite_dit") for k in state_dict):
            return None
        _, metadata = load_torch_file(ckpt_path, return_metadata=True)
        inpaint_masked_input: bool = (metadata or {}).get("lllite.inpaint_masked_input", None) == "true"
        return ControlLLLiteAnimaPatcher(state_dict, inpaint=inpaint_masked_input)

    def __init__(self, state_dict: dict[str, torch.Tensor], inpaint: bool):
        super().__init__()
        self.state_dict = state_dict
        self._is_inpaint = inpaint
        self._lllite_net: ControlNetLLLiteDiT = None
        self._dit = None
        self._skipped = False
        self._swapped = False  # a Hires run: the net may hold a DiT the next batch unloads

    @staticmethod
    def _is_black_on_white(image: torch.Tensor) -> bool:
        grey = image.mean(dim=1).squeeze(0)
        white_ratio = (grey > 0.9).float().mean().item()
        return white_ratio > 0.85

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet
        device, dtype = unet.load_device, unet.model.computation_dtype

        dit = unet.model.diffusion_model
        if getattr(process, "enable_hr", False):
            # A hires pass may reload the model (a Hires checkpoint or modules), also for a unit only in
            # one of the passes: the net would hold that pass's DiT across the next batch's reload
            self._swapped = True
        if self._lllite_net is None and self._dit is dit and self._skipped:
            return  # the hires checkpoint this net does not fit, again (see below)
        rebuilding = self._lllite_net is not None and self._dit is not dit
        if rebuilding:
            # A hires checkpoint swapped the DiT: the net wraps the first pass's modules, which the
            # hires pass never runs (the control would be silently missing). Build it over this one.
            self._lllite_net.restore()
            self._lllite_net = None
            self._swapped = True  # (also a reload for the hires modules, same checkpoint)

        if self._lllite_net is None:
            try:
                view, weights = map_blocks(dit, self.state_dict)
                net = ControlNetLLLiteDiT(view, **infer_anima_config(weights))
                load_lllite_weights_from_dict(net, weights)
            except (AttributeError, NotImplementedError, RuntimeError) as e:
                if not getattr(process, "is_hr_pass", False):
                    raise
                # A hires checkpoint this net does not fit (not Anima, another block count or width):
                # that pass runs without the control, as it did before the net followed the DiT (also
                # for a unit only in the hires pass, and the upscale button, which has no first pass)
                logger.warning(f"Control-LLLite (Anima) skipped for the hires checkpoint: {e}")
                self._dit, self._skipped = dit, True
                return
            self._lllite_net = net.eval().to(device=device, dtype=dtype)
            self._dit, self._skipped = dit, False
        else:
            # Re-upload the cached net if the previous pass offloaded it to CPU.
            self._lllite_net.to(device=device, dtype=dtype)

        if mask is not None and getattr(process, "inpainting_mask_invert", False):
            mask = 1.0 - mask

        if (not self._is_inpaint) and (mask is not None):
            if inv := self._is_black_on_white(cond):
                cond = 1.0 - cond
            cond = cond * mask  # not in place: cond is the unit's cached control, used by every pass
            if inv:
                cond = 1.0 - cond

        cond_image = cond * 2.0 - 1.0
        if self._is_inpaint:
            assert isinstance(mask, torch.Tensor)
            if mask.shape != cond_image.shape:
                mask = F.interpolate(mask, size=(cond_image.shape[2], cond_image.shape[3]), mode="nearest")
            inpaint_mask = mask.to(device=cond_image.device, dtype=cond_image.dtype)
            cond_image = cond_image * (inpaint_mask < 0.5)
            inpaint_mask = inpaint_mask * 2.0 - 1.0
            cond_image = torch.cat([cond_image, inpaint_mask], dim=1)

        process.sd_model.forge_objects.unet = _request_spectrum_window(unet, self.start_percent, self.end_percent, "anima_control_lllite", clone=True)
        self._lllite_net.set_cond_image(cond_image.to(device=device, dtype=dtype))
        self._lllite_net.set_multiplier(self.strength)
        self._lllite_net.set_step_range(num_steps=process.steps, start_percent=self.start_percent, end_percent=self.end_percent, progress=sampler_progress(process))
        self._lllite_net.apply_to()

    def process_after_every_sampling(self, *args, **kwargs):
        if self._lllite_net is not None:
            self._lllite_net.restore()
            if not self._swapped:
                self._lllite_net.to(device=memory_management.unet_offload_device())
        if self._swapped:
            # A Hires run: the net and _dit would keep one of the two DiTs' modules alive
            # while the next batch reloads the other (the next batch builds the net again)
            self._lllite_net, self._dit, self._skipped, self._swapped = None, None, False, False


class ControlLLLitePatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any(k.startswith("lllite") for k in state_dict):
            return None
        return ControlLLLitePatcher(state_dict)

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        if mask is not None:
            if getattr(process, "inpainting_mask_invert", False):
                mask = 1.0 - mask
            cond = cond * mask  # not in place: cond is the unit's cached control, used by every pass

        unet = LLLiteLoader.load_lllite(model=unet, state_dict=self.state_dict, cond_image=cond.movedim(1, -1), strength=self.strength, steps=process.steps, start_percent=self.start_percent, end_percent=self.end_percent, progress=sampler_progress(process))

        process.sd_model.forge_objects.unet = unet


class AnimaControlLoraPatcher(ControlModelPatcher):
    """Anima control-LoRA (``lib_controllllite.anima_control_lora``): the unit's weight scales the
    control image's embedding and its start / end bound it; the LoRA trained with it is applied
    at 1.0 for the whole pass, to the DiT and (before the prompts are encoded) the LLM adapter."""

    anima_compatible = True

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not is_control_lora(state_dict):
            return None
        return AnimaControlLoraPatcher(state_dict, ckpt_path)

    def __init__(self, state_dict: dict[str, torch.Tensor], ckpt_path: str):
        super().__init__()
        embedder, self.lora = split_control_lora(state_dict)
        self.embedder = ControlEmbedder.from_state_dict(embedder)
        self.filename = str(ckpt_path)
        self._injection: ControlInjection = None

    def _patched(self, unet, clip):
        import networks  # sd_forge_lora: depth re-mapping, LLM adapter routing, 3.8B mirroring

        if not self.lora:
            return unet, clip
        try:
            return networks.load_lora_for_models(
                unet,
                clip,
                dict(self.lora),
                1.0,
                1.0,
                online_mode=bool(dynamic_args.online_lora),
                filename=self.filename,
            )
        except networks.LoraNotApplied as error:
            raise ValueError(f"Anima control-LoRA {self.filename}: its LoRA does not fit this model ({error})") from error

    def process_before_conditioning(self, process) -> bool:
        objects = process.sd_model.forge_objects
        _, patched = self._patched(objects.unet, objects.clip)
        changed = patched is not objects.clip
        objects.clip = patched
        return changed

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        objects = process.sd_model.forge_objects
        orig_unet = objects.unet
        objects.unet, _ = self._patched(objects.unet, objects.clip)
        unet = _request_spectrum_window(
            objects.unet,
            self.start_percent,
            self.end_percent,
            "anima_control_lora",
            clone=objects.unet is orig_unet,
        )
        objects.unet = unet

        if mask is not None:
            if getattr(process, "inpainting_mask_invert", False):
                mask = 1.0 - mask
            cond = cond * mask  # not in place: cond is the unit's cached control, used by every pass

        # The Wan VAE takes a 4D batch as the frames of one video: one image at a time
        images = cond.movedim(1, -1)
        latent = torch.cat([objects.vae.encode(images[i : i + 1]) for i in range(images.shape[0])])
        device, dtype = unet.load_device, unet.model.computation_dtype
        try:
            with torch.inference_mode():
                tokens = self.embedder.to(device=device, dtype=dtype)(latent.to(device=device, dtype=dtype))
        finally:
            # The embedder only runs once before sampling; return it to the offload device immediately.
            self.embedder.to(device=memory_management.unet_offload_device())

        progress = sampler_progress(process)
        start, end, hires = self.start_percent, self.end_percent, getattr(process, "is_hr_pass", False)

        def active() -> bool:
            if getattr(process, "is_hr_pass", False) != hires:
                return False  # a pass the unit does not run in (its Hires option)
            return step_active(progress, start, end) is not False

        if self._injection is not None:
            self._injection.detach()
        self._injection = ControlInjection(tokens, self.strength, active)
        self._injection.attach(unet.model.diffusion_model)

    def process_after_every_sampling(self, *args, **kwargs):
        if self._injection is not None:
            self._injection.detach()
            self._injection = None
        self.embedder.to(device=memory_management.unet_offload_device())


class AnimaControlVacePatcher(ControlModelPatcher):
    """Anima VACE ControlNet (spaced-block duplication + zero-conv injection):
    encodes the control image via the Wan VAE + ``Wan21.process_in``, runs the
    parallel VACE blocks before DiT block 0, and injects scaled residual hints
    after the mapped base blocks (supporting 2B, 2.9B and 3.8B depths)."""

    anima_compatible = True

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not is_anima_vace(state_dict):
            return None
        metadata = {}
        if ckpt_path and os.path.isfile(ckpt_path) and str(ckpt_path).lower().endswith((".safetensors", ".sft")):
            try:
                _, raw_meta = load_torch_file(ckpt_path, return_metadata=True)
                metadata = dict(raw_meta or {})
            except Exception:
                metadata = {}
        return AnimaControlVacePatcher(state_dict, metadata=metadata)

    def __init__(self, state_dict: dict[str, torch.Tensor], metadata: dict | None = None):
        super().__init__()
        self.vace = VACEControlNetAnima.from_state_dict(state_dict, metadata=metadata)
        self._injection: VACEInjection = None

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        objects = process.sd_model.forge_objects
        unet = objects.unet
        dit = unet.model.diffusion_model
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            if getattr(process, "is_hr_pass", False):
                logger.warning("Anima VACE ControlNet skipped for the hires checkpoint: not an Anima DiT")
                return
            raise ValueError("Anima VACE ControlNet requires an active Anima checkpoint")

        try:
            target_layers = self.vace.target_layers_for_dit(len(blocks))
        except ValueError as error:
            if getattr(process, "is_hr_pass", False):
                logger.warning(f"Anima VACE ControlNet skipped for the hires checkpoint: {error}")
                return
            raise

        unet = _request_spectrum_window(unet, self.start_percent, self.end_percent, "anima_vace", clone=True)
        objects.unet = unet

        if mask is not None and getattr(process, "inpainting_mask_invert", False):
            mask = 1.0 - mask

        # Wan VAE encodes 4D batches as video frames: encode one image at a time,
        # then normalise into the DiT's Wan21 latent space (matching sd-scripts training).
        images = cond.movedim(1, -1)
        latent = torch.cat([objects.vae.encode(images[i : i + 1]) for i in range(images.shape[0])])
        latent_format = getattr(getattr(process.sd_model, "model_config", None), "latent_format", None)
        if latent_format is not None and hasattr(latent_format, "process_in"):
            latent = latent_format.process_in(latent)
        else:
            from modules_forge.packages.huggingface_guess.latent import Wan21

            latent = Wan21().process_in(latent)

        device, dtype = unet.load_device, unet.model.computation_dtype
        self.vace.to(device=device, dtype=dtype)
        latent = latent.to(device=device, dtype=dtype)

        progress = sampler_progress(process)
        start, end, hires = self.start_percent, self.end_percent, getattr(process, "is_hr_pass", False)

        def active() -> bool:
            if getattr(process, "is_hr_pass", False) != hires:
                return False
            return step_active(progress, start, end) is not False

        if self._injection is not None:
            self._injection.detach()
        self._injection = VACEInjection(self.vace, latent, self.strength, target_layers, mask=mask, active=active)
        self._injection.attach(dit)

    def process_after_every_sampling(self, *args, **kwargs):
        if self._injection is not None:
            self._injection.detach()
            self._injection = None
        self.vace.to(device=memory_management.unet_offload_device())


add_supported_control_model(ControlLLLitePatcher)

add_supported_control_model(ControlLLLiteAnimaPatcher)

add_supported_control_model(AnimaControlLoraPatcher)

add_supported_control_model(AnimaControlVacePatcher)
