# https://github.com/Comfy-Org/ComfyUI/blob/v0.42.0/comfy_extras/nodes_slg.py

import re
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from backend.args import dynamic_args
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling import sampling_function
from modules import scripts
from modules.script_callbacks import on_script_unloaded
from modules.shared import opts

_calc_cond_uncond_batch = sampling_function.calc_cond_uncond_batch


@wraps(_calc_cond_uncond_batch)
def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    if (calc_batch_function := model_options.get("sampler_calc_cond_batch_function", None)) is not None:
        return calc_batch_function(model, cond, uncond, x_in, timestep, model_options)
    return _calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options)


sampling_function.calc_cond_uncond_batch = calc_cond_uncond_batch


def parse_layers(layers: str) -> list[int]:
    return [int(layer) for layer in re.findall(r"\d+", layers)]


_MOSS_FEW_STEP_RE = re.compile(r"(?:^|[-_ /])(?:turbo|dmd|4[-_]?steps?|8[-_]?steps?|16[-_]?step)(?:$|[-_ /.])", re.IGNORECASE)
_MOSS_EDIT_RE = re.compile(r"(?:^|[-_ /])(?:moss[-_ ]?edit|moss[-_ ]?image[-_ ]?edit)(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_TURBO_RE = re.compile(r"(?:^|[-_ /])turbo(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_RAW_RE = re.compile(r"(?:^|[-_ /])raw(?:$|[-_ /.])", re.IGNORECASE)
_KREA2_EDIT_RE = re.compile(r"(?:^|[-_ /])(?:krea[-_ ]?2[-_ ]?edit|krea2edit)(?:$|[-_ /.])", re.IGNORECASE)


def _prompt_and_checkpoint_names(p: "StableDiffusionProcessing") -> tuple[str, list[str]]:
    sd_model = getattr(p, "sd_model", None)
    ckpt_name = str(getattr(getattr(sd_model, "sd_checkpoint_info", None), "name", "") or "")
    lora_names: list[str] = []
    for prompt_text in (getattr(p, "prompt", None), *(getattr(p, "all_prompts", None) or ())):
        if isinstance(prompt_text, str):
            lora_names.extend(re.findall(r"<lora:([^:>]+)", prompt_text))
    return ckpt_name, lora_names


def moss_workflow_advice(p: "StableDiffusionProcessing") -> list[str]:
    """Non-blocking hints for common MOSS-Image configuration pitfalls."""

    if not dynamic_args.moss:
        return []

    advice: list[str] = []
    sd_model = getattr(p, "sd_model", None)
    ref_mode = getattr(opts, "moss_do_reference", "Disable")
    has_refs = bool(getattr(sd_model, "ref_latents", None)) or getattr(sd_model, "ini_latent", None) is not None
    if has_refs and ref_mode == "Disable":
        advice.append(
            "MOSS-Image: reference images are attached while [Moss-Image] Reference Method is 'Disable'; "
            "switch it to 'Moss-Image-Edit' (for chinoll/Edit-HQ LoRAs) or 'ai-toolkit' (for vision+latent Edit LoRAs)."
        )

    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    combined = " ".join([ckpt_name, *lora_names]).strip()

    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if combined and _MOSS_FEW_STEP_RE.search(combined):
        if (isinstance(steps, (int, float)) and steps >= 30) or (isinstance(cfg, (int, float)) and cfg > 2.0):
            advice.append(
                f"MOSS-Image Turbo/DMD detected with steps={steps}, cfg={cfg}: distilled/Turbo variants "
                "typically work best at 4-16 steps and CFG 1.0 (with SLG automatically disabled at CFG 1)."
            )
    if combined and _MOSS_EDIT_RE.search(combined) and ref_mode == "Disable" and not has_refs:
        advice.append(
            "MOSS-Image Edit LoRA detected while [Moss-Image] Reference Method is 'Disable'; "
            "enable 'Moss-Image-Edit' or 'ai-toolkit' in Settings when supplying reference images."
        )
    return advice


def krea2_workflow_advice(p: "StableDiffusionProcessing") -> list[str]:
    """Non-blocking hints for common Krea 2 Raw / Turbo / Edit configuration pitfalls."""

    if not dynamic_args.krea2:
        return []

    advice: list[str] = []
    sd_model = getattr(p, "sd_model", None)
    do_ref = bool(getattr(opts, "krea2_do_reference", False))
    has_refs = bool(getattr(sd_model, "ref_latents", None)) or getattr(sd_model, "ini_latent", None) is not None
    if has_refs and not do_ref:
        advice.append(
            "Krea 2: reference images are attached while [Krea2] Enable Reference is off; "
            "enable it in Settings when using a Krea 2 Edit LoRA."
        )

    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    combined = " ".join([ckpt_name, *lora_names]).strip()
    is_turbo = bool(combined and _KREA2_TURBO_RE.search(combined))
    is_raw_only = bool(ckpt_name and _KREA2_RAW_RE.search(ckpt_name)) and not is_turbo

    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if is_turbo:
        if (isinstance(steps, (int, float)) and steps >= 20) or (isinstance(cfg, (int, float)) and cfg > 1.5):
            advice.append(
                f"Krea-2-Turbo detected with steps={steps}, cfg={cfg}: Turbo is step-distilled and "
                "typically works best at ~8 steps with CFG 1.0."
            )
    elif is_raw_only:
        if (isinstance(steps, (int, float)) and steps <= 12) or (isinstance(cfg, (int, float)) and cfg <= 1.0):
            advice.append(
                f"Krea-2-Raw detected with steps={steps}, cfg={cfg}: the undistilled Raw base typically "
                "needs ~28-52 steps and CFG 3.5-4.5 (or a Krea-2 Turbo LoRA for 8-step CFG 1.0)."
            )
    if combined and _KREA2_EDIT_RE.search(combined) and not do_ref and not has_refs:
        advice.append(
            "Krea 2 Edit LoRA detected while [Krea2] Enable Reference is off; "
            "enable [Krea2] Enable Reference in Settings when supplying reference images."
        )
    return advice


class SkipLayerGuidance(scripts.Script):
    def title(self):
        return "Skip Layer Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process_before_every_sampling(self, p: "StableDiffusionProcessing", *args, **kwargs):
        if not (dynamic_args.moss or dynamic_args.krea2):
            return

        comment = getattr(p, "comment", None)
        if callable(comment) and not getattr(p, "is_hr_pass", False):
            for note in (*moss_workflow_advice(p), *krea2_workflow_advice(p)):
                comment(note)

        # opts.slg_enable ("[Moss-Image] Enable Skip Layer Guidance", default True, layer 22)
        # is specifically tuned for MOSS-Image's 42-layer DiT; never apply it silently to Krea-2-Raw.
        if not dynamic_args.moss or not opts.slg_enable:
            return

        # CFG <= 1.0 (e.g. MOSS-Image-Turbo / 4-step / 8-step DMD) has no unconditional branch.
        cfg_scale = getattr(p, "cfg_scale", None)
        if isinstance(cfg_scale, (int, float)) and cfg_scale <= 1.0:
            return

        double_layers = parse_layers("")  # Moss-Image has no double_block
        single_layers = parse_layers(opts.slg_single_layers)

        unet = p.sd_model.forge_objects.unet
        dit_layers = getattr(getattr(getattr(unet, "model", None), "diffusion_model", None), "layers", None)
        if dit_layers is not None:
            single_layers = [layer for layer in single_layers if 0 <= layer < len(dit_layers)]

        if len(double_layers) == 0 and len(single_layers) == 0:
            return

        p_start = max(min(opts.slg_start_percent, 1.0), 0.0)
        p_end = max(min(opts.slg_end_percent, 1.0), 0.0)

        predictor = unet.model.predictor
        sigma_start = predictor.percent_to_sigma(p_start)
        sigma_end = predictor.percent_to_sigma(p_end)

        calc_batch_function = self.make_calc_cond_batch_function(double_layers, single_layers, sigma_start, sigma_end)

        m = unet.clone()
        m.set_model_sampler_calc_cond_batch_function(calc_batch_function)
        p.sd_model.forge_objects.unet = m

        p.extra_generation_params.update(
            {
                "Skip Layer Guidance": True,
                "SLG Single Layers": ", ".join(map(str, single_layers)) if single_layers else None,
                "SLG Start Percent": p_start,
                "SLG End Percent": p_end,
            }
        )

    @staticmethod
    def make_calc_cond_batch_function(double_layers: list[int], single_layers: list[int], sigma_start: float, sigma_end: float):

        def skip(args, extra_args):
            return args

        def calc_batch_function(model, cond, uncond, x, sigma, model_options):
            sigma_ = sigma[0].item()

            if sigma_end <= sigma_ <= sigma_start and uncond is not None:
                slg_model_options = model_options.copy()
                for layer in double_layers:
                    slg_model_options = set_model_options_patch_replace(slg_model_options, skip, "dit", "double_block", layer)
                for layer in single_layers:
                    slg_model_options = set_model_options_patch_replace(slg_model_options, skip, "dit", "single_block", layer)
                cond_out, _ = _calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)
                _, uncond_out = _calc_cond_uncond_batch(model, [], uncond, x, sigma, slg_model_options)
                return [cond_out, uncond_out]

            return _calc_cond_uncond_batch(model, cond, uncond, x, sigma, model_options)

        return calc_batch_function


def revert():
    sampling_function.calc_cond_uncond_batch = _calc_cond_uncond_batch


on_script_unloaded(revert)
