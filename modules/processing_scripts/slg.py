# https://github.com/Comfy-Org/ComfyUI/blob/v0.42.0/comfy_extras/nodes_slg.py

import re
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from backend.args import dynamic_args
from backend.nn.dit_registry import collect_workflow_advice, krea2_workflow_advice, moss_workflow_advice  # noqa: F401 -- re-exported for callers/tests
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


class SkipLayerGuidance(scripts.Script):
    def title(self):
        return "Skip Layer Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process_before_every_sampling(self, p: "StableDiffusionProcessing", *args, **kwargs):
        comment = getattr(p, "comment", None)
        if callable(comment) and not getattr(p, "is_hr_pass", False):
            for note in collect_workflow_advice(p, opts=opts):
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
