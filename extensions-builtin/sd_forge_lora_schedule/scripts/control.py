import logging
import re

from networks import available_network_aliases

from backend.args import dynamic_args
from backend.logging import setup_logger
from backend.patcher.base import ModelPatcher, OnlineLoRAPatch
from modules import scripts, shared
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser

logger = logging.getLogger("lora_ctl")
setup_logger(logger)

lora_ctl = re.compile(r"<lora:([^\:\>]+):\[([^\]\>]+)\]>")


class LoRAControl(scripts.Script):
    mapping: dict[str, list[tuple[float, float]]] = {}

    def title(self):
        return "LoRA Control Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @staticmethod
    def _parse_schedule(syntax: str) -> list[tuple[float, float]]:
        return [tuple(map(float, chunk.split("@"))) for chunk in syntax.split(",")]

    @staticmethod
    def _prompt_texts(p: StableDiffusionProcessing) -> list[str]:
        all_prompts = getattr(p, "all_prompts", None)
        if isinstance(all_prompts, (list, tuple)) and all_prompts:
            return [text for text in all_prompts if isinstance(text, str)]
        prompt = getattr(p, "prompt", None)
        if isinstance(prompt, str):
            return [prompt]
        if isinstance(prompt, (list, tuple)):
            return [text for text in prompt if isinstance(text, str)]
        return []

    def _collect_mapping(self, p: StableDiffusionProcessing) -> None:
        LoRAControl.mapping.clear()

        prompts = self._prompt_texts(p)
        if not prompts:
            return

        ctl = False
        for prompt_text in prompts:
            for m in re.finditer(lora_ctl, prompt_text):
                ctl = True
                alias: str = m.group(1)

                try:
                    lora: str = available_network_aliases[alias].filename
                    schedule = self._parse_schedule(m.group(2))
                    assert len(schedule) > 1 and all(len(point) == 2 for point in schedule)  # weight@time points
                    LoRAControl.mapping[lora] = schedule
                except Exception:
                    logger.error(f'Invalid Syntax for "{alias}": "{m.group(2)}"')
                    continue

        if ctl and not dynamic_args.online_lora:
            logger.error("LoRA Control requires on-the-fly Patching")
            LoRAControl.mapping.clear()

    def before_process(self, p: StableDiffusionProcessing):
        self._collect_mapping(p)

    def process(self, p: StableDiffusionProcessing, *args):
        if getattr(p, "all_prompts", None):
            self._collect_mapping(p)

    def postprocess(self, p: StableDiffusionProcessing, processed, *args):
        LoRAControl.mapping.clear()

    def cleanup(self, p: StableDiffusionProcessing, *args, **kwargs):
        LoRAControl.mapping.clear()

    @classmethod
    def adjust_lora(cls, params: CFGDenoiserParams):
        if not cls.mapping or not params.total_sampling_steps:
            return

        sd_model = getattr(shared, "sd_model", None)
        forge_objects = getattr(sd_model, "forge_objects", None)
        m: ModelPatcher | None = getattr(forge_objects, "unet", None)
        if m is None or not m.has_online_lora():
            return

        patches: list[list[OnlineLoRAPatch]] = m.weight_wrapper_patches.values()
        t: float = params.sampling_step / params.total_sampling_steps

        for loras in patches:
            for lora in loras:
                for name, schedule in cls.mapping.items():
                    if lora.name != name:
                        continue

                    w = schedule[0][0]  # before the first point: its weight
                    if schedule[-1][1] <= t:
                        w = schedule[-1][0]
                    else:
                        for i in range(len(schedule) - 1):
                            w1, t1 = schedule[i]
                            w2, t2 = schedule[i + 1]

                            if t1 <= t < t2:
                                w = w2 if t2 == t1 else w1 + (w2 - w1) * ((t - t1) / (t2 - t1))
                                break

                    lora.patch[0][0] = w


on_cfg_denoiser(LoRAControl.adjust_lora)
