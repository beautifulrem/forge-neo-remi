from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

import networks

from modules import extra_networks, shared


_ANIMA_SIZES = {"2b": 28, "2.9b": 40, "3.8b": 52}


def anima_source_blocks(value) -> int | None:
    """The ``src=`` of a ``<lora:…>`` tag: the Anima depth the LoRA was trained
    for, as blocks (28/40/52) or the model size (2b/2.9b/3.8b); None when
    absent. Raises ValueError for anything else."""

    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return None
    if text in _ANIMA_SIZES:
        return _ANIMA_SIZES[text]
    if text.isdigit() and int(text) in _ANIMA_SIZES.values():
        return int(text)
    raise ValueError(f"src={value}: expected 28, 40 or 52 (or 2b, 2.9b, 3.8b)")


class ExtraNetworkLora(extra_networks.ExtraNetwork):
    remove_symbols = str.maketrans("", "", ":,")

    def __init__(self):
        super().__init__("lora")

    def activate(self, p: "StableDiffusionProcessing", params_list: list["extra_networks.ExtraNetworkParams"]):
        additional: str = shared.opts.sd_lora
        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names: list[str] = []
        te_multipliers: list[float] = []
        unet_multipliers: list[float] = []
        anima_sources: list = []

        for params in params_list:
            assert params.items

            names.append(params.positional[0])

            te_multiplier = (0.0 if "@" in str(params.positional[1]) else float(params.positional[1])) if len(params.positional) > 1 else 1.0
            te_multiplier = float(params.named.get("te", te_multiplier))

            unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
            unet_multiplier = float(params.named.get("unet", unet_multiplier))

            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            try:
                anima_sources.append(anima_source_blocks(params.named.get("src")))
            except ValueError as e:
                p.comment(f"LoRA {params.positional[0]}: {e}")
                anima_sources.append(None)

        failed = networks.load_networks(names, te_multipliers, unet_multipliers, anima_sources)
        if failed:
            # a job warning (and a note under the image), not just a console line; a LoRA
            # that was found but could not be read also names the reason
            summary = ", ".join(name if reason == "not found" else f"{name} ({reason})" for name, reason in failed.items())
            networks.logger.warning(f"Skipped LoRAs: {summary}")
            p.comment("LoRA not found or failed to load: " + summary)

        if networks.remap_notes:
            # Read back by the infotext paste, so the same image re-maps the same way
            p.extra_generation_params["Lora remap"] = ", ".join(f"{name.translate(self.remove_symbols)}: {note.split(',')[0]}" for name, note in networks.remap_notes.items())

        if shared.opts.lora_add_hashes_to_infotext:
            if not getattr(p, "is_hr_pass", False) or not hasattr(p, "lora_hashes"):
                p.lora_hashes = {}

            for item in networks.loaded_networks:
                if item.network_on_disk.shorthash and item.mentioned_name:
                    p.lora_hashes[item.mentioned_name.translate(self.remove_symbols)] = item.network_on_disk.shorthash

            if p.lora_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(f"{k}: {v}" for k, v in p.lora_hashes.items())

    def deactivate(self, p: "StableDiffusionProcessing"):
        pass  # failures are reported by activate() (also on the Hires pass)
