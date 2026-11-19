"""sd_forge_lora's WebUI hooks: settings, the ``<lora:…>`` extra network,
infotext paste (hashes, AddNet fields) and the ``/sdapi/v1`` routes.

The card browser and the metadata editor are the SPA's network browser
(``remi.networks``, ``/api/v2/networks``); nothing here imports Gradio.
"""

import re

import extra_networks_lora
import lora  # noqa: F401
import network
import networks

from modules import extra_networks, script_callbacks, shared
from remi import ui

shared.options_templates.update(
    shared.options_section(
        ("extra_networks", "Extra Networks"),
        {
            "sd_lora": shared.OptionInfo("None", "Add network to prompt", ui.Choice, lambda: {"choices": ["None", *networks.available_networks]}, refresh=networks.list_available_networks),
            "lora_preferred_name": shared.OptionInfo("Alias from file", "When adding to prompt, refer to Lora by", ui.Choice, {"choices": ["Alias from file", "Filename"], "style": "radio"}),
            "lora_add_hashes_to_infotext": shared.OptionInfo(True, "Add Lora hashes to infotext"),
            "lora_preset_filter": shared.OptionInfo(False, "Filter Lora based on selected Preset").info("the network browser starts with its \"compatible\" filter on"),
            "anima_lora_inserted_blocks": shared.OptionInfo("copy", "Anima LoRA on a deeper model (2B LoRA on 2.9B / 3.8B): inserted blocks", ui.Choice, {"choices": ["copy", "skip"], "style": "radio"}).info("copy: an inserted block gets the LoRA of the block it was copied from; skip: only the original blocks get it."),
        },
    )
)


def create_lora_json(obj: network.NetworkOnDisk) -> dict:
    return {
        "name": obj.name,
        "alias": obj.alias,
        "path": obj.filename,
        "metadata": obj.metadata,
    }


def refresh_loras() -> None:
    networks.list_available_networks()
    try:
        from remi.networks import notify_changed

        notify_changed("lora")
    except ImportError:
        pass


def api_networks(_demo, app) -> None:
    @app.get("/sdapi/v1/loras")
    async def get_loras():
        return [create_lora_json(obj) for obj in networks.available_networks.values()]

    @app.post("/sdapi/v1/refresh-loras")
    async def post_refresh_loras():
        return refresh_loras()


if shared.cmd_opts.api:
    script_callbacks.on_app_started(api_networks)


re_lora = re.compile("<lora:([^:]+):")


def before_ui():
    networks.extra_network_lora = extra_networks_lora.ExtraNetworkLora()
    extra_networks.register_extra_network(networks.extra_network_lora)


re_lora_tag = re.compile(r"<lora:([^:>]+)((?::[^:>]*)*)>")
re_remap = re.compile(r"\s*([^:,]+):\s*(\d+)\s*→\s*\d+")


def remap_pasted(infotext, d: dict):
    """``Lora remap: name: 28→40``: the LoRA was placed as a 28-block one, so the
    pasted prompt says so (``src=28``) and re-maps it the same way again."""

    remap = d.get("Lora remap")
    prompt = d.get("Prompt")
    if not remap or not isinstance(prompt, str):
        return
    sources = {m.group(1).strip(): m.group(2) for m in re_remap.finditer(str(remap))}

    def add_source(m: re.Match) -> str:
        name, rest = m.group(1), m.group(2)
        source = sources.get(name.translate(extra_networks_lora.ExtraNetworkLora.remove_symbols))
        if source is None or ":src=" in rest:
            return m.group(0)
        return f"<lora:{name}{rest}:src={source}>"

    d["Prompt"] = re_lora_tag.sub(add_source, prompt)


def infotext_pasted(infotext, d: dict):
    hashes = d.get("Lora hashes")
    if not hashes:
        return

    hashes = [x.strip().split(":", 1) for x in hashes.split(",")]
    hashes = {x[0].strip().replace(",", ""): x[1].strip() for x in hashes if len(x) == 2}

    def network_replacement(m: re.Match) -> str:
        alias = m.group(1)
        shorthash = hashes.get(alias)
        if shorthash is None:
            return m.group(0)

        network_on_disk = networks.available_network_hash_lookup.get(shorthash)
        if network_on_disk is None:
            return m.group(0)

        return f"<lora:{network_on_disk.get_alias()}:"

    d["Prompt"] = re.sub(re_lora, network_replacement, d["Prompt"])


script_callbacks.on_before_ui(before_ui)
script_callbacks.on_infotext_pasted(networks.infotext_pasted)
script_callbacks.on_infotext_pasted(infotext_pasted)
script_callbacks.on_infotext_pasted(remap_pasted)
