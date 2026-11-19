import logging
import os.path
import re
from typing import TYPE_CHECKING, Optional

import network
import torch

if TYPE_CHECKING:
    from backend.patcher.clip import CLIP
    from backend.patcher.unet import UnetPatcher

from backend.args import dynamic_args
from backend.logging import setup_logger
from backend.nn import anima_layout
from backend.patcher.lora import load_lora, model_lora_keys_clip, model_lora_keys_unet
from backend.state_dict import state_dict_prefix_replace
from backend.utils import load_torch_file
from modules import errors, scripts, sd_models, shared
from modules.anima_lora_support import RESOLVED_LORA_DEPTHS

logger = logging.getLogger("lora")
setup_logger(logger)


class LoraNotApplied(Exception):
    """The LoRA as a whole cannot go on this model (reported per generation, without a traceback)."""


def anima_inserted_blocks_policy() -> str:
    policy = getattr(shared.opts, "anima_lora_inserted_blocks", "copy")
    return policy if policy in anima_layout.INSERTED_POLICIES else "copy"


def process_anima(lora: dict[str, torch.Tensor], blocks: int, *, declared: Optional[int] = None, inserted: Optional[str] = None, notes: Optional[dict] = None) -> Optional[str]:
    """Fit an Anima LoRA's keys to a ``blocks``-deep DiT, in place.

    Moves the LLM adapter keys to the text encoder (where Remi loads the
    adapter), and re-maps the DiT block indices when the LoRA was trained for
    another depth of the 2B / 2.9B / 3.8B family (``backend.nn.anima_layout``).
    The trained depth is ``declared`` (the prompt's ``src=`` or the file's
    metadata) or read from the block indices.

    Returns a short note of the re-mapping for the infotext (``"28→40"``), or
    None when nothing was re-mapped; ``notes["depth"]`` receives the depth the
    LoRA was taken as. Raises ``LoraNotApplied`` when the LoRA cannot be
    placed on this model.
    """

    # LLMAdapter was moved from transformer to text_encoder

    keys = list(lora.keys())
    for k in keys:
        for dotted in ("diffusion_model.", "transformer.", "net.", ""):  # "": bare keys
            if k.startswith(dotted + "llm_adapter"):
                lora["text_encoders.qwen3_06b." + k[len(dotted) :]] = lora.pop(k)
                break
        else:
            for flat in ("lora_unet_", "lycoris_", "lora_transformer_"):
                if k.startswith(flat + "llm_adapter"):
                    lora["lora_te_" + k[len(flat) :]] = lora.pop(k)
                    break
    keys = list(lora.keys())

    parsed: dict[str, tuple[int, str]] = {}
    # Block-key prefixes of the trainers' formats (see model_lora_keys_unet);
    # the most common one in this LoRA is the one to re-map.
    prefixes = (
        ("lora_unet_blocks_", "_"),
        ("lycoris_blocks_", "_"),
        ("lora_transformer_blocks_", "_"),
        ("diffusion_model.blocks.", "."),
        ("transformer.blocks.", "."),
        ("net.blocks.", "."),
        ("blocks.", "."),
    )
    prefix, sep = max(prefixes, key=lambda item: sum(1 for k in keys if k.startswith(item[0])))

    for k in keys:
        if not k.startswith(prefix):
            continue
        num, s, tail = k[len(prefix) :].partition(sep)
        if s and num.isdigit():
            parsed[k] = (int(num), tail)

    if not parsed:  # a text-encoder-only LoRA fits every depth
        return None

    try:
        lora_blocks, guessed = anima_layout.trained_depth({n for n, _ in parsed.values()}, declared=declared)
    except anima_layout.AnimaDepthError as e:
        raise LoraNotApplied(str(e)) from None
    if notes is not None:
        notes["depth"] = lora_blocks

    if lora_blocks == blocks:
        return None

    policy = inserted or anima_inserted_blocks_policy()
    if (layout := anima_layout.block_map(lora_blocks, blocks, policy)) is None:  # type: ignore[arg-type]
        raise LoraNotApplied(f"no Anima block layout maps {lora_blocks} blocks to {blocks}")
    targets, dropped = layout

    if guessed:
        logger.warning(f"Anima LoRA depth guessed as {lora_blocks} blocks from its block indices; write <lora:name:weight:src=40> (or 52) if it was trained on 2.9B (or 3.8B)")
    note = f"{lora_blocks}→{blocks}"
    if dropped:
        note += f", dropped blocks {','.join(map(str, dropped))}"
        logger.warning(f"Re-Mapping Anima LoRA ({lora_blocks} to {blocks}): {len(dropped)} blocks the smaller model lacks are dropped")
    else:
        logger.info(f"Re-Mapping Anima LoRA ({lora_blocks} to {blocks}, inserted blocks: {policy})")

    # The source keys go first: a re-mapped key may reuse the name of another source key.
    # Inserted blocks share their source block's tensors ("copy"): patches only read
    # them, so a copy per target would just double the LoRA's memory.
    source = {k: lora.pop(k) for k in parsed}
    for k, (block, tail) in parsed.items():
        for target in targets.get(block, ()):
            lora[f"{prefix}{target}{sep}{tail}"] = source[k]
    return note


def load_lora_for_models(model: "UnetPatcher", clip: "CLIP", lora: dict[str, torch.Tensor], strength_model: float, strength_clip: float, filename: str = "default", online_mode: bool = False, *, anima_declared_blocks: Optional[int] = None, anima_inserted: Optional[str] = None, notes: Optional[dict[str, str]] = None):
    """Patch ``lora`` onto ``model`` / ``clip``; returns the patched pair.

    Raises ``LoraNotApplied`` when the LoRA does not fit the model (most of
    its keys match nothing, or a part's keys mostly fail to patch), so the
    caller reports it instead of the generation quietly running without it.
    ``notes`` receives ``"remap"``: how an Anima LoRA was re-mapped."""

    if dynamic_args.nunchaku:
        model.model.diffusion_model.loras.append((filename, strength_model))
        return model, clip
    if dynamic_args.anima and model is not None:
        note = process_anima(lora, len(model.model.diffusion_model.blocks), declared=anima_declared_blocks, inserted=anima_inserted, notes=notes)
        if note and notes is not None:
            notes["remap"] = note

    model_flag: str = type(model.model).__name__ if model is not None else "default"

    # A fresh key map each time: the helpers' default dict would keep the keys of
    # every model loaded before (blocks 28-51 of a 3.8B on a 2B, say)
    unet_keys = model_lora_keys_unet(model.model, {}) if model is not None else {}
    clip_keys = model_lora_keys_clip(clip.cond_stage_model, {}) if clip is not None else {}

    lora_unmatch = lora
    lora_unet, lora_unmatch = load_lora(lora_unmatch, unet_keys)
    lora_clip, lora_unmatch = load_lora(lora_unmatch, clip_keys)

    _unmatches = len(lora_unmatch)

    if _unmatches / max(len(lora), 1) > 0.5:
        logger.warning(f"[LORA] LoRA mismatch for {model_flag}: {filename}")
        raise LoraNotApplied(f"{_unmatches} of {len(lora)} keys match nothing in {model_flag}: made for another model?")

    if _unmatches > 0:
        logger.info(f"[LORA] Loading {os.path.basename(filename)} for {model_flag} with {_unmatches} unmatched keys")

    if model is not None and len(lora_unet) > 0:
        new_model = model.clone()
        loaded_keys = new_model.add_patches(filename=filename, patches=lora_unet, strength_patch=strength_model, online_mode=online_mode)
        skipped_keys = [item for item in lora_unet if item not in loaded_keys]
        if len(skipped_keys) / len(lora_unet) > 0.25:
            logger.warning(f"[LORA] Mismatch {filename} for {model_flag}-UNet with {len(skipped_keys)} keys mismatched in {len(loaded_keys)} keys")
            raise LoraNotApplied(f"{len(skipped_keys)} of {len(lora_unet)} diffusion-model keys do not fit {model_flag}")
        else:
            logger.info(f"[LORA] Loaded {os.path.basename(filename)} for {model_flag}-UNet with {len(loaded_keys)} keys at weight {strength_model} (skipped {len(skipped_keys)} keys) with on_the_fly = {online_mode}")
            model = new_model

    # Remi: Anima-3.8B runs a copy of the LLM adapter in the DiT (its semantic connector wraps
    # it), so the adapter's patches go to both copies, at the text-encoder strength
    adapter_prefix = "qwen3_06b.llm_adapter."
    if model is not None and getattr(model.model.diffusion_model, "llm_adapter", None) is not None:
        mirrored = {f"diffusion_model.llm_adapter.{key[len(adapter_prefix) :]}": patch for key, patch in lora_clip.items() if isinstance(key, str) and key.startswith(adapter_prefix)}
        if mirrored:
            new_model = model.clone()
            loaded_keys = new_model.add_patches(filename=filename, patches=mirrored, strength_patch=strength_clip, online_mode=online_mode)
            if len(loaded_keys) != len(mirrored):
                raise LoraNotApplied(f"{len(mirrored) - len(loaded_keys)} of {len(mirrored)} LLM adapter keys do not fit the diffusion model's adapter")
            logger.info(f"[LORA] Loaded {os.path.basename(filename)} for {model_flag}-LLM adapter (in the DiT) with {len(loaded_keys)} keys at weight {strength_clip}")
            model = new_model

    if clip is not None and len(lora_clip) > 0:
        new_clip = clip.clone()
        loaded_keys = new_clip.add_patches(filename=filename, patches=lora_clip, strength_patch=strength_clip, online_mode=online_mode)
        skipped_keys = [item for item in lora_clip if item not in loaded_keys]
        if len(skipped_keys) / len(lora_clip) > 0.25:
            logger.warning(f"[LORA] Mismatch {filename} for {model_flag}-CLIP with {len(skipped_keys)} keys mismatched in {len(loaded_keys)} keys")
            raise LoraNotApplied(f"{len(skipped_keys)} of {len(lora_clip)} text-encoder keys do not fit {model_flag}")
        else:
            logger.info(f"[LORA] Loaded {os.path.basename(filename)} for {model_flag}-CLIP with {len(loaded_keys)} keys at weight {strength_clip} (skipped {len(skipped_keys)} keys) with on_the_fly = {online_mode}")
            clip = new_clip

    return model, clip


def load_network(name: str, network_on_disk: network.NetworkOnDisk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    return net


def load_networks(names: list[str], te_multipliers: list[float] = None, unet_multipliers: list[float] = None, anima_sources: Optional[list[Optional[int]]] = None) -> dict[str, str]:
    """Load ``names`` onto the current model; returns the names that could
    not be loaded (not found, or failed to load), each with the reason.

    A LoRA that failed to load is remembered (``failed_networks``, cleared on
    every rescan) so later generations skip it instead of reading it again;
    it is still returned, so every generation reports it.

    ``anima_sources`` holds each LoRA's ``src=`` (the Anima depth it was
    trained for, overriding the guess from its keys), or None."""
    failed: dict[str, str] = {}
    remap_notes.clear()
    if anima_sources is None:
        anima_sources = [None] * len(names)

    if failed_networks:
        keep = [i for i, name in enumerate(names) if name not in failed_networks]
        if len(keep) != len(names):
            failed.update((name, failed_networks[name]) for name in names if name in failed_networks)
            names = [names[i] for i in keep]
            te_multipliers = [te_multipliers[i] for i in keep]
            unet_multipliers = [unet_multipliers[i] for i in keep]
            anima_sources = [anima_sources[i] for i in keep]

    current_sd = sd_models.model_data.get_sd_model()
    if current_sd is None:
        return failed

    loaded_networks.clear()

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    for network_on_disk, name in zip(networks_on_disk, names):
        if network_on_disk is None:
            logger.error(f'LoRA not found: "{name}"')
            failed[name] = "not found"
            continue
        try:
            net = load_network(name, network_on_disk)
            net.mentioned_name = name
            network_on_disk.read_hash()
            loaded_networks.append(net)
        except Exception as e:
            logger.error(f'Failed to load LoRA: "{name}"')
            failed[name] = failed_networks[name] = str(e) or type(e).__name__
            continue

    online_mode = dynamic_args.online_lora or False

    compiled_lora_targets = []
    compiled_names = []
    # Remi: part of each entry, so changing the setting re-applies the set
    inserted_policy = anima_inserted_blocks_policy()

    for n, u, t, f, src in zip(networks_on_disk, unet_multipliers, te_multipliers, names, anima_sources):
        if n is None or f in failed:
            continue
        compiled_lora_targets.append([n.filename, u, t, online_mode, src, inserted_policy])
        compiled_names.append(f)

    compiled_lora_targets_hash = str(compiled_lora_targets)
    if current_sd.current_lora_hash == compiled_lora_targets_hash:
        # The same set again (the next batch, the hires pass, a rerun): reported again, not reloaded
        failed_hash, apply_failures = getattr(current_sd, "lora_apply_failures", None) or (None, {})
        if failed_hash != compiled_lora_targets_hash:  # the hash was reset (Refiner) and rebuilt since
            apply_failures = {}
        # Named as this prompt names them (another alias of the same file may have failed first)
        failed.update({name: apply_failures[n.filename] for n, name in zip(networks_on_disk, names) if n is not None and n.filename in apply_failures})
        loaded_networks[:] = [item for item in loaded_networks if item.network_on_disk.filename not in apply_failures]
        _restore_remap_notes(current_sd, compiled_lora_targets_hash, zip(networks_on_disk, names))
        return failed

    current_sd.lora_apply_failures = (compiled_lora_targets_hash, {})  # failures of this set (above)
    current_sd.lora_remap_notes = (compiled_lora_targets_hash, {})  # Anima re-mappings of this set, by file

    current_sd.current_lora_hash = compiled_lora_targets_hash
    current_sd.forge_objects.unet = current_sd.forge_objects_original.unet
    current_sd.forge_objects.clip = current_sd.forge_objects_original.clip

    if dynamic_args.nunchaku:
        current_sd.forge_objects.unet.model.diffusion_model.loras.clear()

    for (filename, strength_model, strength_clip, online_mode, source_blocks, inserted_policy), name in zip(compiled_lora_targets, compiled_names):
        try:
            lora_sd, metadata = load_torch_file(filename, return_metadata=True)
        except Exception as e:
            logger.error(f'Failed to read LoRA: "{filename}"')
            failed[name] = failed_networks[name] = str(e) or type(e).__name__
            loaded_networks[:] = [item for item in loaded_networks if item.network_on_disk.filename != filename]
            continue
        if any(key.startswith("lora_unet__") for key in lora_sd):
            lora_sd = state_dict_prefix_replace(lora_sd, {"lora_unet__": "lora_unet_"})
        notes: dict = {}
        try:
            declared = source_blocks if source_blocks is not None else anima_layout.declared_depth(metadata)
            current_sd.forge_objects.unet, current_sd.forge_objects.clip = load_lora_for_models(current_sd.forge_objects.unet, current_sd.forge_objects.clip, lora_sd, strength_model, strength_clip, filename=filename, online_mode=online_mode, anima_declared_blocks=declared, anima_inserted=inserted_policy, notes=notes)
        except LoraNotApplied as e:
            logger.warning(f'LoRA not applied: "{name}": {e}')
            failed[name] = str(e)
            loaded_networks[:] = [item for item in loaded_networks if item.network_on_disk.filename != filename]
            current_sd.lora_apply_failures[1][filename] = failed[name]
            continue
        except Exception as e:
            # Reported for this generation (not remembered: it may load onto another model); the
            # LoRAs applied so far stay, and the objects below are refreshed, or the hash set above
            # would keep the previous LoRA set
            errors.report(f'Failed to apply LoRA: "{filename}"', exc_info=True)
            failed[name] = str(e) or type(e).__name__
            loaded_networks[:] = [item for item in loaded_networks if item.network_on_disk.filename != filename]
            current_sd.lora_apply_failures[1][filename] = failed[name]  # for the same set again (above)
            continue
        if "depth" in notes:
            RESOLVED_LORA_DEPTHS[os.path.abspath(filename)] = notes["depth"]
        if "remap" in notes:
            current_sd.lora_remap_notes[1][filename] = notes["remap"]
            remap_notes[name] = notes["remap"]

    current_sd.forge_objects_after_applying_lora = current_sd.forge_objects.shallow_copy()
    return failed


# Anima LoRAs re-mapped to this model's depth by the last load_networks, by
# prompt name: "28→40" (the infotext's "Lora remap")
remap_notes: dict[str, str] = {}


def _restore_remap_notes(current_sd, compiled_hash: str, named_networks) -> None:
    """The same LoRA set again: its re-mapping notes, named as this prompt names them."""
    remap_notes.clear()
    notes_hash, by_file = getattr(current_sd, "lora_remap_notes", None) or (None, {})
    if notes_hash != compiled_hash:
        return
    for n, name in named_networks:
        if n is not None and n.filename in by_file:
            remap_notes[name] = by_file[n.filename]


def process_network_files(names: Optional[list[str]] = None):
    candidates: set[str] = set()

    for _dir in [shared.cmd_opts.lora_dir, *shared.cmd_opts.lora_dirs]:
        candidates.update(shared.walk_files(_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))

    for filename in sorted(candidates, key=shared.natural_sort_key):
        if os.path.isdir(filename):
            continue
        name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases.add(entry.alias.lower())

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


def update_available_networks_by_names(names: list[str]):
    process_network_files(names)


def list_available_networks():
    failed_networks.clear()
    available_networks.clear()
    available_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.clear()
    forbidden_network_aliases.update(["none", "Addams"])

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    process_network_files()


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params: dict):
    if any("AddNet Module 1" in script.layout.infotext for script in scripts.scripts_txt2img.scripts if script.layout is not None):
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        if m := re_network_name.match(name):
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


extra_network_lora = None

failed_networks: dict[str, str] = {}
available_networks: dict[str, "network.NetworkOnDisk"] = {}
available_network_aliases: dict[str, "network.NetworkOnDisk"] = {}
available_network_hash_lookup: dict[bytes, "network.NetworkOnDisk"] = {}
forbidden_network_aliases: set[str] = set()
loaded_networks: list["network.Network"] = []

list_available_networks()
