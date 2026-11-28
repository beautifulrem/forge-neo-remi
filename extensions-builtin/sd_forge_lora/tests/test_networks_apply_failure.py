"""A LoRA that fails while it is applied (not while it is read) is reported like an unreadable
one, and the model objects still become the ones the LoRAs before it built: otherwise the LoRA hash,
already set, keeps the previous generation's LoRA set active for this run and every rerun."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


@pytest.fixture
def networks(webui, monkeypatch):
    import network
    import networks

    model = SimpleNamespace(current_lora_hash=None)
    model.forge_objects_original = SimpleNamespace(unet="base-unet", clip="base-clip")
    model.forge_objects = SimpleNamespace(unet="unet-with-Y", clip="clip-with-Y", shallow_copy=lambda: SimpleNamespace(unet=model.forge_objects.unet, clip=model.forge_objects.clip))
    model.forge_objects_after_applying_lora = SimpleNamespace(unet="unet-with-Y", clip="clip-with-Y")
    monkeypatch.setattr(networks.sd_models.model_data, "get_sd_model", lambda: model)

    disks = {}
    for name in ("good", "broken"):
        disk = SimpleNamespace(filename=f"/loras/{name}.safetensors", read_hash=lambda: None)
        disks[name] = disk
    monkeypatch.setattr(networks, "available_network_aliases", dict(disks))
    monkeypatch.setattr(networks, "available_networks", dict(disks))
    monkeypatch.setattr(networks, "failed_networks", {})
    monkeypatch.setattr(networks, "loaded_networks", [])
    monkeypatch.setattr(networks, "load_network", lambda name, disk: SimpleNamespace(network_on_disk=disk, mentioned_name=name))
    monkeypatch.setattr(networks, "load_torch_file", lambda filename, return_metadata=False: ({"file": filename}, {}) if return_metadata else {"file": filename})

    def apply(unet, clip, lora_sd, *args, **kwargs):
        if "broken" in lora_sd["file"]:
            raise ZeroDivisionError("division by zero")
        return f"{unet}+good", f"{clip}+good"

    monkeypatch.setattr(networks, "load_lora_for_models", apply)
    monkeypatch.setattr(networks.errors, "report", lambda *args, **kwargs: None)
    del network
    return networks, model


def test_a_lora_failing_to_apply_is_reported_and_the_others_are_used(networks):
    module, model = networks

    failed = module.load_networks(["good", "broken"], [1.0, 1.0], [1.0, 1.0])

    assert failed == {"broken": "division by zero"}
    assert model.forge_objects_after_applying_lora.unet == "base-unet+good"
    assert [net.mentioned_name for net in module.loaded_networks] == ["good"]
    assert module.failed_networks == {}  # not remembered: another model may take it


def test_the_same_set_again_reports_the_failure_without_reloading(networks, monkeypatch):
    """The next batch, the hires pass and a rerun call load_networks with the same set: the failure
    is reported again and the LoRA not listed as applied, without reading and merging every LoRA again."""
    module, model = networks
    reads = []
    read = module.load_torch_file
    monkeypatch.setattr(module, "load_torch_file", lambda filename, **kwargs: reads.append(filename) or read(filename, **kwargs))

    module.load_networks(["good", "broken"], [1.0, 1.0], [1.0, 1.0])
    failed = module.load_networks(["good", "broken"], [1.0, 1.0], [1.0, 1.0])

    assert failed == {"broken": "division by zero"}
    assert [net.mentioned_name for net in module.loaded_networks] == ["good"]
    assert len(reads) == 2  # the first call's two files only
    assert model.forge_objects_after_applying_lora.unet == "base-unet+good"


def test_failures_of_another_set_are_not_reported(networks):
    """The Refiner resets the hash to that of no LoRA and back: a later run without LoRAs (the same
    hash) must not report the failure of the set the Refiner loaded in between."""
    module, model = networks

    module.load_networks(["good", "broken"], [1.0, 1.0], [1.0, 1.0])
    model.current_lora_hash = "[]"

    assert module.load_networks([], [], []) == {}


def test_the_same_set_under_another_alias_names_the_failure_as_this_prompt_does(networks, monkeypatch):
    module, _model = networks
    monkeypatch.setitem(module.available_network_aliases, "broken-alias", module.available_network_aliases["broken"])

    module.load_networks(["good", "broken"], [1.0, 1.0], [1.0, 1.0])
    failed = module.load_networks(["good", "broken-alias"], [1.0, 1.0], [1.0, 1.0])

    assert failed == {"broken-alias": "division by zero"}
    assert [net.mentioned_name for net in module.loaded_networks] == ["good"]


def test_a_lora_that_does_not_fit_is_reported_without_a_traceback(networks, monkeypatch):
    module, _model = networks
    reports = []
    monkeypatch.setattr(module.errors, "report", lambda *args, **kwargs: reports.append(args))

    def apply(unet, clip, lora_sd, *args, **kwargs):
        if "broken" in lora_sd["file"]:
            raise module.LoraNotApplied("40 of 40 keys match nothing")
        kwargs["notes"].update(depth=28, remap="28→40")
        return f"{unet}+good", f"{clip}+good"

    monkeypatch.setattr(module, "load_lora_for_models", apply)
    failed = module.load_networks(["good", "broken"], [1.0, 1.0], [1.0, 1.0])

    assert failed == {"broken": "40 of 40 keys match nothing"}
    assert reports == []
    assert module.remap_notes == {"good": "28→40"}
    # The same set again: the note comes back without re-applying
    failed = module.load_networks(["good", "broken"], [1.0, 1.0], [1.0, 1.0])
    assert module.remap_notes == {"good": "28→40"}
    assert failed == {"broken": "40 of 40 keys match nothing"}


def test_src_override_reaches_the_loader(networks, monkeypatch):
    module, _model = networks
    seen = {}

    def apply(unet, clip, lora_sd, *args, anima_declared_blocks=None, **kwargs):
        seen[lora_sd["file"]] = anima_declared_blocks
        return unet, clip

    monkeypatch.setattr(module, "load_lora_for_models", apply)
    module.load_networks(["good"], [1.0], [1.0], [40])
    assert seen == {"/loras/good.safetensors": 40}


def test_changing_the_inserted_blocks_policy_reapplies_the_set(networks, monkeypatch):
    """The same LoRA set with another ``anima_lora_inserted_blocks`` value is merged again
    (with that value), not taken from the previous generation's cache."""
    module, _model = networks
    seen = []

    def apply(unet, clip, lora_sd, *args, anima_inserted=None, **kwargs):
        seen.append(anima_inserted)
        return unet, clip

    monkeypatch.setattr(module, "load_lora_for_models", apply)
    for policy in ("copy", "copy", "skip"):
        monkeypatch.setattr(module, "anima_inserted_blocks_policy", lambda policy=policy: policy)
        module.load_networks(["good"], [1.0], [1.0])
    assert seen == ["copy", "skip"]


class _Patcher:
    def __init__(self, diffusion_model=None, patches=None):
        self.model = SimpleNamespace(diffusion_model=diffusion_model)
        self.cond_stage_model = None
        self.patches = dict(patches or {})

    def clone(self):
        return _Patcher(self.model.diffusion_model, self.patches)

    def add_patches(self, filename, patches, strength_patch, online_mode=False):
        for key, patch in patches.items():
            self.patches[key] = (patch, strength_patch)
        return list(patches)


def test_anima_3_8b_adapter_patches_reach_the_dit_copy_too(webui, monkeypatch):
    """Anima-3.8B runs a copy of the LLM adapter in the DiT: a LoRA on the adapter must patch
    both copies (at the text-encoder strength), or the connector ignores it."""
    import networks as module

    adapter_key = "qwen3_06b.llm_adapter.blocks.0.cross_attn.q_proj.weight"
    monkeypatch.setattr(module.dynamic_args, "anima", False)
    monkeypatch.setattr(module, "model_lora_keys_unet", lambda model, key_map: {"unet": True})
    monkeypatch.setattr(module, "model_lora_keys_clip", lambda model, key_map: {"clip": True})
    monkeypatch.setattr(module, "load_lora", lambda lora, keys: ({adapter_key: "patch"}, {}) if "clip" in keys else ({}, lora))

    for dit, want in ((SimpleNamespace(llm_adapter=object()), {"diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight": ("patch", 0.5)}), (SimpleNamespace(llm_adapter=None), {})):
        unet, clip = module.load_lora_for_models(_Patcher(dit), _Patcher(), {"x": 1}, 1.0, 0.5)
        assert unet.patches == want
        assert clip.patches == {adapter_key: ("patch", 0.5)}


def test_process_moss_consolidates_orphan_qkv_alphas_and_injects_metadata_alphas(webui):
    import json

    import networks as module
    import torch

    # 1. ai-toolkit fused proj.qkv LoKr with orphan proj.{q,k,v}.alpha
    lokr_sd = {
        "diffusion_model.layers.0.proj.qkv.lokr_w1": torch.zeros(3, 3),
        "diffusion_model.layers.0.proj.qkv.lokr_w2": torch.zeros(12, 4),
        "diffusion_model.layers.0.proj.q.alpha": torch.tensor(1.0),
        "diffusion_model.layers.0.proj.k.alpha": torch.tensor(1.0),
        "diffusion_model.layers.0.proj.v.alpha": torch.tensor(1.0),
    }
    module.process_moss(lokr_sd, metadata=None)
    assert "diffusion_model.layers.0.proj.qkv.alpha" in lokr_sd
    assert not any(k.endswith((".q.alpha", ".k.alpha", ".v.alpha")) for k in lokr_sd)

    # 2. moss-edit-lora per-module adapter_config JSON alpha (respects existing explicit .alpha)
    edit_sd = {
        "layers.0.proj.q.lora_A": torch.zeros(8, 12),
        "layers.0.proj.q.lora_B": torch.zeros(12, 8),
        "layers.0.proj.k.lora_A": torch.zeros(8, 12),
        "layers.0.proj.k.lora_B": torch.zeros(12, 8),
        "layers.0.proj.k.alpha": torch.tensor(16.0),
    }
    meta = {
        "adapter_config": json.dumps(
            {
                "format": "moss-edit-lora",
                "modules": {
                    "layers.0.proj.q": {"rank": 8, "alpha": 4.0},
                    "layers.0.proj.k": {"rank": 8, "alpha": 4.0},
                },
            }
        )
    }
    module.process_moss(edit_sd, metadata=meta)
    assert float(edit_sd["layers.0.proj.q.alpha"]) == pytest.approx(4.0)
    assert float(edit_sd["layers.0.proj.k.alpha"]) == pytest.approx(16.0)

    # 3. hayate_lora_format top-level metadata alpha
    ctrl_sd = {
        "diffusion_model.layers.0.proj.o.lora_up.weight": torch.zeros(12, 8),
        "diffusion_model.layers.0.proj.o.lora_down.weight": torch.zeros(8, 12),
    }
    module.process_moss(ctrl_sd, metadata={"hayate_lora_format": "1", "alpha": "32"})
    assert float(ctrl_sd["diffusion_model.layers.0.proj.o.alpha"]) == pytest.approx(32.0)
