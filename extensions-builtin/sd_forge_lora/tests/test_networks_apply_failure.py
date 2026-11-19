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
    monkeypatch.setattr(networks, "load_torch_file", lambda filename: {"file": filename})

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
    monkeypatch.setattr(module, "load_torch_file", lambda filename: reads.append(filename) or read(filename))

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
