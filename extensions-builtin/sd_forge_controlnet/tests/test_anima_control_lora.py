"""Anima control-LoRA (the Anima-Control-Pose format) as a ControlNet unit: the embedder matches
the reference node's patchify, the control is added to the DiT's patch embedding only while the
unit runs, the LoRA reaches the text side before the prompts are encoded, and the prompt cache
follows it."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import remi_port_harness as harness
import torch
from einops import rearrange

ROOT = harness.REPO / "extensions-builtin"
LLLITE = ROOT / "sd_forge_controlllite"


@pytest.fixture(scope="module")
def lib():
    if str(LLLITE) not in sys.path:
        sys.path.insert(0, str(LLLITE))
    from lib_controllllite import anima_control_lora

    return anima_control_lora


def release_like(width=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "diffusion_model.control_embedder.proj.weight": torch.randn(width, 64, generator=generator),
        "diffusion_model.control_embedder.proj.bias": torch.randn(width, generator=generator),
        "diffusion_model.blocks.0.mlp.layer1.lora_A.weight": torch.zeros(4, width),
        "diffusion_model.blocks.0.mlp.layer1.lora_B.weight": torch.zeros(width, 4),
        "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, 8),
    }


def test_the_embedder_is_read_from_the_file_and_patchifies_like_the_reference(lib):
    state_dict = release_like()
    assert lib.is_control_lora(state_dict) and not lib.is_control_lora({"lllite_dit.x": torch.zeros(1)})
    embedder_sd, lora = lib.split_control_lora(state_dict)
    assert set(embedder_sd) == {"proj.weight", "proj.bias"} and len(lora) == 3

    embedder = lib.ControlEmbedder.from_state_dict(embedder_sd)
    latent = torch.randn(2, 16, 1, 6, 10)
    reference = embedder.proj(rearrange(latent, "b c (t r) (h m) (w n) -> b t h w (c r m n)", r=1, m=2, n=2))

    assert torch.equal(embedder(latent), reference)
    assert embedder(latent[:, :, 0]).shape == (2, 1, 3, 5, 32)  # a 4D latent is one frame
    assert embedder(torch.randn(1, 16, 1, 7, 9)).shape == (1, 1, 4, 5, 32)  # padded as the DiT pads
    with pytest.raises(ValueError, match="channels"):
        embedder(torch.randn(1, 4, 1, 6, 6))


def test_the_control_is_added_to_the_generated_frame_of_every_batched_copy(lib):
    tokens = torch.randn(1, 1, 3, 5, 8)
    output = torch.randn(4, 2, 3, 5, 8)  # cond/uncond x 2 images, a reference frame after the image
    states = {"on": True}
    injection = lib.ControlInjection(tokens, 0.5, lambda: states["on"])
    dit = SimpleNamespace(x_embedder=torch.nn.Identity())

    injection.attach(dit)
    got = dit.x_embedder(output)
    assert torch.allclose(got[:, 0], output[:, 0] + 0.5 * tokens[:, 0])
    assert torch.equal(got[:, 1], output[:, 1])

    states["on"] = False  # outside the unit's start / end
    assert torch.equal(dit.x_embedder(output), output)
    states["on"] = True
    assert torch.equal(dit.x_embedder(output[..., :4, :]), output[..., :4, :])  # another size: skipped

    injection.detach()
    assert torch.equal(dit.x_embedder(output), output)
    zero = lib.ControlInjection(tokens, 0.0)
    zero.attach(dit)
    assert torch.equal(dit.x_embedder(output), output)  # weight 0: no control
    zero.detach()


@pytest.fixture
def lllite(monkeypatch):
    from remi.testing import webui_state

    from modules_forge import shared as forge_shared

    with webui_state():
        monkeypatch.setattr(forge_shared, "supported_control_models", list(forge_shared.supported_control_models))
        yield harness.load_script(LLLITE / "scripts" / "forge_controllllite.py", "forge_controllllite_control_lora")


@pytest.fixture
def networks(monkeypatch):
    """The LoRA extension's loader, recording what it was asked to patch."""

    calls = []
    module = ModuleType("networks")

    class LoraNotApplied(Exception):
        pass

    def load_lora_for_models(unet, clip, lora, strength_model, strength_clip, filename="default", **kwargs):
        calls.append((set(lora), strength_model, strength_clip, filename, kwargs))
        return SimpleNamespace(patched=unet, model=getattr(unet, "model", None), load_device="cpu", model_options=dict(getattr(unet, "model_options", {}))), SimpleNamespace(patched=clip)

    module.LoraNotApplied, module.load_lora_for_models = LoraNotApplied, load_lora_for_models
    monkeypatch.setitem(sys.modules, "networks", module)
    return calls


def test_the_patcher_patches_the_text_side_then_the_dit_and_undoes_its_hook(lllite, networks):
    from modules_forge.shared import supported_control_models

    assert lllite.AnimaControlLoraPatcher in supported_control_models
    patcher = lllite.AnimaControlLoraPatcher.try_build_from_state_dict(release_like(), "/models/ControlNet/anima_pose.safetensors")
    assert lllite.AnimaControlLoraPatcher.try_build_from_state_dict({"lllite_dit.x": torch.zeros(1)}, "x") is None
    assert patcher.anima_compatible

    dit = SimpleNamespace(x_embedder=torch.nn.Identity())
    vae = SimpleNamespace(encode=lambda images: torch.ones(images.shape[0], 16, 1, images.shape[1] // 8, images.shape[2] // 8))
    unet = SimpleNamespace(load_device="cpu", model=SimpleNamespace(computation_dtype=torch.float32, diffusion_model=dit))
    clip = SimpleNamespace(name="clip")
    objects = SimpleNamespace(unet=unet, clip=clip, vae=vae)
    process = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=objects), steps=8)

    assert patcher.process_before_conditioning(process) is True
    assert objects.clip.patched is clip and objects.unet is unet  # only the text side, before the prompts
    files, strength_model, strength_clip, filename, kwargs = networks[0]
    assert "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight" in files and not any("control_embedder" in key for key in files)
    assert (strength_model, strength_clip, filename) == (1.0, 1.0, "/models/ControlNet/anima_pose.safetensors")
    assert kwargs.get("online_mode") is False

    objects.clip = clip
    patcher.strength = 0.75
    patcher.process_before_every_sampling(process, torch.full((2, 3, 32, 48), 0.5), None)
    assert objects.unet.patched is unet and objects.clip is clip  # only the DiT, at sampling
    hooked = dit.x_embedder(torch.zeros(4, 1, 2, 3, 32))
    expected = patcher.embedder(torch.ones(1, 16, 1, 4, 6)) * 0.75
    assert torch.allclose(hooked[0], expected[0]) and torch.allclose(hooked[3], expected[0])

    patcher.process_after_every_sampling()
    patcher.process_after_every_sampling()  # cleanup runs it again: idempotent
    assert not dit.x_embedder(torch.zeros(4, 1, 2, 3, 32)).any()


def test_bias_free_embedder_active_flag_online_lora_and_spectrum_window(lib, lllite, monkeypatch, networks):
    from backend.args import dynamic_args
    from modules import spectrum_force

    # Bias-free community control-LoRAs load cleanly
    sd = release_like()
    sd.pop("diffusion_model.control_embedder.proj.bias")
    embedder_sd, _ = lib.split_control_lora(sd)
    embedder = lib.ControlEmbedder.from_state_dict(embedder_sd)
    assert embedder.proj.bias is None and embedder(torch.zeros(1, 16, 1, 4, 6)).shape == (1, 1, 2, 3, 32)

    # online_lora is forwarded to load_lora_for_models, ACTIVE_CONTROL_LORA tracks hook lifecycle,
    # and windowed start/end notifies Spectrum so it does not forecast across the switch boundary.
    monkeypatch.setattr(dynamic_args, "online_lora", True)
    patcher = lllite.AnimaControlLoraPatcher(sd, "/models/ControlNet/anima_canny.safetensors")
    patcher.start_percent, patcher.end_percent = 0.0, 0.5
    dit = SimpleNamespace(x_embedder=torch.nn.Identity())
    vae = SimpleNamespace(encode=lambda images: torch.ones(images.shape[0], 16, 1, images.shape[1] // 8, images.shape[2] // 8))
    predictor = SimpleNamespace(percent_to_sigma=lambda p: 1.0 - p)
    unet = SimpleNamespace(load_device="cpu", model=SimpleNamespace(computation_dtype=torch.float32, diffusion_model=dit, predictor=predictor), model_options={})
    objects = SimpleNamespace(unet=unet, clip=SimpleNamespace(name="clip"), vae=vae)
    process = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=objects), steps=8)

    patcher.process_before_every_sampling(process, torch.full((1, 3, 32, 48), 0.5), None)
    assert networks[-1][4].get("online_mode") is True
    assert patcher._injection in getattr(dynamic_args, "ACTIVE_CONTROL_LORA", set())
    force = objects.unet.model_options["transformer_options"][spectrum_force.KEY]
    assert spectrum_force.is_forced(force, 0.75) and not spectrum_force.is_forced(force, 0.2)

    patcher.process_after_every_sampling()
    assert not getattr(dynamic_args, "ACTIVE_CONTROL_LORA", set())


def test_the_patcher_reports_a_lora_that_does_not_fit(lllite, monkeypatch, networks):
    def refuse(*args, **kwargs):
        raise sys.modules["networks"].LoraNotApplied("12 of 12 keys match nothing")

    monkeypatch.setattr(sys.modules["networks"], "load_lora_for_models", refuse)
    patcher = lllite.AnimaControlLoraPatcher(release_like(), "pose.safetensors")
    objects = SimpleNamespace(unet=object(), clip=object())

    with pytest.raises(ValueError, match="does not fit this model"):
        patcher.process_before_conditioning(SimpleNamespace(sd_model=SimpleNamespace(forge_objects=objects)))


def test_the_prompt_cache_is_cleared_when_the_text_patching_units_change():
    from remi.testing import webui_state

    with webui_state():
        cn = harness.load_script(ROOT / "sd_forge_controlnet" / "scripts" / "controlnet.py", "controlnet_text_patches")
    script = cn.ControlNetForForgeOfficial()
    patches = {"on": True}
    model = SimpleNamespace(process_before_conditioning=lambda p: patches["on"])
    unit = SimpleNamespace(model="anima_pose_preview2")
    script.get_enabled_units = lambda args: [unit]
    script.current_params = {0: SimpleNamespace(model=model)}
    cleared = []
    p = SimpleNamespace(clear_prompt_cache=lambda: cleared.append(1))

    script.process_batch(p)
    script.process_batch(p)  # the same patches: the cached prompts still hold
    assert len(cleared) == 1
    patches["on"] = False
    script.process_batch(p)
    assert len(cleared) == 2
    p.control_net_disabled = True
    script.process_batch(p)
    assert len(cleared) == 2
    cn._text_patches = ()
