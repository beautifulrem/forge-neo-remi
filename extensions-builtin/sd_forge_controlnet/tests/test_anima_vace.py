"""Anima VACE ControlNet (spaced-block duplication + zero-conv injection):
detection, config inference, strict weight loading, 2B -> 2.9B/3.8B depth
mapping, real tiny-DiT forward & hook lifecycle, odd spatial sizes, reference
frames, spatial masks, Wan21 latent normalisation, and Spectrum/offload wiring."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import remi_port_harness as harness
import torch

from backend.nn import anima_layout
from backend.nn.anima import Anima
from backend.operations import using_forge_operations

ROOT = harness.REPO / "extensions-builtin"
LLLITE = ROOT / "sd_forge_controlllite"


@pytest.fixture(scope="module")
def vace_lib():
    if str(LLLITE) not in sys.path:
        sys.path.insert(0, str(LLLITE))
    from lib_controllllite import anima_vace

    return anima_vace


@pytest.fixture
def lllite(monkeypatch):
    from modules_forge import shared as forge_shared
    from remi.testing import webui_state

    with webui_state():
        monkeypatch.setattr(forge_shared, "supported_control_models", list(forge_shared.supported_control_models))
        yield harness.load_script(LLLITE / "scripts" / "forge_controllllite.py", "forge_controllllite_vace")


def _tiny_dit(*, num_blocks: int = 4, model_channels: int = 64, num_heads: int = 4, crossattn_emb_channels: int = 32, adaln_lora_dim: int = 16) -> Anima:
    torch.manual_seed(0)
    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        dit = Anima(
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            crossattn_emb_channels=crossattn_emb_channels,
            adaln_lora_dim=adaln_lora_dim,
            concat_padding_mask=True,
        ).eval()
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return dit


def _vace_sd(
    vace_lib,
    *,
    num_blocks: int = 4,
    vace_block_every_n: int = 2,
    model_channels: int = 64,
    num_heads: int = 4,
    crossattn_emb_channels: int = 32,
    adaln_lora_dim: int = 16,
    use_after_proj: bool = True,
    zero_proj: bool = False,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    model = vace_lib.VACEControlNetAnima(
        model_channels=model_channels,
        num_heads=num_heads,
        num_blocks=num_blocks,
        crossattn_emb_channels=crossattn_emb_channels,
        vace_block_every_n=vace_block_every_n,
        use_after_proj=use_after_proj,
        adaln_lora_dim=adaln_lora_dim,
    ).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    if zero_proj:
        for cb in model.control_blocks:
            cb.init_weights()
    return {f"{prefix}{k}": v.detach().clone() for k, v in model.state_dict().items()}


def test_detection_prefix_stripping_and_config_inference(vace_lib):
    sd = _vace_sd(vace_lib, num_blocks=28, vace_block_every_n=7, model_channels=64, num_heads=4, prefix="control_model.")
    assert vace_lib.is_anima_vace(sd)
    assert not vace_lib.is_anima_vace({"lllite_dit.0.weight": torch.zeros(1)})
    assert not vace_lib.is_anima_vace({"diffusion_model.control_embedder.proj.weight": torch.zeros(32, 64)})

    stripped = vace_lib.strip_vace_prefix(sd)
    assert "control_embedder.proj.1.weight" in stripped
    assert "control_blocks.0.before_proj.weight" in stripped
    assert "control_blocks.3.after_proj.weight" in stripped

    meta = {
        "vace.condition_strategy": "spaced",
        "vace.num_blocks": "28",
        "vace.block_every_n": "7",
        "vace.num_control_blocks": "4",
        "vace.use_after_proj": "true",
        "vace.concat_padding_mask": "true",
        "vace.in_channels": "16",
        "vace.control_context_scale": "0.85",
    }
    cfg = vace_lib.infer_vace_config(sd, meta)
    assert cfg["model_channels"] == 64
    assert cfg["num_heads"] == 4
    assert cfg["num_blocks"] == 28
    assert cfg["vace_block_every_n"] == 7
    assert cfg["num_control_blocks"] == 4
    assert cfg["use_after_proj"] is True
    assert cfg["concat_padding_mask"] is True
    assert cfg["control_context_scale"] == pytest.approx(0.85)

    # Shape-only inference without safetensors metadata still recovers the exact layout
    cfg_no_meta = vace_lib.infer_vace_config(sd, None)
    assert cfg_no_meta["num_control_blocks"] == 4
    assert cfg_no_meta["vace_block_every_n"] == 7
    assert cfg_no_meta["use_after_proj"] is True

    # first_n + no after_proj
    sd_first = _vace_sd(vace_lib, num_blocks=4, vace_block_every_n=2, use_after_proj=False)
    cfg_first = vace_lib.infer_vace_config(sd_first, {"vace.condition_strategy": "first_n", "vace.num_blocks": "4"})
    assert cfg_first["condition_strategy"] == "first_n" and cfg_first["use_after_proj"] is False
    model_first = vace_lib.VACEControlNetAnima.from_state_dict(sd_first, {"vace.condition_strategy": "first_n", "vace.num_blocks": "4"})
    assert model_first.control_layers == [0, 1]


def test_depth_mapping_across_2b_29b_and_38b(vace_lib):
    sd = _vace_sd(vace_lib, num_blocks=28, vace_block_every_n=7, model_channels=64, num_heads=4)
    vace = vace_lib.VACEControlNetAnima.from_state_dict(sd, {"vace.num_blocks": "28", "vace.block_every_n": "7"})
    assert vace.control_layers == [0, 7, 14, 21]
    assert vace.target_layers_for_dit(28) == [0, 7, 14, 21]

    orig_40 = anima_layout.original_blocks(28, 40)
    orig_52 = anima_layout.original_blocks(28, 52)
    assert orig_40 is not None and orig_52 is not None
    assert vace.target_layers_for_dit(40) == [int(orig_40[i]) for i in (0, 7, 14, 21)]
    assert vace.target_layers_for_dit(52) == [int(orig_52[i]) for i in (0, 7, 14, 21)]

    with pytest.raises(ValueError, match="cannot map"):
        vace.target_layers_for_dit(33)


@torch.inference_mode()
def test_vace_injection_runs_inside_real_dit_with_zero_init_strength_mask_odd_size_and_ref_frames(vace_lib):
    from backend.args import dynamic_args

    dit = _tiny_dit(num_blocks=4)
    x = torch.randn(2, 16, 1, 6, 8)
    t = torch.tensor([0.5, 0.5])
    ctx = torch.randn(2, 5, 32)
    ctrl = torch.randn(1, 16, 1, 6, 8)

    baseline = dit(x, t, ctx)

    # 1. Zero-initialised zero-conv projections leave DiT output identical
    zero_sd = _vace_sd(vace_lib, num_blocks=4, vace_block_every_n=2, zero_proj=True)
    vace_zero = vace_lib.VACEControlNetAnima.from_state_dict(zero_sd, {"vace.num_blocks": "4", "vace.block_every_n": "2"})
    inj_zero = vace_lib.VACEInjection(vace_zero, ctrl, strength=1.0, target_layers=vace_zero.target_layers_for_dit(4))
    inj_zero.attach(dit)
    try:
        assert inj_zero in getattr(dynamic_args, "ACTIVE_ANIMA_VACE", set())
        out_zero = dit(x, t, ctx)
        assert torch.allclose(out_zero, baseline, atol=1e-6)
    finally:
        inj_zero.detach()
    assert inj_zero not in getattr(dynamic_args, "ACTIVE_ANIMA_VACE", set())

    # 2. Non-zero VACE weights modify output; active=False and zero mask suppress control cleanly
    sd = _vace_sd(vace_lib, num_blocks=4, vace_block_every_n=2, zero_proj=False)
    vace = vace_lib.VACEControlNetAnima.from_state_dict(sd, {"vace.num_blocks": "4", "vace.block_every_n": "2"})
    state = {"on": True}
    inj = vace_lib.VACEInjection(vace, ctrl, strength=0.75, target_layers=[0, 2], active=lambda: state["on"])
    inj.attach(dit)
    try:
        out_on = dit(x, t, ctx)
        assert (out_on - baseline).abs().max() > 1e-4
        assert not inj._hints  # hints popped cleanly after block 2

        state["on"] = False
        assert torch.allclose(dit(x, t, ctx), baseline, atol=1e-6)
    finally:
        inj.detach()
        inj.detach()  # idempotent

    # 3. Spatial mask (zeros -> baseline, ones -> matches unmasked)
    inj_zero_mask = vace_lib.VACEInjection(vace, ctrl, strength=0.75, target_layers=[0, 2], mask=torch.zeros(1, 1, 6, 8))
    inj_zero_mask.attach(dit)
    try:
        assert torch.allclose(dit(x, t, ctx), baseline, atol=1e-6)
    finally:
        inj_zero_mask.detach()

    inj_one_mask = vace_lib.VACEInjection(vace, ctrl, strength=0.75, target_layers=[0, 2], mask=torch.ones(1, 1, 6, 8))
    inj_one_mask.attach(dit)
    try:
        assert torch.allclose(dit(x, t, ctx), out_on, atol=1e-5)
    finally:
        inj_one_mask.detach()

    # 4. Odd spatial latent sizes (not divisible by patch_spatial=2) run without shape mismatch
    x_odd = torch.randn(1, 16, 1, 7, 9)
    ctrl_odd = torch.randn(1, 16, 1, 7, 9)
    base_odd = dit(x_odd, torch.tensor([0.5]), ctx[:1])
    inj_odd = vace_lib.VACEInjection(vace, ctrl_odd, strength=1.0, target_layers=[0, 2])
    inj_odd.attach(dit)
    try:
        out_odd = dit(x_odd, torch.tensor([0.5]), ctx[:1])
        assert out_odd.shape == x_odd.shape
        assert (out_odd - base_odd).abs().max() > 1e-4
    finally:
        inj_odd.detach()

    # 5. Reference frames (T=2, Anima Edit): control applies to frame 0 and runs cleanly
    x_ref = torch.randn(1, 16, 2, 6, 8)
    base_ref = dit(x_ref, torch.tensor([0.5]), ctx[:1])
    inj_ref = vace_lib.VACEInjection(vace, ctrl, strength=1.0, target_layers=[0, 2])
    inj_ref.attach(dit)
    try:
        out_ref = dit(x_ref, torch.tensor([0.5]), ctx[:1])
        assert out_ref.shape == x_ref.shape
        assert (out_ref[:, :, 0] - base_ref[:, :, 0]).abs().max() > 1e-4
    finally:
        inj_ref.detach()


def test_patcher_registration_wan21_process_in_spectrum_window_and_hires_skip(vace_lib, lllite, monkeypatch):
    from backend import memory_management
    from backend.args import dynamic_args
    from modules import spectrum_force
    from modules_forge.shared import supported_control_models

    assert lllite.AnimaControlVacePatcher in supported_control_models
    sd = _vace_sd(vace_lib, num_blocks=4, vace_block_every_n=2)
    assert lllite.AnimaControlVacePatcher.try_build_from_state_dict({"lllite_dit.x": torch.zeros(1)}, "x") is None

    patcher = lllite.AnimaControlVacePatcher(sd, metadata={"vace.num_blocks": "4", "vace.block_every_n": "2"})
    assert patcher.anima_compatible is True
    patcher.strength = 0.8
    patcher.start_percent, patcher.end_percent = 0.0, 0.5

    dit = _tiny_dit(num_blocks=4)
    process_in_calls = []

    class FakeLatentFormat:
        def process_in(self, latent):
            process_in_calls.append(tuple(latent.shape))
            return latent * 2.0

    vae = SimpleNamespace(encode=lambda images: torch.ones(images.shape[0], 16, 1, images.shape[1] // 8, images.shape[2] // 8))
    predictor = SimpleNamespace(percent_to_sigma=lambda p: 1.0 - p)
    unet = SimpleNamespace(
        load_device="cpu",
        model=SimpleNamespace(computation_dtype=torch.float32, diffusion_model=dit, predictor=predictor),
        model_options={},
    )
    objects = SimpleNamespace(unet=unet, vae=vae)
    sd_model = SimpleNamespace(forge_objects=objects, model_config=SimpleNamespace(latent_format=FakeLatentFormat()))
    process = SimpleNamespace(sd_model=sd_model, steps=10, is_hr_pass=False)

    offload_calls = []
    monkeypatch.setattr(memory_management, "unet_offload_device", lambda: torch.device("cpu"))
    orig_to = patcher.vace.to

    def track_to(*args, **kwargs):
        offload_calls.append(kwargs.get("device"))
        return orig_to(*args, **kwargs)

    monkeypatch.setattr(patcher.vace, "to", track_to)

    patcher.process_before_every_sampling(process, torch.full((1, 3, 48, 64), 0.5), None)
    assert process_in_calls == [(1, 16, 1, 6, 8)]
    assert torch.allclose(patcher._injection.control_latent, torch.full((1, 16, 1, 6, 8), 2.0))
    assert patcher._injection in getattr(dynamic_args, "ACTIVE_ANIMA_VACE", set())
    force = objects.unet.model_options["transformer_options"][spectrum_force.KEY]
    assert spectrum_force.is_forced(force, 0.75) and not spectrum_force.is_forced(force, 0.2)

    patcher.process_after_every_sampling()
    patcher.process_after_every_sampling()  # idempotent
    assert not getattr(dynamic_args, "ACTIVE_ANIMA_VACE", set())
    assert offload_calls[-1] == torch.device("cpu")

    # Non-Anima hires checkpoint is skipped gracefully; first pass raises clearly
    non_anima_unet = SimpleNamespace(load_device="cpu", model=SimpleNamespace(computation_dtype=torch.float32, diffusion_model=SimpleNamespace()), model_options={})
    hr_process = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=non_anima_unet, vae=vae)), steps=10, is_hr_pass=True)
    patcher.process_before_every_sampling(hr_process, torch.full((1, 3, 48, 64), 0.5), None)
    assert patcher._injection is None

    first_pass = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=non_anima_unet, vae=vae)), steps=10, is_hr_pass=False)
    with pytest.raises(ValueError, match="Anima checkpoint"):
        patcher.process_before_every_sampling(first_pass, torch.full((1, 3, 48, 64), 0.5), None)
