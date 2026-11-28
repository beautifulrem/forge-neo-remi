"""The "Uniform" scheduler (``ForgeScheduleLinker.get_sigmas``) spaces the predictor's own timesteps,
and "Turbo" picks its sigmas from the schedule by index. Both used the indices 999..0 as timesteps:
right for SD's discrete predictor, whose timesteps are those indices, but a flow predictor's are in
[0, multiplier] (Anima's multiplier is 1): Uniform gave sigmas above 1 and a step from 0 to the
appended 0 (a NaN, black image), Turbo sigmas far above 1 (noise)."""

from types import SimpleNamespace

import pytest
import torch

from backend.modules.k_prediction import Prediction, PredictionDiscreteFlow
from k_diffusion.external import ForgeScheduleLinker


def _flow(multiplier):
    return PredictionDiscreteFlow(SimpleNamespace(sampling_settings={"shift": 3.0, "multiplier": multiplier}))


def _sd():
    return Prediction(sigma_data=1.0, prediction_type="epsilon", beta_schedule="linear", linear_start=0.00085, linear_end=0.012, timesteps=1000)


@pytest.fixture
def turbo_scheduler(monkeypatch):
    import k_diffusion.sampling  # noqa: F401 - read by sd_schedulers at import

    from modules import shared

    # sd_schedulers reads the options at import
    monkeypatch.setattr(shared, "opts", SimpleNamespace(hide_schedulers=[]))
    from modules.sd_schedulers import turbo_scheduler

    return turbo_scheduler


def _flow_schedule_ok(sigmas):
    assert sigmas.shape == (11,)
    assert torch.isclose(sigmas[0], torch.tensor(1.0)) and sigmas[-1] == 0
    assert bool((sigmas[:-1] > 0).all()) and bool((sigmas.diff() < 0).all())


@pytest.mark.parametrize("multiplier", [1.0, 1000.0])  # Anima's, SD3's
def test_uniform_on_a_flow_model_runs_from_one_to_one_zero(multiplier):
    _flow_schedule_ok(ForgeScheduleLinker(_flow(multiplier)).get_sigmas(10))


@pytest.mark.parametrize("multiplier", [1.0, 1000.0])
def test_turbo_on_a_flow_model_runs_from_one_to_one_zero(turbo_scheduler, multiplier):
    predictor = _flow(multiplier)
    inner_model = SimpleNamespace(inner_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(predictor=predictor)))))
    _flow_schedule_ok(turbo_scheduler(10, 0.0, 1.0, inner_model, "cpu"))


def test_sd_turbo_schedules_are_unchanged(turbo_scheduler):
    predictor = _sd()
    inner_model = SimpleNamespace(inner_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(predictor=predictor)))))
    timesteps = torch.flip(torch.arange(1, 5) * 250.0 - 1, (0,)).round().long()
    expected = torch.cat([predictor.sigma(timesteps), torch.zeros(1)])

    assert torch.allclose(turbo_scheduler(4, 0.0, 1.0, inner_model, "cpu"), expected)


def _unused():
    sigmas = None

    return sigmas


def test_sd_uniform_schedules_are_unchanged():
    linker = ForgeScheduleLinker(_sd())
    # what the indices gave before
    expected = torch.cat([linker.t_to_sigma(torch.linspace(999, 0, 10)), torch.zeros(1)])

    assert torch.allclose(linker.get_sigmas(10), expected)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 20])
def test_bong_tangent_scheduler_handles_small_and_zero_steps(n):
    from modules import shared

    # Import safely with opts stubbed if needed
    if getattr(shared, "opts", None) is None:
        shared.opts = SimpleNamespace(hide_schedulers=[])
    from modules.sd_schedulers import bong_tangent_scheduler

    for pad in (False, True):
        sigmas = bong_tangent_scheduler(n, 14.6, 0.03, "cpu", pad=pad)
        assert torch.isfinite(sigmas).all()
        assert sigmas[-1].item() == 0.0
        assert len(sigmas) == (n + 2 if pad else n + 1)
        assert (sigmas[:-1] >= sigmas[1:]).all()

    # Zero slope / zero srange should also not divide by zero
    sigmas_flat_slope = bong_tangent_scheduler(5, 14.6, 0.03, "cpu", slope_1=0.0, slope_2=0.0)
    assert torch.isfinite(sigmas_flat_slope).all() and sigmas_flat_slope[-1].item() == 0.0
