"""DDIM and PLMS (CompVis, epsilon only: alphas_cumprod from 1 / (sigma^2 + 1)) and Kohaku LoNyu Yog
(variance-exploding renoise) gave garbled or washed-out images on flow-matching models such as
Anima. They are refused there with a message, as samplers without SDXL support are on SDXL."""

from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


def _model(prediction_type):
    predictor = SimpleNamespace(prediction_type=prediction_type)
    return SimpleNamespace(is_sdxl=False, forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(predictor=predictor))))


def _stub_constructor(sd_samplers, monkeypatch):
    find = sd_samplers.find_sampler_config
    monkeypatch.setattr(sd_samplers, "find_sampler_config", lambda name: find(name)._replace(constructor=lambda model: SimpleNamespace()))


@pytest.mark.parametrize("name", ["DDIM", "PLMS", "Kohaku LoNyu Yog"])
def test_epsilon_only_samplers_are_refused_on_flow_models(webui, name, monkeypatch):
    from modules import sd_samplers

    _stub_constructor(sd_samplers, monkeypatch)

    with pytest.raises(Exception, match="flow-matching"):
        sd_samplers.create_sampler(name, _model("const"))
    assert sd_samplers.create_sampler(name, _model("epsilon")).config.name == name


def test_other_samplers_run_on_flow_models(webui, monkeypatch):
    from modules import sd_samplers

    _stub_constructor(sd_samplers, monkeypatch)
    assert sd_samplers.create_sampler("Euler", _model("const")).config.name == "Euler"
