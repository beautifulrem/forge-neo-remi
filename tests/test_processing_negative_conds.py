"""CFG 1 skips encoding the negative prompt unless a script keeps it (Anima NAG)."""

import types

import pytest

from remi.testing import webui_state


@pytest.fixture(scope="module", autouse=True)
def webui():
    with webui_state() as state:
        yield state


@pytest.mark.parametrize(
    ("cfg", "keep", "skips"),
    [(1, False, True), (1.0, None, True), (1, True, False), (4.0, False, False), (4.0, True, False)],
)
def test_skips_negative_conds(cfg, keep, skips):
    from modules.processing import StableDiffusionProcessing

    p = types.SimpleNamespace() if keep is None else types.SimpleNamespace(keep_negative_conds=keep)
    assert StableDiffusionProcessing.skips_negative_conds(p, cfg) is skips


@pytest.mark.parametrize(("sampler", "skips"), [("Euler", True), ("Euler CFG++", False), ("Euler a CFG++", False), ("DPM++ 2M CFG++", False)])
def test_cfg_pp_samplers_keep_the_negative_prompt_at_cfg_1(sampler, skips):
    # CFG++ takes the noise direction from the unconditional prediction at every scale (it
    # disables the CFG 1 optimisation): without the negative it got zeros, a washed-out image
    from modules import sd_samplers
    from modules.processing import StableDiffusionProcessing

    config = sd_samplers.find_sampler_config(sampler)
    assert StableDiffusionProcessing.skips_negative_conds(types.SimpleNamespace(), 1, config) is skips
