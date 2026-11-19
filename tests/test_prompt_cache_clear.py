"""Clearing the prompt cache (ImageStitch's new references, a model reload) clears the Hires.
fix conds too: their cache key does not see the references edit models encode into the conds."""

import pytest


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


def test_clearing_the_prompt_cache_clears_the_hires_conds(webui, monkeypatch):
    from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img as Txt2Img

    for owner, name in ((StableDiffusionProcessing, "cached_c"), (StableDiffusionProcessing, "cached_uc"), (Txt2Img, "cached_hr_c"), (Txt2Img, "cached_hr_uc")):
        monkeypatch.setattr(owner, name, ["key", "conds", {}])
    p = Txt2Img.__new__(Txt2Img)  # the cache methods need no other state
    p.cached_hr_c, p.cached_hr_uc = Txt2Img.cached_hr_c, Txt2Img.cached_hr_uc

    p.clear_prompt_cache()

    assert Txt2Img.cached_hr_c == Txt2Img.cached_hr_uc == [None, None, None]
    assert p.cached_hr_c == p.cached_hr_uc == p.cached_c == [None, None, None]
    assert StableDiffusionProcessing.cached_c == [None, None, None]
