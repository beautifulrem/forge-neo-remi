"""img2img batch with "Use PNG info": a parse error in one image's infotext does
not leave the ``infotext_styles`` setting at "Ignore" for every later paste."""

import types

import pytest
from PIL import Image

from remi.testing import webui_state


@pytest.fixture(scope="module", autouse=True)
def webui():
    with webui_state() as state:
        yield state


def test_a_parse_error_restores_infotext_styles(monkeypatch):
    from modules import img2img, shared

    def broken(_text):
        raise ValueError("bad infotext")

    monkeypatch.setattr(img2img, "parse_generation_parameters", broken)
    monkeypatch.setattr(img2img.images, "read_info_from_image", lambda _image: ("Steps: 20", {}))
    monkeypatch.setattr(img2img.modules.scripts, "scripts_img2img", types.SimpleNamespace(run=lambda p, *args: None))
    monkeypatch.setattr(img2img, "process_images", lambda p: types.SimpleNamespace(images=[p.init_images[0]], infotexts=["x"]))
    monkeypatch.setattr(shared.opts, "infotext_styles", "Apply if any", raising=False)
    monkeypatch.setattr(img2img, "state", types.SimpleNamespace(job_count=0, job="", skipped=False, interrupted=False, stopping_generation=False))
    p = types.SimpleNamespace(
        prompt="a", negative_prompt="", seed=1, cfg_scale=5.0, sampler_name="Euler", steps=4, width=64, height=64,
        override_settings={}, n_iter=1, batch_size=1, resize_mode=0, subseed=-1,
    )

    result = img2img.process_batch(p, [Image.new("RGB", (64, 64))], "", "", (), use_png_info=True, png_info_props=["Prompt"])

    assert shared.opts.infotext_styles == "Apply if any"
    assert p.prompt == "a" and len(result.images) == 1  # the image is still generated, without the infotext
