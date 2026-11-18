"""Anima references take the generation's size.

Anima stacks its reference latents along the frames of the latent it
denoises, so every reference needs the same height and width. Upstream
caaa2cca resizes reference images to ``reference_max_side`` before encoding
them (for every Edit model), which for Anima no longer matches the image
being generated: Remi keeps the img2img image's own latent and fits
ImageStitch references to the generation in latent space.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.argv = sys.argv[:1]  # modules.shared parses sys.argv on import

from backend.args import dynamic_args  # noqa: E402
from backend.diffusion_engine import base  # noqa: E402
from backend.diffusion_engine.anima import Anima  # noqa: E402
from backend.nn.anima import fit_reference  # noqa: E402


def test_a_reference_of_the_generations_size_is_kept():
    ref = torch.randn(1, 16, 1, 8, 8)
    assert fit_reference(ref, (8, 8)) is ref


@pytest.mark.parametrize("size", [(4, 6), (16, 12), (8, 128)])
def test_a_reference_is_fitted_to_the_generation(size):
    ref = torch.randn(2, 16, 1, 8, 8, dtype=torch.bfloat16)
    fitted = fit_reference(ref, torch.Size(size))
    assert fitted.shape == (2, 16, 1, *size) and fitted.dtype == torch.bfloat16
    # a constant latent stays constant: the channels are resized, not mixed
    constant = torch.ones(1, 16, 1, 128, 128)
    assert torch.allclose(fit_reference(constant, size), torch.ones(1, 16, 1, *size))


@pytest.fixture
def engine(monkeypatch):
    from types import SimpleNamespace

    from backend.diffusion_engine import anima as anima_engine
    from modules import shared

    encoded = []

    def encode(self, x):
        encoded.append(tuple(x.shape[-2:]))
        return x[:, :1, None, ::8, ::8].clone()

    opts = SimpleNamespace(anima_do_reference=True, reference_max_side=1024)
    monkeypatch.setattr(base.ForgeDiffusionEngine, "_encode_first_stage", encode)
    monkeypatch.setattr(anima_engine, "opts", opts)
    monkeypatch.setattr(shared, "opts", opts)  # read by resize_reference
    monkeypatch.setattr(dynamic_args, "is_referencing", False, raising=False)
    anima = Anima.__new__(Anima)
    vars(anima).update(ref_latents=[], ini_latent=None, is_wan=False)
    return anima, encoded


def test_the_img2img_image_is_its_own_reference(engine):
    anima, encoded = engine
    image = torch.rand(2, 3, 1216, 832)

    samples = anima.encode_first_stage(image)

    # encoded once, at the generation's size (upstream: again at ~1024 x 1024)
    assert encoded == [(1216, 832)]
    assert samples.shape == (2, 1, 1, 152, 104)
    assert torch.equal(anima.ini_latent, samples[:1])


def test_imagestitch_references_still_follow_reference_max_side(engine):
    anima, encoded = engine
    dynamic_args.is_referencing = True

    assert anima.encode_first_stage(torch.rand(1, 3, 2048, 2048)) is None

    assert encoded == [(1024, 1024)]
    assert len(anima.ref_latents) == 1 and anima.ini_latent is None
