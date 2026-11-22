"""A branch whose blocks are replaced over part of the pass: Skip Layer
Guidance (``modules/processing_scripts/slg.py``, upstream be6fa5d4) runs a
Moss-Image step as two calls between its start and end percents, cond alone
and then uncond alone with ``patches_replace`` skipping DiT blocks, and one
CFG batch outside them. The uncond features on the two sides of the switch
come from different forwards, so its history must not span it."""

import pathlib
import sys
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import SpectrumNode, _replaced_blocks

STEPS = 16
DEGREE = 1
COND, UNCOND = 0, 1
SLG_STEPS = range(8)  # start 0.0, end 0.5


def _skip(args, _extra):
    return args


SLG_OPTIONS = {"patches_replace": {"dit": {("single_block", 22): _skip, ("single_block", 24): _skip}}}


class _FakePatcher:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
        self.model_options = {}
        self.wrapper = None

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, wrapper):
        self.wrapper = wrapper


def _patch(policy):
    process = SimpleNamespace()
    patcher = SpectrumNode.patch(
        _FakePatcher(), steps=STEPS, weight=1.0, degree=DEGREE, lam=1e-6, window_size=2,
        flex_window=0.0, warmup_steps=2, stop_caching_step=1.0, tail_actual_steps=1,
        history_size=DEGREE + 2, schedule="Window", refresh_ratio=0.0, sea_beta=2.0,
        compat_policy=policy, verbose=False, sea_cache_dir="unused",
        sea_cache_context={}, process=process,
    )
    return patcher


CONTEXTS = {COND: torch.ones(1, 4, 8), UNCOND: torch.zeros(1, 4, 8)}


def _truth(branch, step):
    # Skipping layers shifts the uncond output while SLG is active.
    return 1.0 * step if branch == COND else 2.0 * step + (40.0 if step in SLG_STEPS else 0.0)


def _call(patcher, step, branches, transformer_options, actual):
    def model_function(x, timestep, **_kwargs):
        for branch in branches:
            actual[branch].append(step)
        return torch.cat([torch.full_like(x[:1], _truth(branch, step)) for branch in branches])

    context = torch.cat([CONTEXTS[branch] for branch in branches])
    c = {"c_crossattn": context, "transformer_options": {**transformer_options, "cond_or_uncond": list(branches)}}
    sigma = torch.tensor([1.0 - step * 0.06])
    return patcher.wrapper(model_function, {"input": torch.zeros(len(branches), 1), "timestep": sigma, "c": c, "cond_or_uncond": list(branches)})


def _run(policy):
    patcher = _patch(policy)
    actual = {COND: [], UNCOND: []}
    errors = []
    for step in range(STEPS):
        if step in SLG_STEPS:
            outputs = {COND: _call(patcher, step, [COND], {}, actual), UNCOND: _call(patcher, step, [UNCOND], SLG_OPTIONS, actual)}
        else:
            batch = _call(patcher, step, [UNCOND, COND], {}, actual)
            outputs = {UNCOND: batch[:1], COND: batch[1:]}
        errors.extend((step, branch, abs(float(output[0, 0]) - _truth(branch, step))) for branch, output in outputs.items())
    return actual, errors


def test_replaced_blocks_are_listed_in_order():
    assert _replaced_blocks({}) == () and _replaced_blocks(None) == () and _replaced_blocks({"patches_replace": {"dit": {}}}) == ()
    assert _replaced_blocks(SLG_OPTIONS) == (("dit", "('single_block', 22)"), ("dit", "('single_block', 24)"))


def test_skip_layer_guidance_window_does_not_leak_into_the_forecasts():
    for policy in ("Conservative", "Legacy / fastest"):
        actual, errors = _run(policy)
        assert all(error < 1e-3 for *_where, error in errors), (policy, [item for item in errors if item[2] >= 1e-3])
        # Both branches still cache, inside and outside the SLG window.
        for branch in (COND, UNCOND):
            assert [step for step in SLG_STEPS if step not in actual[branch]], (policy, branch, actual)
            assert [step for step in range(SLG_STEPS.stop, STEPS) if step not in actual[branch]], (policy, branch, actual)
        # The uncond branch rebuilds its history after the switch.
        assert [step for step in actual[UNCOND] if step >= SLG_STEPS.stop][: DEGREE + 2] == list(range(SLG_STEPS.stop, SLG_STEPS.stop + DEGREE + 2)), (policy, actual)


@torch.inference_mode()
def test_moss_image_uses_pre_head_feature_forecasting():
    from backend.nn.moss_image import MOSSImage as MOSSDiT
    from backend.operations import using_forge_operations

    torch.manual_seed(0)
    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        dit = MOSSDiT(dim=32, ffn_dim=64, num_heads=2, num_layers=2, text_dim=16, latent_channels=8).eval()
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.02)

    class _LinearFlowPredictor:
        def timestep(self, sigma):
            return sigma

        def calculate_denoised(self, sigma, model_output, x):
            return x - sigma.view(-1, 1, 1, 1) * model_output

    predictor = _LinearFlowPredictor()

    class _MossPatcher:
        def __init__(self):
            self.model = SimpleNamespace(diffusion_model=dit, predictor=predictor)
            self.model_options = {}
            self.wrapper = None

        def clone(self):
            return self

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper

    patcher = SpectrumNode.patch(
        _MossPatcher(),
        steps=10,
        weight=1.0,
        degree=1,
        lam=1e-6,
        window_size=2,
        flex_window=0.0,
        warmup_steps=3,
        stop_caching_step=1.0,
        tail_actual_steps=1,
        history_size=3,
        schedule="Window",
        refresh_ratio=0.0,
        sea_beta=2.0,
        compat_policy="Conservative",
        verbose=False,
        sea_cache_dir="unused",
        sea_cache_context={},
        process=SimpleNamespace(),
    )
    assert getattr(dit.video_head_input, "_forge_spectrum_hook_installed", False) is True

    x = torch.randn(1, 8, 2, 2)
    ctx = torch.randn(1, 4, 16)
    actual_steps = []

    def model_function(inp, sigma, **_kwargs):
        actual_steps.append(float(sigma[0]))
        out = dit(inp, predictor.timestep(sigma).float(), ctx)
        return predictor.calculate_denoised(sigma, out.float(), inp)

    for step in range(10):
        sigma = torch.tensor([1.0 - step * 0.09])
        c = {"c_crossattn": ctx, "transformer_options": {"cond_or_uncond": [0]}}
        got = patcher.wrapper(model_function, {"input": x, "timestep": sigma, "c": c, "cond_or_uncond": [0]})
        assert got.shape == x.shape and torch.isfinite(got).all()

    # Verify Spectrum actually skipped steps via pre-head _fast_forward_moss
    assert len(actual_steps) < 10

    # Verify compute_video_head matches full forward head numerically
    captured = {}

    def grab(_m, args):
        captured["tokens"] = args[0].detach().clone()

    h = dit.video_head_input.register_forward_pre_hook(grab)
    try:
        full_out = dit(x, torch.tensor([0.5]), ctx)
    finally:
        h.remove()
    head_out = dit.compute_video_head(captured["tokens"], torch.tensor([0.5]), tuple(x.shape))
    assert torch.allclose(head_out, full_out, atol=1e-6)


def test_spectrum_krea2_pre_head_feature_forecasting():
    """Spectrum captures Krea 2's pre-head tokens (``dit.last_input``) and
    evaluates ``compute_last`` on cached steps with fresh timesteps."""

    from backend.modules.k_prediction import PredictionFlux
    from backend.nn.krea import SingleStreamDiT

    gen = torch.Generator().manual_seed(0)
    dit = SingleStreamDiT(features=128, tdim=32, txtdim=64, heads=2, kvheads=2, multiplier=2, layers=2, patch=2, channels=16, txtlayers=12, txtheads=2, txtkvheads=2).eval()
    dit.load_state_dict({k: torch.randn(v.shape, generator=gen) * 0.05 for k, v in dit.state_dict().items()})
    predictor = PredictionFlux(SimpleNamespace(sampling_settings={"shift": 1.15}))

    class _Patcher:
        def __init__(self):
            self.model = SimpleNamespace(diffusion_model=dit, predictor=predictor)
            self.model_options = {}
            self.wrapper = None

        def clone(self):
            return self

        def set_model_unet_function_wrapper(self, fn):
            self.wrapper = fn

    patcher = _Patcher()
    SpectrumNode.patch(
        patcher,
        steps=10,
        weight=0.5,
        degree=2,
        lam=1e-3,
        window_size=2,
        flex_window=0.0,
        warmup_steps=4,
        stop_caching_step=0.9,
        tail_actual_steps=1,
        history_size=4,
        schedule="Uniform",
        refresh_ratio=0.5,
        sea_beta=0.1,
        compat_policy="Conservative",
        verbose=False,
        sea_cache_dir="",
        sea_cache_context={},
    )
    assert getattr(dit.last_input, "_forge_spectrum_hook_installed", False) is True

    x = torch.randn(1, 16, 1, 4, 4, generator=gen)
    ctx = torch.randn(1, 4, 12, 64, generator=gen)
    actual_steps: list[float] = []

    def model_function(inp, sigma, **c):
        actual_steps.append(float(sigma[0].item()))
        t = predictor.timestep(sigma).float()
        out = dit(inp, t, c["c_crossattn"], transformer_options=c.get("transformer_options", {}))
        return predictor.calculate_denoised(sigma, out.float(), inp)

    for step in range(10):
        sigma = torch.tensor([1.0 - step * 0.09])
        c = {"c_crossattn": ctx, "transformer_options": {"cond_or_uncond": [0]}}
        got = patcher.wrapper(model_function, {"input": x, "timestep": sigma, "c": c, "cond_or_uncond": [0]})
        assert got.shape == x.shape and torch.isfinite(got).all()

    assert len(actual_steps) < 10

    captured = {}

    def grab(_m, args):
        captured["tokens"] = args[0].detach().clone()

    h = dit.last_input.register_forward_pre_hook(grab)
    try:
        full_out = dit(x, torch.tensor([0.5]), ctx)
    finally:
        h.remove()
    head_out = dit.compute_last(captured["tokens"], torch.tensor([0.5]), tuple(x.shape))
    assert torch.allclose(head_out, full_out, atol=1e-6)
