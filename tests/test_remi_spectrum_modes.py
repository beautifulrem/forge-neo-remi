"""Unit tests for opt-in Spectrum forecasting modes: TaylorSeer and MagCache
(``extensions-builtin/sd_forge_spectrum/lib_spectrum/modes.py`` and
``forecaster.py``), verifying accuracy on smooth trajectories, adaptive
skipping, and shared pre-head seam execution on DiT models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
SPECTRUM_ROOT = str(REPO / "extensions-builtin" / "sd_forge_spectrum")
if SPECTRUM_ROOT not in sys.path:
    sys.path.insert(0, SPECTRUM_ROOT)

from lib_spectrum.forecaster import SpectrumNode  # noqa: E402
from lib_spectrum.modes import (  # noqa: E402
    FORECAST_MODES,
    MagCacheForecaster,
    MagCacheTracker,
    TaylorSeerForecaster,
)

from backend.nn.dit_registry import DiTCapability, register_dit_capability, unregister_dit_capability  # noqa: E402


def test_taylorseer_exact_polynomial_extrapolation_uniform_and_nonuniform():
    assert set(FORECAST_MODES) == {"Chebyshev", "TaylorSeer", "MagCache"}

    # Quadratic trajectory f(t) = a + b*t + c*t^2: order-2 TaylorSeer must extrapolate EXACTLY!
    a = torch.tensor([[[1.0, -2.0, 0.5]]])
    b = torch.tensor([[[0.4, 0.2, -0.3]]])
    c = torch.tensor([[[-0.05, 0.08, 0.02]]])

    def poly(t: float) -> torch.Tensor:
        return a + b * t + c * (t * t)

    # Uniform steps [0, 1, 2] -> predict 3, 4
    ts = TaylorSeerForecaster(order=2)
    assert ts.ready is False
    for step in (0, 1, 2):
        ts.update(step, poly(float(step)))
    assert ts.ready is True
    torch.testing.assert_close(ts.predict(3), poly(3.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ts.predict(4), poly(4.0), atol=1e-5, rtol=1e-5)

    # Non-uniform actual steps [0, 2, 5] -> predict 6 (verifies Newton divided differences!)
    ts_nonuniform = TaylorSeerForecaster(order=2)
    for step in (0, 2, 5):
        ts_nonuniform.update(step, poly(float(step)))
    torch.testing.assert_close(ts_nonuniform.predict(6), poly(6.0), atol=1e-5, rtol=1e-5)

    # Cubic trajectory with order=3
    d = torch.tensor([[[0.01, -0.02, 0.015]]])

    def cubic(t: float) -> torch.Tensor:
        return poly(t) + d * (t**3)

    ts3 = TaylorSeerForecaster(order=3)
    for step in (0, 1, 2, 3):
        ts3.update(step, cubic(float(step)))
    torch.testing.assert_close(ts3.predict(4), cubic(4.0), atol=1e-4, rtol=1e-4)

    # Writing into pre-allocated out tensor
    out_buf = torch.empty_like(a)
    ret = ts.predict(3, out=out_buf)
    assert ret is out_buf
    torch.testing.assert_close(out_buf, poly(3.0), atol=1e-5, rtol=1e-5)


def test_magcache_tracker_adaptive_skip_and_forecaster():
    tracker = MagCacheTracker(threshold=0.10, max_skip=2, decay=0.9)
    assert tracker.should_skip() is False

    f0 = torch.ones(1, 16) * 10.0
    f1 = f0 + 0.3  # relative L2 drift = 0.03 < 0.10
    tracker.observe_actual(f0, None)
    assert tracker.should_skip() is False

    ratio = tracker.observe_actual(f1, f0)
    assert ratio is not None and 0.02 < ratio < 0.04
    assert tracker.should_skip() is True
    tracker.record_skip()  # skip 1 (acc ~ 0.03)
    assert tracker.should_skip() is True
    tracker.record_skip()  # skip 2 (hits max_skip=2)
    assert tracker.should_skip() is False

    # High drift immediately prevents skipping
    f_jump = f1 + 5.0  # relative drift ~ 0.49 >> 0.10
    ratio_big = tracker.observe_actual(f_jump, f1)
    assert ratio_big is not None and ratio_big > 0.4
    assert tracker.should_skip() is False

    # MagCacheForecaster applies decayed residual extrapolation
    mc = MagCacheForecaster(decay=0.5)
    mc.update(0, torch.tensor([[1.0, 2.0]]))
    mc.update(1, torch.tensor([[3.0, 4.0]]))
    # delta = [2, 2], step=2 -> ahead=1, dt=1, scale = 1 * 0.5 = 0.5 -> [3,4] + [1,1] = [4,5]
    pred = mc.predict(2)
    torch.testing.assert_close(pred, torch.tensor([[4.0, 5.0]]), atol=1e-6, rtol=1e-6)


class _FakePatcher:
    def __init__(self, kmodel):
        self.model = kmodel
        self.model_options = {}

    def clone(self):
        c = _FakePatcher(self.model)
        c.model_options = dict(self.model_options)
        return c

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn


def _run_spectrum_mode_on_pre_head_seam(tmp_path, *, forecast_mode: str):
    class _SeamDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(16, 16)
            self.seam_in = nn.Identity()
            self.head = nn.Linear(16, 16)
            self.actual_body_calls = 0
            self.fast_head_calls = 0

        def forward(self, x, sigma):
            self.actual_body_calls += 1
            feat = self.seam_in(self.body(x) + 0.01 * sigma.reshape(-1, 1))
            return self.head(feat)

        def compute_seam_head(self, feat, _t):
            self.fast_head_calls += 1
            return self.head(feat)

    dit = _SeamDiT().eval()
    cap = DiTCapability(
        name="seam_mode_test",
        dit_cls_name="_SeamDiT",
        model_type_name="_SeamDiT",
        engine_label="SeamTest",
        spectrum_required_attrs=("seam_in", "compute_seam_head"),
        spectrum_capture_attr="seam_in",
        spectrum_fast_forward=lambda d, pred, s, _x, f: d.compute_seam_head(f, pred.timestep(s)),
    )
    register_dit_capability(cap)
    try:
        predictor = SimpleNamespace(timestep=lambda s: s)
        patcher = _FakePatcher(SimpleNamespace(diffusion_model=dit, predictor=predictor))
        patched = SpectrumNode.patch(
            patcher,
            steps=10,
            weight=0.5,
            degree=2,
            lam=0.1,
            window_size=2,
            flex_window=0.0,
            warmup_steps=3,
            stop_caching_step=0.9,
            tail_actual_steps=1,
            history_size=6,
            schedule="Window",
            refresh_ratio=0.5,
            sea_beta=2.0,
            compat_policy="Conservative",
            verbose=False,
            sea_cache_dir=str(tmp_path),
            sea_cache_context={},
            forecast_mode=forecast_mode,
            taylor_order=2,
            magcache_threshold=0.25,
            magcache_max_skip=2,
            magcache_decay=0.9,
        )
        wrapper = patched.model_options["model_function_wrapper"]
        x = torch.ones(1, 16)
        ctx = torch.ones(1, 4, 8)
        for step in range(10):
            sigma = torch.tensor([1.0 - step * 0.09])
            out = wrapper(
                lambda inp, t, **_c: dit(inp, t),
                {"input": x, "timestep": sigma, "c": {"c_crossattn": ctx, "transformer_options": {}}, "cond_or_uncond": [0]},
            )
            assert out.shape == (1, 16) and torch.isfinite(out).all()
        return dit.actual_body_calls, dit.fast_head_calls
    finally:
        unregister_dit_capability("seam_mode_test")


def test_spectrum_node_taylorseer_and_magcache_share_pre_head_seam(tmp_path):
    ts_actual, ts_fast = _run_spectrum_mode_on_pre_head_seam(tmp_path / "ts", forecast_mode="TaylorSeer")
    assert ts_fast >= 2 and ts_actual < 10

    mc_actual, mc_fast = _run_spectrum_mode_on_pre_head_seam(tmp_path / "mc", forecast_mode="MagCache")
    assert mc_fast >= 2 and mc_actual < 10

    # Default Chebyshev mode remains completely intact
    ch_actual, ch_fast = _run_spectrum_mode_on_pre_head_seam(tmp_path / "ch", forecast_mode="Chebyshev")
    assert ch_fast >= 1 and ch_actual < 10
