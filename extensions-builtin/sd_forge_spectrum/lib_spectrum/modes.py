"""Opt-in Spectrum forecasting & adaptive caching modes: TaylorSeer and MagCache.

Both modes share the exact same pre-head feature capture seam and branch
isolation contract as :class:`~lib_spectrum.forecaster.ChebyshevForecaster`:

- **TaylorSeer** (:class:`TaylorSeerForecaster`, arXiv:2503.06923): higher-order
  finite-difference Taylor / Newton divided-difference feature extrapolation
  supporting both uniform and non-uniform step spacing between actual forwards,
  computed in float64 weight space with a single float32 feature accumulator.
- **MagCache** (:class:`MagCacheTracker` + :class:`MagCacheForecaster`,
  arXiv:2506.09045): magnitude-decay adaptive skip caching that tracks
  step-to-step relative residual magnitude ratios and skips DiT evaluations
  when accumulated magnitude drift remains below threshold, extrapolating
  the cached feature with an optional exponentially decayed residual delta.
"""

from __future__ import annotations

import math

import torch

FORECAST_MODES: tuple[str, ...] = ("Chebyshev", "TaylorSeer", "MagCache")

_EPS = 1e-8


def _newton_extrapolation_weights(times: list[float], query: float) -> list[float]:
    """Return scalar combination weights ``w_0, ..., w_{m-1}`` such that
    ``sum(w_i * f(t_i))`` equals the degree-(m-1) Newton interpolating /
    Taylor-extrapolating polynomial evaluated at ``query``.

    Computed in float64 (identical to exact finite-difference Taylor expansion
    for uniform grids, and exact Newton divided differences for non-uniform
    actual-step spacing after skipped steps).
    """

    m = len(times)
    if m == 1:
        return [1.0]

    # Basis vectors e_j in R^m: run divided-difference table on each column
    # (m <= 5, so this is < 50 scalar float operations on CPU).
    weights = [0.0] * m
    for j in range(m):
        # Lagrange basis L_j(query) = prod_{k != j} (query - t_k) / (t_j - t_k)
        # Mathematically identical to Newton divided-difference / finite-difference
        # Taylor expansion of order m - 1, and directly yields the linear weights.
        num = 1.0
        den = 1.0
        tj = times[j]
        for k in range(m):
            if k == j:
                continue
            diff = tj - times[k]
            if abs(diff) < _EPS:
                den = _EPS if diff >= 0 else -_EPS
            else:
                den *= diff
            num *= query - times[k]
        weights[j] = num / den
    return weights


class TaylorSeerForecaster:
    """Higher-order finite-difference feature extrapolation (TaylorSeer).

    Maintains the last ``order + 1`` actual step features ``(t_k, F(t_k))`` and
    extrapolates to future step ``step`` using exact order-``k`` divided
    differences (equivalent to a ``k``-th order Taylor series around the latest
    actual step). Memory footprint is bounded to ``order + 1`` feature tensors
    plus a single float32 accumulator on prediction.
    """

    def __init__(self, order: int = 2, history_size: int | None = None) -> None:
        order = max(1, min(int(order), 4))
        self.order = order
        self.history_size = max(order + 1, int(history_size)) if history_size is not None else order + 1
        self.features: list[torch.Tensor] = []
        self.times: list[float] = []
        self.shape: tuple[int, ...] | None = None
        self._acc_buf: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        return len(self.features) >= self.order + 1

    def update(self, step: int, feature: torch.Tensor) -> None:
        if self.shape is not None and tuple(feature.shape) != self.shape:
            self.clear()
        self.shape = tuple(feature.shape)
        step_f = float(step)
        if self.times and abs(self.times[-1] - step_f) < _EPS:
            self.features[-1] = feature.detach().clone()
            return
        self.features.append(feature.detach().clone())
        self.times.append(step_f)
        if len(self.features) > self.history_size:
            del self.features[0]
            del self.times[0]

    @torch.no_grad()
    def predict(self, step: int, out: torch.Tensor | None = None) -> torch.Tensor:
        if not self.features:
            raise RuntimeError("TaylorSeer forecast requested before warmup")
        if out is not None and tuple(out.shape) != self.shape:
            raise ValueError(f"TaylorSeer forecast of shape {self.shape} cannot be written to {tuple(out.shape)}")

        window = min(len(self.features), self.order + 1)
        recent_times = self.times[-window:]
        recent_features = self.features[-window:]
        weights = _newton_extrapolation_weights(recent_times, float(step))

        latest = recent_features[-1]
        if out is None:
            out = torch.empty(self.shape, dtype=latest.dtype, device=latest.device)
        if out.dtype == torch.float32:
            acc = out
        else:
            buf = self._acc_buf
            if buf is None or tuple(buf.shape) != self.shape or buf.device != out.device:
                buf = torch.empty(self.shape, dtype=torch.float32, device=out.device)
                self._acc_buf = buf
            acc = buf
        acc.zero_()
        for weight, feat in zip(weights, recent_features):
            acc.add_(feat.reshape(self.shape), alpha=float(weight))
        if acc is not out:
            out.copy_(acc)
        return out

    def clear(self) -> None:
        self.features.clear()
        self.times.clear()
        self.shape = None
        self._acc_buf = None


class MagCacheTracker:
    """Magnitude-decay adaptive skip scheduler (MagCache).

    Tracks consecutive actual-step feature change ratios:
    ``delta_t = ||F_t - F_{t-1}||_2 / (||F_{t-1}||_2 + eps)``
    and estimates the relative magnitude drift across skipped steps via a
    geometric decay factor ``decay``. Skips when accumulated error remains below
    ``threshold`` and ``consecutive_cached < max_skip``.
    """

    def __init__(self, *, threshold: float = 0.12, max_skip: int = 2, decay: float = 0.95) -> None:
        self.threshold = max(1e-5, float(threshold))
        self.max_skip = max(1, int(max_skip))
        self.decay = max(0.0, min(1.0, float(decay)))
        self.reset()

    def reset(self) -> None:
        self.last_ratio: float | None = None
        self.prev_norm: float | None = None
        self.accumulated_error: float = 0.0
        self.consecutive_skips: int = 0
        self.ratios: list[float] = []

    def observe_actual(self, current: torch.Tensor, previous: torch.Tensor | None) -> float | None:
        """Record an actual forward feature and return its relative L2 drift ratio."""

        cur_f = current.detach().float()
        cur_norm_t = torch.linalg.vector_norm(cur_f)
        ratio: float | None = None
        if previous is not None and previous.shape == current.shape:
            prev_f = previous.detach().float()
            prev_norm_t = torch.linalg.vector_norm(prev_f)
            diff_norm_t = torch.linalg.vector_norm(cur_f - prev_f)
            cur_norm, prev_norm, diff_norm = (float(v) for v in torch.stack((cur_norm_t, prev_norm_t, diff_norm_t)).cpu().tolist())
            ratio = diff_norm / max(prev_norm, _EPS)
            if math.isfinite(ratio):
                self.last_ratio = ratio
                self.ratios.append(ratio)
        else:
            cur_norm = float(cur_norm_t.cpu().tolist())
        self.prev_norm = cur_norm
        self.accumulated_error = 0.0
        self.consecutive_skips = 0
        return ratio

    def should_skip(self) -> bool:
        """Whether the current step can safely reuse/extrapolate the cached feature."""

        if self.last_ratio is None or self.consecutive_skips >= self.max_skip:
            return False
        step_err = self.last_ratio * (self.decay**self.consecutive_skips)
        return (self.accumulated_error + step_err) < self.threshold

    def record_skip(self) -> None:
        if self.last_ratio is not None:
            self.accumulated_error += self.last_ratio * (self.decay**self.consecutive_skips)
        self.consecutive_skips += 1


class MagCacheForecaster:
    """Magnitude-decayed residual forecaster paired with :class:`MagCacheTracker`.

    Predicts ``F(step) = F_last + (decay ** k) * (step - t_last) / dt * (F_last - F_prev)``
    with optional damping so cached steps smoothly follow magnitude decay
    without overshooting on sharp transitions.
    """

    def __init__(self, *, decay: float = 0.95) -> None:
        self.decay = max(0.0, min(1.0, float(decay)))
        self.features: list[torch.Tensor] = []
        self.times: list[float] = []
        self.shape: tuple[int, ...] | None = None
        self._acc_buf: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        return len(self.features) >= 2

    def update(self, step: int, feature: torch.Tensor) -> None:
        if self.shape is not None and tuple(feature.shape) != self.shape:
            self.clear()
        self.shape = tuple(feature.shape)
        self.features.append(feature.detach().clone())
        self.times.append(float(step))
        if len(self.features) > 2:
            del self.features[0]
            del self.times[0]

    @torch.no_grad()
    def predict(self, step: int, out: torch.Tensor | None = None) -> torch.Tensor:
        if not self.features:
            raise RuntimeError("MagCache forecast requested before warmup")
        if out is not None and tuple(out.shape) != self.shape:
            raise ValueError(f"MagCache forecast of shape {self.shape} cannot be written to {tuple(out.shape)}")

        latest = self.features[-1]
        if out is None:
            out = torch.empty(self.shape, dtype=latest.dtype, device=latest.device)

        if len(self.features) < 2 or self.decay == 0.0:
            out.copy_(latest.reshape(self.shape))
            return out

        dt = max(self.times[-1] - self.times[-2], _EPS)
        ahead = max(0.0, float(step) - self.times[-1])
        # Geometric sum of decayed residual steps: dampens drift as gap grows.
        damped_scale = (ahead / dt) * (self.decay ** max(1.0, ahead))
        if out.dtype == torch.float32:
            acc = out
        else:
            buf = self._acc_buf
            if buf is None or tuple(buf.shape) != self.shape or buf.device != out.device:
                buf = torch.empty(self.shape, dtype=torch.float32, device=out.device)
                self._acc_buf = buf
            acc = buf
        acc.copy_(latest.reshape(self.shape))
        acc.add_(latest.reshape(self.shape) - self.features[-2].reshape(self.shape), alpha=float(damped_scale))
        if acc is not out:
            out.copy_(acc)
        return out

    def clear(self) -> None:
        self.features.clear()
        self.times.clear()
        self.shape = None
        self._acc_buf = None
