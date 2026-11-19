import torch
from torch import nn

from . import sampling


class ForgeScheduleLinker(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    @property
    def sigmas(self):
        return self.predictor.sigmas

    @property
    def log_sigmas(self):
        return self.predictor.sigmas.log()

    @property
    def sigma_min(self):
        return self.predictor.sigma_min()

    @property
    def sigma_max(self):
        return self.predictor.sigma_max()

    def get_sigmas(self, n=None):
        if n is None:
            return sampling.append_zero(self.sigmas.flip(0))
        # Remi: the predictor's own timesteps from sigma_max to sigma_min (as ComfyUI's "normal"
        # scheduler), not the indices 999..0: for a flow predictor index 0 is sigma 0, and the
        # step from it to the appended zero divides by zero (a black image)
        start = float(self.sigma_to_t(self.sigmas[-1]))
        end = float(self.sigma_to_t(self.sigmas[0]))
        t = torch.linspace(start, end, n, device=self.sigmas.device)
        return sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        return self.predictor.timestep(sigma)

    def t_to_sigma(self, t):
        return self.predictor.sigma(t)
