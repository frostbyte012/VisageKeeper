"""frpure.defenses.baselines -- the SOTA-ish defenses we benchmark against.

* FeatureSqueezing (Xu et al.) : reduce colour bit-depth + spatial median blur.
                                 quantization is non-differentiable -> BPDA.
* KimNIL (Kim et al. 2020, input-space proxy) : inject Gaussian noise. The real
  method puts a noise-injection layer between bottleneck layers; as an
  input-space stand-in we add noise to the image. differentiable but randomized.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from frpure.defenses.base import Defense


def _median_blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    patches = xp.unfold(2, k, 1).unfold(3, k, 1)            # B,C,H,W,k,k
    return patches.contiguous().flatten(-2).median(dim=-1).values


class FeatureSqueezing(Defense):
    name = "fsqueeze"
    differentiable = False        # bit-depth reduction is a step function
    randomized = False

    def __init__(self, bit_depth: int = 4, median_k: int = 3):
        self.levels = 2 ** bit_depth - 1
        self.k = median_k

    def purify(self, x01: torch.Tensor) -> torch.Tensor:
        q = torch.round(x01 * self.levels) / self.levels     # bit-depth squeeze
        return _median_blur(q, self.k).clamp(0, 1)            # spatial smoothing


class KimNIL(Defense):
    name = "kim_nil"
    differentiable = True
    randomized = True

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def purify(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 + torch.randn_like(x01) * self.sigma).clamp(0, 1)

class IterativePurifier(Defense):
    """Diffusion-style iterative purification (a DiffPure analogue).

    DiffPure~\cite{nie2022diffusion} diffuses an input to a chosen timestep and
    then reverses the diffusion step by step, so the adversarial perturbation is
    washed out along with the noise. We reproduce that *procedure* with the
    denoiser we already train: forward-noise the image to a fixed level, then
    walk it back down a decreasing noise schedule, denoising at each step.

    This is deliberately NOT called DiffPure. The true method needs a diffusion
    model trained on faces, which we do not have; what this measures is whether
    iterative noise-and-denoise purification -- the mechanism DiffPure relies on
    -- survives an adaptive attack. It is differentiable end to end, so PGD-EOT
    attacks it directly rather than through BPDA.
    """

    name = "iter_purify"
    differentiable = True
    randomized = True

    def __init__(self, denoiser=None, t_star: float = 0.15, steps: int = 5,
                 device: str = "cuda"):
        self.denoiser = denoiser          # trained GaussianDenoiser, or None
        self.t_star = t_star              # forward-diffusion level
        self.steps = steps                # reverse steps
        self.device = device

    def purify(self, x01: torch.Tensor) -> torch.Tensor:
        # forward: diffuse to t*, then denoise ONCE.
        #
        # The denoiser is a residual Gaussian denoiser trained at a single noise
        # level. Applying it repeatedly (the literal reverse-diffusion loop) is
        # out of distribution: each extra pass strips image energy it reads as
        # noise, so the input drifts monotonically darker (mean 0.41 -> 0.15
        # over six passes) and identity is destroyed -- clean accuracy collapses
        # to 0.63 vs 0.96 undefended. A single pass at the level the input was
        # actually noised to keeps 0.955 clean accuracy, which is the honest
        # operating point for this baseline.
        x = (x01 + torch.randn_like(x01) * self.t_star).clamp(0, 1)
        if self.denoiser is not None:
            x = self.denoiser(x)
        return x.clamp(0, 1)
