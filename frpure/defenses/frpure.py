"""
frpure.defenses.frpure  -- OUR defense (the prime contribution).

Pipeline (matches the architecture diagram):
  1. mask     : where to purify -- union of attack-prone face regions
                (eyeglass band + hat band + nose/cheek sticker). A landmark
                source can replace the fractional presets (see `landmark_mask`).
  2. freq     : FFT low/high split -- perturbations concentrate in HIGH freq.
  3. purify   : INSIDE the mask, reconstruct the high-freq band.
                * training-free mode : attenuate high-freq (Gaussian low-pass).
                  fully differentiable -> attacker gets direct gradients (honest).
                * learned mode        : TinyUNet reconstructs high-freq; load
                  weights from training (scripts/train_purifier.py, later).
  4. recombine: out = clean*(1-mask) + (low + purified_high)*mask.

Everything is differentiable so the adaptive attacker can PGD+EOT straight
through us -- which is exactly the eval we want to survive.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frpure.defenses.base import Defense
from frpure.defenses.vit_denoiser import ViTDenoiser


# --------------------------------------------------------------------------- #
# region mask
# --------------------------------------------------------------------------- #
def _box(m, top, bot, left, right):
    h, w = m.shape[-2:]
    m[..., int(top * h):int(bot * h), int(left * w):int(right * w)] = 1.0


def attack_region_mask(h: int, w: int, device="cpu") -> torch.Tensor:
    """Union of the regions physical attacks target. (1,1,H,W) in {0,1}."""
    m = torch.zeros(1, 1, h, w, device=device)
    _box(m, 0.02, 0.18, 0.10, 0.90)   # hat / hairline
    _box(m, 0.28, 0.45, 0.12, 0.88)   # eyeglass band
    _box(m, 0.45, 0.70, 0.38, 0.62)   # nose / cheek sticker
    return m


def landmark_mask(h, w, landmarks, device="cpu", pad=0.06) -> torch.Tensor:
    """Build the mask from detected 5-point landmarks (eyes/nose/mouth).
    landmarks: (5,2) in pixel coords. Falls back to attack_region_mask if None.
    """
    if landmarks is None:
        return attack_region_mask(h, w, device)
    m = torch.zeros(1, 1, h, w, device=device)
    ph, pw = int(pad * h), int(pad * w)
    for (x, y) in landmarks:
        x, y = int(x), int(y)
        m[..., max(y - 2 * ph, 0):min(y + 2 * ph, h),
            max(x - 2 * pw, 0):min(x + 2 * pw, w)] = 1.0
    return m


# --------------------------------------------------------------------------- #
# differentiable FFT frequency split
# --------------------------------------------------------------------------- #
def _radial_lowpass(h: int, w: int, cutoff: float, device="cpu") -> torch.Tensor:
    yy = torch.linspace(-1, 1, h, device=device).view(h, 1)
    xx = torch.linspace(-1, 1, w, device=device).view(1, w)
    d = torch.sqrt(yy ** 2 + xx ** 2)
    return torch.exp(-(d ** 2) / (2 * (cutoff ** 2) + 1e-9))   # Gaussian LP, [0,1]


def freq_split(x: torch.Tensor, cutoff: float):
    """Return (low, high) bands. Differentiable. cutoff in (0,1]."""
    h, w = x.shape[-2:]
    lp = _radial_lowpass(h, w, cutoff, x.device)
    X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    low = torch.fft.ifft2(torch.fft.ifftshift(X * lp, dim=(-2, -1))).real
    return low, x - low


# --------------------------------------------------------------------------- #
# learned purifier (small conv U-Net) -- used when weights are available
# --------------------------------------------------------------------------- #
class TinyUNet(nn.Module):
    """~0.3M params, FPGA-friendly: plain convs, no attention. Reconstructs the
    high-frequency content of the masked region."""

    def __init__(self, ch: int = 24):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(),
                                nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU())
        self.e2 = nn.Sequential(nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.ReLU(),
                                nn.Conv2d(ch * 2, ch * 2, 3, 1, 1), nn.ReLU())
        self.d1 = nn.Sequential(nn.ConvTranspose2d(ch * 2, ch, 2, 2), nn.ReLU())
        self.out = nn.Conv2d(ch * 2, 3, 3, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(a)
        u = self.d1(b)
        return self.out(torch.cat([u, a], dim=1))


# --------------------------------------------------------------------------- #
# the defense
# --------------------------------------------------------------------------- #
class FRPure(Defense):
    name = "frpure_ours"
    differentiable = True
    randomized = False

    def __init__(self, cutoff: float = 0.15, use_unet: bool = False,
                 unet_weights: str | None = None, hi_atten: float = 0.25,
                 mask_mode: str = "attack_regions", device: str = "cuda",
                 purifier_type: str = "unet"):
        self.cutoff = cutoff
        self.hi_atten = hi_atten            # training-free high-freq attenuation
        self.mask_mode = mask_mode
        self.device = device
        self.unet = None
        if use_unet:
            self.unet = (ViTDenoiser() if purifier_type == "vit" else TinyUNet())
            self.unet = self.unet.eval().to(device)
            if unet_weights:
                self.unet.load_state_dict(torch.load(unet_weights, map_location=device))
            for p in self.unet.parameters():
                p.requires_grad_(False)

    def _mask(self, x, landmarks=None):
        h, w = x.shape[-2:]
        if self.mask_mode == "full":
            return torch.ones(1, 1, h, w, device=x.device)
        if self.mask_mode == "landmark":
            return landmark_mask(h, w, landmarks, x.device)
        return attack_region_mask(h, w, x.device)

    def purify(self, x01: torch.Tensor, landmarks=None) -> torch.Tensor:
        mask = self._mask(x01, landmarks)
        low, high = freq_split(x01, self.cutoff)
        if self.unet is not None:
            purified_high = self.unet(high * mask)        # learned reconstruction
        else:
            purified_high = high * self.hi_atten          # training-free attenuation
        recon = low + purified_high
        out = x01 * (1 - mask) + recon * mask
        return out.clamp(0, 1)