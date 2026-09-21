"""
frpure.defenses.smoothing
==========================
Denoised randomized smoothing for FACE VERIFICATION, with a provable certificate.

Wraps a FROZEN FR backbone (+ optional Gaussian denoiser) into a smoothed
verifier. For a probe a vs an enrolled (clean) gallery embedding e_b, the base
binary decision is
    h(a) = 1[ cos( emb(D(a)), e_b ) >= tau ]        (1 = same, 0 = different)
and the smoothed decision is the majority of h(a + eps), eps ~ N(0, sigma^2 I).

certify() returns the predicted label and an L2 radius R (Cohen et al., 2019):
the smoothed decision provably cannot change for any perturbation of the probe
with ||delta||_2 <= R. R is a property of the smoothed function, not of the
attacker -- so no adaptive attack can flip the decision inside R.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from scipy.stats import norm, beta


class GaussianDenoiser(nn.Module):
    """Lightweight full-image residual denoiser (DnCNN-style, FPGA-friendly).
    Predicts the noise; denoised = (x - noise).clamp(0,1)."""

    def __init__(self, ch: int = 32, depth: int = 4):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return (x - self.net(x)).clamp(0, 1)


class ViTGaussianDenoiser(nn.Module):
    """ViT bottleneck denoiser for denoised smoothing: same residual contract as
    GaussianDenoiser (predicts noise; denoised = (x - noise).clamp(0,1)), with
    a transformer core so denoising is non-local. Certificate is unchanged --
    only the base classifier under noise gets better."""

    def __init__(self, patch: int = 8, dim: int = 192, depth: int = 4, heads: int = 6,
                 pos_mode: str = "2d"):
        super().__init__()
        from frpure.defenses.vit_denoiser import ViTDenoiser
        self.core = ViTDenoiser(patch=patch, dim=dim, depth=depth, heads=heads,
                                pos_mode=pos_mode)

    def forward(self, x):
        return (x - self.core(x)).clamp(0, 1)


def build_denoiser(arch: str = "dncnn", ch: int = 32, **vit_kw) -> nn.Module:
    if arch == "dncnn":
        return GaussianDenoiser(ch=ch)
    if arch == "vit":
        return ViTGaussianDenoiser(**vit_kw)
    raise ValueError(arch)


class SmoothedVerifier:
    """Frozen backbone (+ optional denoiser) -> certified smoothed verifier."""

    def __init__(self, backbone, denoiser=None, sigma: float = 0.5,
                 tau: float = 0.3, device: str = "cuda"):
        self.backbone = backbone
        self.denoiser = denoiser
        self.sigma = sigma
        self.tau = tau
        self.device = getattr(backbone, "device", device)

    @torch.no_grad()
    def _count_same(self, probe, gallery_emb, n, batch):
        """Number of 'same' votes over n Gaussian-noised copies of the probe."""
        n_same, remaining = 0, n
        g = gallery_emb.unsqueeze(0)
        while remaining > 0:
            m = min(batch, remaining)
            x = probe.unsqueeze(0).repeat(m, 1, 1, 1)
            x = (x + torch.randn_like(x) * self.sigma).clamp(0, 1)
            if self.denoiser is not None:
                x = self.denoiser(x)
            cos = (self.backbone.embed(x) * g).sum(1)
            n_same += int((cos >= self.tau).sum().item())
            remaining -= m
        return n_same

    @torch.no_grad()
    def certify(self, probe, gallery_emb, n0=100, n=500, alpha=1e-3, batch=256):
        """Cohen-style certify. Returns (pred, radius).
        pred: 1=same, 0=different, -1=ABSTAIN (cannot certify)."""
        probe = probe.to(self.device)
        gallery_emb = gallery_emb.to(self.device)
        c0 = self._count_same(probe, gallery_emb, n0, batch)
        top = 1 if c0 >= (n0 - c0) else 0
        cnt = self._count_same(probe, gallery_emb, n, batch)
        nA = cnt if top == 1 else (n - cnt)
        pA = float(beta.ppf(alpha, nA, n - nA + 1)) if nA > 0 else 0.0
        if not (pA > 0.5):
            return -1, 0.0
        return top, self.sigma * float(norm.ppf(pA))

    @torch.no_grad()
    def predict(self, probe, gallery_emb, n=200, batch=256):
        probe = probe.to(self.device); gallery_emb = gallery_emb.to(self.device)
        c = self._count_same(probe, gallery_emb, n, batch)
        return 1 if c >= (n - c) else 0

    def soft_score(self, probe_batch, gallery_emb, n_eot: int = 8):
        """DIFFERENTIABLE Monte-Carlo estimate of the expected smoothed cosine
        E_eps[ cos( emb(D(probe + eps)), gallery ) ],  eps ~ N(0, sigma^2 I).

        Unlike _count_same/certify/predict this is NOT wrapped in no_grad and
        returns a continuous score, so an attacker can backprop through the same
        Gaussian-noise -> denoiser -> backbone chain the certificate relies on
        (Salman et al. 2019, "attacking the smoothed classifier"). Used ONLY to
        craft adversarial probes; the certificate itself is unchanged.
        """
        gallery_emb = gallery_emb.to(self.device)
        g = gallery_emb.unsqueeze(0)
        scores = []
        for _ in range(max(n_eot, 1)):
            noise = torch.randn_like(probe_batch) * self.sigma
            x = (probe_batch + noise).clamp(0, 1)
            if self.denoiser is not None:
                x = self.denoiser(x)
            emb = self.backbone.embed(x)
            scores.append((emb * g).sum(1))
        return torch.stack(scores, dim=0).mean(0)  # (B,) soft cosine, grad flows