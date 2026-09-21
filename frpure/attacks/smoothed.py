"""
frpure.attacks.smoothed
=======================
Adaptive attack on the SMOOTHED verifier (PGD + EOT). At each step the gradient
is averaged over `eot` fresh Gaussian draws through (denoiser -> backbone), so
the attacker isn't fooled by the randomness. Objective = expected cosine:
  genuine  (label=1): MINIMIZE cos (dodging)
  impostor (label=0): MAXIMIZE cos (impersonation)
Empirical robust accuracy at eps must sit >= certified accuracy at the same L2 radius.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _project(delta, probe, eps, norm):
    if norm == "linf":
        delta = delta.clamp(-eps, eps)
    else:
        n = delta.flatten().norm()
        if n > eps:
            delta = delta * (eps / (n + 1e-12))
    return (probe + delta).clamp(0, 1) - probe


def pgd_smoothed(sv, probe, gallery_emb, label, eps,
                 norm="l2", steps=20, eot=8, alpha=None):
    """Return an adversarial probe (detached) within the eps-ball (L2 or Linf)."""
    dev = sv.device
    probe = probe.to(dev)
    gallery_emb = gallery_emb.to(dev).unsqueeze(0)
    if alpha is None:
        alpha = 2.5 * eps / steps
    s = -1.0 if label == 1 else 1.0
    delta = torch.zeros_like(probe)
    for _ in range(steps):
        d = delta.clone().requires_grad_(True)
        x = (probe + d).clamp(0, 1)
        xb = x.unsqueeze(0).repeat(eot, 1, 1, 1)
        xb = (xb + torch.randn_like(xb) * sv.sigma).clamp(0, 1)
        if sv.denoiser is not None:
            xb = sv.denoiser(xb)
        cos = (sv.backbone.embed(xb) * gallery_emb).sum(1).mean()
        g, = torch.autograd.grad(cos, d)
        with torch.no_grad():
            if norm == "linf":
                step = alpha * s * g.sign()
            else:
                step = alpha * s * g / (g.flatten().norm() + 1e-12)
            delta = _project(delta + step, probe, eps, norm)
    return (probe + delta).clamp(0, 1).detach()