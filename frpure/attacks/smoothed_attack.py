"""
frpure.attacks.smoothed_attack
==============================
Adaptive PGD+EOT against the CERTIFIED smoothed verifier (frpure.defenses.
smoothing.SmoothedVerifier). This is the missing empirical counterpart to
certification: the attacker differentiates through the *smoothed* decision by
averaging gradients over the same Gaussian noise the certificate uses
(Salman et al., 2019). It optimizes the FR verification objective:

  dodging (genuine pair, want REJECT)      -> push smoothed cosine DOWN
  impersonation (impostor pair, want ACCEPT) -> push smoothed cosine UP

IMPORTANT: succeeding here does NOT contradict the certificate. Cohen's theorem
guarantees the smoothed decision is constant inside the certified L2 radius R;
an attack can only flip the *empirical* prediction by spending perturbation
budget that exceeds R. This experiment illustrates exactly that boundary.
"""
from __future__ import annotations

import torch

from frpure.attacks.base import project_linf, project_l2, clamp01


def pgd_eot_smoothed(sv, probe, gallery_emb, eps, alpha, steps, mode,
                     n_eot: int = 8, norm: str = "linf",
                     rand_init: bool = True) -> torch.Tensor:
    """PGD+EOT against SmoothedVerifier.soft_score.

    sv          : SmoothedVerifier
    probe       : (C,H,W) image in [0,1] being perturbed
    gallery_emb : (D,) fixed enrolled embedding of the partner image
    eps, alpha, steps : L-inf/L2 budget, step size, iterations
    mode        : "dodging" | "impersonation"
    n_eot       : EOT samples per step (>1 essential -- smoothing is randomized)
    """
    dev = sv.device
    x_clean = probe.to(dev).clone().detach()
    gallery_emb = gallery_emb.to(dev)
    x = x_clean.clone()
    if rand_init:
        if norm == "linf":
            x = x + torch.empty_like(x).uniform_(-eps, eps)
        else:
            noise = torch.randn_like(x)
            x = x + eps * noise / noise.flatten().norm().clamp_min(1e-12)
        x = clamp01(x).detach()

    # dodging minimizes similarity, impersonation maximizes it
    sign = -1.0 if mode == "dodging" else 1.0

    for _ in range(steps):
        x.requires_grad_(True)
        # soft_score expects a batch; probe is a single image
        score = sv.soft_score(x.unsqueeze(0), gallery_emb, n_eot=n_eot).squeeze(0)
        grad, = torch.autograd.grad(sign * score, x)
        x = x.detach()
        if norm == "linf":
            x = x + alpha * grad.sign()
            x = project_linf(x, x_clean, eps)
        elif norm == "l2":
            gn = grad.flatten().norm().clamp_min(1e-12)
            x = x + alpha * grad / gn
            x = project_l2(x.unsqueeze(0), x_clean.unsqueeze(0), eps).squeeze(0)
        else:
            raise ValueError(norm)
        x = clamp01(x).detach()
    return x
