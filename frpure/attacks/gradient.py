"""
frpure.attacks.gradient
=======================
The "different types of attacks", digital family:
  * FGSM         -- one-step L-inf (the classic baseline)
  * PGD (L-inf)  -- iterative, the workhorse; with EOT -> PRIMARY adaptive attack
  * PGD (L2)     -- L2 variant
  * BPDA+EOT     -- just PGD with a BPDA Target (see attacks/base.Target)

All optimize the verification objective (dodging / impersonation). The attacker
perturbs `x_adv` (the image being modified) while `x_other` (the partner in the
pair) stays fixed.
"""
from __future__ import annotations

import torch

from frpure.attacks.base import (Target, eot_grad, project_linf, project_l2,
                                 clamp01)


def fgsm(target: Target, x_adv: torch.Tensor, x_other: torch.Tensor,
         eps: float, mode: str, n_eot: int = 1) -> torch.Tensor:
    x_clean = x_adv.clone().detach()
    emb_other = target.backbone.embed(x_other).detach()
    g = eot_grad(target, x_clean, emb_other, mode, n_eot)
    x = x_clean + eps * g.sign()
    return clamp01(project_linf(x, x_clean, eps)).detach()


def pgd(target: Target, x_adv: torch.Tensor, x_other: torch.Tensor,
        eps: float, alpha: float, steps: int, mode: str,
        norm: str = "linf", n_eot: int = 1, rand_init: bool = True) -> torch.Tensor:
    """Iterative gradient attack with EOT. norm in {'linf','l2'}."""
    x_clean = x_adv.clone().detach()
    emb_other = target.backbone.embed(x_other).detach()
    x = x_clean.clone()
    if rand_init:
        if norm == "linf":
            x = x + torch.empty_like(x).uniform_(-eps, eps)
        else:
            noise = torch.randn_like(x)
            x = x + eps * noise / noise.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        x = clamp01(x).detach()

    for _ in range(steps):
        g = eot_grad(target, x, emb_other, mode, n_eot)
        if norm == "linf":
            x = x + alpha * g.sign()
            x = project_linf(x, x_clean, eps)
        elif norm == "l2":
            gflat = g.flatten(1)
            gn = gflat.norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
            x = x + alpha * g / gn
            x = project_l2(x, x_clean, eps)
        else:
            raise ValueError(norm)
        x = clamp01(x).detach()
    return x


def bpda_eot(target: Target, x_adv: torch.Tensor, x_other: torch.Tensor,
             eps: float, alpha: float, steps: int, mode: str,
             n_eot: int = 10) -> torch.Tensor:
    """BPDA+EOT: corroboration attack for non-differentiable / randomized
    defenses. Requires the Target to have bpda=True."""
    assert target.bpda, "bpda_eot needs Target(..., bpda=True)"
    return pgd(target, x_adv, x_other, eps, alpha, steps, mode,
               norm="linf", n_eot=n_eot, rand_init=True)


# Standard L-inf budget from the purification literature
DEFAULT_EPS_LINF = 8 / 255
DEFAULT_ALPHA_LINF = 2 / 255