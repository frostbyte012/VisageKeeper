"""
frpure.attacks.base
===================
Shared machinery for every attack. The FR-specific bit: attacks optimize a
*verification* objective on a pair, not a classification loss.

  dodging (genuine pair, attacker wants REJECT) -> push cosine DOWN
  impersonation (impostor pair, wants ACCEPT)   -> push cosine UP

`Target` is the function the attacker differentiates. For an *adaptive* attack
it wraps  backbone.embed(defense.purify(x)) :
  * differentiable defense -> autograd flows straight through (PGD+EOT)
  * non-differentiable defense (quantization, classical inpaint, randomness)
    -> BPDA straight-through: forward uses the real defense, backward
       approximates d(defense)/dx ~= Identity.
"""
from __future__ import annotations

from typing import Callable

import torch


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=1)


def pair_objective(emb_adv: torch.Tensor, emb_other: torch.Tensor, mode: str) -> torch.Tensor:
    """Scalar the attacker MAXIMIZES (summed over the batch)."""
    s = cos_sim(emb_adv, emb_other)
    if mode == "dodging":
        return (-s).sum()            # minimize similarity
    if mode == "impersonation":
        return (s).sum()             # maximize similarity
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# adaptive target (model, optionally + defense, optionally BPDA)
# --------------------------------------------------------------------------- #
class Target:
    def __init__(self, backbone, defense=None, bpda: bool = False):
        """
        backbone : FRBackbone with .embed(x01) -> normalized embeddings
        defense  : object with .purify(x01) -> x01, or None for undefended attack
        bpda     : if True, treat defense as identity on the backward pass
        """
        self.backbone = backbone
        self.defense = defense
        self.bpda = bpda

    def embed(self, x01: torch.Tensor) -> torch.Tensor:
        if self.defense is None:
            return self.backbone.embed(x01)
        if not self.bpda:
            # differentiable defense: autograd flows through purify
            return self.backbone.embed(self.defense.purify(x01))
        # BPDA straight-through: forward = real defense, backward = identity
        with torch.no_grad():
            y = self.defense.purify(x01)
        y_st = x01 + (y - x01).detach()
        return self.backbone.embed(y_st)


def eot_grad(
    target: Target,
    x_adv: torch.Tensor,
    emb_other: torch.Tensor,
    mode: str,
    n_eot: int = 1,
) -> torch.Tensor:
    """Expectation-over-Transformation gradient of the objective wrt x_adv.

    Averages gradients over n_eot stochastic forward passes -- essential when
    the defense (or attack pipeline) has randomness, else the attacker chases
    noise and reports fake robustness.
    """
    grad = torch.zeros_like(x_adv)
    for _ in range(max(n_eot, 1)):
        x = x_adv.clone().detach().requires_grad_(True)
        emb = target.embed(x)
        J = pair_objective(emb, emb_other, mode)
        g, = torch.autograd.grad(J, x)
        grad += g
    return grad / max(n_eot, 1)


# --------------------------------------------------------------------------- #
# norm projections
# --------------------------------------------------------------------------- #
def project_linf(x_adv: torch.Tensor, x_clean: torch.Tensor, eps: float) -> torch.Tensor:
    return x_clean + torch.clamp(x_adv - x_clean, -eps, eps)


def project_l2(x_adv: torch.Tensor, x_clean: torch.Tensor, eps: float) -> torch.Tensor:
    delta = x_adv - x_clean
    flat = delta.flatten(1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    factor = (eps / norm).clamp(max=1.0)
    return x_clean + (flat * factor).view_as(delta)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


EmbedFn = Callable[[torch.Tensor], torch.Tensor]