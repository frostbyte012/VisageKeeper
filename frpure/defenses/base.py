"""frpure.defenses.base -- common defense interface.

Every defense maps [0,1] images -> [0,1] images via .purify(x). Two flags drive
how the attacker should adapt:

  differentiable : if True, attacks can backprop straight through (PGD+EOT,
                   direct gradients -- the STRONGEST test). If False, attacks
                   fall back to BPDA straight-through.
  randomized     : if True, the runner uses EOT (n_eot>1) so the attacker
                   averages over the defense's stochasticity (no fake robustness).
"""
from __future__ import annotations

import torch


class Defense:
    name: str = "base"
    differentiable: bool = True
    randomized: bool = False

    def purify(self, x01: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Identity(Defense):
    name = "none"
    differentiable = True
    randomized = False

    def purify(self, x01: torch.Tensor) -> torch.Tensor:
        return x01