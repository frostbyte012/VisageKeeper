"""
frpure.attacks.patch
====================
Physical-attack family, digital twin. Perturbation is confined to a facial
REGION (where real attacks live) and optimized to be robust to small geometric
jitter (EOT) -- the digital stand-in for "survives being printed and worn".

Region presets are fractional boxes on the *aligned* crop (faces are roughly
canonical after MTCNN alignment), so they approximate:
  eyeglass band  -> Sharif et al. eyeglass-frame attack
  hat band       -> AdvHat sticker
  sticker        -> adversarial patch on nose/cheek

For a landmark-exact mask, swap in the geometry from frpure.defenses.frpure
(same landmark source) -- kept consistent so attack region and defense mask are
defined the same way.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from frpure.attacks.base import Target, eot_grad, clamp01


# --------------------------------------------------------------------------- #
# region masks  (1 = perturbable, 0 = locked)   shape (1,1,H,W)
# --------------------------------------------------------------------------- #
def _box_mask(h: int, w: int, top: float, bot: float, left: float, right: float,
              device="cpu") -> torch.Tensor:
    m = torch.zeros(1, 1, h, w, device=device)
    m[..., int(top * h):int(bot * h), int(left * w):int(right * w)] = 1.0
    return m


def region_mask(kind: str, h: int, w: int, device="cpu") -> torch.Tensor:
    if kind == "eyeglass":      # horizontal band across the eyes
        return _box_mask(h, w, 0.28, 0.45, 0.12, 0.88, device)
    if kind == "hat":           # band across forehead / hairline
        return _box_mask(h, w, 0.02, 0.18, 0.10, 0.90, device)
    if kind == "sticker":       # square on nose / cheek
        return _box_mask(h, w, 0.45, 0.70, 0.38, 0.62, device)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# EOT transforms for physical robustness (small affine jitter)
# --------------------------------------------------------------------------- #
def _jitter(x: torch.Tensor, max_rot=0.05, max_trans=0.03, max_scale=0.04) -> torch.Tensor:
    b = x.shape[0]
    ang = (torch.rand(b, device=x.device) * 2 - 1) * max_rot
    tx = (torch.rand(b, device=x.device) * 2 - 1) * max_trans
    ty = (torch.rand(b, device=x.device) * 2 - 1) * max_trans
    sc = 1 + (torch.rand(b, device=x.device) * 2 - 1) * max_scale
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


# --------------------------------------------------------------------------- #
# masked PGD patch attack
# --------------------------------------------------------------------------- #
def patch_attack(
    target: Target,
    x_adv: torch.Tensor,
    x_other: torch.Tensor,
    mode: str,
    kind: str = "eyeglass",
    steps: int = 100,
    alpha: float = 4 / 255,
    n_eot: int = 5,
    physical_eot: bool = True,
) -> torch.Tensor:
    """Optimize an unbounded perturbation INSIDE `region_mask(kind)`.

    No L-inf cap (patches are visible by design); robustness comes from EOT
    jitter when physical_eot=True. Returns the patched image in [0,1].
    """
    h, w = x_adv.shape[-2:]
    mask = region_mask(kind, h, w, device=x_adv.device)
    x_clean = x_adv.clone().detach()
    emb_other = target.backbone.embed(x_other).detach()

    delta = (torch.rand_like(x_clean) * mask).detach()  # init patch
    for _ in range(steps):
        # accumulate EOT gradient (optionally through physical jitter)
        grad = torch.zeros_like(delta)
        for _ in range(max(n_eot, 1)):
            d = delta.clone().requires_grad_(True)
            x = clamp01(x_clean * (1 - mask) + (x_clean + d) * mask)
            x = _jitter(x) if physical_eot else x
            emb = target.embed(x)
            from frpure.attacks.base import pair_objective
            J = pair_objective(emb, emb_other, mode)
            g, = torch.autograd.grad(J, d)
            grad += g
        grad /= max(n_eot, 1)
        delta = (delta + alpha * grad.sign() * mask).clamp(-1, 1).detach()

    return clamp01(x_clean * (1 - mask) + (x_clean + delta) * mask).detach()