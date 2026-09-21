"""
frpure.defenses.gallery_denoiser
================================
Gallery-CONDITIONED denoiser for certified face verification.

The asymmetry we exploit
------------------------
Denoised smoothing (Salman et al. 2020) was designed for classification, where
the base classifier sees only the noisy input. Face verification is different:
at decision time we also hold a CLEAN, TRUSTED, ENROLLED gallery embedding e_b.
It is fixed before any probe arrives and is never touched by the attacker.
Standard denoised smoothing throws this information away -- D(x) is a generic
"remove Gaussian noise" map that does not know whose face it is reconstructing.

This module conditions the denoiser on e_b via FiLM modulation, so at sigma=1.0
(where the probe is barely visible) the network can reconstruct TOWARD the
enrolled identity instead of guessing from pixels alone.

Certificate safety
------------------
The Cohen et al. (2019) guarantee certifies g(x) = majority_eps h(x + eps) for
any DETERMINISTIC base classifier h. Conditioning on e_b keeps h deterministic:
for a fixed enrollment, D_{e_b}(.) is a fixed function of the noisy sample. The
gallery is not a function of the probe, so the attacker cannot steer it, and the
per-sample i.i.d. structure the Monte-Carlo bound needs is untouched. Each
enrolled identity simply gets its own base classifier -- exactly as in
per-enrollment sigma.

Honest caveat
-------------
Conditioning on the claimed identity introduces a CONFIRMATION-BIAS risk: the
denoiser could learn to pull every probe toward e_b, which would raise genuine-
pair similarity and impostor similarity alike, inflating FAR rather than
accuracy. Training therefore uses BOTH genuine and impostor conditioning, with
an explicit impostor-repulsion term, and evaluation must report impostor
accuracy separately. See scripts/train_gallery_denoiser.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation: per-channel (scale, shift) from a vector."""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, ch)
        self.to_beta = nn.Linear(cond_dim, ch)
        # start as identity so a warm-started denoiser is unchanged at init
        nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)

    def forward(self, feat, cond):
        g = self.to_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.to_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return feat * g + b


class GalleryConditionedDenoiser(nn.Module):
    """DnCNN-style residual denoiser with FiLM conditioning on the gallery embedding.

    forward(x, cond) -> denoised in [0,1], where cond is the (B, cond_dim)
    enrolled embedding. cond=None falls back to unconditional behaviour, so the
    same module can be evaluated as an ablation.
    """

    def __init__(self, ch: int = 32, depth: int = 4, cond_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        for _ in range(depth - 2):
            self.blocks.append(nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1),
                                             nn.ReLU(inplace=True)))
            self.films.append(FiLM(cond_dim, ch))
        self.tail = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x, cond=None):
        h = self.head(x)
        for blk, film in zip(self.blocks, self.films):
            h = blk(h)
            if cond is not None:
                h = film(h, cond)
        return (x - self.tail(h)).clamp(0, 1)


class GalleryWrapper(nn.Module):
    """Binds a fixed gallery embedding so the module satisfies the plain
    denoiser contract D(x) expected by SmoothedVerifier.

    This is what makes the certificate argument concrete: once the enrollment is
    fixed, this object IS a deterministic single-argument denoiser.
    """

    def __init__(self, core: GalleryConditionedDenoiser, gallery_emb: torch.Tensor):
        super().__init__()
        self.core = core
        self.register_buffer("cond", gallery_emb.detach().clone())

    def forward(self, x):
        return self.core(x, self.cond.unsqueeze(0).expand(x.shape[0], -1))
