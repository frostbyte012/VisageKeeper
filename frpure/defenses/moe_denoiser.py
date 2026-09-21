"""
frpure.defenses.moe_denoiser
============================
Mixture-of-experts denoiser (DnCNN + ViT) with a LEARNED gate, trained
end-to-end with an explicit decorrelation penalty.

Why this exists
---------------
Post-hoc routing between independently trained DnCNN/ViT denoisers does not pay
off: their errors are nested, so the oracle ceiling is ~+1% over ViT alone
(scripts/diag_error_overlap.py, scripts/probe_cross_sigma.py). But the niche
probe (scripts/probe_niche_coherence.py) found that the draws DnCNN wins are
highly structured -- split-half reliability r=0.95 at sigma=0.5, with 18% of
images won by DnCNN a majority of the time. The niche EXISTS; independent
training simply never grew it, and hand-designed gates could not find it
(best AUC 0.595).

This module trains the pair jointly to (a) widen that niche via a decorrelation
penalty that rewards experts making errors on DIFFERENT inputs, and (b) learn
the gate rather than hand-design it.

Certificate safety
------------------
The gate reads ONLY the noisy sample -- no clean image, no label, no pooling
across the n smoothing draws. So the whole module stays a deterministic
per-sample map and plugs into SmoothedVerifier with the Cohen et al. (2019)
guarantee unchanged, exactly like a single denoiser. See cascade.py for the
longer argument; it applies verbatim here.

Soft vs hard routing
--------------------
Training uses a SOFT convex blend (differentiable, both experts always run).
Inference can use either: soft (best accuracy) or hard top-1 (cheaper, since
only the selected expert runs). `set_hard(True)` switches modes; the certificate
holds for both because each is still a deterministic function of the sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateNet(nn.Module):
    """Small conv net: noisy image -> P(use ViT). Deliberately cheap (~15k
    params) so routing overhead stays far below the ViT it may avoid."""

    def __init__(self, ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, 2, 1), nn.ReLU(inplace=True),      # /2
            nn.Conv2d(ch, ch, 3, 2, 1), nn.ReLU(inplace=True),     # /4
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.ReLU(inplace=True), # /8
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(ch * 2, 2)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.head(h)                       # (B,2) logits [cheap, expensive]


class MoEDenoiser(nn.Module):
    """DnCNN + ViT experts with a learned gate.

    Residual contract matches GaussianDenoiser: forward(x) -> denoised in [0,1].
    """

    def __init__(self, cheap: nn.Module, expensive: nn.Module,
                 gate_ch: int = 16, hard: bool = False, temp: float = 1.0):
        super().__init__()
        self.cheap = cheap
        self.expensive = expensive
        self.gate = GateNet(gate_ch)
        self.hard = hard
        self.temp = temp
        from frpure.defenses.cascade import _disable_fused_attention
        _disable_fused_attention(self)     # NVML-safe transformer path

    def set_hard(self, hard: bool):
        self.hard = hard

    def gate_probs(self, x):
        return F.softmax(self.gate(x) / self.temp, dim=1)   # (B,2)

    def forward(self, x, return_parts: bool = False):
        p = self.gate_probs(x)                              # (B,2)
        yc = self.cheap(x)
        ye = self.expensive(x)
        if self.hard:
            pick = p.argmax(1)                              # (B,)
            m = pick.view(-1, 1, 1, 1).float()
            out = (1 - m) * yc + m * ye
        else:
            w = p.view(-1, 2, 1, 1)
            out = w[:, 0:1] * yc + w[:, 1:2] * ye
        out = out.clamp(0, 1)
        if return_parts:
            return out, yc, ye, p
        return out


def moe_losses(out, yc, ye, p, clean, backbone, lambda_emb: float = 0.5,
               lambda_dec: float = 0.0, lambda_bal: float = 0.0,
               lambda_anchor: float = 1.0, eps: float = 1e-8):
    """Training objective.

    task      : MSE + embedding-consistency on the ROUTED output (what we deploy)
    decorrel  : rewards experts whose per-sample errors are NEGATIVELY correlated
                across the batch, i.e. that fail on different inputs. This is the
                term meant to grow the niche the coherence probe found.
    balance   : keeps the gate from collapsing onto one expert (the standard MoE
                failure mode, and exactly the degenerate cascade we measured).
    """
    from frpure.attacks.base import cos_sim

    with torch.no_grad():
        e_clean = backbone.embed(clean)

    loss_mse = F.mse_loss(out, clean)
    loss_emb = (1 - cos_sim(backbone.embed(out), e_clean)).mean()
    task = loss_mse + lambda_emb * loss_emb

    # Each expert must stay individually competent. Without this, the optimiser
    # satisfies the decorrelation term by DEGRADING both experts (measured: 10
    # steps at lambda_dec=0.3 drove corr to -0.94 while both experts got worse
    # and DnCNN's win-rate FELL 0.375 -> 0.250). Anchoring both to the clean
    # target means decorrelation can only be earned by specialising.
    anchor = F.mse_loss(yc, clean) + F.mse_loss(ye, clean)
    task = task + lambda_anchor * anchor

    # --- decorrelation: per-sample expert error, centred, correlated ---------
    ec = (yc - clean).pow(2).mean(dim=(1, 2, 3))          # (B,)
    ee = (ye - clean).pow(2).mean(dim=(1, 2, 3))          # (B,)
    ec_c = ec - ec.mean()
    ee_c = ee - ee.mean()
    corr = (ec_c * ee_c).mean() / (ec_c.std() * ee_c.std() + eps)
    # Penalise POSITIVE correlation (both bad on the same inputs = nested
    # errors), but only down to 0: driving corr toward -1 is not the goal and
    # is what the degenerate solution exploits.
    loss_dec = corr.clamp(min=0.0)

    # --- load balance: gate usage should not collapse ------------------------
    usage = p.mean(0)                                      # (2,)
    loss_bal = (usage * usage).sum() * 2 - 1               # 0 at 50/50, 1 at collapse

    total = task + lambda_dec * loss_dec + lambda_bal * loss_bal
    return total, {"mse": float(loss_mse), "emb": float(loss_emb),
                   "corr": float(corr), "bal": float(loss_bal),
                   "usage_vit": float(usage[1]),
                   # per-expert pixel error: watch these to catch the degenerate
                   # "decorrelate by getting worse" solution as it happens
                   "mse_dncnn": float(ec.mean()), "mse_vit": float(ee.mean())}
