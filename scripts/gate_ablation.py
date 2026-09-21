#!/usr/bin/env python3
"""
scripts/gate_ablation.py -- which label-free signal predicts WHICH denoiser wins?

For each Gaussian draw z = x + eps we ask: does DnCNN(z) land closer to the
clean identity (in backbone embedding space) than ViT(z)? That is the ground
truth a per-input router would need. We then score six candidate gate signals by
AUC against it.

A signal is only USEFUL for a cost-saving cascade if it is (a) predictive and
(b) cheaper than the ViT it avoids. Embedding self-similarity is the strongest
signal but needs a backbone pass, so it fails (b).

Run:
  python scripts/gate_ablation.py --sigma 0.5 --n_pairs 100 --out results/paper/gate_ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser
from scripts.run_certify import load_pairs, balanced_subset
from scripts.diag_error_overlap import CKPT


def load_den(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    for p in d.parameters():
        p.requires_grad_(False)
    return d


def signals(z, cheap_out, backbone):
    """All candidate gate signals for a batch of noisy draws."""
    r = z - cheap_out
    pe = F.avg_pool2d(r.pow(2).mean(1, keepdim=True), 8, 8).flatten(1)
    lap = cheap_out - F.avg_pool2d(cheap_out, 3, 1, 1)
    ec = F.normalize(backbone.embed(cheap_out), dim=1)
    mu = F.normalize(ec.mean(0, keepdim=True), dim=1)
    return {
        "flatness":     -(pe.std(1) / (pe.mean(1) + 1e-8)),
        "resid_energy": r.pow(2).mean((1, 2, 3)),
        "resid_tv":     (r[:, :, 1:, :] - r[:, :, :-1, :]).abs().mean((1, 2, 3)),
        "out_tv":       (cheap_out[:, :, 1:, :] - cheap_out[:, :, :-1, :]).abs().mean((1, 2, 3)),
        "out_sharp":    lap.pow(2).mean((1, 2, 3)),
        "selfsim":      (ec * mu).sum(1),
    }


NEEDS_BACKBONE = {"selfsim"}


def auc(pos, neg, cap=2000, seed=0):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    sub = lambda a: a[rng.choice(len(a), min(cap, len(a)), replace=False)]
    p, n = sub(pos), sub(neg)
    return float(np.mean([(x > y) + 0.5 * (x == y) for x, y in product(p, n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/paper/gate_ablation.json")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)
    img1, _, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs * 2, len(same)))[:args.n_pairs]

    dn_w, vit_w = CKPT[args.sigma]
    cheap, exp = load_den("dncnn", dn_w, device), load_den("vit", vit_w, device)

    acc, better = {}, []
    with torch.no_grad():
        for i in idx:
            x0 = img1[i].unsqueeze(0).to(device)
            z = (x0.repeat(args.draws, 1, 1, 1)
                 + torch.randn(args.draws, 3, *x0.shape[-2:], device=device) * args.sigma).clamp(0, 1)
            cc, ee = cheap(z), exp(z)
            for k, v in signals(z, cc, backbone).items():
                acc.setdefault(k, []).append(v.cpu().numpy())
            e0 = backbone.embed(x0)
            better.append((((cheap_c := backbone.embed(cc)) * e0).sum(1)
                           >= (backbone.embed(ee) * e0).sum(1)).cpu().numpy())
            del cheap_c
    better = np.concatenate(better)

    rows = {}
    for k, v in acc.items():
        g = np.concatenate(v)
        a = auc(g[better], g[~better])
        rows[k] = {"auc": a, "abs_dev": abs(a - 0.5),
                   "needs_backbone": k in NEEDS_BACKBONE}
        print(f"  {k:>13}: AUC={a:.4f}  dev={abs(a-0.5):.4f}"
              f"{'   (needs backbone pass -- too costly for a cascade)' if k in NEEDS_BACKBONE else ''}")

    usable = {k: v for k, v in rows.items() if not v["needs_backbone"]}
    best = max(usable, key=lambda k: usable[k]["abs_dev"])
    print(f"\nDnCNN better on {better.mean():.3f} of {len(better)} draws")
    print(f"best backbone-free gate: {best} (AUC={rows[best]['auc']:.4f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"sigma": args.sigma, "n_draws": int(len(better)),
                   "dncnn_better_frac": float(better.mean()),
                   "signals": rows, "best_backbone_free": best}, f, indent=2)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
