#!/usr/bin/env python3
"""
scripts/measure_pbar.py -- measure p_bar and the implied certified radius directly.

The certified radius is R = sigma * Phi^-1(p_bar), where p_bar is the fraction of
Gaussian draws voting for the correct decision. Today's investigation repeatedly
showed that intermediate metrics (pixel MSE, embedding cosine) can improve while
verification accuracy does not move -- so for the consistency experiment we
measure the quantity the certificate actually depends on, BEFORE committing to
long training runs.

Reports per denoiser, over a balanced pair subset:
  mean p_bar (correct-vote fraction)      <- what sets R
  mean implied R = sigma * Phi^-1(p_bar)
  embedding dispersion across draws       <- what the consistency loss targets
  vote-margin distribution

Run:
  python scripts/measure_pbar.py --sigma 0.5 --n_pairs 100 \
      --denoisers baseline=results/denoiser_vit_s050_v3.pth \
                  cons=results/denoiser_vit_s050_cons.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser
from frpure.defenses.cascade import _disable_fused_attention
from scripts.run_certify import load_pairs, balanced_subset, embed_all


def load_den(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(d)
    for p in d.parameters():
        p.requires_grad_(False)
    return d


@torch.no_grad()
def pbar_stats(den, backbone, probe, gallery, label, sigma, tau, n, batch, device):
    """Vote fraction for the CORRECT decision, plus embedding dispersion."""
    votes, embs = 0, []
    remaining = n
    while remaining > 0:
        m = min(batch, remaining)
        z = probe.unsqueeze(0).repeat(m, 1, 1, 1).to(device)
        z = (z + torch.randn_like(z) * sigma).clamp(0, 1)
        e = backbone.embed(den(z))
        embs.append(e)
        cos = (e * gallery.unsqueeze(0)).sum(1)
        pred = (cos >= tau).long()
        votes += int((pred == label).sum().item())
        remaining -= m
    e = torch.cat(embs)
    mu = F.normalize(e.mean(0, keepdim=True), dim=1)
    dispersion = float(1 - (e * mu).sum(1).mean())   # 0 = all draws agree
    return votes / n, dispersion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--arch", default="vit")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--n", type=int, default=400, help="draws per pair")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--denoisers", nargs="+", required=True,
                    help="name=path pairs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    imp = cos_clean[lab == 0]
    tau = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3
    print(f"sigma={args.sigma}  tau={tau:.4f}  pairs={len(idx)}  draws={args.n}\n")

    out = {"sigma": args.sigma, "tau": tau, "n_pairs": len(idx), "n": args.n,
           "models": {}}
    for spec in args.denoisers:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[skip] {name}: missing {path}")
            continue
        den = load_den(args.arch, path, device)
        pb, disp = [], []
        for j, i in enumerate(idx):
            p, d = pbar_stats(den, backbone, img1[i], e2[j].to(device), lab[j],
                              args.sigma, tau, args.n, args.batch, device)
            pb.append(p); disp.append(d)
        pb = np.array(pb); disp = np.array(disp)
        # implied radius; only pairs with p_bar>0.5 are certifiable at all
        cert = pb > 0.5
        R = np.where(cert, args.sigma * norm.ppf(np.clip(pb, 1e-6, 1 - 1e-6)), 0.0)
        rec = {"mean_pbar": float(pb.mean()),
               "mean_pbar_certifiable": float(pb[cert].mean()) if cert.any() else 0.0,
               "frac_certifiable": float(cert.mean()),
               "mean_R": float(R[cert].mean()) if cert.any() else 0.0,
               "mean_dispersion": float(disp.mean()),
               "majority_correct": float((pb > 0.5).mean())}
        out["models"][name] = rec
        print(f"=== {name} ===")
        print(f"  mean p_bar          : {rec['mean_pbar']:.4f}")
        print(f"  frac majority-right : {rec['majority_correct']:.4f}")
        print(f"  mean implied R      : {rec['mean_R']:.4f}")
        print(f"  embedding dispersion: {rec['mean_dispersion']:.4f}  (lower = draws agree)")
        del den
        torch.cuda.empty_cache()

    names = list(out["models"])
    if len(names) >= 2:
        a, b = names[0], names[-1]
        ra, rb = out["models"][a], out["models"][b]
        print(f"\n{b} vs {a}:")
        print(f"  d(p_bar)      = {rb['mean_pbar'] - ra['mean_pbar']:+.4f}")
        print(f"  d(R)          = {rb['mean_R'] - ra['mean_R']:+.4f}")
        print(f"  d(dispersion) = {rb['mean_dispersion'] - ra['mean_dispersion']:+.4f}")

    path = args.out or f"results/paper/pbar_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
