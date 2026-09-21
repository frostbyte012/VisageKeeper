#!/usr/bin/env python3
"""
scripts/eval_attenuation_aware.py -- did training FOR a(sigma) actually raise it?

The attenuation-aware objective (scripts/train_attenuation_aware.py) maximises a
BATCH-LEVEL slope between clean and denoised pairwise cosines. This script checks
that proxy against the real quantity: a(sigma) fitted the way the law defines it
(clean pair cosine vs mean denoised cosine for the SAME pair), plus the certified
accuracy that follows.

The DnCNN run already showed the two can come apart -- training a_hat rose to
0.83 while the true a FELL from 0.322 to 0.249, because the batch penalty is
dominated by near-orthogonal impostor-like pairs and can be satisfied by
inflating those rather than by preserving identity separation. So this
measurement, not the training log, is the verdict.

Run:
  python scripts/eval_attenuation_aware.py --arch vit --sigma 0.75 \
      --models baseline=results/denoiser_vit_s075_v3.pth \
               att0.3=results/denoiser_vit_s075_att0.3.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser, SmoothedVerifier
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--arch", default="vit")
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)
    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    tau_clean = float(np.quantile(clean[lab == 0], 1 - args.far))
    print(f"sigma={args.sigma}  tau_clean={tau_clean:.4f}  pairs={len(idx)}\n")

    out = {"sigma": args.sigma, "tau_clean": tau_clean, "models": {}}
    for spec in args.models:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[skip] {name}: missing {path}")
            continue
        den = load_den(args.arch, path, device)
        noisy = []
        with torch.no_grad():
            for j, i in enumerate(idx):
                z = img1[i].unsqueeze(0).repeat(args.draws, 1, 1, 1).to(device)
                z = (z + torch.randn_like(z) * args.sigma).clamp(0, 1)
                noisy.append(float((backbone.embed(den(z))
                                    * e2[j].to(device).unsqueeze(0)).sum(1).mean()))
        noisy = np.array(noisy)
        A = np.vstack([clean, np.ones_like(clean)]).T
        (a_hat, b_hat), *_ = np.linalg.lstsq(A, noisy, rcond=None)
        pred = A @ np.array([a_hat, b_hat])
        r2 = 1 - ((noisy - pred) ** 2).sum() / ((noisy - noisy.mean()) ** 2).sum()
        tau_an = a_hat * tau_clean + b_hat

        accs = {}
        for tname, t in (("clean", tau_clean), ("analytic", tau_an)):
            sv = SmoothedVerifier(backbone, den, sigma=args.sigma, tau=t, device=device)
            accs[tname] = float(np.mean([
                sv.predict(img1[i], e2[j].to(device), n=args.eval_n) == lab[j]
                for j, i in enumerate(idx)]))
        out["models"][name] = {"a": float(a_hat), "b": float(b_hat), "r2": float(r2),
                               "tau_analytic": float(tau_an), **accs}
        print(f"{name:>10}: a={a_hat:.4f}  R2={r2:.4f}  tau_an={tau_an:.4f}  "
              f"acc@clean={accs['clean']:.4f}  acc@analytic={accs['analytic']:.4f}")
        del den
        torch.cuda.empty_cache()

    names = list(out["models"])
    if len(names) >= 2:
        b, t = out["models"][names[0]], out["models"][names[-1]]
        print(f"\n{names[-1]} vs {names[0]}:  d_a={t['a']-b['a']:+.4f}  "
              f"d_acc@analytic={t['analytic']-b['analytic']:+.4f}")
        print("VERDICT: " + ("attenuation-aware training HELPED"
                             if t["a"] > b["a"] and t["analytic"] >= b["analytic"]
                             else "attenuation-aware training did NOT help"))

    path = args.out or f"results/paper/att_aware_{args.arch}_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
