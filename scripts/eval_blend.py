#!/usr/bin/env python3
"""
scripts/eval_blend.py -- ENSEMBLE (blend) the two denoisers instead of routing.

Why this experiment exists
--------------------------
Every routing attempt failed: post-hoc cascade (no gain), cross-sigma
specialisation (oracle +1%), learned MoE gate (dead -- logit std 0.005). The
diagnostic that explains all three:

    soft 50/50 blend cos = 0.7365
    ORACLE routing   cos = 0.7225      <-- perfect selection is WORSE
    ViT alone        cos = 0.6936
    DnCNN alone      cos = 0.6628

Averaging both experts beats perfectly CHOOSING between them. The learned gate
is dead because the loss is already minimised by a constant blend -- there is
no gradient toward discrimination. Routing was simply the wrong operation:
these denoisers' errors are partly independent, so averaging cancels noise,
while selection throws half the information away.

This script measures whether that advantage survives at the VERIFICATION level
(smoothed accuracy), which is what the paper actually claims on. A static blend
has no gate, so it is trivially certificate-safe: it is a single deterministic
denoiser D(x) = w*ViT(x) + (1-w)*DnCNN(x).

Run:
  python scripts/eval_blend.py --sigma 0.5 --n_pairs 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser, SmoothedVerifier
from frpure.defenses.cascade import _disable_fused_attention
from scripts.run_certify import load_pairs, balanced_subset, embed_all, RADII
from scripts.diag_error_overlap import CKPT


class Blend(nn.Module):
    """D(x) = w*ViT(x) + (1-w)*DnCNN(x). Deterministic -> certificate-safe."""

    def __init__(self, cheap, expensive, w):
        super().__init__()
        self.cheap = cheap
        self.expensive = expensive
        self.w = w

    def forward(self, x):
        return (self.w * self.expensive(x) + (1 - self.w) * self.cheap(x)).clamp(0, 1)


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
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0])
    ap.add_argument("--certify_best", action="store_true",
                    help="also run full Cohen certification for the best w and ViT-only")
    ap.add_argument("--n0", type=int, default=100)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=1e-3)
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
    print(f"sigma={args.sigma}  tau={tau:.4f}  pairs={len(idx)}\n")

    dn_w, vit_w = CKPT[args.sigma]
    cheap, exp = load_den("dncnn", dn_w, device), load_den("vit", vit_w, device)

    accs = {}
    for w in args.weights:
        sv = SmoothedVerifier(backbone, Blend(cheap, exp, w), sigma=args.sigma,
                              tau=tau, device=device)
        ok = np.array([sv.predict(img1[i], e2[j].to(device), n=args.eval_n) == lab[j]
                       for j, i in enumerate(idx)])
        accs[w] = float(ok.mean())
        tag = "  (DnCNN-only)" if w == 0 else "  (ViT-only)" if w == 1 else ""
        print(f"  w_vit={w:.2f}: acc={accs[w]:.3f}{tag}")

    vit_acc = accs.get(1.0, float("nan"))
    best_w = max(accs, key=accs.get)
    gain = accs[best_w] - vit_acc
    print(f"\nBEST w={best_w:.2f}  acc={accs[best_w]:.3f}   ViT-only={vit_acc:.3f}   "
          f"gain={gain:+.3f}")
    # 200-pair binomial noise band, so we do not over-read a 1-2 pair difference
    se = float(np.sqrt(vit_acc * (1 - vit_acc) / len(idx)))
    print(f"(1 s.e. on {len(idx)} pairs ~ {se:.3f}; "
          f"{'SIGNIFICANT' if abs(gain) > 2 * se else 'WITHIN NOISE'} at 2 s.e.)")

    out = {"sigma": args.sigma, "tau": tau, "n_pairs": len(idx),
           "acc_by_weight": accs, "best_w": best_w, "vit_only": vit_acc,
           "gain_over_vit": gain, "se": se,
           "significant": bool(abs(gain) > 2 * se)}

    # ---- optional: full certification for best blend vs ViT-only ----------
    if args.certify_best:
        print("\ncertifying best blend and ViT-only ...")
        out["certified"] = {}
        for name, w in (("ViT-only", 1.0), (f"Blend(w={best_w:.2f})", best_w)):
            sv = SmoothedVerifier(backbone, Blend(cheap, exp, w), sigma=args.sigma,
                                  tau=tau, device=device)
            preds, radii = [], []
            for j, i in enumerate(idx):
                p, R = sv.certify(img1[i], e2[j].to(device), n0=args.n0,
                                  n=args.n, alpha=args.alpha)
                preds.append(p); radii.append(R)
            preds = np.array(preds); radii = np.array(radii)
            corr = (preds == lab) & (preds != -1)
            rec = {"smoothed_acc": float(corr.mean()),
                   "abstention": float((preds == -1).mean()),
                   "mean_radius": float(radii[corr].mean()) if corr.any() else 0.0,
                   "certified_accuracy": {str(r): float((corr & (radii >= r)).mean())
                                          for r in RADII}}
            out["certified"][name] = rec
            print(f"  {name}: smoothed={rec['smoothed_acc']:.3f} "
                  f"meanR={rec['mean_radius']:.3f}")
            for r in RADII:
                print(f"      R>={r:<4}: {rec['certified_accuracy'][str(r)]:.3f}")

    path = args.out or f"results/paper/blend_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
