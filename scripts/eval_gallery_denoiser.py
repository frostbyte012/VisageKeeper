#!/usr/bin/env python3
"""
scripts/eval_gallery_denoiser.py -- does conditioning on the enrolled embedding
actually help certified verification?

Reports GENUINE and IMPOSTOR accuracy SEPARATELY, because the failure mode of
identity conditioning is confirmation bias: a denoiser that pulls every probe
toward the claimed identity raises genuine similarity and impostor similarity
together. That looks like a gain if you only read overall accuracy, but it is
just an inflated FAR. A real win must not lose impostor accuracy.

Arms compared (all through the same smoothing code path):
  uncond-DnCNN   the unconditional baseline the model was warm-started from
  gallery-cond   the trained conditional model, conditioned on the TRUE claim
  gallery-null   the same model with cond=None (isolates the conv-stack change
                 from the conditioning itself)

Run:
  python scripts/eval_gallery_denoiser.py --sigma 0.5 --n_pairs 200 \
      --weights results/gallery_den_s050.pth
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
from frpure.defenses.gallery_denoiser import GalleryConditionedDenoiser, GalleryWrapper
from scripts.run_certify import load_pairs, balanced_subset, embed_all
from scripts.diag_error_overlap import CKPT


class NullCond(nn.Module):
    """Conditional model run WITHOUT conditioning (ablation)."""
    def __init__(self, core):
        super().__init__()
        self.core = core
    def forward(self, x):
        return self.core(x, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--weights", default="results/gallery_den_s050.pth")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=None)
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
    tau = args.tau if args.tau is not None else (
        float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3)
    print(f"sigma={args.sigma}  tau={tau:.4f}  pairs={len(idx)} "
          f"({int((lab==1).sum())} genuine / {int((lab==0).sum())} impostor)\n")

    core = GalleryConditionedDenoiser(ch=args.ch).to(device).eval()
    core.load_state_dict(torch.load(args.weights, map_location=device))
    for p in core.parameters():
        p.requires_grad_(False)

    base = build_denoiser("dncnn", ch=args.ch).to(device).eval()
    base.load_state_dict(torch.load(CKPT[args.sigma][0], map_location=device))
    for p in base.parameters():
        p.requires_grad_(False)

    def run(make_den, name):
        """make_den(j) -> denoiser for pair j (conditioning may depend on j)."""
        ok = np.zeros(len(idx), dtype=bool)
        for j, i in enumerate(idx):
            sv = SmoothedVerifier(backbone, make_den(j), sigma=args.sigma,
                                  tau=tau, device=device)
            ok[j] = sv.predict(img1[i], e2[j].to(device), n=args.eval_n) == lab[j]
        g = ok[lab == 1].mean(); m = ok[lab == 0].mean()
        print(f"  {name:>14}: overall={ok.mean():.4f}  genuine={g:.4f}  impostor={m:.4f}")
        return {"overall": float(ok.mean()), "genuine": float(g),
                "impostor": float(m)}

    res = {}
    res["uncond-DnCNN"] = run(lambda j: base, "uncond-DnCNN")
    res["gallery-null"] = run(lambda j: NullCond(core), "gallery-null")
    res["gallery-cond"] = run(lambda j: GalleryWrapper(core, e2[j].to(device)).to(device),
                              "gallery-cond")

    b, c = res["uncond-DnCNN"], res["gallery-cond"]
    print(f"\ngallery-cond vs uncond baseline:")
    for k in ("overall", "genuine", "impostor"):
        print(f"  d_{k:<9} = {c[k] - b[k]:+.4f}")
    n = len(idx)
    se = float(np.sqrt(b["overall"] * (1 - b["overall"]) / n))
    print(f"  (1 s.e. ~ {se:.4f} on {n} pairs)")
    if c["impostor"] < b["impostor"] - 2 * se:
        print("  WARNING: impostor accuracy dropped -- confirmation bias, not a win.")

    out = {"sigma": args.sigma, "tau": tau, "n_pairs": n, "arms": res, "se": se}
    path = args.out or f"results/paper/gallery_eval_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
