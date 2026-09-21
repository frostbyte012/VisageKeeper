#!/usr/bin/env python3
"""
scripts/probe_cross_sigma.py -- do NOISE-SPECIALISED denoisers fail differently?

Same-sigma DnCNN/ViT pairs have nested errors (ViT wins ~everywhere DnCNN does),
capping any router at ~+1.5% over ViT alone (see diag_error_overlap.py). This
probe asks whether SPECIALISATION creates the complementarity that identical
training did not: pair a DnCNN trained at LOW sigma with a ViT trained at HIGH
sigma, then evaluate both at an intermediate test sigma.

That mismatched pair is exactly what a noise-intensity router would switch
between, so the oracle ceiling here is the headroom such a router could ever
have. We report it against the best same-sigma single model, which is the bar
any dynamic algorithm must clear to be worth building.

Run:
  python scripts/probe_cross_sigma.py --test_sigma 0.5 --n_pairs 200
"""
from __future__ import annotations

import argparse
import itertools
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

DNCNN = {0.05: "results/denoiser_dncnn_s005_30ep.pth",
         0.1:  "results/denoiser_dncnn_s010_30ep.pth",
         0.25: "results/denoiser_dncnn_s025_30ep.pth",
         0.5:  "results/denoiser_dncnn_s050_30ep.pth",
         0.75: "results/denoiser_dncnn_s075_30ep.pth",
         1.0:  "results/denoiser_dncnn_s100_30ep.pth"}
VIT = {0.05: "results/denoiser_vit_s005_v3.pth",
       0.1:  "results/denoiser_vit_s010_v3.pth",
       0.25: "results/denoiser_vit_s025_v3.pth",
       0.5:  "results/denoiser_vit_s050_v3.pth",
       0.75: "results/denoiser_vit_s075_v3.pth",
       1.0:  "results/denoiser_vit_s100_v3.pth"}


def load_den(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(d)          # NVML-safe on this host
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
    ap.add_argument("--test_sigma", type=float, default=0.5)
    ap.add_argument("--train_sigmas", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75, 1.0],
                    help="checkpoint sigmas to pair up")
    ap.add_argument("--n_pairs", type=int, default=200)
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
    cos_clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    imp = cos_clean[lab == 0]
    tau = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3
    print(f"test_sigma={args.test_sigma}  tau={tau:.4f}  pairs={len(idx)}\n")

    # per-model correctness vector over the SAME pairs, so any pair of models
    # can be cross-tabulated afterwards
    correct = {}
    for arch, table in (("dncnn", DNCNN), ("vit", VIT)):
        for s in args.train_sigmas:
            if s not in table or not os.path.exists(table[s]):
                continue
            den = load_den(arch, table[s], device)
            sv = SmoothedVerifier(backbone, den, sigma=args.test_sigma,
                                  tau=tau, device=device)
            ok = np.array([sv.predict(img1[i], e2[j].to(device), n=args.eval_n) == lab[j]
                           for j, i in enumerate(idx)])
            correct[f"{arch}@{s}"] = ok
            print(f"  {arch:>5} trained@{s:<5} -> acc {ok.mean():.3f}")
            del den
            torch.cuda.empty_cache()

    best_single = max(correct.items(), key=lambda kv: kv[1].mean())
    print(f"\nbest single model: {best_single[0]} = {best_single[1].mean():.3f}")

    # cross-tabulate every DnCNN x ViT pair; the interesting cell is a pair
    # whose oracle clears the best single model by a wide margin
    print("\n=== pairwise oracle ceilings (DnCNN x ViT) ===")
    rows = []
    dn = [k for k in correct if k.startswith("dncnn")]
    vt = [k for k in correct if k.startswith("vit")]
    for a, b in itertools.product(dn, vt):
        oa, ob = correct[a], correct[b]
        neither = float((~oa & ~ob).mean())
        oracle = 1 - neither
        gain = oracle - best_single[1].mean()
        rows.append({"dncnn": a, "vit": b, "acc_dncnn": float(oa.mean()),
                     "acc_vit": float(ob.mean()),
                     "only_dncnn": float((oa & ~ob).mean()),
                     "only_vit": float((~oa & ob).mean()),
                     "oracle": oracle, "gain_over_best_single": gain})
    rows.sort(key=lambda r: -r["gain_over_best_single"])
    for r in rows[:12]:
        print(f"  {r['dncnn']:>12} ({r['acc_dncnn']:.3f}) x {r['vit']:>10} ({r['acc_vit']:.3f})"
              f" | onlyD={r['only_dncnn']:.3f} onlyV={r['only_vit']:.3f}"
              f" | oracle={r['oracle']:.3f}  gain={r['gain_over_best_single']:+.3f}")

    best = rows[0]
    print(f"\nBEST PAIRING: {best['dncnn']} x {best['vit']}")
    print(f"  oracle ceiling      : {best['oracle']:.3f}")
    print(f"  best single model   : {best_single[1].mean():.3f}")
    print(f"  MAX router headroom : {best['gain_over_best_single']:+.3f}")
    verdict = ("PROMISING - specialisation created real complementarity"
               if best["gain_over_best_single"] >= 0.03 else
               "DEAD END - errors still nested, router cannot pay off")
    print(f"  VERDICT: {verdict}")

    out = args.out or f"results/paper/probe_cross_sigma_s{int(args.test_sigma*100):03d}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"test_sigma": args.test_sigma, "tau": tau, "n_pairs": len(idx),
                   "acc": {k: float(v.mean()) for k, v in correct.items()},
                   "best_single": {"model": best_single[0],
                                   "acc": float(best_single[1].mean())},
                   "pairs": rows, "verdict": verdict}, f, indent=2)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
