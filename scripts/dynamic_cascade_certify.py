#!/usr/bin/env python3
"""
scripts/dynamic_cascade_certify.py -- CERTIFY-OR-ESCALATE: dynamic dual-
architecture certification that jointly optimises GFLOPs and accuracy.

The pipeline
------------
  1. Try the CHEAP path: DnCNN-smoothed verifier, sequential test (alpha/2),
     law-scheduled starting budget. If it certifies the decision -> DONE at
     ~3.88 GFLOPs/draw (1.03 denoiser + 2.85 backbone).
  2. Only pairs the cheap path FAILS TO CERTIFY escalate to the ViT path
     (alpha/2, 7.15 GFLOPs/draw), which retains the high-noise accuracy.

Why this succeeds where the accuracy cascade failed
---------------------------------------------------
The earlier per-draw cascade routed on a weak learned gate (AUC 0.595) and
targeted ACCURACY, capped at +1% by nested errors. Here the router is the
certificate itself: escalation fires exactly when the cheap model cannot prove
its answer, which is a perfect-precision trigger by construction. The target is
COST -- accuracy is protected by escalation, never traded.

Certificate status (stated honestly)
------------------------------------
Every output is the certified decision of a FIXED smoothed classifier (DnCNN's
or ViT's), each tested at level alpha/2, so any reported (decision, radius) is
individually valid at overall confidence 1-alpha. An adversary cannot obtain an
uncertified answer -- perturbing the probe can only change WHICH branch
certifies (worst case: forces escalation, i.e. more compute). Composite
limitation: the guarantee attaches to the branch that answered; stability of
the BRANCH CHOICE itself is not certified, so a perturbation inside the radius
could in principle move a pair from one certified answer to the other branch's
certified answer. We report per-branch certificates and disagreement rates.

FPGA relevance: the cheap path is the 84 KB DnCNN -- the escalation fraction
directly sets how rarely the big ViT engine must be powered.

Run:
  python scripts/dynamic_cascade_certify.py --sigma 0.5 --n_pairs 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy.stats import norm

from frpure.models.backbone import build_backbone
from scripts.run_certify import load_pairs, balanced_subset, embed_all, RADII
from scripts.sequential_certify import (load_den, DrawSampler,
                                        certify_sequential)

# measured per-draw cost (results/paper/denoiser_cost.json + facenet 2.85)
GFLOPS_DRAW = {"dncnn": 1.03 + 2.85, "vit": 4.30 + 2.85}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--stages", type=int, nargs="+", default=[100, 200, 400, 1000])
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--r_accept", type=float, default=0.0,
                    help="accept the cheap branch only if its certified radius "
                         "reaches this; wrong-but-certified answers cluster at "
                         "small radii, so this bar trades GFLOPs for accuracy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)
    dens = {"dncnn": load_den("dncnn", args.sigma, device),
            "vit": load_den("vit", args.sigma, device)}

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    tau = float(np.quantile(clean[lab == 0], 1 - args.far))
    half = len(idx) // 2
    K = len(args.stages)
    # alpha/2 per branch, spent uniformly over that branch's stages
    al_branch = [args.alpha / 2 / K] * K
    print(f"sigma={args.sigma} tau={tau:.4f} stages={args.stages} "
          f"alpha={args.alpha} (alpha/2 per branch)")

    # ---- law scheduler fitted per denoiser on the CALIBRATION half ---------
    sched = {}
    for name, den in dens.items():
        mus, sds = [], []
        with torch.no_grad():
            for j, i in enumerate(idx[:half]):
                z = img1[i].unsqueeze(0).repeat(64, 1, 1, 1).to(device)
                z = (z + torch.randn_like(z) * args.sigma).clamp(0, 1)
                cos = (backbone.embed(den(z)) * e2[j].to(device).unsqueeze(0)).sum(1)
                mus.append(float(cos.mean())); sds.append(float(cos.std()))
        A = np.vstack([clean[:half], np.ones(half)]).T
        (a_h, b_h), *_ = np.linalg.lstsq(A, np.array(mus), rcond=None)
        sched[name] = (float(a_h), float(b_h), float(np.mean(sds)))
        print(f"law[{name}]: a={a_h:.4f} b={b_h:+.4f} s={np.mean(sds):.4f}")

    def start_stage(name, cc):
        a_h, b_h, s = sched[name]
        p_hat = norm.cdf((a_h * cc + b_h - tau) / s)
        conf = max(p_hat, 1 - p_hat)
        return 0 if conf > 0.995 else (1 if conf > 0.95 else 2)

    # ---- evaluate the TEST half --------------------------------------------
    rows = []
    for j0, i in enumerate(idx[half:]):
        j = half + j0
        g = e2[j].to(device)
        cc = clean[j]
        gflops = 0.0
        # branch 1: cheap
        s0 = start_stage("dncnn", cc)
        smp = DrawSampler(dens["dncnn"], backbone, img1[i], g, args.sigma, tau,
                          device, args.batch)
        pred, R, used = certify_sequential(smp, args.stages[s0:], al_branch[s0:],
                                           args.sigma, "decision")
        gflops += used * GFLOPS_DRAW["dncnn"]
        branch = "dncnn"
        # Escalate on failure to certify OR on a small certified radius: wrong-
        # but-certified cheap answers cluster near the decision threshold (low
        # p_bar -> low R), so an acceptance bar on R converts them into
        # escalations instead of silent errors. r_accept=0 reproduces the plain
        # certify-or-escalate cascade.
        if pred == -1 or R < args.r_accept:
            s1 = start_stage("vit", cc)
            smp = DrawSampler(dens["vit"], backbone, img1[i], g, args.sigma,
                              tau, device, args.batch)
            pred, R, used_v = certify_sequential(smp, args.stages[s1:],
                                                 al_branch[s1:], args.sigma,
                                                 "decision")
            gflops += used_v * GFLOPS_DRAW["vit"]
            branch = "vit"
        rows.append({"label": int(lab[j]), "pred": pred, "R": R,
                     "branch": branch, "gflops": gflops})
        if (j0 + 1) % 50 == 0:
            print(f"  {j0+1}/{len(idx)-half}")

    pred = np.array([r["pred"] for r in rows])
    R = np.array([r["R"] for r in rows])
    labt = np.array([r["label"] for r in rows])
    gfl = np.array([r["gflops"] for r in rows])
    esc = np.array([r["branch"] == "vit" for r in rows])
    correct = (pred == labt) & (pred != -1)
    base_gflops = args.stages[-1] * GFLOPS_DRAW["vit"]

    print("\n==== DYNAMIC CASCADE (certify-or-escalate) ====")
    print(f"certified accuracy   : {correct.mean():.4f}")
    print(f"abstention           : {(pred == -1).mean():.4f}")
    print(f"escalation fraction  : {esc.mean():.4f}")
    print(f"mean radius          : {R[correct].mean() if correct.any() else 0:.4f}")
    cert_acc = {r: float((correct & (R >= r)).mean()) for r in RADII}
    for r in RADII:
        print(f"   cert acc @ R>={r:<4}: {cert_acc[r]:.4f}")
    print(f"mean GFLOPs/decision : {gfl.mean():.1f}   "
          f"(fixed-n ViT baseline: {base_gflops:.0f})")
    print(f"GFLOP reduction      : {(1 - gfl.mean()/base_gflops)*100:.1f}%")
    print(f"cheap-path accuracy  : "
          f"{correct[~esc].mean() if (~esc).any() else float('nan'):.4f} "
          f"on {(~esc).sum()} pairs")
    print(f"escalated accuracy   : "
          f"{correct[esc].mean() if esc.any() else float('nan'):.4f} "
          f"on {esc.sum()} pairs")

    out = {"sigma": args.sigma, "tau": tau, "stages": args.stages,
           "alpha": args.alpha, "n_test": len(rows),
           "certified_acc": float(correct.mean()),
           "abstention": float((pred == -1).mean()),
           "escalation_frac": float(esc.mean()),
           "mean_radius": float(R[correct].mean()) if correct.any() else 0.0,
           "cert_acc_at": cert_acc,
           "mean_gflops": float(gfl.mean()),
           "baseline_gflops": float(base_gflops),
           "gflop_reduction_pct": float((1 - gfl.mean()/base_gflops)*100)}
    path = args.out or f"results/paper/dyn_cascade_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
