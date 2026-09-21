#!/usr/bin/env python3
"""
scripts/run_cascade.py -- calibrate AND evaluate the cost-aware dual-architecture
(DnCNN + ViT) cascade for denoised smoothing.

Protocol (leakage-free):
  1. split the balanced pair subset into CALIBRATION and TEST halves (disjoint),
  2. on CALIBRATION only, sweep the gate threshold t and pick the largest t*
     (= most traffic on the cheap branch) whose smoothed accuracy stays within
     --tol of the ViT-only accuracy,
  3. on TEST, report smoothed accuracy + certified accuracy + realised cheap
     fraction for: DnCNN-only, ViT-only, cascade(t*).

The gate lives inside the base classifier (see frpure/defenses/cascade.py), so
the Cohen certificate applies to the cascade unchanged -- we certify it with the
SAME SmoothedVerifier.certify() used for the single-architecture baselines.

Run:
  python scripts/run_cascade.py --aligned_dir DATA/lfw_aligned_160 \
      --pairs DATA/pairs.txt --sigma 0.5 --n_pairs 400 --n 1000
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
from frpure.defenses.cascade import CascadedDenoiser
from scripts.run_certify import load_pairs, balanced_subset, embed_all, RADII
from scripts.diag_error_overlap import CKPT


def load_den(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    for p in d.parameters():
        p.requires_grad_(False)
    return d


def smoothed_acc(sv, img1, idx, e2, labels, eval_n):
    """Plain smoothed (majority-vote) accuracy over a pair subset."""
    ok = 0
    for j, i in enumerate(idx):
        if sv.predict(img1[i], e2[j].to(sv.device), n=eval_n) == labels[j]:
            ok += 1
    return ok / len(idx)


def certify_all(sv, img1, idx, e2, labels, n0, n, alpha, batch):
    preds, radii = [], []
    for j, i in enumerate(idx):
        p, R = sv.certify(img1[i], e2[j].to(sv.device), n0=n0, n=n,
                          alpha=alpha, batch=batch)
        preds.append(p); radii.append(R)
    preds = np.array(preds); radii = np.array(radii); labels = np.asarray(labels)
    correct = (preds == labels) & (preds != -1)
    return {
        "smoothed_acc": float(correct.mean()),
        "abstention": float((preds == -1).mean()),
        "mean_radius": float(radii[correct].mean()) if correct.any() else 0.0,
        "certified_accuracy": {str(r): float((correct & (radii >= r)).mean())
                               for r in RADII},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir"); ap.add_argument("--pairs")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=400, help="split 50/50 calib/test")
    ap.add_argument("--eval_n", type=int, default=200, help="votes for calibration")
    ap.add_argument("--n0", type=int, default=100)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max smoothed-acc drop vs ViT-only allowed on CALIB")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy(); sub_same = same[idx]
    imp = cos_clean[sub_same == 0]
    tau = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3

    half = len(idx) // 2
    cal_sl, test_sl = slice(0, half), slice(half, len(idx))
    print(f"sigma={args.sigma}  tau={tau:.4f}  calib={half}  test={len(idx)-half}")

    dn_w, vit_w = CKPT[args.sigma]
    cheap = load_den("dncnn", dn_w, device)
    exp = load_den("vit", vit_w, device)
    casc = CascadedDenoiser(cheap, exp, thresh=0.0, mode="cascade").to(device).eval()

    def sv_for(mode, thresh=0.0):
        casc.mode, casc.thresh = mode, thresh
        return SmoothedVerifier(backbone, casc, sigma=args.sigma, tau=tau,
                                device=device)

    # ---- gate-score distribution on CALIB, to pick a sensible sweep grid ----
    casc.mode = "cascade"
    scores = []
    with torch.no_grad():
        for i in idx[cal_sl][:64]:
            x = img1[i].unsqueeze(0).repeat(32, 1, 1, 1).to(device)
            x = (x + torch.randn_like(x) * args.sigma).clamp(0, 1)
            scores.append(CascadedDenoiser.gate_score(x, cheap(x)).cpu())
    scores = torch.cat(scores).numpy()
    # gate is output TV: cheap branch takes scores BELOW thresh, so a HIGH
    # quantile = more traffic on the cheap path.
    grid = [float(np.quantile(scores, q)) for q in
            (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]
    print(f"gate score range [{scores.min():.4f}, {scores.max():.4f}]")

    cal_idx, cal_lab = idx[cal_sl], sub_same[cal_sl]
    cal_e2 = e2[cal_sl]

    # ---- calibration: baselines then threshold sweep ----------------------
    acc_vit_cal = smoothed_acc(sv_for("expensive"), img1, cal_idx, cal_e2,
                               cal_lab, args.eval_n)
    acc_dn_cal = smoothed_acc(sv_for("cheap"), img1, cal_idx, cal_e2,
                              cal_lab, args.eval_n)
    print(f"[calib] ViT-only={acc_vit_cal:.3f}  DnCNN-only={acc_dn_cal:.3f}")

    sweep, best_t, best_frac = [], grid[-1], 0.0
    for t in grid:
        sv = sv_for("cascade", t)
        casc.reset_stats()
        a = smoothed_acc(sv, img1, cal_idx, cal_e2, cal_lab, args.eval_n)
        frac = casc.cheap_fraction
        sweep.append({"thresh": t, "acc": a, "cheap_frac": frac})
        print(f"[calib] t={t:+.4f}  acc={a:.3f}  cheap_frac={frac:.3f}")
        if a >= acc_vit_cal - args.tol and frac > best_frac:
            best_t, best_frac = t, frac
    print(f"\nselected t* = {best_t:+.4f}  (calib cheap_frac={best_frac:.3f})")

    # ---- TEST: certify all three ------------------------------------------
    test_idx, test_lab = idx[test_sl], sub_same[test_sl]
    test_e2 = e2[test_sl]
    out = {"sigma": args.sigma, "tau": tau, "thresh": best_t,
           "tol": args.tol, "n_calib": half, "n_test": len(test_idx),
           "calib_sweep": sweep, "calib_acc_vit": acc_vit_cal,
           "calib_acc_dncnn": acc_dn_cal, "arms": {}}

    for name, mode in (("DnCNN-only", "cheap"), ("ViT-only", "expensive"),
                       ("Cascade", "cascade")):
        sv = sv_for(mode, best_t)
        casc.reset_stats()
        res = certify_all(sv, img1, test_idx, test_e2, test_lab,
                          args.n0, args.n, args.alpha, args.batch)
        res["cheap_fraction"] = casc.cheap_fraction
        out["arms"][name] = res
        print(f"\n[test] {name}: smoothed={res['smoothed_acc']:.3f} "
              f"abstain={res['abstention']:.3f} meanR={res['mean_radius']:.3f} "
              f"cheap_frac={res['cheap_fraction']:.3f}")
        for r in RADII:
            print(f"        R>={r:<4}: {res['certified_accuracy'][str(r)]:.3f}")

    path = args.out or f"results/cascade_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
