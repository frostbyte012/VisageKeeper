#!/usr/bin/env python3
"""
scripts/probe_niche_coherence.py -- is there a NICHE for DnCNN to specialise into?

Decorrelation training can only create complementarity if a region of input
space exists where local convolution genuinely beats global attention. This
probe asks whether the draws DnCNN currently wins are a COHERENT region or just
noise.

Two tests, both on data we can generate from existing checkpoints:

  1. IMAGE-LEVEL CONSISTENCY. Split the Gaussian draws for each image into two
     halves. If DnCNN's wins are a property of the IMAGE, the per-image win-rate
     on half A predicts the win-rate on half B (high correlation). If wins are
     per-draw luck, the correlation is ~0.
     -> This is the decisive test. A niche must be a property of the input.

  2. PREDICTABILITY FROM IMAGE FEATURES. Fit a small logistic model from
     label-free image statistics (sharpness, contrast, edge density, ...) to
     "DnCNN wins", and report held-out AUC. A coherent niche is predictable;
     random wins are not (AUC ~0.5).

Interpretation: if split-half correlation is near zero AND feature AUC is near
0.5, DnCNN's wins are coin-flips on draws ViT also mostly gets right, there is
no niche, and decorrelation training has no target.

Run:
  python scripts/probe_niche_coherence.py --sigma 0.5 --n_pairs 150 --draws 64
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

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser
from frpure.defenses.cascade import _disable_fused_attention
from scripts.run_certify import load_pairs, balanced_subset
from scripts.diag_error_overlap import CKPT


def load_den(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(d)
    for p in d.parameters():
        p.requires_grad_(False)
    return d


def image_features(x):
    """Label-free statistics of the CLEAN image (1,3,H,W) -> feature vector."""
    g = x.mean(1, keepdim=True)
    lap = g - F.avg_pool2d(g, 3, 1, 1)
    gx = (g[:, :, :, 1:] - g[:, :, :, :-1]).abs().mean()
    gy = (g[:, :, 1:, :] - g[:, :, :-1, :]).abs().mean()
    hi = F.avg_pool2d(g, 4, 4)
    return np.array([
        float(g.mean()),                    # brightness
        float(g.std()),                     # contrast
        float(lap.pow(2).mean()),           # sharpness / high-freq energy
        float(gx), float(gy),               # edge density (h / v)
        float(hi.std()),                    # coarse-scale structure
        float((g > g.mean()).float().mean()),  # bright-pixel fraction
    ], dtype=np.float64)


FEATURE_NAMES = ["brightness", "contrast", "sharpness", "edge_h", "edge_v",
                 "coarse_std", "bright_frac"]


def auc(pos, neg, cap=4000, seed=0):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    sub = lambda a: a[rng.choice(len(a), min(cap, len(a)), replace=False)]
    p, n = sub(np.asarray(pos)), sub(np.asarray(neg))
    # rank-based AUC (no O(n^2) blowup)
    allv = np.concatenate([p, n])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    rp = ranks[:len(p)].sum()
    return float((rp - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=150)
    ap.add_argument("--draws", type=int, default=64, help="even; split in half")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)
    img1, _, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs * 2, len(same)))[:args.n_pairs]

    dn_w, vit_w = CKPT[args.sigma]
    cheap, exp = load_den("dncnn", dn_w, device), load_den("vit", vit_w, device)

    half = args.draws // 2
    rateA, rateB, feats, overall = [], [], [], []
    with torch.no_grad():
        for i in idx:
            x0 = img1[i].unsqueeze(0).to(device)
            e0 = backbone.embed(x0)
            z = (x0.repeat(args.draws, 1, 1, 1)
                 + torch.randn(args.draws, 3, *x0.shape[-2:], device=device) * args.sigma).clamp(0, 1)
            dc = (backbone.embed(cheap(z)) * e0).sum(1)
            dv = (backbone.embed(exp(z)) * e0).sum(1)
            win = (dc >= dv).float().cpu().numpy()
            rateA.append(win[:half].mean()); rateB.append(win[half:].mean())
            overall.append(win.mean())
            feats.append(image_features(x0))

    rateA = np.array(rateA); rateB = np.array(rateB)
    overall = np.array(overall); feats = np.stack(feats)

    # ---- test 1: split-half consistency ----------------------------------
    r = float(np.corrcoef(rateA, rateB)[0, 1])
    # Spearman-Brown correction for the full-length reliability
    rel = 2 * r / (1 + r) if r > -1 else float("nan")
    print("=== TEST 1: split-half image-level consistency ===")
    print(f"  DnCNN overall win-rate      : {overall.mean():.3f}")
    print(f"  per-image win-rate spread   : sd={overall.std():.3f} "
          f"[min {overall.min():.3f}, max {overall.max():.3f}]")
    print(f"  corr(halfA, halfB)          : {r:+.3f}")
    print(f"  Spearman-Brown reliability  : {rel:+.3f}")
    print("  (near 0 => wins are per-draw luck, not a property of the image)")

    # ---- test 2: predictability from clean-image features ----------------
    y = (overall >= 0.5).astype(int)
    print(f"\n=== TEST 2: predictability from image features ===")
    print(f"  images where DnCNN wins majority: {y.sum()}/{len(y)}")
    per_feature = {}
    for k, name in enumerate(FEATURE_NAMES):
        a = auc(feats[y == 1, k], feats[y == 0, k])
        per_feature[name] = a
        print(f"    {name:>12}: AUC={a:.3f}")

    # held-out logistic regression on all features
    cv_auc = float("nan")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score
        if 0 < y.sum() < len(y):
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            preds = np.zeros(len(y))
            for tr, te in skf.split(feats, y):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000))
                m.fit(feats[tr], y[tr])
                preds[te] = m.predict_proba(feats[te])[:, 1]
            cv_auc = float(roc_auc_score(y, preds))
            print(f"  5-fold held-out logistic AUC : {cv_auc:.3f}  (0.5 = no signal)")
    except ImportError:
        print("  (sklearn unavailable -- skipping multivariate fit)")

    coherent = (abs(rel) >= 0.30) or (not np.isnan(cv_auc) and cv_auc >= 0.65)
    verdict = ("NICHE EXISTS - DnCNN wins are structured; decorrelation has a target"
               if coherent else
               "NO NICHE - wins are per-draw noise; decorrelation has nothing to aim at")
    print(f"\nVERDICT: {verdict}")

    out = args.out or f"results/paper/probe_niche_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"sigma": args.sigma, "n_images": len(idx), "draws": args.draws,
                   "dncnn_win_rate": float(overall.mean()),
                   "win_rate_sd": float(overall.std()),
                   "split_half_corr": r, "spearman_brown": rel,
                   "feature_auc": per_feature, "cv_logistic_auc": cv_auc,
                   "verdict": verdict}, f, indent=2)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
