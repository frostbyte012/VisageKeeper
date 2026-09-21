#!/usr/bin/env python3
"""
scripts/attenuation_law.py -- the cosine ATTENUATION LAW under denoised smoothing,
and the analytic threshold it implies.

Observation
-----------
Gaussian smoothing destroys identity signal asymmetrically. Genuine pairs, whose
similarity comes from that signal, collapse toward the threshold; impostor pairs
are already near-orthogonal and barely move. Measured at sigma=1.0 the genuine
mean falls by -0.302 while the impostor mean rises only +0.031.

The useful part is that the induced map is close to AFFINE:

    E[cos_noisy]  ~=  a(sigma) * cos_clean  +  b(sigma)

with a(sigma) the fraction of clean cosine signal the denoiser retains. Fitting
(a, b) once per sigma lets us set the verification threshold ANALYTICALLY,

    tau(sigma) = a(sigma) * tau_clean + b(sigma)

instead of brute-force sweeping tau at every noise level. This explains WHY the
clean-FAR threshold is miscalibrated under smoothing (it ignores a, b entirely)
and by how much.

This script:
  1. fits (a, b) and R^2 per sigma, per denoiser architecture -> does the law
     generalise, or is it a ViT artefact?
  2. compares CERTIFIED ACCURACY at three thresholds -- clean-FAR tau, analytic
     tau, and the empirically swept optimum -- on held-out pairs. The analytic
     tau is only useful if it captures most of the swept tau's gain.

Run:
  python scripts/attenuation_law.py --sigmas 0.25 0.5 0.75 1.0 --arch vit dncnn
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
from frpure.defenses.smoothing import build_denoiser
from frpure.defenses.cascade import _disable_fused_attention
from scripts.run_certify import load_pairs, balanced_subset, embed_all
from scripts.diag_error_overlap import CKPT


def load_den(arch, sigma, device):
    dn_w, vit_w = CKPT[sigma]
    path = vit_w if arch == "vit" else dn_w
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(d)
    for p in d.parameters():
        p.requires_grad_(False)
    return d


@torch.no_grad()
def draw_cosines(den, backbone, img1, idx, e2, sigma, draws, batch, device):
    """(P, draws) cosines of denoised noisy probes against their gallery."""
    out = []
    for j, i in enumerate(idx):
        g = e2[j].to(device).unsqueeze(0)
        cs, rem = [], draws
        while rem > 0:
            m = min(batch, rem)
            z = img1[i].unsqueeze(0).repeat(m, 1, 1, 1).to(device)
            z = (z + torch.randn_like(z) * sigma).clamp(0, 1)
            cs.append((backbone.embed(den(z)) * g).sum(1).cpu().numpy())
            rem -= m
        out.append(np.concatenate(cs))
    return np.stack(out)


def cert_stats(cos, lab, tau, sigma):
    pbar = ((cos >= tau).astype(int) == lab[:, None]).mean(1)
    cert = pbar > 0.5
    R = np.where(cert, sigma * norm.ppf(np.clip(pbar, 1e-6, 1 - 1e-6)), 0.0)
    return {"acc": float(cert.mean()),
            "meanR": float(R[cert].mean()) if cert.any() else 0.0,
            "pbar": float(pbar.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--arch", nargs="+", default=["vit", "dncnn"])
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/paper/attenuation_law.json")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    tau_clean = float(np.quantile(clean[lab == 0], 1 - args.far))
    half = len(idx) // 2
    cal, test = slice(0, half), slice(half, len(idx))
    print(f"pairs={len(idx)}  draws={args.draws}  tau_clean={tau_clean:.4f}")
    print(f"fit on {half} calibration pairs, evaluate on {len(idx)-half} held out\n")

    results = {"tau_clean": tau_clean, "n_pairs": len(idx), "draws": args.draws,
               "arch": {}}

    for arch in args.arch:
        print(f"########## {arch.upper()} ##########")
        rows = []
        for sigma in args.sigmas:
            den = load_den(arch, sigma, device)
            cos = draw_cosines(den, backbone, img1, idx, e2, sigma,
                               args.draws, args.batch, device)
            mean_noisy = cos.mean(1)

            # --- fit the affine law on CALIBRATION pairs only ---------------
            A = np.vstack([clean[cal], np.ones(half)]).T
            (a_hat, b_hat), *_ = np.linalg.lstsq(A, mean_noisy[cal], rcond=None)
            pred_cal = A @ np.array([a_hat, b_hat])
            r2_cal = 1 - ((mean_noisy[cal] - pred_cal) ** 2).sum() / \
                ((mean_noisy[cal] - mean_noisy[cal].mean()) ** 2).sum()
            # held-out R^2: does the law PREDICT, not just fit?
            A_t = np.vstack([clean[test], np.ones(len(idx) - half)]).T
            pred_test = A_t @ np.array([a_hat, b_hat])
            r2_test = 1 - ((mean_noisy[test] - pred_test) ** 2).sum() / \
                ((mean_noisy[test] - mean_noisy[test].mean()) ** 2).sum()

            tau_analytic = a_hat * tau_clean + b_hat

            # --- swept optimum, chosen on CALIBRATION ----------------------
            grid = np.arange(0.05, 0.65, 0.0125)
            sweep = [(t, cert_stats(cos[cal], lab[cal], t, sigma)["acc"]) for t in grid]
            tau_swept = max(sweep, key=lambda r: r[1])[0]

            # --- held-out comparison of the three thresholds ---------------
            s_clean = cert_stats(cos[test], lab[test], tau_clean, sigma)
            s_ana = cert_stats(cos[test], lab[test], tau_analytic, sigma)
            s_swp = cert_stats(cos[test], lab[test], tau_swept, sigma)

            gain_swept = s_swp["acc"] - s_clean["acc"]
            gain_ana = s_ana["acc"] - s_clean["acc"]
            captured = (gain_ana / gain_swept * 100) if abs(gain_swept) > 1e-9 else float("nan")

            print(f"sigma={sigma}:  a={a_hat:.4f} b={b_hat:+.4f}  "
                  f"R2_fit={r2_cal:.4f} R2_heldout={r2_test:.4f}")
            print(f"   tau: clean={tau_clean:.4f}  analytic={tau_analytic:.4f}  "
                  f"swept={tau_swept:.4f}")
            print(f"   acc: clean={s_clean['acc']:.4f}  analytic={s_ana['acc']:.4f} "
                  f"({gain_ana:+.4f})  swept={s_swp['acc']:.4f} ({gain_swept:+.4f})")
            print(f"   analytic captures {captured:.0f}% of the swept gain")

            rows.append({"sigma": sigma, "a": float(a_hat), "b": float(b_hat),
                         "r2_fit": float(r2_cal), "r2_heldout": float(r2_test),
                         "tau_analytic": float(tau_analytic),
                         "tau_swept": float(tau_swept),
                         "acc_clean": s_clean["acc"], "acc_analytic": s_ana["acc"],
                         "acc_swept": s_swp["acc"],
                         "gain_analytic": gain_ana, "gain_swept": gain_swept,
                         "captured_pct": captured,
                         "R_clean": s_clean["meanR"], "R_analytic": s_ana["meanR"],
                         "R_swept": s_swp["meanR"]})
            del den
            torch.cuda.empty_cache()
        results["arch"][arch] = rows
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
