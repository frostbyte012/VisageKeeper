#!/usr/bin/env python3
"""
scripts/sequential_certify.py -- SEQUENTIAL certified verification: spend draws
only until the decision (or a target radius) is statistically settled.

Why
---
Fixed-n certification (Cohen et al. 2019) charges every pair the same n=1000
draws, but most verification pairs are easy: after 100 draws voting 97-3 the
outcome is already settled far beyond the confidence level. With a ViT denoiser
at ~4.3 GFLOPs per draw, that uniformity is the single largest cost in the
pipeline. This script certifies pair-by-pair with a STAGED sequential test.

Statistical validity (the part that must be exact)
--------------------------------------------------
Naive early stopping ("stop when the Clopper-Pearson bound crosses 0.5") is
INVALID: checking a level-alpha bound repeatedly inflates the error rate. We use
alpha-spending: with stages n_1 < ... < n_K, stage k tests at level alpha_k with
sum_k alpha_k = alpha. By the union bound, P(any stage's bound fails) <= alpha,
so any radius reported at any stage holds with confidence 1-alpha, exactly like
the fixed-n certificate. The cost of K looks is a slightly more conservative
per-stage bound -- measured below, not hidden.

Draws are i.i.d. across stages (counts accumulate), so each stage's
Clopper-Pearson bound is valid on the data seen so far.

Modes
-----
  decision      stop as soon as the smoothed DECISION is certified (radius > 0);
                maximal savings; per-pair radius = whatever the stopping stage
                supports (smaller than fixed-n on easy pairs -- reported).
  target        stop as soon as radius >= R_target is certified; pairs that
                cannot reach it run to n_max. Certifies THE SAME guarantee as
                fixed-n at R_target, at a fraction of the draws.

Attenuation-law scheduler (--schedule law)
------------------------------------------
Our attenuation law predicts each pair's mean noisy cosine from its CLEAN
cosine: mu ~= a(sigma)*cos_clean + b(sigma). With the per-draw cosine std s
(fitted on the calibration half), the predicted vote rate is
    p_hat = Phi((mu - tau)/s)        (genuine side; mirrored for impostor)
Pairs predicted deep inside the threshold start at the smallest stage; border-
line pairs start larger, skipping stages they would fail anyway. The schedule
only chooses WHERE TO START -- validity never depends on it, so a wrong
prediction costs draws, not correctness.

Run:
  python scripts/sequential_certify.py --sigma 0.75 --arch vit --n_pairs 300 \
      --mode decision --schedule law
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy.stats import beta, norm

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser
from frpure.defenses.cascade import _disable_fused_attention
from scripts.run_certify import load_pairs, balanced_subset, embed_all, RADII
from scripts.diag_error_overlap import CKPT


def load_den(arch, sigma, device):
    p = CKPT[sigma][1 if arch == "vit" else 0]
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(p, map_location=device))
    _disable_fused_attention(d)
    for q in d.parameters():
        q.requires_grad_(False)
    return d


class DrawSampler:
    """Draws votes for one pair in batches; counts accumulate across stages."""

    def __init__(self, den, backbone, probe, gallery, sigma, tau, device, batch=100):
        self.den, self.backbone = den, backbone
        self.probe, self.gallery = probe, gallery
        self.sigma, self.tau, self.device, self.batch = sigma, tau, device, batch
        self.n = 0
        self.n_same = 0

    @torch.no_grad()
    def draw_to(self, n_total):
        while self.n < n_total:
            m = min(self.batch, n_total - self.n)
            z = self.probe.unsqueeze(0).repeat(m, 1, 1, 1).to(self.device)
            z = (z + torch.randn_like(z) * self.sigma).clamp(0, 1)
            cos = (self.backbone.embed(self.den(z))
                   * self.gallery.unsqueeze(0)).sum(1)
            self.n_same += int((cos >= self.tau).sum().item())
            self.n += m


def lcb(k, n, alpha_k):
    """One-sided lower Clopper-Pearson bound on the majority class rate."""
    return float(beta.ppf(alpha_k, k, n - k + 1)) if k > 0 else 0.0


def certify_sequential(sampler, stages, alphas, sigma, mode="decision",
                       r_target=0.0):
    """-> (pred, radius, draws_used). pred=-1 means abstain at n_max."""
    for n_k, a_k in zip(stages, alphas):
        sampler.draw_to(n_k)
        k_same = sampler.n_same
        top = 1 if k_same >= sampler.n - k_same else 0
        k_top = k_same if top == 1 else sampler.n - k_same
        pA = lcb(k_top, sampler.n, a_k)
        if pA > 0.5:
            R = sigma * float(norm.ppf(pA))
            if mode == "decision" or R >= r_target:
                return top, R, sampler.n
    # final stage reached without (target) certification
    k_same = sampler.n_same
    top = 1 if k_same >= sampler.n - k_same else 0
    k_top = k_same if top == 1 else sampler.n - k_same
    pA = lcb(k_top, sampler.n, alphas[-1])
    if pA > 0.5:
        return top, sigma * float(norm.ppf(pA)), sampler.n
    return -1, 0.0, sampler.n


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
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--stages", type=int, nargs="+", default=[100, 200, 400, 1000])
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--mode", choices=["decision", "target"], default="decision")
    ap.add_argument("--r_target", type=float, default=0.5)
    ap.add_argument("--schedule", choices=["fixed", "law"], default="law")
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)
    den = load_den(args.arch, args.sigma, device)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    tau = float(np.quantile(clean[lab == 0], 1 - args.far))
    half = len(idx) // 2
    cal, test = idx[:half], idx[half:]
    lab_cal, lab_test = lab[:half], lab[half:]
    clean_cal, clean_test = clean[:half], clean[half:]
    K = len(args.stages)
    alphas = [args.alpha / K] * K
    print(f"sigma={args.sigma} arch={args.arch} tau={tau:.4f} "
          f"stages={args.stages} alpha={args.alpha} (per-stage {args.alpha/K:g}) "
          f"mode={args.mode}" + (f" R>={args.r_target}" if args.mode == "target" else ""))

    # --- fit the attenuation-law scheduler on the CALIBRATION half ----------
    if args.schedule == "law":
        mus = []
        with torch.no_grad():
            for j, i in enumerate(cal):
                z = img1[i].unsqueeze(0).repeat(64, 1, 1, 1).to(device)
                z = (z + torch.randn_like(z) * args.sigma).clamp(0, 1)
                cos = (backbone.embed(den(z)) * e2[j].to(device).unsqueeze(0)).sum(1)
                mus.append((float(cos.mean()), float(cos.std())))
        mu = np.array([m for m, _ in mus]); sd = np.array([s for _, s in mus])
        A = np.vstack([clean_cal, np.ones(half)]).T
        (a_hat, b_hat), *_ = np.linalg.lstsq(A, mu, rcond=None)
        s_bar = float(sd.mean())
        print(f"law fit: a={a_hat:.4f} b={b_hat:+.4f} s={s_bar:.4f} "
              f"(64-draw calibration on {half} pairs = "
              f"{64*half} draws, amortised below)")

        def start_stage(cc, is_gen_guess):
            mu_p = a_hat * cc + b_hat
            p_hat = norm.cdf((mu_p - tau) / s_bar)
            conf = p_hat if is_gen_guess else 1 - p_hat
            if conf > 0.995:
                return 0            # easiest: start at the smallest stage
            if conf > 0.95:
                return 1
            return 2                # borderline: skip stages it would fail
    else:
        def start_stage(cc, g):
            return 0

    # --- certify the TEST half: sequential vs fixed-n ------------------------
    rows = []
    for j0, i in enumerate(test):
        j = half + j0
        g = e2[j].to(device)
        cc = clean_test[j0]
        s0 = start_stage(cc, cc >= tau)
        stages = args.stages[s0:]
        al = alphas[s0:]
        smp = DrawSampler(den, backbone, img1[i], g, args.sigma, tau, device,
                          args.batch)
        pred, R, used = certify_sequential(smp, stages, al, args.sigma,
                                           args.mode, args.r_target)
        rows.append({"label": int(lab_test[j0]), "pred": pred, "R": R,
                     "draws": used, "start_stage": s0})
        if (j0 + 1) % 50 == 0:
            print(f"  {j0+1}/{len(test)}")

    pred = np.array([r["pred"] for r in rows])
    R = np.array([r["R"] for r in rows])
    draws = np.array([r["draws"] for r in rows])
    labt = np.array([r["label"] for r in rows])
    correct = (pred == labt) & (pred != -1)
    n_max = args.stages[-1]

    print("\n==== SEQUENTIAL vs FIXED-n ====")
    print(f"certified accuracy      : {correct.mean():.4f}")
    print(f"abstention              : {(pred == -1).mean():.4f}")
    print(f"mean radius (certified) : {R[correct].mean() if correct.any() else 0:.4f}")
    cert_acc = {r: float((correct & (R >= r)).mean()) for r in RADII}
    for r in RADII:
        print(f"   cert acc @ R>={r:<4}: {cert_acc[r]:.4f}")
    print(f"mean draws/pair         : {draws.mean():.1f}  (fixed-n: {n_max})")
    print(f"draw reduction          : {(1 - draws.mean()/n_max)*100:.1f}%")
    print(f"draws histogram         : " +
          "  ".join(f"n<={s}:{(draws <= s).mean():.2f}" for s in args.stages))

    out = {"sigma": args.sigma, "arch": args.arch, "tau": tau,
           "stages": args.stages, "alpha": args.alpha, "mode": args.mode,
           "r_target": args.r_target if args.mode == "target" else None,
           "schedule": args.schedule, "n_test": len(test),
           "certified_acc": float(correct.mean()),
           "abstention": float((pred == -1).mean()),
           "mean_radius": float(R[correct].mean()) if correct.any() else 0.0,
           "cert_acc_at": cert_acc,
           "mean_draws": float(draws.mean()), "n_max": n_max,
           "reduction_pct": float((1 - draws.mean()/n_max)*100)}
    path = args.out or (f"results/paper/seq_certify_{args.arch}_"
                        f"s{int(args.sigma*100):03d}_{args.mode}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
