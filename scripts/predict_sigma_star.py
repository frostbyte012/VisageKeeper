#!/usr/bin/env python3
"""
scripts/predict_sigma_star.py -- ANALYTIC design-space exploration: predict the
certified accuracy/radius curve over sigma from a two-point calibration, and
select the deployment sigma* without ever sweeping.

The gap this fills (SOTA positioning)
-------------------------------------
Choosing the smoothing level sigma is an open problem. Existing approaches are
empirical: Variational RS trains an auxiliary network to pick sigma per input;
Dual RS restricts to locally-constant variance with confidence corrections;
naive input-dependent sigma is known to be unsound. All of them EVALUATE
candidate sigmas by Monte-Carlo certification -- thousands of GFLOPs per grid
point per dataset.

Our attenuation law makes the sweep unnecessary. The certified radius is
R = sigma * Phi^-1(p_bar), and the law predicts, for a pair with clean cosine c,

    p_bar(sigma, c) ~= Phi( (a(sigma)*c + b(sigma) - tau(sigma)) / s(sigma) )

with a(sigma) in closed form (DnCNN: 1/(1+k*sigma^2); ViT: exp-family), b
near-linear, tau(sigma) = a*tau_clean + b (our analytic threshold), and s the
per-draw cosine std. Everything on the right is measurable from ONE cheap
calibration at TWO sigma points (to pin k and the s trend). The whole
certified-performance surface -- accuracy at any radius, mean radius, the
optimal deployment sigma* -- then comes from a formula.

Soundness: sigma* is fixed at DESIGN time from clean gallery statistics, before
any probe exists. Every certificate is issued by the standard fixed-sigma Cohen
machinery at sigma*. No per-input noise selection, so none of the soundness
issues of input-dependent smoothing arise.

Validation protocol
-------------------
Fit k, b-trend, s-trend from sigma in {0.25, 0.5} ONLY. Predict certified
accuracy and mean radius at {0.05, 0.1, 0.75, 1.0} (never seen). Compare with
the measured 6-sigma sweeps (attenuation_law_*6.json). Then report the
predicted sigma* under two deployment objectives and check it against the
measured optimum.

Run:  python scripts/predict_sigma_star.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper")
SRC = {"vit": "attenuation_law_vit6.json", "dncnn": "attenuation_law_dncnn6.json"}
CAL_SIGMAS = [0.25, 0.5]           # the only points the predictor may see
FORM = {"vit": ("exp", lambda s, k: np.exp(-k * s)),
        "dncnn": ("lorentz2", lambda s, k: 1.0 / (1.0 + k * s ** 2))}
# per-draw cosine std: measured to grow roughly linearly in sigma; fitted from
# the same two calibration points via the p_bar identity (see fit_s below).
RADII_EVAL = [0.25, 0.5]


def fit_k(fn, sig, a):
    grid = np.linspace(0.01, 10.0, 5000)
    return min(grid, key=lambda k: np.sum((fn(np.array(sig), k) - np.array(a)) ** 2))


_COS_CACHE = None


def load_clean_cosines():
    """Empirical clean genuine/impostor cosines (same pairs as the sweeps)."""
    global _COS_CACHE
    if _COS_CACHE is not None:
        return _COS_CACHE
    import argparse
    import torch
    from frpure.models.backbone import build_backbone
    from scripts.run_certify import load_pairs, balanced_subset, embed_all
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = build_backbone("facenet", device=dev, pretrained="vggface2")
    a = argparse.Namespace(synthetic=False, aligned_dir="DATA/lfw_aligned_160",
                           pairs="DATA/pairs.txt", pairs_npz=None, bin_path=None)
    i1, i2, same = load_pairs(a)
    idx = balanced_subset(same, 300)
    e1 = embed_all(bb, i1[idx]); e2 = embed_all(bb, i2[idx])
    lab = same[idx].astype(int)
    c = (e1 * e2).sum(1).numpy()
    _COS_CACHE = (c[lab == 1], c[lab == 0])
    return _COS_CACHE


def main():
    results = {}
    for arch, src in SRC.items():
        p = os.path.join(OUT, src)
        if not os.path.exists(p):
            print(f"[skip] {src}")
            continue
        d = json.load(open(p))
        rows = next(iter(d["arch"].values()))
        tau_clean = d["tau_clean"]
        by_sig = {r["sigma"]: r for r in rows}
        sig_all = sorted(by_sig)

        # ---- calibration: only CAL_SIGMAS visible ---------------------------
        cal = [by_sig[s] for s in CAL_SIGMAS]
        fname, fn = FORM[arch]
        k = fit_k(fn, [r["sigma"] for r in cal], [r["a"] for r in cal])
        # b: linear through the two calibration points
        b_slope = (cal[1]["b"] - cal[0]["b"]) / (CAL_SIGMAS[1] - CAL_SIGMAS[0])
        b_icpt = cal[0]["b"] - b_slope * CAL_SIGMAS[0]
        b_of = lambda s: b_slope * s + b_icpt

        # s(sigma): invert the p_bar identity at the calibration points.
        # acc_analytic at sigma is the fraction of pairs with p_bar>0.5, i.e.
        # the fraction with a*c+b > tau_an -- threshold crossing in c. We fit s
        # from the STEEPNESS of accuracy: use the two calibration accuracies to
        # solve for an effective per-draw std trend s(sigma)=s0+s1*sigma, with
        # the clean-cosine population statistics providing the c distribution.
        # We approximate the genuine/impostor clean cosine populations by their
        # calibration-time summary (mean/std), stored in the 6-sigma files'
        # implied stats; here we reconstruct from acc via a normal quantile
        # matching at the two points, which is enough for a two-parameter trend.
        # For prediction we hold the pair population fixed and vary sigma only.
        # Practical shortcut: at each cal sigma, s_eff solves
        #   acc_meas = P_c[ Phi((a*c+b - tau_an)/s) > 0.5 ] = P_c[a*c+b > tau_an]
        # which is s-independent at threshold 0.5, so instead we match the
        # MEAN RADIUS:  R_mean = E[ sigma*Phi^-1(pbar) | pbar>0.5 ].
        # We calibrate s numerically so predicted mean radius matches at the
        # cal points, then extrapolate s linearly.
        # EMPIRICAL clean-cosine population (the tails carry the hard pairs
        # that determine accuracy; a Gaussian toy saturates acc at ~1).
        c_gen, c_imp = load_clean_cosines()
        c_all = np.concatenate([c_gen, c_imp])

        # Residual scatter around the affine law. With the analytic tau,
        # p_bar>0.5 iff a*c+b+eps > a*tau_clean+b, so WITHOUT eps predicted
        # accuracy would be sigma-independent by construction; the measured
        # decay comes entirely from the residual eps whose variance we can
        # recover from the stored fit R^2:  var_eps = a^2 var(c) (1-R2)/R2.
        var_c = float(np.var(c_all))

        def sres_of(sigma_q, a_q):
            # linear-in-sigma extrapolation of residual std from the cal points
            pts = []
            for s0, r in zip(CAL_SIGMAS, cal):
                r2v = max(min(r["r2_fit"], 0.999), 1e-3)
                pts.append(np.sqrt(r["a"] ** 2 * var_c * (1 - r2v) / r2v))
            slope = (pts[1] - pts[0]) / (CAL_SIGMAS[1] - CAL_SIGMAS[0])
            return max(1e-4, pts[0] + slope * (sigma_q - CAL_SIGMAS[0]))

        rng_eps = np.random.default_rng(1)
        eps_unit = rng_eps.standard_normal(len(c_all))   # fixed draws, scaled per sigma

        def predict(sigma, s_eff):
            a = float(fn(np.array([sigma]), k)[0]); b = b_of(sigma)
            tau_an = a * tau_clean + b
            eps = eps_unit * sres_of(sigma, a)
            mu_g = a * c_gen + b + eps[:len(c_gen)]
            mu_i = a * c_imp + b + eps[len(c_gen):]
            pb_g = norm.cdf((mu_g - tau_an) / s_eff)                  # genuine votes same
            pb_i = norm.cdf((tau_an - mu_i) / s_eff)                  # impostor votes diff
            pb = np.concatenate([pb_g, pb_i])
            ok = pb > 0.5
            acc = ok.mean()
            R = sigma * norm.ppf(np.clip(pb[ok], 1e-6, 1 - 1e-6))
            cert_at = {r: float((pb > norm.cdf(r / sigma)).mean()) for r in RADII_EVAL}
            return float(acc), float(R.mean()) if ok.any() else 0.0, cert_at

        def s_for(sigma, target_R):
            f = lambda s: predict(sigma, s)[1] - target_R
            try:
                return brentq(f, 0.01, 0.5)
            except ValueError:
                return 0.08
        s_pts = [s_for(s, by_sig[s]["R_analytic"]) for s in CAL_SIGMAS]
        s_slope = (s_pts[1] - s_pts[0]) / (CAL_SIGMAS[1] - CAL_SIGMAS[0])
        s_of = lambda s: max(0.02, s_pts[0] + s_slope * (s - CAL_SIGMAS[0]))

        print(f"\n########## {arch.upper()} ##########")
        print(f"calibrated on sigma={CAL_SIGMAS}: k={k:.4f} ({fname}), "
              f"b={b_slope:+.4f}*s{b_icpt:+.4f}, s_eff={s_pts[0]:.4f},{s_pts[1]:.4f}")
        print(f"{'sigma':>6} {'acc_meas':>9} {'acc_pred':>9} {'R_meas':>8} {'R_pred':>8}  seen?")
        rows_out = []
        for s in sig_all:
            acc_p, R_p, cert_at = predict(s, s_of(s))
            r = by_sig[s]
            seen = "cal" if s in CAL_SIGMAS else "PREDICTED"
            print(f"{s:>6} {r['acc_analytic']:>9.4f} {acc_p:>9.4f} "
                  f"{r['R_analytic']:>8.4f} {R_p:>8.4f}  {seen}")
            rows_out.append({"sigma": s, "acc_meas": r["acc_analytic"],
                             "acc_pred": acc_p, "R_meas": r["R_analytic"],
                             "R_pred": R_p, "cert_at_pred": cert_at,
                             "seen": s in CAL_SIGMAS})
        # errors on UNSEEN sigmas only
        uns = [r for r in rows_out if not r["seen"]]
        mae_acc = np.mean([abs(r["acc_pred"] - r["acc_meas"]) for r in uns])
        mae_R = np.mean([abs(r["R_pred"] - r["R_meas"]) for r in uns])
        print(f"unseen-sigma MAE: acc={mae_acc:.4f}  R={mae_R:.4f}")

        # ---- sigma* selection under two objectives -------------------------
        grid = np.linspace(0.1, 1.2, 111)
        curves = [(s, *predict(s, s_of(s))[:2]) for s in grid]
        # objective A: max mean radius s.t. acc >= 0.85
        featA = [(s, R) for s, acc, R in curves if acc >= 0.85]
        star_A = max(featA, key=lambda t: t[1])[0] if featA else None
        # objective B: max cert-acc @ R>=0.5
        featB = [(s, predict(s, s_of(s))[2][0.5]) for s in grid]
        star_B = max(featB, key=lambda t: t[1])[0]
        # measured optima from the 6-sigma sweep
        measA = [(s, by_sig[s]["R_analytic"]) for s in sig_all
                 if by_sig[s]["acc_analytic"] >= 0.85]
        measA_star = max(measA, key=lambda t: t[1])[0] if measA else None
        print(f"sigma* (max R | acc>=0.85): predicted={star_A}  "
              f"measured-grid={measA_star}")
        print(f"sigma* (max certacc@R>=0.5): predicted={star_B:.2f}")
        results[arch] = {"k": float(k), "form": fname,
                         "b": [float(b_slope), float(b_icpt)],
                         "s_eff": [float(x) for x in s_pts],
                         "rows": rows_out,
                         "mae_unseen": {"acc": float(mae_acc), "R": float(mae_R)},
                         "sigma_star_pred_A": star_A,
                         "sigma_star_meas_A": measA_star,
                         "sigma_star_pred_B": float(star_B)}

    path = os.path.join(OUT, "sigma_star_prediction.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
