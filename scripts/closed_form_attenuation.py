#!/usr/bin/env python3
"""
scripts/closed_form_attenuation.py -- can a(sigma) be PREDICTED instead of fitted?

Where this sits
---------------
attenuation_law.py established that smoothing maps clean cosine to noisy cosine
approximately affinely, E[cos_noisy] ~= a(sigma)*cos_clean + b(sigma), and that
setting tau(sigma) = a(sigma)*tau_clean + b(sigma) recovers up to +20.8 points of
certified accuracy. But a(sigma) is currently FITTED separately at every noise
level, which means every new sigma costs a calibration pass.

If a(sigma) follows a simple functional form, one or two constants replace the
whole table and tau becomes available at ANY sigma -- including noise levels
never measured. That is the difference between an empirical curve and a law.

Candidates (all monotone decreasing, a(0)=1):
    exp        a = exp(-k*sigma)
    exp2       a = exp(-k*sigma^2)
    lorentz    a = 1/(1 + k*sigma)
    lorentz2   a = 1/(1 + k*sigma^2)            <- signal-to-noise intuition
    power      a = (1 + k*sigma)^(-p)           (2 params)

Validation is LEAVE-ONE-SIGMA-OUT: fit the constant on three noise levels and
predict the fourth. A form that only interpolates its own fit proves nothing;
we need it to predict a sigma it never saw, and then we check that the tau it
implies still delivers the certified-accuracy gain.

Run:
  python scripts/closed_form_attenuation.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper")
# prefer the wider 6-sigma sweep when present: discriminating between candidate
# functional forms needs points at BOTH ends, especially near sigma=0 where the
# exponential and Lorentzian shapes differ most.
SRC = {"ViT": ["attenuation_law_vit6.json", "attenuation_law_vit500.json"],
       "DnCNN": ["attenuation_law_dncnn6.json", "attenuation_law_dncnn500.json"]}

FORMS = {
    "exp":      (lambda s, k: np.exp(-k * s), [1.0]),
    "exp2":     (lambda s, k: np.exp(-k * s ** 2), [1.0]),
    "lorentz":  (lambda s, k: 1.0 / (1.0 + k * s), [1.0]),
    "lorentz2": (lambda s, k: 1.0 / (1.0 + k * s ** 2), [1.0]),
}
# two-parameter form handled separately
def power(s, k, p):
    return (1.0 + k * s) ** (-p)


def fit_1p(fn, sig, a, grid=np.linspace(0.01, 6.0, 2000)):
    err = [(np.sum((fn(sig, k) - a) ** 2), k) for k in grid]
    return min(err)[1]


def fit_2p(sig, a, kg=np.linspace(0.05, 6.0, 200), pg=np.linspace(0.2, 4.0, 120)):
    best = None
    for k, p in itertools.product(kg, pg):
        e = np.sum((power(sig, k, p) - a) ** 2)
        if best is None or e < best[0]:
            best = (e, k, p)
    return best[1], best[2]


def main():
    data = {}
    for name, candidates in SRC.items():
        p = next((os.path.join(OUT, c) for c in candidates
                  if os.path.exists(os.path.join(OUT, c))), None)
        if p is None:
            print(f"[warn] no source file for {name}")
            continue
        print(f"[{name}] using {os.path.basename(p)}")
        with open(p) as f:
            d = json.load(f)
        rows = next(iter(d["arch"].values()))
        data[name] = {"sigma": np.array([r["sigma"] for r in rows]),
                      "a": np.array([r["a"] for r in rows]),
                      "b": np.array([r["b"] for r in rows]),
                      "rows": rows, "tau_clean": d["tau_clean"]}

    results = {}
    for arch, d in data.items():
        sig, a = d["sigma"], d["a"]
        print(f"\n########## {arch} ##########")
        print(f"  measured a(sigma): " +
              "  ".join(f"{s}:{v:.4f}" for s, v in zip(sig, a)))

        # ---- full fit, for reference ----------------------------------------
        print("\n  full fit (all sigmas):")
        full = {}
        for fname, (fn, _) in FORMS.items():
            k = fit_1p(fn, sig, a)
            pred = fn(sig, k)
            r2 = 1 - ((a - pred) ** 2).sum() / ((a - a.mean()) ** 2).sum()
            rmse = float(np.sqrt(((a - pred) ** 2).mean()))
            full[fname] = {"k": float(k), "r2": float(r2), "rmse": rmse}
            print(f"    {fname:>9}: k={k:.4f}  R2={r2:+.4f}  rmse={rmse:.4f}")
        k2, p2 = fit_2p(sig, a)
        pred = power(sig, k2, p2)
        r2 = 1 - ((a - pred) ** 2).sum() / ((a - a.mean()) ** 2).sum()
        full["power"] = {"k": float(k2), "p": float(p2), "r2": float(r2),
                         "rmse": float(np.sqrt(((a - pred) ** 2).mean()))}
        print(f"    {'power':>9}: k={k2:.4f} p={p2:.4f}  R2={r2:+.4f}  "
              f"rmse={full['power']['rmse']:.4f}")

        # ---- leave-one-sigma-out: does it PREDICT an unseen noise level? ----
        print("\n  leave-one-sigma-out prediction error:")
        loo = {}
        for fname, (fn, _) in FORMS.items():
            errs, preds = [], {}
            for i in range(len(sig)):
                m = np.ones(len(sig), bool); m[i] = False
                k = fit_1p(fn, sig[m], a[m])
                ph = float(fn(sig[i], k))
                preds[float(sig[i])] = ph
                errs.append(abs(ph - a[i]))
            loo[fname] = {"mean_abs_err": float(np.mean(errs)),
                          "max_abs_err": float(np.max(errs)), "preds": preds}
            print(f"    {fname:>9}: mean|err|={np.mean(errs):.4f}  "
                  f"max|err|={np.max(errs):.4f}   " +
                  " ".join(f"s{s}:{v:.3f}" for s, v in preds.items()))
        errs, preds = [], {}
        for i in range(len(sig)):
            m = np.ones(len(sig), bool); m[i] = False
            k, p = fit_2p(sig[m], a[m])
            ph = float(power(sig[i], k, p))
            preds[float(sig[i])] = ph
            errs.append(abs(ph - a[i]))
        loo["power"] = {"mean_abs_err": float(np.mean(errs)),
                        "max_abs_err": float(np.max(errs)), "preds": preds}
        print(f"    {'power':>9}: mean|err|={np.mean(errs):.4f}  "
              f"max|err|={np.max(errs):.4f}   " +
              " ".join(f"s{s}:{v:.3f}" for s, v in preds.items()))

        best = min(loo, key=lambda f: loo[f]["mean_abs_err"])
        print(f"\n  BEST predictive form: {best} "
              f"(mean|err|={loo[best]['mean_abs_err']:.4f})")

        # Model selection with a parameter penalty: leave-one-out error already
        # guards against overfitting, but `power` has 2 free constants vs 1 for
        # the others, so we also report AIC on the full fit to make the
        # comparison explicit rather than letting the extra parameter win by
        # default.
        n = len(sig)
        print("  AIC (full fit, lower is better; k = #params):")
        for fname in list(FORMS) + ["power"]:
            rmse = full[fname]["rmse"]
            npar = 2 if fname == "power" else 1
            aic = n * np.log(max(rmse ** 2, 1e-12)) + 2 * npar
            full[fname]["aic"] = float(aic)
            print(f"    {fname:>9}: AIC={aic:+8.2f}  (k={npar})")
        best_aic = min(list(FORMS) + ["power"], key=lambda f: full[f]["aic"])
        print(f"  best by AIC: {best_aic}")
        agree = "AGREE" if best_aic == best else "DISAGREE"
        print(f"  LOO vs AIC: {agree}"
              + ("" if agree == "AGREE" else
                 "  -- form choice is not robust; report as an observation"))

        # ---- does the PREDICTED a still give a good tau? --------------------
        # b(sigma) is small and near-linear; fit it linearly for completeness.
        bk = np.polyfit(sig, d["b"], 1)
        print(f"  b(sigma) ~ {bk[0]:+.4f}*sigma {bk[1]:+.4f}")
        print("\n  tau from PREDICTED a (leave-one-out) vs fitted a:")
        tau_rows = []
        for r in d["rows"]:
            s = r["sigma"]
            a_pred = loo[best]["preds"][s]
            b_pred = float(np.polyval(bk, s))
            tau_pred = a_pred * d["tau_clean"] + b_pred
            tau_rows.append({"sigma": s, "tau_fitted": r["tau_analytic"],
                             "tau_predicted": tau_pred,
                             "d_tau": tau_pred - r["tau_analytic"],
                             "a_true": r["a"], "a_pred": a_pred})
            print(f"    sigma={s}: a_true={r['a']:.4f} a_pred={a_pred:.4f}  "
                  f"tau_fit={r['tau_analytic']:.4f} tau_pred={tau_pred:.4f} "
                  f"(d={tau_pred - r['tau_analytic']:+.4f})")

        results[arch] = {"full_fit": full, "loo": loo, "best_form": best,
                         "b_linear": [float(bk[0]), float(bk[1])],
                         "tau_rows": tau_rows}

    path = os.path.join(OUT, "closed_form_attenuation.json")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
