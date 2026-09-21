#!/usr/bin/env python3
"""
scripts/cross_dataset_law.py -- does the attenuation law GENERALISE?

On LFW we found E[cos_noisy] ~= a(sigma)*cos_clean + b(sigma), with
    ViT    a(sigma) = exp(-k*sigma)        k ~ 0.61
    DnCNN  a(sigma) = 1/(1 + k*sigma^2)    k ~ 1.91
(both selected by leave-one-sigma-out AND AIC).

The claim only matters if it is a property of denoised smoothing rather than of
one benchmark. This script re-runs the model selection on CFP-FP (frontal-
profile) and AgeDB-30 (large age gap) and asks two separate questions:

  1. Does the FUNCTIONAL FORM survive? (same winner per architecture)
     -> if yes, the law generalises and only the constant is dataset-specific.
  2. Does the CONSTANT k change, and in an interpretable direction?
     -> harder benchmarks should attenuate faster (larger k).

A form that survives with a shifted constant is a much stronger result than one
tuned per dataset: it means a single calibration pass on a new benchmark gives
tau at every sigma.

Run:  python scripts/cross_dataset_law.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper")

DATASETS = {
    "LFW":      {"ViT": "attenuation_law_vit6.json",
                 "DnCNN": "attenuation_law_dncnn6.json"},
    "CFP-FP":   {"both": "attenuation_cfpfp.json"},
    "AgeDB-30": {"both": "attenuation_agedb.json"},
}

FORMS = {
    "exp":      lambda s, k: np.exp(-k * s),
    "exp2":     lambda s, k: np.exp(-k * s ** 2),
    "lorentz":  lambda s, k: 1.0 / (1.0 + k * s),
    "lorentz2": lambda s, k: 1.0 / (1.0 + k * s ** 2),
}


def power(s, k, p):
    return (1.0 + k * s) ** (-p)


def fit_1p(fn, sig, a, grid=np.linspace(0.01, 10.0, 4000)):
    return min(((np.sum((fn(sig, k) - a) ** 2), k) for k in grid))[1]


def fit_2p(sig, a):
    best = None
    for k, p in itertools.product(np.linspace(0.05, 8, 200),
                                  np.linspace(0.2, 5, 120)):
        e = np.sum((power(sig, k, p) - a) ** 2)
        if best is None or e < best[0]:
            best = (e, k, p)
    return best[1], best[2]


def analyse(sig, a):
    """Return per-form full-fit and leave-one-out metrics + the two winners."""
    n = len(sig)
    full, loo = {}, {}
    for fname, fn in FORMS.items():
        k = fit_1p(fn, sig, a)
        pred = fn(sig, k)
        rmse = float(np.sqrt(((a - pred) ** 2).mean()))
        r2 = float(1 - ((a - pred) ** 2).sum() / ((a - a.mean()) ** 2).sum())
        full[fname] = {"k": float(k), "rmse": rmse, "r2": r2,
                       "aic": float(n * np.log(max(rmse ** 2, 1e-12)) + 2)}
        errs = []
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            kk = fit_1p(fn, sig[m], a[m])
            errs.append(abs(float(fn(sig[i], kk)) - a[i]))
        loo[fname] = float(np.mean(errs))
    k2, p2 = fit_2p(sig, a)
    pred = power(sig, k2, p2)
    rmse = float(np.sqrt(((a - pred) ** 2).mean()))
    full["power"] = {"k": float(k2), "p": float(p2), "rmse": rmse,
                     "r2": float(1 - ((a - pred) ** 2).sum() / ((a - a.mean()) ** 2).sum()),
                     "aic": float(n * np.log(max(rmse ** 2, 1e-12)) + 4)}
    errs = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        kk, pp = fit_2p(sig[m], a[m])
        errs.append(abs(float(power(sig[i], kk, pp)) - a[i]))
    loo["power"] = float(np.mean(errs))
    return full, loo, min(loo, key=loo.get), min(full, key=lambda f: full[f]["aic"])


def load_rows():
    """-> {(dataset, arch): [rows]}"""
    out = {}
    for ds, spec in DATASETS.items():
        for key, fn in spec.items():
            p = os.path.join(OUT, fn)
            if not os.path.exists(p):
                print(f"[warn] missing {fn} ({ds})")
                continue
            with open(p) as f:
                d = json.load(f)
            for arch, rows in d["arch"].items():
                label = {"vit": "ViT", "dncnn": "DnCNN"}.get(arch, arch)
                if key != "both" and label != key:
                    continue
                out[(ds, label)] = rows
    return out


def main():
    rows = load_rows()
    if not rows:
        print("no data")
        return

    results = {}
    print(f"{'dataset':>10} {'arch':>6} {'best(LOO)':>10} {'best(AIC)':>10} "
          f"{'k':>8} {'R2':>8} {'LOOerr':>8}  agree")
    for (ds, arch), rr in sorted(rows.items()):
        sig = np.array([r["sigma"] for r in rr])
        a = np.array([r["a"] for r in rr])
        if len(sig) < 4:
            print(f"{ds:>10} {arch:>6}  (only {len(sig)} sigmas -- skipped)")
            continue
        full, loo, b_loo, b_aic = analyse(sig, a)
        agree = "YES" if b_loo == b_aic else "no"
        results[f"{ds}|{arch}"] = {
            "sigmas": sig.tolist(), "a": a.tolist(),
            "best_loo": b_loo, "best_aic": b_aic, "agree": b_loo == b_aic,
            "full": full, "loo": loo}
        print(f"{ds:>10} {arch:>6} {b_loo:>10} {b_aic:>10} "
              f"{full[b_loo].get('k', float('nan')):>8.4f} "
              f"{full[b_loo]['r2']:>8.4f} {loo[b_loo]:>8.4f}  {agree}")

    # ---- the generalisation verdict ---------------------------------------
    print("\n=== does the FORM survive across datasets? ===")
    for arch in ("ViT", "DnCNN"):
        forms = {ds: r["best_loo"] for (ds, a_), r in
                 ((k.split("|"), v) for k, v in results.items()) if a_ == arch}
        if not forms:
            continue
        uniq = set(forms.values())
        ks = {ds: results[f"{ds}|{arch}"]["full"][results[f"{ds}|{arch}"]["best_loo"]].get("k")
              for ds in forms}
        print(f"  {arch}: " + ", ".join(f"{d}={f}" for d, f in forms.items()))
        print(f"        k: " + ", ".join(f"{d}={v:.3f}" for d, v in ks.items()))
        if len(uniq) == 1:
            print(f"        -> FORM GENERALISES ({uniq.pop()}); only k is dataset-specific")
        else:
            print(f"        -> form NOT stable across datasets; report per-dataset")

    path = os.path.join(OUT, "cross_dataset_law.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
