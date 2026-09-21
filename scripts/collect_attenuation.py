#!/usr/bin/env python3
"""
scripts/collect_attenuation.py -- paper package for the attenuation law +
analytic threshold calibration.

Reads results/paper/attenuation_law_{vit,dncnn}500.json and writes:
  table_attenuation.csv   a(sigma), b(sigma), held-out R^2, per architecture
  table_analytic_tau.csv  certified accuracy at clean / analytic / swept tau
  fig_attenuation.pdf/png three panels:
      (a) a(sigma) decay per architecture  -- retained identity signal
      (b) held-out R^2 per architecture    -- where the affine model is valid
      (c) certified accuracy: clean vs analytic tau
  attenuation_summary.md  headline numbers and the honest caveats

Run:  python scripts/collect_attenuation.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper")
SRC = {"ViT": "attenuation_law_vit500.json",
       "DnCNN": "attenuation_law_dncnn500.json"}


def load():
    data = {}
    for name, fn in SRC.items():
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print(f"[warn] missing {fn}")
            continue
        with open(p) as f:
            d = json.load(f)
        # each file was run with a single --arch, so take whatever key is there
        rows = next(iter(d["arch"].values()))
        data[name] = {"rows": rows, "tau_clean": d["tau_clean"],
                      "n_pairs": d["n_pairs"], "draws": d["draws"]}
    return data


def main():
    data = load()
    if not data:
        print("nothing to collect")
        return
    os.makedirs(OUT, exist_ok=True)

    # ---- table 1: the law --------------------------------------------------
    with open(os.path.join(OUT, "table_attenuation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "sigma", "a", "b", "r2_fit", "r2_heldout"])
        for arch, d in data.items():
            for r in d["rows"]:
                w.writerow([arch, r["sigma"], round(r["a"], 4), round(r["b"], 4),
                            round(r["r2_fit"], 4), round(r["r2_heldout"], 4)])

    # ---- table 2: what the analytic tau buys -------------------------------
    with open(os.path.join(OUT, "table_analytic_tau.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "sigma", "tau_clean", "tau_analytic", "tau_swept",
                    "acc_clean", "acc_analytic", "acc_swept",
                    "gain_analytic", "gain_swept", "captured_pct"])
        for arch, d in data.items():
            for r in d["rows"]:
                w.writerow([arch, r["sigma"], round(d["tau_clean"], 4),
                            round(r["tau_analytic"], 4), round(r["tau_swept"], 4),
                            round(r["acc_clean"], 4), round(r["acc_analytic"], 4),
                            round(r["acc_swept"], 4),
                            round(r["gain_analytic"], 4), round(r["gain_swept"], 4),
                            round(r["captured_pct"], 1)])

    for arch, d in data.items():
        print(f"\n########## {arch}  (n={d['n_pairs']}, draws={d['draws']}) ##########")
        print(f"{'sigma':>6} {'a':>7} {'b':>8} {'R2_out':>7} | "
              f"{'tau_an':>7} {'acc_cl':>7} {'acc_an':>7} {'gain':>7} {'cap%':>6}")
        for r in d["rows"]:
            print(f"{r['sigma']:>6} {r['a']:>7.4f} {r['b']:>+8.4f} "
                  f"{r['r2_heldout']:>7.4f} | {r['tau_analytic']:>7.4f} "
                  f"{r['acc_clean']:>7.4f} {r['acc_analytic']:>7.4f} "
                  f"{r['gain_analytic']:>+7.4f} {r['captured_pct']:>6.0f}")

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {"ViT": "#2a78d6", "DnCNN": "#eb6834"}
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for arch, d in data.items():
        s = [r["sigma"] for r in d["rows"]]
        ax[0].plot(s, [r["a"] for r in d["rows"]], marker="o", lw=2,
                   color=COL.get(arch), label=arch)
        ax[1].plot(s, [r["r2_heldout"] for r in d["rows"]], marker="o", lw=2,
                   color=COL.get(arch), label=arch)
        ax[2].plot(s, [r["acc_clean"] for r in d["rows"]], marker="o", lw=2,
                   ls="--", color=COL.get(arch), alpha=0.6,
                   label=f"{arch}: clean $\\tau$")
        ax[2].plot(s, [r["acc_analytic"] for r in d["rows"]], marker="s", lw=2,
                   color=COL.get(arch), label=f"{arch}: analytic $\\tau$")

    ax[0].set_ylabel(r"attenuation $a(\sigma)$")
    ax[0].set_title("Retained identity signal", fontsize=10)
    ax[1].set_ylabel(r"held-out $R^2$")
    ax[1].set_title("Validity of the affine model", fontsize=10)
    ax[1].axhline(0.8, color="#888", ls=":", lw=1)
    ax[2].set_ylabel("certified accuracy")
    ax[2].set_title(r"Clean-FAR $\tau$ vs analytic $\tau$", fontsize=10)
    for a_ in ax:
        a_.set_xlabel(r"noise level $\sigma$")
        a_.grid(True, lw=0.4, alpha=0.4)
        a_.spines[["top", "right"]].set_visible(False)
        a_.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_attenuation.{ext}"), dpi=200)

    # ---- summary -----------------------------------------------------------
    md = ["# Cosine attenuation under denoised smoothing", "",
          "Gaussian smoothing shrinks genuine-pair cosine similarity by a",
          "predictable factor while leaving impostor pairs nearly unchanged.",
          "The induced map is approximately affine:", "",
          "    E[cos_noisy] ~= a(sigma) * cos_clean + b(sigma)", "",
          "so the verification threshold follows analytically:", "",
          "    tau(sigma) = a(sigma) * tau_clean + b(sigma)", ""]
    for arch, d in data.items():
        md += [f"## {arch}", "",
               "| sigma | a | b | held-out R2 | tau_analytic | acc (clean tau) | acc (analytic tau) | gain |",
               "|---|---|---|---|---|---|---|---|"]
        for r in d["rows"]:
            md.append(f"| {r['sigma']} | {r['a']:.4f} | {r['b']:+.4f} | "
                      f"{r['r2_heldout']:.4f} | {r['tau_analytic']:.4f} | "
                      f"{r['acc_clean']:.4f} | {r['acc_analytic']:.4f} | "
                      f"{r['gain_analytic']:+.4f} |")
        md.append("")
    md += ["## Caveats", "",
           "* The affine approximation degrades as sigma grows (held-out R^2 falls);",
           "  it should be reported as a good model for moderate noise and an",
           "  approximation at the high-noise end, where the correction is largest.",
           "* At sigma=0.25 the clean-FAR threshold is already close to optimal, so",
           "  the correction is negligible or slightly negative -- this bounds the claim.",
           "* Single backbone (facenet) and dataset (LFW); the mechanism should",
           "  generalise but that is not demonstrated here."]
    with open(os.path.join(OUT, "attenuation_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nwrote table_attenuation.csv, table_analytic_tau.csv, "
          f"fig_attenuation.{{pdf,png}}, attenuation_summary.md -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
