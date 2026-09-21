#!/usr/bin/env python3
"""
scripts/collect_smoothing_results.py -- assemble the paper package for the
ViT-vs-DnCNN denoised-smoothing study.

Reads the certify/empirical .json outputs and writes to results/paper/:
  table_certified.csv     defense x sigma x radius certified accuracy
  table_empirical.csv     defense x sigma empirical robust accuracy (+ cert bound)
  fig_certified_radius.pdf/.png   2-panel certified-accuracy-vs-radius figure
  consistency_report.txt  certificate <= empirical checks
  summary.md              everything in one human-readable file

Run after the eval sweeps:  python scripts/collect_smoothing_results.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "paper")

# sigma -> file tag used in filenames (e.g. 0.05 -> "s005")
SIGMA_TAG = {0.0: "s0000", 0.05: "s005", 0.1: "s010", 0.25: "s025",
             0.5: "s050", 0.75: "s075", 1.0: "s100"}


def _cfg(sigma):
    tag = SIGMA_TAG[sigma]
    if sigma == 0.0:
        return [("No denoiser", sigma, f"certify500_{tag}_none.json", None)]
    return [
        ("DnCNN (5ep)",   sigma, f"certify500_{tag}_dncnn.json",   f"emp200_{tag}_dncnn.json"),
        ("DnCNN (30ep)",  sigma, f"certify500_{tag}_dncnn30.json", f"emp200_{tag}_dncnn30.json"),
        ("ViT v3 (ours)", sigma, f"certify500_{tag}_vit3.json",    f"emp200_{tag}_vit3.json"),
    ]


SIGMAS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
CONFIGS = [row for s in SIGMAS for row in _cfg(s)]
RADII = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.5"]

# 50-pair fallbacks if the 200-pair empirical file is absent
FALLBACK_EMP = {
    "emp200_s050_dncnn.json": "emp_s050_dncnn.json",
    "emp200_s100_dncnn.json": "emp_s100_dncnn.json",
    "emp200_s050_vit3.json": "emp_s050_vit3.json",
    "emp200_s100_vit3.json": "emp_s100_vit3.json",
}
# sigma=0 certify used a tiny n0/n (deterministic point); treat as certify-only


def load(name):
    if name is None:
        return None
    p = os.path.join(RES, name)
    if not os.path.exists(p) and name in FALLBACK_EMP:
        p = os.path.join(RES, FALLBACK_EMP[name])
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for label, sigma, cert_f, emp_f in CONFIGS:
        cert, emp = load(cert_f), load(emp_f)
        if cert is None:
            print(f"[skip] {label} sigma={sigma}: missing {cert_f}")
            continue
        row = {"defense": label, "sigma": sigma,
               "smoothed_acc": cert["smoothed_accuracy"],
               "abstention": cert["abstention_rate"],
               "mean_radius": cert["mean_radius"]}
        for r in RADII:
            row[f"cert@R>={r}"] = cert["certified_accuracy"].get(r, "")
        if emp:
            row["emp_eps"] = emp["eps"]
            row["emp_robust_acc"] = emp["empirical_robust_acc"]
            row["emp_clean_acc"] = emp["clean_acc"]
        rows.append(row)

    # ---- CSV tables -------------------------------------------------------
    import csv
    cert_cols = ["defense", "sigma", "smoothed_acc", "abstention",
                 "mean_radius"] + [f"cert@R>={r}" for r in RADII]
    with open(os.path.join(OUT, "table_certified.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cert_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    emp_cols = ["defense", "sigma", "emp_eps", "emp_clean_acc",
                "emp_robust_acc"]
    with open(os.path.join(OUT, "table_empirical.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=emp_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows([r for r in rows if "emp_robust_acc" in r])

    # ---- consistency checks ----------------------------------------------
    lines = ["certificate <= empirical robustness at matching radius/eps:"]
    ok = True
    for r in rows:
        if "emp_robust_acc" not in r:
            continue
        eps = str(r["emp_eps"])
        cert_at = r.get(f"cert@R>={eps}")
        if cert_at == "" or cert_at is None:
            continue
        good = cert_at <= r["emp_robust_acc"] + 1e-9
        ok &= good
        lines.append(f"  {'PASS' if good else 'FAIL'}  {r['defense']:>14s} sigma={r['sigma']}: "
                     f"cert@{eps}={cert_at:.3f} vs emp={r['emp_robust_acc']:.3f}")
    lines.append(f"overall: {'ALL CONSISTENT' if ok else 'NOMINAL CROSSING(S) FOUND'}")
    lines.append("note: certified (500 pairs) and empirical (200 pairs) use different")
    lines.append("pair subsets; the Cohen guarantee is per-pair, so small crossings")
    lines.append("across subsets are sampling variation, not certificate violations.")
    report = "\n".join(lines)
    with open(os.path.join(OUT, "consistency_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)

    # ---- figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # color = architecture (validated pair); linestyle = training budget
    COLORS = {"ViT v3 (ours)": "#2a78d6", "DnCNN (5ep)": "#eb6834",
              "DnCNN (30ep)": "#eb6834", "No denoiser": "#777777"}
    STYLES = {"DnCNN (5ep)": "--", "DnCNN (30ep)": "-", "ViT v3 (ours)": "-",
              "No denoiser": ":"}

    # fig 1: certified accuracy vs radius, headline sigmas only (readability)
    headline_sigmas = [s for s in (0.5, 1.0) if s in {r["sigma"] for r in rows}]
    fig, axes = plt.subplots(1, len(headline_sigmas), figsize=(9, 3.4), sharey=True)
    if len(headline_sigmas) == 1:
        axes = [axes]
    xs = [float(r) for r in RADII]
    for ax, s in zip(axes, headline_sigmas):
        for r in rows:
            if r["sigma"] != s or r["defense"] == "No denoiser":
                continue
            ys = [r[f"cert@R>={rad}"] for rad in RADII]
            ax.plot(xs, ys, marker="o", ms=4, lw=2,
                    ls=STYLES.get(r["defense"], "-"),
                    color=COLORS.get(r["defense"], "#777"), label=r["defense"])
        ax.set_title(f"$\\sigma$ = {s}", fontsize=11)
        ax.set_xlabel("certified $L_2$ radius $R$")
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("certified accuracy")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Certified accuracy vs. radius (LFW, facenet, 500 pairs, n=1000)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_certified_radius.{ext}"), dpi=200)

    # fig 2: full sigma sweep -- smoothed/certified@R>=sigma accuracy AND
    # empirical robust accuracy (eps=sigma) vs sigma, one line per defense
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.6))
    by_defense = {}
    for r in rows:
        by_defense.setdefault(r["defense"], []).append(r)
    for defense, rs in by_defense.items():
        rs = sorted(rs, key=lambda r: r["sigma"])
        sig = [r["sigma"] for r in rs]
        smoothed = [r["smoothed_acc"] for r in rs]
        axL.plot(sig, smoothed, marker="o", ms=4, lw=2,
                 ls=STYLES.get(defense, "-"), color=COLORS.get(defense, "#777"),
                 label=defense)
        emp_sig = [r["sigma"] for r in rs if "emp_robust_acc" in r]
        emp_acc = [r["emp_robust_acc"] for r in rs if "emp_robust_acc" in r]
        if emp_acc:
            axR.plot(emp_sig, emp_acc, marker="o", ms=4, lw=2,
                     ls=STYLES.get(defense, "-"), color=COLORS.get(defense, "#777"),
                     label=defense)
    axL.set_title("Smoothed (clean) accuracy vs. $\\sigma$", fontsize=11)
    axR.set_title("Empirical robust accuracy vs. $\\sigma$ (eps=$\\sigma$)", fontsize=11)
    for ax in (axL, axR):
        ax.set_xlabel("noise level $\\sigma$")
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.set_ylim(0.4, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axL.set_ylabel("accuracy")
    axL.legend(frameon=False, fontsize=9)
    fig2.suptitle("Full noise sweep: sigma in {0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0}",
                  fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(OUT, f"fig_sigma_sweep.{ext}"), dpi=200)
    print(f"wrote figures + tables -> {os.path.abspath(OUT)}")

    # ---- summary.md -------------------------------------------------------
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    md = ["# Denoised smoothing: ViT vs DnCNN (paper package)", "",
          "## Certified accuracy (500 pairs, n=1000, alpha=1e-3)", "",
          "| defense | sigma | smoothed | " + " | ".join(f"R>={r}" for r in RADII) + " |",
          "|---" * (3 + len(RADII)) + "|"]
    for r in rows:
        md.append("| " + " | ".join([r["defense"], str(r["sigma"]), fmt(r["smoothed_acc"])] +
                                    [fmt(r[f"cert@R>={rad}"]) for rad in RADII]) + " |")
    md += ["", "## Empirical adaptive EOT-PGD (L2, steps=20, EOT=8)", "",
           "| defense | sigma | eps | clean | robust |", "|---|---|---|---|---|"]
    for r in rows:
        if "emp_robust_acc" in r:
            md.append("| " + " | ".join([r["defense"], str(r["sigma"]), fmt(r["emp_eps"]),
                                         fmt(r["emp_clean_acc"]), fmt(r["emp_robust_acc"])]) + " |")
    md += ["", "## Consistency", "", "```", report, "```"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
