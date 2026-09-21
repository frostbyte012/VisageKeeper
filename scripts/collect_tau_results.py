#!/usr/bin/env python3
"""
scripts/collect_tau_results.py -- assemble the tau-calibration paper package.

Reads results/paper/tau_opt_s*.json (one per sigma) and writes:
  table_tau.csv          per-sigma clean-tau vs certification-tau, held-out
  fig_tau.pdf/.png       two panels: accuracy vs tau, and the accuracy/radius
                         frontier the choice of tau moves along
  tau_summary.md         human-readable summary with the headline deltas

Run after the sweep:  python scripts/collect_tau_results.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper")


def load_all():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "tau_opt_s*.json"))):
        m = re.search(r"tau_opt_s(\d+)\.json$", p)
        if not m:
            continue
        with open(p) as f:
            d = json.load(f)
        rows.append((d.get("sigma", int(m.group(1)) / 100), d))
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    rows = load_all()
    if not rows:
        print("no tau_opt_s*.json found -- run scripts/optimize_tau.py first")
        return

    # ---- table -------------------------------------------------------------
    cols = ["sigma", "tau_clean", "acc_clean", "R_clean",
            "tau_cert", "acc_cert", "R_cert", "d_acc", "d_R_pct", "se"]
    table = []
    for sigma, d in rows:
        t = d["test"]
        b, c = t["tau_clean"], t["tau_best_radius"]
        table.append({
            "sigma": sigma,
            "tau_clean": round(b["tau"], 4), "acc_clean": round(b["acc"], 4),
            "R_clean": round(b["meanR"], 4),
            "tau_cert": round(c["tau"], 4), "acc_cert": round(c["acc"], 4),
            "R_cert": round(c["meanR"], 4),
            "d_acc": round(c["acc"] - b["acc"], 4),
            "d_R_pct": round((c["meanR"] - b["meanR"]) / max(b["meanR"], 1e-9) * 100, 2),
            "se": round(d.get("se", float("nan")), 4),
        })
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "table_tau.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(table)

    print(f"{'sigma':>6} {'tau_cl':>7} {'acc_cl':>7} {'tau_ct':>7} {'acc_ct':>7} "
          f"{'d_acc':>7} {'dR%':>7} {'se':>6}")
    for r in table:
        flag = "*" if abs(r["d_acc"]) > 2 * r["se"] else " "
        print(f"{r['sigma']:>6} {r['tau_clean']:>7.4f} {r['acc_clean']:>7.4f} "
              f"{r['tau_cert']:>7.4f} {r['acc_cert']:>7.4f} {r['d_acc']:>+7.4f} "
              f"{r['d_R_pct']:>+7.2f} {r['se']:>6.4f} {flag}")
    print("(* = beyond 2 s.e.)")

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("viridis")
    for k, (sigma, d) in enumerate(rows):
        sw = d["calib_sweep"]
        taus = [s["tau"] for s in sw]
        accs = [s["acc"] for s in sw]
        Rs = [s["meanR"] for s in sw]
        col = cmap(k / max(n - 1, 1))
        axes[0].plot(taus, accs, lw=2, color=col, label=f"$\\sigma$={sigma}")
        axes[0].axvline(d["tau_clean"], color=col, ls=":", lw=1.2, alpha=0.8)
        axes[1].plot(Rs, accs, lw=2, color=col, label=f"$\\sigma$={sigma}")

    axes[0].set_xlabel(r"verification threshold $\tau$")
    axes[0].set_ylabel("certified accuracy")
    axes[0].set_title(r"Accuracy vs $\tau$ (dotted = clean-FAR $\tau$)", fontsize=10)
    axes[1].set_xlabel(r"mean certified radius $R$")
    axes[1].set_ylabel("certified accuracy")
    axes[1].set_title(r"Accuracy/radius frontier traced by $\tau$", fontsize=10)
    for ax in axes:
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_tau.{ext}"), dpi=200)

    # ---- summary -----------------------------------------------------------
    md = ["# Certification-aware threshold calibration", "",
          "Held-out results: tau chosen on a disjoint calibration half.", "",
          "| sigma | tau (clean FAR) | acc | tau (certification) | acc | d_acc | d_R |",
          "|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['sigma']} | {r['tau_clean']:.4f} | {r['acc_clean']:.4f} "
                  f"| {r['tau_cert']:.4f} | {r['acc_cert']:.4f} | "
                  f"{r['d_acc']:+.4f} | {r['d_R_pct']:+.2f}% |")
    mean_d = sum(r["d_acc"] for r in table) / len(table)
    md += ["", f"Mean certified-accuracy gain: **{mean_d:+.4f}** "
           f"across {len(table)} noise levels, at zero additional inference cost."]
    with open(os.path.join(OUT, "tau_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nwrote table_tau.csv, fig_tau.{{pdf,png}}, tau_summary.md -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
