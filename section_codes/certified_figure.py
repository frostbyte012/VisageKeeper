#!/usr/bin/env python3
"""
section_codes/certified_figure.py -- replaces the certified-robustness TABLE with
one dense figure in the style of the reference plots: grouped bars on the left
axis, an overlaid trend line on the right axis, and vertical dividers separating
the three backbones into labelled regimes.

Left axis / bars : certified accuracy at three radii (C@0.25, C@0.5, C@1.0),
                   grouped per dataset, hatched by radius. The left-to-right
                   fall within a group IS the accuracy-radius trade-off, so the
                   whole certified curve is legible per dataset.
Right axis / line: mean certified radius R-bar. It stays flat near 1.0 across
                   every dataset and backbone, which is the paper's point --
                   harder faces cost accuracy, not guarantee strength.
Dividers + bands : one region per backbone, so the ArcFace > VGGFace2 > CASIA
                   ordering is a visual step-down rather than a column scan.

Values are Table `tab:visagekeeper_certified` (sigma=0.5, n=1000).

Run:  python section_codes/certified_figure.py --out Figures/certified.pdf
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DATASETS = ["LFW", "CALFW", "CFP-FP", "CPLFW", "AgeDB"]
BACKBONES = ["ArcFace-R50", "FaceNet-VGGFace2", "FaceNet-CASIA"]

# C@0.25, C@0.5, C@1.0  (percent)
CERT = {
    "ArcFace-R50":      {"LFW": [97.6, 96.0, 81.8], "CALFW": [90.8, 87.8, 73.6],
                         "CFP-FP": [89.6, 84.2, 64.8], "CPLFW": [88.4, 83.2, 65.2],
                         "AgeDB": [76.8, 71.8, 58.4]},
    "FaceNet-VGGFace2": {"LFW": [82.0, 76.8, 60.4], "CALFW": [58.6, 55.6, 48.6],
                         "CFP-FP": [70.2, 63.4, 49.4], "CPLFW": [60.8, 57.0, 49.2],
                         "AgeDB": [58.2, 53.8, 45.8]},
    "FaceNet-CASIA":    {"LFW": [64.6, 60.0, 52.8], "CALFW": [56.0, 53.4, 48.8],
                         "CFP-FP": [62.8, 58.0, 47.6], "CPLFW": [58.2, 54.4, 45.2],
                         "AgeDB": [53.6, 50.2, 48.2]},
}
RBAR = {  # mean certified radius
    "ArcFace-R50":      [1.126, 1.089, 1.033, 1.026, 1.018],
    "FaceNet-VGGFace2": [1.000, 1.050, 0.971, 1.028, 1.031],
    "FaceNet-CASIA":    [1.015, 1.095, 1.019, 1.002, 1.116],
}
SMOOTH = {  # smoothed accuracy, annotated on the tallest bar of each group
    "ArcFace-R50":      [98.4, 93.2, 92.0, 91.8, 81.6],
    "FaceNet-VGGFace2": [87.0, 63.0, 76.2, 65.4, 62.2],
    "FaceNet-CASIA":    [71.6, 58.6, 66.6, 63.6, 55.8],
}
ABST = {  # abstention rate (%): pairs whose vote was too close to certify
    "ArcFace-R50":      [0.2, 0.8, 1.8, 0.6, 3.2],
    "FaceNet-VGGFace2": [2.0, 2.0, 4.0, 3.6, 2.0],
    "FaceNet-CASIA":    [2.4, 1.2, 2.4, 2.2, 1.8],
}
NO_DENOISER = 49.8  # LFW, sigma=0.5, denoiser removed

RAD_C = ["#9ecae1", "#4292c6", "#08519c"]      # C@0.25 / C@0.5 / C@1.0
RAD_H = ["", "///", "xxx"]
BAND = ["#eb6834", "#2a78d6", "#1b9e77"]
EDGE = "#2b2b2b"
RLINE = "#b8299a"
ABST_C = "#f0a202"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Figures/certified.pdf")
    ap.add_argument("--width", type=float, default=9.6)
    ap.add_argument("--height", type=float, default=4.35)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10.5,
        "ytick.labelsize": 11, "axes.linewidth": 0.9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # two stacked axes: the tall certified panel, and a thin abstention strip
    # underneath so the two quantities are never read as one stacked bar
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(args.width, args.height), sharex=True,
        gridspec_kw=dict(height_ratios=[6.4, 1.0], hspace=0.10))
    ax2 = ax.twinx()

    n_ds = len(DATASETS)
    pitch = 1.0
    x = np.arange(len(BACKBONES) * n_ds) * pitch
    w = 0.26
    offs = np.array([-1.0, 0.0, 1.0]) * w

    # ---- backbone regions, drawn first so bars sit on top -----------------
    for b, col in enumerate(BAND):
        lo = b * n_ds * pitch - 0.55
        hi = (b + 1) * n_ds * pitch - 0.45
        ax.axvspan(lo, hi, color=col, alpha=0.055, zorder=0)
        if b:
            ax.axvline(lo, color="#666", lw=1.2, ls=(0, (6, 4)), zorder=2)
        ax.text((lo + hi) / 2, 126.0, BACKBONES[b], ha="center", va="center",
                fontsize=11.5, fontweight="700", color="#333")

    # ---- bars: three radii per dataset -------------------------------------
    for b, bb in enumerate(BACKBONES):
        for d, ds in enumerate(DATASETS):
            xi = (b * n_ds + d) * pitch
            for r in range(3):
                ax.bar(xi + offs[r], CERT[bb][ds][r], w, color=RAD_C[r],
                       hatch=RAD_H[r], edgecolor=EDGE, linewidth=0.6, zorder=3)
            # smoothed accuracy above the group: the ceiling the radii fall from
            t = ax.annotate(f"{SMOOTH[bb][d]:.0f}", (xi, CERT[bb][ds][0]),
                            textcoords="offset points", xytext=(-13, 5),
                            ha="center", fontsize=9, color="#333",
                            fontweight="600", zorder=7)
            t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    # ---- abstention: its own strip, true scale --------------------------------
    for b, bb in enumerate(BACKBONES):
        lo = b * n_ds * pitch - 0.55
        hi = (b + 1) * n_ds * pitch - 0.45
        axb.axvspan(lo, hi, color=BAND[b], alpha=0.055, zorder=0)
        if b:
            axb.axvline(lo, color="#666", lw=1.2, ls=(0, (6, 4)), zorder=2)
        for d in range(n_ds):
            xi = (b * n_ds + d) * pitch
            axb.bar(xi, ABST[bb][d], w * 3.05, color=ABST_C, edgecolor=EDGE,
                    linewidth=0.5, zorder=3)
            axb.annotate(f"{ABST[bb][d]:.1f}", (xi, ABST[bb][d]),
                         textcoords="offset points", xytext=(0, 1.5),
                         ha="center", fontsize=7.6, color="#6b4a00", zorder=5)
    axb.set_ylabel("abst.\n(%)", fontsize=9.5, linespacing=0.95)
    axb.set_ylim(0, 5.6); axb.set_yticks([0, 4])
    axb.tick_params(axis="y", labelsize=9)
    axb.grid(axis="y", lw=0.4, alpha=0.30, zorder=0)
    axb.spines[["top", "right"]].set_visible(False)

    # ---- mean certified radius on the right axis ---------------------------
    rbar = np.concatenate([RBAR[b] for b in BACKBONES])
    ax2.plot(x, rbar, color=RLINE, lw=2.1, marker="D", ms=5.0,
             mec="white", mew=1.0, zorder=6)
    ax2.axhline(1.0, color=RLINE, lw=0.9, ls=":", alpha=0.55, zorder=1)
    t = ax2.annotate("$\\bar{R}$ stays $\\approx 1$ everywhere: harder faces\n"
                     "cost accuracy, not guarantee strength",
                     xy=(11.6, rbar[11]), xytext=(8.35, 1.60), fontsize=10,
                     color=RLINE, linespacing=1.0, ha="left", zorder=9,
                     arrowprops=dict(arrowstyle="->", lw=1.1, color=RLINE,
                                     connectionstyle="arc3,rad=0.2"))
    t.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

    # ---- the ablation: without a denoiser everything collapses to chance ---
    ax.axhline(NO_DENOISER, color="#c62828", lw=1.3, ls=(0, (5, 3)), zorder=4)
    t = ax.annotate("no denoiser: 49.8% (chance)", xy=(0.35, NO_DENOISER),
                    xytext=(-0.45, 41.0), fontsize=10, color="#c62828",
                    fontweight="600", ha="left", zorder=8)
    t.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

    # ---- axes ---------------------------------------------------------------
    axb.set_xticks(x)
    axb.set_xticklabels(DATASETS * len(BACKBONES), rotation=30, ha="right")
    ax.set_ylabel("certified accuracy (%)")
    ax.set_ylim(0, 132)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(axis="y", lw=0.4, alpha=0.30, zorder=0)
    ax.spines[["top", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)

    ax2.set_ylabel("mean certified radius  $\\bar{R}$", color=RLINE, fontsize=12)
    ax2.tick_params(axis="y", labelsize=11, colors=RLINE)
    ax2.set_ylim(0, 1.97)
    ax2.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax2.spines[["top"]].set_visible(False)
    ax2.spines["right"].set_color(RLINE)

    # ---- legend -------------------------------------------------------------
    handles = [Patch(fc=RAD_C[i], ec=EDGE, lw=0.6, hatch=RAD_H[i],
                     label=f"C@{r}") for i, r in enumerate(["0.25", "0.5", "1.0"])]
    handles += [Line2D([], [], color=RLINE, lw=2.1, marker="D", ms=5,
                       label="mean radius $\\bar{R}$"),
                Patch(fc=ABST_C, ec=EDGE, lw=0.5, label="abstention"),
                Line2D([], [], color="#c62828", lw=1.3, ls=(0, (5, 3)),
                       label="no denoiser")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.015),
               ncol=6, frameon=False, handlelength=1.7, columnspacing=1.5,
               handletextpad=0.6, fontsize=10.5)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=400, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
