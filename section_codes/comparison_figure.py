#!/usr/bin/env python3
"""
section_codes/comparison_figure.py -- three-panel comparison with prior work.

(a) COST vs GUARANTEE, log-log scatter. Each method is a point: x = time to one
    decision, y = certified radius delivered. Methods with no certificate sit on
    the y=0 floor, drawn as a shaded "no guarantee" band -- they buy speed, or
    not, but never a proof. Marker area encodes on-chip memory where known.
    The dashed frontier is what VisageKeeper achieves; everything above-left of
    it is strictly better, and nothing published occupies that region.

(b) THE SAMPLING BILL. Stacked bars show where the certification cost goes and
    what alpha-spending removes, per sigma. The right axis carries the draws
    actually used, measured at 203 (sigma=0.5) and 242 (sigma=0.75).

(c) ACCURACY-RADIUS TRADE, measured. Certified accuracy at three radii per
    configuration, with the mean radius overlaid, showing the guarantee holds
    as the cheaper denoiser is selected.

PROVENANCE. Solid markers and all bars are measured in this work. Hollow grey
markers are QUOTED from the cited papers on their own hardware and task, and
are labelled as such -- no like-for-like accuracy or GFLOP figure exists for
them on our benchmarks, which is why they appear only in panel (a).

Run:  python section_codes/comparison_figure.py --out Figures/comparison.pdf
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

# ---- measured, this work (ZCU104 Fixed16, 100 MHz, 112x112) ----------------
DNCNN_MS, VIT_MS = 129.9, 1158.5
DRAWS, DRAWS_MAX = 203, 1000
OURS = [
    # label,           s/decision,                     Rbar,  BRAM, colour
    ("VisageKeeper\nDnCNN", DNCNN_MS*DRAWS/1000,        1.126,  32,  "#2a78d6"),
    ("VisageKeeper\nViT",   VIT_MS*DRAWS/1000,          1.089, 334,  "#1b9e77"),
]
# ---- quoted from the cited papers (different HW/task; no certificate) ------
QUOTED = [
    ("DiffPure",     11.13, 0.0, "purification, no proof"),
    ("FeatSqueeze",   0.04, 0.0, "purification, no proof"),
    ("KimNIL",        0.30, 0.0, "purification, no proof"),
]
C_QUOT, EDGE, RLINE = "#8a8f94", "#2b2b2b", "#b8299a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Figures/comparison.pdf")
    ap.add_argument("--width", type=float, default=9.8)
    ap.add_argument("--height", type=float, default=3.5)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 10.5, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(args.width, args.height),
        gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.05], wspace=0.42))

    # =====================================================================
    # (a) cost vs guarantee
    # =====================================================================
    axA.axhspan(-0.12, 0.10, color="#c62828", alpha=0.07, zorder=0)
    t = axA.annotate("no guarantee at any price", (3.0, -0.075), fontsize=8.6,
                     color="#a02020", fontweight="600", zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

    dy = [16, 30, 16]
    for k, (name, sec, r, note) in enumerate(QUOTED):
        axA.scatter(sec, 0.0, s=140, facecolor="white", edgecolor=C_QUOT,
                    linewidth=1.5, marker="o", zorder=5, hatch="xx")
        t = axA.annotate(name, (sec, 0.0), textcoords="offset points",
                         xytext=(0, dy[k]), ha="center", fontsize=8.6,
                         color="#555", zorder=7)
        t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    for name, sec, r, bram, col in OURS:
        axA.scatter(sec, r, s=90 + bram*1.1, facecolor=col, edgecolor=EDGE,
                    linewidth=1.1, alpha=0.92, zorder=6)
        off = (-52, -22) if "DnCNN" in name else (44, -24)
        t = axA.annotate(name, (sec, r), textcoords="offset points",
                         xytext=off, ha="center", fontsize=8.8,
                         fontweight="700", color=col, linespacing=0.95, zorder=8)
        t.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

    # the region nobody occupies
    axA.plot([OURS[0][1], OURS[1][1]], [OURS[0][2], OURS[1][2]],
             color="#444", lw=1.3, ls=(0, (5, 3)), zorder=4)
    t = axA.annotate("only region with\na proof at all", (230, 1.40), fontsize=8.8,
                     color="#333", ha="center", linespacing=1.0, zorder=8)
    t.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

    axA.set_xscale("log")
    axA.set_xlim(0.02, 3000); axA.set_ylim(-0.12, 1.62)
    axA.set_xlabel("time to one decision (s, log)")
    axA.set_ylabel("certified radius $\\bar{R}$")
    axA.grid(lw=0.4, alpha=0.28, zorder=0)
    axA.spines[["top", "right"]].set_visible(False)
    axA.set_title("(a) cost vs guarantee", fontsize=11.5, pad=7)

    # =====================================================================
    # (b) the sampling bill
    # =====================================================================
    sig = ["$\\sigma{=}0.5$", "$\\sigma{=}0.75$"]
    used_d = [203, 242]
    xb = np.arange(len(sig))
    w = 0.46
    for i, u in enumerate(used_d):
        axB.bar(xb[i], u, w, color="#2a78d6", edgecolor=EDGE, lw=0.9, zorder=3)
        axB.bar(xb[i], DRAWS_MAX - u, w, bottom=u, color="#2a78d6", alpha=0.20,
                hatch="///", edgecolor=EDGE, lw=0.8, zorder=2)
        t = axB.annotate(f"{u}", (xb[i], u), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=10,
                         fontweight="700", color="#2a78d6", zorder=7)
        t.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])
        pct = 100*(DRAWS_MAX-u)/DRAWS_MAX
        t = axB.annotate(f"$-{pct:.0f}\\%$", (xb[i], u + (DRAWS_MAX-u)/2),
                         ha="center", fontsize=10.5, fontweight="700",
                         color="#1a4f8a", zorder=7)
        t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    axB.axhline(DRAWS_MAX, color="#c62828", lw=1.2, ls=(0, (5, 3)), zorder=4)
    t = axB.annotate("fixed $n{=}1000$", (0.5, DRAWS_MAX), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9, color="#c62828",
                     fontweight="600", zorder=8)
    t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

    axB.set_xticks(xb); axB.set_xticklabels(sig)
    axB.set_ylabel("noisy copies per decision")
    axB.set_ylim(0, 1230)
    axB.grid(axis="y", lw=0.4, alpha=0.28, zorder=0)
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title("(b) sampling avoided", fontsize=11.5, pad=7)

    # =====================================================================
    # (c) accuracy-radius trade, measured
    # =====================================================================
    cfg = ["DnCNN\n$\\sigma{=}0.5$", "ViT\n$\\sigma{=}0.5$", "ViT\n$\\sigma{=}0.75$"]
    c025 = [97.6, 93.5, 89.0]
    c050 = [96.0, 89.0, 83.0]
    c100 = [81.8, 74.5, 70.0]
    rbar = [1.126, 1.089, 1.033]
    xc = np.arange(len(cfg)); bw = 0.24
    cols3 = ["#9ecae1", "#4292c6", "#08519c"]
    for k, (vals, lab) in enumerate(((c025, "C@0.25"), (c050, "C@0.5"), (c100, "C@1.0"))):
        axC.bar(xc + (k-1)*bw, vals, bw, color=cols3[k], edgecolor=EDGE,
                lw=0.7, zorder=3, label=lab)

    ax2 = axC.twinx()
    ax2.plot(xc, rbar, color=RLINE, lw=2.0, marker="D", ms=5.0,
             mec="white", mew=1.0, zorder=6)
    ax2.axhline(1.0, color=RLINE, lw=0.9, ls=":", alpha=0.55, zorder=1)
    t = ax2.annotate("$\\bar{R}\\!\\approx\\!1$ throughout", (1.0, 1.089),
                     textcoords="offset points", xytext=(0, 30), ha="center",
                     fontsize=9, color=RLINE, fontweight="600", zorder=9)
    t.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

    axC.set_xticks(xc); axC.set_xticklabels(cfg)
    axC.set_ylabel("certified accuracy (%)")
    axC.set_ylim(0, 126); axC.set_yticks([0, 25, 50, 75, 100])
    axC.grid(axis="y", lw=0.4, alpha=0.28, zorder=0)
    axC.spines[["top"]].set_visible(False)
    ax2.set_ylabel("$\\bar{R}$", color=RLINE, fontsize=11)
    ax2.tick_params(axis="y", colors=RLINE, labelsize=9.5)
    ax2.set_ylim(0, 1.85); ax2.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax2.spines[["top"]].set_visible(False); ax2.spines["right"].set_color(RLINE)
    axC.set_title("(c) what the proof delivers", fontsize=11.5, pad=7)

    handles = [
        Line2D([], [], ls="", marker="o", mfc="#2a78d6", mec=EDGE, ms=9,
               label="ours, measured (area $\\propto$ BRAM)"),
        Line2D([], [], ls="", marker="o", mfc="white", mec=C_QUOT, mew=1.6, ms=9,
               label="prior work, quoted (other HW)"),
        Patch(fc="white", ec=EDGE, lw=0.8, hatch="///", label="avoided by early stopping"),
        Patch(fc=cols3[0], ec=EDGE, lw=0.7, label="C@0.25"),
        Patch(fc=cols3[1], ec=EDGE, lw=0.7, label="C@0.5"),
        Patch(fc=cols3[2], ec=EDGE, lw=0.7, label="C@1.0"),
        Line2D([], [], color=RLINE, lw=2.0, marker="D", ms=4.5, label="$\\bar{R}$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=7, frameon=False, handlelength=1.6, columnspacing=1.0,
               handletextpad=0.5, fontsize=9)

    fig.tight_layout(rect=[0, 0.02, 1, 0.86])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=400, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
