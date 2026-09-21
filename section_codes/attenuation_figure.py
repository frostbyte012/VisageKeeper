#!/usr/bin/env python3
"""
section_codes/attenuation_figure.py -- replaces the attenuation TABLE with one
dense two-panel figure, in the style of the reference plots (grouped bars on the
left axis, overlaid trend lines on the right axis, annotated regime bands).

Panel (a): identity retained a(sigma). Bars = per-denoiser retention at each
noise level, lines = the fitted closed form 1/(1+k*sigma^2) extended through the
sweep. The shaded band marks where the two denoisers are statistically tied, so
the crossover is visible rather than inferred from a column of numbers.

Panel (b): what the threshold correction is worth. Grouped bars = certified
accuracy gain (points) per dataset, line = mean gain across datasets. The
left-to-right rise is the paper's claim -- the worse the miscalibration, the more
the correction recovers -- and it reads instantly as a slope.

Values are the 500-pair LFW runs (attenuation_law_{vit,dncnn}500.json) and the
300-pair CFP-FP / AgeDB runs (attenuation_{cfpfp,agedb}.json), matching the
numbers quoted in the text.

Run:  python section_codes/attenuation_figure.py --out Figures/attenuation.pdf
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

# ---- measured values (see module docstring for provenance) -----------------
SIG = np.array([0.25, 0.50, 0.75, 1.00])

A = {  # identity retained a(sigma)
    "DnCNN": {"LFW":   [0.864, 0.666, 0.478, 0.319],
              "CFP-FP":[0.865, 0.657, 0.446, 0.272],
              "AgeDB": [0.847, 0.652, 0.501, 0.336]},
    "ViT":   {"LFW":   [0.861, 0.706, 0.601, 0.515],
              "CFP-FP":[0.855, 0.612, 0.342, 0.198],
              "AgeDB": [0.796, 0.595, 0.413, 0.348]},
}
GAIN = {  # certified-accuracy gain from the corrected threshold, in points
    "DnCNN": {"LFW":   [0.4, 3.2, 12.0, 20.8],
              "CFP-FP":[0.0, 11.3, 14.7, 13.3],
              "AgeDB": [6.0, 14.0, 23.3, 20.0]},
    "ViT":   {"LFW":   [0.8, 2.4, 6.0, 10.4],
              "CFP-FP":[0.0, 6.0, 14.0, 11.3],
              "AgeDB": [12.0, 19.3, 18.7, 16.7]},
}
K_FIT = {"LFW": 1.909, "AgeDB": 1.994, "CFP-FP": 2.303}   # DnCNN closed form
# held-out R^2 of the affine fit at sigma=1.0 -- how well the law still describes
# each denoiser at the hardest noise level (LFW; see Table in the text)
R2_AT_1 = {"DnCNN": 0.55, "ViT": 0.81}

C = {"LFW": "#2a78d6", "CFP-FP": "#eb6834", "AgeDB": "#1b9e77"}
EDGE = "#2b2b2b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Figures/attenuation.pdf")
    ap.add_argument("--width", type=float, default=9.2)
    ap.add_argument("--height", type=float, default=3.65)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 10,
        "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(args.width, args.height))
    x = np.arange(len(SIG))

    # =====================================================================
    # (a) identity retained: bars per dataset, closed-form curves on top
    # =====================================================================
    w = 0.13
    offs = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * w
    k = 0
    for arch, hatch, alpha in (("DnCNN", "", 0.95), ("ViT", "////", 0.95)):
        for ds in ("LFW", "CFP-FP", "AgeDB"):
            axA.bar(x + offs[k], A[arch][ds], w, color=C[ds], alpha=alpha,
                    hatch=hatch, edgecolor=EDGE, linewidth=0.5, zorder=3)
            k += 1

    # the fitted closed form, drawn continuously so the "law" is visible
    xs_sig = np.linspace(0.25, 1.00, 200)
    xs_pos = np.interp(xs_sig, SIG, x)
    for j, ds in enumerate(("LFW", "CFP-FP", "AgeDB")):
        # anchor each curve on its own DnCNN bar centre so line and bars align
        axA.plot(xs_pos + offs[j], 1.0 / (1.0 + K_FIT[ds] * xs_sig ** 2),
                 color=C[ds], lw=1.5, ls="-", zorder=5,
                 path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    axA.plot([], [], color="#444", lw=1.5, label="fit")

    # tie region: where the two denoisers are within measurement noise
    axA.axvspan(-0.55, 0.5, color="#888888", alpha=0.10, zorder=0)
    axA.text(-0.03, 1.20, "tied\n(use DnCNN)", ha="center", va="top",
             fontsize=9.5, color="#444", linespacing=1.0)
    # fit quality at the hardest noise level, on the two LFW bars it describes
    # DnCNN: leader line down-left into clear space; ViT: directly above its bar
    tA = axA.annotate(f"$R^2{{=}}{R2_AT_1['DnCNN']:.2f}$",
                      xy=(3 + offs[0], A["DnCNN"]["LFW"][-1]), xytext=(2.16, 0.16),
                      fontsize=9, color=C["LFW"], fontweight="600", zorder=8,
                      ha="left", va="center",
                      arrowprops=dict(arrowstyle="-", lw=0.8, color=C["LFW"],
                                      shrinkA=1, shrinkB=2))
    tB = axA.annotate(f"$R^2{{=}}{R2_AT_1['ViT']:.2f}$",
                      (3 + offs[3], A["ViT"]["LFW"][-1]), textcoords="offset points",
                      xytext=(0, 7), ha="center", fontsize=9, color=C["LFW"],
                      fontweight="600", zorder=8)
    for t in (tA, tB):
        t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

    axA.annotate("ViT keeps $+61\\%$\nmore than DnCNN", xy=(3 + offs[3], 0.515),
                 xytext=(1.45, 1.03), fontsize=9.5, color="#1b5e20",
                 linespacing=0.95, ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="#1b5e20",
                                 connectionstyle="arc3,rad=-0.25"))

    axA.set_xticks(x); axA.set_xticklabels([f"{s:g}" for s in SIG])
    axA.set_xlabel("noise level $\\sigma$")
    axA.set_ylabel("identity retained  $a(\\sigma)$")
    axA.set_ylim(0, 1.24); axA.set_xlim(-0.55, len(SIG) - 0.45)
    axA.grid(axis="y", lw=0.4, alpha=0.30, zorder=0)
    axA.spines[["top", "right"]].set_visible(False)
    axA.set_title("(a) the law: how much identity survives", fontsize=12, pad=6)

    # =====================================================================
    # (b) what the correction buys: bars per dataset + mean trend line
    # =====================================================================
    w2 = 0.13
    k = 0
    for arch, hatch in (("DnCNN", ""), ("ViT", "////")):
        for ds in ("LFW", "CFP-FP", "AgeDB"):
            axB.bar(x + offs[k], GAIN[arch][ds], w2, color=C[ds],
                    hatch=hatch, edgecolor=EDGE, linewidth=0.5, zorder=3)
            k += 1

    mean_gain = [np.mean([GAIN[a][d][i] for a in GAIN for d in GAIN[a]])
                 for i in range(len(SIG))]
    axB.plot(x, mean_gain, color="#7c2d92", lw=2.0, marker="o", ms=4.5,
             mec="white", mew=1.0, zorder=6)
    # place each label in the gap beside its marker, alternating to dodge bars
    lab_off = [(-20, 2), (-21, 3), (-22, 4), (-22, 5)]
    for xi, mv, (dx, dy) in zip(x, mean_gain, lab_off):
        t = axB.annotate(f"{mv:.1f}", (xi, mv), textcoords="offset points",
                         xytext=(dx, dy), ha="center", fontsize=9.5,
                         color="#7c2d92", fontweight="700", zorder=8)
        t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
    axB.annotate("the worse the miscalibration,\nthe more it recovers",
                 xy=(2.62, 14.9), xytext=(0.12, 26.6), fontsize=9.5,
                 color="#7c2d92", linespacing=0.95, ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="#7c2d92",
                                 connectionstyle="arc3,rad=0.2"))

    axB.axhline(0, color=EDGE, lw=0.6)
    axB.set_xticks(x); axB.set_xticklabels([f"{s:g}" for s in SIG])
    axB.set_xlabel("noise level $\\sigma$")
    axB.set_ylabel("certified acc. gain $\\Delta$ (pts)")
    axB.set_ylim(0, 29); axB.set_xlim(-0.55, len(SIG) - 0.45)
    axB.grid(axis="y", lw=0.4, alpha=0.30, zorder=0)
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title("(b) what the corrected threshold recovers", fontsize=12, pad=6)

    # =====================================================================
    # one shared legend: colour = dataset, hatch = denoiser
    # =====================================================================
    handles = [Patch(fc=C[d], ec=EDGE, lw=0.5, label=d) for d in ("LFW", "CFP-FP", "AgeDB")]
    handles += [Patch(fc="white", ec=EDGE, lw=0.5, label="DnCNN"),
                Patch(fc="white", ec=EDGE, lw=0.5, hatch="////", label="ViT"),
                Line2D([], [], color="#444", lw=1.5, label="closed form $1/(1{+}k\\sigma^2)$"),
                Line2D([], [], color="#7c2d92", lw=2.0, marker="o", ms=4, label="mean gain")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=7, frameon=False, handlelength=1.9, columnspacing=1.4,
               handletextpad=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=400, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
