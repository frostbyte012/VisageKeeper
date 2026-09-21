#!/usr/bin/env python3
"""
adversarial_dense_chart.py -- dense, layered motivation figure (Fig.5-style):
  * grouped verification-accuracy bars (Base / FGSM / iFGSM / BPDA), left axis,
  * overlaid smooth spline of MEAN adversarial accuracy per model, right axis
    (the "robustness trend" -- low = collapses under attack),
  * CNN | ViT region shading + band labels,
  * per-model worst-case collapse deltas (Base -> iFGSM).

CNN numbers: VGGFace2. ViT numbers: LFW (timm face ViTs). Dataset caveat in caption.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from scipy.interpolate import make_interp_spline

# ── Data ─────────────────────────────────────────────────────────────────────
cnn_models = ["ResNet50-FT", "SENet50-FT", "ResNet50-S", "SENet50-S", "FaceNet"]
vit_models = ["ViT-Small", "ViT-Tiny"]
models = cnn_models + vit_models
# short tick labels so they never collide horizontally at large font size
xtick_labels = ["R50-FT", "SE50-FT", "R50-S", "SE50-S", "FaceNet", "ViT-S", "ViT-Ti"]
n_cnn = len(cnn_models)
GROUP_PITCH = 1.85                                  # >1 spreads groups apart
x = np.arange(len(models)) * GROUP_PITCH

base_acc  = [76.2, 87.5, 88.1, 84.8, 83.8,  75.8, 60.6]
fgsm_005  = [41.7, 33.1, 24.9, 36.9, 39.2,  48.4, 45.2]
ifgsm_005 = [4.0,  3.0,  2.0,  3.0,  2.0,    1.2,  0.0]
bpda_010  = [1.0,  1.0,  1.0,  1.0,  1.0,   18.0,  3.8]

all_data = [base_acc, fgsm_005, ifgsm_005, bpda_010]
labels   = ['Base', 'FGSM (ε=0.05)', 'iFGSM (ε=0.05)', 'BPDA (ε=0.10)']
colors   = ["#9AA0A6", "#eb6834", "#2a78d6", "#7c4d9c"]
edge     = "#2b2b2b"

# mean adversarial accuracy per model (the robustness trend line)
mean_adv = [np.mean([f, i, b]) for f, i, b in zip(fgsm_005, ifgsm_005, bpda_010)]

# ── Figure ───────────────────────────────────────────────────────────────────
plt.style.use('default')
fig, ax = plt.subplots(figsize=(20, 9))
fig.patch.set_facecolor('white')
ax2 = ax.twinx()                                    # secondary axis for the trend

# region shading behind everything
div_x   = (n_cnn - 0.5) * GROUP_PITCH               # boundary between CNN & ViT
left_x  = -0.6 * GROUP_PITCH
right_x = (len(models) - 1) * GROUP_PITCH + 0.6 * GROUP_PITCH
ax.axvspan(left_x, div_x, color="#eb6834", alpha=0.045, zorder=0)
ax.axvspan(div_x, right_x, color="#2a78d6", alpha=0.06, zorder=0)

width   = 0.36                                      # thick bars
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
bars_list = []
for i, (data, color, label) in enumerate(zip(all_data, colors, labels)):
    b = ax.bar(x + offsets[i], data, width, label=label, color=color,
               edgecolor=edge, linewidth=1.5, zorder=3)
    bars_list.append(b)

# value labels -- placed just above each bar top, rotated, no overlaps
for si, (bars, data) in enumerate(zip(bars_list, all_data)):
    for bar, val in zip(bars, data):
        if val >= 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f'{val:g}', ha='center', va='bottom', fontsize=28,
                    rotation=90, color='#222', zorder=4)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 1.4, f'{val:g}',
                    ha='center', va='bottom', fontsize=24, rotation=90,
                    color='#555', zorder=4)

# ── secondary-axis robustness spline (mean adversarial accuracy) ─────────────
xs = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, mean_adv, k=2)
ys = np.clip(spl(xs), 0, None)
ax2.plot(xs, ys, color="#1b5e20", lw=4.0, zorder=6, solid_capstyle='round')
ax2.scatter(x, mean_adv, color="#1b5e20", s=130, zorder=7, edgecolor='white', linewidth=2.0)
import matplotlib.patheffects as pe
# label below-right of each node, in the open gap between the spline and the
# small iFGSM/BPDA bars -- clear of the tall FGSM value labels above.
for xi, mv in zip(x, mean_adv):
    t = ax2.annotate(f'{mv:.1f}', (xi, mv), textcoords="offset points",
                     xytext=(34, -20), ha='center', fontsize=26, color="#1b5e20",
                     fontweight='700', zorder=9)
    t.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])
ax2.plot([], [], color="#1b5e20", lw=3.0, marker='o',
         label='Mean adversarial acc. (robustness)')

# ── worst-case collapse callout (Base -> iFGSM), in a clean band above bars ──
# Placed at a FIXED height in the headroom so it never touches any bar/label.
for xi, b0, bi in zip(x, base_acc, ifgsm_005):
    drop = (b0 - bi) / b0 * 100
    ax.annotate(f'↓{drop:.0f}%', xy=(xi, 109), ha='center', va='center',
                fontsize=23, fontweight='700', color="#b71c1c", zorder=8,
                bbox=dict(boxstyle="round,pad=0.28", fc="#fdecea",
                          ec="#e57373", lw=1.2))

# ── CNN | ViT divider + band labels ──────────────────────────────────────────
ax.axvline(div_x, color='#555', lw=1.8, ls=(0, (6, 4)), zorder=2)
ax.text((n_cnn - 1) / 2 * GROUP_PITCH, 122, "CNN backbones  (VGGFace2)",
        ha='center', va='center', fontsize=25, fontweight='700', color='#444')
ax.text((n_cnn + (len(vit_models) - 1) / 2) * GROUP_PITCH, 122,
        "ViT backbones  (LFW)", ha='center', va='center', fontsize=25,
        fontweight='700', color='#444')

# ── Axes styling ─────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(xtick_labels, fontsize=29, fontweight='600')
ax.set_ylabel("Verification Accuracy (%)", fontsize=32, fontweight='600', labelpad=40)
ax.tick_params(axis='y', labelsize=29, pad=8)
ax.set_yticks(np.arange(0, 101, 10))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
ax.set_ylim(0, 130)
ax.set_xlim(left_x, right_x)
ax.grid(axis='y', lw=0.6, alpha=0.30, zorder=0)
ax.spines['top'].set_visible(False)

ax2.set_ylabel("Mean Adversarial\nAccuracy (%)", fontsize=30, fontweight='600',
               labelpad=22, color="#1b5e20", linespacing=0.95)
ax2.tick_params(axis='y', labelsize=28, colors="#1b5e20")
ax2.set_ylim(0, 62)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_edgecolor("#1b5e20")
ax2.spines['right'].set_linewidth(2.0)

# ── Legend (bars + trend line together) ──────────────────────────────────────
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fontsize=24, frameon=False, ncol=5, handletextpad=0.5,
          columnspacing=1.1)

plt.subplots_adjust(left=0.115, right=0.915, top=0.82, bottom=0.10)
plt.savefig("results/adversarial_dense_chart.pdf", dpi=300, bbox_inches='tight')
plt.savefig("results/adversarial_dense_chart.png", dpi=200, bbox_inches='tight')
print("saved -> results/adversarial_dense_chart.{pdf,png}")
