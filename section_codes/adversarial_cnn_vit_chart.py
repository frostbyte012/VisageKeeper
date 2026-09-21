#!/usr/bin/env python3
"""
adversarial_cnn_vit_chart.py -- motivation figure: CNN vs ViT face backbones
under FGSM / iFGSM / BPDA.

CNN numbers: VGGFace2 (from the earlier experiment, adversarial_face_eval.py).
ViT numbers: LFW aligned, timm face ViTs (vit_adversarial_eval.py). The dataset
differs across the CNN|ViT divide -- stated in the caption and marked by the
vertical separator in the plot.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.ticker as mtick

# ── Data ─────────────────────────────────────────────────────────────────────
cnn_models = ["ResNet50-FT", "SENet50-FT", "ResNet50-S", "SENet50-S", "FaceNet"]
vit_models = ["ViT-Small", "ViT-Tiny"]
models = cnn_models + vit_models
n_cnn = len(cnn_models)
x = np.arange(len(models))

#              ResNet-FT SENet-FT ResNet-S SENet-S FaceNet | ViT-S  ViT-Ti
base_acc  = [76.2, 87.5, 88.1, 84.8, 83.8,   75.8, 60.6]
fgsm_005  = [41.7, 33.1, 24.9, 36.9, 39.2,   48.4, 45.2]
ifgsm_005 = [4.0,  3.0,  2.0,  3.0,  2.0,     1.2,  0.0]
bpda_010  = [1.0,  1.0,  1.0,  1.0,  1.0,     18.0, 3.8]

all_data = [base_acc, fgsm_005, ifgsm_005, bpda_010]
labels   = ['Base', 'FGSM (ε=0.05)', 'iFGSM (ε=0.05)', 'BPDA (ε=0.10)']
# gray baseline (reference) + three validated attack hues
colors   = ["#9AA0A6", "#eb6834", "#2a78d6", "#7c4d9c"]
edge     = "#2b2b2b"

# ── Figure ───────────────────────────────────────────────────────────────────
plt.style.use('default')
fig, ax = plt.subplots(figsize=(17, 7))
fig.patch.set_facecolor('white')

width   = 0.20
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

bars_list = []
for i, (data, color, label) in enumerate(zip(all_data, colors, labels)):
    b = ax.bar(x + offsets[i], data, width, label=label, color=color,
               edgecolor=edge, linewidth=1.2, zorder=3)
    bars_list.append(b)

# ── Data labels ──────────────────────────────────────────────────────────────
for bars, data in zip(bars_list, all_data):
    for bar, val in zip(bars, data):
        if val >= 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                    f'{val:g}', ha='center', va='bottom', fontsize=19,
                    rotation=90, color='#222')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 1.0, f'{val:g}',
                    ha='center', va='bottom', fontsize=14, rotation=90, color='#555')

# ── CNN | ViT divider + family band labels ───────────────────────────────────
div_x = n_cnn - 0.5
ax.axvline(div_x, color='#555', lw=1.6, ls=(0, (6, 4)), zorder=2)
ax.text((n_cnn - 1) / 2, 112, "CNN backbones  (VGGFace2)", ha='center',
        va='center', fontsize=18, fontweight='600', color='#444')
ax.text(n_cnn + (len(vit_models) - 1) / 2, 112, "ViT backbones  (LFW)",
        ha='center', va='center', fontsize=18, fontweight='600', color='#444')
ax.axvspan(div_x, len(models) - 0.5, color="#2a78d6", alpha=0.05, zorder=0)

# ── Axes styling ─────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=20, fontweight='500')
ax.set_ylabel("Verification Accuracy (%)", fontsize=23, fontweight='500', labelpad=18)
ax.tick_params(axis='y', labelsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks(np.arange(0, 101, 10))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
ax.set_ylim(0, 120)
ax.set_xlim(-0.6, len(models) - 0.4)
ax.grid(axis='y', lw=0.5, alpha=0.35, zorder=0)

# ── Legend ───────────────────────────────────────────────────────────────────
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=19,
          frameon=False, ncol=4, handletextpad=0.5, columnspacing=1.1)

# ── Outer border ─────────────────────────────────────────────────────────────
border = patches.Rectangle((0, 0), 1, 1, linewidth=2.2, edgecolor='black',
                           facecolor='none', transform=fig.transFigure,
                           figure=fig, zorder=100)
fig.patches.append(border)

plt.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.10)
plt.savefig("results/adversarial_cnn_vit_chart.pdf", dpi=300, bbox_inches='tight')
plt.savefig("results/adversarial_cnn_vit_chart.png", dpi=200, bbox_inches='tight')
print("saved -> results/adversarial_cnn_vit_chart.{pdf,png}")
