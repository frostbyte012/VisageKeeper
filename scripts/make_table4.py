#!/usr/bin/env python3
"""Build Table IV (baseline comparison) from the measured shard CSVs.

Presents clean AND robust accuracy side by side. This is deliberate: the
smoothing-only ablation reaches 49.5 robust TAR purely because its clean
accuracy is already 50.0 (chance) -- reading the robust column alone would
rank a non-functional verifier above the working one.
"""
import csv, glob, os

rows = []
for f in sorted(glob.glob("results/shards/*.csv")):
    tag = os.path.basename(f)
    with open(f) as fh:
        for r in csv.DictReader(fh):
            r["_src"] = tag
            rows.append(r)

with open("results/table4_baselines.csv", "w", newline="") as fh:
    keys = sorted({k for r in rows for k in r})
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader(); w.writerows(rows)

def pick(defense, attack, src_contains=None):
    for r in rows:
        if r["defense"] == defense and r["attack"] == attack:
            if src_contains and src_contains not in r["_src"]:
                continue
            return r
    return None

def val(r, k):
    if not r: return None
    try: return float(r.get(k, "") or "") * 100
    except ValueError: return None

ROWS = [
    ("none",                  None,        "No defense"),
    ("fsqueeze",              None,        "Feature squeezing~\\cite{xu2018feature}"),
    ("kim_nil",               None,        "Noise-injection layer"),
    ("iter_purify",           None,        "Iterative purification (DiffPure-style)"),
    ("frpure_ours_unet",      None,        "\\quad Ours: frequency filter only"),
    ("visagekeeper_smoothed", "visagekeeper", "\\textbf{Ours: denoise + smoothing}"),
    ("visagekeeper_smoothed", "ablation",  "\\quad\\textit{ablation: smoothing, no denoiser}"),
]

def fmt(v, bold_zero=False):
    if v is None: return "--"
    if bold_zero and v < 0.05: return r"\textbf{0.0}"
    return f"{v:.1f}"

L = []
L.append(r"\begin{table}[t]")
L.append(r"\centering")
L.append(r"\caption{Baseline comparison under one identical adaptive protocol "
         r"(LFW, FaceNet/VGGFace2, $\ell_\infty$, $\epsilon=8/255$, 200 dodging "
         r"+ 200 impersonation pairs, threshold frozen at the clean "
         r"TAR@FAR$=10^{-3}$ operating point). PGD-EOT is the primary attack and "
         r"differentiates through each defense; BPDA-EOT corroborates. Every "
         r"deterministic defense, including our own frequency filter in "
         r"isolation, is driven to $0.0$. Clean accuracy is reported alongside "
         r"robustness because the smoothing-only ablation attains $49.5$ robust "
         r"TAR solely by being at chance ($52.7$ clean) -- it has no accuracy "
         r"left to lose. Smoothing costs clean accuracy ($97.1\to91.8$); the "
         r"denoiser is what makes a certifiable noise level usable at all.}")
L.append(r"\label{tab:baselines}")
L.append(r"\begin{tabular}{lccccc}")
L.append(r"\toprule")
L.append(r" & \multicolumn{2}{c}{Clean} & \multicolumn{3}{c}{Robust TAR (\%)} \\")
L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}")
L.append(r"Defense & Acc. & TAR & FGSM & PGD-EOT & BPDA-EOT \\")
L.append(r"\midrule")
for dk, src, label in ROWS:
    c  = pick(dk, "clean", src)
    fg = pick(dk, "fgsm", src)
    pg = pick(dk, "pgd_eot", src)
    bp = pick(dk, "bpda_eot", src)
    cells = [fmt(val(c, "ver_acc")), fmt(val(c, "robust_tar")),
             fmt(val(fg, "robust_tar")), fmt(val(pg, "robust_tar"), True),
             fmt(val(bp, "robust_tar"), True)]
    L.append(f"{label} & " + " & ".join(cells) + r" \\")
L.append(r"\bottomrule")
L.append(r"\end{tabular}")
L.append(r"\end{table}")

tex = "\n".join(L)
open("results/table4_baselines.tex", "w").write(tex + "\n")
print(tex)
print(f"\n-> results/table4_baselines.csv ({len(rows)} rows)")
print( "-> results/table4_baselines.tex")
