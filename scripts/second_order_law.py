#!/usr/bin/env python3
"""
scripts/second_order_law.py -- fix the affine model where it breaks.

The single-population affine fit E[cos_noisy] ~= a*cos_clean + b degrades badly
at high noise on hard benchmarks (CFP-FP ViT: held-out R^2 = 0.15 at sigma=1.0,
vs 0.78 on LFW). That is exactly where the threshold correction matters most, so
the weakness is worth fixing rather than caveating.

Why it breaks: genuine and impostor pairs are NOT one population. Measured on
LFW at sigma=1.0, the genuine mean falls by -0.302 while the impostor mean rises
by +0.031. Forcing one line through both mixes a steep negative slope with a
nearly flat one; as sigma grows the two clouds converge and a single line
explains less and less of the variance.

Three models compared, all fitted on CALIBRATION pairs and scored on held-out:
  affine      cos_noisy ~ a*cos_clean + b                (current)
  quadratic   cos_noisy ~ a2*cos^2 + a*cos + b           (curvature, 1 pop)
  two-pop     separate (a,b) for genuine and impostor    (the hypothesis)

For two-pop we ALSO report the threshold it implies. Note the subtlety: at
decision time the label is unknown, so a two-population fit cannot be applied
per-pair. It is still useful in two ways -- it tells us whether population
mixing is the cause of the low R^2, and the IMPOSTOR fit alone is the correct
one for setting tau, because tau is calibrated against the impostor
distribution (FAR), not the genuine one.

Run:  python scripts/second_order_law.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy.stats import norm

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser
from frpure.defenses.cascade import _disable_fused_attention
from scripts.run_certify import load_pairs, balanced_subset, embed_all
from scripts.diag_error_overlap import CKPT


def load_den(arch, sigma, device):
    dn_w, vit_w = CKPT[sigma]
    path = vit_w if arch == "vit" else dn_w
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(d)
    for p in d.parameters():
        p.requires_grad_(False)
    return d


@torch.no_grad()
def draw_cosines(den, backbone, img1, idx, e2, sigma, draws, batch, device):
    out = []
    for j, i in enumerate(idx):
        g = e2[j].to(device).unsqueeze(0)
        cs, rem = [], draws
        while rem > 0:
            m = min(batch, rem)
            z = img1[i].unsqueeze(0).repeat(m, 1, 1, 1).to(device)
            z = (z + torch.randn_like(z) * sigma).clamp(0, 1)
            cs.append((backbone.embed(den(z)) * g).sum(1).cpu().numpy())
            rem -= m
        out.append(np.concatenate(cs))
    return np.stack(out)


def r2(y, pred):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def cert_stats(cos, lab, tau, sigma):
    pbar = ((cos >= tau).astype(int) == lab[:, None]).mean(1)
    cert = pbar > 0.5
    R = np.where(cert, sigma * norm.ppf(np.clip(pbar, 1e-6, 1 - 1e-6)), 0.0)
    return float(cert.mean()), float(R[cert].mean()) if cert.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--arch", nargs="+", default=["vit", "dncnn"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="lfw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)
    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    tau_clean = float(np.quantile(clean[lab == 0], 1 - args.far))
    half = len(idx) // 2
    cal, test = slice(0, half), slice(half, len(idx))
    print(f"[{args.tag}] pairs={len(idx)} draws={args.draws} tau_clean={tau_clean:.4f}\n")

    results = {}
    for arch in args.arch:
        print(f"########## {arch.upper()} ##########")
        rows = []
        for sigma in args.sigmas:
            den = load_den(arch, sigma, device)
            cos = draw_cosines(den, backbone, img1, idx, e2, sigma,
                               args.draws, args.batch, device)
            y = cos.mean(1)
            del den; torch.cuda.empty_cache()

            xc, yc = clean[cal], y[cal]
            xt, yt = clean[test], y[test]
            lc, lt = lab[cal], lab[test]

            # 1. affine (current model)
            A = np.vstack([xc, np.ones_like(xc)]).T
            (a1, b1), *_ = np.linalg.lstsq(A, yc, rcond=None)
            r2_aff = r2(yt, a1 * xt + b1)

            # 2. quadratic
            q = np.polyfit(xc, yc, 2)
            r2_quad = r2(yt, np.polyval(q, xt))

            # 3. two-population
            pop = {}
            for name, mask_c, mask_t in (("gen", lc == 1, lt == 1),
                                         ("imp", lc == 0, lt == 0)):
                Ac = np.vstack([xc[mask_c], np.ones(mask_c.sum())]).T
                (aa, bb), *_ = np.linalg.lstsq(Ac, yc[mask_c], rcond=None)
                pop[name] = {"a": float(aa), "b": float(bb),
                             "r2": r2(yt[mask_t], aa * xt[mask_t] + bb)}
            # pooled held-out R^2 when each pair uses its own population's line
            pred_two = np.where(lt == 1,
                                pop["gen"]["a"] * xt + pop["gen"]["b"],
                                pop["imp"]["a"] * xt + pop["imp"]["b"])
            r2_two = r2(yt, pred_two)

            # thresholds implied by each model, scored on held-out pairs
            taus = {
                "clean": tau_clean,
                "affine": a1 * tau_clean + b1,
                # tau is an FAR-side quantity -> the impostor line is the right one
                "impostor_fit": pop["imp"]["a"] * tau_clean + pop["imp"]["b"],
            }
            accs = {k: cert_stats(cos[test], lt, t, sigma)[0] for k, t in taus.items()}

            print(f"sigma={sigma}:  R2_heldout  affine={r2_aff:+.4f} "
                  f"quad={r2_quad:+.4f} two-pop={r2_two:+.4f}")
            print(f"           gen: a={pop['gen']['a']:.4f} R2={pop['gen']['r2']:+.4f}   "
                  f"imp: a={pop['imp']['a']:.4f} R2={pop['imp']['r2']:+.4f}")
            print(f"           tau: clean={taus['clean']:.4f} affine={taus['affine']:.4f} "
                  f"impfit={taus['impostor_fit']:.4f}")
            print(f"           acc: clean={accs['clean']:.4f} affine={accs['affine']:.4f} "
                  f"impfit={accs['impostor_fit']:.4f}")

            rows.append({"sigma": sigma, "r2_affine": r2_aff, "r2_quad": r2_quad,
                         "r2_twopop": r2_two, "pop": pop,
                         "taus": {k: float(v) for k, v in taus.items()},
                         "accs": accs})
        results[arch] = rows
        print()

    out = args.out or f"results/paper/second_order_{args.tag}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"tag": args.tag, "tau_clean": tau_clean, "arch": results}, f, indent=2)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
