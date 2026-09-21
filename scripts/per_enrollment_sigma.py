#!/usr/bin/env python3
"""
scripts/per_enrollment_sigma.py -- choose the smoothing noise level PER ENROLLED
IDENTITY instead of globally.

The idea
--------
R = sigma * Phi^-1(p_bar). A single global sigma is a compromise: raising it
grows the sigma factor but shrinks p_bar, and the best trade-off is NOT the same
for every identity. Distinctive faces keep a high p_bar even under heavy noise
and could be certified at a much larger radius; borderline faces need a smaller
sigma just to stay correct. A global sigma leaves both cases sub-optimal.

Why this is SOUND (the part that matters)
-----------------------------------------
The certificate is a statement about the smoothed function g_sigma applied to a
probe. Choosing sigma from the PROBE would let the attacker steer the choice and
would invalidate the guarantee. Here sigma is a function of the ENROLLED GALLERY
embedding only -- fixed at enrollment time, before any probe (adversarial or
not) is seen, and never influenced by the attacker. Each identity therefore has
its own fixed smoothed classifier g_{sigma(id)}, and Cohen et al. (2019) applies
to each one unchanged. This is exactly the asymmetry face verification has and
generic classification smoothing does not: a clean, trusted, attacker-
independent reference is available at decision time.

What this script measures
-------------------------
For each pair, the vote fraction at EVERY candidate sigma (one pass), then:
  * global-best sigma        (current practice: one sigma for all)
  * per-enrollment sigma     (chosen on CALIBRATION pairs, keyed by a gallery-
                              only statistic, applied to held-out TEST pairs)
  * ORACLE per-pair sigma    (upper bound; uses the answer, so not deployable)

Reporting the oracle first tells us whether the idea is worth engineering at all.

Run:
  python scripts/per_enrollment_sigma.py --n_pairs 300 --draws 300
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


def load_den(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(d)
    for p in d.parameters():
        p.requires_grad_(False)
    return d


@torch.no_grad()
def votes_at_sigma(den, backbone, probe, gallery, sigma, tau, draws, batch, device):
    correct_cos = []
    rem = draws
    while rem > 0:
        m = min(batch, rem)
        z = probe.unsqueeze(0).repeat(m, 1, 1, 1).to(device)
        z = (z + torch.randn_like(z) * sigma).clamp(0, 1)
        correct_cos.append((backbone.embed(den(z)) * gallery.unsqueeze(0)).sum(1).cpu().numpy())
        rem -= m
    return np.concatenate(correct_cos)


def radius(pbar, sigma):
    return sigma * norm.ppf(np.clip(pbar, 1e-6, 1 - 1e-6)) if pbar > 0.5 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--arch", default="vit")
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--tau", type=float, default=None,
                    help="default: certification-optimised tau per sigma")
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/paper/per_enrollment_sigma.json")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    imp = cos_clean[lab == 0]
    tau = args.tau if args.tau is not None else (
        float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3)
    print(f"pairs={len(idx)}  draws={args.draws}  tau={tau:.4f}")
    print(f"sigmas={args.sigmas}\n")

    # gallery-only statistic available at ENROLLMENT time (no probe involved):
    # how distinctive this identity is against the rest of the gallery.
    E2 = torch.nn.functional.normalize(e2, dim=1)
    sim = (E2 @ E2.t()).numpy()
    np.fill_diagonal(sim, -np.inf)
    distinctiveness = -sim.max(1)          # high = far from nearest neighbour

    # pbar[p, s] for every pair p and candidate sigma s
    P, S = len(idx), len(args.sigmas)
    pbar = np.zeros((P, S))
    for si, s in enumerate(args.sigmas):
        dn_w, vit_w = CKPT[s]
        den = load_den(args.arch, vit_w if args.arch == "vit" else dn_w, device)
        for j, i in enumerate(idx):
            cs = votes_at_sigma(den, backbone, img1[i], e2[j].to(device), s,
                                tau, args.draws, args.batch, device)
            pred = (cs >= tau).astype(int)
            pbar[j, si] = (pred == lab[j]).mean()
        print(f"  sigma={s}: mean pbar={pbar[:, si].mean():.4f} "
              f"maj_correct={(pbar[:, si] > 0.5).mean():.4f}")
        del den; torch.cuda.empty_cache()

    R = np.array([[radius(pbar[p, s], args.sigmas[s]) for s in range(S)]
                  for p in range(P)])

    half = P // 2
    cal, test = slice(0, half), slice(half, P)

    # --- baseline: single global sigma, chosen on calibration ---------------
    glob_scores = [R[cal, s].mean() for s in range(S)]
    gs = int(np.argmax(glob_scores))
    global_R = float(R[test, gs].mean())
    global_acc = float((pbar[test, gs] > 0.5).mean())
    print(f"\nglobal best sigma = {args.sigmas[gs]} "
          f"-> test meanR={global_R:.4f} acc={global_acc:.4f}")

    # --- oracle: best sigma per pair (upper bound, not deployable) ----------
    oracle_R = float(R[test].max(1).mean())
    oracle_acc = float((pbar[test].max(1) > 0.5).mean())
    print(f"ORACLE per-pair sigma -> test meanR={oracle_R:.4f} acc={oracle_acc:.4f}"
          f"   (headroom {oracle_R - global_R:+.4f} R)")

    # --- deployable: sigma keyed by gallery distinctiveness -----------------
    # split calibration identities into quantile bins, pick the best sigma per
    # bin, then apply those bins to held-out pairs.
    nb = 3
    qs = np.quantile(distinctiveness[cal], np.linspace(0, 1, nb + 1)[1:-1])
    bin_cal = np.digitize(distinctiveness[cal], qs)
    bin_test = np.digitize(distinctiveness[test], qs)
    bin_sigma = {}
    for b in range(nb):
        m = bin_cal == b
        if m.sum() == 0:
            bin_sigma[b] = gs
            continue
        bin_sigma[b] = int(np.argmax([R[cal][m, s].mean() for s in range(S)]))
    chosen = np.array([bin_sigma[b] for b in bin_test])
    per_R = float(R[test][np.arange(len(chosen)), chosen].mean())
    per_acc = float((pbar[test][np.arange(len(chosen)), chosen] > 0.5).mean())
    print(f"per-enrollment sigma (gallery-keyed) -> test meanR={per_R:.4f} "
          f"acc={per_acc:.4f}")
    print(f"  bins -> sigma: " +
          ", ".join(f"bin{b}={args.sigmas[bin_sigma[b]]}" for b in range(nb)))
    print(f"  vs global: d_R={per_R - global_R:+.4f} "
          f"({(per_R - global_R) / max(global_R, 1e-9) * 100:+.1f}%)  "
          f"d_acc={per_acc - global_acc:+.4f}")

    out = {"tau": tau, "sigmas": args.sigmas, "n_pairs": P, "draws": args.draws,
           "global": {"sigma": args.sigmas[gs], "meanR": global_R, "acc": global_acc},
           "oracle": {"meanR": oracle_R, "acc": oracle_acc},
           "per_enrollment": {"meanR": per_R, "acc": per_acc,
                              "bin_sigma": {str(b): args.sigmas[v]
                                            for b, v in bin_sigma.items()}},
           "headroom_R": oracle_R - global_R}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
