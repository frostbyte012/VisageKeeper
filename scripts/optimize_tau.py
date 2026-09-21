#!/usr/bin/env python3
"""
scripts/optimize_tau.py -- calibrate the verification threshold FOR THE
CERTIFICATE instead of for clean FAR.

The idea
--------
Randomized smoothing certifies R = sigma * Phi^-1(p_bar), where p_bar is the
fraction of Gaussian draws voting for the correct decision. In FACE
VERIFICATION the vote is  1[cos(emb(D(x+eps)), e_gallery) >= tau],  so p_bar --
and therefore the whole certificate -- depends on tau.

Standard practice (and our earlier runs) sets tau from CLEAN cosines at a target
FAR. That optimises a different objective: clean operating point, not certified
robustness. Nothing forces the two to coincide, and measurement says they do not.

This is a degree of freedom that exists only in verification. Classification
smoothing (Cohen et al. 2019, Salman et al. 2020) decides by argmax and has no
threshold to tune, which is why the FR literature has inherited the clean-FAR
tau without questioning it. Re-tuning it costs ZERO extra inference compute.

Protocol (leakage-free)
-----------------------
Split pairs into CALIBRATION and TEST halves. Sweep tau on CALIBRATION only,
pick the optimum under the chosen objective, then report on TEST. We report both
objectives because they trade off:
    --objective acc     maximise certified/majority accuracy
    --objective radius  maximise mean certified radius at non-inferior accuracy

Run:
  python scripts/optimize_tau.py --sigma 0.5 --n_pairs 400 --draws 400
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


def load_den(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(d)
    for p in d.parameters():
        p.requires_grad_(False)
    return d


@torch.no_grad()
def collect_cosines(den, backbone, img1, idx, e2, sigma, draws, batch, device):
    """Cosine of every noisy draw against its gallery embedding -> (P, draws).

    Collected ONCE; every candidate tau is then evaluated offline on the same
    samples, so the sweep costs one forward pass over the data.
    """
    out = []
    for j, i in enumerate(idx):
        g = e2[j].to(device).unsqueeze(0)
        cs = []
        rem = draws
        while rem > 0:
            m = min(batch, rem)
            z = img1[i].unsqueeze(0).repeat(m, 1, 1, 1).to(device)
            z = (z + torch.randn_like(z) * sigma).clamp(0, 1)
            cs.append((backbone.embed(den(z)) * g).sum(1).cpu().numpy())
            rem -= m
        out.append(np.concatenate(cs))
    return np.stack(out)


def stats_at_tau(cos, lab, tau, sigma):
    pred = (cos >= tau).astype(int)
    pbar = (pred == lab[:, None]).mean(1)
    cert = pbar > 0.5
    R = np.where(cert, sigma * norm.ppf(np.clip(pbar, 1e-6, 1 - 1e-6)), 0.0)
    return {"pbar": float(pbar.mean()),
            "acc": float(cert.mean()),
            "meanR": float(R[cert].mean()) if cert.any() else 0.0,
            "meanR_all": float(R.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--arch", default="vit")
    ap.add_argument("--weights", default="results/denoiser_vit_s050_v3.pth")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=400)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    imp = cos_clean[lab == 0]
    tau_clean = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3

    den = load_den(args.arch, args.weights, device)
    print(f"sigma={args.sigma}  pairs={len(idx)}  draws={args.draws}")
    print(f"tau_clean (FAR={args.far}) = {tau_clean:.4f}\ncollecting draw cosines ...")
    cos = collect_cosines(den, backbone, img1, idx, e2, args.sigma,
                          args.draws, args.batch, device)

    half = len(idx) // 2
    cal, test = slice(0, half), slice(half, len(idx))
    grid = np.arange(0.10, 0.70, 0.0125)

    # calibration sweep
    rows = []
    for t in grid:
        s = stats_at_tau(cos[cal], lab[cal], t, args.sigma)
        rows.append({"tau": float(t), **s})

    base_cal = stats_at_tau(cos[cal], lab[cal], tau_clean, args.sigma)
    # objective 1: max accuracy (ties -> larger radius)
    best_acc = max(rows, key=lambda r: (r["acc"], r["meanR"]))
    # objective 2: max radius subject to accuracy >= clean-tau accuracy
    ok = [r for r in rows if r["acc"] >= base_cal["acc"] - 1e-9]
    best_rad = max(ok, key=lambda r: r["meanR"]) if ok else best_acc

    print(f"\n[calib] tau_clean={tau_clean:.4f}: acc={base_cal['acc']:.4f} "
          f"meanR={base_cal['meanR']:.4f} pbar={base_cal['pbar']:.4f}")
    print(f"[calib] best-acc    tau={best_acc['tau']:.4f}: acc={best_acc['acc']:.4f} "
          f"meanR={best_acc['meanR']:.4f}")
    print(f"[calib] best-radius tau={best_rad['tau']:.4f}: acc={best_rad['acc']:.4f} "
          f"meanR={best_rad['meanR']:.4f}")

    # held-out evaluation
    print("\n=== TEST (held out) ===")
    res = {}
    for name, t in (("tau_clean", tau_clean),
                    ("tau_best_acc", best_acc["tau"]),
                    ("tau_best_radius", best_rad["tau"])):
        s = stats_at_tau(cos[test], lab[test], t, args.sigma)
        res[name] = {"tau": float(t), **s}
        print(f"  {name:>16} (tau={t:.4f}): acc={s['acc']:.4f} "
              f"meanR={s['meanR']:.4f} pbar={s['pbar']:.4f}")

    b = res["tau_clean"]
    for k in ("tau_best_acc", "tau_best_radius"):
        d_acc = res[k]["acc"] - b["acc"]
        d_R = res[k]["meanR"] - b["meanR"]
        print(f"  {k} vs tau_clean: d_acc={d_acc:+.4f}  d_R={d_R:+.4f} "
              f"({d_R / b['meanR'] * 100:+.1f}% radius)")
    n_test = len(idx) - half
    se = float(np.sqrt(b["acc"] * (1 - b["acc"]) / n_test))
    print(f"  (1 s.e. on {n_test} test pairs ~ {se:.4f})")

    out = {"sigma": args.sigma, "tau_clean": tau_clean, "n_pairs": len(idx),
           "draws": args.draws, "calib_sweep": rows, "test": res, "se": se}
    path = args.out or f"results/paper/tau_opt_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
