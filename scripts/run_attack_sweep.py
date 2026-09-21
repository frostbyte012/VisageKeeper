#!/usr/bin/env python3
"""
scripts/run_attack_sweep.py  --  ATTACK-STRENGTH SWEEP of the smoothed verifier.
Runs the adaptive EOT-PGD attack at a RANGE of eps on one dataset -> an
empirical-robustness-vs-eps curve (one CSV row per eps). Overlay on the certified
curve: empirical must sit >= certified at the same L2 radius; the gap = how tight
your certificate is across budgets.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import GaussianDenoiser, SmoothedVerifier
from frpure.attacks.smoothed import pgd_smoothed


def load_pairs(args):
    if args.synthetic:
        g = torch.Generator().manual_seed(1)
        n_id, n_pairs = 40, 120
        protos = torch.rand(n_id, 3, 64, 64, generator=g)
        jit = lambda p: (p + torch.randn(p.shape, generator=g) * 0.04).clamp(0, 1)
        a1, a2, same = [], [], []
        for k in range(n_pairs):
            i = k % n_id
            a1.append(jit(protos[i]))
            if k % 2 == 0:
                a2.append(jit(protos[i])); same.append(1)
            else:
                a2.append(jit(protos[(i + 1) % n_id])); same.append(0)
        return torch.stack(a1), torch.stack(a2), np.array(same)
    if args.bin_path:
        from frpure.data.bin_pairs import load_bin_pairs
        return load_bin_pairs(args.bin_path, size=160)
    if args.pairs_npz:
        d = np.load(args.pairs_npz)
        i1 = torch.from_numpy(d["img1"]).permute(0, 3, 1, 2).float() / 255
        i2 = torch.from_numpy(d["img2"]).permute(0, 3, 1, 2).float() / 255
        return i1, i2, d["same"].astype(int)
    from frpure.data.lfw import LFWPairs
    ds = LFWPairs(args.aligned_dir, args.pairs)
    i1 = torch.stack([ds[i].img1 for i in range(len(ds))])
    i2 = torch.stack([ds[i].img2 for i in range(len(ds))])
    same = np.array([int(ds.pairs[i][2]) for i in range(len(ds))])
    return i1, i2, same


def balanced_subset(same, n_pairs, seed=0):
    rng = np.random.default_rng(seed)
    gen = np.where(same == 1)[0]; imp = np.where(same == 0)[0]
    rng.shuffle(gen); rng.shuffle(imp)
    half = n_pairs // 2
    idx = np.concatenate([gen[:half], imp[:n_pairs - half]])
    rng.shuffle(idx)
    return idx


@torch.no_grad()
def embed_all(backbone, imgs, batch=128):
    dev = getattr(backbone, "device", "cpu")
    return torch.cat([backbone.embed(imgs[i:i + batch].to(dev)).cpu()
                      for i in range(0, len(imgs), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aligned_dir"); ap.add_argument("--pairs")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--backbone", default="dummy")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--denoiser_weights", default=None)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--norm", choices=["l2", "linf"], default="l2")
    ap.add_argument("--eps", default="0,0.25,0.5,0.75,1.0,1.5",
                    help="comma-separated list of attack budgets to sweep")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eot", type=int, default=8)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--out", default="results/attack_sweep.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    eps_list = [float(x) for x in str(args.eps).split(",")]
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    bb_kw = {"device": device}
    if args.backbone == "facenet":
        bb_kw["pretrained"] = args.pretrained
    backbone = build_backbone(args.backbone, **bb_kw)

    denoiser = None
    if args.denoiser_weights:
        denoiser = GaussianDenoiser(ch=args.ch).to(device).eval()
        denoiser.load_state_dict(torch.load(args.denoiser_weights, map_location=device))
        for p in denoiser.parameters():
            p.requires_grad_(False)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy(); sub_same = same[idx]
    if args.tau is not None:
        tau = args.tau
    else:
        imp = cos_clean[sub_same == 0]
        tau = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3

    sv = SmoothedVerifier(backbone, denoiser, sigma=args.sigma, tau=tau, device=device)

    clean_ok = 0
    for j, i in enumerate(idx):
        if sv.predict(img1[i], e2[j].to(device), n=args.eval_n) == int(sub_same[j]):
            clean_ok += 1
    clean_acc = clean_ok / len(idx)
    print(f"{args.norm.upper()} sweep on {len(idx)} pairs (sigma={args.sigma}, "
          f"steps={args.steps}, eot={args.eot})  clean={clean_acc:.3f}")

    rows = []
    for eps in eps_list:
        if eps == 0.0:
            robust = clean_acc
        else:
            ok = 0
            for j, i in enumerate(idx):
                e_b = e2[j].to(device); label = int(sub_same[j])
                adv = pgd_smoothed(sv, img1[i], e_b, label, eps,
                                   norm=args.norm, steps=args.steps, eot=args.eot)
                if sv.predict(adv, e_b, n=args.eval_n) == label:
                    ok += 1
            robust = ok / len(idx)
        print(f"  eps={eps:<6}  empirical_robust_acc={robust:.3f}")
        rows.append({"norm": args.norm, "eps": eps,
                     "clean_acc": clean_acc, "empirical_robust_acc": robust})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["norm", "eps", "clean_acc", "empirical_robust_acc"])
        w.writeheader(); w.writerows(rows)
    with open(args.out.replace(".csv", ".json"), "w") as f:
        json.dump({"tau": tau, "sigma": args.sigma, "rows": rows}, f, indent=2)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()