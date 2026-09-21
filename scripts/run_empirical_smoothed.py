#!/usr/bin/env python3
"""
scripts/run_empirical_smoothed.py  --  empirical robustness of the SMOOTHED verifier
under adaptive EOT-PGD (L2 or Linf). Overlay on the certified curve.

Run (L2 eps=0.5, to compare with certified R>=0.5):
  python scripts/run_empirical_smoothed.py --aligned_dir DATA/lfw_aligned_160 \
      --pairs DATA/pairs.txt --backbone facenet --denoiser_weights results/denoiser_s050.pth \
      --sigma 0.5 --norm l2 --eps 0.5 --n_pairs 200 --out results/lfw_emp_l2_050.csv
CPU self-test:
  python scripts/run_empirical_smoothed.py --synthetic --device cpu --n_pairs 6 \
      --steps 5 --eot 3 --eval_n 60 --sigma 0.1 --eps 0.3
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
from frpure.defenses.smoothing import build_denoiser, SmoothedVerifier
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
    ap.add_argument("--pairs_npz")
    ap.add_argument("--backbone", default="dummy")
    ap.add_argument("--denoiser_weights", default=None)
    ap.add_argument("--bin_path", help="InsightFace .bin pack (cfp_fp/agedb_30/calfw/cplfw)")
    ap.add_argument("--pretrained", default="vggface2", help="facenet weights: vggface2 | casia-webface")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--arch", choices=["dncnn", "vit"], default="dncnn")
    ap.add_argument("--vit_patch", type=int, default=8)
    ap.add_argument("--vit_pos", choices=["2d", "1d"], default="2d")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--norm", choices=["l2", "linf"], default="l2")
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eot", type=int, default=8)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--out", default="results/empirical.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device)

    denoiser = None
    if args.denoiser_weights:
        denoiser = build_denoiser(args.arch, ch=args.ch, patch=args.vit_patch, pos_mode=args.vit_pos).to(device).eval()
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
    print(f"adaptive {args.norm.upper()} EOT-PGD on smoothed verifier  "
          f"(eps={args.eps}, steps={args.steps}, eot={args.eot}, sigma={args.sigma}) "
          f"on {len(idx)} pairs")

    clean_ok = adv_ok = 0
    for j, i in enumerate(idx):
        e_b = e2[j].to(device)
        label = int(sub_same[j])
        if sv.predict(img1[i], e_b, n=args.eval_n) == label:
            clean_ok += 1
        adv = pgd_smoothed(sv, img1[i], e_b, label, args.eps,
                           norm=args.norm, steps=args.steps, eot=args.eot)
        if sv.predict(adv, e_b, n=args.eval_n) == label:
            adv_ok += 1
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(idx)}  (clean_ok={clean_ok} adv_ok={adv_ok})")

    clean_acc = clean_ok / len(idx)
    emp_robust = adv_ok / len(idx)
    print("\n==== EMPIRICAL (adaptive attack on smoothed verifier) ====")
    print(f"norm={args.norm}  eps={args.eps}")
    print(f"clean smoothed accuracy   : {clean_acc:.3f}")
    print(f"empirical robust accuracy : {emp_robust:.3f}")
    print("(must be >= certified accuracy at the same L2 radius)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["norm", "eps", "clean_acc", "empirical_robust_acc"])
        w.writerow([args.norm, args.eps, clean_acc, emp_robust])
    with open(args.out.replace(".csv", ".json"), "w") as f:
        json.dump({"norm": args.norm, "eps": args.eps, "tau": tau,
                   "clean_acc": clean_acc, "empirical_robust_acc": emp_robust}, f, indent=2)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()