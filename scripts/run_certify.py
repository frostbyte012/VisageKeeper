#!/usr/bin/env python3
"""
scripts/run_certify.py  --  evaluate CERTIFIED robustness of the smoothed verifier.

Reports over a balanced subset of pairs:
  * smoothed accuracy (abstain = wrong)   * abstention rate
  * certified accuracy @ L2 radius grid   * mean certified radius
tau is calibrated on CLEAN cosines at --far unless --tau is given.
Match --sigma to the denoiser's training sigma.

Run (real):
  python scripts/run_certify.py --aligned_dir DATA/lfw_aligned_160 --pairs DATA/pairs.txt \
      --backbone facenet --denoiser_weights results/denoiser_s050.pth \
      --sigma 0.5 --n_pairs 500 --n 1000 --out results/lfw_certify.csv
CPU self-test:
  python scripts/run_certify.py --synthetic --device cpu --n_pairs 8 --n0 20 --n 80 --sigma 0.1
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

RADII = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


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
    out = []
    for i in range(0, len(imgs), batch):
        out.append(backbone.embed(imgs[i:i + batch].to(dev)).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aligned_dir"); ap.add_argument("--pairs")
    ap.add_argument("--pairs_npz")
    ap.add_argument("--backbone", default="dummy")
    ap.add_argument("--denoiser_weights", default=None)
    ap.add_argument("--bin_path", help="InsightFace .bin pack (cfp_fp/agedb_30/calfw/cplfw)")
    ap.add_argument("--pretrained", default="vggface2", help="facenet weights: vggface2 | casia-webface")
    ap.add_argument("--onnx_path", help="path to .onnx FR model (for --backbone onnx)")
    ap.add_argument("--onnx_bgr", action="store_true", help="flip channel order if clean acc is near chance")
    ap.add_argument("--ch", type=int, default=32, help="must match the trained denoiser")
    ap.add_argument("--arch", choices=["dncnn", "vit"], default="dncnn")
    ap.add_argument("--vit_patch", type=int, default=8)
    ap.add_argument("--vit_pos", choices=["2d", "1d"], default="2d")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--n0", type=int, default=100)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--n_pairs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="results/certify.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    bb_kw = {"device": device}
    if args.backbone == "facenet":
        bb_kw["pretrained"] = args.pretrained
    if args.backbone == "onnx":
        bb_kw["onnx_path"] = args.onnx_path
        bb_kw["bgr"] = args.onnx_bgr
    backbone = build_backbone(args.backbone, **bb_kw)

    denoiser = None
    if args.denoiser_weights:
        denoiser = build_denoiser(args.arch, ch=args.ch, patch=args.vit_patch, pos_mode=args.vit_pos).to(device).eval()
        denoiser.load_state_dict(torch.load(args.denoiser_weights, map_location=device))
        for p in denoiser.parameters():
            p.requires_grad_(False)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    print(f"certifying {len(idx)} pairs  (sigma={args.sigma}, n={args.n}, "
          f"denoiser={'yes' if denoiser else 'NONE'})")

    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy()
    sub_same = same[idx]
    if args.tau is not None:
        tau = args.tau
    else:
        imp = cos_clean[sub_same == 0]
        tau = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3
    print(f"tau = {tau:.4f}  (clean acc @ tau = "
          f"{np.mean((cos_clean >= tau).astype(int) == sub_same):.3f})")

    sv = SmoothedVerifier(backbone, denoiser, sigma=args.sigma, tau=tau, device=device)

    preds, radii, labels = [], [], []
    for j, i in enumerate(idx):
        e_b = e2[j].to(device)
        pred, R = sv.certify(img1[i], e_b, n0=args.n0, n=args.n,
                             alpha=args.alpha, batch=args.batch)
        preds.append(pred); radii.append(R); labels.append(int(sub_same[j]))
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(idx)} certified")

    preds = np.array(preds); radii = np.array(radii); labels = np.array(labels)
    not_abstain = preds != -1
    correct = (preds == labels) & not_abstain

    smoothed_acc = float(correct.mean())
    abstain_rate = float((~not_abstain).mean())
    cert_acc = {r: float((correct & (radii >= r)).mean()) for r in RADII}
    mean_R = float(radii[correct].mean()) if correct.any() else 0.0

    print("\n==== CERTIFICATION RESULTS ====")
    print(f"smoothed accuracy   : {smoothed_acc:.3f}")
    print(f"abstention rate     : {abstain_rate:.3f}")
    print(f"mean certified R    : {mean_R:.3f}  (over certified-correct)")
    print("certified accuracy @ L2 radius:")
    for r in RADII:
        print(f"   R >= {r:<4}: {cert_acc[r]:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = [{"radius": r, "certified_accuracy": cert_acc[r]} for r in RADII]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["radius", "certified_accuracy"])
        w.writeheader(); w.writerows(rows)
    with open(args.out.replace(".csv", ".json"), "w") as f:
        json.dump({"sigma": args.sigma, "tau": tau, "n": args.n,
                   "smoothed_accuracy": smoothed_acc, "abstention_rate": abstain_rate,
                   "mean_radius": mean_R, "certified_accuracy": cert_acc}, f, indent=2)
    print(f"\nwrote -> {args.out} (+ .json)")


if __name__ == "__main__":
    main()