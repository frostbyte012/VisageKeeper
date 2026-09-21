#!/usr/bin/env python3
"""
scripts/profile_denoisers.py -- cost side of the dual-architecture claim.

Reports, per 160x160 image: parameters, MACs/FLOPs (thop), and measured GPU
latency for DnCNN, ViT, and the cascade at a given cheap-fraction. Smoothing
multiplies all of these by n (the number of noise draws), so the per-pass number
is what actually decides deployability.

Run:  python scripts/profile_denoisers.py --cheap_frac 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from frpure.defenses.smoothing import build_denoiser


def macs_params(model, size, device):
    from thop import profile
    x = torch.randn(1, 3, size, size, device=device)
    macs, params = profile(model, inputs=(x,), verbose=False)
    return float(macs), float(params)


@torch.no_grad()
def latency_ms(model, size, device, batch=32, iters=50, warmup=10):
    x = torch.randn(batch, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        model(x)
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / batch   # ms per image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cheap_frac", type=float, default=0.5,
                    help="measured cascade cheap fraction, for the blended cost")
    ap.add_argument("--n_draws", type=int, default=1000,
                    help="smoothing draws per decision, for the per-decision cost")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/paper/denoiser_cost.json")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    models = {
        "DnCNN": build_denoiser("dncnn", ch=32).to(device).eval(),
        "ViT": build_denoiser("vit", patch=4, pos_mode="2d").to(device).eval(),
    }

    rows = {}
    for name, m in models.items():
        macs, params = macs_params(m, args.size, device)
        lat = latency_ms(m, args.size, device, args.batch)
        rows[name] = {"params_M": params / 1e6, "GMACs": macs / 1e9,
                      "GFLOPs": 2 * macs / 1e9, "latency_ms_per_img": lat}
        print(f"{name:>8}: {params/1e6:8.4f} M params  {2*macs/1e9:8.4f} GFLOPs  "
              f"{lat:7.4f} ms/img")

    f = args.cheap_frac
    # cascade always pays DnCNN; pays ViT only on the escalated fraction
    casc = {
        "cheap_frac": f,
        "GFLOPs": rows["DnCNN"]["GFLOPs"] + (1 - f) * rows["ViT"]["GFLOPs"],
        "latency_ms_per_img": rows["DnCNN"]["latency_ms_per_img"]
                              + (1 - f) * rows["ViT"]["latency_ms_per_img"],
        "params_M": rows["DnCNN"]["params_M"] + rows["ViT"]["params_M"],
    }
    rows["Cascade"] = casc
    print(f"{'Cascade':>8}: {casc['params_M']:8.4f} M params  {casc['GFLOPs']:8.4f} GFLOPs  "
          f"{casc['latency_ms_per_img']:7.4f} ms/img   (cheap_frac={f:.2f})")

    print(f"\nper smoothed decision (n={args.n_draws} draws):")
    for name in ("DnCNN", "ViT", "Cascade"):
        print(f"  {name:>8}: {rows[name]['GFLOPs']*args.n_draws:9.1f} GFLOPs  "
              f"{rows[name]['latency_ms_per_img']*args.n_draws/1000:7.3f} s")
    sav = 1 - rows["Cascade"]["GFLOPs"] / rows["ViT"]["GFLOPs"]
    print(f"\ncascade vs ViT-only: {sav*100:+.1f}% FLOPs")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"size": args.size, "n_draws": args.n_draws, "rows": rows,
                   "flop_saving_vs_vit": sav}, fh, indent=2)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
