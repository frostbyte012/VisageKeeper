#!/usr/bin/env python3
"""
scripts/train_moe.py -- joint DnCNN+ViT mixture-of-experts denoiser training.

Warm-starts both experts from the existing single-architecture checkpoints (so
we GROW the measured niche rather than relearning denoising from scratch), then
fine-tunes them together with a learned gate under:

    loss = MSE + lambda_emb * (1 - cos(emb(D(noisy)), emb(clean)))
           + lambda_dec * corr(expert errors)      <- widen the niche
           + lambda_bal * usage-collapse penalty   <- keep both experts alive

Run:
  python scripts/train_moe.py --aligned_dir DATA/lfw_aligned_160 --sigma 0.5 \
      --epochs 12 --lambda_dec 0.3 --out results/moe_s050_dec03.pth
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser
from frpure.defenses.moe_denoiser import MoEDenoiser, moe_losses
from scripts.diag_error_overlap import CKPT


class FaceFolder(Dataset):
    def __init__(self, root, size, limit=None):
        self.paths = []
        for ext in ("png", "jpg", "jpeg"):
            self.paths += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
        self.paths.sort()
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise RuntimeError(f"no images under {root}")
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size))
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gate_lr", type=float, default=1e-3)
    ap.add_argument("--lambda_emb", type=float, default=0.5)
    ap.add_argument("--lambda_dec", type=float, default=0.3)
    ap.add_argument("--lambda_bal", type=float, default=0.1)
    ap.add_argument("--lambda_anchor", type=float, default=1.0,
                    help="keeps each expert individually competent; 0 lets the "
                         "decorrelation term degrade both (measured failure)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warm_start", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/moe_s050.pth")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)
    for p in backbone.model.parameters() if hasattr(backbone, "model") else []:
        p.requires_grad_(False)

    cheap = build_denoiser("dncnn", ch=32)
    exp = build_denoiser("vit", patch=4, pos_mode="2d")
    if args.warm_start and args.sigma in CKPT:
        dn_w, vit_w = CKPT[args.sigma]
        cheap.load_state_dict(torch.load(dn_w, map_location="cpu"))
        exp.load_state_dict(torch.load(vit_w, map_location="cpu"))
        print(f"warm-started experts from {dn_w} / {vit_w}")

    moe = MoEDenoiser(cheap, exp).to(device).train()
    gate_params = list(moe.gate.parameters())
    gate_ids = {id(p) for p in gate_params}
    expert_params = [p for p in moe.parameters() if id(p) not in gate_ids]
    opt = torch.optim.Adam([
        {"params": expert_params, "lr": args.lr},
        {"params": gate_params, "lr": args.gate_lr},
    ])

    ds = FaceFolder(args.aligned_dir, args.size, args.limit)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True)
    print(f"{len(ds)} images  sigma={args.sigma}  lambda_dec={args.lambda_dec} "
          f"lambda_bal={args.lambda_bal}  epochs={args.epochs}")

    hist, step = [], 0
    for epoch in range(args.epochs):
        agg = {}
        for x in dl:
            x = x.to(device, non_blocking=True)
            noisy = (x + torch.randn_like(x) * args.sigma).clamp(0, 1)
            out, yc, ye, p = moe(noisy, return_parts=True)
            loss, parts = moe_losses(out, yc, ye, p, x, backbone,
                                     lambda_emb=args.lambda_emb,
                                     lambda_dec=args.lambda_dec,
                                     lambda_bal=args.lambda_bal,
                                     lambda_anchor=args.lambda_anchor)
            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(moe.parameters(), args.grad_clip)
            opt.step()
            step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            if step % args.log_every == 0:
                n = args.log_every
                msg = "  ".join(f"{k}={agg[k]/n:.4f}" for k in
                                ("mse", "emb", "corr", "bal", "usage_vit",
                                 "mse_dncnn", "mse_vit"))
                print(f"ep{epoch} step{step}  loss={float(loss):.4f}  {msg}")
                hist.append({"step": step, **{k: agg[k] / n for k in agg}})
                agg = {}
        print(f"-- epoch {epoch} done --")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(moe.state_dict(), args.out)
    with open(args.out.replace(".pth", "_hist.json"), "w") as f:
        json.dump({"args": vars(args), "history": hist}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
