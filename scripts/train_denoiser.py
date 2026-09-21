#!/usr/bin/env python3
"""
scripts/train_denoiser.py  --  trains the Gaussian denoiser for denoised smoothing.

Pure regression, NO attacker in the loop:
  noisy = x + eps, eps ~ N(0, sigma^2 I)
  loss  = MSE(D(noisy), x) + lambda_emb * (1 - cos(emb(D(noisy)), emb(x)))
IMPORTANT: train at the SAME sigma you will certify with.

Run (real):
  python scripts/train_denoiser.py --aligned_dir DATA/lfw_aligned_160 --backbone facenet \
      --size 160 --sigma 0.5 --epochs 5 --out results/denoiser_s050.pth
CPU self-test:
  python scripts/train_denoiser.py --synthetic --device cpu --iters 3
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from frpure.models.backbone import build_backbone
from frpure.attacks.base import cos_sim
from frpure.defenses.smoothing import build_denoiser


class FaceFolder(Dataset):
    def __init__(self, root, size):
        self.paths = []
        for ext in ("png", "jpg", "jpeg"):
            self.paths += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
        self.paths.sort()
        if not self.paths:
            raise RuntimeError(f"no images under {root}")
        self.size = size

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size))
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


class SyntheticFaces(Dataset):
    def __init__(self, n=32, size=64):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(n, 3, size, size, generator=g)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def train(args):
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    
    bb_kw = {"device": device}
    if args.backbone == "facenet":
        bb_kw["pretrained"] = args.pretrained
    backbone = build_backbone(args.backbone, **bb_kw)

    ds = SyntheticFaces(size=args.size) if args.synthetic else FaceFolder(args.aligned_dir, args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    denoiser = build_denoiser(args.arch, ch=args.ch, patch=args.vit_patch, pos_mode=args.vit_pos).to(device).train()
    opt = torch.optim.Adam(denoiser.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        for x in dl:
            x = x.to(device)
            noisy = (x + torch.randn_like(x) * args.sigma).clamp(0, 1)
            xhat = denoiser(noisy)
            loss_mse = F.mse_loss(xhat, x)
            with torch.no_grad():
                e_clean = backbone.embed(x)
            loss_emb = (1 - cos_sim(backbone.embed(xhat), e_clean)).mean()
            loss = loss_mse + args.lambda_emb * loss_emb

            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), args.grad_clip)
            opt.step()
            step += 1
            if step % args.log_every == 0 or args.synthetic:
                print(f"ep{epoch} step{step}  loss={loss.item():.4f} "
                      f"(mse={loss_mse.item():.4f} emb={loss_emb.item():.4f})")
            if args.synthetic and step >= args.iters:
                break
        if args.synthetic and step >= args.iters:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(denoiser.state_dict(), args.out)
    print(f"saved denoiser -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aligned_dir")
    ap.add_argument("--backbone", default="dummy")
    ap.add_argument("--pretrained", default="vggface2", help="facenet weights: vggface2 | casia-webface")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.5, help="Gaussian noise std (match certify)")
    ap.add_argument("--ch", type=int, default=32, help="denoiser width (FPGA-friendly knob)")
    ap.add_argument("--vit_patch", type=int, default=8)
    ap.add_argument("--vit_pos", choices=["2d", "1d"], default="2d")
    ap.add_argument("--arch", choices=["dncnn", "vit"], default="dncnn",
                    help="denoiser architecture: DnCNN-style convs or ViT bottleneck")
    ap.add_argument("--lambda_emb", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="max grad norm; 0 = off")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default="results/denoiser.pth")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.synthetic and args.backbone == "dummy":
        args.backbone, args.size = "facenet", 160
    train(args)


if __name__ == "__main__":
    main()