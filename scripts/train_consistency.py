#!/usr/bin/env python3
"""
scripts/train_consistency.py -- draw-consistency regularised denoiser training.

Motivation (mathematical, not just empirical)
---------------------------------------------
Randomized smoothing certifies the radius

    R = sigma * Phi^-1(p_bar)

where p_bar is the probability that a Gaussian draw votes for the correct class.
Phi^-1 is steep near p_bar -> 1, so pushing the vote distribution to concentrate
is the single most direct way to enlarge the certificate.

The standard denoiser loss (MSE + embedding consistency to the CLEAN image)
optimises each draw in isolation. It never asks that two INDEPENDENT draws of
the same face agree with EACH OTHER. Yet the smoothed decision is literally a
majority vote over such draws: if independent draws scatter across the decision
threshold, p_bar sags and R shrinks, even when every individual reconstruction
looks good on average.

So we add a variance-reduction term over paired draws:

    loss = MSE(D(z1), x) + lambda_emb * (1 - cos(emb(D(z1)), emb(x)))
           + lambda_cons * (1 - cos(emb(D(z1)), emb(D(z2))))

with z1, z2 two independent noisy copies of the same x. The third term directly
minimises embedding dispersion across the noise distribution -- exactly the
quantity that sets p_bar.

This is FR-specific: for a classifier you would regularise logits, but face
verification decides on embedding COSINE, so the consistency must be enforced in
embedding space against another draw, not against a label.

Run:
  python scripts/train_consistency.py --sigma 0.5 --arch vit --epochs 12 \
      --lambda_cons 1.0 --out results/denoiser_vit_s050_cons.pth
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from frpure.models.backbone import build_backbone
from frpure.attacks.base import cos_sim
from frpure.defenses.smoothing import build_denoiser
from frpure.defenses.cascade import _disable_fused_attention
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
    ap.add_argument("--arch", choices=["dncnn", "vit"], default="vit")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_emb", type=float, default=0.5)
    ap.add_argument("--lambda_cons", type=float, default=1.0,
                    help="weight on the paired-draw embedding-agreement term")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warm_start", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every_epoch", action="store_true", default=True,
                    help="checkpoint each epoch (a killed run then loses <1 epoch)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/denoiser_vit_s050_cons.pth")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    kw = dict(patch=4, pos_mode="2d") if args.arch == "vit" else {}
    den = build_denoiser(args.arch, ch=32, **kw)
    if args.warm_start and args.sigma in CKPT:
        dn_w, vit_w = CKPT[args.sigma]
        w = vit_w if args.arch == "vit" else dn_w
        den.load_state_dict(torch.load(w, map_location="cpu"))
        print(f"warm-started from {w}")
    den = den.to(device).train()
    _disable_fused_attention(den)
    opt = torch.optim.Adam(den.parameters(), lr=args.lr)

    ds = FaceFolder(args.aligned_dir, args.size, args.limit)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True)
    print(f"{len(ds)} images  arch={args.arch}  sigma={args.sigma}  "
          f"lambda_cons={args.lambda_cons}  epochs={args.epochs}")

    hist, step = [], 0
    for epoch in range(args.epochs):
        agg, nagg = {}, 0
        for x in dl:
            x = x.to(device, non_blocking=True)
            # two INDEPENDENT draws of the same image
            z1 = (x + torch.randn_like(x) * args.sigma).clamp(0, 1)
            z2 = (x + torch.randn_like(x) * args.sigma).clamp(0, 1)
            y1, y2 = den(z1), den(z2)

            with torch.no_grad():
                e_clean = backbone.embed(x)
            e1 = backbone.embed(y1)
            e2 = backbone.embed(y2)

            loss_mse = F.mse_loss(y1, x)
            loss_emb = (1 - cos_sim(e1, e_clean)).mean()
            loss_cons = (1 - cos_sim(e1, e2)).mean()     # <- the new term
            loss = loss_mse + args.lambda_emb * loss_emb + args.lambda_cons * loss_cons

            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(den.parameters(), args.grad_clip)
            opt.step()
            step += 1

            for k, v in (("mse", float(loss_mse)), ("emb", float(loss_emb)),
                         ("cons", float(loss_cons))):
                agg[k] = agg.get(k, 0.0) + v
            nagg += 1
            if step % args.log_every == 0:
                msg = "  ".join(f"{k}={agg[k]/nagg:.4f}" for k in ("mse", "emb", "cons"))
                print(f"ep{epoch} step{step}  loss={float(loss):.4f}  {msg}")
                hist.append({"step": step, **{k: agg[k] / nagg for k in agg}})
                agg, nagg = {}, 0
        print(f"-- epoch {epoch} done --")
        if args.save_every_epoch:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(den.state_dict(), args.out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(den.state_dict(), args.out)
    with open(args.out.replace(".pth", "_hist.json"), "w") as f:
        json.dump({"args": vars(args), "history": hist}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
