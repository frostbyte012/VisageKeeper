#!/usr/bin/env python3
"""
scripts/train_gallery_denoiser.py -- train the gallery-CONDITIONED denoiser.

Setup
-----
Each training sample is (probe image x, conditioning embedding e_c, is_genuine).
  genuine  (50%): e_c = embedding of ANOTHER image of the SAME identity
  impostor (50%): e_c = embedding of a DIFFERENT identity

The denoiser sees only the NOISY probe plus e_c, and must reconstruct x. The
conditioning is meant to help it resolve ambiguity under heavy noise -- but only
when the claim is true.

The confirmation-bias problem (why the impostor half exists)
------------------------------------------------------------
Conditioning on a claimed identity invites a degenerate solution: pull EVERY
probe toward e_c. That raises genuine similarity and impostor similarity alike,
so verification accuracy does not improve -- FAR just inflates. Two guards:

  1. impostor samples are trained with the SAME reconstruction target x, so
     conditioning on a wrong identity must NOT drag the output toward e_c;
  2. an explicit repulsion term penalises impostor-conditioned outputs whose
     embedding moved TOWARD e_c relative to the unconditional reconstruction.

Evaluation reports genuine and impostor accuracy separately; a method that wins
only on genuine pairs is not a win.

Run:
  python scripts/train_gallery_denoiser.py --sigma 0.5 --epochs 10 \
      --out results/gallery_den_s050.pth
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from frpure.models.backbone import build_backbone
from frpure.attacks.base import cos_sim
from frpure.defenses.gallery_denoiser import GalleryConditionedDenoiser


class IdentityPairs(Dataset):
    """Yields (probe, partner, is_genuine). Partner supplies the conditioning."""

    def __init__(self, root, size, min_imgs=2, limit_ids=None):
        self.size = size
        self.by_id = {}
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            imgs = sorted(glob.glob(os.path.join(d, "*.jpg"))) + \
                sorted(glob.glob(os.path.join(d, "*.png")))
            if len(imgs) >= min_imgs:
                self.by_id[os.path.basename(d)] = imgs
        self.ids = sorted(self.by_id)
        if limit_ids:
            self.ids = self.ids[:limit_ids]
            self.by_id = {k: self.by_id[k] for k in self.ids}
        # flat index of every image belonging to a multi-image identity
        self.items = [(i, p) for i in self.ids for p in self.by_id[i]]
        if not self.items:
            raise RuntimeError(f"no multi-image identities under {root}")
        print(f"{len(self.ids)} identities, {len(self.items)} images "
              f"(>= {min_imgs} images each)")

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()

    def __getitem__(self, k):
        ident, path = self.items[k]
        x = self._load(path)
        genuine = random.random() < 0.5
        if genuine:
            others = [p for p in self.by_id[ident] if p != path]
            partner = self._load(random.choice(others)) if others else x.clone()
        else:
            oid = random.choice([i for i in self.ids if i != ident])
            partner = self._load(random.choice(self.by_id[oid]))
        return x, partner, float(genuine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--lambda_emb", type=float, default=0.5)
    ap.add_argument("--lambda_rep", type=float, default=0.5,
                    help="impostor-repulsion weight (guards confirmation bias)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--limit_ids", type=int, default=None)
    ap.add_argument("--warm_start", action="store_true", default=True,
                    help="init conv stack from the trained unconditional DnCNN")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/gallery_den_s050.pth")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    ds = IdentityPairs(args.aligned_dir, args.size, limit_ids=args.limit_ids)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True)

    den = GalleryConditionedDenoiser(ch=args.ch)
    if args.warm_start:
        # Reuse the trained unconditional DnCNN weights: its conv stack has the
        # same shape as ours (head / depth-2 body blocks / tail), so the model
        # starts as a competent denoiser and training only has to learn what the
        # CONDITIONING adds. FiLM is zero-init (identity), so at step 0 this is
        # exactly the unconditional baseline.
        from scripts.diag_error_overlap import CKPT
        sd = torch.load(CKPT[args.sigma][0], map_location="cpu")
        src = [v for k, v in sd.items() if k.startswith("net.")]
        with torch.no_grad():
            den.head[0].weight.copy_(src[0]); den.head[0].bias.copy_(src[1])
            for bi, blk in enumerate(den.blocks):
                blk[0].weight.copy_(src[2 + 2 * bi]); blk[0].bias.copy_(src[3 + 2 * bi])
            den.tail.weight.copy_(src[-2]); den.tail.bias.copy_(src[-1])
        print(f"warm-started conv stack from {CKPT[args.sigma][0]}")
    den = den.to(device).train()
    opt = torch.optim.Adam(den.parameters(), lr=args.lr)
    print(f"sigma={args.sigma}  lambda_rep={args.lambda_rep}  epochs={args.epochs}")

    hist, step = [], 0
    for epoch in range(args.epochs):
        agg, nagg = {}, 0
        for x, partner, gen in dl:
            x = x.to(device, non_blocking=True)
            partner = partner.to(device, non_blocking=True)
            gen = gen.to(device).view(-1, 1)

            with torch.no_grad():
                e_clean = backbone.embed(x)
                e_cond = backbone.embed(partner)      # the "enrolled" embedding

            z = (x + torch.randn_like(x) * args.sigma).clamp(0, 1)
            y_cond = den(z, e_cond)
            y_uncond = den(z, None)

            loss_mse = F.mse_loss(y_cond, x)
            loss_emb = (1 - cos_sim(backbone.embed(y_cond), e_clean)).mean()

            # repulsion: on IMPOSTOR samples, conditioning must not drag the
            # output toward e_cond relative to the unconditional reconstruction
            ec = backbone.embed(y_cond)
            eu = backbone.embed(y_uncond)
            drift = (cos_sim(ec, e_cond) - cos_sim(eu, e_cond)).view(-1, 1)
            loss_rep = (F.relu(drift) * (1 - gen)).mean()

            loss = loss_mse + args.lambda_emb * loss_emb + args.lambda_rep * loss_rep
            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(den.parameters(), args.grad_clip)
            opt.step()
            step += 1

            gd = float((drift * gen).sum() / gen.sum().clamp(min=1))
            idr = float((drift * (1 - gen)).sum() / (1 - gen).sum().clamp(min=1))
            for k, v in (("mse", float(loss_mse)), ("emb", float(loss_emb)),
                         ("rep", float(loss_rep)), ("drift_gen", gd),
                         ("drift_imp", idr)):
                agg[k] = agg.get(k, 0.0) + v
            nagg += 1
            if step % args.log_every == 0:
                msg = "  ".join(f"{k}={agg[k]/nagg:+.4f}" for k in
                                ("mse", "emb", "rep", "drift_gen", "drift_imp"))
                print(f"ep{epoch} step{step}  {msg}")
                hist.append({"step": step, **{k: agg[k] / nagg for k in agg}})
                agg, nagg = {}, 0
        print(f"-- epoch {epoch} done --")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(den.state_dict(), args.out)      # checkpoint EVERY epoch

    with open(args.out.replace(".pth", "_hist.json"), "w") as f:
        json.dump({"args": vars(args), "history": hist}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
