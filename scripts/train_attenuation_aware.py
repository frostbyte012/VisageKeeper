#!/usr/bin/env python3
"""
scripts/train_attenuation_aware.py -- train the denoiser to maximise a(sigma).

Rationale
---------
Certified accuracy is set by p_bar, and p_bar is set by where the denoised
cosine distribution sits relative to tau. Our measurements show that smoothing
maps clean cosine to noisy cosine as

    E[cos_noisy] ~= a(sigma) * cos_clean + b(sigma)

so a(sigma) -- the retained identity signal -- is the quantity that actually
controls the certificate. Every denoiser we have trained optimises reconstruction
(pixel MSE + cosine to the clean embedding of the SAME image). Neither term asks
the network to preserve the SEPARATION between identities, which is what a
measures.

Concretely: MSE and embedding-cosine are both computed per image against its own
clean target. A denoiser can score well on both while compressing all embeddings
toward a common mean -- that shrinks genuine and impostor cosines alike, lowers
a, and costs certified accuracy. This objective adds a term that explicitly
resists that compression.

Objective
---------
    loss = MSE(D(z), x)
         + lambda_emb  * (1 - cos(emb(D(z)), emb(x)))
         + lambda_att  * attenuation_penalty

The penalty operates on a BATCH of distinct identities. For all pairs (i, j) in
the batch we compare the clean cosine c_ij = cos(emb(x_i), emb(x_j)) with the
denoised cosine d_ij = cos(emb(D(z_i)), emb(D(z_j))), and penalise shrinkage of
the spread. Fitting a per-batch slope a_hat = <d, c> / <c, c> over the off-
diagonal entries, the penalty is  (1 - a_hat)^2  -- minimised when the denoised
cosine geometry matches the clean geometry one-for-one.

This is trained on the pairwise structure, not on labels, so it needs no
identity supervision beyond "these are different images".

Run:
  python scripts/train_attenuation_aware.py --sigma 0.75 --arch vit --epochs 8 \
      --lambda_att 1.0 --out results/denoiser_vit_s075_att.pth
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


def offdiag(M):
    n = M.shape[0]
    return M[~torch.eye(n, dtype=torch.bool, device=M.device)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--arch", choices=["vit", "dncnn"], default="vit")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=24,
                    help="also the number of identities in the pairwise penalty")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_emb", type=float, default=0.5)
    ap.add_argument("--lambda_att", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warm_start", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/denoiser_att.pth")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    kw = dict(patch=4, pos_mode="2d") if args.arch == "vit" else {}
    den = build_denoiser(args.arch, ch=32, **kw)
    if args.warm_start and args.sigma in CKPT:
        w = CKPT[args.sigma][1 if args.arch == "vit" else 0]
        den.load_state_dict(torch.load(w, map_location="cpu"))
        print(f"warm-started from {w}")
    den = den.to(device).train()
    _disable_fused_attention(den)
    opt = torch.optim.Adam(den.parameters(), lr=args.lr)

    ds = FaceFolder(args.aligned_dir, args.size, args.limit)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True)
    print(f"{len(ds)} images  arch={args.arch}  sigma={args.sigma}  "
          f"lambda_att={args.lambda_att}  epochs={args.epochs}")

    hist, step = [], 0
    for epoch in range(args.epochs):
        agg, nagg = {}, 0
        for x in dl:
            x = x.to(device, non_blocking=True)
            z = (x + torch.randn_like(x) * args.sigma).clamp(0, 1)
            y = den(z)

            with torch.no_grad():
                e_clean = backbone.embed(x)
            e_den = backbone.embed(y)

            loss_mse = F.mse_loss(y, x)
            loss_emb = (1 - cos_sim(e_den, e_clean)).mean()

            # pairwise geometry: clean vs denoised cosine between DIFFERENT images
            ec = F.normalize(e_clean, dim=1)
            ed = F.normalize(e_den, dim=1)
            c = offdiag(ec @ ec.t())
            dcos = offdiag(ed @ ed.t())
            # least-squares slope of denoised-vs-clean cosine, no intercept
            a_hat = (dcos * c).sum() / (c * c).sum().clamp(min=1e-8)
            loss_att = (1.0 - a_hat) ** 2

            loss = loss_mse + args.lambda_emb * loss_emb + args.lambda_att * loss_att
            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(den.parameters(), args.grad_clip)
            opt.step()
            step += 1

            for k, v in (("mse", float(loss_mse)), ("emb", float(loss_emb)),
                         ("a_hat", float(a_hat)), ("att", float(loss_att))):
                agg[k] = agg.get(k, 0.0) + v
            nagg += 1
            if step % args.log_every == 0:
                msg = "  ".join(f"{k}={agg[k]/nagg:.4f}" for k in
                                ("mse", "emb", "a_hat", "att"))
                print(f"ep{epoch} step{step}  {msg}")
                hist.append({"step": step, **{k: agg[k] / nagg for k in agg}})
                agg, nagg = {}, 0
        print(f"-- epoch {epoch} done --")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(den.state_dict(), args.out)   # checkpoint every epoch

    with open(args.out.replace(".pth", "_hist.json"), "w") as f:
        json.dump({"args": vars(args), "history": hist}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
