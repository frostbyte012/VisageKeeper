#!/usr/bin/env python3
"""
hls_vit/export_vit_weights.py -- dump the trained ViT denoiser to the flat
binary layout that vit_denoiser_hls.cpp expects, plus a golden I/O pair.

Unlike the DnCNN (0.02 M weights -> denoiser_weights.h static ROM), the ViT has
1.80 M weights = 3.5 MB at 16-bit, well over the ZCU104's 11 Mb of BRAM. So the
weights go to a .bin that the kernel streams from DDR through an m_axi port.

Layout must match LAYER_W_SIZE in vit_denoiser_hls.h exactly:
    embed_w, embed_b
    per layer: in_proj_w, in_proj_b, out_proj_w, out_proj_b,
               norm1_w, norm1_b, lin1_w, lin1_b, lin2_w, lin2_b,
               norm2_w, norm2_b
    unembed_w, unembed_b

Run:
    python hls_vit/export_vit_weights.py \
        --ckpt results/denoiser_vit_s050_v3.pth --outdir hls_vit
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frpure.defenses.vit_denoiser import ViTDenoiser   # noqa: E402

DEPTH = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/denoiser_vit_s050_v3.pth")
    ap.add_argument("--outdir", default="hls_vit")
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sd = torch.load(args.ckpt, map_location="cpu")
    # Checkpoints saved from ViTGaussianDenoiser hold the core as `.core`.
    if any(k.startswith("core.") for k in sd):
        sd = {k[len("core."):]: v for k, v in sd.items() if k.startswith("core.")}

    patch = sd["embed.weight"].shape[-1]
    dim = sd["embed.weight"].shape[0]
    d_ff = sd["encoder.layers.0.linear1.weight"].shape[0]
    print(f"patch={patch} dim={dim} depth={DEPTH} d_ff={d_ff}")

    net = ViTDenoiser(patch=patch, dim=dim, depth=DEPTH,
                      heads=6, mlp_ratio=d_ff / dim)
    net.load_state_dict(sd)
    net.eval()

    # ---- flatten in the kernel's expected order --------------------------
    blobs = [sd["embed.weight"].flatten(), sd["embed.bias"].flatten()]
    for l in range(DEPTH):
        p = f"encoder.layers.{l}."
        for key in ("self_attn.in_proj_weight", "self_attn.in_proj_bias",
                    "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                    "norm1.weight", "norm1.bias",
                    "linear1.weight", "linear1.bias",
                    "linear2.weight", "linear2.bias",
                    "norm2.weight", "norm2.bias"):
            blobs.append(sd[p + key].flatten())
    blobs += [sd["unembed.weight"].flatten(), sd["unembed.bias"].flatten()]

    flat = torch.cat(blobs).float().numpy().astype("<f4")
    wpath = os.path.join(args.outdir, "vit_weights.bin")
    flat.tofile(wpath)
    print(f"wrote {wpath}: {flat.size} floats ({flat.nbytes/1e6:.2f} MB)")

    # sanity: does the flat size match the header's arithmetic?
    n_tok = (args.size // patch) ** 2
    layer_sz = (3*dim*dim + 3*dim + dim*dim + dim + 2*dim
                + d_ff*dim + d_ff + dim*d_ff + dim + 2*dim)
    expect = (dim*3*patch*patch + dim) + DEPTH*layer_sz + (dim*3*patch*patch + 3)
    if flat.size != expect:
        print(f"  !! size mismatch: got {flat.size}, header expects {expect}")
    else:
        print(f"  size matches TOTAL_W_SIZE ({expect}), N_TOK={n_tok}")

    # ---- golden I/O for the C testbench ----------------------------------
    torch.manual_seed(args.seed)
    clean = torch.rand(1, 3, args.size, args.size)
    noisy = (clean + args.sigma * torch.randn_like(clean)).clamp(0, 1)
    with torch.no_grad():
        noise_pred = net(noisy)
        out = (noisy - noise_pred).clamp(0, 1)

    ipath = os.path.join(args.outdir, "io_input.bin")
    epath = os.path.join(args.outdir, "io_expected.bin")
    noisy.numpy().astype("<f4").tofile(ipath)
    out.numpy().astype("<f4").tofile(epath)
    print(f"wrote {ipath} and {epath}  ({noisy.numel()} floats each)")


if __name__ == "__main__":
    main()
