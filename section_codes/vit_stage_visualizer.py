#!/usr/bin/env python3
"""
section_codes/vit_stage_visualizer.py -- what the ViT denoiser does, stage by stage.

Takes ONE clean face image, adds Gaussian noise, then walks the noisy image
through every stage of frpure.defenses.vit_denoiser.ViTDenoiser, saving a PNG
for each stage. This reproduces the internal computation of `ViTDenoiser.forward`
step by step rather than calling it, so every intermediate tensor is available.

Stages saved (one PNG each, plus a summary sheet):
  00  clean input                       3x160x160
  01  noisy input      x = clean + eps  3x160x160
  02  patch grid       Conv2d(3->192,k=4,s=4)   192x40x40
  03  tokens           flatten+transpose        1600x192
  04  positional enc   fixed 2D sin-cos         1600x192
  05  tokens + pos                              1600x192
  06  attention maps   layer 1, all 6 heads     6x1600x1600 (probe rows shown)
  07  after layer k    k = 1..4                 1600x192
  08  reshaped         transpose+reshape        192x40x40
  09  predicted noise  ConvTranspose(192->3)    3x160x160
  10  denoised         clamp(x - noise, 0, 1)   3x160x160

Run:
  python section_codes/vit_stage_visualizer.py \
      --image DATA/lfw_aligned_160/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
      --sigma 0.75 --weights results/denoiser_vit_s075_v3.pth \
      --outdir results/vit_stages
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frpure.defenses.vit_denoiser import ViTDenoiser
from frpure.defenses.cascade import _disable_fused_attention


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def to_img(t):
    """(3,H,W) tensor in [0,1] -> (H,W,3) numpy for imshow."""
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def save_rgb(t, path, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    ax.imshow(to_img(t))
    ax.set_title(title, fontsize=11, fontweight="600")
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=9, color="#555")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_channels(feat, path, title, subtitle="", n=16):
    """feat (C,h,w): show the first n channels as a grid of heatmaps.

    Each channel is one of the 192 'descriptions' the patch-embed produced --
    channel c at grid cell (i,j) is filter c's response to the 4x4 patch there.
    """
    f = feat.detach().cpu()
    n = min(n, f.shape[0])
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
    axes = np.atleast_1d(axes).ravel()
    for k in range(len(axes)):
        axes[k].set_xticks([]); axes[k].set_yticks([])
        if k < n:
            axes[k].imshow(f[k], cmap="viridis")
            axes[k].set_title(f"ch {k}", fontsize=7)
        else:
            axes[k].axis("off")
    fig.suptitle(title, fontsize=11, fontweight="600")
    if subtitle:
        fig.text(0.5, 0.005, subtitle, ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_tokens(tok, path, title, subtitle="", max_tok=200):
    """tok (N,D): the token matrix itself, as a heatmap (rows=tokens, cols=features)."""
    t = tok.detach().cpu().numpy()[:max_tok]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(t, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(t).max(), vmax=np.abs(t).max())
    ax.set_xlabel("feature index (0..191)", fontsize=9)
    ax.set_ylabel(f"token index (0..{max_tok-1} of {tok.shape[0]})", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="600")
    fig.colorbar(im, ax=ax, fraction=0.03)
    if subtitle:
        fig.text(0.5, 0.005, subtitle, ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_token_norms(tok, gh, gw, path, title, subtitle=""):
    """Per-token magnitude, folded back onto the 40x40 grid -> spatial view."""
    n = tok.detach().cpu().norm(dim=-1).reshape(gh, gw).numpy()
    fig, ax = plt.subplots(figsize=(4.4, 4.8))
    im = ax.imshow(n, cmap="magma")
    ax.set_title(title, fontsize=11, fontweight="600")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_attention(attn, gh, gw, path, probes=(820, 0, 1599)):
    """attn (heads,N,N): for a few probe tokens, show WHICH patches they attend to.

    Row i of the attention matrix says how much token i draws from every other
    token. Folding that row back onto the 40x40 grid shows, spatially, where on
    the face this patch is gathering information from -- the thing a DnCNN
    cannot do.
    """
    a = attn.detach().cpu()
    H = a.shape[0]
    fig, axes = plt.subplots(len(probes), H, figsize=(H * 1.7, len(probes) * 1.95))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(probes):
        for h in range(H):
            ax = axes[r, h]
            ax.imshow(a[h, p].reshape(gh, gw), cmap="inferno")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"head {h}", fontsize=8)
            if h == 0:
                ax.set_ylabel(f"token {p}\n(row {p//gw}, col {p%gw})", fontsize=7.5)
    fig.suptitle("Layer 1 self-attention: where each probe patch looks "
                 "(bright = high attention)", fontsize=11, fontweight="600")
    fig.text(0.5, 0.005, "every patch can read every other patch in ONE step -- "
             "this is what a local convolution cannot do",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(items, path):
    """One contact sheet of the RGB-space stages, for the paper figure."""
    fig, axes = plt.subplots(1, len(items), figsize=(3.0 * len(items), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (img, title, sub) in zip(axes, items):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="600")
        ax.set_xlabel(sub, fontsize=8, color="#555")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("ViT denoiser: pixel-space stages", fontsize=12, fontweight="700")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="path to a face image; default = first LFW aligned image")
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--weights", default="results/denoiser_vit_s075_v3.pth")
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/vit_stages")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # ---- load one clean face ---------------------------------------------
    path = args.image
    if path is None:
        cand = sorted(glob.glob(os.path.join(args.aligned_dir, "*", "*.jpg"))) + \
               sorted(glob.glob(os.path.join(args.aligned_dir, "*", "*.png")))
        if not cand:
            raise RuntimeError(f"no images under {args.aligned_dir}")
        path = cand[0]
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((args.size, args.size))
    clean = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    clean = clean.permute(2, 0, 1).unsqueeze(0).to(device)     # (1,3,H,W)
    print(f"image      : {path}")
    print(f"clean      : {tuple(clean.shape)}")

    # ---- build the denoiser ----------------------------------------------
    den = ViTDenoiser(patch=args.patch, dim=args.dim, depth=args.depth,
                      heads=args.heads, pos_mode="2d").to(device).eval()
    if os.path.exists(args.weights):
        sd = torch.load(args.weights, map_location=device)
        # Checkpoints are saved from ViTGaussianDenoiser, which holds the
        # ViTDenoiser as `.core`, so every key is prefixed. Strip it.
        if any(k.startswith("core.") for k in sd):
            sd = {k[len("core."):]: v for k, v in sd.items() if k.startswith("core.")}
        den.load_state_dict(sd)
        print(f"weights    : {args.weights}")
    else:
        print(f"weights    : NOT FOUND ({args.weights}) -- using random init; "
              f"shapes are still correct but the denoised output will be garbage")
    _disable_fused_attention(den)
    for p in den.parameters():
        p.requires_grad_(False)

    # ---- STAGE 00/01: clean -> noisy --------------------------------------
    noise = torch.randn_like(clean) * args.sigma
    x = (clean + noise).clamp(0, 1)
    save_rgb(clean[0], f"{args.outdir}/00_clean.png", "00  clean input",
             f"3x{args.size}x{args.size} = {3*args.size*args.size:,} values")
    save_rgb(x[0], f"{args.outdir}/01_noisy.png",
             f"01  noisy input (sigma={args.sigma})",
             "x = clamp(clean + N(0, sigma^2), 0, 1)   shape unchanged")
    print(f"01 noisy   : {tuple(x.shape)}")

    with torch.no_grad():
        # ---- STAGE 02: patch embedding ------------------------------------
        # Conv2d(3->dim, k=patch, s=patch): stride == kernel, so the 4x4 patches
        # TILE the image without overlap. 48 input numbers -> 192 output numbers.
        grid = den.embed(x)                                   # (1,dim,gh,gw)
        gh, gw = grid.shape[-2:]
        n_tok = gh * gw
        save_channels(grid[0], f"{args.outdir}/02_patch_grid.png",
                      f"02  patch grid: {args.dim}x{gh}x{gw}  (first 16 of {args.dim} channels)",
                      f"Conv2d(3->{args.dim}, k={args.patch}, s={args.patch}): each 4x4x3=48-value patch "
                      f"-> {args.dim} numbers; {args.size}/{args.patch}={gh} positions per side")
        print(f"02 grid    : {tuple(grid.shape)}  ({gh}x{gw} = {n_tok} patches)")

        # ---- STAGE 03: flatten to a token sequence -------------------------
        # Pure reshape: same numbers, new arrangement. No parameters.
        tok = grid.flatten(2).transpose(1, 2)                  # (1,N,dim)
        save_tokens(tok[0], f"{args.outdir}/03_tokens.png",
                    f"03  tokens: {n_tok}x{args.dim}  (first 200 tokens shown)",
                    "flatten(2)+transpose: grid -> sequence. Same numbers, zero parameters. "
                    "Row i = the 192-number description of patch i")
        print(f"03 tokens  : {tuple(tok.shape)}")

        # ---- STAGE 04/05: positional encoding ------------------------------
        # Flattening threw away geometry; attention treats its input as an
        # unordered set. The encoding re-injects "which row/col was I?".
        pos = den._pos_embed(gh, gw, x.device, x.dtype)        # (1,N,dim)
        save_tokens(pos[0], f"{args.outdir}/04_pos_encoding.png",
                    f"04  positional encoding: {n_tok}x{args.dim}",
                    "fixed 2D sin-cos, ZERO learned params. Channels 0-95 encode the ROW, "
                    "96-191 encode the COLUMN")
        tok_pos = tok + pos
        save_tokens(tok_pos[0], f"{args.outdir}/05_tokens_plus_pos.png",
                    f"05  tokens + position: {n_tok}x{args.dim}",
                    "elementwise addition -- shape unchanged, but each token now carries its address")
        print(f"04 pos     : {tuple(pos.shape)}")

        # ---- STAGE 06: attention maps from layer 1 -------------------------
        # Re-run layer 1's attention with need_weights=True to expose the
        # (heads, N, N) matrix. Row i shows where token i gathers from.
        layer0 = den.encoder.layers[0]
        _, attn = layer0.self_attn(tok_pos, tok_pos, tok_pos,
                                   need_weights=True, average_attn_weights=False)
        centre = (gh // 2) * gw + (gw // 2)
        save_attention(attn[0], gh, gw, f"{args.outdir}/06_attention_layer1.png",
                       probes=(centre, 0, n_tok - 1))
        print(f"06 attn    : {tuple(attn[0].shape)}  (heads, N, N) = "
              f"{attn.shape[1]}x{n_tok}x{n_tok}")

        # ---- STAGE 07: run the 4 encoder layers ----------------------------
        # NOT a loop over one layer: 4 SEPARATE layers, each with its own
        # ~444k parameters, executed once each in sequence.
        h = tok_pos
        for li, layer in enumerate(den.encoder.layers, start=1):
            h = layer(h)
            save_token_norms(h[0], gh, gw,
                             f"{args.outdir}/07_after_layer{li}.png",
                             f"07.{li}  after encoder layer {li}: {n_tok}x{args.dim}",
                             "per-token magnitude folded back to the 40x40 grid "
                             "(shape never changes; only values do)")
            print(f"07 layer{li} : {tuple(h.shape)}")

        # ---- STAGE 08: reshape back to a grid ------------------------------
        back = h.transpose(1, 2).reshape(1, den.dim, gh, gw)   # (1,dim,gh,gw)
        save_channels(back[0], f"{args.outdir}/08_reshaped.png",
                      f"08  reshaped: {args.dim}x{gh}x{gw}  (first 16 channels)",
                      "exact inverse of stage 03 -- sequence -> grid, zero parameters")
        print(f"08 reshape : {tuple(back.shape)}")

        # ---- STAGE 09: conv-transpose -> predicted NOISE -------------------
        # Mirror of the patch embed: 1 grid cell (192 numbers) -> one 3x4x4
        # output patch. The result is the noise to REMOVE, not the clean face.
        pred_noise = den.unembed(back)                         # (1,3,H,W)
        pn = pred_noise[0]
        pn_vis = (pn - pn.min()) / (pn.max() - pn.min() + 1e-8)
        save_rgb(pn_vis, f"{args.outdir}/09_predicted_noise.png",
                 f"09  predicted noise: 3x{args.size}x{args.size}",
                 f"ConvTranspose2d({args.dim}->3, k={args.patch}, s={args.patch}); "
                 f"min={pn.min():.3f} max={pn.max():.3f} (contrast-stretched for display)")
        print(f"09 noise   : {tuple(pred_noise.shape)}")

        # ---- STAGE 10: subtract + clamp ------------------------------------
        raw = x - pred_noise
        out = raw.clamp(0, 1)
        n_clipped = int(((raw < 0) | (raw > 1)).sum())
        save_rgb(out[0], f"{args.outdir}/10_denoised.png",
                 f"10  denoised: 3x{args.size}x{args.size}",
                 f"clamp(x - predicted_noise, 0, 1); {n_clipped:,} of "
                 f"{raw.numel():,} values were out of range and clipped")
        print(f"10 denoised: {tuple(out.shape)}  ({n_clipped:,} values clipped)")

    # ---- contact sheet -----------------------------------------------------
    save_summary([
        (to_img(clean[0]), "00 clean", f"3x{args.size}x{args.size}"),
        (to_img(x[0]), f"01 noisy (s={args.sigma})", "3x160x160"),
        (to_img(pn_vis), "09 predicted noise", "3x160x160"),
        (to_img(out[0]), "10 denoised", "3x160x160"),
    ], f"{args.outdir}/summary_pixel_stages.png")

    # ---- fidelity numbers --------------------------------------------------
    mse_noisy = float(F.mse_loss(x, clean))
    mse_den = float(F.mse_loss(out, clean))
    print(f"\nMSE(noisy, clean)    = {mse_noisy:.5f}")
    print(f"MSE(denoised, clean) = {mse_den:.5f}"
          f"   ({'improved' if mse_den < mse_noisy else 'WORSE'} "
          f"{mse_noisy/max(mse_den,1e-9):.2f}x)")
    print(f"\nwrote {len(os.listdir(args.outdir))} files -> {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
