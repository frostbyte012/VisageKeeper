#!/usr/bin/env python3
"""
scripts/visualize_compare.py  --  clean vs attacked vs rectified panels.

Saves, per dataset:
  outdir/<dataset>/sample_XX/{clean,adv,perturbation,noisy_clean,rectified_clean,
                              noisy_adv,rectified_adv}.png
  outdir/<dataset>/sample_XX_panel.png      (per-sample grid)
  outdir/<dataset>/montage.png              (all samples stacked)
  outdir/<dataset>/analysis.csv             (numbers per sample)
"rectified" = denoiser output on the smoothing-noised image (what the defended
pipeline feeds the backbone). Per-sample: cosine-to-gallery at every stage, the
defended decision clean vs adv, and L2/Linf perturbation size.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import GaussianDenoiser, SmoothedVerifier
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
    from frpure.data.lfw import LFWPairs
    ds = LFWPairs(args.aligned_dir, args.pairs)
    i1 = torch.stack([ds[i].img1 for i in range(len(ds))])
    i2 = torch.stack([ds[i].img2 for i in range(len(ds))])
    same = np.array([int(ds.pairs[i][2]) for i in range(len(ds))])
    return i1, i2, same


def to_img(t):
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def save_png(t, path):
    plt.imsave(path, to_img(t))


def cos(a, b):
    return float((a * b).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aligned_dir"); ap.add_argument("--pairs"); ap.add_argument("--bin_path")
    ap.add_argument("--backbone", default="dummy")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--denoiser_weights", default=None)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--norm", choices=["l2", "linf"], default="l2")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eot", type=int, default=8)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--outdir", default="results/compare")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    bb_kw = {"device": device}
    if args.backbone == "facenet":
        bb_kw["pretrained"] = args.pretrained
    backbone = build_backbone(args.backbone, **bb_kw)

    denoiser = None
    if args.denoiser_weights:
        denoiser = GaussianDenoiser(ch=args.ch).to(device).eval()
        denoiser.load_state_dict(torch.load(args.denoiser_weights, map_location=device))
        for p in denoiser.parameters():
            p.requires_grad_(False)

    img1, img2, same = load_pairs(args)
    tag = (os.path.splitext(os.path.basename(args.bin_path))[0] if args.bin_path
           else ("synthetic" if args.synthetic else "lfw"))
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(0)
    gen = np.where(same == 1)[0]; imp = np.where(same == 0)[0]
    rng.shuffle(gen); rng.shuffle(imp)
    half = args.n_samples // 2
    sample_idx = list(gen[:half]) + list(imp[:args.n_samples - half])

    with torch.no_grad():
        sub = np.concatenate([gen[:200], imp[:200]])
        e1 = backbone.embed(img1[sub].to(device)).cpu()
        e2 = backbone.embed(img2[sub].to(device)).cpu()
        c = (e1 * e2).sum(1).numpy(); ssame = same[sub]
        tau = (args.tau if args.tau is not None
               else float(np.quantile(c[ssame == 0], 1 - args.far)) if (ssame == 0).any() else 0.3)

    sv = SmoothedVerifier(backbone, denoiser, sigma=args.sigma, tau=tau, device=device)
    print(f"[{tag}] tau={tau:.3f}  sigma={args.sigma}  attack={args.norm} eps={args.eps}  "
          f"{len(sample_idx)} samples -> {outdir}")

    torch.manual_seed(0)
    rows = []; panels = []
    for k, i in enumerate(sample_idx):
        probe = img1[i].to(device)
        with torch.no_grad():
            g_emb = backbone.embed(img2[i:i + 1].to(device))[0]
        label = int(same[i])

        adv = pgd_smoothed(sv, probe, g_emb, label, args.eps,
                           norm=args.norm, steps=args.steps, eot=args.eot)

        noise = torch.randn_like(probe) * args.sigma
        noisy_clean = (probe + noise).clamp(0, 1)
        noisy_adv = (adv + noise).clamp(0, 1)
        with torch.no_grad():
            rect_clean = denoiser(noisy_clean.unsqueeze(0))[0] if denoiser is not None else noisy_clean
            rect_adv = denoiser(noisy_adv.unsqueeze(0))[0] if denoiser is not None else noisy_adv
            e_clean = backbone.embed(probe.unsqueeze(0))[0]
            e_adv = backbone.embed(adv.unsqueeze(0))[0]
            e_radv = backbone.embed(rect_adv.unsqueeze(0))[0]

        pert = (adv - probe).detach().cpu()
        l2 = float(pert.flatten().norm()); linf = float(pert.abs().max())
        pmin, pmax = float(pert.min()), float(pert.max())
        pert_vis = (pert - pmin) / (pmax - pmin + 1e-9)

        cos_clean = cos(e_clean, g_emb); cos_adv = cos(e_adv, g_emb); cos_radv = cos(e_radv, g_emb)
        dec_clean = sv.predict(probe, g_emb, n=args.eval_n)
        dec_adv = sv.predict(adv, g_emb, n=args.eval_n)
        undef_clean = int(cos_clean >= tau); undef_adv = int(cos_adv >= tau)

        sdir = os.path.join(outdir, f"sample_{k:02d}"); os.makedirs(sdir, exist_ok=True)
        save_png(probe, f"{sdir}/clean.png"); save_png(adv, f"{sdir}/adv.png")
        plt.imsave(f"{sdir}/perturbation.png", pert_vis.permute(1, 2, 0).numpy())
        save_png(noisy_clean, f"{sdir}/noisy_clean.png"); save_png(rect_clean, f"{sdir}/rectified_clean.png")
        save_png(noisy_adv, f"{sdir}/noisy_adv.png"); save_png(rect_adv, f"{sdir}/rectified_adv.png")

        rows.append(dict(
            sample=k, pair_kind=("genuine" if label == 1 else "impostor"),
            l2_pert=round(l2, 4), linf_pert=round(linf, 4),
            cos_clean=round(cos_clean, 4), cos_adv=round(cos_adv, 4),
            cos_rect_adv=round(cos_radv, 4), tau=round(tau, 4),
            undef_clean=undef_clean, undef_adv=undef_adv,
            defended_clean=dec_clean, defended_adv=dec_adv, label=label,
            attack_flipped_undefended=int(undef_clean == label and undef_adv != label),
            defense_held=int(dec_adv == label)))
        panels.append((k, label, to_img(probe), to_img(adv), pert_vis.permute(1, 2, 0).numpy(),
                       to_img(rect_clean), to_img(rect_adv), cos_clean, cos_adv, cos_radv, dec_adv))

        titles = ["clean", f"adv ({args.norm} {args.eps})", "perturbation", "rectified clean", "rectified adv"]
        imgs = [to_img(probe), to_img(adv), pert_vis.permute(1, 2, 0).numpy(),
                to_img(rect_clean), to_img(rect_adv)]
        fig, axs = plt.subplots(1, 5, figsize=(15, 3.4))
        for ax, im, t in zip(axs, imgs, titles):
            ax.imshow(im); ax.set_title(t, fontsize=10); ax.axis("off")
        kind = "genuine" if label == 1 else "impostor"
        fig.suptitle(f"{tag} sample {k} [{kind}]  cos: clean={cos_clean:.3f} -> adv={cos_adv:.3f} "
                     f"-> rect_adv={cos_radv:.3f}  tau={tau:.3f}  defended adv={'OK' if dec_adv==label else 'BROKEN'}",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"{outdir}/sample_{k:02d}_panel.png", dpi=110); plt.close(fig)

    ncol = 5
    fig, axs = plt.subplots(len(panels), ncol, figsize=(ncol * 2.6, len(panels) * 2.6))
    if len(panels) == 1:
        axs = axs[None, :]
    for r, (k, label, cl, ad, pv, rc, ra, cc, ca, cra, dadv) in enumerate(panels):
        for cax, im in zip(axs[r], [cl, ad, pv, rc, ra]):
            cax.imshow(im); cax.axis("off")
        kind = "gen" if label == 1 else "imp"
        axs[r][0].set_ylabel(f"#{k} {kind}", rotation=0, ha="right", va="center", fontsize=9)
        axs[r][1].set_title(f"cos {cc:.2f}->{ca:.2f}", fontsize=8)
        axs[r][4].set_title(f"rect {cra:.2f} [{'OK' if dadv==label else 'X'}]", fontsize=8)
    for c, t in enumerate(["clean", "attacked", "perturbation", "rectified clean", "rectified adv"]):
        axs[0][c].set_title((axs[0][c].get_title() + "\n" + t) if axs[0][c].get_title() else t, fontsize=9)
    fig.suptitle(f"{tag}: clean vs attacked vs rectified ({args.norm} eps={args.eps}, sigma={args.sigma})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{outdir}/montage.png", dpi=120); plt.close(fig)

    with open(f"{outdir}/analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    held = sum(r["defense_held"] for r in rows)
    flipped = sum(r["attack_flipped_undefended"] for r in rows)
    print(f"  undefended flips: {flipped}/{len(rows)}   defense held: {held}/{len(rows)}")
    print(f"  wrote images + montage.png + analysis.csv under {outdir}")


if __name__ == "__main__":
    main()