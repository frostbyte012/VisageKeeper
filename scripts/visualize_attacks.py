#!/usr/bin/env python3
"""
scripts/visualize_attacks.py  --  compare ALL attacks on the same faces.
Per sample, per attack: saves attacked image, perturbation, rectified (denoised)
image, plus cosine trajectory + defended decision.
  outdir/<dataset>/sample_XX/<attack>/{adv,perturbation,rectified}.png
  outdir/<dataset>/sample_XX/clean.png , rectified_clean.png
  outdir/<dataset>/sample_XX_attacks.png   (grid: rows=attacks, cols=image|pert|rectified)
  outdir/<dataset>/analysis.csv
Attacks: fgsm_linf, pgd_linf, pgd_l2, patch (raw backbone) + smoothed_l2,
smoothed_linf (adaptive). bpda_linf added if --purifier_weights given.
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
from frpure.attacks.base import Target
from frpure.attacks.gradient import fgsm, pgd, bpda_eot, DEFAULT_EPS_LINF
from frpure.attacks.patch import patch_attack
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


def pert_to_img(p):
    pmin, pmax = float(p.min()), float(p.max())
    return ((p - pmin) / (pmax - pmin + 1e-9)).detach().cpu().permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aligned_dir"); ap.add_argument("--pairs"); ap.add_argument("--bin_path")
    ap.add_argument("--backbone", default="dummy")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--denoiser_weights", default=None)
    ap.add_argument("--purifier_weights", default=None, help="enables bpda_linf vs the purifier")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--eps_linf", type=float, default=DEFAULT_EPS_LINF)
    ap.add_argument("--eps_l2", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eot", type=int, default=8)
    ap.add_argument("--patch_steps", type=int, default=60)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--outdir", default="results/attack_compare")
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

    purifier = None
    if args.purifier_weights:
        from frpure.defenses.frpure import FRPure
        purifier = FRPure().to(device).eval()
        purifier.load_state_dict(torch.load(args.purifier_weights, map_location=device))

    img1, img2, same = load_pairs(args)
    tag = (os.path.splitext(os.path.basename(args.bin_path))[0] if args.bin_path
           else ("synthetic" if args.synthetic else "lfw"))
    outdir = os.path.join(args.outdir, tag); os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(0)
    gen = np.where(same == 1)[0]; imp = np.where(same == 0)[0]
    rng.shuffle(gen); rng.shuffle(imp)
    half = args.n_samples // 2
    sample_idx = list(gen[:half]) + list(imp[:args.n_samples - half])

    with torch.no_grad():
        sub = np.concatenate([gen[:200], imp[:200]])
        c = (backbone.embed(img1[sub].to(device)).cpu() * backbone.embed(img2[sub].to(device)).cpu()).sum(1).numpy()
        ss = same[sub]
        tau = (args.tau if args.tau is not None
               else float(np.quantile(c[ss == 0], 1 - args.far)) if (ss == 0).any() else 0.3)

    sv = SmoothedVerifier(backbone, denoiser, sigma=args.sigma, tau=tau, device=device)
    tgt = Target(backbone, defense=None)
    tgt_bpda = Target(backbone, defense=purifier, bpda=True) if purifier is not None else None
    alpha_linf = 2.5 * args.eps_linf / args.steps
    alpha_l2 = 2.5 * args.eps_l2 / args.steps

    def attacks_for(probe_b, other_b, mode, probe1, g_emb, label):
        out = {}
        out["fgsm_linf"] = fgsm(tgt, probe_b, other_b, args.eps_linf, mode, n_eot=1)[0]
        out["pgd_linf"] = pgd(tgt, probe_b, other_b, args.eps_linf, alpha_linf, args.steps,
                              mode, norm="linf", n_eot=args.eot)[0]
        out["pgd_l2"] = pgd(tgt, probe_b, other_b, args.eps_l2, alpha_l2, args.steps,
                            mode, norm="l2", n_eot=args.eot)[0]
        out["patch"] = patch_attack(tgt, probe_b, other_b, mode, kind="eyeglass",
                                    steps=args.patch_steps, n_eot=max(args.eot // 2, 1))[0]
        out["smoothed_l2"] = pgd_smoothed(sv, probe1, g_emb, label, args.eps_l2,
                                          norm="l2", steps=args.steps, eot=args.eot)
        out["smoothed_linf"] = pgd_smoothed(sv, probe1, g_emb, label, args.eps_linf,
                                            norm="linf", steps=args.steps, eot=args.eot)
        if tgt_bpda is not None:
            out["bpda_linf"] = bpda_eot(tgt_bpda, probe_b, other_b, args.eps_linf,
                                        alpha_linf, args.steps, mode, n_eot=args.eot)[0]
        return out

    print(f"[{tag}] tau={tau:.3f} sigma={args.sigma}  {len(sample_idx)} samples, "
          f"{6 + (tgt_bpda is not None)} attacks each -> {outdir}")

    torch.manual_seed(0)
    rows = []
    for k, i in enumerate(sample_idx):
        probe = img1[i].to(device); other = img2[i:i + 1].to(device)
        label = int(same[i]); mode = "dodging" if label == 1 else "impersonation"
        with torch.no_grad():
            g_emb = backbone.embed(other)[0]
            cos_clean = float((backbone.embed(probe.unsqueeze(0))[0] * g_emb).sum())
        sdir = os.path.join(outdir, f"sample_{k:02d}"); os.makedirs(sdir, exist_ok=True)
        plt.imsave(f"{sdir}/clean.png", to_img(probe))

        advs = attacks_for(probe.unsqueeze(0), other, mode, probe, g_emb, label)
        noise = torch.randn_like(probe) * args.sigma
        with torch.no_grad():
            rect_clean = denoiser((probe + noise).clamp(0, 1).unsqueeze(0))[0] if denoiser is not None else probe
        plt.imsave(f"{sdir}/rectified_clean.png", to_img(rect_clean))

        names = list(advs.keys())
        grid = [("clean", probe, torch.zeros_like(probe), rect_clean, cos_clean, cos_clean, sv.predict(probe, g_emb, n=args.eval_n))]
        for name in names:
            adv = advs[name].to(device)
            pert = (adv - probe)
            with torch.no_grad():
                rect_adv = denoiser((adv + noise).clamp(0, 1).unsqueeze(0))[0] if denoiser is not None else adv
                cos_adv = float((backbone.embed(adv.unsqueeze(0))[0] * g_emb).sum())
                cos_radv = float((backbone.embed(rect_adv.unsqueeze(0))[0] * g_emb).sum())
            dec_adv = sv.predict(adv, g_emb, n=args.eval_n)
            adir = os.path.join(sdir, name); os.makedirs(adir, exist_ok=True)
            plt.imsave(f"{adir}/adv.png", to_img(adv))
            plt.imsave(f"{adir}/perturbation.png", pert_to_img(pert))
            plt.imsave(f"{adir}/rectified.png", to_img(rect_adv))
            rows.append(dict(sample=k, attack=name, pair_kind=("genuine" if label == 1 else "impostor"),
                             l2_pert=round(float(pert.flatten().norm()), 4),
                             linf_pert=round(float(pert.abs().max()), 4),
                             cos_clean=round(cos_clean, 4), cos_adv=round(cos_adv, 4),
                             cos_rect_adv=round(cos_radv, 4), tau=round(tau, 4),
                             undef_adv=int((cos_adv >= tau) == label),
                             defended_adv=dec_adv, label=label, defense_held=int(dec_adv == label)))
            grid.append((name, adv, pert, rect_adv, cos_adv, cos_radv, dec_adv))

        nr = len(grid)
        fig, axs = plt.subplots(nr, 3, figsize=(3 * 2.7, nr * 2.7))
        for r, (name, im, pt, rc, ca, cra, dadv) in enumerate(grid):
            axs[r][0].imshow(to_img(im)); axs[r][0].axis("off")
            axs[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
            axs[r][0].set_title(f"cos={ca:.3f}" + ("" if r == 0 else f" [{'OK' if dadv==label else 'X'}]"), fontsize=8)
            axs[r][1].imshow(pert_to_img(pt) if r else np.ones_like(to_img(im))); axs[r][1].axis("off")
            axs[r][2].imshow(to_img(rc)); axs[r][2].axis("off")
            axs[r][2].set_title(f"rect cos={cra:.3f}", fontsize=8)
        axs[0][0].annotate("image", (0.5, 1.25), xycoords="axes fraction", ha="center", fontsize=10)
        axs[0][1].annotate("perturbation", (0.5, 1.25), xycoords="axes fraction", ha="center", fontsize=10)
        axs[0][2].annotate("rectified", (0.5, 1.25), xycoords="axes fraction", ha="center", fontsize=10)
        fig.suptitle(f"{tag} sample {k} [{mode}]  tau={tau:.3f}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{outdir}/sample_{k:02d}_attacks.png", dpi=120); plt.close(fig)

    with open(f"{outdir}/analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    by_attack = {}
    for r in rows:
        by_attack.setdefault(r["attack"], []).append(r["defense_held"])
    print("  defense held by attack:", {a: f"{sum(v)}/{len(v)}" for a, v in by_attack.items()})
    print(f"  wrote per-attack images + sample_XX_attacks.png + analysis.csv under {outdir}")


if __name__ == "__main__":
    main()