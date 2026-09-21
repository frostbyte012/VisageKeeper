#!/usr/bin/env python3
"""
scripts/eval_moe.py -- did joint decorrelation training actually pay off?

Evaluates each trained MoE against the bar that matters: ViT-only smoothed
accuracy at the same sigma (0.890 at sigma=0.5, 200 held-out pairs).

Reports three things per checkpoint:
  1. SMOOTHED ACCURACY of the MoE (soft routing, and hard top-1 routing) --
     does the deployed system beat ViT alone?
  2. ORACLE CEILING of its two trained experts -- did decorrelation actually
     WIDEN the niche? This is the diagnostic that separates "the gate got
     better" from "the experts became genuinely complementary". If the ceiling
     is still ~+1% over the best expert, the penalty did not do its job even if
     accuracy moved a little.
  3. GATE QUALITY -- how much of the oracle the learned gate actually captures.

Also evaluates the ViT-only and DnCNN-only baselines through the SAME code path
so the comparison is apples-to-apples.

Run:
  python scripts/eval_moe.py --sigma 0.5 --n_pairs 200 \
      --ckpts results/moe_s050_dec0.0.pth results/moe_s050_dec1.0.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser, SmoothedVerifier
from frpure.defenses.moe_denoiser import MoEDenoiser
from frpure.defenses.cascade import _disable_fused_attention
from scripts.run_certify import load_pairs, balanced_subset, embed_all
from scripts.diag_error_overlap import CKPT


def load_moe(path, device):
    cheap = build_denoiser("dncnn", ch=32)
    exp = build_denoiser("vit", patch=4, pos_mode="2d")
    moe = MoEDenoiser(cheap, exp).to(device).eval()
    moe.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(moe)
    for p in moe.parameters():
        p.requires_grad_(False)
    return moe


def load_single(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    _disable_fused_attention(d)
    for p in d.parameters():
        p.requires_grad_(False)
    return d


def acc_vector(denoiser, backbone, img1, idx, e2, lab, sigma, tau, eval_n, device):
    """Per-pair correctness of the smoothed verifier using `denoiser`."""
    sv = SmoothedVerifier(backbone, denoiser, sigma=sigma, tau=tau, device=device)
    return np.array([sv.predict(img1[i], e2[j].to(device), n=eval_n) == lab[j]
                     for j, i in enumerate(idx)])


class _Expert(torch.nn.Module):
    """Expose one MoE expert as a standalone denoiser (for the oracle ceiling)."""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy(); lab = same[idx].astype(int)
    imp = cos_clean[lab == 0]
    tau = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3
    print(f"sigma={args.sigma}  tau={tau:.4f}  pairs={len(idx)}\n")

    ev = lambda d: acc_vector(d, backbone, img1, idx, e2, lab, args.sigma, tau,
                              args.eval_n, device)

    # ---- baselines --------------------------------------------------------
    dn_w, vit_w = CKPT[args.sigma]
    base = {}
    for name, arch, w in (("DnCNN-only", "dncnn", dn_w), ("ViT-only", "vit", vit_w)):
        d = load_single(arch, w, device)
        base[name] = ev(d)
        print(f"  {name:>12}: {base[name].mean():.3f}")
        del d; torch.cuda.empty_cache()
    bar = base["ViT-only"].mean()
    print(f"\nBAR TO BEAT (ViT-only) = {bar:.3f}\n")

    results = {"sigma": args.sigma, "tau": tau, "n_pairs": len(idx),
               "baselines": {k: float(v.mean()) for k, v in base.items()},
               "bar": float(bar), "runs": {}}

    for ck in args.ckpts:
        if not os.path.exists(ck):
            print(f"[skip] missing {ck}")
            continue
        tag = os.path.basename(ck).replace(".pth", "")
        moe = load_moe(ck, device)

        moe.set_hard(False); soft = ev(moe)
        moe.set_hard(True);  hard = ev(moe)
        # oracle over the TRAINED experts -> did the niche widen?
        ok_c = ev(_Expert(moe.cheap))
        ok_e = ev(_Expert(moe.expensive))
        neither = float((~ok_c & ~ok_e).mean())
        oracle = 1 - neither
        best_expert = max(ok_c.mean(), ok_e.mean())

        # gate usage on real data
        with torch.no_grad():
            usage = []
            for i in idx[:64]:
                x = img1[i].unsqueeze(0).repeat(16, 1, 1, 1).to(device)
                x = (x + torch.randn_like(x) * args.sigma).clamp(0, 1)
                usage.append(moe.gate_probs(x)[:, 1].mean().item())
        usage_vit = float(np.mean(usage))

        r = {"soft_acc": float(soft.mean()), "hard_acc": float(hard.mean()),
             "expert_dncnn": float(ok_c.mean()), "expert_vit": float(ok_e.mean()),
             "oracle": oracle, "best_expert": float(best_expert),
             "oracle_headroom": float(oracle - best_expert),
             "only_dncnn": float((ok_c & ~ok_e).mean()),
             "only_vit": float((~ok_c & ok_e).mean()),
             "usage_vit": usage_vit,
             "vs_bar_soft": float(soft.mean() - bar),
             "vs_bar_hard": float(hard.mean() - bar)}
        results["runs"][tag] = r

        print(f"=== {tag} ===")
        print(f"  MoE soft routing : {r['soft_acc']:.3f}   ({r['vs_bar_soft']:+.3f} vs bar)")
        print(f"  MoE hard routing : {r['hard_acc']:.3f}   ({r['vs_bar_hard']:+.3f} vs bar)")
        print(f"  trained experts  : dncnn={r['expert_dncnn']:.3f} vit={r['expert_vit']:.3f}")
        print(f"  only-dncnn       : {r['only_dncnn']:.3f}   only-vit: {r['only_vit']:.3f}")
        print(f"  ORACLE ceiling   : {oracle:.3f}  (headroom over best expert "
              f"{r['oracle_headroom']:+.3f})")
        print(f"  gate usage_vit   : {usage_vit:.3f}")
        del moe; torch.cuda.empty_cache()

    out = args.out or f"results/paper/moe_eval_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
