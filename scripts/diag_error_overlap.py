#!/usr/bin/env python3
"""
scripts/diag_error_overlap.py -- does a DnCNN/ViT hybrid have ANY headroom?

A router can only beat both single-architecture baselines if the two denoisers
fail on DIFFERENT pairs. If their errors coincide, the best any router can do is
match the stronger one, and the hybrid is not worth writing up.

For each pair we record the smoothed (majority-vote) decision under each
denoiser and cross-tabulate:
      both right | only DnCNN | only ViT | both wrong
"oracle" = 1 - P(both wrong) is the ceiling for ANY per-input router.

Also dumps a per-pair gate signal so we can check (in the next step) whether the
"only ViT" cases are actually PREDICTABLE without labels.

Run:
  python scripts/diag_error_overlap.py --aligned_dir DATA/lfw_aligned_160 \
      --pairs DATA/pairs.txt --sigma 0.5 --n_pairs 200
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
from scripts.run_certify import load_pairs, balanced_subset, embed_all

# sigma -> (dncnn weights, vit weights)
CKPT = {
    0.05: ("results/denoiser_dncnn_s005_30ep.pth", "results/denoiser_vit_s005_v3.pth"),
    0.1:  ("results/denoiser_dncnn_s010_30ep.pth", "results/denoiser_vit_s010_v3.pth"),
    0.25: ("results/denoiser_dncnn_s025_30ep.pth", "results/denoiser_vit_s025_v3.pth"),
    0.5:  ("results/denoiser_dncnn_s050_30ep.pth", "results/denoiser_vit_s050_v3.pth"),
    0.75: ("results/denoiser_dncnn_s075_30ep.pth", "results/denoiser_vit_s075_v3.pth"),
    1.0:  ("results/denoiser_dncnn_s100_30ep.pth", "results/denoiser_vit_s100_v3.pth"),
}


def load_den(arch, path, device):
    kw = dict(patch=4, pos_mode="2d") if arch == "vit" else {}
    d = build_denoiser(arch, ch=32, **kw).to(device).eval()
    d.load_state_dict(torch.load(path, map_location=device))
    for p in d.parameters():
        p.requires_grad_(False)
    return d


@torch.no_grad()
def gate_stats(denoiser, backbone, probe, gallery, sigma, n=64, batch=64):
    """Label-free signals describing how the denoiser behaves on THIS probe.

    self_consistency: mean pairwise cosine between embeddings of independently
        denoised noisy copies. High = the denoiser maps every noise draw to the
        same identity (confident); low = it is guessing.
    margin: |mean cos(probe_emb, gallery) - tau| proxy, returned raw so the
        caller can center it.
    """
    x = probe.unsqueeze(0).repeat(n, 1, 1, 1)
    x = (x + torch.randn_like(x) * sigma).clamp(0, 1)
    embs = []
    for i in range(0, n, batch):
        embs.append(backbone.embed(denoiser(x[i:i + batch])))
    e = torch.cat(embs)                       # (n, d) already L2-normalised
    mu = torch.nn.functional.normalize(e.mean(0, keepdim=True), dim=1)
    self_cons = float((e * mu).sum(1).mean())  # concentration about the mean
    cos_gal = (e * gallery.unsqueeze(0)).sum(1)
    return self_cons, float(cos_gal.mean()), float(cos_gal.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir"); ap.add_argument("--pairs")
    ap.add_argument("--pairs_npz"); ap.add_argument("--bin_path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    backbone = build_backbone(args.backbone, device=device, pretrained=args.pretrained)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy(); sub_same = same[idx]
    imp = cos_clean[sub_same == 0]
    tau = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3
    print(f"sigma={args.sigma}  tau={tau:.4f}  pairs={len(idx)}")

    dn_w, vit_w = CKPT[args.sigma]
    dens = {"dncnn": load_den("dncnn", dn_w, device),
            "vit": load_den("vit", vit_w, device)}
    svs = {k: SmoothedVerifier(backbone, d, sigma=args.sigma, tau=tau, device=device)
           for k, d in dens.items()}

    recs = []
    for j, i in enumerate(idx):
        e_b = e2[j].to(device)
        label = int(sub_same[j])
        r = {"label": label}
        for k, sv in svs.items():
            r[f"pred_{k}"] = sv.predict(img1[i], e_b, n=args.eval_n)
            sc, cg, cs = gate_stats(dens[k], backbone, img1[i].to(device), e_b,
                                    args.sigma)
            r[f"selfcons_{k}"] = sc; r[f"cosgal_{k}"] = cg; r[f"cosstd_{k}"] = cs
        recs.append(r)
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(idx)}")

    lab = np.array([r["label"] for r in recs])
    ok_d = np.array([r["pred_dncnn"] for r in recs]) == lab
    ok_v = np.array([r["pred_vit"] for r in recs]) == lab

    both = float((ok_d & ok_v).mean())
    only_d = float((ok_d & ~ok_v).mean())
    only_v = float((~ok_d & ok_v).mean())
    neither = float((~ok_d & ~ok_v).mean())

    print("\n==== ERROR OVERLAP ====")
    print(f"  DnCNN acc           : {ok_d.mean():.3f}")
    print(f"  ViT   acc           : {ok_v.mean():.3f}")
    print(f"  both correct        : {both:.3f}")
    print(f"  only DnCNN correct  : {only_d:.3f}   <-- headroom ViT alone loses")
    print(f"  only ViT   correct  : {only_v:.3f}")
    print(f"  neither correct     : {neither:.3f}")
    print(f"  ORACLE router ceiling: {1 - neither:.3f}")
    print(f"  best single         : {max(ok_d.mean(), ok_v.mean()):.3f}")
    print(f"  => max possible gain: {(1 - neither) - max(ok_d.mean(), ok_v.mean()):+.3f}")

    out = args.out or f"results/diag_overlap_s{int(args.sigma*100):03d}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"sigma": args.sigma, "tau": tau, "n_pairs": len(idx),
                   "acc_dncnn": float(ok_d.mean()), "acc_vit": float(ok_v.mean()),
                   "both": both, "only_dncnn": only_d, "only_vit": only_v,
                   "neither": neither, "oracle": 1 - neither,
                   "records": recs}, f, indent=2)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
