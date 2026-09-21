#!/usr/bin/env python3
"""
frpure.eval.runner  --  fills the comparison matrix.

For each (defense x attack) it:
  1. fixes the operating threshold on CLEAN pairs at TAR@FAR=1e-3 (frozen),
  2. generates adversarial probes ADAPTIVELY (PGD+EOT through the defense if it
     is differentiable, else BPDA straight-through; BPDA+EOT column always),
  3. scores verification (probe purified, gallery clean) and computes the row.

Run on the GPU server:
  python -m frpure.eval.runner --pairs_npz DATA/lfw_pairs_160.npz --backbone facenet \
      --frpure_weights results/purifier.pth --out results/lfw_trained.csv
CPU smoke (no data/weights):
  python -m frpure.eval.runner --synthetic --device cpu --out results/synthetic.csv
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from frpure.eval import metrics as M
from frpure.attacks.base import Target
from frpure.attacks import gradient as G
from frpure.attacks.patch import patch_attack
from frpure.defenses.base import Identity
from frpure.defenses.baselines import FeatureSqueezing, KimNIL, IterativePurifier
from frpure.defenses.frpure import FRPure


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score(backbone, defense, probe01, gallery01, batch=64):
    """cosine sim; probe purified, gallery clean (enrolled). Batched + device-safe
    (moves each chunk to the backbone device) so large pair sets don't OOM."""
    dev = getattr(backbone, "device", probe01.device)
    sims = []
    for i in range(0, probe01.shape[0], batch):
        p = probe01[i:i + batch].to(dev)
        g = gallery01[i:i + batch].to(dev)
        ep = backbone.embed(defense.purify(p))
        eg = backbone.embed(g)
        sims.append((ep * eg).sum(dim=1).cpu())
    return torch.cat(sims)


# --------------------------------------------------------------------------- #
# attack dispatch
# --------------------------------------------------------------------------- #
def make_target(backbone, defense, force_bpda: bool):
    bpda = force_bpda or (not defense.differentiable)
    return Target(backbone, defense, bpda=bpda)


def run_attack(name, backbone, defense, probe01, gallery01, mode, device):
    eps, alpha = G.DEFAULT_EPS_LINF, G.DEFAULT_ALPHA_LINF
    n_eot = 10 if defense.randomized else 1
    if name == "clean":
        return probe01
    if name == "fgsm":
        t = make_target(backbone, defense, force_bpda=False)
        return G.fgsm(t, probe01, gallery01, eps, mode, n_eot)
    if name == "pgd_eot":                       # PRIMARY: direct gradients
        t = make_target(backbone, defense, force_bpda=False)
        return G.pgd(t, probe01, gallery01, eps, alpha, steps=40, mode=mode, n_eot=max(n_eot, 1))
    if name == "bpda_eot":                      # corroboration
        t = make_target(backbone, defense, force_bpda=True)
        return G.bpda_eot(t, probe01, gallery01, eps, alpha, steps=40, mode=mode, n_eot=10)
    if name.startswith("patch_"):
        kind = name.split("_", 1)[1]            # eyeglass | hat | sticker
        t = make_target(backbone, defense, force_bpda=not defense.differentiable)
        return patch_attack(t, probe01, gallery01, mode, kind=kind, steps=60, n_eot=5)
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #
def run_cell(backbone, defense, attack, pools, device) -> M.ResultRow:
    g1, g2, same = pools["all"]                 # all clean pairs for threshold
    s_clean = score(backbone, defense, g1, g2).cpu().numpy()
    gen = s_clean[same.cpu().numpy()]
    imp = s_clean[~same.cpu().numpy()]
    ver_acc, _ = M.verification_accuracy(gen, imp)
    taf = M.tar_at_far(gen, imp, (1e-2, 1e-3))
    op_thr = taf[1e-3]["threshold"]

    row = M.ResultRow(defense=defense.name, attack=attack,
                      threat="Linf_8/255_whitebox_adaptive",
                      ver_acc=ver_acc, tar_at_far=taf)

    if attack == "clean":
        row.robust_tar = M.robust_tar(gen, imp, op_thr)
        return row

    # dodging on genuine probes
    dg1, dg2 = pools["dodging"]
    adv_d = run_attack(attack, backbone, defense, dg1, dg2, "dodging", device)
    sd = score(backbone, defense, adv_d, dg2).cpu().numpy()
    row.asr_dodging = M.attack_success_rate(sd, op_thr, "dodging")

    # impersonation on impostor probes
    ip1, ip2 = pools["impersonation"]
    adv_i = run_attack(attack, backbone, defense, ip1, ip2, "impersonation", device)
    si = score(backbone, defense, adv_i, ip2).cpu().numpy()
    row.asr_impersonation = M.attack_success_rate(si, op_thr, "impersonation")

    # robust_tar over the attacked pools, at the frozen clean threshold
    row.robust_tar = M.robust_tar(sd, si, op_thr)

    # fidelity of the purified adversarial probe vs clean probe (dodging pool)
    with torch.no_grad():
        pur = defense.purify(adv_d)
    row.psnr = M.psnr(dg1.cpu().numpy(), pur.cpu().numpy())
    row.ssim = M.ssim(dg1.cpu().numpy(), pur.cpu().numpy())
    pn = M.perturbation_norms(dg1.cpu().numpy(), adv_d.cpu().numpy())
    row.pert_l2, row.pert_linf = pn["l2"], pn["linf"]

    # software latency of a purify pass
    with M.Throughput(n_items=dg1.shape[0]) as tp:
        with torch.no_grad():
            _ = defense.purify(dg1)
    row.latency_ms, row.fps = tp.latency_ms, tp.fps
    return row


# --------------------------------------------------------------------------- #
# data providers
# --------------------------------------------------------------------------- #
def synthetic_pools(device, n_id=60, n_pairs=120, n_attack=40):
    """Planted-identity synthetic pairs so DummyBackbone separates gen/imp."""
    g = torch.Generator().manual_seed(1)
    protos = torch.rand(n_id, 3, 64, 64, generator=g)

    def jit(p):  # a noisy 'capture' of an identity
        return (p + torch.randn(p.shape, generator=g) * 0.05).clamp(0, 1)

    a1, a2, same = [], [], []
    for k in range(n_pairs):
        i = k % n_id
        if k % 2 == 0:
            a1.append(jit(protos[i])); a2.append(jit(protos[i])); same.append(True)
        else:
            j = (i + 1) % n_id
            a1.append(jit(protos[i])); a2.append(jit(protos[j])); same.append(False)
    all1 = torch.stack(a1).to(device); all2 = torch.stack(a2).to(device)
    same_t = torch.tensor(same)
    gi = [k for k in range(n_pairs) if same[k]][:n_attack]
    ii = [k for k in range(n_pairs) if not same[k]][:n_attack]
    return {
        "all": (all1, all2, same_t),
        "dodging": (all1[gi], all2[gi]),
        "impersonation": (all1[ii], all2[ii]),
    }


def lfw_pools(aligned_dir, pairs_txt, device, n_attack=200):
    from frpure.data.lfw import LFWPairs, sample_attack_subset
    ds = LFWPairs(aligned_dir, pairs_txt)
    all1 = torch.stack([ds[i].img1 for i in range(len(ds))])   # CPU (can be large)
    all2 = torch.stack([ds[i].img2 for i in range(len(ds))])
    same = torch.tensor([ds.pairs[i][2] for i in range(len(ds))])
    sub = sample_attack_subset(ds, n_attack, n_attack)
    d = sub["dodging"]; im = sub["impersonation"]
    return {
        "all": (all1, all2, same),                             # CPU; score() moves per-batch
        "dodging": (all1[d].to(device), all2[d].to(device)),   # small -> device for attacks
        "impersonation": (all1[im].to(device), all2[im].to(device)),
    }


def npz_pools(npz_path, device, n_attack=200):
    """Load aligned eval pairs saved by prepare_data.py (img1,img2,same; uint8)."""
    import numpy as np
    d = np.load(npz_path)
    img1 = torch.from_numpy(d["img1"]).permute(0, 3, 1, 2).float().div(255).to(device)
    img2 = torch.from_numpy(d["img2"]).permute(0, 3, 1, 2).float().div(255).to(device)
    same = torch.from_numpy(d["same"]).bool()
    gi = torch.where(same)[0].tolist()[:n_attack]
    ii = torch.where(~same)[0].tolist()[:n_attack]
    return {
        "all": (img1, img2, same),
        "dodging": (img1[gi], img2[gi]),
        "impersonation": (img1[ii], img2[ii]),
    }


def build_defenses(device, frpure_weights=None, purifier_type="unet",
                   iter_denoiser=None):
    frpure = FRPure(use_unet=frpure_weights is not None,
                    unet_weights=frpure_weights, device=device,
                    purifier_type=purifier_type)
    frpure.name = f"frpure_ours_{purifier_type}" if frpure_weights else frpure.name
    return [Identity(), FeatureSqueezing(), KimNIL(),
            IterativePurifier(denoiser=iter_denoiser, device=device), frpure]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aligned_dir"); ap.add_argument("--pairs")
    ap.add_argument("--pairs_npz", help="aligned eval pairs .npz from prepare_data.py")
    ap.add_argument("--frpure_weights", default=None,
                    help="path to trained purifier .pth; omit = training-free FRPure")
    ap.add_argument("--purifier_type", choices=["unet", "vit"], default="unet")
    ap.add_argument("--n_attack", type=int, default=200,
                    help="probes per dodging/impersonation pool")
    ap.add_argument("--iter_denoiser", default=None,
                    help="denoiser .pth used by the iterative (DiffPure-style) purifier")
    ap.add_argument("--defenses", default="none,fsqueeze,kim_nil,frpure",
                    help="comma list to include; e.g. 'none,frpure' to skip baselines")
    ap.add_argument("--backbone", default="dummy")
    ap.add_argument("--attacks", default="clean,fgsm,pgd_eot,bpda_eot,patch_eyeglass")
    ap.add_argument("--out", default="results/run.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    from frpure.models.backbone import build_backbone
    backbone = build_backbone(args.backbone, device=device) if args.backbone != "dummy" \
        else build_backbone("dummy", device=device)

    if args.synthetic:
        pools = synthetic_pools(device)
    elif args.pairs_npz:
        pools = npz_pools(args.pairs_npz, device, n_attack=args.n_attack)
    else:
        pools = lfw_pools(args.aligned_dir, args.pairs, device, n_attack=args.n_attack)

    attacks = args.attacks.split(",")
    want = set(args.defenses.split(","))
    iter_den = None
    if args.iter_denoiser:
        from frpure.defenses.smoothing import build_denoiser
        iter_den = build_denoiser("dncnn").to(device).eval()
        sd = torch.load(args.iter_denoiser, map_location=device)
        if any(k.startswith("core.") for k in sd):
            sd = {k[5:]: v for k, v in sd.items() if k.startswith("core.")}
        iter_den.load_state_dict(sd)
        for prm in iter_den.parameters():
            prm.requires_grad_(False)
    all_defenses = build_defenses(device, args.frpure_weights, args.purifier_type,
                                  iter_denoiser=iter_den)
    name_key = {"none": "none", "fsqueeze": "fsqueeze", "kim_nil": "kim_nil",
                "iter_purify": "iter_purify"}
    rows = []
    for defense in all_defenses:
        key = name_key.get(defense.name, "frpure")
        if key not in want:
            continue
        for atk in attacks:
            row = run_cell(backbone, defense, atk, pools, device)
            rows.append(row.to_dict())
            print(f"[{defense.name:>11} | {atk:>14}] "
                  f"ver_acc={row.ver_acc:.3f} robust_tar={row.robust_tar:.3f} "
                  f"asr_dodge={row.asr_dodging:.3f} asr_imp={row.asr_impersonation:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}
            w.writerow(r)
    with open(args.out.replace(".csv", ".json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()