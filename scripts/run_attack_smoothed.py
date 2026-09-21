#!/usr/bin/env python3
"""
scripts/run_attack_smoothed.py  --  EMPIRICAL robustness of the SMOOTHED verifier
under an adaptive PGD+EOT attack (the missing counterpart to run_certify.py).

For each balanced pair we:
  1. calibrate tau on CLEAN cosines at --far (same as run_certify),
  2. craft an adversarial probe with pgd_eot_smoothed (dodging for genuine
     pairs, impersonation for impostor pairs),
  3. compare the smoothed prediction on the CLEAN vs ADVERSARIAL probe, and
  4. re-certify the adversarial probe.

Reports:
  * clean smoothed accuracy               (baseline, matches run_certify)
  * robust smoothed accuracy under attack (predict() on adv probe)
  * attack success rate (ASR)             (fraction of decisions flipped wrong)
  * certified accuracy @ radius under attack (should track the certificate:
    attacks only flip predictions once budget exceeds the certified R)

NOTE: a nonzero ASR does NOT break the certificate. Cohen's guarantee holds
inside radius R; the eps=8/255 L-inf budget here has L2 norm ~1.7 for 112x112x3,
which is comparable to the mean certified R, so some flips are expected and
consistent with theory.

Run (real):
  python scripts/run_attack_smoothed.py --aligned_dir DATA/lfw_aligned_160 \
      --pairs DATA/pairs.txt --backbone facenet \
      --denoiser_weights results/denoiser_s050.pth --sigma 0.5 \
      --n_pairs 200 --n 500 --steps 40 --n_eot 8 \
      --out results/lfw_attack_smoothed.csv
CPU self-test:
  python scripts/run_attack_smoothed.py --synthetic --device cpu \
      --n_pairs 8 --n0 20 --n 60 --sigma 0.1 --steps 5 --n_eot 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import GaussianDenoiser, SmoothedVerifier
from frpure.attacks.smoothed_attack import pgd_eot_smoothed

# reuse the identical data / subset helpers from the certify script
from run_certify import load_pairs, balanced_subset, embed_all, RADII

DEFAULT_EPS_LINF = 8 / 255
DEFAULT_ALPHA_LINF = 2 / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aligned_dir"); ap.add_argument("--pairs")
    ap.add_argument("--pairs_npz")
    ap.add_argument("--bin_path")
    ap.add_argument("--backbone", default="dummy")
    ap.add_argument("--denoiser_weights", default=None)
    ap.add_argument("--pretrained", default="vggface2")
    ap.add_argument("--onnx_path"); ap.add_argument("--onnx_bgr", action="store_true")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--n0", type=int, default=100)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--alpha_cert", type=float, default=1e-3, help="Clopper-Pearson conf")
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    # attack knobs
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS_LINF)
    ap.add_argument("--step_size", type=float, default=DEFAULT_ALPHA_LINF)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--n_eot", type=int, default=8)
    ap.add_argument("--norm", default="linf", choices=["linf", "l2"])
    ap.add_argument("--out", default="results/attack_smoothed.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    bb_kw = {"device": device}
    if args.backbone == "facenet":
        bb_kw["pretrained"] = args.pretrained
    if args.backbone == "onnx":
        bb_kw["onnx_path"] = args.onnx_path; bb_kw["bgr"] = args.onnx_bgr
    backbone = build_backbone(args.backbone, **bb_kw)

    denoiser = None
    if args.denoiser_weights:
        denoiser = GaussianDenoiser(ch=args.ch).to(device).eval()
        denoiser.load_state_dict(torch.load(args.denoiser_weights, map_location=device))
        for p in denoiser.parameters():
            p.requires_grad_(False)

    img1, img2, same = load_pairs(args)
    idx = balanced_subset(same, min(args.n_pairs, len(same)))
    print(f"attacking {len(idx)} pairs  (sigma={args.sigma}, steps={args.steps}, "
          f"n_eot={args.n_eot}, eps={args.eps:.4f}, "
          f"denoiser={'yes' if denoiser else 'NONE'})")

    e1 = embed_all(backbone, img1[idx]); e2 = embed_all(backbone, img2[idx])
    cos_clean = (e1 * e2).sum(1).numpy()
    sub_same = same[idx]
    if args.tau is not None:
        tau = args.tau
    else:
        imp = cos_clean[sub_same == 0]
        tau = float(np.quantile(imp, 1 - args.far)) if len(imp) else 0.3
    print(f"tau = {tau:.4f}")

    sv = SmoothedVerifier(backbone, denoiser, sigma=args.sigma, tau=tau, device=device)

    clean_pred, robust_pred, adv_cert_pred, adv_radius, labels = [], [], [], [], []
    for j, i in enumerate(idx):
        lbl = int(sub_same[j])
        e_b = e2[j].to(device)
        probe = img1[i].to(device)

        # clean smoothed prediction (baseline)
        cp = sv.predict(probe, e_b, n=args.n, batch=args.batch)

        # attacker goal depends on the pair: break genuine (dodging) or forge impostor
        mode = "dodging" if lbl == 1 else "impersonation"
        x_adv = pgd_eot_smoothed(sv, probe, e_b, eps=args.eps,
                                 alpha=args.step_size, steps=args.steps,
                                 mode=mode, n_eot=args.n_eot, norm=args.norm)

        rp = sv.predict(x_adv, e_b, n=args.n, batch=args.batch)
        cpred, R = sv.certify(x_adv, e_b, n0=args.n0, n=args.n,
                              alpha=args.alpha_cert, batch=args.batch)

        clean_pred.append(cp); robust_pred.append(rp)
        adv_cert_pred.append(cpred); adv_radius.append(R); labels.append(lbl)
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(idx)} attacked")

    clean_pred = np.array(clean_pred); robust_pred = np.array(robust_pred)
    adv_cert_pred = np.array(adv_cert_pred); adv_radius = np.array(adv_radius)
    labels = np.array(labels)

    clean_acc = float((clean_pred == labels).mean())
    robust_acc = float((robust_pred == labels).mean())
    # ASR: of pairs correct when clean, fraction flipped wrong by the attack
    was_correct = clean_pred == labels
    asr = float(((robust_pred != labels) & was_correct).mean()) if was_correct.any() else 0.0

    not_abstain = adv_cert_pred != -1
    cert_correct = (adv_cert_pred == labels) & not_abstain
    cert_acc = {r: float((cert_correct & (adv_radius >= r)).mean()) for r in RADII}
    mean_R = float(adv_radius[cert_correct].mean()) if cert_correct.any() else 0.0

    print("\n==== SMOOTHED-MODEL ATTACK RESULTS ====")
    print(f"clean smoothed accuracy   : {clean_acc:.3f}")
    print(f"robust smoothed accuracy  : {robust_acc:.3f}   (under PGD+EOT)")
    print(f"attack success rate (ASR) : {asr:.3f}")
    print(f"mean certified R (adv)    : {mean_R:.3f}")
    print("certified accuracy @ L2 radius (on adversarial probes):")
    for r in RADII:
        print(f"   R >= {r:<4}: {cert_acc[r]:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = [{"radius": r, "certified_accuracy_under_attack": cert_acc[r]} for r in RADII]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["radius", "certified_accuracy_under_attack"])
        w.writeheader(); w.writerows(rows)
    with open(args.out.replace(".csv", ".json"), "w") as f:
        json.dump({"sigma": args.sigma, "tau": tau, "eps": args.eps,
                   "steps": args.steps, "n_eot": args.n_eot, "norm": args.norm,
                   "clean_smoothed_accuracy": clean_acc,
                   "robust_smoothed_accuracy": robust_acc,
                   "attack_success_rate": asr, "mean_radius_adv": mean_R,
                   "certified_accuracy_under_attack": cert_acc}, f, indent=2)
    print(f"\nwrote -> {args.out} (+ .json)")


if __name__ == "__main__":
    main()
