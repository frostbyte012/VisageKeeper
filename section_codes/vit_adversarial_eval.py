#!/usr/bin/env python3
"""
vit_adversarial_eval.py -- motivation-figure numbers for ViT face backbones.

Mirrors the CNN protocol in adversarial_face_eval.py EXACTLY, only the backbone
changes:
  * verification accuracy on N face pairs at a fixed cosine threshold,
  * genuine pair (label 1) -> impersonation attack (push apart),
  * impostor pair (label 0) -> dodging attack (pull together),
  * FGSM / I-FGSM (eps=0.05, 10 steps) + BPDA (eps=0.10, 10 steps, avg-pool
    blur as the non-differentiable stand-in).

Backbones: timm face-recognition ViTs (ArcFace/CosFace, MS1MV3), which are real
identity embedders (unlike ImageNet ViTs). Eval set: LFW aligned (present in
DATA/); the CNN row was on VGGFace2 -- state this dataset caveat in the caption.

Run:
  python section_codes/vit_adversarial_eval.py --aligned_dir DATA/lfw_aligned_160 \
      --num_pairs 500 --out results/vit_motivation.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTACK_STEPS = 10
COSINE_THRESHOLD = 0.6            # same operating threshold as the CNN experiment

# timm face ViTs (real face embedders). label -> (repo, display name)
VIT_MODELS = {
    "ViT-S": "hf-hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3",
    "ViT-Ti": "hf-hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3",
}


# --------------------------------------------------------------------------- #
# backbone wrapper -- same embed_tensor(x01) contract as the CNN wrappers.
# x is in [0,1] pixel space; the ViT normalizes to mean/std 0.5 internally so
# the attack perturbs honest pixel space (like the CNN _project does).
# --------------------------------------------------------------------------- #
class ViTFace(torch.nn.Module):
    def __init__(self, repo, size=112):
        super().__init__()
        import timm
        self.net = timm.create_model(repo, pretrained=True, num_classes=0).eval().to(DEVICE)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.size = size

    def embed_tensor(self, x01):
        x = x01
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                              align_corners=False)
        x = (x - 0.5) / 0.5
        return F.normalize(self.net(x), dim=1)


# --------------------------------------------------------------------------- #
# attacks -- byte-for-byte the same math as adversarial_face_eval.py, but in
# [0,1] pixel space (eps is a plain L-inf pixel budget; the CNN version used
# ImageNet-std-scaled eps, i.e. the SAME perceptual budget).
# --------------------------------------------------------------------------- #
def _project(x_adv, x_orig, eps):
    return (x_orig + (x_adv - x_orig).clamp(-eps, eps)).clamp(0, 1).detach()


def fgsm_attack(model, x, x_ref, eps, label):
    sign = -1.0 if label == 1 else +1.0
    x_adv = x.clone().detach().requires_grad_(True)
    (model.embed_tensor(x_adv) * model.embed_tensor(x_ref).detach()).sum().backward()
    x_pert = x.detach() + sign * eps * x_adv.grad.sign()
    return _project(x_pert, x.detach(), eps)


def ifgsm_attack(model, x, x_ref, eps, label, steps=ATTACK_STEPS):
    alpha = eps / steps
    sign = -1.0 if label == 1 else +1.0
    with torch.no_grad():
        emb_ref = model.embed_tensor(x_ref).detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        (model.embed_tensor(x_adv) * emb_ref).sum().backward()
        with torch.no_grad():
            x_adv = _project(x_adv.detach() + sign * alpha * x_adv.grad.sign(),
                             x.detach(), eps)
    return x_adv


def bpda_attack(model, x, x_ref, eps, label, steps=ATTACK_STEPS):
    alpha = eps / steps
    sign = -1.0 if label == 1 else +1.0
    with torch.no_grad():
        emb_ref = model.embed_tensor(x_ref).detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.no_grad():
            x_blur = F.avg_pool2d(x_adv.detach(), 3, 1, 1)
        x_bpda = x_adv + (x_blur - x_adv).detach()   # BPDA straight-through
        (model.embed_tensor(x_bpda) * emb_ref).sum().backward()
        with torch.no_grad():
            x_adv = _project(x_adv.detach() + sign * alpha * x_adv.grad.sign(),
                             x.detach(), eps)
    return x_adv


ATTACKS = {
    "FGSM (eps=0.05)":  lambda m, x, xr, l: fgsm_attack(m, x, xr, 0.05, l),
    "iFGSM (eps=0.05)": lambda m, x, xr, l: ifgsm_attack(m, x, xr, 0.05, l),
    "BPDA (eps=0.10)":  lambda m, x, xr, l: bpda_attack(m, x, xr, 0.10, l),
}
COLUMNS = ["Base"] + list(ATTACKS.keys())


# --------------------------------------------------------------------------- #
# data / eval -- same pairing scheme as load_vggface2_pairs
# --------------------------------------------------------------------------- #
def load_pairs(aligned_dir, num_pairs, seed=42):
    random.seed(seed)
    id_to_imgs = {}
    for d in sorted(glob.glob(os.path.join(aligned_dir, "*"))):
        imgs = sorted(glob.glob(os.path.join(d, "*.png"))) + \
            sorted(glob.glob(os.path.join(d, "*.jpg")))
        if len(imgs) >= 2:
            id_to_imgs[d] = imgs
    ids = list(id_to_imgs.keys())
    pairs, half = [], num_pairs // 2
    for _ in range(half):
        i = random.choice(ids)
        a, b = random.sample(id_to_imgs[i], 2)
        pairs.append((a, b, 1))
    for _ in range(num_pairs - half):
        i1, i2 = random.sample(ids, 2)
        pairs.append((random.choice(id_to_imgs[i1]), random.choice(id_to_imgs[i2]), 0))
    random.shuffle(pairs)
    gen = sum(1 for *_, l in pairs if l == 1)
    print(f"[INFO] {len(pairs)} pairs -- {gen} genuine / {num_pairs - gen} impostor")
    return pairs


def load_img(path):
    img = Image.open(path).convert("RGB").resize((112, 112))
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)
    return t.unsqueeze(0).to(DEVICE)


def classify(e1, e2, thr):
    return 1 if F.cosine_similarity(e1, e2).item() >= thr else 0


def evaluate(model, pairs, thr):
    tal = {k: [0, 0] for k in COLUMNS}
    for p1, p2, label in pairs:
        x1, x2 = load_img(p1), load_img(p2)
        with torch.no_grad():
            e1, e2 = model.embed_tensor(x1), model.embed_tensor(x2)
        tal["Base"][0] += int(classify(e1, e2, thr) == label)
        tal["Base"][1] += 1
        for name, fn in ATTACKS.items():
            x1a = fn(model, x1, x2, label)
            with torch.no_grad():
                e1a, e2c = model.embed_tensor(x1a), model.embed_tensor(x2)
            tal[name][0] += int(classify(e1a, e2c, thr) == label)
            tal[name][1] += 1
    return {k: (v[0] / v[1] * 100 if v[1] else float("nan")) for k, v in tal.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--num_pairs", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=COSINE_THRESHOLD)
    ap.add_argument("--out", default="results/vit_motivation.csv")
    args = ap.parse_args()

    pairs = load_pairs(args.aligned_dir, args.num_pairs)
    results = {}
    for label, repo in VIT_MODELS.items():
        print(f"\n[INFO] Evaluating {label}  ({repo})")
        model = ViTFace(repo)
        res = evaluate(model, pairs, args.threshold)
        results[label] = res
        for c in COLUMNS:
            print(f"    {c:<18} {res[c]:6.2f}")
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + COLUMNS)
        for name, res in results.items():
            w.writerow([name] + [f"{res[c]:.2f}" for c in COLUMNS])
    print(f"\n[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()
