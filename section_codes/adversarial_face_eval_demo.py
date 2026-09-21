"""
adversarial_face_eval_demo.py
─────────────────────────────
Smoke-test / dry-run of the full evaluation pipeline.
Uses randomly-initialised backbone stubs so you can verify the
attack loop, pair sampling, and result table WITHOUT needing:
  • real model weights (.pkl files)
  • the VGGFace2 dataset

All 5 model names match the paper exactly:
  ResNet50-FT, SENet50-FT, ResNet50-Scratch, SENet50-Scratch, FaceNet

Usage:
    python adversarial_face_eval_demo.py
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE         = 224
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
COSINE_THRESHOLD = 0.5
ATTACK_STEPS     = 10
NUM_PAIRS        = 60    # kept small for quick demo

print(f"[INFO] Using device: {DEVICE}")

base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Normalisation helpers ─────────────────────────────────────

def denorm(t):
    m = torch.tensor(IMAGENET_MEAN, device=t.device).view(1,3,1,1)
    s = torch.tensor(IMAGENET_STD,  device=t.device).view(1,3,1,1)
    return t * s + m

def renorm(t):
    m = torch.tensor(IMAGENET_MEAN, device=t.device).view(1,3,1,1)
    s = torch.tensor(IMAGENET_STD,  device=t.device).view(1,3,1,1)
    return (t - m) / s

def std_t():
    return torch.tensor(IMAGENET_STD, device=DEVICE).view(1,3,1,1)

def project(x_adv, x_orig, eps_norm):
    delta  = (x_adv - x_orig).clamp(-eps_norm, eps_norm)
    return renorm(denorm(x_orig + delta).clamp(0,1)).detach()

def cosine_dist_loss(e1, e2):
    return F.cosine_similarity(e1, e2).mean()

# ── Stub models ───────────────────────────────────────────────

class StubModel(nn.Module):
    """Random ResNet-18 → 512-d L2-normalised embedding. Stand-in for any FR model."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        base    = models.resnet18(weights=None)
        base.fc = nn.Linear(512, 512)
        self.net = base.to(DEVICE).eval()

    def embed_tensor(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(x.requires_grad):
            return F.normalize(self.net(x), dim=1)

MODEL_NAMES = [
    "ResNet50-FT",
    "SENet50-FT",
    "ResNet50-Scratch",
    "SENet50-Scratch",
    "FaceNet",
]

# ── Attacks ───────────────────────────────────────────────────

def fgsm(model, x, x_ref, epsilon):
    eps_n = epsilon / std_t()
    x_adv = x.clone().detach().requires_grad_(True)
    loss  = cosine_dist_loss(model.embed_tensor(x_adv),
                              model.embed_tensor(x_ref).detach())
    loss.backward()
    return project(x.detach() + eps_n * x_adv.grad.sign(), x.detach(), eps_n)

def ifgsm(model, x, x_ref, epsilon, steps=ATTACK_STEPS):
    alpha   = epsilon / steps
    eps_n   = epsilon / std_t()
    alpha_n = alpha   / std_t()
    emb_ref = model.embed_tensor(x.clone().detach()).detach()
    x_adv   = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        cosine_dist_loss(model.embed_tensor(x_adv), emb_ref).backward()
        with torch.no_grad():
            x_adv = project(x_adv.detach() + alpha_n * x_adv.grad.sign(),
                            x.detach(), eps_n)
    return x_adv

def bpda(model, x, x_ref, epsilon, steps=ATTACK_STEPS):
    alpha   = epsilon / steps
    eps_n   = epsilon / std_t()
    alpha_n = alpha   / std_t()
    emb_ref = model.embed_tensor(x.clone().detach()).detach()
    x_adv   = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.no_grad():
            x_blur = F.avg_pool2d(x_adv.detach(), 3, 1, 1)
        x_bpda = x_adv + (x_blur - x_adv).detach()
        cosine_dist_loss(model.embed_tensor(x_bpda), emb_ref).backward()
        with torch.no_grad():
            x_adv = project(x_adv.detach() + alpha_n * x_adv.grad.sign(),
                            x.detach(), eps_n)
    return x_adv

ATTACKS = {
    "FGSM (ε=0.05)":    lambda m,x,xr: fgsm (m, x, xr, 0.05),
    "FGSM (ε=0.10)":    lambda m,x,xr: fgsm (m, x, xr, 0.10),
    "I-FGSM (ε=0.05)":  lambda m,x,xr: ifgsm(m, x, xr, 0.05),
    "I-FGSM (ε=0.10)":  lambda m,x,xr: ifgsm(m, x, xr, 0.10),
    "BPDA (ε=0.10)":    lambda m,x,xr: bpda (m, x, xr, 0.10),
}

COLUMNS = ["clean"] + list(ATTACKS.keys())

# ── Synthetic pair generator ──────────────────────────────────

def make_pairs(n=NUM_PAIRS, seed=42):
    """Pairs are (rng_seed_1, rng_seed_2, label).
    Same seed ≈ same identity; different seed ≈ different identity."""
    random.seed(seed)
    pairs, half = [], n // 2
    for _ in range(half):           # genuine
        s = random.randint(0, 9999)
        pairs.append((s, s, 1))
    for _ in range(n - half):       # impostor
        s1 = random.randint(0, 9999)
        s2 = random.randint(10000, 19999)
        pairs.append((s1, s2, 0))
    random.shuffle(pairs)
    return pairs

def seed_to_tensor(seed: int) -> torch.Tensor:
    arr = np.random.default_rng(seed).integers(0, 255,
          (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return base_transform(Image.fromarray(arr)).unsqueeze(0).to(DEVICE)

# ── Evaluation ────────────────────────────────────────────────

def classify(e1, e2):
    return 1 if F.cosine_similarity(
        F.normalize(e1,dim=1), F.normalize(e2,dim=1)).item() >= COSINE_THRESHOLD else 0

def evaluate(model, pairs):
    tallies = {k: [0, 0] for k in COLUMNS}
    for s1, s2, label in tqdm(pairs, desc=f"  {model.name}", leave=False):
        x1 = seed_to_tensor(s1)
        x2 = seed_to_tensor(s2)
        with torch.no_grad():
            e1 = model.embed_tensor(x1)
            e2 = model.embed_tensor(x2)
        tallies["clean"][0] += int(classify(e1, e2) == label)
        tallies["clean"][1] += 1
        for atk_name, atk_fn in ATTACKS.items():
            x1_adv = atk_fn(model, x1, x2)
            with torch.no_grad():
                e1_adv = model.embed_tensor(x1_adv)
                e2_c   = model.embed_tensor(x2)
            tallies[atk_name][0] += int(classify(e1_adv, e2_c) == label)
            tallies[atk_name][1] += 1
    return {k: v[0]/v[1]*100 for k, v in tallies.items()}

# ── Main ──────────────────────────────────────────────────────

def print_table(results):
    col_w = 16
    width = 22 + col_w * len(COLUMNS)
    print("\n" + "=" * width)
    print("DEMO RESULTS (random weights / synthetic data)")
    print("Verification Accuracy (%) on VGGFace2  [pipeline smoke-test]")
    print("=" * width)
    print(f"{'Model':<22}" + "".join(f"{c:>{col_w}}" for c in COLUMNS))
    print("-" * width)
    for name, res in results.items():
        print(f"{name:<22}" + "".join(f"{res[c]:>{col_w}.2f}" for c in COLUMNS))
    print("=" * width)

print(f"\n[INFO] Generating {NUM_PAIRS} synthetic pairs ...")
pairs = make_pairs(NUM_PAIRS)
gen   = sum(1 for *_,l in pairs if l==1)
imp   = sum(1 for *_,l in pairs if l==0)
print(f"[INFO] {len(pairs)} pairs — {gen} genuine / {imp} impostor")

all_results = {}
for name in MODEL_NAMES:
    print(f"\n[INFO] Evaluating stub model: {name}")
    model = StubModel(name)
    all_results[name] = evaluate(model, pairs)
    del model

print_table(all_results)
print("\n[OK] Pipeline verified.")
print("     Replace StubModel with real weights and run adversarial_face_eval.py")
print("     with your VGGFace2 --data_dir for paper results.")