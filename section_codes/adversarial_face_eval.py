"""
Adversarial Attack Evaluation on Face Recognition Models
=========================================================
Models:
  1. ResNet50-FT       — resnet50_ft_weight.pkl
  2. SENet50-FT        — senet50_ft_weight.pkl
  3. ResNet50-Scratch  — resnet50_scratch_weight.pkl
  4. SENet50-Scratch   — senet50_scratch_weight.pkl
  5. FaceNet           — InceptionResnetV1 (facenet-pytorch, auto-downloads)

Backbone architecture confirmed from pkl inspection:
  • ResNet50 : standard torchvision ResNet-50 layout (layer1..layer4 + fc)
               fc.weight shape (8631, 2048) → we use avgpool output as embedding
  • SENet50  : ResNet-50 with SE blocks using conv4 (squeeze) + conv5 (excite)
               per block — NOT timm legacy_seresnet50; custom implementation below.

Usage:
    python adversarial_face_eval.py \
        --data_dir  vggface2_112x112/ \
        --weights   weights/ \
        --output    results.csv \
        --num_pairs 1000
"""

import csv, argparse, random, pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ══════════════════════════════════════════════════════════════
# 0.  CONFIG
# ══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

IMG_SIZE         = 224
FACENET_SIZE     = 160
VGG_MEAN_RGB     = [91.4953, 103.8827, 131.0912]   # R, G, B  (0-255)
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
COSINE_THRESHOLD = 0.6
ATTACK_STEPS     = 10

base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ══════════════════════════════════════════════════════════════
# 1.  NORMALISATION HELPERS
# ══════════════════════════════════════════════════════════════

def denorm(t):
    m = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=t.device).view(1,3,1,1)
    s = torch.tensor(IMAGENET_STD,  dtype=torch.float32, device=t.device).view(1,3,1,1)
    return t * s + m

def renorm(t):
    m = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=t.device).view(1,3,1,1)
    s = torch.tensor(IMAGENET_STD,  dtype=torch.float32, device=t.device).view(1,3,1,1)
    return (t - m) / s

def to_vgg_space(x):
    """ImageNet-normalised tensor → VGGFace2 input (BGR mean-subtracted, 0-255)."""
    pixel = denorm(x).clamp(0, 1) * 255.0
    mean  = torch.tensor(VGG_MEAN_RGB, dtype=torch.float32,
                         device=x.device).view(1,3,1,1)
    return pixel - mean

# ══════════════════════════════════════════════════════════════
# 2.  RESNET-50 BACKBONE
#     Identical to torchvision ResNet-50.  We keep the fc layer
#     so strict=True loading works, then extract avgpool output
#     as the 2048-d embedding.
# ══════════════════════════════════════════════════════════════

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        # Keep ALL layers including fc (8631 classes) so weights load cleanly
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        self.fc      = nn.Linear(2048, 8631)   # matches pkl fc.weight (8631,2048)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return x.flatten(1)   # (B, 2048) — embedding before fc


# ══════════════════════════════════════════════════════════════
# 3.  SENET-50 BACKBONE
#
#  From the pkl inspection:
#    • Standard ResNet-50 conv1/bn1/layer1..4 layout  ✓
#    • Each bottleneck block has EXTRA keys:
#        layerX.Y.conv4   (squeeze): shape (r, C, 1, 1)  where r = C//16
#        layerX.Y.conv5   (excite) : shape (C, r, 1, 1)
#      e.g. layer3.1.conv4.weight  (64, 1024, 1,1)   r=64=1024//16
#           layer3.1.conv5.weight  (1024, 64, 1,1)  ← excite back to C
#    • No separate BN on SE path (bias only on conv4/conv5)
#
#  We build a Bottleneck with an SE block matching this exactly.
# ══════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block whose weights match the pkl layout:
        conv4: (planes, in_channels, 1, 1)   — global avg pool + squeeze FC
        conv5: (in_channels, planes, 1, 1)   — excite FC
    Both use bias (confirmed by conv4.bias in pkl).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = channels // reduction
        # conv4 = squeeze:  channels → r
        self.conv4 = nn.Conv2d(channels, r, kernel_size=1, bias=True)
        # conv5 = excite:   r → channels
        self.conv5 = nn.Conv2d(r, channels, kernel_size=1, bias=True)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)   # global avg pool
        s = F.relu(self.conv4(s), inplace=True)
        s = torch.sigmoid(self.conv5(s))
        return x * s


class SEBottleneck(nn.Module):
    """
    ResNet-50 Bottleneck + SE block.
    Keys per block in pkl:
        conv1, bn1, conv2, bn2, conv3, bn3   — standard bottleneck
        conv4, conv5                          — SE squeeze / excite
        downsample.0 / downsample.1           — optional projection
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.se    = SEBlock(planes * self.expansion, reduction)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

    def remap_keys(self, prefix, state):
        """
        The SE block keys live directly on the block (conv4 / conv5),
        but in our nn.Module they live under self.se.conv4 / self.se.conv5.
        Remap so load_state_dict works with strict=True.
        """
        remapped = {}
        for k, v in state.items():
            if k.startswith(prefix + ".conv4") or k.startswith(prefix + ".conv5"):
                # e.g. "layer1.0.conv4.weight" → "layer1.0.se.conv4.weight"
                new_k = k.replace(prefix + ".conv4", prefix + ".se.conv4", 1)
                new_k = new_k.replace(prefix + ".conv5", prefix + ".se.conv5", 1)
                remapped[new_k] = v
            else:
                remapped[k] = v
        return remapped


def _make_se_layer(inplanes, planes, blocks, stride=1, reduction=16):
    downsample = None
    outplanes  = planes * SEBottleneck.expansion
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride=stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )
    layers = [SEBottleneck(inplanes, planes, stride, downsample, reduction)]
    for _ in range(1, blocks):
        layers.append(SEBottleneck(outplanes, planes, reduction=reduction))
    return nn.Sequential(*layers)


class SENet50(nn.Module):
    """
    SE-ResNet-50 whose state_dict keys match the VGGFace2 pkl files exactly
    after the conv4/conv5 → se.conv4/se.conv5 remapping.
    """
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1  = _make_se_layer(64,   64,  3)
        self.layer2  = _make_se_layer(256,  128, 4, stride=2)
        self.layer3  = _make_se_layer(512,  256, 6, stride=2)
        self.layer4  = _make_se_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(2048, 8631)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return x.flatten(1)   # (B, 2048)


# ══════════════════════════════════════════════════════════════
# 4.  WEIGHT LOADING
# ══════════════════════════════════════════════════════════════

def remap_se_keys(state: dict) -> dict:
    """
    pkl keys:   layerX.Y.conv4  /  layerX.Y.conv5
    our keys:   layerX.Y.se.conv4  /  layerX.Y.se.conv5
    """
    out = {}
    for k, v in state.items():
        nk = k
        # Match pattern  layerX.Y.conv4  or  layerX.Y.conv5
        parts = k.split(".")
        if (len(parts) >= 3
                and parts[0].startswith("layer")
                and parts[2] in ("conv4", "conv5")):
            parts.insert(2, "se")   # inject 'se' before conv4/conv5
            nk = ".".join(parts)
        out[nk] = v
    return out


def load_pkl(model: nn.Module, pkl_path: str, is_senet: bool = False):
    with open(pkl_path, "rb") as f:
        state = pickle.load(f, encoding="latin1")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip DataParallel prefix
    state = {(k[7:] if k.startswith("module.") else k): v
             for k, v in state.items()}

    # Convert numpy arrays
    state = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
             for k, v in state.items()}

    # SE key remapping
    if is_senet:
        state = remap_se_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Only warn about non-fc mismatches
    real_missing    = [k for k in missing    if "fc" not in k]
    real_unexpected = [k for k in unexpected if "fc" not in k]

    if real_missing:
        print(f"    [WARN] {len(real_missing)} missing keys (ex: {real_missing[:3]})")
    if real_unexpected:
        print(f"    [WARN] {len(real_unexpected)} unexpected keys "
              f"(ex: {real_unexpected[:3]})")

    loaded = len(state) - len(unexpected)
    print(f"    Loaded {loaded}/{len(state)} tensors from {Path(pkl_path).name}")


# ══════════════════════════════════════════════════════════════
# 5.  MODEL WRAPPERS  (.embed_tensor → L2-normalised 2048-d vec)
# ══════════════════════════════════════════════════════════════

class VGGFace2Wrapper(nn.Module):
    def __init__(self, name: str, backbone: nn.Module, pkl_path: str,
                 is_senet: bool = False):
        super().__init__()
        self.name     = name
        self.backbone = backbone
        print(f"  [{name}] Loading weights ...")
        load_pkl(self.backbone, pkl_path, is_senet=is_senet)
        self.backbone = self.backbone.to(DEVICE).eval()

    def embed_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(to_vgg_space(x)), dim=1)

    def forward(self, x):
        return self.embed_tensor(x)


class FaceNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "FaceNet"
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError("pip install facenet-pytorch")
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    def embed_tensor(self, x: torch.Tensor) -> torch.Tensor:
        pixel   = denorm(x)          # no clamp — keep gradients intact
        resized = F.interpolate(pixel, size=(FACENET_SIZE, FACENET_SIZE),
                                mode="bilinear", align_corners=False)
        return F.normalize(self.net((resized - 0.5) / 0.5), dim=1)

    def forward(self, x):
        return self.embed_tensor(x)


# ══════════════════════════════════════════════════════════════
# 6.  MODEL FACTORY
# ══════════════════════════════════════════════════════════════

PKL_CFG = {
    "ResNet50-FT":      ("resnet50_ft_weight.pkl",      False),
    "SENet50-FT":       ("senet50_ft_weight.pkl",       True),
    "ResNet50-Scratch": ("resnet50_scratch_weight.pkl", False),
    "SENet50-Scratch":  ("senet50_scratch_weight.pkl",  True),
}

def build_models(weights_dir: str) -> dict:
    weights_dir = Path(weights_dir)
    loaded = {}

    for name, (pkl_name, is_senet) in PKL_CFG.items():
        pkl_path = weights_dir / pkl_name
        if not pkl_path.exists():
            print(f"  [SKIP] {name}: {pkl_path} not found")
            continue
        backbone = SENet50() if is_senet else ResNet50()
        try:
            loaded[name] = VGGFace2Wrapper(name, backbone, str(pkl_path), is_senet)
            print(f"  [{name}] Ready.")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    try:
        print("  [FaceNet] Loading (auto-download if needed) ...")
        loaded["FaceNet"] = FaceNetWrapper()
        print("  [FaceNet] Ready.")
    except Exception as e:
        print(f"  [ERROR] FaceNet: {e}")

    return loaded


# ══════════════════════════════════════════════════════════════
# 7.  ATTACKS
# ══════════════════════════════════════════════════════════════

def _en(epsilon, device):
    """epsilon in pixel space → per-channel normalised space."""
    return epsilon / torch.tensor(IMAGENET_STD, dtype=torch.float32,
                                   device=device).view(1,3,1,1)

def _project(x_adv, x_orig, eps_n):
    delta = (x_adv - x_orig).clamp(-eps_n, eps_n)
    return renorm(denorm(x_orig + delta).clamp(0, 1)).detach()

def cosine_sim(e1, e2):
    return F.cosine_similarity(e1, e2).mean()

# ── Label-aware attack helpers ─────────────────────────────────────────────
# For a verification task, "worst-case" means:
#   genuine pair  (label=1): PUSH embeddings apart  → minimise similarity
#   impostor pair (label=0): PULL embeddings together → maximise similarity
#
# We implement this by passing `sign` to each attack:
#   sign = -1  → gradient step minimises similarity  (genuine: push apart)
#   sign = +1  → gradient step maximises similarity  (impostor: pull together)
#
# In both cases accuracy drops because the model makes the wrong decision.
# ──────────────────────────────────────────────────────────────────────────

def fgsm_attack(model, x, x_ref, epsilon, label):
    """
    label=1 (genuine)  → push x away from x_ref  (sign = -1)
    label=0 (impostor) → pull x toward x_ref      (sign = +1)
    """
    eps_n  = _en(epsilon, x.device)
    sign   = -1.0 if label == 1 else +1.0
    x_adv  = x.clone().detach().requires_grad_(True)
    # We always differentiate through cosine_sim; sign flips the step direction
    loss   = cosine_sim(model.embed_tensor(x_adv),
                        model.embed_tensor(x_ref).detach())
    loss.backward()
    with torch.no_grad():
        # sign=-1: subtract → reduce similarity (push apart)
        # sign=+1: add      → increase similarity (pull together)
        x_pert = x.detach() + sign * eps_n * x_adv.grad.sign()
    return _project(x_pert, x.detach(), eps_n)

def ifgsm_attack(model, x, x_ref, epsilon, label, steps=ATTACK_STEPS):
    eps_n   = _en(epsilon, x.device)
    alpha_n = _en(epsilon / steps, x.device)
    sign    = -1.0 if label == 1 else +1.0
    with torch.no_grad():
        emb_ref = model.embed_tensor(x_ref).detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        cosine_sim(model.embed_tensor(x_adv), emb_ref).backward()
        with torch.no_grad():
            x_adv = _project(x_adv.detach() + sign * alpha_n * x_adv.grad.sign(),
                             x.detach(), eps_n)
    return x_adv

def bpda_attack(model, x, x_ref, epsilon, label, steps=ATTACK_STEPS):
    eps_n   = _en(epsilon, x.device)
    alpha_n = _en(epsilon / steps, x.device)
    sign    = -1.0 if label == 1 else +1.0
    with torch.no_grad():
        emb_ref = model.embed_tensor(x_ref).detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.no_grad():
            x_blur = F.avg_pool2d(x_adv.detach(), 3, 1, 1)
        x_bpda = x_adv + (x_blur - x_adv).detach()
        cosine_sim(model.embed_tensor(x_bpda), emb_ref).backward()
        with torch.no_grad():
            x_adv = _project(x_adv.detach() + sign * alpha_n * x_adv.grad.sign(),
                             x.detach(), eps_n)
    return x_adv

# Attack registry — label passed at call time inside evaluate_model
ATTACKS = {
    "FGSM (ε=0.05)":    lambda m,x,xr,lbl: fgsm_attack (m,x,xr, 0.05, lbl),
    "FGSM (ε=0.10)":    lambda m,x,xr,lbl: fgsm_attack (m,x,xr, 0.10, lbl),
    "I-FGSM (ε=0.05)":  lambda m,x,xr,lbl: ifgsm_attack(m,x,xr, 0.05, lbl),
    "I-FGSM (ε=0.10)":  lambda m,x,xr,lbl: ifgsm_attack(m,x,xr, 0.10, lbl),
    "BPDA (ε=0.10)":    lambda m,x,xr,lbl: bpda_attack (m,x,xr, 0.10, lbl),
}

# ══════════════════════════════════════════════════════════════
# 8.  DATASET
# ══════════════════════════════════════════════════════════════

def load_vggface2_pairs(data_dir, num_pairs=1000, seed=42):
    random.seed(seed)
    data_dir = Path(data_dir)
    id_to_imgs = {}
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir(): continue
        imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        if len(imgs) >= 2:
            id_to_imgs[d.name] = imgs
    if len(id_to_imgs) < 2:
        raise ValueError(f"Need ≥2 identity folders in {data_dir}")
    ids   = list(id_to_imgs.keys())
    pairs = []
    half  = num_pairs // 2
    for _ in range(half):
        i = random.choice(ids)
        a, b = random.sample(id_to_imgs[i], 2)
        pairs.append((str(a), str(b), 1))
    for _ in range(num_pairs - half):
        i1, i2 = random.sample(ids, 2)
        pairs.append((str(random.choice(id_to_imgs[i1])),
                      str(random.choice(id_to_imgs[i2])), 0))
    random.shuffle(pairs)
    gen = sum(1 for *_,l in pairs if l==1)
    print(f"[INFO] {len(pairs)} pairs — {gen} genuine / {num_pairs-gen} impostor")
    return pairs

def load_image(path):
    return base_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

# ══════════════════════════════════════════════════════════════
# 9.  EVALUATION
# ══════════════════════════════════════════════════════════════

COLUMNS = ["clean"] + list(ATTACKS.keys())

def classify(e1, e2, thr):
    return 1 if F.cosine_similarity(e1, e2).item() >= thr else 0

def evaluate_model(model, pairs, thr):
    tallies = {k: [0,0] for k in COLUMNS}
    for p1, p2, label in tqdm(pairs, desc=f"  {model.name}", leave=False):
        try:
            x1, x2 = load_image(p1), load_image(p2)
            with torch.no_grad():
                e1, e2 = model.embed_tensor(x1), model.embed_tensor(x2)
            tallies["clean"][0] += int(classify(e1, e2, thr) == label)
            tallies["clean"][1] += 1
            for atk_name, atk_fn in ATTACKS.items():
                # Pass label so attack direction matches pair type:
                #   genuine  (1) → push apart   (impersonation attack)
                #   impostor (0) → pull together (dodging attack)
                x1_adv = atk_fn(model, x1, x2, label)
                with torch.no_grad():
                    e1a = model.embed_tensor(x1_adv)
                    e2c = model.embed_tensor(x2)
                tallies[atk_name][0] += int(classify(e1a, e2c, thr) == label)
                tallies[atk_name][1] += 1
        except Exception as exc:
            print(f"\n  [WARN] Skipping ({Path(p1).name},{Path(p2).name}): {exc}")
    return {k: (v[0]/v[1]*100 if v[1]>0 else float("nan"))
            for k,v in tallies.items()}

# ══════════════════════════════════════════════════════════════
# 10. OUTPUT
# ══════════════════════════════════════════════════════════════

COL_W = 16

def print_table(results):
    w = 22 + COL_W * len(COLUMNS)
    print("\n" + "="*w)
    print("RESULTS — Verification Accuracy (%) on VGGFace2 Test Set")
    print("="*w)
    print(f"{'Model':<22}" + "".join(f"{c:>{COL_W}}" for c in COLUMNS))
    print("-"*w)
    for name, res in results.items():
        print(f"{name:<22}" + "".join(
            f"{res.get(c,float('nan')):>{COL_W}.2f}" for c in COLUMNS))
    print("="*w)

def save_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + COLUMNS)
        for name, res in results.items():
            w.writerow([name] + [f"{res.get(c,float('nan')):.2f}" for c in COLUMNS])
    print(f"[INFO] CSV saved → {path}")

# ══════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--weights",   default="weights/")
    parser.add_argument("--output",    default="results.csv")
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD)
    parser.add_argument("--models",    nargs="+",
                        default=["ResNet50-FT","SENet50-FT",
                                 "ResNet50-Scratch","SENet50-Scratch","FaceNet"])
    args = parser.parse_args()

    pairs = load_vggface2_pairs(args.data_dir, args.num_pairs)

    print("\n[INFO] Building models ...")
    all_models  = build_models(args.weights)
    eval_models = {k:v for k,v in all_models.items() if k in args.models}
    if not eval_models:
        print("[ERROR] No models loaded."); return

    all_results = {}
    for name, model in eval_models.items():
        print(f"\n[INFO] ── Evaluating: {name} ──")
        res = evaluate_model(model, pairs, args.threshold)
        all_results[name] = res
        print(f"  {'Condition':<22} {'Accuracy (%)':>12}")
        print("  " + "-"*36)
        for col in COLUMNS:
            print(f"  {col:<22} {res.get(col,float('nan')):>12.2f}")
        del model; torch.cuda.empty_cache()

    print_table(all_results)
    save_csv(all_results, args.output)

if __name__ == "__main__":
    main()