"""
frpure.models.backbone
======================
FR backbones expose a single differentiable interface so attacks can backprop
through model (and, for adaptive attacks, through defense -> model).

Convention everywhere in this repo:
  images are float tensors, shape (B, 3, H, W), pixel range [0, 1].
Each backbone resizes / normalizes internally to whatever it was trained on.

Why FaceNet is the *primary* attack backbone
---------------------------------------------
White-box adaptive attacks (PGD+EOT through the defense) need gradients through
the FR model. insightface's ArcFace ships as ONNX -> not differentiable, so it
is great as a *transfer / black-box* eval target but cannot generate white-box
adversaries directly. facenet-pytorch (InceptionResnetV1) is fully torch,
weights auto-download, and is the de-facto differentiable FR model in the
adversarial-FR literature. We attack FaceNet white-box and *transfer* to ArcFace
for the black-box columns of the matrix.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FRBackbone(nn.Module):
    """Base: subclasses implement `_forward_features` on a model-space tensor.

    `embed` returns L2-normalized embeddings so cosine similarity == dot product.
    Keep everything differentiable.
    """

    input_size: int = 160          # H=W the backbone expects
    embed_dim: int = 512

    def preprocess(self, imgs01: torch.Tensor) -> torch.Tensor:
        """[0,1] (B,3,H,W) -> model space. Override per-backbone normalization."""
        x = imgs01
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def embed(self, imgs01: torch.Tensor) -> torch.Tensor:
        feat = self._forward_features(self.preprocess(imgs01))
        return F.normalize(feat, p=2, dim=1)

    @torch.no_grad()
    def cosine(self, a01: torch.Tensor, b01: torch.Tensor) -> torch.Tensor:
        return (self.embed(a01) * self.embed(b01)).sum(dim=1)


class FaceNetBackbone(FRBackbone):
    """InceptionResnetV1 from facenet-pytorch. Differentiable. weights auto-DL."""

    input_size = 160
    embed_dim = 512

    def __init__(self, pretrained: Literal["vggface2", "casia-webface"] = "vggface2",
                 device: str = "cuda"):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1  # lazy import
        self.net = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)            # model frozen; we attack the INPUT
        self.device = device

    def preprocess(self, imgs01: torch.Tensor) -> torch.Tensor:
        x = super().preprocess(imgs01)
        return x * 2.0 - 1.0                   # facenet expects ~[-1,1]

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArcFaceBackbone(FRBackbone):
    """ArcFace IR backbone. Two modes:
      * 'onnx'  : insightface buffalo_l recognition model -- NON-differentiable,
                  use only for transfer / black-box scoring (no white-box attack).
      * 'torch' : insightface arcface_torch IResNet, differentiable, if a .pth
                  weight path is supplied (download on your server).
    """

    input_size = 112
    embed_dim = 512

    def __init__(self, mode: Literal["onnx", "torch"] = "onnx",
                 torch_weights: str | None = None, arch: str = "r100",
                 device: str = "cuda"):
        super().__init__()
        self.mode = mode
        self.device = device
        if mode == "torch":
            from frpure.models.iresnet import iresnet  # local IResNet defs
            self.net = iresnet(arch).eval().to(device)
            if torch_weights:
                self.net.load_state_dict(torch.load(torch_weights, map_location=device))
            for p in self.net.parameters():
                p.requires_grad_(False)
        else:
            import insightface
            self.app = insightface.app.FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0 if device.startswith("cuda") else -1)

    def preprocess(self, imgs01: torch.Tensor) -> torch.Tensor:
        x = super().preprocess(imgs01)
        return (x - 0.5) / 0.5                  # arcface_torch normalization

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode != "torch":
            raise RuntimeError("ONNX ArcFace is non-differentiable; use mode='torch' "
                               "for white-box attacks, or score via embed_numpy().")
        return self.net(x)


class DummyBackbone(FRBackbone):
    """Tiny differentiable CNN -> embedding. CPU-only, no weights, for tests.

    Identity is faked by a fixed random projection of a downsampled image, so
    two crops of the 'same' synthetic identity land near each other.
    """

    input_size = 64
    embed_dim = 128

    def __init__(self, device: str = "cpu", seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(32 * 4 * 4, self.embed_dim)
        for p in self.parameters():
            p.requires_grad_(False)
        # deterministic-ish init so embeddings are stable across calls
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.1, generator=g)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.to(device)
        self.device = device

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.fc(h)


def build_backbone(name: str, **kw) -> FRBackbone:
    name = name.lower()
    if name == "facenet":
        return FaceNetBackbone(**kw)
    if name == "arcface":
        return ArcFaceBackbone(**kw)
    if name == "dummy":
        return DummyBackbone(**kw)
    if name == "onnx":
        from frpure.models.onnx_backbone import OnnxBackbone
        return OnnxBackbone(**kw)
    raise ValueError(f"unknown backbone {name!r}")