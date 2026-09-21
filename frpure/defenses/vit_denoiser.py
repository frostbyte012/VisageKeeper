"""
frpure.defenses.vit_denoiser -- ViT-based alternative to TinyUNet.

Same interface as TinyUNet in frpure.py: forward(x) maps (B,3,H,W) -> (B,3,H,W),
where x is the masked high-frequency band. Drop-in replacement so it plugs into
FRPure.purify and scripts/train_purifier.py unchanged.

Architecture: conv patch-embed -> transformer encoder over patch tokens ->
conv-transpose head back to pixel space. Input H,W must be divisible by `patch`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTDenoiser(nn.Module):
    def __init__(self, patch: int = 8, dim: int = 192, depth: int = 4,
                 heads: int = 6, mlp_ratio: float = 4.0, pos_mode: str = "2d"):
        super().__init__()
        self.patch = patch
        self.dim = dim
        self.pos_mode = pos_mode   # "2d" (new) | "1d" (reproduces old checkpoints)
        self.embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.unembed = nn.ConvTranspose2d(dim, 3, kernel_size=patch, stride=patch)

    def _pos_embed(self, gh, gw, device, dtype):
        if self.pos_mode == "1d":
            pos = torch.arange(gh * gw, device=device, dtype=dtype).unsqueeze(1)
            i = torch.arange(self.dim, device=device, dtype=dtype).unsqueeze(0)
            angle = pos / (10000 ** (2 * (i // 2) / self.dim))
            return torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle)).unsqueeze(0)
        # fixed 2D sin-cos encoding (half the channels encode row, half column)
        # -- no learned params, so state_dicts stay independent of resolution.
        d = self.dim // 2
        i = torch.arange(d, device=device, dtype=dtype).unsqueeze(0)
        freq = 10000 ** (2 * (i // 2) / d)
        ys = torch.arange(gh, device=device, dtype=dtype).unsqueeze(1) / freq
        xs = torch.arange(gw, device=device, dtype=dtype).unsqueeze(1) / freq
        pe_y = torch.where(i % 2 == 0, torch.sin(ys), torch.cos(ys))  # (gh, d)
        pe_x = torch.where(i % 2 == 0, torch.sin(xs), torch.cos(xs))  # (gw, d)
        pe = torch.cat([pe_y.unsqueeze(1).expand(gh, gw, d),
                        pe_x.unsqueeze(0).expand(gh, gw, d)], dim=-1)
        return pe.reshape(1, gh * gw, self.dim)

    def forward(self, x):
        b, _, h, w = x.shape
        tok = self.embed(x)                          # (B, dim, h/p, w/p)
        gh, gw = tok.shape[-2:]
        tok = tok.flatten(2).transpose(1, 2)          # (B, N, dim)
        pos = self._pos_embed(gh, gw, x.device, x.dtype)
        tok = self.encoder(tok + pos)
        tok = tok.transpose(1, 2).reshape(b, self.dim, gh, gw)
        return self.unembed(tok)
