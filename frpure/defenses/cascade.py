"""
frpure.defenses.cascade
=======================
Cost-aware DUAL-ARCHITECTURE denoiser for denoised randomized smoothing.

Motivation (measured, see results/diag_overlap_s*.json): the ViT denoiser's
errors are very nearly a SUBSET of the DnCNN's, so no per-input router can beat
ViT-only accuracy by more than ~1.5% (the oracle ceiling). The exploitable axis
is therefore COST, not accuracy: DnCNN is ~40x smaller and much cheaper, and on
"easy" noise draws it is already sufficient. The cascade spends the expensive
ViT only where the cheap denoiser is unreliable.

    for each Gaussian draw z = x + eps:
        xd = DnCNN(z)
        if gate(xd, z) >= t:  use xd            (cheap path)
        else:                 use ViT(z)        (expensive path)

CERTIFICATE SAFETY -- the part that is easy to get wrong.
The Cohen et al. (2019) guarantee certifies the smoothed function
g(x) = majority_{eps} h(x + eps) for ANY deterministic base classifier h. It
does not care how complicated h is. So the cascade is certifiable IFF the gate
is part of h, i.e. it is a deterministic function of the SINGLE noisy sample
(x + eps) alone. It must NOT look at:
  * the clean x (unavailable to the verifier at certify time),
  * the label or the gallery decision,
  * statistics pooled ACROSS the n noise draws (that would make the per-sample
    decisions dependent, breaking the i.i.d. Monte-Carlo bound).
`route_mask` below therefore takes only the noisy batch. Each row is routed
independently, so h stays a deterministic per-sample map and the standard
Clopper-Pearson/Cohen machinery in SmoothedVerifier applies UNCHANGED.

The gate signal is patch-wise residual energy of the cheap denoiser: how much
noise DnCNN thinks it removed, normalised per image. Draws that DnCNN explains
cleanly (low residual disagreement) are kept; the ragged ones escalate. This is
label-free and computable from the noisy sample alone.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _slow_encoder_layer_forward(self, src, src_mask=None,
                                src_key_padding_mask=None, is_causal=False):
    """Explicit (non-fused) TransformerEncoderLayer forward, post-norm.

    Mirrors nn.TransformerEncoderLayer's own slow path for norm_first=False.
    At eval() the dropout submodules are identities, so this is numerically
    equivalent to the fused kernel -- it just never enters it.
    """
    x = src
    x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask,
                                      is_causal=is_causal))
    x = self.norm2(x + self._ff_block(x))
    return x


def _slow_sa_block(self, x, attn_mask=None, key_padding_mask=None,
                   is_causal=False):
    """Self-attention block that avoids torch._native_multi_head_attention.

    nn.MultiheadAttention has its own fused fast path, separate from the
    encoder layer's, and it trips the same NVML assert. Requesting attention
    weights disables that path; we discard them, so the output is unchanged.
    """
    out = self.self_attn(x, x, x, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask,
                         need_weights=True, average_attn_weights=False)[0]
    return self.dropout1(out)


def _disable_fused_attention(module: nn.Module) -> None:
    """Keep TransformerEncoder off PyTorch's fused fast path.

    That path calls nvmlInit_v2_() inside the CUDA caching allocator and
    hard-asserts on hosts whose NVML runtime mismatches the loaded driver
    (`nvidia-smi` failing is the tell). It is only reachable under eval()+
    no_grad(), which is exactly the certify/predict regime -- so smoothing
    would crash while gradient-based code paths appeared fine.
    """
    for m in module.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
        if isinstance(m, nn.TransformerEncoderLayer):
            # MultiheadAttention's own fused path trips the same assert, so both
            # levels have to be routed around.
            m._sa_block = types.MethodType(_slow_sa_block, m)
            if m.norm_first:            # only the post-norm variant is patched
                continue
            m.forward = types.MethodType(_slow_encoder_layer_forward, m)


class CascadedDenoiser(nn.Module):
    """DnCNN-first, ViT-on-demand denoiser with a per-sample gate.

    Parameters
    ----------
    cheap, expensive : nn.Module
        Denoisers with the smoothing residual contract (x -> denoised in [0,1]).
    thresh : float
        Gate threshold t. Samples with gate score >= t take the cheap path.
        Calibrated on a HELD-OUT split by `scripts/calibrate_cascade.py`.
    mode : {"cascade", "cheap", "expensive"}
        "cheap"/"expensive" force a single branch (baselines share this code
        path so the comparison is apples-to-apples).
    """

    def __init__(self, cheap: nn.Module, expensive: nn.Module,
                 thresh: float = 0.0, mode: str = "cascade"):
        super().__init__()
        self.cheap = cheap
        self.expensive = expensive
        self.thresh = thresh
        self.mode = mode
        _disable_fused_attention(self)
        # usage accounting (not part of the decision; for the cost table only)
        self.register_buffer("_n_seen", torch.zeros((), dtype=torch.long),
                             persistent=False)
        self.register_buffer("_n_cheap", torch.zeros((), dtype=torch.long),
                             persistent=False)

    # -- gate ---------------------------------------------------------------
    @staticmethod
    def gate_score(noisy: torch.Tensor, cheap_out: torch.Tensor) -> torch.Tensor:
        """Per-sample confidence of the CHEAP denoiser. Higher = more reliable.

        Depends only on (noisy, DnCNN(noisy)) -- no clean image, no label, no
        cross-draw pooling -- so it is a valid part of the base classifier h.

        Signal: total variation of the cheap denoiser's OUTPUT. Selected
        empirically over six candidates (residual flatness, residual energy,
        residual TV, output TV, output sharpness, embedding self-similarity) by
        AUC for predicting "DnCNN reconstructs this draw at least as well as the
        ViT in embedding space"; see scripts/gate_ablation.py. Output TV was the
        best signal computable WITHOUT a backbone forward pass -- the stronger
        embedding-self-similarity signal costs as much as the ViT it is meant to
        avoid, so it cannot fund a cost saving.

        Intuition: when the cheap net fails under heavy noise it leaves residual
        speckle in the output, raising TV. Clean reconstructions are smoother.
        NOTE: this gate is weak (AUC ~0.58); see the cascade analysis for what
        that implies about achievable savings.
        """
        del noisy  # signal reads the output only; kept in the signature so the
                   # gate contract stays "function of the noisy sample alone"
        d = cheap_out[:, :, 1:, :] - cheap_out[:, :, :-1, :]
        return d.abs().mean(dim=(1, 2, 3))               # (B,) smooth => LOW

    @torch.no_grad()
    def route_mask(self, noisy: torch.Tensor, cheap_out: torch.Tensor) -> torch.Tensor:
        """Boolean (B,) mask: True = handled by the cheap branch.

        gate_score is output TV, which is LOW when the cheap denoiser is doing
        well, so the cheap branch keeps scores BELOW the threshold. Raising
        `thresh` therefore sends more traffic down the cheap path.
        """
        return self.gate_score(noisy, cheap_out) <= self.thresh

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "cheap":
            self._tally(x.shape[0], x.shape[0])
            return self.cheap(x)
        if self.mode == "expensive":
            self._tally(x.shape[0], 0)
            return self.expensive(x)

        out = self.cheap(x)
        keep = self.route_mask(x, out)
        self._tally(x.shape[0], int(keep.sum().item()))
        if bool(keep.all()):
            return out
        idx = (~keep).nonzero(as_tuple=True)[0]
        # only the escalated rows pay for the transformer
        out = out.clone()
        out[idx] = self.expensive(x[idx])
        return out

    # -- accounting ---------------------------------------------------------
    def _tally(self, seen: int, cheap: int):
        self._n_seen += seen
        self._n_cheap += cheap

    def reset_stats(self):
        self._n_seen.zero_(); self._n_cheap.zero_()

    @property
    def cheap_fraction(self) -> float:
        n = int(self._n_seen.item())
        return float(self._n_cheap.item()) / n if n else float("nan")
