"""
frpure.eval.metrics
===================
Operationalizes every comparison parameter used by the SOTA papers we benchmark
against, so that an undefended model, a baseline defense (DiffPure-adapted,
Kim-style noise injection, feature squeezing), and our landmark/frequency-aware
purifier are all measured on *identical* axes.

Design notes
------------
* Core metrics are pure-numpy so they run anywhere (incl. CI / a CPU sandbox)
  with no heavy deps. Embedding extraction itself lives in frpure.models; this
  module only consumes cosine similarities / images / timings.
* FR is a *verification* task: we score genuine vs. impostor pairs by cosine
  similarity of L2-normalized embeddings, then threshold. This is why TAR@FAR
  (not top-1 accuracy) is the headline utility metric -- the gap the
  diffusion-purification papers never filled for identity verification.

Metric groups (mirror the comparison table)
  A. Utility    : verification_accuracy, tar_at_far
  B. Robustness : attack_success_rate (dodging & impersonation), robust_tar
  E. Fidelity   : psnr, ssim, perturbation_norms
  F. Efficiency : Throughput (latency / fps helper)
Hardware FPGA metrics (LUT/DSP/power) are emitted by frpure.hardware after
synthesis and merged into the same results row; see hardware/report.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


# --------------------------------------------------------------------------- #
# A. Utility metrics (clean behaviour)
# --------------------------------------------------------------------------- #
def _as_np(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def verification_accuracy(
    genuine_scores: ArrayLike,
    impostor_scores: ArrayLike,
) -> tuple[float, float]:
    """LFW-style verification accuracy at the best cosine-similarity threshold.

    Returns (accuracy, threshold). Higher similarity => same identity.
    """
    g = _as_np(genuine_scores)
    i = _as_np(impostor_scores)
    # candidate thresholds = all observed scores
    thr_candidates = np.unique(np.concatenate([g, i]))
    best_acc, best_thr = 0.0, 0.0
    n = len(g) + len(i)
    for thr in thr_candidates:
        tp = np.sum(g >= thr)        # genuine accepted
        tn = np.sum(i < thr)         # impostor rejected
        acc = (tp + tn) / n
        if acc > best_acc:
            best_acc, best_thr = float(acc), float(thr)
    return best_acc, best_thr


def tar_at_far(
    genuine_scores: ArrayLike,
    impostor_scores: ArrayLike,
    far_targets: Sequence[float] = (1e-2, 1e-3, 1e-4),
) -> dict[float, dict[str, float]]:
    """True Accept Rate at fixed False Accept Rate -- the real FR metric.

    For each target FAR we find the score threshold on the impostor
    distribution that yields that FAR, then report the TAR (fraction of
    genuine pairs accepted) at that threshold.
    """
    g = _as_np(genuine_scores)
    i = _as_np(impostor_scores)
    out: dict[float, dict[str, float]] = {}
    i_sorted = np.sort(i)[::-1]  # descending: highest impostor scores first
    for far in far_targets:
        # threshold s.t. exactly `far` fraction of impostors exceed it
        k = max(int(np.ceil(far * len(i_sorted))) - 1, 0)
        thr = float(i_sorted[k]) if len(i_sorted) else np.inf
        tar = float(np.mean(g >= thr)) if len(g) else 0.0
        out[far] = {"tar": tar, "threshold": thr}
    return out


# --------------------------------------------------------------------------- #
# B. Robustness metrics (behaviour under attack)
# --------------------------------------------------------------------------- #
def attack_success_rate(
    scores_after: ArrayLike,
    threshold: float,
    mode: str,
) -> float:
    """Attack Success Rate.

    mode='dodging'      : genuine pair, attacker wants it REJECTED
                          (score pushed BELOW threshold). Success = score < thr.
    mode='impersonation': impostor pair, attacker wants it ACCEPTED
                          (score pushed ABOVE threshold). Success = score >= thr.
    `threshold` should be the operating threshold fixed on CLEAN data
    (e.g. the TAR@FAR=1e-3 threshold) -- never re-tuned on adversarial scores.
    """
    s = _as_np(scores_after)
    if mode == "dodging":
        return float(np.mean(s < threshold))
    elif mode == "impersonation":
        return float(np.mean(s >= threshold))
    raise ValueError("mode must be 'dodging' or 'impersonation'")


def robust_tar(
    genuine_scores_adv: ArrayLike,
    impostor_scores_adv: ArrayLike,
    threshold: float,
) -> float:
    """Robust verification accuracy: accuracy on adversarial pairs evaluated at
    the CLEAN operating threshold (no re-tuning -- that would be cheating)."""
    g = _as_np(genuine_scores_adv)
    i = _as_np(impostor_scores_adv)
    tp = np.sum(g >= threshold)
    tn = np.sum(i < threshold)
    return float((tp + tn) / (len(g) + len(i)))


# --------------------------------------------------------------------------- #
# E. Fidelity of purified images
# --------------------------------------------------------------------------- #
def perturbation_norms(clean: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """L2 / Linf between two images (or batches), assuming pixel range [0,1]."""
    d = (other.astype(np.float64) - clean.astype(np.float64)).ravel()
    return {"l2": float(np.linalg.norm(d)), "linf": float(np.max(np.abs(d)))}


def psnr(clean: np.ndarray, other: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((clean.astype(np.float64) - other.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(clean: np.ndarray, other: np.ndarray, data_range: float = 1.0) -> float:
    """Global SSIM (single-window approximation, numpy-only).

    For per-paper-faithful MSSIM use scikit-image (structural_similarity) on the
    GPU box; this fallback keeps the harness dependency-free for smoke tests.
    """
    x = clean.astype(np.float64).ravel()
    y = other.astype(np.float64).ravel()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov = np.mean((x - mu_x) * (y - mu_y))
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / den)


# --------------------------------------------------------------------------- #
# F. Efficiency (software-side; FPGA metrics merged later)
# --------------------------------------------------------------------------- #
class Throughput:
    """Context manager to time a block and convert to latency / FPS."""

    def __init__(self, n_items: int = 1):
        self.n_items = n_items
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.elapsed / max(self.n_items, 1)

    @property
    def fps(self) -> float:
        return self.n_items / self.elapsed if self.elapsed > 0 else float("inf")


# --------------------------------------------------------------------------- #
# Results row: one defense x one attack x one threat-model = one comparable row
# --------------------------------------------------------------------------- #
@dataclass
class ResultRow:
    defense: str                 # 'none' | 'diffpure_fr' | 'kim_nil' | 'fsqueeze' | 'frpure_ours'
    attack: str                  # 'clean' | 'pgd_eot' | 'bpda_eot' | 'autoattack_rand' | 'advhat' | ...
    threat: str                  # e.g. 'Linf_8/255_whitebox_adaptive'
    # A. utility
    ver_acc: float = float("nan")
    tar_at_far: dict = field(default_factory=dict)
    # B. robustness
    asr_dodging: float = float("nan")
    asr_impersonation: float = float("nan")
    robust_tar: float = float("nan")
    # D. purification knobs
    diffusion_t: float = float("nan")
    n_denoise_steps: int = -1
    # E. fidelity
    psnr: float = float("nan")
    ssim: float = float("nan")
    pert_l2: float = float("nan")
    pert_linf: float = float("nan")
    # F. efficiency (software)
    latency_ms: float = float("nan")
    fps: float = float("nan")
    # F. hardware (filled post-synthesis)
    fpga_lut: int = -1
    fpga_dsp: int = -1
    fpga_bram: int = -1
    power_w: float = float("nan")
    energy_mj: float = float("nan")
    bitwidth: int = -1

    def to_dict(self) -> dict:
        return asdict(self)