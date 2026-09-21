"""Synthetic smoke test for frpure.eval.metrics -- runs on CPU, no weights/data.

Verifies the comparison-harness logic is correct before it touches the GPU box:
  * a well-separated FR system gets high TAR@FAR and verification accuracy
  * an attack that collapses the scores produces high ASR + low robust_tar
  * a defense that restores the scores recovers robust_tar
  * fidelity metrics behave (identical -> inf PSNR / SSIM 1.0)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from frpure.eval import metrics as M

rng = np.random.default_rng(0)


def section(t): print(f"\n=== {t} ===")

# --- A. Utility: a decent FR system: genuine sims high, impostor sims low ----
section("A. Utility (clean)")
genuine = rng.normal(0.65, 0.10, size=3000).clip(-1, 1)   # same identity
impostor = rng.normal(0.05, 0.10, size=3000).clip(-1, 1)  # different identity

acc, thr = M.verification_accuracy(genuine, impostor)
taf = M.tar_at_far(genuine, impostor, far_targets=(1e-2, 1e-3))
print(f"verification_accuracy = {acc:.4f} at thr={thr:.3f}")
for far, d in taf.items():
    print(f"  TAR@FAR={far:.0e}: TAR={d['tar']:.4f} (thr={d['threshold']:.3f})")
assert acc > 0.95, "well-separated system should verify near-perfectly"
assert taf[1e-2]["tar"] > 0.8

# operating threshold we will FREEZE for the robustness eval (clean FAR=1e-2)
op_thr = taf[1e-2]["threshold"]

# --- B. Robustness: a dodging attack pushes genuine scores down --------------
section("B. Robustness under attack (no defense)")
genuine_adv = (genuine - 0.75).clip(-1, 1)     # attacker suppresses similarity
impostor_adv = (impostor + 0.75).clip(-1, 1)   # impersonation pushes sims up

asr_dodge = M.attack_success_rate(genuine_adv, op_thr, mode="dodging")
asr_imp = M.attack_success_rate(impostor_adv, op_thr, mode="impersonation")
rtar_nodef = M.robust_tar(genuine_adv, impostor_adv, op_thr)
print(f"ASR dodging        = {asr_dodge:.4f}  (want HIGH -> attack works)")
print(f"ASR impersonation  = {asr_imp:.4f}  (want HIGH -> attack works)")
print(f"robust_tar (nodef) = {rtar_nodef:.4f} (want LOW -> system broken)")
assert asr_dodge > 0.9 and asr_imp > 0.9
assert rtar_nodef < 0.2

# --- B'. A defense that partially restores the clean scores ------------------
section("B'. Robustness WITH a (mock) defense")
genuine_def = (genuine - 0.10).clip(-1, 1)     # defense recovers most signal
impostor_def = (impostor + 0.10).clip(-1, 1)
rtar_def = M.robust_tar(genuine_def, impostor_def, op_thr)
asr_dodge_def = M.attack_success_rate(genuine_def, op_thr, mode="dodging")
print(f"robust_tar (defended) = {rtar_def:.4f} (want HIGH -> defense holds)")
print(f"ASR dodging (defended) = {asr_dodge_def:.4f} (want LOW -> attack foiled)")
assert rtar_def > rtar_nodef + 0.5, "defense must recover robust accuracy"

# --- E. Fidelity --------------------------------------------------------------
section("E. Fidelity")
img = rng.random((3, 112, 112)).astype(np.float32)          # CHW in [0,1]
noise = (rng.random(img.shape).astype(np.float32) - 0.5) * (8 / 255 * 2)
adv = (img + noise).clip(0, 1)

print(f"PSNR(identical)     = {M.psnr(img, img)}")
print(f"SSIM(identical)     = {M.ssim(img, img):.4f}")
print(f"PSNR(adv)           = {M.psnr(img, adv):.2f} dB")
print(f"SSIM(adv)           = {M.ssim(img, adv):.4f}")
print(f"pert norms(adv)     = {M.perturbation_norms(img, adv)}")
assert M.psnr(img, img) == float("inf")
assert abs(M.ssim(img, img) - 1.0) < 1e-9
assert M.perturbation_norms(img, adv)["linf"] <= 8 / 255 + 1e-6

# --- F. Efficiency ------------------------------------------------------------
section("F. Efficiency timer")
with M.Throughput(n_items=64) as tp:
    _ = np.fft.fft2(rng.random((64, 112, 112)))   # stand-in for a purifier pass
print(f"latency = {tp.latency_ms:.3f} ms/img   throughput = {tp.fps:.1f} fps")
assert tp.fps > 0

# --- ResultRow round-trips to a dict (for CSV/JSON logging) ------------------
section("ResultRow serialization")
row = M.ResultRow(defense="frpure_ours", attack="pgd_eot",
                  threat="Linf_8/255_whitebox_adaptive",
                  ver_acc=acc, tar_at_far=taf, robust_tar=rtar_def,
                  asr_dodging=asr_dodge_def, psnr=M.psnr(img, adv),
                  ssim=M.ssim(img, adv), latency_ms=tp.latency_ms, fps=tp.fps)
d = row.to_dict()
print(f"row keys ({len(d)}): {sorted(d)[:6]} ...")
assert d["defense"] == "frpure_ours"

print("\nALL SMOKE TESTS PASSED ✔")