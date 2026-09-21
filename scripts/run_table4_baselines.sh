#!/usr/bin/env bash
# Measures every baseline row of Table IV under one identical protocol:
# LFW, ArcFace-R50, adaptive PGD-EOT and BPDA-EOT.
#
#   bash scripts/run_table4_baselines.sh
#
# Requires a working GPU. If nvidia-smi reports a driver/library mismatch,
# reboot first -- the kernel module and userspace NVML are out of step.
set -e

ALIGNED="${ALIGNED:-DATA/lfw_aligned_160}"
PAIRS="${PAIRS:-DATA/pairs.txt}"
DEN="${DEN:-results/denoiser_s050.pth}"
OUT="${OUT:-results/table4_baselines.csv}"
N="${N:-200}"

echo "== GPU check =="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || {
  echo "GPU unavailable -- reboot required before this can run."; exit 1; }

echo
echo "== running all defenses x all attacks =="
python -m frpure.eval.runner \
  --aligned_dir "$ALIGNED" --pairs "$PAIRS" \
  --backbone facenet --pretrained vggface2 \
  --defenses none,fsqueeze,kim_nil,iter_purify \
  --attacks clean,fgsm,pgd_eot,bpda_eot \
  --iter_denoiser "$DEN" \
  --n_attack "$N" \
  --out "$OUT"

echo
echo "== results =="
column -s, -t < "$OUT"
