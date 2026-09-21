#!/usr/bin/env bash
# scripts/run_pipeline.sh -- full FRPure pipeline: self-test -> prep -> train -> eval.
# Run from anywhere:  bash scripts/run_pipeline.sh
# Override settings:  EPOCHS=10 BATCH=64 bash scripts/run_pipeline.sh
#                     SELF_TEST=0 bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DATA="${DATA:-DATA}"
SIZE="${SIZE:-160}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-32}"
BACKBONE="${BACKBONE:-facenet}"
ATTACKS="${ATTACKS:-clean,fgsm,pgd_eot,bpda_eot,patch_eyeglass}"
PURIFIER="${PURIFIER:-results/purifier.pth}"
OUT="${OUT:-results/lfw_trained.csv}"
SELF_TEST="${SELF_TEST:-1}"

ALIGNED="${DATA}/lfw_aligned_${SIZE}"
PAIRS="${DATA}/pairs.txt"

banner() { echo; echo "========================================================"; echo "  $1"; echo "========================================================"; }

if [ "${SELF_TEST}" = "1" ]; then
  banner "STEP 0  self-test trainer (CPU, synthetic)"
  python scripts/train_purifier.py --synthetic --device cpu --iters 3
fi

banner "STEP 1  fetch+align LFW (figshare, direct)  ->  ${ALIGNED}"
python scripts/prepare_data.py --out "${DATA}" --size "${SIZE}" --device "${DEVICE}"

banner "STEP 2  train purifier  ->  ${PURIFIER}"
python scripts/train_purifier.py --aligned_dir "${ALIGNED}" \
    --backbone "${BACKBONE}" --size "${SIZE}" --epochs "${EPOCHS}" \
    --batch "${BATCH}" --out "${PURIFIER}" --device "${DEVICE}"

banner "STEP 3  run matrix  ->  ${OUT}"
python -m frpure.eval.runner --aligned_dir "${ALIGNED}" --pairs "${PAIRS}" \
    --backbone "${BACKBONE}" --frpure_weights "${PURIFIER}" \
    --attacks "${ATTACKS}" --device "${DEVICE}" --out "${OUT}"

banner "DONE.  results -> ${OUT} (+ .json)"
echo "Compare the frpure_ours rows vs none / fsqueeze / kim_nil on robust_tar."