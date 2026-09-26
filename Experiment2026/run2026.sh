#!/usr/bin/env bash
# Fixed 61/31/46 PET transfer-learning rerun.
# This launcher writes exactly one results workbook: Experiment2026/metrics2026.xlsx.
# Run it inside the existing polarmaps2024 environment, preferably from tmux.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/../dataparent}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/metrics2026.xlsx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_SEED="${START_SEED:-1}"
END_SEED="${END_SEED:-100}"

MODELS=(
  VGG16
  VGG19
  ResNet50
  ResNet101
  ResNet152
  InceptionV3
  InceptionResNetV2
  DenseNet169
  DenseNet201
  MobileNetV2
  Xception
)

if [[ "${START_SEED}" -lt 1 || "${END_SEED}" -gt 100 || "${START_SEED}" -gt "${END_SEED}" ]]; then
  echo "START_SEED and END_SEED must define an inclusive range within 1-100." >&2
  exit 2
fi

printf '2026 fixed-split TL rerun\n'
printf 'Data root: %s\n' "${ROOT_DIR}"
printf 'One output workbook: %s\n' "${OUTPUT_FILE}"
printf 'Seeds: %s-%s\n' "${START_SEED}" "${END_SEED}"
printf 'Models: %s\n' "${MODELS[*]}"
printf 'Protocol: 128 input | Adam 0.0003 | dropout 0.50 | class weights on | no augmentation\n\n'

for model_name in "${MODELS[@]}"; do
  for seed in $(seq "${START_SEED}" "${END_SEED}"); do
    printf '\n===== %s | seed %03d =====\n' "${model_name}" "${seed}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/experiments2026.py" \
      --root "${ROOT_DIR}" \
      --output-file "${OUTPUT_FILE}" \
      --model-name "${model_name}" \
      --seed "${seed}" \
      --input-size 128 \
      --batch-size 5 \
      --phase1-epochs 100 \
      --phase2-epochs 100 \
      --phase3-epochs 100 \
      --resume
  done
done

printf '\nAll requested model/seed runs have completed.\n'
printf 'Single workbook: %s\n' "${OUTPUT_FILE}"
