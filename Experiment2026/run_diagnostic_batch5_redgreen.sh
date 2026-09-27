#!/usr/bin/env bash
# Diagnostic preflight only: legacy-style red/green x1.10 training-image shift.
# Writes to a separate workbook. It never modifies metrics2026.xlsx.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/../dataparent}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/metrics2026_diagnostic_batch5_redgreen.xlsx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_SEED="${START_SEED:-1}"
END_SEED="${END_SEED:-5}"

MODELS=(
  VGG16 VGG19 ResNet50 ResNet101 ResNet152
  InceptionV3 InceptionResNetV2 DenseNet169 DenseNet201 MobileNetV2 Xception
)

if [[ "${START_SEED}" -lt 1 || "${END_SEED}" -gt 100 || "${START_SEED}" -gt "${END_SEED}" ]]; then
  echo "START_SEED and END_SEED must define an inclusive range within 1-100." >&2
  exit 2
fi

printf 'Diagnostic preflight: batch 5 + red/green x1.10 training shift\n'
printf 'Data root: %s\n' "${ROOT_DIR}"
printf 'Separate diagnostic workbook: %s\n' "${OUTPUT_FILE}"
printf 'Seeds: %s-%s\n\n' "${START_SEED}" "${END_SEED}"

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
      --red-green-shift \
      --diagnostic-label diagnostic_batch5_redgreen \
      --resume
  done
done

printf '\nDiagnostic batch-5/red-green preflight completed.\n'
printf 'Separate workbook: %s\n' "${OUTPUT_FILE}"
