#!/usr/bin/env bash
# INTERNAL DIAGNOSTIC ONLY — do not use this output in the manuscript.
# Retains legacy batch size 5 and red/green shift while correcting only
# architecture-specific ImageNet preprocessing.
# Default 1-5 runs are a structural preflight. After review, resume with
# START_REPEAT=6 END_REPEAT=100 to complete the descriptive 100-repeat set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/../dataparent}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/metrics_preprocessing_only.xlsx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_REPEAT="${START_REPEAT:-1}"
END_REPEAT="${END_REPEAT:-5}"

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

if [[ "${START_REPEAT}" -lt 1 || "${END_REPEAT}" -gt 100 || "${START_REPEAT}" -gt "${END_REPEAT}" ]]; then
  echo "START_REPEAT and END_REPEAT must define an inclusive range within 1-100." >&2
  exit 2
fi

printf '%s\n' 'INTERNAL DIAGNOSTIC ONLY — never manuscript input'
printf 'Data root: %s\n' "${ROOT_DIR}"
printf 'Output workbook: %s\n' "${OUTPUT_FILE}"
printf 'Repeats: %s-%s\n' "${START_REPEAT}" "${END_REPEAT}"
printf 'Models: %s\n' "${MODELS[*]}"
printf '%s\n\n' 'Protocol: 128 input | batch 5 | legacy red/green x1.1 training shift | unseeded repeats | correct architecture-specific preprocessing'

for model_name in "${MODELS[@]}"; do
  for repeat in $(seq "${START_REPEAT}" "${END_REPEAT}"); do
    printf '\n===== %s | repeat %03d =====\n' "${model_name}" "${repeat}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/preprocessing_only_ablation.py" \
      --root "${ROOT_DIR}" \
      --output-file "${OUTPUT_FILE}" \
      --model-name "${model_name}" \
      --repeat "${repeat}" \
      --input-size 128 \
      --batch-size 5 \
      --phase1-epochs 100 \
      --phase2-epochs 100 \
      --phase3-epochs 100 \
      --resume
  done
done

printf '\nCompleted requested internal diagnostic repeats.\n'
printf 'Diagnostic workbook: %s\n' "${OUTPUT_FILE}"
