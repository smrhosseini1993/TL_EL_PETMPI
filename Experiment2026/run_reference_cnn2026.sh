#!/usr/bin/env bash
# Run the seed-varied four-convolution reference-CNN reanalysis for seeds 1-100.
# This launcher writes exactly one secure local workbook: CNN_metrics2026.xlsx.
# Run it in the existing polarmaps2024 TensorFlow environment, preferably in tmux.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/../dataparent}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/CNN_metrics2026.xlsx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_SEED="${START_SEED:-1}"
END_SEED="${END_SEED:-100}"

if [[ "${START_SEED}" -lt 1 || "${END_SEED}" -gt 100 || "${START_SEED}" -gt "${END_SEED}" ]]; then
  echo "START_SEED and END_SEED must define an inclusive range within 1-100." >&2
  exit 2
fi

printf '2026 seed-varied reference-CNN reanalysis\n'
printf 'Data root: %s\n' "${ROOT_DIR}"
printf 'One output workbook: %s\n' "${OUTPUT_FILE}"
printf 'Random seeds: %s-%s\n' "${START_SEED}" "${END_SEED}"
printf 'Protocol: four-convolution source CNN | input 256 | batch 20 | 35 epochs | SGD lr 0.005, momentum 0.9, decay 1e-8 | class weights 0:1, 1:3\n\n'

for seed in $(seq "${START_SEED}" "${END_SEED}"); do
  printf '\n===== Reference CNN | seed %03d =====\n' "${seed}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/reference_cnn2026.py" \
    --root "${ROOT_DIR}" \
    --output-file "${OUTPUT_FILE}" \
    --seed "${seed}" \
    --input-size 256 \
    --batch-size 20 \
    --epochs 35 \
    --resume
done

printf '\nAll requested reference-CNN seeds have completed.\n'
printf 'Single workbook: %s\n' "${OUTPUT_FILE}"
