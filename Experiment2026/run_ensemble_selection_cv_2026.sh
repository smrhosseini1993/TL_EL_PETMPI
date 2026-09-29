#!/usr/bin/env bash
# Development-only CV launcher for ensemble-configuration selection.
# It intentionally never loads root/data/test. Use a distinct secure output directory
# for the VGG16 technical preflight and the complete 55-fit run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/../dataparent}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_STAGE="${RUN_STAGE:-preflight}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../secure_outputs}"
BASE_SEED="${BASE_SEED:-20260929}"
CV_SEED="${CV_SEED:-42}"

if [[ ! -d "${ROOT_DIR}/data/training" ]]; then
  echo "Development data directory not found: ${ROOT_DIR}/data/training" >&2
  exit 2
fi

case "${RUN_STAGE}" in
  preflight)
    OUTPUT_DIR="${OUTPUT_ROOT}/ensemble_selection_cv_2026_preflight"
    MODELS="VGG16"
    EXTRA_ARGS=(--allow-partial)
    ;;
  full)
    OUTPUT_DIR="${OUTPUT_ROOT}/ensemble_selection_cv_2026_full"
    MODELS="VGG16,VGG19,ResNet50,ResNet101,ResNet152,InceptionV3,InceptionResNetV2,DenseNet169,DenseNet201,MobileNetV2,Xception"
    EXTRA_ARGS=()
    ;;
  *)
    echo "RUN_STAGE must be 'preflight' or 'full'; received: ${RUN_STAGE}" >&2
    exit 2
    ;;
esac

printf '2026 development-only ensemble-selection CV\n'
printf 'Stage: %s\n' "${RUN_STAGE}"
printf 'Secure development root: %s\n' "${ROOT_DIR}"
printf 'Secure output directory: %s\n' "${OUTPUT_DIR}"
printf 'Models: %s\n' "${MODELS}"
printf 'Plan: 5-fold CV; 128 input | batch 10 | Adam 0.0003 | dropout 0.50 | class weights | no augmentation\n'
printf 'Test data: never loaded by this launcher or runner\n\n'

"${PYTHON_BIN}" "${SCRIPT_DIR}/ensemble_selection_cv_2026.py" \
  --root "${ROOT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --models "${MODELS}" \
  --base-seed "${BASE_SEED}" \
  --cv-seed "${CV_SEED}" \
  --n-folds 5 \
  "${EXTRA_ARGS[@]}"

printf '\nCompleted requested %s stage.\n' "${RUN_STAGE}"
printf 'Output directory: %s\n' "${OUTPUT_DIR}"
if [[ "${RUN_STAGE}" == "preflight" ]]; then
  printf '%s\n' 'This preflight is only a technical check. Run the full stage in a new output directory before opening the ensemble-selection notebook.'
else
  printf '%s\n' 'Use ensemble_selection_cv_raw_results.xlsx from this full output directory in the ensemble-selection notebook.'
fi
