#!/usr/bin/env bash
# R1 locked historic 61/31/46 TL benchmark.
# Results are written to one durable SQLite database in RESULTS_DIR, not thousands of files.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the secure directory containing data/training and data/test}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the verified secure 61/31/46 CSV}"
CV_OUTPUT_ROOT="${CV_OUTPUT_ROOT:?Set CV_OUTPUT_ROOT to the complete CV output directory}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to the secure R1_final_TL output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_STAGE="${RUN_STAGE:-preflight}"  # Allowed values: preflight, production
PROTOCOL_FILE="${CV_OUTPUT_ROOT}/selected_protocol.json"
MODELS="VGG16,VGG19,ResNet50,ResNet101,ResNet152,InceptionV3,InceptionResNetV2,DenseNet169,DenseNet201,MobileNetV2,Xception"

mkdir -p "${RESULTS_DIR}"

if [[ "${RUN_STAGE}" == "preflight" ]]; then
  LOG_FILE="${RESULTS_DIR}/preflight_console.log"
  set +e
  "${PYTHON_BIN}" TL_fixedvalidation.py \
    --root "${PROJECT_ROOT}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --protocol-file "${PROTOCOL_FILE}" \
    --results-dir "${RESULTS_DIR}" \
    --batch-name seeds_001_005 \
    --models "${MODELS}" \
    --seeds 1-5 \
    --preflight \
    2>&1 | tee "${LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
  set -e
  printf '\nPreflight exit status: %s\n' "${STATUS}"
  exit "${STATUS}"
elif [[ "${RUN_STAGE}" == "production" ]]; then
  LOG_FILE="${RESULTS_DIR}/production_console.log"
  set +e
  "${PYTHON_BIN}" TL_fixedvalidation.py \
    --root "${PROJECT_ROOT}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --protocol-file "${PROTOCOL_FILE}" \
    --results-dir "${RESULTS_DIR}" \
    --batch-name seeds_006_100 \
    --models "${MODELS}" \
    --seeds 6-100 \
    2>&1 | tee "${LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
  set -e
  printf '\nProduction exit status: %s\n' "${STATUS}"
  exit "${STATUS}"
else
  echo "RUN_STAGE must be 'preflight' or 'production'; received: ${RUN_STAGE}" >&2
  exit 2
fi
