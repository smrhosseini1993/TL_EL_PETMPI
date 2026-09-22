#!/usr/bin/env bash
# R1 locked historic 61/31/46 TL benchmark.
# It can begin only after the full CV run has created selected_protocol.json.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the secure directory containing images}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the verified secure 61/31/46 CSV}"
CV_OUTPUT_ROOT="${CV_OUTPUT_ROOT:?Set CV_OUTPUT_ROOT to the complete CV output directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to secure final-output storage}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROTOCOL_FILE="${CV_OUTPUT_ROOT}/selected_protocol.json"
MODELS="VGG16,VGG19,ResNet50,ResNet101,ResNet152,InceptionV3,InceptionResNetV2,DenseNet169,DenseNet201,MobileNetV2,Xception"

# Step 1: five-seed technical preflight for every architecture (55 fits).
# It does not select an architecture or a seed. Inspect expected patient-level files,
# phase parameter counts, logs, and any GPU-memory failures before production.
"${PYTHON_BIN}" TL_fixedvalidation.py \
  --root "${PROJECT_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --protocol-file "${PROTOCOL_FILE}" \
  --output-dir "${OUTPUT_ROOT}/fixed_preflight_seeds_001_005" \
  --models "${MODELS}" \
  --seeds 1-5 \
  --preflight

# Step 2: after the entire 55-fit preflight is accepted technically, run seeds 6–100.
# This produces 11 × 95 = 1,045 remaining fits. It must use the same locked protocol
# and locked split manifest; do not alter settings based on preflight performance.
# "${PYTHON_BIN}" TL_fixedvalidation.py \
#   --root "${PROJECT_ROOT}" \
#   --split-manifest "${SPLIT_MANIFEST}" \
#   --protocol-file "${PROTOCOL_FILE}" \
#   --output-dir "${OUTPUT_ROOT}/fixed_production_seeds_006_100" \
#   --models "${MODELS}" \
#   --seeds 6-100
