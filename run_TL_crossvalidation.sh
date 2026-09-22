#!/usr/bin/env bash
# R1 development-only TL protocol selection.
# This script never supplies a test-data path to TL_crossvalidation.py.
# Set PROJECT_ROOT and OUTPUT_ROOT on the secure GPU server; do not commit patient data.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the secure directory that contains data/training}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a secure empty output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Step 1: technical preflight (all 11 architectures, one central protocol, all five folds).
# This executes 55 fits and verifies every architecture-specific preprocessing and
# fine-tuning-stage registry before the complete 1,485-fit development search.
# Verify the output schema, phase-parameter files, and logs before proceeding.
"${PYTHON_BIN}" TL_crossvalidation.py \
  --root "${PROJECT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/cv_preflight_all_models" \
  --models VGG16,VGG19,ResNet50,ResNet101,ResNet152,InceptionV3,InceptionResNetV2,DenseNet169,DenseNet201,MobileNetV2,Xception \
  --protocol-ids hlr1e-04_flr3e-06_do50 \
  --allow-partial

# Step 2: after inspecting the preflight, run this full, development-only search.
# Uncomment only after the preflight has passed. It executes 27 × 5 × 11 = 1,485 fits.
# "${PYTHON_BIN}" TL_crossvalidation.py \
#   --root "${PROJECT_ROOT}" \
#   --output-dir "${OUTPUT_ROOT}/cv_full_27x5x11" \
#   --models VGG16,VGG19,ResNet50,ResNet101,ResNet152,InceptionV3,InceptionResNetV2,DenseNet169,DenseNet201,MobileNetV2,Xception
#
# The complete run writes selected_protocol.json. Copy neither test labels nor images
# into the CV output directory; the runner does not load test data.
