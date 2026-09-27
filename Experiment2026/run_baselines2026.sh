#!/usr/bin/env bash
# Conventional baseline runner for Reviewer 1, Comment 2.
# Clinical data remain outside Git. Supply their secure local path only through
# CLINICAL_CSV when running this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLINICAL_CSV="${CLINICAL_CSV:-}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/baseline_metrics2026.xlsx}"

if [[ -z "${CLINICAL_CSV}" ]]; then
  echo "CLINICAL_CSV is required and must point to the secure local clinical CSV." >&2
  echo "Example:" >&2
  echo "  CLINICAL_CSV=/secure/path/clinical_data.csv ./run_baselines2026.sh" >&2
  exit 2
fi

if [[ ! -f "${CLINICAL_CSV}" ]]; then
  echo "Clinical CSV not found: ${CLINICAL_CSV}" >&2
  exit 2
fi

printf 'EJPH-D-26-00179 conventional baselines A-F\n'
printf 'Secure clinical CSV: %s\n' "${CLINICAL_CSV}"
printf 'Secure output workbook: %s\n' "${OUTPUT_FILE}"
printf 'Split: historic 61 training / 31 validation / 46 independent test\n'
printf 'Selection: train fit + validation AUC; test data excluded from selection\n\n'

"${PYTHON_BIN}" "${SCRIPT_DIR}/baselines2026.py" \
  --clinical-csv "${CLINICAL_CSV}" \
  --output-file "${OUTPUT_FILE}"
