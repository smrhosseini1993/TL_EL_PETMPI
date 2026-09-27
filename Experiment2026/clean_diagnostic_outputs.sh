#!/usr/bin/env bash
# Data-only cleanup for the 2026 fixed-split rerun.
# It never edits experiments2026.py or any launcher.
#
# Retained final workbook (the file expected by run2026.sh):
#   metrics2026.xlsx
#
# Deleted diagnostic-only workbooks:
#   metrics2026_diagnostic_batch10_redgreen.xlsx
#   metrics2026_diagnostic_batch5_redgreen.xlsx
#
# Usage:
#   ./clean_diagnostic_outputs.sh --apply
#
# The script first verifies that metrics2026.xlsx is the clean 55-run,
# batch-10/no-shift preflight workbook. It refuses to delete anything otherwise.

set -euo pipefail

if [[ "${1:-}" != "--apply" || $# -ne 1 ]]; then
  echo "Usage: $0 --apply" >&2
  echo "This data-only script validates metrics2026.xlsx, then deletes the two diagnostic workbooks." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_WORKBOOK="${SCRIPT_DIR}/metrics2026.xlsx"
DIAGNOSTICS=(
  "${SCRIPT_DIR}/metrics2026_diagnostic_batch10_redgreen.xlsx"
  "${SCRIPT_DIR}/metrics2026_diagnostic_batch5_redgreen.xlsx"
)

if [[ ! -f "${FINAL_WORKBOOK}" ]]; then
  echo "Refusing cleanup: retained final workbook is missing: ${FINAL_WORKBOOK}" >&2
  exit 1
fi

python - "${FINAL_WORKBOOK}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
sheets = pd.read_excel(path, sheet_name=None)
expected_sheets = {"Run_results", "Patient_manifest", "Study_lock"}
if set(sheets) != expected_sheets:
    raise SystemExit("Refusing cleanup: retained workbook has unexpected sheets: {0}".format(sorted(sheets)))

runs = sheets["Run_results"]
if len(runs) != 55:
    raise SystemExit("Refusing cleanup: expected exactly 55 clean preflight rows, found {0}.".format(len(runs)))
if runs.duplicated(["model_name", "seed"]).any():
    raise SystemExit("Refusing cleanup: retained workbook contains duplicate model/seed rows.")
if sorted(runs["batch_size"].dropna().astype(int).unique().tolist()) != [10]:
    raise SystemExit("Refusing cleanup: retained workbook is not batch size 10 only.")
if "diagnostic_label" in runs.columns:
    raise SystemExit("Refusing cleanup: retained workbook contains diagnostic rows/columns.")
if "augmentation" not in runs.columns or set(runs["augmentation"].dropna().astype(str).unique()) != {"none"}:
    raise SystemExit("Refusing cleanup: retained workbook is not the no-augmentation/no-shift protocol.")

print("Validated retained workbook: {0}".format(path.name))
print("  rows=55; batch_size=10; augmentation=none; no diagnostic label")
PY

echo
for file in "${DIAGNOSTICS[@]}"; do
  if [[ -f "${file}" ]]; then
    rm -f -- "${file}"
    echo "Deleted diagnostic workbook: $(basename "${file}")"
  else
    echo "Already absent: $(basename "${file}")"
  fi
done

echo
echo "Cleanup complete. Retained final workbook (unchanged): ${FINAL_WORKBOOK}"
