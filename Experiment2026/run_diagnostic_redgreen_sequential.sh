#!/usr/bin/env bash
# Overnight diagnostic sequence. Runs one GPU job at a time:
#   1) batch 10 + legacy-style red/green x1.10 training shift
#   2) batch 5 + legacy-style red/green x1.10 training shift
# The second diagnostic starts only when the first exits successfully.
# Neither job changes metrics2026.xlsx (the preserved batch-10/no-shift workbook).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SEED="${START_SEED:-1}"
END_SEED="${END_SEED:-5}"

printf 'Sequential overnight red/green diagnostics\n'
printf 'Seeds: %s-%s\n' "${START_SEED}" "${END_SEED}"
printf 'Preserved final-protocol workbook: %s/metrics2026.xlsx\n' "${SCRIPT_DIR}"
printf 'First: batch 10 + red/green diagnostic\n'
printf 'Second: batch 5 + red/green diagnostic, only after the first succeeds\n\n'

START_SEED="${START_SEED}" END_SEED="${END_SEED}" \
  "${SCRIPT_DIR}/run_diagnostic_batch10_redgreen.sh"

printf '\nBatch-10 red/green diagnostic completed successfully. Starting batch-5 diagnostic.\n\n'

START_SEED="${START_SEED}" END_SEED="${END_SEED}" \
  "${SCRIPT_DIR}/run_diagnostic_batch5_redgreen.sh"

printf '\nBoth sequential red/green diagnostic preflights completed successfully.\n'
printf 'Review separately: metrics2026_diagnostic_batch10_redgreen.xlsx and metrics2026_diagnostic_batch5_redgreen.xlsx\n'
