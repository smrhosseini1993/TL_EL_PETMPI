#!/usr/bin/env python3
"""Create a non-destructive internal metrics2026 comparison-normalized workbook."""
from __future__ import print_function

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.crossyear_metrics_normalizer import normalize_workbook


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a schema-matched internal metrics2026 comparison workbook without overwriting the source."
    )
    parser.add_argument("--source-workbook", type=Path, required=True, help="Full-precision source metrics2026.xlsx")
    parser.add_argument("--output-workbook", type=Path, required=True, help="New internal comparison-normalized workbook")
    parser.add_argument(
        "--probability-representation",
        choices=("full_precision", "four_decimal"),
        default="full_precision",
        help="Use full_precision for actual 2025-versus-2026 comparison (default); four_decimal only for legacy-rule sensitivity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source_workbook.resolve() == args.output_workbook.resolve():
        raise ValueError("Output must be a new workbook; never overwrite metrics2026.xlsx")
    summary = normalize_workbook(
        args.source_workbook.resolve(),
        args.output_workbook.resolve(),
        args.probability_representation,
    )
    print("Created internal comparison-normalized workbook:", summary["output"])
    print("Probability representation:", summary["probability_representation"])
    print("Run rows:", summary["run_rows"])
    print("Shared metrics2025 core columns:", summary["core_columns"])
    print("Additional retained 2026 columns:", summary["extra_columns"])
    print("Rows whose normalized test metrics differ from full precision:", summary["metric_difference_rows"])
    print("Rows with literal 0.5000 after normalization:", summary["literal_half_rows"])


if __name__ == "__main__":
    main()
