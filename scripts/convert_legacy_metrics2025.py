#!/usr/bin/env python3
"""Convert the submitted legacy metrics workbook for one ensemble-rule regression check."""
from __future__ import print_function

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.legacy_metrics2025_adapter import convert_workbooks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a three-sheet metrics2025.xlsx for legacy Top-5 Max Rule reproduction."
    )
    parser.add_argument("--legacy-workbook", type=Path, required=True, help="Submitted one-sheet legacy metrics.xlsx")
    parser.add_argument("--test-labels", type=Path, required=True, help="Ordered 46-label historic test ica_lables.txt")
    parser.add_argument("--output-workbook", type=Path, required=True, help="New metrics2025.xlsx output path")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.legacy_workbook.resolve() == args.output_workbook.resolve():
        raise ValueError("Output workbook must be a new file; never overwrite the legacy source")
    summary = convert_workbooks(
        legacy_workbook=args.legacy_workbook.resolve(),
        test_labels_file=args.test_labels.resolve(),
        output_workbook=args.output_workbook.resolve(),
    )
    print("Created legacy reproduction workbook:", summary["output"])
    print("Source rows:", summary["source_rows"])
    print("Output rows:", summary["output_rows"], "(11 manuscript architectures x 100 positional runs)")
    print("Test patients:", summary["test_patients"])
    print("Excluded source-only architecture:", summary["excluded_source_model"])
    print("Run pairing: converted seed 1-100 equals original within-model row position, not legacy tag.")


if __name__ == "__main__":
    main()
