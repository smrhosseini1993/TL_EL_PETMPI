#!/usr/bin/env python3
"""Validate final-run SQLite coverage without printing model-performance results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Permit `python scripts/...` from the repository root without requiring
# PYTHONPATH. This is the documented invocation on the hospital server.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_results_store import connect_results_database, seed_coverage_frame, validate_results_database
from tl_reanalysis_core import SUPPORTED_MODELS


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seed_specification(value: str) -> List[int]:
    seeds: List[int] = []
    for token in parse_csv(value):
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid seed range: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    return sorted(set(seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the structural completeness of compact R1 final TL outputs.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Secure R1_final_TL directory containing r1_final_tl_runs.sqlite.")
    parser.add_argument("--models", default=",".join(SUPPORTED_MODELS), help="Expected architectures; default is all 11.")
    parser.add_argument("--seeds", required=True, help="Expected seed range, e.g. 1-5 or 1-100.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    seeds = parse_seed_specification(args.seeds)
    connection = connect_results_database(args.results_dir)
    try:
        result = validate_results_database(connection, models=models, seeds=seeds, expected_test_patients=46)
        coverage = seed_coverage_frame(connection, models=models, seeds=seeds, expected_test_patients=46)
    finally:
        connection.close()
    summary = result["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nPer-model structural coverage:")
    for model_name, group in coverage.groupby("model_name", sort=True):
        completed = int((group["coverage_status"] == "PASS").sum())
        print(f"{model_name}: {completed}/{len(group)} complete; status={'PASS' if completed == len(group) else 'INCOMPLETE'}")
    if not summary["valid"]:
        raise SystemExit(1)
    print("\nStructural validation passed. No performance metric was used for this result.")


if __name__ == "__main__":
    main()
