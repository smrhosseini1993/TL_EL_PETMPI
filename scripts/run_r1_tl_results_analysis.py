#!/usr/bin/env python3
"""Create the secure R1 TL Version 1 result package from a validated SQLite store.

The source database is opened read-only. The output includes aggregated patient-level
predictions, so it must remain in approved secure storage and must never be committed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.r1_tl_results import AnalysisSettings, run_v1_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create R1 TL results analysis Sections 0–6 from the complete final SQLite store.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Secure R1_final_TL directory containing r1_final_tl_runs.sqlite.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New secure directory for tables, figures and an analysis manifest.")
    parser.add_argument("--metric-bootstrap-iterations", type=int, default=2000, help="Stratified bootstrap iterations for patient-level CIs (default: 2000).")
    parser.add_argument("--dca-bootstrap-iterations", type=int, default=1000, help="Stratified bootstrap iterations for DCA bands (default: 1000).")
    parser.add_argument("--random-seed", type=int, default=20260926, help="Fixed analysis random seed.")
    parser.add_argument("--calibration-bins", type=int, default=5, help="Uniform bins for calibration plot/ECE (default: 5).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metric_bootstrap_iterations < 100 or args.dca_bootstrap_iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations for the analysis package.")
    if args.calibration_bins < 2:
        raise ValueError("Calibration requires at least two bins.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {args.output_dir}. Use a new directory; do not overwrite an analysis package.")
    settings = AnalysisSettings(
        metric_bootstrap_iterations=args.metric_bootstrap_iterations,
        dca_bootstrap_iterations=args.dca_bootstrap_iterations,
        random_seed=args.random_seed,
        calibration_bins=args.calibration_bins,
    )
    tables = run_v1_analysis(args.results_dir, args.output_dir, settings)
    print("R1 TL Version 1 analysis complete.")
    print(f"Source database (read-only): {args.results_dir / 'r1_final_tl_runs.sqlite'}")
    print(f"Secure analysis output: {args.output_dir}")
    print(f"Tables: {len(tables)}; figures: 9")
    print("Scope: Sections 0–6 only. No baseline/CNN/reader/ensemble analysis was performed.")


if __name__ == "__main__":
    main()
