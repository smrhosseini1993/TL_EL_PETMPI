#!/usr/bin/env python3
"""Parse revision Python files using the Python 3.8 grammar feature set."""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FILES = [
    REPO_ROOT / "TL_fixedvalidation.py",
    REPO_ROOT / "final_results_store.py",
    REPO_ROOT / "scripts" / "test_runner_dry_runs.py",
    REPO_ROOT / "scripts" / "test_final_results_store.py",
    REPO_ROOT / "scripts" / "export_final_results_workbook.py",
    REPO_ROOT / "scripts" / "validate_final_results_store.py",
    REPO_ROOT / "scripts" / "create_verified_historic_manifest.py",
    REPO_ROOT / "scripts" / "test_historic_manifest_generator.py",
    REPO_ROOT / "analysis" / "r1_tl_results.py",
    REPO_ROOT / "scripts" / "run_r1_tl_results_analysis.py",
    REPO_ROOT / "scripts" / "create_r1_tl_results_notebook.py",
    REPO_ROOT / "scripts" / "test_r1_tl_results_analysis.py",
    REPO_ROOT / "Experiment2026" / "ensemble_selection_cv_2026.py",
    REPO_ROOT / "analysis" / "r1_ensemble_core.py",
    REPO_ROOT / "scripts" / "create_r1_ensemble_selection_notebook.py",
    REPO_ROOT / "scripts" / "create_r1_final_ensemble_notebook.py",
    REPO_ROOT / "scripts" / "test_r1_ensemble_selection.py",
    REPO_ROOT / "analysis" / "metrics2026_legacy_converter.py",
    REPO_ROOT / "scripts" / "create_metrics2026_legacy_converter_notebook.py",
    REPO_ROOT / "scripts" / "test_metrics2026_legacy_converter.py",
    REPO_ROOT / "analysis" / "results_publish_core.py",
    REPO_ROOT / "scripts" / "create_results_publish_notebook.py",
    REPO_ROOT / "scripts" / "test_results_publish.py",
]

for path in FILES:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
    print(f"Python 3.8 grammar OK: {path.relative_to(REPO_ROOT)}")
