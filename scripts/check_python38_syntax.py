#!/usr/bin/env python3
"""Parse only active revision-workflow Python files using Python 3.8 grammar."""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FILES = [
    REPO_ROOT / "Experiment2026" / "experiments2026.py",
    REPO_ROOT / "Experiment2026" / "preprocessing_only_ablation.py",
    REPO_ROOT / "Experiment2026" / "ensemble_selection_cv_2026.py",
    REPO_ROOT / "Experiment2026" / "baselines2026.py",
    REPO_ROOT / "Experiment2026" / "reference_cnn2026.py",
    REPO_ROOT / "analysis" / "r1_ensemble_core.py",
    REPO_ROOT / "analysis" / "legacy_metrics2025_adapter.py",
    REPO_ROOT / "analysis" / "crossyear_metrics_normalizer.py",
    REPO_ROOT / "analysis" / "results_publish_core.py",
    REPO_ROOT / "scripts" / "create_r1_ensemble_selection_notebook.py",
    REPO_ROOT / "scripts" / "create_r1_final_ensemble_notebook.py",
    REPO_ROOT / "scripts" / "create_results_publish_notebook.py",
    REPO_ROOT / "scripts" / "test_r1_ensemble_selection.py",
    REPO_ROOT / "scripts" / "convert_legacy_metrics2025.py",
    REPO_ROOT / "scripts" / "verify_legacy_top5_max_rule.py",
    REPO_ROOT / "scripts" / "test_legacy_metrics2025_adapter.py",
    REPO_ROOT / "scripts" / "normalize_metrics2026_for_2025_comparison.py",
    REPO_ROOT / "scripts" / "test_crossyear_metrics_normalizer.py",
    REPO_ROOT / "scripts" / "test_preprocessing_only_ablation.py",
    REPO_ROOT / "scripts" / "test_reference_cnn2026.py",
    REPO_ROOT / "scripts" / "test_results_publish.py",
]

for path in FILES:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
    print("Python 3.8 grammar OK: {0}".format(path.relative_to(REPO_ROOT)))
