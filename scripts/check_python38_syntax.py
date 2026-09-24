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
]

for path in FILES:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
    print(f"Python 3.8 grammar OK: {path.relative_to(REPO_ROOT)}")
