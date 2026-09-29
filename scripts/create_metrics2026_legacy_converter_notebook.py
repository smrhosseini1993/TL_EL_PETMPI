#!/usr/bin/env python3
"""Create the version-controlled metrics2026 legacy-compatibility notebook."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "notebooks" / "convert_metrics2026_to_legacy.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown("""# Convert final `metrics2026.xlsx` to legacy plotting layout

**Purpose:** Create a separate `metrics_legacy_compatible.xlsx` workbook whose `sheet1` matches the column layout of the original legacy `metrics.xlsx`. It can be used by old **visualisation** cells that expect names such as `test_predicts`, `test_auc`, and `tag`.

> **Safety boundary:** `metrics2026.xlsx` is the master audited workbook and is never edited. The compatibility copy rounds prediction text to the legacy four-decimal display format; retain the source workbook for all final statistics and patient-level analysis. Do not reuse any old best-run, test-set-selection, Borda-count, or run-level inferential cells.
"""),
    markdown("""## 0. Paths and run switch

This assumes the recommended Mac folder layout:

```text
PETMPI_R1_final_analysis/
├── code/            # GitHub clone
└── secure_inputs/
    └── metrics2026.xlsx
```

Change paths only if your local layout differs.
"""),
    code("""from pathlib import Path
import sys
import pandas as pd
from IPython.display import display

# Run from the repository root or its `notebooks/` folder.
CWD = Path.cwd().resolve()
if (CWD / 'analysis').exists():
    REPO_ROOT = CWD
elif (CWD.parent / 'analysis').exists():
    REPO_ROOT = CWD.parent
else:
    raise RuntimeError('Open this notebook from the repository root or notebooks/.')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.metrics2026_legacy_converter import (
    LEGACY_COLUMNS,
    convert_metrics2026_to_legacy,
    read_and_validate_metrics2026,
)

PROJECT_ROOT = REPO_ROOT.parent
SOURCE_WORKBOOK = PROJECT_ROOT / 'secure_inputs' / 'metrics2026.xlsx'
OUTPUT_WORKBOOK = PROJECT_ROOT / 'secure_inputs' / 'metrics_legacy_compatible.xlsx'
RUN_CONVERSION = False

print('Source master workbook:', SOURCE_WORKBOOK)
print('Separate compatibility output:', OUTPUT_WORKBOOK)
print('Source will not be modified.')
"""),
    markdown("""## 1. Validate the final 1,100-run source workbook

This checks the source is the complete final 11 architectures × 100 seeds workbook with its 61/31/46 patient manifest. It performs no writing.
"""),
    code("""source_runs, labels_by_split, source_lock = read_and_validate_metrics2026(SOURCE_WORKBOOK)
print(f'Validated {len(source_runs)} runs: {source_runs.model_name.nunique()} architectures × {source_runs.seed.nunique()} seeds.')
print('Patient counts:', {split: len(labels) for split, labels in labels_by_split.items()})
print('Source sheets:', ['Run_results', 'Patient_manifest', 'Study_lock'])
"""),
    markdown("""## 2. Create the separate legacy-compatible copy

Set `RUN_CONVERSION=True` only after the validation cell succeeds. The output contains:

```text
metrics_legacy_compatible.xlsx
├── sheet1            # 1,100 rows in the original legacy metrics.xlsx layout
└── Conversion_lock   # provenance and statistical-use warning
```
"""),
    code("""if RUN_CONVERSION:
    converted = convert_metrics2026_to_legacy(SOURCE_WORKBOOK, OUTPUT_WORKBOOK)
    print('Created:', OUTPUT_WORKBOOK)
    print('Legacy-style rows:', len(converted))
    print('Columns match legacy schema:', tuple(converted.columns) == LEGACY_COLUMNS)
    display(converted.head())
else:
    print('Conversion not run. Set RUN_CONVERSION = True after validating the source workbook.')
"""),
    markdown("""## What can and cannot be reused

**Can reuse:** old code cells that only read run metrics and draw figures/tables.

**Do not reuse without revision:** cells that select a best seed/model using test performance, select ensemble configurations using test performance, use Borda Count, or treat 100 runs as independent patients for p-values.

For final results, use the audited `metrics2026.xlsx` and later combined reporting notebook as the statistical source of truth.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print("Wrote {0}".format(OUTPUT))
