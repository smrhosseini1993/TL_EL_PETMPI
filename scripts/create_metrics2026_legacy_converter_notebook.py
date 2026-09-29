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
    markdown("""# Convert final results workbooks to legacy plotting layout

**Purpose:** Create a separate legacy-style workbook whose `sheet1` matches the original `metrics.xlsx` column layout. It accepts either the final 1,100-run transfer-learning master workbook or the 100-seed reference-CNN master workbook. It can be used by old **visualisation** cells that expect names such as `test_predicts`, `test_auc`, and `tag`.

> **Safety boundary:** the source master workbook is never edited. The compatibility copy rounds prediction text to the legacy four-decimal display format; retain the source workbook for all final statistics and patient-level analysis. Do not reuse any old best-run, test-set-selection, Borda-count, or run-level inferential cells.
"""),
    markdown("""## 0. Paths and run switch

This assumes the recommended Mac folder layout:

```text
PETMPI_R1_final_analysis/
├── code/            # GitHub clone
└── secure_inputs/
    ├── metrics2026.xlsx
    └── CNN_metrics2026.xlsx
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
    convert_reference_cnn_metrics2026_to_legacy,
    read_and_validate_metrics2026,
    read_and_validate_reference_cnn_metrics2026,
)

PROJECT_ROOT = REPO_ROOT.parent
# Select exactly one source type: 'transfer_learning' or 'reference_cnn'.
SOURCE_TYPE = 'transfer_learning'
SOURCE_WORKBOOK = PROJECT_ROOT / 'secure_inputs' / ('metrics2026.xlsx' if SOURCE_TYPE == 'transfer_learning' else 'CNN_metrics2026.xlsx')
OUTPUT_WORKBOOK = PROJECT_ROOT / 'secure_inputs' / ('metrics_legacy_compatible.xlsx' if SOURCE_TYPE == 'transfer_learning' else 'CNN_metrics2026_legacy_compatible.xlsx')
RUN_CONVERSION = False

if SOURCE_TYPE not in {'transfer_learning', 'reference_cnn'}:
    raise ValueError("SOURCE_TYPE must be 'transfer_learning' or 'reference_cnn'.")
print('Source master workbook:', SOURCE_WORKBOOK)
print('Source type:', SOURCE_TYPE)
print('Separate compatibility output:', OUTPUT_WORKBOOK)
print('Source will not be modified.')
"""),
    markdown("""## 1. Validate the source workbook

This checks the selected source workbook has the expected completed seeds, run rows, and 61/31/46 patient manifest. It performs no writing.
"""),
    code("""if SOURCE_TYPE == 'transfer_learning':
    source_runs, labels_by_split, source_lock = read_and_validate_metrics2026(SOURCE_WORKBOOK)
else:
    source_runs, labels_by_split, source_lock = read_and_validate_reference_cnn_metrics2026(SOURCE_WORKBOOK)
print(f'Validated {len(source_runs)} runs: {source_runs.model_name.nunique()} architecture(s) × {source_runs.seed.nunique()} seeds.')
print('Patient counts:', {split: len(labels) for split, labels in labels_by_split.items()})
print('Source sheets:', ['Run_results', 'Patient_manifest', 'Study_lock'])
"""),
    markdown("""## 2. Create the separate legacy-compatible copy

Set `RUN_CONVERSION=True` only after the validation cell succeeds. The output contains:

```text
<source>_legacy_compatible.xlsx
├── sheet1            # 1,100 TL rows or 100 reference-CNN rows in the original legacy layout
└── Conversion_lock   # provenance and statistical-use warning
```
"""),
    code("""if RUN_CONVERSION:
    if SOURCE_TYPE == 'transfer_learning':
        converted = convert_metrics2026_to_legacy(SOURCE_WORKBOOK, OUTPUT_WORKBOOK)
    else:
        converted = convert_reference_cnn_metrics2026_to_legacy(SOURCE_WORKBOOK, OUTPUT_WORKBOOK)
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
