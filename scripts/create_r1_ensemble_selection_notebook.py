#!/usr/bin/env python3
"""Create the version-controlled display-only ensemble-ranking notebook."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "notebooks" / "R1_ensemble_configuration_selection.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown("""# R1 ensemble configuration ranking — development data only

**Purpose:** This notebook reads the raw secure results workbook from the completed 92-patient, five-fold CV. It first displays the ranking of the 11 transfer-learning architectures. You then manually enter the Top-3 and Top-5 pools in a later cell. The notebook displays the ranking of the resulting 10 ensemble configurations.

> **Boundary:** This notebook never reads the 46-patient test cohort or `metrics2026.xlsx`. It writes no files. It is used only to inspect development-only rankings before the selected pool and rule are manually entered into the final EL notebook.

The candidate set contains **10 configurations**: Sum, Median, Max, Majority Voting, and Weighted Sum across the manually entered Top-3 and Top-5 pools. Borda Count is excluded.
"""),
    markdown("""## 0. Secure raw CV workbook

Edit only the secure local path. The workbook contains patient-linked OOF predictions and remains outside Git.
"""),
    code("""from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import display

# Run from the repository root, `notebooks/`, or `Experiment2026/`.
CWD = Path.cwd().resolve()
if (CWD / 'analysis').exists():
    REPO_ROOT = CWD
elif (CWD.parent / 'analysis').exists():
    REPO_ROOT = CWD.parent
else:
    raise RuntimeError('Open this notebook from the repository root, notebooks/, or Experiment2026/.')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.r1_ensemble_core import architecture_oof_ranking, configuration_ranking, validate_oof_predictions

# One secure workbook created after the complete 11 architectures × 5 folds run.
CV_WORKBOOK = REPO_ROOT.parent / 'secure_outputs' / 'ensemble_selection_cv_2026_full' / 'ensemble_selection_cv_raw_results.xlsx'

print('Repository:', REPO_ROOT)
print('Development-only raw CV workbook:', CV_WORKBOOK)
print('Test data and metrics2026.xlsx are not inputs to this notebook.')
"""),
    markdown("""## 1. Read and validate the complete development-only CV workbook

The workbook must represent a complete 11-architecture × 5-fold run: one OOF probability per development patient and architecture. A partial technical preflight is rejected.
"""),
    code("""if not CV_WORKBOOK.exists():
    raise FileNotFoundError('Completed raw CV workbook was not found. Update CV_WORKBOOK before continuing.')

sheets = pd.read_excel(CV_WORKBOOK, sheet_name=None, engine='openpyxl')
required_sheets = {'Study_lock', 'OOF_predictions'}
missing_sheets = required_sheets - set(sheets)
if missing_sheets:
    raise RuntimeError(f'CV workbook lacks required sheets: {sorted(missing_sheets)}')

study_lock = dict(zip(sheets['Study_lock']['item'].astype(str), sheets['Study_lock']['value'].astype(str)))
if study_lock.get('test_data_accessed') != 'false':
    raise RuntimeError('The CV workbook does not establish that test data were excluded.')
if study_lock.get('partial_technical_preflight') != 'false':
    raise RuntimeError('A partial CV preflight cannot be used for ensemble configuration ranking.')

oof = validate_oof_predictions(sheets['OOF_predictions'])
if oof.model_name.nunique() != 11 or oof.patient_id.nunique() != 92 or len(oof) != 92 * 11 or oof.fold.nunique() != 5:
    raise RuntimeError('Expected 1,012 OOF rows from 11 architectures, 92 patients, and 5 folds.')

print(f'Validated {len(oof)} OOF predictions from {oof.model_name.nunique()} architectures and {oof.patient_id.nunique()} development patients.')
print('Independent test data accessed: False')
"""),
    markdown("""## 2. Display the 11-model development ranking

This ranking is calculated from pooled OOF AUC. Inspect it before completing the manual pool cell below.
"""),
    code("""architecture_ranking = architecture_oof_ranking(oof)
display(architecture_ranking)
"""),
    markdown("""## 3. Manually enter the Top-3 and Top-5 pools

After reviewing the displayed 11-model ranking, enter the architecture names below in development-rank order. The Top-3 list must be the first three names in the Top-5 list.

This manual cell is intentional: it gives you direct review control while keeping the source of every choice visible in the notebook. Do **not** use any test results to fill these lists.
"""),
    code("""# Enter these names manually after reviewing the Section 2 ranking.
TOP3_MODELS = [
    # 'Architecture_1',
    # 'Architecture_2',
    # 'Architecture_3',
]
TOP5_MODELS = [
    # 'Architecture_1',
    # 'Architecture_2',
    # 'Architecture_3',
    # 'Architecture_4',
    # 'Architecture_5',
]

if len(TOP3_MODELS) != 3 or len(TOP5_MODELS) != 5:
    raise ValueError('Enter exactly three Top-3 models and five Top-5 models.')
if TOP3_MODELS != TOP5_MODELS[:3]:
    raise ValueError('Top-3 must equal the first three names in Top-5.')
if not set(TOP5_MODELS).issubset(set(oof.model_name.astype(str))):
    raise ValueError('Every manually entered model must appear in the Section 2 development ranking.')

print('Manually entered Top-3:', TOP3_MODELS)
print('Manually entered Top-5:', TOP5_MODELS)
"""),
    markdown("""## 4. Display the 10 development-only ensemble configurations

**Primary ranking criterion:** pooled OOF AUC.

**Exact-tie rules:** at an exact tie to 12 decimal places, the smaller pool is ranked first (Top-3 before Top-5), then the fixed rule order is applied: Sum, Median, Max, Majority Voting, and Weighted Sum.

For Weighted Sum, weights are calculated only from development OOF AUC values. This cell displays the 10-row table only. It writes no file and does not select a test-set winner.
"""),
    code("""pools = {'Top-3': TOP3_MODELS, 'Top-5': TOP5_MODELS}
configuration_table, configuration_predictions, weights_by_configuration = configuration_ranking(oof, pools)
if len(configuration_table) != 10:
    raise RuntimeError(f'Expected 10 configurations; found {len(configuration_table)}.')

display(configuration_table)
print('Copy your selected pool and rule, plus these exact Top-3/Top-5 lists, into R1_final_ensemble_analysis.ipynb.')
print('This notebook has not written a file or used test data.')
"""),
    markdown("""## Next step

Open `notebooks/R1_final_ensemble_analysis.ipynb`. Manually copy the pool lists and your chosen configuration:

```text
TOP3_MODELS = [...]
TOP5_MODELS = [...]
SELECTED_POOL = 'Top-3' or 'Top-5'
SELECTED_RULE = 'sum' | 'median' | 'max' | 'majority_vote' | 'weighted_sum'
```

Then set `RUN_ANALYSIS=True`. The final notebook reads `metrics2026.xlsx`, applies the one manually entered configuration to run-matched predictions, and creates one **raw 100-run ensemble Excel workbook** for later combined paper reporting.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (polarmaps2024)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print("Wrote {0}".format(OUTPUT))
