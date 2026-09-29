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

**Purpose:** This notebook reads the single secure results workbook from the completed 92-patient development-only five-fold CV. It displays the architecture ranking, the Top-3 and Top-5 pools, and the ranking of the 10 candidate ensemble configurations.

> **Boundary:** This notebook never reads the 46-patient test cohort or `metrics2026.xlsx`. It does not write a selected configuration. After reviewing the displayed development-only ranking, copy the selected Top-3/Top-5 list and rule manually into the final ensemble-analysis notebook.

The candidate set contains **10 configurations**: Sum, Median, Max, Majority Voting, and Weighted Sum across the Top-3 and Top-5 development-ranked pools. Borda Count is excluded.
"""),
    markdown("""## 0. Secure input workbook

Edit only the secure local path. The input workbook includes patient-linked OOF predictions and remains outside Git.
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

from analysis.r1_ensemble_core import configuration_ranking, validate_oof_predictions

# One secure workbook created after the complete 11 architectures × 5 folds run.
CV_WORKBOOK = REPO_ROOT.parent / 'secure_outputs' / 'ensemble_selection_cv_2026_full' / 'ensemble_selection_cv_results.xlsx'

print('Repository:', REPO_ROOT)
print('Development-only CV workbook:', CV_WORKBOOK)
print('Test data and metrics2026.xlsx are not inputs to this notebook.')
"""),
    markdown("""## 1. Read and validate the completed development-only CV workbook

The workbook must represent a complete 11-architecture × 5-fold run: one OOF probability per development patient and architecture. A partial technical preflight is rejected.
"""),
    code("""if not CV_WORKBOOK.exists():
    raise FileNotFoundError('Completed CV results workbook was not found. Update CV_WORKBOOK before continuing.')

sheets = pd.read_excel(CV_WORKBOOK, sheet_name=None, engine='openpyxl')
required_sheets = {'Study_lock', 'Architecture_ranking', 'Candidate_pools', 'OOF_predictions'}
missing_sheets = required_sheets - set(sheets)
if missing_sheets:
    raise RuntimeError(f'CV workbook lacks required sheets: {sorted(missing_sheets)}')

study_lock = dict(zip(sheets['Study_lock']['item'].astype(str), sheets['Study_lock']['value'].astype(str)))
if study_lock.get('test_data_accessed') != 'false':
    raise RuntimeError('The CV workbook does not establish that test data were excluded.')
if study_lock.get('partial_technical_preflight') != 'false':
    raise RuntimeError('A partial CV preflight cannot be used for configuration ranking.')

oof = validate_oof_predictions(sheets['OOF_predictions'])
architecture_ranking = sheets['Architecture_ranking'].sort_values('development_rank').reset_index(drop=True)
candidate_pools = sheets['Candidate_pools'].sort_values(['pool_name', 'pool_rank']).reset_index(drop=True)

if oof.model_name.nunique() != 11 or oof.patient_id.nunique() != 92 or len(oof) != 92 * 11 or oof.fold.nunique() != 5:
    raise RuntimeError('Expected 1,012 OOF rows from 11 architectures, 92 patients, and 5 folds.')
if len(architecture_ranking) != 11:
    raise RuntimeError('Architecture_ranking must contain 11 architectures.')
if len(candidate_pools) != 8:
    raise RuntimeError('Candidate_pools must contain the Top-3 and Top-5 rows.')

print(f'Validated {len(oof)} OOF predictions from {oof.model_name.nunique()} architectures and {oof.patient_id.nunique()} development patients.')
print('Independent test data accessed: False')
"""),
    markdown("""## 2. Inspect the 11-model development ranking and candidate pools

The CV runner has already created this ranking in the input workbook. The Top-3 and Top-5 lists below are the exact lists to copy into the final ensemble-analysis notebook after configuration ranking is reviewed.
"""),
    code("""display(architecture_ranking)
display(candidate_pools)

TOP3_MODELS = candidate_pools.loc[candidate_pools['pool_name'] == 'Top-3'].sort_values('pool_rank')['model_name'].astype(str).tolist()
TOP5_MODELS = candidate_pools.loc[candidate_pools['pool_name'] == 'Top-5'].sort_values('pool_rank')['model_name'].astype(str).tolist()
if TOP3_MODELS != TOP5_MODELS[:3]:
    raise RuntimeError('The workbook Top-3 list is not nested within Top-5.')
print('Top-3 models:', TOP3_MODELS)
print('Top-5 models:', TOP5_MODELS)
"""),
    markdown("""## 3. Rank the 10 development-only ensemble configurations

**Primary criterion:** highest pooled OOF AUC.

**Exact-tie rules:** at an exact tie to 12 decimal places, select the smaller pool (Top-3 before Top-5), then use the fixed rule order: Sum, Median, Max, Majority Voting, and Weighted Sum.

For Weighted Sum, the weights are calculated solely from pooled development OOF AUC values. This cell displays the ranking only; it creates no files and does not select a test-set winner.
"""),
    code("""pools = {'Top-3': TOP3_MODELS, 'Top-5': TOP5_MODELS}
configuration_table, configuration_predictions, weights_by_configuration = configuration_ranking(oof, pools)
if len(configuration_table) != 10:
    raise RuntimeError(f'Expected 10 configurations; found {len(configuration_table)}.')

display(configuration_table)
print('Copy the chosen pool/rule and the displayed Top-3/Top-5 lists into R1_final_ensemble_analysis.ipynb.')
print('This notebook has not written a selection file or used test data.')
"""),
    markdown("""## Next step

Open `notebooks/R1_final_ensemble_analysis.ipynb`. Manually enter:

```text
TOP3_MODELS = [...]       # copied from Section 2
TOP5_MODELS = [...]       # copied from Section 2
SELECTED_POOL = 'Top-3' or 'Top-5'
SELECTED_RULE = 'sum' | 'median' | 'max' | 'majority_vote' | 'weighted_sum'
```

Then, and only then, set `RUN_ANALYSIS=True`. The final notebook reads the completed 1,100-run `metrics2026.xlsx` workbook, applies this single configuration to run-matched seed predictions, and creates 100 ensemble replicates plus patient-level results.
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
