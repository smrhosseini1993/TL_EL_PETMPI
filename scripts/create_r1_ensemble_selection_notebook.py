#!/usr/bin/env python3
"""Create the version-controlled development-only ensemble-selection notebook."""
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
    markdown("""# R1 ensemble configuration selection — development data only

**Purpose:** This notebook selects one ensemble configuration using **only** the 92-patient development cohort. It never reads the 46-patient independent test cohort or the 1,100 historic-split test predictions.

> **Selection boundary:** The selected Top-3/Top-5 pool, ensemble rule, weighted-sum weights if applicable, and 0.50 threshold are fixed here before the final test analysis. The later final-ensemble notebook may read only the resulting lock file and must not test alternative configurations.

The candidate set contains **10 configurations**: five rules (Sum, Median, Max, Majority Voting, and Weighted Sum) applied to Top-3 and Top-5 pools. Borda Count is deliberately excluded.
"""),
    markdown("""## 0. Configuration

Edit only the secure local paths. The CV outputs and every notebook output contain patient-linked predictions and must remain on approved secure storage; none are saved in Git.
"""),
    code("""from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Markdown

# Run from the repository `Experiment2026/`, `notebooks/`, or repository root.
CWD = Path.cwd().resolve()
if (CWD / 'analysis').exists():
    REPO_ROOT = CWD
elif (CWD.parent / 'analysis').exists():
    REPO_ROOT = CWD.parent
else:
    raise RuntimeError('Open this notebook from the repository root, notebooks/, or Experiment2026/.')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.r1_ensemble_core import (
    architecture_oof_ranking,
    candidate_pools,
    configuration_ranking,
    selection_lock,
    validate_oof_predictions,
    write_json,
)

# Secure output of the completed full 11-model × 5-fold CV run.
CV_OUTPUT_DIR = REPO_ROOT.parent / 'secure_outputs' / 'ensemble_selection_cv_2026_full'
# Separate secure output directory for this notebook's rankings and lock file.
SELECTION_OUTPUT_DIR = REPO_ROOT.parent / 'secure_outputs' / 'ensemble_selection_2026'
OOF_FILE = CV_OUTPUT_DIR / 'predictions' / 'oof_predictions_all.csv'
CV_MANIFEST_FILE = CV_OUTPUT_DIR / 'manifests' / 'cv_run_manifest.json'

# This remains False until the complete 10-row table has been reviewed.
# When changed to True, the lock still selects the rank-1 configuration automatically;
# do not type model names, rules, or weights manually.
WRITE_SELECTION_LOCK = False

print('Repository:', REPO_ROOT)
print('Development OOF predictions:', OOF_FILE)
print('CV manifest:', CV_MANIFEST_FILE)
print('Secure selection output:', SELECTION_OUTPUT_DIR)
print('Test data is not an input to this notebook.')
"""),
    markdown("""## 1. Load and validate the complete development-only CV output

The full CV run must contain one OOF probability for each of 92 development patients and all 11 architectures. A partial technical preflight is rejected and cannot be used to select an ensemble.
"""),
    code("""if not OOF_FILE.exists() or not CV_MANIFEST_FILE.exists():
    raise FileNotFoundError('Completed full CV outputs were not found. Update CV_OUTPUT_DIR before continuing.')

cv_manifest = __import__('json').loads(CV_MANIFEST_FILE.read_text(encoding='utf-8'))
if cv_manifest.get('test_data_accessed') is not False:
    raise RuntimeError('The CV manifest does not establish that test data were excluded.')
if cv_manifest.get('partial_technical_preflight'):
    raise RuntimeError('A partial CV preflight cannot be used for configuration selection.')

raw_oof = pd.read_csv(OOF_FILE)
oof = validate_oof_predictions(raw_oof)
if oof['model_name'].nunique() != 11:
    raise RuntimeError(f'Expected 11 architectures in the complete CV output; found {oof.model_name.nunique()}.')
if oof['patient_id'].nunique() != 92:
    raise RuntimeError(f'Expected 92 development patients; found {oof.patient_id.nunique()}.')
if len(oof) != 92 * 11:
    raise RuntimeError(f'Expected 1,012 OOF rows; found {len(oof)}.')
if oof['fold'].nunique() != 5:
    raise RuntimeError(f'Expected 5 folds; found {oof.fold.nunique()}.')

print(f'Validated {len(oof)} development OOF predictions.')
print(f'Architectures: {oof.model_name.nunique()} | patients: {oof.patient_id.nunique()} | folds: {oof.fold.nunique()}')
print('Independent test data accessed: False')
"""),
    markdown("""## 2. Rank the 11 individual architectures using pooled OOF AUC

This table defines the Top-3 and Top-5 candidate pools. The ranking uses one out-of-fold probability per development patient and architecture. It does not use the historic 46-patient test cohort.
"""),
    code("""architecture_ranking = architecture_oof_ranking(oof)
pools = candidate_pools(architecture_ranking)

display(architecture_ranking)
print('Top-3 pool:', pools['Top-3'])
print('Top-5 pool:', pools['Top-5'])
"""),
    markdown("""## 3. Evaluate and rank the 10 prespecified ensemble configurations

**Primary selection criterion:** highest pooled out-of-fold AUC.

**Exact-tie rules:** a tie at 12 decimal places is resolved first by selecting the smaller pool (Top-3 before Top-5), then by the fixed rule order: Sum, Median, Max, Majority Voting, and Weighted Sum. These rules are declared before the selection table is viewed.

For Weighted Sum, weights are calculated from pooled OOF AUC values within the development cohort only.
"""),
    code("""configuration_table, configuration_predictions, weights_by_configuration = configuration_ranking(oof, pools)
if len(configuration_table) != 10:
    raise RuntimeError(f'Expected exactly 10 configurations; found {len(configuration_table)}.')

SELECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
architecture_ranking.to_csv(SELECTION_OUTPUT_DIR / 'architecture_oof_ranking.csv', index=False)
configuration_table.to_csv(SELECTION_OUTPUT_DIR / 'configuration_oof_ranking.csv', index=False)
configuration_predictions.to_csv(SELECTION_OUTPUT_DIR / 'oof_configuration_predictions.csv', index=False)

display(configuration_table)
selected_preview = configuration_table.iloc[0]
print('Rank-1 development-only configuration:')
display(pd.DataFrame([selected_preview]))
print('Selection tables were written to:', SELECTION_OUTPUT_DIR)
"""),
    markdown("""## 4. Review and write the final selection lock

Review the complete 10-row table above. When the table is complete and the prespecified selection rule has been accepted, change `WRITE_SELECTION_LOCK` to `True` and run this cell.

The cell does **not** permit manually choosing a different configuration. It writes the rank-1 configuration, its Top-3/Top-5 constituent models, weighted-sum weights when applicable, the 0.50 threshold, source-file hashes, and the instruction that only this configuration may be evaluated later on the independent test cohort.
"""),
    code("""if WRITE_SELECTION_LOCK:
    lock = selection_lock(
        architecture_ranking_frame=architecture_ranking,
        configuration_ranking_frame=configuration_table,
        pools=pools,
        weights_by_configuration=weights_by_configuration,
        oof_file=OOF_FILE,
        cv_manifest_file=CV_MANIFEST_FILE,
    )
    lock_file = SELECTION_OUTPUT_DIR / 'ensemble_selection_lock.json'
    write_json(lock_file, lock)
    print('Wrote development-only ensemble-selection lock:', lock_file)
    display(pd.DataFrame([lock['selected_configuration']]))
else:
    print('Selection lock not written. Set WRITE_SELECTION_LOCK = True only after reviewing the complete 10-row development-only ranking.')
"""),
    markdown("""## Deliverables

```text
secure_outputs/ensemble_selection_2026/
├── architecture_oof_ranking.csv
├── configuration_oof_ranking.csv
├── oof_configuration_predictions.csv
└── ensemble_selection_lock.json  # created only after WRITE_SELECTION_LOCK=True
```

**Next step:** The final ensemble-analysis notebook will read `ensemble_selection_lock.json` and the existing 1,100 historic 61/31/46 TL runs. It will create 100 run-matched ensemble prediction replicates only for this selected configuration. It will not compare any unselected configuration on the independent test cohort.
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
