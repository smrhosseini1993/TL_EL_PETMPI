#!/usr/bin/env python3
"""Create the version-controlled final raw ensemble-run notebook."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "notebooks" / "R1_final_ensemble_analysis.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown("""# R1 final ensemble runs — historic 61/31/46 split

**Purpose:** This notebook applies **one manually entered, development-selected ensemble configuration** to the completed `metrics2026.xlsx` workbook. It creates 100 run-matched ensemble runs and writes one simple raw Excel workbook.

> **Boundary:** This notebook does not rank architectures, rank ensemble configurations, calculate confidence intervals, or choose a test-set winner. Its final two cells display descriptive median (IQR) seed-stability summaries only. Later paper reporting will combine this raw 100-run EL workbook with the 1,100 TL runs, 100 CNN runs, and the clinical-reader output.

For seed 1, it takes seed-1 predictions from every selected constituent model and applies the selected rule. It repeats this for seeds 2–100. No model is retrained and no additional patients are used.
"""),
    markdown("""## 0. Secure paths and manually entered configuration

Copy the Top-3/Top-5 lists and selected pool/rule from the middle development-only notebook. The Top-3 list must be the first three entries of Top-5. The selected rule must be one of `sum`, `median`, `max`, `majority_vote`, or `weighted_sum`.
"""),
    code("""from pathlib import Path
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

from analysis.r1_ensemble_core import (
    create_run_matched_ensemble_predictions,
    load_historic_tl_workbook,
    raw_ensemble_run_results,
    raw_ensemble_study_lock,
    run_stability_summary,
    validate_manual_configuration,
)

# Secure local Mac inputs/outputs. These files must never be committed to Git.
# With the recommended layout, PROJECT_ROOT is PETMPI_R1_final_analysis/.
PROJECT_ROOT = REPO_ROOT.parent
INPUT_DIR = PROJECT_ROOT / 'secure_inputs'
TL_WORKBOOK = INPUT_DIR / 'metrics2026.xlsx'
OUTPUT_WORKBOOK = INPUT_DIR / 'R1_final_ensemble_raw_runs.xlsx'

# Enter these names manually after reviewing the middle development-only notebook.
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
SELECTED_POOL = 'Top-5'      # 'Top-3' or 'Top-5'
SELECTED_RULE = 'max'        # 'sum', 'median', 'max', 'majority_vote', or 'weighted_sum'
# Only for SELECTED_RULE='weighted_sum': enter the development-only weights.
WEIGHTED_SUM_WEIGHTS = {}

# Keep False until the manually entered configuration above is complete.
RUN_ANALYSIS = False

EXPECTED_MODELS = (
    'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'InceptionV3',
    'InceptionResNetV2', 'DenseNet169', 'DenseNet201', 'MobileNetV2', 'Xception',
)

print('Historic 1,100-run workbook:', TL_WORKBOOK)
print('Raw ensemble output workbook:', OUTPUT_WORKBOOK)
print('Selected pool/rule:', SELECTED_POOL, '/', SELECTED_RULE)
print('This notebook runs one manually entered configuration only.')
"""),
    markdown("""## 1. Validate the manually entered configuration

This checks the pool/rule structure only. It does not read test predictions.
"""),
    code("""configuration = validate_manual_configuration(
    top3_models=TOP3_MODELS,
    top5_models=TOP5_MODELS,
    selected_pool=SELECTED_POOL,
    selected_rule=SELECTED_RULE,
    weighted_sum_weights=WEIGHTED_SUM_WEIGHTS,
)
display(pd.DataFrame([configuration]))
"""),
    markdown("""## 2. Read the completed 1,100-run TL workbook

The input must contain exactly 11 architectures × 100 seeds and 46 ordered test predictions per model/seed. It is opened read-only.
"""),
    code("""if not RUN_ANALYSIS:
    print('Set RUN_ANALYSIS = True only after confirming the manual configuration above.')
else:
    historic_predictions, test_manifest, study_lock = load_historic_tl_workbook(TL_WORKBOOK, EXPECTED_MODELS)
    print(f'Validated {len(historic_predictions)} prediction records.')
    print(f'Architectures: {historic_predictions.model_name.nunique()} | seeds: {historic_predictions.seed.nunique()} | test patients: {historic_predictions.patient_id.nunique()}')
"""),
    markdown("""## 3. Create the 100 run-matched ensemble runs

Each ensemble seed uses only same-seed constituent predictions. This cell creates raw prediction and metric records for the 100 ensemble runs; it does not calculate any paper summaries.
"""),
    code("""if RUN_ANALYSIS:
    ensemble_predictions, ensemble_run_metrics = create_run_matched_ensemble_predictions(
        historic_predictions=historic_predictions,
        configuration=configuration,
    )
    assert len(ensemble_predictions) == 100 * 46
    assert len(ensemble_run_metrics) == 100
    display(ensemble_run_metrics.head())
    print('Created 100 run-matched ensemble runs with 46 predictions per run.')
"""),
    markdown("""## 4. Write the raw EL workbook

The workbook is intentionally simple. It contains only the raw ensemble runs, their ordered 46-patient probabilities, the test manifest, and a short provenance sheet. A later combined-analysis notebook will merge this with TL, CNN, and clinical-reader inputs for medians, patient-level 95% CIs, tables, and figures.
"""),
    code("""if RUN_ANALYSIS:
    OUTPUT_WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
    raw_run_results = raw_ensemble_run_results(
        ensemble_predictions=ensemble_predictions,
        ensemble_run_metrics=ensemble_run_metrics,
        configuration=configuration,
    )
    raw_lock = raw_ensemble_study_lock(configuration, TL_WORKBOOK)
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine='openpyxl', mode='w') as writer:
        raw_run_results.to_excel(writer, sheet_name='Run_results', index=False)
        test_manifest.to_excel(writer, sheet_name='Patient_manifest', index=False)
        raw_lock.to_excel(writer, sheet_name='Study_lock', index=False)
    print('Wrote raw 100-run ensemble workbook:', OUTPUT_WORKBOOK)
    print('Sheets: Run_results | Patient_manifest | Study_lock')
"""),
    markdown("""## 5. Descriptive stability across the 100 ensemble runs

This table is the familiar median (IQR) summary across the 100 run-matched ensemble realizations. It is **descriptive training-stability information only**. It is not a patient-level confidence interval and it is not used for any hypothesis test.
"""),
    code("""if RUN_ANALYSIS:
    stability_summary = run_stability_summary(ensemble_run_metrics).iloc[0]
    stability_table = pd.DataFrame([
        {
            'Metric': metric,
            'Median': float(stability_summary['{0}_median'.format(metric)]),
            'IQR': float(stability_summary['{0}_iqr'.format(metric)]),
        }
        for metric in ('ACC', 'PRE', 'SEN', 'SPE', 'F1S', 'AUC')
    ])
    display(stability_table.style.format({'Median': '{:.3f}', 'IQR': '{:.3f}'}))
    print('Descriptive summary across {0} run-matched ensemble seeds.'.format(int(stability_summary['n_runs'])))
"""),
    markdown("""## 6. Descriptive confusion-matrix stability across the 100 ensemble runs

Each seed produces one confusion matrix for the same 46 test patients. This table summarizes the run-level TP, TN, FP, and FN counts using the median, first quartile (Q1), third quartile (Q3), IQR, minimum, and maximum. It is descriptive only; the final patient-level confusion matrix is produced later by `results_publish.ipynb` from seed-mean probabilities.
"""),
    code("""if RUN_ANALYSIS:
    confusion_stability_table = pd.DataFrame([
        {
            'Component': component,
            'Median': float(ensemble_run_metrics[column].median()),
            'Q1': float(ensemble_run_metrics[column].quantile(0.25)),
            'Q3': float(ensemble_run_metrics[column].quantile(0.75)),
            'IQR': float(ensemble_run_metrics[column].quantile(0.75) - ensemble_run_metrics[column].quantile(0.25)),
            'Minimum': int(ensemble_run_metrics[column].min()),
            'Maximum': int(ensemble_run_metrics[column].max()),
        }
        for component, column in (('True positives (TP)', 'tp'), ('True negatives (TN)', 'tn'), ('False positives (FP)', 'fp'), ('False negatives (FN)', 'fn'))
    ])
    display(confusion_stability_table.style.format({'Median': '{:.1f}', 'Q1': '{:.1f}', 'Q3': '{:.1f}', 'IQR': '{:.1f}'}))
"""),
    markdown("""## Raw output

```text
secure_inputs/
└── R1_final_ensemble_raw_runs.xlsx
    ├── Run_results       # 100 raw ensemble runs with metrics and 46 ordered probabilities
    ├── Patient_manifest  # test-patient order and labels
    └── Study_lock        # manually entered pool/rule and seed-matching provenance
```

This is the EL input for the later combined reporting notebook. It will be combined with the original 1,100 TL runs, 100 CNN runs, and the clinical reader—not used to select any further ensemble configuration.
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
