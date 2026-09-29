#!/usr/bin/env python3
"""Create the version-controlled final historic-split ensemble analysis notebook."""
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
    markdown("""# R1 final ensemble analysis — historic 61/31/46 split

**Purpose:** This notebook applies **one already development-selected ensemble configuration** to the completed `metrics2026.xlsx` workbook. It reads the 100 historic-split test prediction runs for the selected constituent models and creates 100 **run-matched ensemble replicates**.

> **Boundary:** This notebook does not rank architectures, rank ensemble configurations, or select a winner from test results. Enter the Top-3/Top-5 lists and the one selected rule from the development-only selection notebook. Do not use this notebook to try alternative configurations on the 46-patient test cohort.

It reports both valid result types:

1. **Stability:** median (IQR) across 100 run-matched ensemble replicates.
2. **Patient-level performance:** for each of 46 test patients, the arithmetic mean of 100 ensemble probabilities, followed by bootstrap 95% confidence intervals.
"""),
    markdown("""## 0. Secure paths and manually entered, development-selected configuration

Set the secure workbook and output paths. Then enter the exact Top-3 list, Top-5 list, selected pool, and rule shown by the completed development-only selection notebook.

The Top-3 list must be the first three names in the Top-5 list. This notebook validates that all named models exist in the completed 11×100 workbook and creates no test-set ranking.
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

from analysis.r1_ensemble_core import (
    create_run_matched_ensemble_predictions,
    load_historic_tl_workbook,
    patient_level_summary,
    run_stability_summary,
    validate_manual_configuration,
    write_json,
)

# Secure local paths. These files must never be committed to Git.
TL_WORKBOOK = REPO_ROOT / 'Experiment2026' / 'metrics2026.xlsx'
ANALYSIS_OUTPUT_DIR = REPO_ROOT.parent / 'secure_analysis' / 'R1_final_ensemble_2026'
OUTPUT_WORKBOOK = ANALYSIS_OUTPUT_DIR / 'R1_final_ensemble_results.xlsx'
OUTPUT_CONFIGURATION_RECORD = ANALYSIS_OUTPUT_DIR / 'final_ensemble_configuration_record.json'

# Copy these exact lists from the completed development-only selection notebook.
# Keep the CV ranking order: Top-3 must be the first three entries of Top-5.
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

# Copy the one selected configuration from the development-only ranking.
SELECTED_POOL = 'Top-5'      # 'Top-3' or 'Top-5'
SELECTED_RULE = 'max'        # 'sum', 'median', 'max', 'majority_vote', or 'weighted_sum'
# Only for SELECTED_RULE='weighted_sum': copy the development-OOF weights.
WEIGHTED_SUM_WEIGHTS = {}

# The final analysis is intentionally off until the configuration above is complete.
RUN_ANALYSIS = False
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260929

EXPECTED_MODELS = (
    'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'InceptionV3',
    'InceptionResNetV2', 'DenseNet169', 'DenseNet201', 'MobileNetV2', 'Xception',
)

print('Historic 1,100-run workbook:', TL_WORKBOOK)
print('Secure ensemble output directory:', ANALYSIS_OUTPUT_DIR)
print('Selected pool/rule:', SELECTED_POOL, '/', SELECTED_RULE)
print('This notebook evaluates exactly one manually entered, development-selected configuration.')
"""),
    markdown("""## 1. Validate the manually entered configuration

This validates only the configuration structure. It does not access test predictions.
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
    markdown("""## 2. Read and validate the completed historic TL workbook

The workbook must contain exactly 11 architectures × 100 seeds and 46 test probabilities per model/seed. The reader opens it read-only. The historic 61/31/46 allocation is retained so the ensemble is compared fairly with its constituent models and the reference CNN.
"""),
    code("""if not RUN_ANALYSIS:
    print('Set RUN_ANALYSIS = True only after confirming the development-selected configuration above.')
else:
    historic_predictions, test_manifest, study_lock = load_historic_tl_workbook(TL_WORKBOOK, EXPECTED_MODELS)
    print(f'Validated {len(historic_predictions)} test-prediction records.')
    print(f'Architectures: {historic_predictions.model_name.nunique()} | seeds: {historic_predictions.seed.nunique()} | test patients: {historic_predictions.patient_id.nunique()}')
    display(study_lock)
"""),
    markdown("""## 3. Create the 100 run-matched ensemble replicates

For every seed from 1 to 100, this takes the same-numbered test prediction from each selected constituent model and applies the one selected rule. For example, seed 17 uses seed-17 predictions from every constituent model. No models are retrained and no additional training patients are introduced.
"""),
    code("""if RUN_ANALYSIS:
    ensemble_predictions, ensemble_run_metrics = create_run_matched_ensemble_predictions(
        historic_predictions=historic_predictions,
        configuration=configuration,
    )
    assert ensemble_predictions.seed.nunique() == 100
    assert ensemble_predictions.patient_id.nunique() == 46
    assert len(ensemble_predictions) == 100 * 46
    display(ensemble_run_metrics.head())
    print('Created 100 run-matched ensemble replicates with 46 predictions each.')
"""),
    markdown("""## 4. Descriptive 100-run stability

This reproduces the prior median (IQR) stability presentation. These 100 replicates describe stochastic-training variability; they are not independent clinical samples and are not used for run-level hypothesis testing.
"""),
    code("""if RUN_ANALYSIS:
    ensemble_stability = run_stability_summary(ensemble_run_metrics)
    display(ensemble_stability)
"""),
    markdown("""## 5. Patient-level final ensemble result with 95% confidence intervals

For each independent test patient, this averages the 100 run-matched ensemble probabilities. This produces one final probability per patient before the fixed 0.50 threshold is applied. Bootstrap resampling is at the patient level.
"""),
    code("""if RUN_ANALYSIS:
    ensemble_patient_predictions, ensemble_patient_metrics = patient_level_summary(
        ensemble_predictions=ensemble_predictions,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        random_seed=BOOTSTRAP_SEED,
    )
    display(ensemble_patient_metrics)
    display(ensemble_patient_predictions.head())
"""),
    markdown("""## 6. Write the secure final ensemble package

This writes one human-readable Excel workbook and one JSON configuration record. The workbook contains the selected configuration, run-level stability record, all 4,600 run-matched ensemble predictions, the 46 patient-level mean probabilities, and patient-level confidence intervals.
"""),
    code("""if RUN_ANALYSIS:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configuration_record = {
        'analysis_scope': 'one development-selected ensemble configuration evaluated on historic 61/31/46 test predictions',
        'test_configuration_selection_performed': False,
        'configuration': configuration,
        'input_workbook': str(TL_WORKBOOK),
        'n_historic_tl_runs': 1100,
        'n_ensemble_replicates': 100,
        'n_test_patients': 46,
        'seed_matching': 'same seed across every selected constituent model',
        'patient_level_aggregation': 'mean probability across 100 run-matched ensemble replicates',
        'classification_threshold': 0.50,
        'bootstrap_iterations': BOOTSTRAP_ITERATIONS,
        'bootstrap_seed': BOOTSTRAP_SEED,
    }
    write_json(OUTPUT_CONFIGURATION_RECORD, configuration_record)
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine='openpyxl', mode='w') as writer:
        pd.DataFrame([configuration_record]).to_excel(writer, sheet_name='Study_lock', index=False)
        pd.DataFrame([configuration]).to_excel(writer, sheet_name='Selected_configuration', index=False)
        ensemble_run_metrics.to_excel(writer, sheet_name='Ensemble_run_metrics', index=False)
        ensemble_stability.to_excel(writer, sheet_name='Ensemble_stability', index=False)
        ensemble_predictions.to_excel(writer, sheet_name='Ensemble_predictions_100', index=False)
        ensemble_patient_predictions.to_excel(writer, sheet_name='Patient_predictions', index=False)
        ensemble_patient_metrics.to_excel(writer, sheet_name='Patient_level_results', index=False)
        test_manifest.to_excel(writer, sheet_name='Test_manifest', index=False)
    print('Wrote secure final ensemble workbook:', OUTPUT_WORKBOOK)
    print('Wrote secure configuration record:', OUTPUT_CONFIGURATION_RECORD)
"""),
    markdown("""## Deliverables

```text
secure_analysis/R1_final_ensemble_2026/
├── R1_final_ensemble_results.xlsx
└── final_ensemble_configuration_record.json
```

The next results-integration notebook can combine the final ensemble output with the 11-model TL patient-level outputs, reference-CNN/clinical-reader data, and conventional baseline workbook. It must not recreate or select alternative ensemble configurations on the independent test cohort.
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
