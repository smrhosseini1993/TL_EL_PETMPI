#!/usr/bin/env python3
"""Create the version-controlled R1 TL Sections 0–6 review notebook without patient data."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "notebooks" / "R1_TL_results_V1.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown("""# R1 transfer-learning results — Version 1

**Scope:** completed locked TL experiment only (Sections 0–6). This notebook reads the final SQLite store **read-only**, creates reproducible local tables/figures, and does not train models or select a winner from test results.

> **Interpretation boundary:** the 100 seeds describe stochastic-training stability. They are not independent clinical samples. Patient-level reporting uses one fixed final probability per test patient/model: the arithmetic mean of the 100 seed probabilities, then the locked 0.50 threshold.

Sections on conventional PET baselines, the reference CNN/reader, and the ensemble are deliberately excluded from Version 1 and will be analysed in a separate later notebook.
"""),
    markdown("""## 0. Configuration

Edit only the secure local paths below if your folder names differ. These paths are not saved in Git. The output package contains aggregated patient-level predictions and must remain on approved secure storage.
"""),
    code("""from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import display, Image, Markdown

# Run this notebook from the repository's `code/` folder or its `notebooks/` folder.
CWD = Path.cwd().resolve()
REPO_ROOT = CWD if (CWD / 'analysis').exists() else CWD.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.r1_tl_results import AnalysisSettings, load_r1_results, run_v1_analysis

# Secure local folders beside the repository clone. Do not put patient data/outputs in Git.
RESULTS_DIR = REPO_ROOT.parent / 'R1_final_TL'
CV_OUTPUT_DIR = REPO_ROOT.parent / 'secure_outputs' / 'cv_full_27x5x11'
ANALYSIS_OUTPUT_DIR = REPO_ROOT.parent / 'secure_analysis' / 'R1_TL_V1'

SETTINGS = AnalysisSettings(
    aggregation_rule='mean_probability_across_100_seeds',
    threshold=0.50,
    metric_bootstrap_iterations=2000,
    dca_bootstrap_iterations=1000,
    random_seed=20260926,
    calibration_bins=5,
)

print('Repository:', REPO_ROOT)
print('Final TL database:', RESULTS_DIR / 'r1_final_tl_runs.sqlite')
print('CV selection output:', CV_OUTPUT_DIR)
print('Secure analysis output:', ANALYSIS_OUTPUT_DIR)
print('Patient-level aggregation:', SETTINGS.aggregation_rule)
"""),
    markdown("""## 0A. Read and validate the locked final experiment

This fails if the database is not exactly 11 architectures × 100 completed seeds with 46 test predictions and three phase records per run.
"""),
    code("""results = load_r1_results(RESULTS_DIR)
print(f'Validated {len(results.run_metrics)} completed runs.')
print(f'Architectures: {results.run_metrics.model_name.nunique()}')
print(f'Test patients: {results.test_predictions.patient_id.nunique()}')
"""),
    markdown("""## 0B. Create the Version 1 analysis package

This is the only cell that writes files. It opens the SQLite database read-only and writes tables/figures to `ANALYSIS_OUTPUT_DIR`. It never changes the database.
"""),
    code("""analysis_tables = run_v1_analysis(RESULTS_DIR, ANALYSIS_OUTPUT_DIR, SETTINGS)
print('Created secure table/figure package at:', ANALYSIS_OUTPUT_DIR)
print('Manifest:', ANALYSIS_OUTPUT_DIR / 'analysis_manifest.json')
"""),
    markdown("""## 0C. Locked protocol, split, preprocessing, and development-only tuning

The protocol/split table documents the final run. The CV output below documents how the common protocol was selected using the 92-patient development cohort—not the 46-patient test cohort.
"""),
    code("""display(analysis_tables['integrity'])
display(analysis_tables['preprocessing'])

selected_protocol = CV_OUTPUT_DIR / 'selected_protocol.json'
cv_summary = CV_OUTPUT_DIR / 'summaries' / 'common_protocol_summary.csv'
if selected_protocol.exists() and cv_summary.exists():
    print('Development-only selected protocol:')
    display(pd.DataFrame([json.loads(selected_protocol.read_text())]))
    print('All 27 common protocols, ranked by mean pooled out-of-fold AUC across the 11 architectures:')
    display(pd.read_csv(cv_summary).sort_values('mean_pooled_oof_auc', ascending=False))
else:
    print('CV files were not found at the configured secure path. Update CV_OUTPUT_DIR above if needed.')
"""),
    markdown("""# 1. Training and computational audit

Total/trainable parameters, actual early-stopping duration, and observed runtime by architecture.
"""),
    code("""display(analysis_tables['phase_audit'])
display(analysis_tables['runtime'])
display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'runtime_by_model.png')))
display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'phase_epochs_by_model.png')))
"""),
    markdown("""# 2. 100-run descriptive stability

These results reproduce the useful median (IQR) reporting across the 100 seeds. They are descriptive stability results only: no run-level p-values and no best-seed selection.
"""),
    code("""display(analysis_tables['seed_stability_main'])
display(analysis_tables['seed_confusion_main'])
display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'seed_stability_boxplots.png')))
"""),
    markdown("""# 3. Sensitivity–specificity and error trade-off

The figures show the operating trade-off rather than presenting sensitivity as automatic overall superiority.
"""),
    code("""display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'sensitivity_specificity_tradeoff.png')))
display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'false_positive_negative_tradeoff.png')))
"""),
    markdown("""# 4. Locked patient-level TL results

For each model and each of the 46 test patients, the final probability is the arithmetic mean of 100 prespecified seed probabilities. This creates exactly one output per patient/model for patient-level confidence intervals and later paired comparisons.
"""),
    code("""display(analysis_tables['patient_metrics_ci'])
display(analysis_tables['patient_confusion'])
print('Secure aggregated patient-level predictions were written to:')
print(ANALYSIS_OUTPUT_DIR / 'tables' / 'patient_aggregated_predictions.csv')
"""),
    markdown("""# 5. ROC and calibration

ROC, AUC confidence intervals, calibration figure, Brier score, calibration intercept/slope, and expected calibration error are calculated from the locked patient-level probabilities.
"""),
    code("""display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'patient_level_roc.png')))
display(analysis_tables['calibration'])
display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'patient_level_calibration.png')))
"""),
    markdown("""# 6. Decision-curve analysis

DCA is calculated from the locked patient-level probabilities. The overview compares models with treat-all/treat-none. The multi-panel figure shows stratified-bootstrap 95% uncertainty bands for each model.
"""),
    code("""display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'patient_level_dca_overview.png')))
display(Image(filename=str(ANALYSIS_OUTPUT_DIR / 'figures' / 'patient_level_dca_uncertainty.png')))
"""),
    markdown("""# Version 1 deliverables

```text
secure_analysis/R1_TL_V1/
├── analysis_manifest.json
├── tables/      # manuscript/supplementary CSV tables; includes secure patient-level aggregate file
└── figures/     # nine PNG figures
```

**Not included here:** MBF/clinical baselines, reference-CNN/reader comparison, or ensemble analysis. Those require their respective locked inputs and will be handled in a separate follow-on notebook.
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
print(f"Wrote {OUTPUT}")
