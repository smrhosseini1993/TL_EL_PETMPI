#!/usr/bin/env python3
"""Create the version-controlled paper-order results_publish notebook."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "notebooks" / "results_publish.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown("""# Paper results package — EJPH-D-26-00179

This is the final **paper-order reporting notebook**. It preserves the visual language of the previous plotting notebooks—wide figures, bold labels, outside legends, the fixed metric order `ACC / PRE / SEN / SPE / F1S / AUC`, and the ensemble/CNN/clinical colour conventions—while using the reviewer-corrected statistical approach.

> **Statistical boundary:** the 100 random-seed runs are used only to describe training stability. Clinical performance, ROC curves, calibration, decision curves, 95% confidence intervals, paired DeLong tests, and McNemar tests are calculated from **one seed-mean prediction per test patient and method**. This notebook contains no Borda Count, best run, test-set ensemble selection, or run-level inferential p-values.

The notebook writes a paper-ready package after all required inputs have been copied to the local Mac analysis folder. It does **not** modify any input workbook.
"""),
    markdown("""## 0. Configure the six inputs and output folder

Recommended local Mac structure:

```text
PETMPI_R1_final_analysis/
├── code/                         # clone of the GitHub branch
└── secure_inputs/
    ├── metrics2026.xlsx          # final 11 × 100 TL output
    ├── baseline_metrics2026.xlsx # six conventional baselines
    ├── R1_final_ensemble_raw_runs.xlsx
    ├── reference_cnn_runs.xlsx   # old 100 CNN runs, one CNN method only
    └── clinical_reader.xlsx      # optional; 46 rows if available
```

`clinical_reader.xlsx` is optional. If it contains only binary classifications, the notebook reports its ACC/F1S/SEN/SPE and confusion matrix, but intentionally leaves AUC and DCA unavailable.
"""),
    code("""from pathlib import Path
import sys
from IPython.display import display, Image, Markdown

# Run this notebook from the repository root or its notebooks/ folder.
CWD = Path.cwd().resolve()
if (CWD / 'analysis').exists():
    REPO_ROOT = CWD
elif (CWD.parent / 'analysis').exists():
    REPO_ROOT = CWD.parent
else:
    raise RuntimeError('Open this notebook from the repository root or notebooks/.')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.results_publish_core import ReportingSettings, load_all_sources, run_results_publish

PROJECT_ROOT = REPO_ROOT.parent
INPUT_DIR = PROJECT_ROOT / 'secure_inputs'
OUTPUT_DIR = PROJECT_ROOT / 'secure_outputs' / 'results_publish'

# Required audited inputs.
TL_WORKBOOK = INPUT_DIR / 'metrics2026.xlsx'
BASELINE_WORKBOOK = INPUT_DIR / 'baseline_metrics2026.xlsx'

# Required before producing the final ensemble/final-comparison package.
ENSEMBLE_WORKBOOK = INPUT_DIR / 'R1_final_ensemble_raw_runs.xlsx'

# Add the filenames after copying these inputs. Set to None only while unavailable.
REFERENCE_CNN_WORKBOOK = INPUT_DIR / 'reference_cnn_runs.xlsx'
CLINICAL_READER_FILE = None  # e.g., INPUT_DIR / 'clinical_reader.xlsx'

# These two prespecified conventional quantitative PET models appear in Table T3/T4 and F3.
PRINCIPAL_BASELINES = (
    'segmental17_mbf_logistic',
    'clinical_plus_segmental17_logistic',
)

# Optional figure SF2. Keep false unless the final calibration plot is readable with n=46.
MAKE_CALIBRATION_FIGURE = False

SETTINGS = ReportingSettings(
    threshold=0.50,
    bootstrap_iterations=2000,
    random_seed=20260929,
    calibration_bins=5,
)

# This is the sole write switch. Keep False until all local paths are correct.
RUN_ANALYSIS = False

print('Repository:', REPO_ROOT)
print('Input folder:', INPUT_DIR)
print('Output package:', OUTPUT_DIR)
print('TL workbook:', TL_WORKBOOK)
print('Baseline workbook:', BASELINE_WORKBOOK)
print('Ensemble workbook:', ENSEMBLE_WORKBOOK)
print('Reference CNN workbook:', REFERENCE_CNN_WORKBOOK)
print('Clinical reader file:', CLINICAL_READER_FILE)
"""),
    markdown("""## 0A. Check source-file availability

This cell does not read or modify any workbook. It simply verifies that the local paths in the configuration cell are correct.
"""),
    code("""input_paths = {
    'Final TL 1,100-run workbook': TL_WORKBOOK,
    'Conventional-baseline workbook': BASELINE_WORKBOOK,
    'Final raw 100-run ensemble workbook': ENSEMBLE_WORKBOOK,
    'Reference CNN workbook': REFERENCE_CNN_WORKBOOK,
    'Optional clinical-reader file': CLINICAL_READER_FILE,
}
for label, path in input_paths.items():
    if path is None:
        print(f'{label}: not configured')
    else:
        print(f'{label}: {"FOUND" if path.exists() else "MISSING"} — {path}')
"""),
    markdown("""## 0B. Generate the complete paper-order package

This is the only cell that writes outputs. It validates all schemas, confirms the 46 test patients and their labels align across sources, creates patient-level seed-aggregated predictions, then writes all tables and figures in the agreed Results/Supplementary order.

If the reference CNN or clinical reader are not yet available, set their path to `None`. The notebook will still create the TL and conventional-comparator outputs, but Table T3/T4 and Figure F3 will not include the missing comparator.
"""),
    code("""if RUN_ANALYSIS:
    sources = load_all_sources(
        tl_workbook=TL_WORKBOOK,
        baseline_workbook=BASELINE_WORKBOOK,
        ensemble_workbook=ENSEMBLE_WORKBOOK,
        reference_cnn_workbook=REFERENCE_CNN_WORKBOOK if REFERENCE_CNN_WORKBOOK is not None else None,
        clinical_reader_file=CLINICAL_READER_FILE,
        settings=SETTINGS,
    )
    tables = run_results_publish(
        sources=sources,
        output_dir=OUTPUT_DIR,
        settings=SETTINGS,
        principal_baselines=PRINCIPAL_BASELINES,
        make_calibration_figure=MAKE_CALIBRATION_FIGURE,
    )
    print('Created complete Results/Supplementary package:', OUTPUT_DIR)
    print('Tables workbook:', OUTPUT_DIR / 'results_publish_tables.xlsx')
else:
    print('No files were read or written. Set RUN_ANALYSIS = True after confirming the input paths.')
"""),
    markdown("""# Results — Transfer learning

## N2 / T1. Performance of all pre-trained models

> Table T1 presents a comparison of all pre-trained models using ACC, AUC, F1S, SEN, and SPE. Values were calculated from seed-aggregated patient-level predictions in the independent test cohort and are reported with 95% bootstrap confidence intervals.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['T1_TL_performance'])
"""),
    markdown("""## N3 / T2. Confusion-matrix results for all pre-trained models

> Table T2 presents TP, TN, FP, and FN for all pre-trained models. Each count was calculated once per test patient from the seed-aggregated predictions.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['T2_TL_confusion'])
"""),
    markdown("""## N4 / F1. Receiver operating characteristic curves

> Figure F1 presents receiver operating characteristic curves for all pre-trained models, derived from seed-aggregated patient-level predictions in the independent test cohort.
"""),
    code("""if RUN_ANALYSIS:
    display(Image(filename=str(OUTPUT_DIR / 'figures' / 'F1_TL_ROC.png')))
"""),
    markdown("""## N5 / F2. Decision-curve analysis for all pre-trained models

> Figure F2 illustrates decision-curve analysis through net-benefit curves for all pre-trained models. Each curve represents the net benefit of using the respective model for classifying significant CAD, compared with treat-all and treat-none strategies.
"""),
    code("""if RUN_ANALYSIS:
    display(Image(filename=str(OUTPUT_DIR / 'figures' / 'F2_TL_DCA.png')))
"""),
    markdown("""## N6 / ST1. Conventional quantitative-PET and clinical benchmarks

> Six prespecified conventional comparators were evaluated in the independent test cohort. Table T3 uses the two full 17-segment models because they are the most direct conventional counterparts to the regional PET information represented in the polar maps. The complete six-comparator results are retained in Supplementary Table ST1.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['ST1_all_conventional'])
"""),
    markdown("""# Results — Ensemble learning

## N7. Development-selected ensemble configuration

This records the already selected development-only configuration. The raw ensemble workbook's `Study_lock` sheet remains the durable source for its Top-3/Top-5 pool, rule, constituent models, any development-only weights, and seed matching.
"""),
    code("""if RUN_ANALYSIS:
    configuration = dict(sources.ensemble_metadata)
    display(Markdown(
        '**Final ensemble:** {pool} {rule}<br>'
        '**Constituent models:** {models}'.format(
            pool=configuration.get('pool_name', 'not recorded'),
            rule=configuration.get('rule_id', 'not recorded'),
            models=configuration.get('constituent_models', 'not recorded'),
        )
    ))
"""),
    markdown("""## N8 / T3. Final comparison of principal approaches

> Table T3 presents the final ensemble, its constituent models, the reference CNN when supplied, and the two prespecified conventional models based on the complete 17-segment MBF pattern, with and without clinical variables. The clinical reader is included as a descriptive comparator only if its 46-patient input file is supplied.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['T3_principal_comparison'])
"""),
    markdown("""## N9 / T4. Final-comparison confusion matrices

> Table T4 shows TP, TN, FP, and FN for the principal comparison approaches in the independent test cohort.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['T4_principal_confusion'])
"""),
    markdown("""## N10. Sensitivity-specificity trade-off

The manuscript should interpret sensitivity and specificity together with the corresponding false-positive and false-negative classifications. The patient-level paired comparisons are provided in Supplementary Table ST3.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['ST3_paired_comparisons'])
"""),
    markdown("""## N11 / F3. Decision-curve analysis of final probabilistic approaches

> Figure F3 shows decision-curve analysis of the final ensemble, reference CNN when supplied, the first development-ranked constituent model, and the two prespecified full-segment logistic-regression comparators. Binary-only clinical-reader classifications are deliberately excluded from DCA.
"""),
    code("""if RUN_ANALYSIS:
    display(Image(filename=str(OUTPUT_DIR / 'figures' / 'F3_final_comparison_DCA.png')))
"""),
    markdown("""# Supplementary Material

## ST2 / SF1. 100-seed stability

These seed-level results reproduce the useful median (IQR) stability presentation from the previous paper, but they are explicitly descriptive. They are not used for patient-level hypothesis tests.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['ST2_TL_seed_stability'])
    display(Image(filename=str(OUTPUT_DIR / 'figures' / 'SF1_TL_seed_stability.png')))
"""),
    markdown("""## ST3. Patient-level paired comparisons

The table includes the final ensemble against the reference CNN when supplied, each constituent model, all six conventional comparators, and the clinical reader when supplied. AUC comparisons use paired DeLong testing. Accuracy comparisons use McNemar testing.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['ST3_paired_comparisons'])
"""),
    markdown("""## ST4 / optional SF2. Calibration

Calibration measures are reported for methods that provide probabilistic predictions. The fixed MBF threshold rule and a binary-only clinical reader are not included in calibration analysis. Set `MAKE_CALIBRATION_FIGURE = True` only after checking that the optional figure is readable with the 46-patient test cohort.
"""),
    code("""if RUN_ANALYSIS:
    display(tables['ST4_calibration'])
    optional = OUTPUT_DIR / 'figures' / 'SF2_calibration.png'
    if optional.exists():
        display(Image(filename=str(optional)))
"""),
    markdown("""# Final files for manuscript writing

```text
secure_outputs/results_publish/
├── results_publish_tables.xlsx       # all T/ST tables; easy to copy into manuscript
├── captions_and_results_text.md      # N1–N11/captions in agreed wording
├── analysis_manifest.json            # reporting decisions and exclusions
├── tables/
│   ├── T1_TL_performance.csv
│   ├── T2_TL_confusion.csv
│   ├── T3_principal_comparison.csv
│   ├── T4_principal_confusion.csv
│   ├── ST1_all_conventional.csv
│   ├── ST2_TL_seed_stability.csv
│   ├── ST3_paired_comparisons.csv
│   └── ST4_calibration.csv
└── figures/
    ├── F1_TL_ROC.png
    ├── F2_TL_DCA.png
    ├── F3_final_comparison_DCA.png
    ├── SF1_TL_seed_stability.png
    └── SF2_calibration.png            # only when enabled
```

The `patient_level_predictions_secure.csv` and numeric comparison tables are supporting calculation records. Keep them secured; do not submit them as public supplementary material without checking the journal and local data-governance requirements.
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
