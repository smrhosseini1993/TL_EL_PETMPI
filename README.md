# Transfer Learning and Ensemble Learning for PET-MPI Polar Map Classification

Code for the revision-stage transfer-learning (TL) reanalysis of PET polar-map classification. Patient images, labels, split manifests, model weights, and run outputs must remain on approved secure storage and are not included in this repository.

## R1 transfer-learning workflow

The current `r1/tl-reanalysis` branch implements the revised TL workflow and the development-only ensemble-configuration selection workflow. `EL_ensemble.ipynb` is legacy Version 1 code and **must not be used for the R1 ensemble analysis**. The valid R1 ensemble process ranks candidate configurations using development-only out-of-fold predictions before any selected configuration is evaluated with the historic 61/31/46 test predictions.

| File | Role |
|---|---|
| `TL_crossvalidation.py` | Development-only five-fold stratified CV. It loads only the 92-patient development cohort and evaluates the predeclared 27 common protocols. It writes patient-level out-of-fold probabilities and `selected_protocol.json` only after the complete 27 × 5 × 11 run. |
| `TL_fixedvalidation.py` | Locked historic 61/31/46 runner. It accepts only the CV-selected protocol, writes each completed model/seed result atomically to one secure SQLite database, and performs no model, protocol, threshold, or best-seed selection. |
| `final_results_store.py` | Compact, resumable SQLite storage layer for final run metrics, full-precision test predictions, phase parameters, and completion status. |
| `tl_reanalysis_core.py` | Shared architecture registry, official Keras preprocessing registry, retained legacy classifier head, three-phase fine-tuning functions, reproducibility utilities, and output schema. |
| `configs/r1_tl_protocol_grid.json` | Version-controlled declaration of the fixed protocol and 3 × 3 × 3 development-only grid. |
| `configs/fixed_split_manifest_TEMPLATE.csv` | Non-sensitive schema for the secure locked 61/31/46 split manifest. Do not place real IDs or labels in Git. |
| `scripts/create_verified_historic_manifest.py` | Creates one secure manifest from the audited legacy alphabetical ordering and refuses any mismatch from the historic 61/31/46 class counts. |
| `scripts/validate_tl_setup.py` | Fast no-GPU validator for secure data counts, labels, and (optionally) the locked split manifest. |
| `run_TL_crossvalidation.sh` | CV technical-preflight and full-search launcher template. |
| `run_TL_fixedvalidation.sh` | 5-seed technical-preflight and remaining 95-seed production launcher template. |
| `scripts/validate_final_results_store.py` | Performs structural coverage validation without printing or using performance metrics. |
| `scripts/export_final_results_workbook.py` | Validates the secure SQLite store and exports one readable final Excel workbook only after all expected runs are complete. |
| `Experiment2026/ensemble_selection_cv_2026.py` | Development-only 5-fold CV runner matching the final retained 128×128 Experiment2026 protocol. It creates one secure Excel workbook containing raw OOF predictions, fold metrics, phase parameters, and the study lock. It never opens test data. |
| `Experiment2026/run_ensemble_selection_cv_2026.sh` | Separate VGG16 technical-preflight and complete 11-architecture × 5-fold CV launcher. The preflight and complete runs always use different secure output directories. |
| `analysis/r1_ensemble_core.py` | Shared implementations of the five allowed combination rules, OOF architecture/configuration ranking, development-only AUC weights, lock-file creation, and structural validation. Borda Count is not implemented. |
| `notebooks/R1_ensemble_configuration_selection.ipynb` | Reads only the raw final-protocol CV workbook, displays the 11-model development ranking, accepts manually entered Top-3/Top-5 pools, then displays the 10 candidate configurations. It writes no output and never reads historic test predictions. |
| `notebooks/R1_final_ensemble_analysis.ipynb` | Reads the completed historic `metrics2026.xlsx` workbook and one manually entered development-selected pool/rule. It creates one raw 100-run ensemble workbook for later combined reporting. It never ranks alternative configurations on the test cohort. |
| `notebooks/convert_metrics2026_to_legacy.ipynb` | Creates a separate old-layout `metrics_legacy_compatible.xlsx` from the final 1,100-run `metrics2026.xlsx`, for legacy plotting-cell compatibility only. It never modifies the audited source workbook. |
| `Experiment2026/reference_cnn2026.py` | Retained four-convolution reference-CNN runner based on the supplied historical AUC/ACC implementations. It preserves their architecture, literal image path, NumPy/TensorFlow random states, 256×256 resize path, 35-epoch SGD protocol, class weights, and 92-patient `validation_split=1/3` fit behavior while writing a secure 100-repeat `CNN_metrics2026.xlsx` workbook with full-precision patient predictions. |
| `Experiment2026/run_reference_cnn2026.sh` | Resumable launcher for reference-CNN repeat identifiers 1–100. |
| `notebooks/results_publish.ipynb` | The single paper-order reporting notebook. It validates the final TL, selected raw ensemble, conventional baseline, and optional CNN/clinical-reader inputs; then writes Tables T1–T4 and ST1–ST4, Figures F1–F3 and SF1, paired patient-level statistics, captions, and a provenance manifest. |

### Locked R1 protocol

- **Images:** 256 × 256 RGB JPEG polar maps.
- **Preprocessing:** the official Keras `preprocess_input` function for the selected ImageNet backbone.
- **Architectures:** VGG16, VGG19, ResNet50, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, DenseNet169, DenseNet201, MobileNetV2, and Xception.
- **Retained classifier-head topology:** `Flatten → Dense(1024) → Dropout(d) → Dense(512) → Dropout(d) → Dense(256) → Dropout(d) → sigmoid`.
- **Fixed settings:** Adam; unweighted binary cross-entropy; batch size 5; no augmentation; 0.50 individual-model threshold; three maximum 30-epoch phases; early stopping on validation binary accuracy with patience 4 and best-weight restoration.
- **Development-only grid:** head learning rate `{3e-5, 1e-4, 3e-4}` × fine-tuning learning rate `{1e-6, 3e-6, 1e-5}` × dropout `{0.30, 0.50, 0.60}`.
- **Protocol selection:** highest mean pooled out-of-fold AUC across the eleven architectures. The 46-patient test cohort is not loaded by the CV runner.
- **Final runs:** 100 prespecified seeds per architecture. They are descriptive stability runs, not independent clinical samples and not a source of best-seed or run-level inferential tests.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and a TensorFlow/Keras environment compatible with the hospital GPU; the revision workflow was validated in the existing `polarmaps2024` environment with TensorFlow 2.7.1.

## Secure data setup

The CV runner expects the secure development data to be organized as:

```
data/
├── training/
│   ├── *.jpg                    # Training images
│   └── ica_lables.txt           # Labels (one per line: 0 or 1)
```

The locked final runner does **not** infer the 61/31/46 split by filename order. It requires a secure CSV with these columns:

```text
patient_id,relative_path,split,observed_label
```

Its exact shape is shown in `configs/fixed_split_manifest_TEMPLATE.csv`. Keep the actual manifest out of Git. Use `scripts/validate_tl_setup.py` to validate the secure data and manifest before GPU fitting.

After confirming the old split rule and boundary filenames, create the secure manifest outside the repository, for example in a `secure_config` directory beside the clone:

```bash
python scripts/create_verified_historic_manifest.py \
  --root "${PROJECT_ROOT}" \
  --output /secure/path/secure_config/locked_historic_61_31_46.csv
```

## R1 usage

All commands below are executed on the approved GPU environment after cloning this branch. Set variables to secure paths; do not edit the repository with patient identifiers or outputs.

```bash
export PROJECT_ROOT=/secure/path/project
export OUTPUT_ROOT=/secure/path/r1_outputs
export SPLIT_MANIFEST=/secure/path/locked_61_31_46.csv
export CV_OUTPUT_ROOT="${OUTPUT_ROOT}/cv_full_27x5x11"
export RESULTS_DIR="${OUTPUT_ROOT}/R1_final_TL"

# Validate inputs before GPU work.
python scripts/validate_tl_setup.py \
  --root "${PROJECT_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}"
```

### Stage 1: CV technical preflight and complete development-only search

```bash
bash run_TL_crossvalidation.sh
```

The launcher runs every architecture with one central protocol across five folds first (55 technical fits). Inspect the resulting manifests, patient-level out-of-fold prediction files, phase parameter counts, and GPU logs. Then uncomment the complete 1,485-fit command in the launcher and run it. A complete CV run writes `selected_protocol.json`.

### Stage 2: locked fixed-split technical preflight and production runs

The launcher is controlled by `RUN_STAGE`, keeping the two batches separate while writing into the same secure SQLite results file:

```bash
# Stage 2a: technical acceptance batch; exactly seeds 1-5.
RUN_STAGE=preflight bash run_TL_fixedvalidation.sh

# Stage 2b: only after accepting the technical preflight; exactly seeds 6-100.
RUN_STAGE=production bash run_TL_fixedvalidation.sh
```

The first batch contains 55 fits and is part of the final 100-seed record only if it passes technically without changing code, the split, or locked settings. The second batch contains the remaining 1,045 fits. Never use interim test performance to decide whether to alter settings.

Validate the preflight structurally—without using AUC or other performance results—before the production batch:

```bash
python scripts/validate_final_results_store.py \
  --results-dir "${RESULTS_DIR}" \
  --seeds 1-5
```

### R1 ensemble configuration selection: development data only

The retained 128×128 `Experiment2026` benchmark is the historic 61/31/46 analysis used for the final TL and EL comparison. Its ensemble configuration is selected separately with a five-fold CV that opens only the 92 development patients. It must use a new secure output directory and must never read `data/test`.

```bash
# Technical preflight: VGG16 × 5 folds only. Do not use its output for selection.
RUN_STAGE=preflight bash Experiment2026/run_ensemble_selection_cv_2026.sh

# Complete selection run: 11 architectures × 5 folds = 55 fits.
RUN_STAGE=full bash Experiment2026/run_ensemble_selection_cv_2026.sh
```

The complete CV run writes one secure `ensemble_selection_cv_raw_results.xlsx` workbook containing only raw OOF predictions, fold metrics, phase parameters, and the study lock. Open `notebooks/R1_ensemble_configuration_selection.ipynb` only after this workbook exists. Its early cells display the 11-architecture ranking. Enter the Top-3 and Top-5 pools manually in its later pool cell, then use the displayed 10 configuration rows—Sum, Median, Max, Majority Voting, and Weighted Sum across both pools—to choose the rule. The notebook writes no output. Copy the manually entered pools and selected rule into the final ensemble notebook; no configuration can be selected by the 46-patient test cohort.

For the historic-split final ensemble result, open `notebooks/R1_final_ensemble_analysis.ipynb`. Copy the manually entered Top-3/Top-5 pools and one development-selected rule into its configuration cell, then set `RUN_ANALYSIS=True`. The notebook reads `Experiment2026/metrics2026.xlsx`, validates all 11 architectures × 100 runs, and writes one simple `R1_final_ensemble_raw_runs.xlsx` workbook containing the 100 run-matched ensemble runs, ordered probabilities, test manifest, and study lock. It does not rank or evaluate any alternative configuration on the test set.

### Legacy plotting compatibility

`notebooks/convert_metrics2026_to_legacy.ipynb` can create a separate `metrics_legacy_compatible.xlsx` workbook from the final `metrics2026.xlsx`. Its `sheet1` has the legacy metrics column names and four-decimal prediction text so old plotting cells can be adapted quickly. The source workbook is read-only and remains the auditable master dataset. Do not use the compatibility output to revive any legacy best-run, test-set-selection, Borda-count, or run-level inferential workflow.

The same notebook accepts `SOURCE_TYPE = 'reference_cnn'` to create a separate `CNN_metrics2026_legacy_compatible.xlsx` from the 100-seed reference-CNN master workbook. The source CNN workbook remains read-only; use the master workbook for patient-level reporting.

### Retained reference-CNN rerun

The original supplied AUC and ACC scripts have the same four-convolution architecture and identical training protocol. They differ only in the Keras metric displayed while fitting; neither metric controls early stopping or selection. `Experiment2026/reference_cnn2026.py` preserves the common source protocol, including its effective NumPy=1 and TensorFlow=2 random-state calls for every repeat, while recording both post-fit AUC and accuracy in `CNN_metrics2026.xlsx`. The workbook's `seed` field is therefore a repeat identifier (1–100), not a replacement model-initialisation seed.

Run the structural preflight first:

```bash
python Experiment2026/reference_cnn2026.py \
  --root /secure/path/dataparent \
  --output-file /secure/path/CNN_metrics2026.xlsx \
  --seed 1 \
  --dry-run
```

Then run all prespecified seeds in tmux:

```bash
ROOT_DIR=/secure/path/dataparent \
OUTPUT_FILE=/secure/path/CNN_metrics2026.xlsx \
bash Experiment2026/run_reference_cnn2026.sh
```

The result has the same three-sheet master-workbook design as `metrics2026.xlsx`: `Run_results`, `Patient_manifest`, and `Study_lock`. It contains one retained reference-CNN architecture × 100 prespecified seeds, full-precision ordered probabilities, and all train/validation/test metrics. It is directly accepted by `notebooks/results_publish.ipynb` as `CNN_metrics2026.xlsx`.

### Paper-order reporting package

After the development-only ensemble configuration has been selected and `R1_final_ensemble_raw_runs.xlsx` has been created, open `notebooks/results_publish.ipynb` on the secure analysis computer. Its one configuration cell accepts the final 1,100-run `metrics2026.xlsx`, the six-comparator `baseline_metrics2026.xlsx`, the raw selected-ensemble workbook, and optional reference-CNN and clinical-reader files. It reproduces the submitted figures' visual language while applying the revision-stage analysis boundary: 100 seeds are descriptive stability only; all CIs, ROC/DCA/calibration curves, paired DeLong comparisons, and McNemar comparisons are calculated from one seed-mean prediction per patient. It writes a separate `results_publish` package and never modifies source workbooks.

## R1 outputs

The CV runner produces its existing secure CSV/JSON outputs. The final locked runner keeps its complete record in exactly one secure durable database, `r1_final_tl_runs.sqlite`, stored in `RESULTS_DIR`. It records one full-precision test probability per patient/model/seed, the 1,100 run-metric rows, phase parameters, and completion status. It also writes exactly two batch manifests and the two console logs. It does **not** create thousands of per-seed prediction files.

After seeds 1-100 are complete, validate and export the one human-facing workbook:

```bash
python scripts/export_final_results_workbook.py \
  --results-dir "${RESULTS_DIR}"
```

This creates `R1_TL_final_results.xlsx` only when every expected model/seed run is structurally complete. Do not manually reorder, round, overwrite, or merge database records.

## License

MIT License - see LICENSE file.

## Author

**Seyed M. Hosseini**  
Turku PET Centre, University of Turku  
smhoss@utu.fi
or smrh.1372@gmail.com

For questions, open an issue on GitHub or contact via email.
