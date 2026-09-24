# Transfer Learning and Ensemble Learning for PET-MPI Polar Map Classification

Code for the revision-stage transfer-learning (TL) reanalysis of PET polar-map classification. Patient images, labels, split manifests, model weights, and run outputs must remain on approved secure storage and are not included in this repository.

## R1 transfer-learning workflow

The current `r1/tl-reanalysis` branch implements the TL portion of the revised experiment. The ensemble notebook is legacy Version 1 code and **must not be used for the R1 ensemble analysis**. A separate revision-stage ensemble workflow will be added only after the TL outputs have been verified.

| File | Role |
|---|---|
| `TL_crossvalidation.py` | Development-only five-fold stratified CV. It loads only the 92-patient development cohort and evaluates the predeclared 27 common protocols. It writes patient-level out-of-fold probabilities and `selected_protocol.json` only after the complete 27 × 5 × 11 run. |
| `TL_fixedvalidation.py` | Locked historic 61/31/46 runner. It accepts only the CV-selected protocol, writes each completed model/seed result atomically to one secure SQLite database, and performs no model, protocol, threshold, or best-seed selection. |
| `final_results_store.py` | Compact, resumable SQLite storage layer for final run metrics, full-precision test predictions, phase parameters, and completion status. |
| `tl_reanalysis_core.py` | Shared architecture registry, official Keras preprocessing registry, retained legacy classifier head, three-phase fine-tuning functions, reproducibility utilities, and output schema. |
| `configs/r1_tl_protocol_grid.json` | Version-controlled declaration of the fixed protocol and 3 × 3 × 3 development-only grid. |
| `configs/fixed_split_manifest_TEMPLATE.csv` | Non-sensitive schema for the secure locked 61/31/46 split manifest. Do not place real IDs or labels in Git. |
| `scripts/validate_tl_setup.py` | Fast no-GPU validator for secure data counts, labels, and (optionally) the locked split manifest. |
| `run_TL_crossvalidation.sh` | CV technical-preflight and full-search launcher template. |
| `run_TL_fixedvalidation.sh` | 5-seed technical-preflight and remaining 95-seed production launcher template. |
| `scripts/validate_final_results_store.py` | Performs structural coverage validation without printing or using performance metrics. |
| `scripts/export_final_results_workbook.py` | Validates the secure SQLite store and exports one readable final Excel workbook only after all expected runs are complete. |

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
