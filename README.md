# PET-MPI transfer learning and ensemble learning — revision workflow

This branch contains **only the active code used for the EJPH-D-26-00179 major revision**. Patient images, labels, clinical data, model weights, and analysis workbooks are intentionally excluded from Git and must remain on approved secure storage.

> **Removed from this branch:** the legacy `EL_ensemble.ipynb`, the SQLite/256×256 transfer-learning workflow, old legacy-format conversion notebooks, and red/green diagnostic scripts. They must not be used for the revised manuscript. Git history preserves prior development records without leaving competing runnable workflows in the active codebase.

## Active workflow

| Order | File | Purpose | Input | Output |
|---:|---|---|---|---|
| 1 | `Experiment2026/experiments2026.py` + `run2026.sh` | Final 128×128 fixed-split TL rerun: 11 architectures × 100 seeds. | Secure `data/training` and `data/test` | `metrics2026.xlsx` |
| 2 | `Experiment2026/ensemble_selection_cv_2026.py` + `run_ensemble_selection_cv_2026.sh` | Development-only five-fold CV for selecting EL constituents and rule. It never opens test data. | Secure 92-patient `data/training` | `ensemble_selection_cv_raw_results.xlsx` |
| 3 | `notebooks/R1_ensemble_configuration_selection.ipynb` | Displays 11-architecture and 10-configuration development rankings. You manually enter Top-3/Top-5 after inspecting the displayed ranking. | CV workbook from step 2 | No file written |
| 4 | `notebooks/R1_final_ensemble_analysis.ipynb` | Creates one 100-run, seed-matched raw ensemble workbook using one development-selected configuration; then displays descriptive median (IQR) metrics and confusion-matrix stability. | `secure_inputs/metrics2026.xlsx` | `secure_inputs/R1_final_ensemble_raw_runs.xlsx` |
| 5 | `Experiment2026/baselines2026.py` + `run_baselines2026.sh` | Six prespecified conventional clinical/quantitative-PET benchmarks. | Secure clinical CSV | `baseline_metrics2026.xlsx` |
| 6 | `Experiment2026/reference_cnn2026.py` + `run_reference_cnn2026.sh` | Seed-varied 100-run rerun of the retained four-convolution reference CNN. | Secure `data/training` and `data/test` | `CNN_metrics2026.xlsx` |
| 7 | `notebooks/results_publish.ipynb` | Final paper-order tables, figures, patient-level bootstrap CIs, paired DeLong tests, and McNemar tests. | Raw workbooks from steps 1, 4–6 | Separate Results/Supplementary package |

## Internal preprocessing-only attribution diagnostic

`Experiment2026/preprocessing_only_ablation.py` is a **separate internal diagnostic**, not an active manuscript-analysis workflow. It retains the archived legacy learning conditions—128×128 input, batch size 5, training-only red/green channel multiplication by 1.1, the frozen-backbone behavior, and no explicit random seed—while replacing only the universal legacy `/255` image scaling with the correct architecture-specific Keras `preprocess_input` function. Its output must be named `metrics_preprocessing_only.xlsx`, remain outside Git, and must never be used for manuscript results, EL selection, clinical comparison, or inferential statistics. `run_preprocessing_only_ablation.sh` defaults to five repeats per architecture as a preflight and can resume to 100 repeats using its `START_REPEAT` and `END_REPEAT` environment variables.

## Core analysis rules

- The historic fixed split is **61 training / 31 validation / 46 test patients**.
- The final TL experiment uses 128×128 RGB inputs, architecture-specific Keras preprocessing, Adam learning rate 0.0003, batch size 10, dropout 0.50, class weights, and **no augmentation**.
- The final TL and reference-CNN 100-seed runs describe training stability. They are **not** treated as 100 independent clinical samples.
- Paper-level ROC curves, decision curves, calibration, 95% CIs, paired DeLong tests, and McNemar tests use **one seed-mean prediction per test patient and method**.
- The ensemble pool/rule is selected with 92-patient development-only out-of-fold predictions before the 46-patient test predictions are read. The allowed rules are Sum, Median, Max, Majority Voting, and Weighted Sum. **Borda Count is not implemented.**
- No best run, test-set configuration selection, or run-level inferential p-value is permitted.

## Secure data layout

The TL, EL-CV, and reference-CNN runners expect this secure structure:

```text
dataparent/
└── data/
    ├── training/
    │   ├── *.jpg                 # 92 alphabetically ordered development images
    │   └── ica_lables.txt        # 92 ordered binary labels: 56 label-0, 36 label-1
    └── test/
        ├── *.jpg                 # 46 alphabetically ordered test images
        └── ica_lables.txt        # 46 ordered binary labels: 26 label-0, 20 label-1
```

## Mac analysis-folder layout

Use the following local structure after copying only the required secure workbooks:

```text
PETMPI_R1_final_analysis/
├── TL_EL_PETMPI/                         # this Git clone
└── secure_inputs/
    ├── metrics2026.xlsx                  # final 11 × 100 TL workbook
    ├── baseline_metrics2026.xlsx         # conventional baselines
    ├── CNN_metrics2026.xlsx              # seed-varied reference CNN, 100 runs
    └── R1_final_ensemble_raw_runs.xlsx   # created after Step 4
```

`results_publish.ipynb` reads those raw workbooks directly. Do not create, convert, or use a legacy `metrics.xlsx` for final analysis.

## Running the two current notebooks on the Mac

Open the clone folder in VS Code, choose a Python environment with the dependencies in `requirements.txt`, then open:

1. `notebooks/R1_final_ensemble_analysis.ipynb` only **after** development-only ensemble selection is complete. Its first code cell reads `../secure_inputs/metrics2026.xlsx` and writes `../secure_inputs/R1_final_ensemble_raw_runs.xlsx`.
2. `notebooks/results_publish.ipynb` only after all required raw workbooks are present. Keep `RUN_ANALYSIS = False` until its input-check cell reports the expected source files.

## Validation

The active structural tests are:

```bash
python scripts/check_python38_syntax.py
python scripts/test_r1_ensemble_selection.py
python scripts/test_reference_cnn2026.py
python scripts/test_results_publish.py
```

### Legacy Top-5 Max Rule parity check

`analysis/legacy_metrics2025_adapter.py` and the two command-line scripts below
exist **only** to verify that the revised shared ensemble implementation preserves
the historical Top-5 Max Rule applied to the submitted legacy `metrics.xlsx`.
They are not part of the final revision analysis or manuscript reporting workflow.

The source file has one sheet with 12 architectures × 100 rows. `EfficientNetB0`
is excluded because it was not part of the submitted manuscript's 11-architecture
comparison. Several old `tag` values are duplicated; the old notebook matched runs
by their row position within each architecture. The converter deliberately maps that
position to seeds 1–100 and retains the old `tag` only as provenance.

```bash
python scripts/convert_legacy_metrics2025.py \
  --legacy-workbook /secure/path/metrics.xlsx \
  --test-labels /secure/path/ica_lables.txt \
  --output-workbook /secure/path/metrics2025.xlsx

python scripts/verify_legacy_top5_max_rule.py \
  --legacy-workbook /secure/path/metrics.xlsx \
  --test-labels /secure/path/ica_lables.txt \
  --metrics2025 /secure/path/metrics2025.xlsx \
  --output-workbook /secure/path/legacy_top5_max_rule_reproduction.xlsx
```

The latter command requires exact equality for all 100 saved-probability vectors,
hard calls, metric rows, and confusion counts between the direct old row-position
calculation and the current shared-rule code. It does not claim to reconstruct
unavailable full-precision old predictions.

### Internal 2025-versus-2026 comparison normalization

For an internal like-for-like comparison only, create a separate derivative of the
full-precision final `metrics2026.xlsx`. The source is never replaced. The resulting
workbook places the same 72 `Run_results` core columns as `metrics2025.xlsx` first
and uses the same ordered 46-patient comparison manifest. By default it preserves
the actual full-precision 2026 test probabilities and source test metrics, so no
continuous-score information is lost in a meaningful old-versus-new comparison. The
original full patient manifest and study lock are retained as separate source sheets.

```bash
python scripts/normalize_metrics2026_for_2025_comparison.py \
  --source-workbook /secure/path/metrics2026.xlsx \
  --output-workbook /secure/path/metrics2026_comparable_to_metrics2025.xlsx
```

If checking the sensitivity of the old four-decimal representation specifically, add
`--probability-representation four_decimal`. That creates a separate legacy-style
copy and must not replace the full-precision comparison workbook, because it can
change AUC through probability ties.

Use this derivative only when comparing the old and new analyses through the same
legacy-style processing code. Use the original full-precision `metrics2026.xlsx` for
the final revised manuscript analyses.

## License

MIT License — see [LICENSE](LICENSE).
