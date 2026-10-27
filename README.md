# Transfer Learning and Ensemble Learning for PET Polar-Map Classification

Code accompanying the study of transfer learning (TL) and ensemble learning (EL) for coronary artery disease classification from $^{15}$O-H$_2$O PET perfusion polar maps.

## Contents

- `TL_fixedvalidation.py`: fixed train/validation-split TL experiments.
- `TL_crossvalidation.py`: cross-validation TL experiments used to define development-ranked model pools.
- `run_TL_fixedvalidation.sh` and `run_TL_crossvalidation.sh`: example shell launchers that record the seed for every run.
- `baselines.py`: six clinical and myocardial-blood-flow (MBF) comparator methods on the fixed 61/31/46 split.
- `notebooks/ensemble_analysis.ipynb`: run-matched Top-3 and Top-5 EL analyses using Sum, Median, Max, and Majority Voting rules.
- `notebooks/evaluate_results.ipynb`: descriptive repeated-run summaries, patient-level evaluation, confidence intervals, paired comparisons, and calibration analyses.
- `notebooks/visualization.ipynb`: patient-level ROC curves and repeated-run stability figures.
- `data/README.md`: required local data and result-file structure.

## Data availability

The study images and participant-level data are not included because they contain patient information. The code uses a local `data/` directory; see [`data/README.md`](data/README.md) for the expected structure. All local data, result workbooks, predictions, figures, logs, and model files are excluded from version control.

## Environment

The study implementation used Python 3.10 and TensorFlow 2.7. Install the project dependencies in an isolated environment compatible with the TensorFlow build available for your platform:

```bash
pip install -r requirements.txt
```

## Running the transfer-learning experiments

Place local images and binary labels in the structure described in `data/README.md`. The training scripts determine the project root from their own location. The fixed-split launcher performs 100 independent runs for each of the 11 architectures. The CV launcher performs 10 repeated 5-fold CV runs for each architecture. Then run:

```bash
bash run_TL_fixedvalidation.sh
# or
bash run_TL_crossvalidation.sh
```

The scripts use a shared 128 × 128 RGB representation scaled to the 0–1 range. No image augmentation is applied. Each launcher passes a distinct integer seed, which is saved in the resulting run metadata.

## Clinical and MBF comparator methods

`baselines.py` creates the six prespecified comparator outputs used by the evaluation notebook: a fixed global-MBF threshold rule and five regularised logistic-regression models based on global MBF, territorial MBF, 17-segment MBF, clinical variables, or clinical variables plus 17-segment MBF. It accepts a local clinical/MBF CSV in the fixed 61/31/46 study order and writes a local workbook:

```bash
python baselines.py \
  --clinical-csv data/clinical_baselines.csv \
  --output-file data/results_baselines.xlsx
```

Regularisation is selected using the validation subset only. The test subset is excluded from comparator fitting and selection. The fixed global-MBF rule classifies with global stress MBF <2.3 mL/g/min; its negative global-MBF value is retained only as a continuous ranking score for AUC, not as a calibrated probability.

## Ensemble, evaluation, and visualization notebooks

Start Jupyter from the project root:

```bash
jupyter notebook notebooks/ensemble_analysis.ipynb
# or
jupyter notebook notebooks/evaluate_results.ipynb
# or
jupyter notebook notebooks/visualization.ipynb
```

The ensemble notebook expects a TL workbook and an ordered test-label file in the local `data/` directory. The evaluation notebook accepts local de-identified repeated-run and baseline workbooks, writes aggregate tables to `analysis_outputs/`, and exports local patient-order prediction scores needed by the visualization notebook. These local outputs are excluded from version control. To create the ROC figures, run the evaluation notebook first and then run the ROC cells in `notebooks/visualization.ipynb`; its default `REPORTING_DIR` is `analysis_outputs/`. The ROC figures use one mean predicted probability per test patient across the 100 runs; they do not average ROC curves across runs.

## Reference CNN

The custom Reference CNN is the implementation reported by Teuho et al. for the same cohort. Its source code is separately available through the University of Turku GitLab portal: <https://gitlab.utu.fi/jarmo.teuho>.

## Citation

If you use this code, please cite the associated article.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
