#!/usr/bin/env python3
"""Locked conventional baseline comparison for EJPH-D-26-00179.

This runner creates one secure workbook containing the approved conventional
comparators for the historic 61/31/46 PET-MPI split.  It never writes raw
clinical data to Git and accepts the clinical CSV only through a local path.

Approved baselines
A. Fixed global stress-MBF <2.3 mL/g/min rule.
B. Logistic regression using continuous global stress MBF.
C. Logistic regression using standard-AHA LAD, LCX, and RCA territorial means.
D. L2-regularised logistic regression using all 17 segmental stress-MBF values.
E. L2-regularised logistic regression using approved clinical variables.
F. L2-regularised logistic regression using approved clinical + all-17 MBF inputs.

The first 61 patients train each logistic model.  The next 31 patients select
its prespecified L2 regularisation strength.  The independent 46-patient test
cohort is never used for any model or hyperparameter choice.
"""
from __future__ import print_function

import argparse
import hashlib
import json
from collections import OrderedDict
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FILE = SCRIPT_DIRECTORY / "baseline_metrics2026.xlsx"

OUTCOME_COLUMN = "Outcome (Ischemic or not)"
ID_COLUMN = "ID"
SEGMENT_COLUMNS = ["STR {0}".format(index) for index in range(1, 18)]
CLINICAL_COLUMNS = [
    "Sex (M=1. F=2)", "Age", "BMI", "Family history of CAD", "Angina",
    "Type 1 diabetes", "Smoking", "Dyslipidemia", "Hypertension",
]
RISK_COLUMNS = [
    "Family history of CAD", "Angina", "Type 1 diabetes", "Smoking",
    "Dyslipidemia", "Hypertension",
]
EXPECTED_SPLITS = OrderedDict([
    ("train", (0, 61, 36, 25)),
    ("validation", (61, 92, 20, 11)),
    ("test", (92, 138, 26, 20)),
])
REGULARISATION_GRID = (0.01, 0.1, 1.0, 10.0)
GLOBAL_THRESHOLD = 2.3

# Cerqueira et al., Circulation 2002;105:539-542. DOI: 10.1161/hc0402.102975.
# Standard population-level assignment; the manuscript will state its limitation
# that actual coronary supply can vary between patients.
AHA_TERRITORIES = OrderedDict([
    ("LAD", (1, 2, 7, 8, 13, 14, 17)),
    ("LCX", (5, 6, 11, 12, 16)),
    ("RCA", (3, 4, 9, 10, 15)),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked conventional PET/clinical baselines on the historic 61/31/46 split.")
    parser.add_argument("--clinical-csv", type=Path, required=True, help="Secure local path to the clinical CSV; never committed to Git.")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Secure workbook written locally (default: baseline_metrics2026.xlsx).")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing baseline output workbook.")
    parser.add_argument("--dry-run", action="store_true", help="Validate CSV, split, inputs and protocol without fitting models or writing a workbook.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_columns(data: pd.DataFrame) -> None:
    required = [ID_COLUMN, OUTCOME_COLUMN] + SEGMENT_COLUMNS + CLINICAL_COLUMNS
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Clinical CSV is missing required columns: {0}".format(", ".join(missing)))


def read_clinical_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Clinical CSV does not exist: {0}".format(path))
    data = pd.read_csv(str(path))
    validate_columns(data)
    if len(data) != 138:
        raise ValueError("Expected exactly 138 clinical rows in historic order; found {0}.".format(len(data)))
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError("Clinical ID column must contain 138 unique, non-missing values.")
    labels = pd.to_numeric(data[OUTCOME_COLUMN], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Outcome column must contain only binary 0/1 values.")
    data = data.copy()
    data[OUTCOME_COLUMN] = labels.astype(int)
    for split_name, (start, stop, expected_zero, expected_one) in EXPECTED_SPLITS.items():
        labels_here = data.iloc[start:stop][OUTCOME_COLUMN]
        observed = (len(labels_here), int((labels_here == 0).sum()), int((labels_here == 1).sum()))
        expected = (stop - start, expected_zero, expected_one)
        if observed != expected:
            raise ValueError("Historic {0} split mismatch; expected n/0/1={1}, observed={2}.".format(split_name, expected, observed))
    return data


def split_frame(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return OrderedDict((name, data.iloc[start:stop].copy().reset_index(drop=True))
                       for name, (start, stop, _, _) in EXPECTED_SPLITS.items())


def clean_segments(data: pd.DataFrame) -> pd.DataFrame:
    segments = data[SEGMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # The supplied clinical-data documentation states that missing PET variables
    # are represented as zero; zero cannot be a physiological stress-MBF value.
    return segments.mask(segments <= 0.0)


def global_mbf(data: pd.DataFrame) -> pd.DataFrame:
    values = clean_segments(data).mean(axis=1, skipna=True)
    return pd.DataFrame({"global_stress_mbf": values})


def territorial_mbf(data: pd.DataFrame) -> pd.DataFrame:
    segments = clean_segments(data)
    values: Dict[str, pd.Series] = {}
    for territory, indices in AHA_TERRITORIES.items():
        columns = ["STR {0}".format(index) for index in indices]
        values["{0}_stress_mbf".format(territory)] = segments[columns].mean(axis=1, skipna=True)
    return pd.DataFrame(values)


def segmental_mbf(data: pd.DataFrame) -> pd.DataFrame:
    return clean_segments(data).copy()


def clinical_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    sex = pd.to_numeric(data["Sex (M=1. F=2)"], errors="coerce")
    if not sex.dropna().isin([1, 2]).all():
        raise ValueError("Sex must be coded 1 (male) or 2 (female), with missing values as blank/NaN.")
    features["female"] = sex.map({1: 0.0, 2: 1.0})
    features["age"] = pd.to_numeric(data["Age"], errors="coerce")
    features["bmi"] = pd.to_numeric(data["BMI"], errors="coerce")
    for column in RISK_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        if not values.dropna().isin([0, 1, 2]).all():
            raise ValueError("{0} must use the documented 0/1/2 coding.".format(column))
        # Jarmo's supplied documentation: 1=normal, 2=abnormal, 0=missing.
        name = column.lower().replace(" ", "_").replace("(", "").replace(")", "")
        features["{0}_abnormal".format(name)] = values.map({1: 0.0, 2: 1.0, 0: np.nan})
    return features


def combined_clinical_segmental(data: pd.DataFrame) -> pd.DataFrame:
    clinical = clinical_features(data).reset_index(drop=True)
    segments = segmental_mbf(data).reset_index(drop=True)
    return pd.concat([clinical, segments], axis=1)


def build_feature_sets(data: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    return OrderedDict([
        ("global_mbf_logistic", global_mbf(data)),
        ("territorial_mbf_logistic", territorial_mbf(data)),
        ("segmental17_mbf_logistic", segmental_mbf(data)),
        ("clinical_logistic", clinical_features(data)),
        ("clinical_plus_segmental17_logistic", combined_clinical_segmental(data)),
    ])


def make_pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2", C=float(c_value), solver="liblinear", max_iter=10000,
            random_state=0, class_weight=None,
        )),
    ])


def safe_auc(labels: Sequence[int], score: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=int)
    score_array = np.asarray(score, dtype=float)
    if len(np.unique(labels_array)) != 2 or np.isnan(score_array).any():
        return float("nan")
    return float(roc_auc_score(labels_array, score_array))


def calculate_metrics(labels: Sequence[int], scores: Sequence[float], binary_predictions: Sequence[int], probability_available: bool) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    y_pred = np.asarray(binary_predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if probability_available:
        probabilities = np.clip(y_score, 1e-15, 1 - 1e-15)
        metrics["brier_score"] = float(np.mean((probabilities - y_true) ** 2))
    else:
        metrics["brier_score"] = float("nan")
    return metrics


def select_regularisation(train_x: pd.DataFrame, train_y: pd.Series, validation_x: pd.DataFrame, validation_y: pd.Series) -> Tuple[float, float]:
    candidates: List[Tuple[float, float]] = []
    for c_value in REGULARISATION_GRID:
        pipeline = make_pipeline(c_value)
        pipeline.fit(train_x, train_y)
        validation_probability = pipeline.predict_proba(validation_x)[:, 1]
        candidates.append((float(c_value), safe_auc(validation_y, validation_probability)))
    finite = [(c_value, auc) for c_value, auc in candidates if not np.isnan(auc)]
    if not finite:
        raise RuntimeError("Validation AUC could not be calculated for any prespecified regularisation value.")
    # Exact best validation AUC; ties select the smaller C (stronger regularisation).
    best_auc = max(auc for _, auc in finite)
    selected_c = min(c_value for c_value, auc in finite if auc == best_auc)
    return selected_c, best_auc


def logistic_baseline(name: str, features: Mapping[str, pd.DataFrame], splits: Mapping[str, pd.DataFrame]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_x = features["train"]
    validation_x = features["validation"]
    test_x = features["test"]
    train_y = splits["train"][OUTCOME_COLUMN]
    validation_y = splits["validation"][OUTCOME_COLUMN]
    test_y = splits["test"][OUTCOME_COLUMN]
    selected_c, validation_selection_auc = select_regularisation(train_x, train_y, validation_x, validation_y)
    pipeline = make_pipeline(selected_c)
    pipeline.fit(train_x, train_y)

    summary: Dict[str, Any] = {
        "baseline_id": name,
        "baseline_type": "regularised_logistic_regression",
        "input_features": ";".join(train_x.columns.tolist()),
        "n_raw_features": int(train_x.shape[1]),
        "selection_data": "61-patient training fit; 31-patient validation AUC selection; test excluded",
        "regularisation_grid": ";".join(str(value) for value in REGULARISATION_GRID),
        "selected_C": float(selected_c),
        "validation_selection_auc": float(validation_selection_auc),
        "prediction_scale": "probability",
        "classification_threshold": 0.50,
    }
    prediction_rows: List[Dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        frame = splits[split_name]
        probability = pipeline.predict_proba(features[split_name])[:, 1]
        binary = (probability >= 0.50).astype(int)
        metrics = calculate_metrics(frame[OUTCOME_COLUMN], probability, binary, probability_available=True)
        for key, value in metrics.items():
            summary["{0}_{1}".format(split_name, key)] = value
        for index, (score, prediction) in enumerate(zip(probability, binary), start=1):
            prediction_rows.append({
                "baseline_id": name, "split": split_name, "within_split_order": index,
                "observed_label": int(frame.iloc[index - 1][OUTCOME_COLUMN]),
                "score": float(score), "probability": float(score), "binary_prediction": int(prediction),
                "score_definition": "logistic_regression_probability",
            })
    validation_detail = pd.DataFrame([{
        "baseline_id": name, "selected_C": selected_c,
        "validation_selection_auc": validation_selection_auc,
    }])
    return summary, pd.DataFrame(prediction_rows), validation_detail


def fixed_global_threshold_baseline(splits: Mapping[str, pd.DataFrame]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    global_features = {name: global_mbf(frame) for name, frame in splits.items()}
    train_values = global_features["train"]["global_stress_mbf"]
    if train_values.isna().all():
        raise RuntimeError("Cannot impute global MBF: every training value is missing.")
    train_median = float(train_values.median())
    summary: Dict[str, Any] = {
        "baseline_id": "global_mbf_lt_2_3_rule",
        "baseline_type": "prespecified_threshold_rule",
        "input_features": "global_stress_mbf_mean_of_STR_1_to_STR_17",
        "n_raw_features": 1,
        "selection_data": "No fitted parameter; prespecified global stress-MBF threshold <2.3 mL/g/min",
        "regularisation_grid": "not_applicable",
        "selected_C": float("nan"),
        "validation_selection_auc": float("nan"),
        "prediction_scale": "continuous_negative_global_mbf_score; not a calibrated probability",
        "classification_threshold": GLOBAL_THRESHOLD,
        "development_median_imputation_for_missing_global_mbf": train_median,
    }
    prediction_rows: List[Dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        frame = splits[split_name]
        raw_global = global_features[split_name]["global_stress_mbf"]
        imputed_global = raw_global.fillna(train_median).to_numpy(dtype=float)
        # Lower MBF indicates higher likelihood of disease; negate for an AUC-oriented score.
        score = -imputed_global
        binary = (imputed_global < GLOBAL_THRESHOLD).astype(int)
        metrics = calculate_metrics(frame[OUTCOME_COLUMN], score, binary, probability_available=False)
        for key, value in metrics.items():
            summary["{0}_{1}".format(split_name, key)] = value
        summary["{0}_missing_global_mbf_count".format(split_name)] = int(raw_global.isna().sum())
        for index, (raw_value, value, score_value, prediction) in enumerate(zip(raw_global, imputed_global, score, binary), start=1):
            prediction_rows.append({
                "baseline_id": "global_mbf_lt_2_3_rule", "split": split_name, "within_split_order": index,
                "observed_label": int(frame.iloc[index - 1][OUTCOME_COLUMN]),
                "score": float(score_value), "probability": float("nan"), "binary_prediction": int(prediction),
                "score_definition": "negative_global_mbf_for_auc; fixed_rule_uses_global_mbf_lt_2_3",
                "raw_global_mbf": float(raw_value) if not pd.isna(raw_value) else float("nan"),
                "global_mbf_imputed": bool(pd.isna(raw_value)),
            })
    validation_detail = pd.DataFrame([{
        "baseline_id": "global_mbf_lt_2_3_rule", "selected_C": np.nan,
        "validation_selection_auc": np.nan,
    }])
    return summary, pd.DataFrame(prediction_rows), validation_detail


def patient_manifest(splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        frame = splits[split_name]
        for index, row in frame.iterrows():
            rows.append({
                "split": split_name,
                "within_split_order": int(index + 1),
                "clinical_id": int(row[ID_COLUMN]),
                "observed_label": int(row[OUTCOME_COLUMN]),
            })
    return pd.DataFrame(rows)


def data_quality_rows(data: pd.DataFrame, splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, frame in splits.items():
        segment_values = clean_segments(frame)
        rows.append({
            "split": split_name,
            "n_patients": len(frame),
            "label_0": int((frame[OUTCOME_COLUMN] == 0).sum()),
            "label_1": int((frame[OUTCOME_COLUMN] == 1).sum()),
            "patients_with_any_missing_segmental_mbf": int(segment_values.isna().any(axis=1).sum()),
            "patients_with_all_segmental_mbf_missing": int(segment_values.isna().all(axis=1).sum()),
            "patients_with_missing_BMI": int(pd.to_numeric(frame["BMI"], errors="coerce").isna().sum()),
        })
    return pd.DataFrame(rows)


def study_lock_rows(clinical_csv_hash: str) -> pd.DataFrame:
    mapping = "; ".join("{0}: {1}".format(key, ",".join(str(value) for value in values)) for key, values in AHA_TERRITORIES.items())
    rows = [
        ("study", "EJPH-D-26-00179 Comment 2 conventional baseline analysis"),
        ("split", "Historic row-order split: first 61=train, next 31=validation, final 46=test"),
        ("split_class_counts", "train: 36 label-0 / 25 label-1; validation: 20 / 11; test: 26 / 20"),
        ("clinical_csv_sha256", clinical_csv_hash),
        ("test_data_used_for_selection", "false"),
        ("baseline_A", "Fixed global stress-MBF <2.3 mL/g/min rule; global mean uses all 17 segments"),
        ("baseline_B", "L2-regularised logistic regression: continuous global stress MBF"),
        ("baseline_C", "L2-regularised logistic regression: AHA standard LAD/LCX/RCA territorial stress-MBF means"),
        ("baseline_D", "L2-regularised logistic regression: all 17 stress-MBF segment values"),
        ("baseline_E", "L2-regularised logistic regression: sex, age, BMI, family history, angina, type-1 diabetes, smoking, dyslipidaemia, hypertension"),
        ("baseline_F", "L2-regularised logistic regression: approved clinical variables plus all 17 stress-MBF segment values"),
        ("regularisation", "Prespecified C grid 0.01, 0.1, 1.0, 10.0; select highest validation AUC; exact ties select smaller C"),
        ("imputation", "All fitted baselines: median imputation learned from 61 training patients only, with missingness indicators. Fixed threshold rule: training-median global-MBF substitution only when global MBF is unavailable; count recorded."),
        ("scaling", "All fitted baselines: StandardScaler fit on 61 training patients only."),
        ("clinical_coding", "Sex: 1=male, 2=female. Risk variables: 1=normal, 2=abnormal, 0=missing; missing values handled by training-only imputation."),
        ("territory_mapping", "Cerqueira et al., Circulation 2002, DOI:10.1161/hc0402.102975. " + mapping),
        ("territory_mapping_limitation", "Population-standard AHA territory assignment; does not claim patient-specific coronary anatomy."),
        ("excluded_inputs", "CTA stenosis variables and coronary calcium score excluded from primary baselines."),
        ("patient_level_output", "One locked score/prediction per baseline per patient; no repeated seeds."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def write_workbook(output_file: Path, summaries: pd.DataFrame, predictions: pd.DataFrame, manifest: pd.DataFrame, selection: pd.DataFrame, quality: pd.DataFrame, lock: pd.DataFrame) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(output_file), engine="openpyxl", mode="w") as writer:
        summaries.to_excel(writer, sheet_name="Baseline_results", index=False)
        predictions.to_excel(writer, sheet_name="Patient_predictions", index=False)
        manifest.to_excel(writer, sheet_name="Patient_manifest", index=False)
        selection.to_excel(writer, sheet_name="Validation_selection", index=False)
        quality.to_excel(writer, sheet_name="Data_quality", index=False)
        lock.to_excel(writer, sheet_name="Study_lock", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                cell.font = font


def main() -> None:
    args = parse_args()
    args.clinical_csv = args.clinical_csv.resolve()
    args.output_file = args.output_file.resolve()
    if args.output_file.exists() and not args.overwrite:
        raise FileExistsError("Output workbook already exists. Use --overwrite only after deliberately reviewing it: {0}".format(args.output_file))

    data = read_clinical_data(args.clinical_csv)
    splits = split_frame(data)
    quality = data_quality_rows(data, splits)
    print("Clinical CSV validated: rows=138; historic train/validation/test counts=61/31/46.")
    print("All approved clinical and STR 1-17 inputs are present.")
    print("Missing all-segmental-MBF patients by split: train={0}, validation={1}, test={2}.".format(
        int(quality.loc[quality["split"] == "train", "patients_with_all_segmental_mbf_missing"].iloc[0]),
        int(quality.loc[quality["split"] == "validation", "patients_with_all_segmental_mbf_missing"].iloc[0]),
        int(quality.loc[quality["split"] == "test", "patients_with_all_segmental_mbf_missing"].iloc[0]),
    ))
    print("Test data used for model/regularisation selection: false")
    if args.dry_run:
        print("Dry run complete: no model was fitted and no workbook was written.")
        return

    feature_sets = {name: build_feature_sets(frame) for name, frame in splits.items()}
    summaries: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    selection_frames: List[pd.DataFrame] = []

    threshold_summary, threshold_predictions, threshold_selection = fixed_global_threshold_baseline(splits)
    summaries.append(threshold_summary)
    prediction_frames.append(threshold_predictions)
    selection_frames.append(threshold_selection)

    for baseline_id in (
        "global_mbf_logistic", "territorial_mbf_logistic", "segmental17_mbf_logistic",
        "clinical_logistic", "clinical_plus_segmental17_logistic",
    ):
        by_split = {split_name: feature_sets[split_name][baseline_id] for split_name in ("train", "validation", "test")}
        summary, predictions, selection = logistic_baseline(baseline_id, by_split, splits)
        summaries.append(summary)
        prediction_frames.append(predictions)
        selection_frames.append(selection)

    summaries_frame = pd.DataFrame(summaries).sort_values("baseline_id").reset_index(drop=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True).sort_values(["baseline_id", "split", "within_split_order"]).reset_index(drop=True)
    selection_frame = pd.concat(selection_frames, ignore_index=True).sort_values("baseline_id").reset_index(drop=True)
    manifest = patient_manifest(splits)
    lock = study_lock_rows(sha256_file(args.clinical_csv))
    write_workbook(args.output_file, summaries_frame, predictions_frame, manifest, selection_frame, quality, lock)
    print("Completed approved baselines A-F. Secure workbook written: {0}".format(args.output_file))


if __name__ == "__main__":
    main()
