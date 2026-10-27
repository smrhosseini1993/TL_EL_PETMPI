"""Fit conventional clinical and myocardial blood-flow baseline comparators.

The script uses the fixed 61/31/46 development and test partition. It expects a
local clinical CSV in that row order and writes a local results workbook. Study
inputs and generated workbooks are excluded from version control.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ID_COLUMN = "ID"
OUTCOME_COLUMN = "Outcome (Ischemic or not)"
SEGMENT_COLUMNS = [f"STR {index}" for index in range(1, 18)]
CLINICAL_COLUMNS = [
    "Sex (M=1. F=2)",
    "Age",
    "BMI",
    "Family history of CAD",
    "Angina",
    "Type 1 diabetes",
    "Smoking",
    "Dyslipidemia",
    "Hypertension",
]
RISK_COLUMNS = [
    "Family history of CAD",
    "Angina",
    "Type 1 diabetes",
    "Smoking",
    "Dyslipidemia",
    "Hypertension",
]
FIXED_SPLITS = OrderedDict(
    [
        ("train", (0, 61)),
        ("validation", (61, 92)),
        ("test", (92, 138)),
    ]
)
REGULARISATION_GRID = (0.01, 0.1, 1.0, 10.0)
GLOBAL_MBF_THRESHOLD = 2.3
AHA_TERRITORIES = OrderedDict(
    [
        ("LAD", (1, 2, 7, 8, 13, 14, 17)),
        ("LCX", (5, 6, 11, 12, 16)),
        ("RCA", (3, 4, 9, 10, 15)),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit conventional clinical and MBF baseline comparators on a fixed 61/31/46 split."
    )
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        required=True,
        help="Local CSV containing the required clinical and segmental MBF columns.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("results_baselines.xlsx"),
        help="Local output workbook (default: results_baselines.xlsx).",
    )
    parser.add_argument(
        "--id-column",
        default=ID_COLUMN,
        help=f"Unique identifier column (default: {ID_COLUMN!r}).",
    )
    parser.add_argument(
        "--outcome-column",
        default=OUTCOME_COLUMN,
        help="Binary 0/1 CAD-label column.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output workbook.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and the fixed split without fitting models or writing output.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_columns(id_column: str, outcome_column: str) -> list[str]:
    return [id_column, outcome_column, *SEGMENT_COLUMNS, *CLINICAL_COLUMNS]


def read_clinical_data(path: Path, id_column: str, outcome_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Clinical CSV does not exist: {path}")
    data = pd.read_csv(path)
    missing = [column for column in required_columns(id_column, outcome_column) if column not in data.columns]
    if missing:
        raise ValueError(f"Clinical CSV is missing required columns: {', '.join(missing)}")
    if len(data) != 138:
        raise ValueError(f"Expected exactly 138 rows in fixed study order; found {len(data)}.")
    if data[id_column].isna().any() or data[id_column].duplicated().any():
        raise ValueError("Identifier column must contain 138 unique, non-missing values.")
    labels = pd.to_numeric(data[outcome_column], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Outcome column must contain binary 0/1 labels.")
    result = data.copy()
    result[outcome_column] = labels.astype(int)
    return result


def split_frame(data: pd.DataFrame) -> OrderedDict[str, pd.DataFrame]:
    return OrderedDict(
        (name, data.iloc[start:stop].copy().reset_index(drop=True))
        for name, (start, stop) in FIXED_SPLITS.items()
    )


def clean_segments(data: pd.DataFrame) -> pd.DataFrame:
    segments = data[SEGMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return segments.mask(segments <= 0.0)


def global_mbf(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"global_stress_mbf": clean_segments(data).mean(axis=1, skipna=True)})


def territorial_mbf(data: pd.DataFrame) -> pd.DataFrame:
    segments = clean_segments(data)
    values: dict[str, pd.Series] = {}
    for territory, indices in AHA_TERRITORIES.items():
        columns = [f"STR {index}" for index in indices]
        values[f"{territory}_stress_mbf"] = segments[columns].mean(axis=1, skipna=True)
    return pd.DataFrame(values)


def segmental_mbf(data: pd.DataFrame) -> pd.DataFrame:
    return clean_segments(data).copy()


def clinical_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    sex = pd.to_numeric(data["Sex (M=1. F=2)"], errors="coerce")
    if not sex.dropna().isin([1, 2]).all():
        raise ValueError("Sex must be coded 1 (male) or 2 (female), with blank values represented as missing.")
    features["female"] = sex.map({1: 0.0, 2: 1.0})
    features["age"] = pd.to_numeric(data["Age"], errors="coerce")
    features["bmi"] = pd.to_numeric(data["BMI"], errors="coerce")
    for column in RISK_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        if not values.dropna().isin([0, 1, 2]).all():
            raise ValueError(f"{column} must use documented 0/1/2 coding.")
        name = column.lower().replace(" ", "_").replace("(", "").replace(")", "")
        features[f"{name}_abnormal"] = values.map({1: 0.0, 2: 1.0, 0: np.nan})
    return features


def combined_clinical_segmental(data: pd.DataFrame) -> pd.DataFrame:
    clinical = clinical_features(data).reset_index(drop=True)
    segments = segmental_mbf(data).reset_index(drop=True)
    return pd.concat([clinical, segments], axis=1)


def build_feature_sets(data: pd.DataFrame) -> OrderedDict[str, pd.DataFrame]:
    return OrderedDict(
        [
            ("global_mbf_logistic", global_mbf(data)),
            ("territorial_mbf_logistic", territorial_mbf(data)),
            ("segmental17_mbf_logistic", segmental_mbf(data)),
            ("clinical_logistic", clinical_features(data)),
            ("clinical_plus_segmental17_logistic", combined_clinical_segmental(data)),
        ]
    )


def make_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=float(c_value),
                    solver="liblinear",
                    max_iter=10000,
                    random_state=0,
                    class_weight=None,
                ),
            ),
        ]
    )


def safe_auc(labels: pd.Series | np.ndarray, scores: np.ndarray) -> float:
    labels_array = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    if len(np.unique(labels_array)) != 2 or np.isnan(score_array).any():
        return float("nan")
    return float(roc_auc_score(labels_array, score_array))


def calculate_metrics(
    labels: pd.Series | np.ndarray,
    scores: np.ndarray,
    binary_predictions: np.ndarray,
    probability_available: bool,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    y_pred = np.asarray(binary_predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if probability_available:
        probabilities = np.clip(y_score, 1e-15, 1 - 1e-15)
        metrics["brier_score"] = float(np.mean((probabilities - y_true) ** 2))
    else:
        metrics["brier_score"] = float("nan")
    return metrics


def select_regularisation(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    for c_value in REGULARISATION_GRID:
        pipeline = make_pipeline(c_value)
        pipeline.fit(train_x, train_y)
        validation_probability = pipeline.predict_proba(validation_x)[:, 1]
        candidates.append((float(c_value), safe_auc(validation_y, validation_probability)))
    finite = [(c_value, auc) for c_value, auc in candidates if not np.isnan(auc)]
    if not finite:
        raise RuntimeError("Validation AUC could not be calculated for any prespecified regularisation value.")
    best_auc = max(auc for _, auc in finite)
    selected_c = min(c_value for c_value, auc in finite if auc == best_auc)
    return selected_c, best_auc


def logistic_baseline(
    name: str,
    features: dict[str, pd.DataFrame],
    splits: OrderedDict[str, pd.DataFrame],
    outcome_column: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_x = features["train"]
    validation_x = features["validation"]
    train_y = splits["train"][outcome_column]
    validation_y = splits["validation"][outcome_column]
    selected_c, validation_selection_auc = select_regularisation(train_x, train_y, validation_x, validation_y)
    pipeline = make_pipeline(selected_c)
    pipeline.fit(train_x, train_y)
    summary: dict[str, Any] = {
        "baseline_id": name,
        "baseline_type": "regularised_logistic_regression",
        "input_features": ";".join(train_x.columns.tolist()),
        "n_raw_features": int(train_x.shape[1]),
        "selection_data": "61-patient training fit; 31-patient validation AUC selection; test excluded",
        "regularisation_grid": ";".join(str(value) for value in REGULARISATION_GRID),
        "selected_C": float(selected_c),
        "validation_selection_auc": float(validation_selection_auc),
        "prediction_scale": "probability",
        "classification_threshold": 0.5,
    }
    prediction_rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        frame = splits[split_name]
        probability = pipeline.predict_proba(features[split_name])[:, 1]
        binary = (probability >= 0.5).astype(int)
        metrics = calculate_metrics(frame[outcome_column], probability, binary, probability_available=True)
        for key, value in metrics.items():
            summary[f"{split_name}_{key}"] = value
        for index, (score, prediction) in enumerate(zip(probability, binary), start=1):
            prediction_rows.append(
                {
                    "baseline_id": name,
                    "split": split_name,
                    "within_split_order": index,
                    "observed_label": int(frame.iloc[index - 1][outcome_column]),
                    "score": float(score),
                    "probability": float(score),
                    "binary_prediction": int(prediction),
                    "score_definition": "logistic_regression_probability",
                }
            )
    selection = pd.DataFrame(
        [{"baseline_id": name, "selected_C": selected_c, "validation_selection_auc": validation_selection_auc}]
    )
    return summary, pd.DataFrame(prediction_rows), selection


def fixed_global_threshold_baseline(
    splits: OrderedDict[str, pd.DataFrame],
    outcome_column: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    global_features = {name: global_mbf(frame) for name, frame in splits.items()}
    train_values = global_features["train"]["global_stress_mbf"]
    if train_values.isna().all():
        raise RuntimeError("Cannot impute global MBF: every training value is missing.")
    train_median = float(train_values.median())
    summary: dict[str, Any] = {
        "baseline_id": "global_mbf_lt_2_3_rule",
        "baseline_type": "prespecified_threshold_rule",
        "input_features": "global_stress_mbf_mean_of_STR_1_to_STR_17",
        "n_raw_features": 1,
        "selection_data": "No fitted parameter; prespecified global stress-MBF threshold <2.3 mL/g/min",
        "regularisation_grid": "not_applicable",
        "selected_C": float("nan"),
        "validation_selection_auc": float("nan"),
        "prediction_scale": "continuous_negative_global_mbf_score; not a calibrated probability",
        "classification_threshold": GLOBAL_MBF_THRESHOLD,
        "development_median_imputation_for_missing_global_mbf": train_median,
    }
    prediction_rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        frame = splits[split_name]
        raw_global = global_features[split_name]["global_stress_mbf"]
        imputed_global = raw_global.fillna(train_median).to_numpy(dtype=float)
        score = -imputed_global
        binary = (imputed_global < GLOBAL_MBF_THRESHOLD).astype(int)
        metrics = calculate_metrics(frame[outcome_column], score, binary, probability_available=False)
        for key, value in metrics.items():
            summary[f"{split_name}_{key}"] = value
        summary[f"{split_name}_missing_global_mbf_count"] = int(raw_global.isna().sum())
        for index, (raw_value, imputed_value, score_value, prediction) in enumerate(
            zip(raw_global, imputed_global, score, binary), start=1
        ):
            prediction_rows.append(
                {
                    "baseline_id": "global_mbf_lt_2_3_rule",
                    "split": split_name,
                    "within_split_order": index,
                    "observed_label": int(frame.iloc[index - 1][outcome_column]),
                    "score": float(score_value),
                    "probability": float("nan"),
                    "binary_prediction": int(prediction),
                    "score_definition": "negative_global_mbf_for_auc; fixed_rule_uses_global_mbf_lt_2_3",
                    "raw_global_mbf": float(raw_value) if not pd.isna(raw_value) else float("nan"),
                    "global_mbf_imputed": bool(pd.isna(raw_value)),
                }
            )
    selection = pd.DataFrame(
        [{"baseline_id": "global_mbf_lt_2_3_rule", "selected_C": np.nan, "validation_selection_auc": np.nan}]
    )
    return summary, pd.DataFrame(prediction_rows), selection


def patient_manifest(
    splits: OrderedDict[str, pd.DataFrame], id_column: str, outcome_column: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, frame in splits.items():
        for index, (_, row) in enumerate(frame.iterrows(), start=1):
            rows.append(
                {
                    "split": split_name,
                    "within_split_order": index,
                    "clinical_id": int(row[id_column]),
                    "observed_label": int(row[outcome_column]),
                }
            )
    return pd.DataFrame(rows)


def data_quality_rows(
    data: pd.DataFrame, splits: OrderedDict[str, pd.DataFrame], outcome_column: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, frame in splits.items():
        segment_values = clean_segments(frame)
        rows.append(
            {
                "split": split_name,
                "n_patients": len(frame),
                "label_0": int((frame[outcome_column] == 0).sum()),
                "label_1": int((frame[outcome_column] == 1).sum()),
                "patients_with_any_missing_segmental_mbf": int(segment_values.isna().any(axis=1).sum()),
                "patients_with_all_segmental_mbf_missing": int(segment_values.isna().all(axis=1).sum()),
                "patients_with_missing_BMI": int(pd.to_numeric(frame["BMI"], errors="coerce").isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def study_lock_rows(clinical_csv_hash: str) -> pd.DataFrame:
    territory_mapping = "; ".join(
        f"{name}: {','.join(str(value) for value in indices)}" for name, indices in AHA_TERRITORIES.items()
    )
    rows = [
        ("split", "Fixed row-order split: first 61=train, next 31=validation, final 46=test"),
        ("clinical_csv_sha256", clinical_csv_hash),
        ("test_data_used_for_selection", "false"),
        ("baseline_A", "Fixed global stress-MBF <2.3 mL/g/min rule; global mean uses all 17 segments"),
        ("baseline_B", "L2-regularised logistic regression: continuous global stress MBF"),
        ("baseline_C", "L2-regularised logistic regression: AHA standard LAD/LCX/RCA territorial stress-MBF means"),
        ("baseline_D", "L2-regularised logistic regression: all 17 stress-MBF segment values"),
        ("baseline_E", "L2-regularised logistic regression: sex, age, BMI, family history, angina, type-1 diabetes, smoking, dyslipidaemia, hypertension"),
        ("baseline_F", "L2-regularised logistic regression: clinical variables plus all 17 stress-MBF segment values"),
        ("regularisation", "C grid 0.01, 0.1, 1.0, 10.0; select highest validation AUC; exact ties select smaller C"),
        ("imputation", "Fitted baselines use training-only median imputation with missingness indicators. The fixed rule uses training-median global-MBF substitution only when global MBF is unavailable."),
        ("scaling", "Fitted baselines use StandardScaler fit on 61 training patients only."),
        ("clinical_coding", "Sex: 1=male, 2=female. Risk variables: 1=normal, 2=abnormal, 0=missing."),
        ("territory_mapping", territory_mapping),
        ("territory_mapping_limitation", "Population-standard AHA territory assignment; does not claim patient-specific coronary anatomy."),
        ("excluded_inputs", "CTA stenosis variables and coronary calcium score are not included in the primary baselines."),
        ("patient_level_output", "One score and classification per baseline per patient; no repeated training runs."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def write_workbook(
    output_file: Path,
    summaries: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    selection: pd.DataFrame,
    quality: pd.DataFrame,
    lock: pd.DataFrame,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
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
    clinical_csv = args.clinical_csv.resolve()
    output_file = args.output_file.resolve()
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output workbook already exists. Use --overwrite to replace it: {output_file}")
    data = read_clinical_data(clinical_csv, args.id_column, args.outcome_column)
    splits = split_frame(data)
    quality = data_quality_rows(data, splits, args.outcome_column)
    print("Clinical CSV validated: 138 rows; fixed split=61/31/46.")
    print("Test data are excluded from regularisation selection.")
    if args.dry_run:
        print("Dry run complete: no model was fitted and no workbook was written.")
        return
    feature_sets = {name: build_feature_sets(frame) for name, frame in splits.items()}
    summaries: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    threshold_summary, threshold_predictions, threshold_selection = fixed_global_threshold_baseline(
        splits, args.outcome_column
    )
    summaries.append(threshold_summary)
    prediction_frames.append(threshold_predictions)
    selection_frames.append(threshold_selection)
    for baseline_id in (
        "global_mbf_logistic",
        "territorial_mbf_logistic",
        "segmental17_mbf_logistic",
        "clinical_logistic",
        "clinical_plus_segmental17_logistic",
    ):
        by_split = {name: feature_sets[name][baseline_id] for name in ("train", "validation", "test")}
        summary, predictions, selection = logistic_baseline(baseline_id, by_split, splits, args.outcome_column)
        summaries.append(summary)
        prediction_frames.append(predictions)
        selection_frames.append(selection)
    summaries_frame = pd.DataFrame(summaries).sort_values("baseline_id").reset_index(drop=True)
    predictions_frame = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values(["baseline_id", "split", "within_split_order"])
        .reset_index(drop=True)
    )
    selection_frame = pd.concat(selection_frames, ignore_index=True).sort_values("baseline_id").reset_index(drop=True)
    manifest = patient_manifest(splits, args.id_column, args.outcome_column)
    lock = study_lock_rows(sha256_file(clinical_csv))
    write_workbook(output_file, summaries_frame, predictions_frame, manifest, selection_frame, quality, lock)
    print(f"Completed six baseline comparators. Local workbook written: {output_file}")


if __name__ == "__main__":
    main()
