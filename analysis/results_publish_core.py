"""Patient-level paper reporting for the EJPH-D-26-00179 revision.

The module converts the locked 1,100-run TL workbook, one selected 100-run
ensemble workbook, conventional baseline predictions, and optional reference-CNN
and clinical-reader inputs into a reproducible Results/Supplementary package.

Statistical boundary
--------------------
* Seed-level 100-run results are descriptive stability results only.
* Clinical inference uses one seed-mean probability per test patient/model.
* Ensemble configuration selection is never performed here.
* No best run, Borda Count, or run-level p-value is implemented.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, norm
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

TL_MODELS: Tuple[str, ...] = (
    "VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "InceptionV3",
    "InceptionResNetV2", "DenseNet169", "DenseNet201", "MobileNetV2", "Xception",
)
BASELINE_ORDER: Tuple[str, ...] = (
    "global_mbf_lt_2_3_rule",
    "global_mbf_logistic",
    "territorial_mbf_logistic",
    "segmental17_mbf_logistic",
    "clinical_logistic",
    "clinical_plus_segmental17_logistic",
)
BASELINE_DISPLAY: Mapping[str, str] = {
    "global_mbf_lt_2_3_rule": "Global stress-MBF <2.3 mL/g/min rule",
    "global_mbf_logistic": "Global MBF logistic regression",
    "territorial_mbf_logistic": "Territorial MBF logistic regression",
    "segmental17_mbf_logistic": "17-segment MBF logistic regression",
    "clinical_logistic": "Clinical logistic regression",
    "clinical_plus_segmental17_logistic": "Clinical + 17-segment MBF logistic regression",
}
METRIC_ORDER: Tuple[str, ...] = ("accuracy", "auc", "f1", "sensitivity", "specificity", "precision")
METRIC_LABELS: Mapping[str, str] = {
    "accuracy": "ACC", "auc": "AUC", "f1": "F1S", "sensitivity": "SEN",
    "specificity": "SPE", "precision": "PRE",
}


@dataclass(frozen=True)
class ReportingSettings:
    """Predeclared reporting choices; these must not be tuned from test results."""

    threshold: float = 0.50
    bootstrap_iterations: int = 2000
    random_seed: int = 20260929
    calibration_bins: int = 5
    dca_threshold_start: float = 0.01
    dca_threshold_end: float = 0.99
    dca_threshold_step: float = 0.01


@dataclass
class SourceData:
    """Validated raw and patient-level data for the reporting notebook."""

    tl_runs: pd.DataFrame
    tl_patient_predictions: pd.DataFrame
    test_manifest: pd.DataFrame
    ensemble_runs: Optional[pd.DataFrame]
    ensemble_patient_predictions: Optional[pd.DataFrame]
    ensemble_metadata: Mapping[str, str]
    baseline_patient_predictions: pd.DataFrame
    baseline_metadata: pd.DataFrame
    reference_cnn_patient_predictions: Optional[pd.DataFrame]
    clinical_reader_patient_predictions: Optional[pd.DataFrame]


# ---------------------------------------------------------------------------
# Input handling and integrity checks
# ---------------------------------------------------------------------------

def _parse_probability_csv(value: Any, expected_length: int) -> np.ndarray:
    text = str(value).strip().strip("[](){}")
    if not text or text.lower() == "nan":
        raise ValueError("Probability text is empty")
    values = np.asarray([float(part.strip()) for part in text.split(",") if part.strip()], dtype=float)
    if len(values) != expected_length:
        raise ValueError("Expected {0} probabilities; found {1}".format(expected_length, len(values)))
    if not np.isfinite(values).all() or not np.logical_and(values >= 0.0, values <= 1.0).all():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return values


def _read_sheets(workbook: Path, required: Sequence[str]) -> Mapping[str, pd.DataFrame]:
    if not workbook.exists():
        raise FileNotFoundError("Input workbook not found: {0}".format(workbook))
    sheets = pd.read_excel(str(workbook), sheet_name=None, engine="openpyxl")
    missing = sorted(set(required) - set(sheets))
    if missing:
        raise ValueError("{0} lacks sheets: {1}".format(workbook.name, missing))
    return sheets


def _test_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "within_split_order", "observed_label"}
    if not required.issubset(manifest.columns):
        raise ValueError("Patient manifest lacks required columns: {0}".format(sorted(required - set(manifest.columns))))
    patient_column = "file_name" if "file_name" in manifest.columns else ("patient_id" if "patient_id" in manifest.columns else "clinical_id")
    if patient_column not in manifest.columns:
        raise ValueError("Patient manifest needs file_name or patient_id")
    result = manifest.loc[manifest["split"].astype(str) == "test"].copy()
    result["within_split_order"] = pd.to_numeric(result["within_split_order"], errors="raise").astype(int)
    result["observed_label"] = pd.to_numeric(result["observed_label"], errors="raise").astype(int)
    result = result.sort_values("within_split_order").reset_index(drop=True)
    if len(result) != 46 or result["within_split_order"].tolist() != list(range(1, 47)):
        raise ValueError("Test manifest must contain 46 ordered patients")
    if (int((result["observed_label"] == 0).sum()), int((result["observed_label"] == 1).sum())) != (26, 20):
        raise ValueError("Test manifest must contain 26 label-0 and 20 label-1 patients")
    if not np.isin(result["observed_label"], [0, 1]).all():
        raise ValueError("Test labels must be binary")
    result = result.rename(columns={patient_column: "patient_id"})
    result["patient_id"] = result["patient_id"].astype(str)
    return result[["patient_id", "within_split_order", "observed_label"]].copy()


def load_tl_metrics2026(workbook: Path, settings: ReportingSettings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and validate the final historic 11 × 100 Experiment2026 workbook."""
    sheets = _read_sheets(workbook, ("Run_results", "Patient_manifest", "Study_lock"))
    runs = sheets["Run_results"].copy()
    required = {
        "model_name", "seed", "test_probabilities", "test_accuracy", "test_precision",
        "test_sensitivity", "test_specificity", "test_f1", "test_auc", "test_tn",
        "test_fp", "test_fn", "test_tp",
    }
    if not required.issubset(runs.columns):
        raise ValueError("TL Run_results lacks required columns: {0}".format(sorted(required - set(runs.columns))))
    runs["model_name"] = runs["model_name"].astype(str)
    runs["seed"] = pd.to_numeric(runs["seed"], errors="raise").astype(int)
    if len(runs) != 1100 or set(runs["model_name"]) != set(TL_MODELS):
        raise ValueError("TL workbook must contain exactly 11 expected architectures × 100 seeds")
    if runs.duplicated(["model_name", "seed"]).any():
        raise ValueError("TL Run_results contains duplicate model/seed rows")
    for model_name in TL_MODELS:
        if set(runs.loc[runs["model_name"] == model_name, "seed"]) != set(range(1, 101)):
            raise ValueError("{0} does not contain exactly seeds 1-100".format(model_name))
    manifest = _test_manifest(sheets["Patient_manifest"])
    prediction_rows: List[Dict[str, Any]] = []
    for row in runs.itertuples(index=False):
        values = row._asdict()
        probabilities = _parse_probability_csv(values["test_probabilities"], 46)
        for manifest_row, probability in zip(manifest.itertuples(index=False), probabilities):
            prediction_rows.append({
                "method_id": str(values["model_name"]), "method_label": str(values["model_name"]),
                "method_type": "transfer_learning", "patient_id": manifest_row.patient_id,
                "within_split_order": int(manifest_row.within_split_order),
                "observed_label": int(manifest_row.observed_label), "seed": int(values["seed"]),
                "probability": float(probability), "score": float(probability),
                "binary_prediction": int(float(probability) >= settings.threshold),
                "probability_available": True,
            })
    long = pd.DataFrame(prediction_rows).sort_values(["method_id", "seed", "within_split_order"]).reset_index(drop=True)
    patient = aggregate_repeated_predictions(long, settings, expected_seeds=100)
    return runs.sort_values(["model_name", "seed"]).reset_index(drop=True), patient, manifest


def load_ensemble_raw_runs(workbook: Path, reference_manifest: pd.DataFrame, settings: ReportingSettings) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, str]]:
    """Read one manually configured, 100-run raw ensemble workbook."""
    sheets = _read_sheets(workbook, ("Run_results", "Patient_manifest", "Study_lock"))
    runs = sheets["Run_results"].copy()
    required = {"seed", "test_probabilities", "test_accuracy", "test_precision", "test_sensitivity", "test_specificity", "test_f1", "test_auc", "test_tn", "test_fp", "test_fn", "test_tp", "pool_name", "rule_id", "constituent_models"}
    if not required.issubset(runs.columns):
        raise ValueError("Ensemble Run_results lacks required columns: {0}".format(sorted(required - set(runs.columns))))
    runs["seed"] = pd.to_numeric(runs["seed"], errors="raise").astype(int)
    if len(runs) != 100 or set(runs["seed"]) != set(range(1, 101)) or runs.duplicated("seed").any():
        raise ValueError("Ensemble workbook must contain exactly one row for each seed 1-100")
    manifest = _test_manifest(sheets["Patient_manifest"])
    verify_matching_manifests(reference_manifest, manifest, "TL", "ensemble")
    if runs["pool_name"].nunique() != 1 or runs["rule_id"].nunique() != 1 or runs["constituent_models"].nunique() != 1:
        raise ValueError("Ensemble raw workbook has inconsistent pool, rule, or constituent metadata")
    prediction_rows: List[Dict[str, Any]] = []
    for row in runs.itertuples(index=False):
        values = row._asdict()
        probabilities = _parse_probability_csv(values["test_probabilities"], 46)
        for manifest_row, probability in zip(manifest.itertuples(index=False), probabilities):
            prediction_rows.append({
                "method_id": "Ensemble", "method_label": "Ensemble", "method_type": "ensemble",
                "patient_id": manifest_row.patient_id, "within_split_order": int(manifest_row.within_split_order),
                "observed_label": int(manifest_row.observed_label), "seed": int(values["seed"]),
                "probability": float(probability), "score": float(probability),
                "binary_prediction": int(float(probability) >= settings.threshold), "probability_available": True,
            })
    long = pd.DataFrame(prediction_rows).sort_values(["seed", "within_split_order"]).reset_index(drop=True)
    metadata = {
        "pool_name": str(runs["pool_name"].iloc[0]), "rule_id": str(runs["rule_id"].iloc[0]),
        "constituent_models": str(runs["constituent_models"].iloc[0]),
    }
    return runs.sort_values("seed").reset_index(drop=True), aggregate_repeated_predictions(long, settings, expected_seeds=100), metadata


def load_baseline_predictions(workbook: Path, reference_manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the locked conventional-baseline workbook and align its test patients."""
    sheets = _read_sheets(workbook, ("Baseline_results", "Patient_predictions", "Patient_manifest", "Study_lock"))
    raw = sheets["Patient_predictions"].copy()
    required = {"baseline_id", "split", "within_split_order", "observed_label", "score", "probability", "binary_prediction"}
    if not required.issubset(raw.columns):
        raise ValueError("Baseline Patient_predictions lacks required columns: {0}".format(sorted(required - set(raw.columns))))
    manifest = _test_manifest(sheets["Patient_manifest"])
    verify_matching_manifests(reference_manifest, manifest, "TL", "baseline")
    test = raw.loc[raw["split"].astype(str) == "test"].copy()
    test["baseline_id"] = test["baseline_id"].astype(str)
    test["within_split_order"] = pd.to_numeric(test["within_split_order"], errors="raise").astype(int)
    test["observed_label"] = pd.to_numeric(test["observed_label"], errors="raise").astype(int)
    test["score"] = pd.to_numeric(test["score"], errors="raise").astype(float)
    test["probability"] = pd.to_numeric(test["probability"], errors="coerce")
    test["binary_prediction"] = pd.to_numeric(test["binary_prediction"], errors="raise").astype(int)
    if set(test["baseline_id"]) != set(BASELINE_ORDER):
        raise ValueError("Baseline workbook does not contain exactly the six prespecified comparators")
    if test.duplicated(["baseline_id", "within_split_order"]).any() or not test.groupby("baseline_id").size().eq(46).all():
        raise ValueError("Each baseline must have exactly one test prediction for each of 46 ordered patients")
    expected_labels = reference_manifest.set_index("within_split_order")["observed_label"]
    for baseline_id, group in test.groupby("baseline_id"):
        group = group.sort_values("within_split_order")
        if group["within_split_order"].tolist() != list(range(1, 47)) or not np.array_equal(group["observed_label"].to_numpy(), expected_labels.to_numpy()):
            raise ValueError("Baseline {0} does not match the TL test manifest".format(baseline_id))
    result = test.merge(reference_manifest[["patient_id", "within_split_order"]], on="within_split_order", how="left", validate="many_to_one")
    result["method_id"] = result["baseline_id"]
    result["method_label"] = result["baseline_id"].map(BASELINE_DISPLAY)
    result["method_type"] = "conventional_baseline"
    result["seed"] = np.nan
    result["probability_available"] = result["probability"].between(0.0, 1.0, inclusive="both")
    result = result[["method_id", "method_label", "method_type", "patient_id", "within_split_order", "observed_label", "seed", "probability", "score", "binary_prediction", "probability_available"]]
    return result.sort_values(["method_id", "within_split_order"]).reset_index(drop=True), sheets["Baseline_results"].copy()


def load_reference_cnn_runs(workbook: Path, reference_manifest: pd.DataFrame, settings: ReportingSettings) -> pd.DataFrame:
    """Read a legacy-style 100-run reference-CNN workbook into patient-level predictions.

    Accepts the original one-sheet metrics layout or a workbook with a `Run_results`
    sheet. It needs an ordered 46-value probability column named one of `predicts`,
    `test_predicts`, or `test_probabilities` and optional `tag`/`seed` run IDs.
    Its binary predictions retain the source CNN's strict ``probability > 0.50``
    convention rather than the general ``>= 0.50`` reporting convention.
    """
    if not workbook.exists():
        raise FileNotFoundError("Reference CNN workbook not found: {0}".format(workbook))
    sheets = pd.read_excel(str(workbook), sheet_name=None, engine="openpyxl")
    if "Run_results" in sheets:
        runs = sheets["Run_results"].copy()
    elif "sheet1" in sheets:
        runs = sheets["sheet1"].copy()
    else:
        runs = next(iter(sheets.values())).copy()
    probability_column = next((name for name in ("test_probabilities", "test_predicts", "predicts") if name in runs.columns), None)
    if probability_column is None:
        raise ValueError("Reference CNN workbook needs test_probabilities, test_predicts, or predicts")
    if "model_name" in runs.columns:
        names = runs["model_name"].dropna().astype(str).unique().tolist()
        if len(names) > 1:
            raise ValueError("Reference CNN workbook must contain one CNN method only")
    seed_column = "seed" if "seed" in runs.columns else ("tag" if "tag" in runs.columns else None)
    rows: List[Dict[str, Any]] = []
    for default_seed, row in enumerate(runs.itertuples(index=False), start=1):
        values = row._asdict()
        seed = int(values[seed_column]) if seed_column is not None and not pd.isna(values[seed_column]) else default_seed
        probabilities = _parse_probability_csv(values[probability_column], 46)
        for manifest_row, probability in zip(reference_manifest.itertuples(index=False), probabilities):
            rows.append({
                "method_id": "Reference CNN", "method_label": "Reference CNN", "method_type": "reference_cnn",
                "patient_id": manifest_row.patient_id, "within_split_order": int(manifest_row.within_split_order),
                "observed_label": int(manifest_row.observed_label), "seed": seed, "probability": float(probability),
                "score": float(probability), "binary_prediction": int(float(probability) > settings.threshold),
                "probability_available": True,
            })
    long = pd.DataFrame(rows)
    if long.empty or long.duplicated(["seed", "within_split_order"]).any():
        raise ValueError("Reference CNN workbook contains invalid duplicated or empty run predictions")
    seed_count = int(long["seed"].nunique())
    if seed_count not in (1, 100) or not long.groupby("seed").size().eq(46).all():
        raise ValueError("Reference CNN workbook must contain either one or 100 complete 46-patient runs")
    return aggregate_repeated_predictions(long, settings, expected_seeds=seed_count)


def load_clinical_reader(path: Path, reference_manifest: pd.DataFrame) -> pd.DataFrame:
    """Read an optional one-row-per-test-patient clinical-reader CSV/XLSX file."""
    if not path.exists():
        raise FileNotFoundError("Clinical-reader file not found: {0}".format(path))
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(str(path))
    else:
        frame = pd.read_excel(str(path), engine="openpyxl")
    required = {"within_split_order", "observed_label", "binary_prediction"}
    if not required.issubset(frame.columns):
        raise ValueError("Clinical-reader input needs columns: within_split_order, observed_label, binary_prediction; probability is optional")
    result = frame.copy()
    result["within_split_order"] = pd.to_numeric(result["within_split_order"], errors="raise").astype(int)
    result["observed_label"] = pd.to_numeric(result["observed_label"], errors="raise").astype(int)
    result["binary_prediction"] = pd.to_numeric(result["binary_prediction"], errors="raise").astype(int)
    if len(result) != 46 or result.duplicated("within_split_order").any() or result["within_split_order"].tolist() != list(range(1, 47)):
        raise ValueError("Clinical-reader input must contain exactly one row for each ordered test patient 1-46")
    expected = reference_manifest.sort_values("within_split_order")
    if not np.array_equal(result["observed_label"].to_numpy(), expected["observed_label"].to_numpy()):
        raise ValueError("Clinical-reader labels do not match the TL test manifest")
    probability = pd.to_numeric(result["probability"], errors="coerce") if "probability" in result.columns else pd.Series(np.nan, index=result.index)
    probability_available = probability.between(0.0, 1.0, inclusive="both")
    if probability_available.any() and not probability_available.all():
        raise ValueError("Clinical-reader probability must be complete for all 46 patients or omitted entirely")
    result = result.merge(expected[["patient_id", "within_split_order"]], on="within_split_order", how="left", validate="one_to_one")
    result["method_id"] = "Clinical reader"
    result["method_label"] = "Clinical reader"
    result["method_type"] = "clinical_reader"
    result["seed"] = np.nan
    result["probability"] = probability.to_numpy()
    result["score"] = probability.where(probability_available, result["binary_prediction"].astype(float)).to_numpy()
    result["probability_available"] = bool(probability_available.all())
    return result[["method_id", "method_label", "method_type", "patient_id", "within_split_order", "observed_label", "seed", "probability", "score", "binary_prediction", "probability_available"]].sort_values("within_split_order").reset_index(drop=True)


def verify_matching_manifests(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> None:
    left_sorted = left.sort_values("within_split_order").reset_index(drop=True)
    right_sorted = right.sort_values("within_split_order").reset_index(drop=True)
    compare = ["within_split_order", "observed_label"]
    if not left_sorted[compare].equals(right_sorted[compare]):
        raise ValueError("{0} and {1} test manifests do not share order and labels".format(left_name, right_name))


def aggregate_repeated_predictions(long: pd.DataFrame, settings: ReportingSettings, expected_seeds: int) -> pd.DataFrame:
    """Average repeat probabilities into one patient-level record per method.

    This is the only place repeated neural-network seeds are collapsed for
    patient-level inference. It does not use model performance to choose a seed.
    """
    required = {"method_id", "method_label", "method_type", "patient_id", "within_split_order", "observed_label", "seed", "probability"}
    if not required.issubset(long.columns):
        raise ValueError("Repeated prediction data lacks: {0}".format(sorted(required - set(long.columns))))
    group_columns = ["method_id", "method_label", "method_type", "patient_id", "within_split_order", "observed_label"]
    counts = long.groupby(group_columns, dropna=False)["seed"].nunique()
    if not counts.eq(expected_seeds).all():
        raise ValueError("Every method/patient must have exactly {0} repeated predictions".format(expected_seeds))
    result = long.groupby(group_columns, as_index=False).agg(
        probability=("probability", "mean"),
        score=("probability", "mean"),
        n_runs=("seed", "nunique"),
    )
    result["binary_prediction"] = (result["probability"] >= settings.threshold).astype(int)
    result["probability_available"] = True
    return result.sort_values(["method_id", "within_split_order"]).reset_index(drop=True)


def load_all_sources(
    tl_workbook: Path,
    baseline_workbook: Path,
    settings: ReportingSettings,
    ensemble_workbook: Optional[Path] = None,
    reference_cnn_workbook: Optional[Path] = None,
    clinical_reader_file: Optional[Path] = None,
) -> SourceData:
    """Load all available sources without modifying any input workbook."""
    tl_runs, tl_patient, manifest = load_tl_metrics2026(tl_workbook, settings)
    baseline_patient, baseline_metadata = load_baseline_predictions(baseline_workbook, manifest)
    ensemble_runs: Optional[pd.DataFrame] = None
    ensemble_patient: Optional[pd.DataFrame] = None
    ensemble_metadata: Mapping[str, str] = {}
    cnn_patient: Optional[pd.DataFrame] = None
    reader_patient: Optional[pd.DataFrame] = None
    if ensemble_workbook is not None and str(ensemble_workbook):
        ensemble_runs, ensemble_patient, ensemble_metadata = load_ensemble_raw_runs(ensemble_workbook, manifest, settings)
    if reference_cnn_workbook is not None and str(reference_cnn_workbook):
        cnn_patient = load_reference_cnn_runs(reference_cnn_workbook, manifest, settings)
    if clinical_reader_file is not None and str(clinical_reader_file):
        reader_patient = load_clinical_reader(clinical_reader_file, manifest)
    return SourceData(
        tl_runs=tl_runs, tl_patient_predictions=tl_patient, test_manifest=manifest,
        ensemble_runs=ensemble_runs, ensemble_patient_predictions=ensemble_patient,
        ensemble_metadata=ensemble_metadata, baseline_patient_predictions=baseline_patient,
        baseline_metadata=baseline_metadata, reference_cnn_patient_predictions=cnn_patient,
        clinical_reader_patient_predictions=reader_patient,
    )


# ---------------------------------------------------------------------------
# Patient-level metrics, bootstrap confidence intervals, and paired tests
# ---------------------------------------------------------------------------

def _metric_values(labels: np.ndarray, scores: np.ndarray, binary: np.ndarray, auc_available: bool) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, binary, labels=[0, 1]).ravel()
    values = {
        "accuracy": float(accuracy_score(labels, binary)),
        "precision": float(precision_score(labels, binary, zero_division=0)),
        "sensitivity": float(recall_score(labels, binary, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(labels, binary, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)) if auc_available else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return values


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    negatives = np.flatnonzero(labels == 0)
    positives = np.flatnonzero(labels == 1)
    return np.concatenate((rng.choice(negatives, size=len(negatives), replace=True), rng.choice(positives, size=len(positives), replace=True)))


def _method_seed(method_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(method_id.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2 ** 32 - 1)


def patient_metrics_ci(patient_predictions: pd.DataFrame, settings: ReportingSettings) -> pd.DataFrame:
    """Compute point estimates and stratified patient-bootstrap 95% confidence intervals."""
    required = {"method_id", "method_label", "observed_label", "score", "binary_prediction", "probability_available"}
    if not required.issubset(patient_predictions.columns):
        raise ValueError("Patient-level data lacks: {0}".format(sorted(required - set(patient_predictions.columns))))
    rows: List[Dict[str, Any]] = []
    for method_id, group in patient_predictions.groupby("method_id", sort=False):
        group = group.sort_values("within_split_order")
        if len(group) != 46 or group["within_split_order"].tolist() != list(range(1, 47)):
            raise ValueError("{0} does not have exactly one record for each test patient".format(method_id))
        labels = group["observed_label"].to_numpy(dtype=int)
        scores = group["score"].to_numpy(dtype=float)
        binary = group["binary_prediction"].to_numpy(dtype=int)
        auc_available = bool(group["probability_available"].all()) or str(group["method_type"].iloc[0]) == "conventional_baseline"
        point = _metric_values(labels, scores, binary, auc_available)
        rng = np.random.default_rng(_method_seed(str(method_id), settings.random_seed))
        boot: Dict[str, List[float]] = {metric: [] for metric in METRIC_ORDER}
        for _ in range(settings.bootstrap_iterations):
            index = _stratified_indices(labels, rng)
            values = _metric_values(labels[index], scores[index], binary[index], auc_available)
            for metric in METRIC_ORDER:
                if np.isfinite(values[metric]):
                    boot[metric].append(values[metric])
        row: Dict[str, Any] = {
            "method_id": method_id, "method_label": group["method_label"].iloc[0],
            "method_type": group["method_type"].iloc[0], "n_patients": int(len(group)),
            "probability_available": bool(group["probability_available"].all()),
            "auc_available": bool(auc_available),
        }
        for metric in METRIC_ORDER:
            row[metric + "_point"] = point[metric]
            if boot[metric]:
                low, high = np.quantile(np.asarray(boot[metric], dtype=float), (0.025, 0.975))
                row[metric + "_ci_low"] = float(low)
                row[metric + "_ci_high"] = float(high)
            else:
                row[metric + "_ci_low"] = float("nan")
                row[metric + "_ci_high"] = float("nan")
        for item in ("tp", "tn", "fp", "fn"):
            row[item] = int(point[item])
        rows.append(row)
    return pd.DataFrame(rows)


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def _fast_delong(predictions_sorted: np.ndarray, n_positive: int) -> Tuple[np.ndarray, np.ndarray]:
    n_examples = predictions_sorted.shape[1]
    n_negative = n_examples - n_positive
    positive = predictions_sorted[:, :n_positive]
    negative = predictions_sorted[:, n_positive:]
    tx = np.asarray([_midrank(row) for row in positive])
    ty = np.asarray([_midrank(row) for row in negative])
    tz = np.asarray([_midrank(row) for row in predictions_sorted])
    auc = tz[:, :n_positive].sum(axis=1) / n_positive / n_negative - (n_positive + 1.0) / (2.0 * n_negative)
    v01 = (tz[:, :n_positive] - tx) / n_negative
    v10 = 1.0 - (tz[:, n_positive:] - ty) / n_positive
    sx = np.cov(v01)
    sy = np.cov(v10)
    if np.ndim(sx) == 0:
        sx = np.asarray([[float(sx)]])
    if np.ndim(sy) == 0:
        sy = np.asarray([[float(sy)]])
    covariance = sx / n_positive + sy / n_negative
    return auc, covariance


def paired_delong(labels: Sequence[int], score_a: Sequence[float], score_b: Sequence[float]) -> Tuple[float, float, float]:
    """Return AUC difference (a-b), paired DeLong z statistic, and two-sided p value."""
    labels_array = np.asarray(labels, dtype=int)
    scores = np.vstack((np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)))
    order = np.argsort(-labels_array)
    ordered_labels = labels_array[order]
    n_positive = int(ordered_labels.sum())
    if n_positive == 0 or n_positive == len(ordered_labels):
        return float("nan"), float("nan"), float("nan")
    aucs, covariance = _fast_delong(scores[:, order], n_positive)
    difference = float(aucs[0] - aucs[1])
    variance = float(np.asarray([1.0, -1.0]).dot(covariance).dot(np.asarray([1.0, -1.0])))
    if variance <= 0.0 or not np.isfinite(variance):
        z_value = 0.0 if abs(difference) < 1e-12 else float("inf")
        p_value = 1.0 if z_value == 0.0 else 0.0
    else:
        z_value = difference / math.sqrt(variance)
        p_value = float(2.0 * norm.sf(abs(z_value)))
    return difference, float(z_value), p_value


def _mcnemar_p_value(binary_a: np.ndarray, binary_b: np.ndarray, labels: np.ndarray) -> float:
    correct_a = binary_a == labels
    correct_b = binary_b == labels
    discordant_a = int(np.sum(correct_a & ~correct_b))
    discordant_b = int(np.sum(~correct_a & correct_b))
    if discordant_a + discordant_b == 0:
        return 1.0
    return float(binomtest(min(discordant_a, discordant_b), n=discordant_a + discordant_b, p=0.5, alternative="two-sided").pvalue)


def paired_comparison_table(patient_predictions: pd.DataFrame, ensemble_method_id: str, comparator_method_ids: Sequence[str], settings: ReportingSettings) -> pd.DataFrame:
    """Patient-level paired comparison table; seed rows are never used as observations."""
    all_methods = set(patient_predictions["method_id"])
    if ensemble_method_id not in all_methods:
        raise ValueError("Selected ensemble method is not present")
    ensemble = patient_predictions.loc[patient_predictions["method_id"] == ensemble_method_id].sort_values("within_split_order")
    labels = ensemble["observed_label"].to_numpy(dtype=int)
    rows: List[Dict[str, Any]] = []
    for comparator_id in comparator_method_ids:
        if comparator_id not in all_methods:
            continue
        comparator = patient_predictions.loc[patient_predictions["method_id"] == comparator_id].sort_values("within_split_order")
        if not np.array_equal(labels, comparator["observed_label"].to_numpy(dtype=int)):
            raise ValueError("Outcome labels differ for paired comparison: {0}".format(comparator_id))
        a_scores = ensemble["score"].to_numpy(dtype=float)
        b_scores = comparator["score"].to_numpy(dtype=float)
        a_binary = ensemble["binary_prediction"].to_numpy(dtype=int)
        b_binary = comparator["binary_prediction"].to_numpy(dtype=int)
        auc_available = bool(ensemble["probability_available"].all()) and (bool(comparator["probability_available"].all()) or str(comparator["method_type"].iloc[0]) == "conventional_baseline")
        rng = np.random.default_rng(_method_seed("{0}__{1}".format(ensemble_method_id, comparator_id), settings.random_seed + 31))
        differences: Dict[str, List[float]] = {metric: [] for metric in METRIC_ORDER}
        for _ in range(settings.bootstrap_iterations):
            index = _stratified_indices(labels, rng)
            a_metrics = _metric_values(labels[index], a_scores[index], a_binary[index], auc_available)
            b_metrics = _metric_values(labels[index], b_scores[index], b_binary[index], auc_available)
            for metric in METRIC_ORDER:
                if np.isfinite(a_metrics[metric]) and np.isfinite(b_metrics[metric]):
                    differences[metric].append(float(a_metrics[metric] - b_metrics[metric]))
        row: Dict[str, Any] = {
            "comparison": "Ensemble versus {0}".format(comparator["method_label"].iloc[0]),
            "ensemble_method_id": ensemble_method_id, "comparator_method_id": comparator_id,
            "mcnemar_p_value": _mcnemar_p_value(a_binary, b_binary, labels),
        }
        for metric in METRIC_ORDER:
            a_point = _metric_values(labels, a_scores, a_binary, auc_available)[metric]
            b_point = _metric_values(labels, b_scores, b_binary, auc_available)[metric]
            row[metric + "_difference"] = float(a_point - b_point) if np.isfinite(a_point) and np.isfinite(b_point) else float("nan")
            if differences[metric]:
                low, high = np.quantile(np.asarray(differences[metric]), (0.025, 0.975))
                row[metric + "_ci_low"] = float(low)
                row[metric + "_ci_high"] = float(high)
            else:
                row[metric + "_ci_low"] = float("nan")
                row[metric + "_ci_high"] = float("nan")
        if auc_available:
            _, z_value, delong_p = paired_delong(labels, a_scores, b_scores)
            row["delong_z"] = z_value
            row["delong_p_value"] = delong_p
        else:
            row["delong_z"] = float("nan")
            row["delong_p_value"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table formatting and calibration / DCA data
# ---------------------------------------------------------------------------

def _format_ci(point: float, low: float, high: float, decimals: int = 3) -> str:
    if not (np.isfinite(point) and np.isfinite(low) and np.isfinite(high)):
        return "—"
    template = "{{0:.{0}f}} ({{1:.{0}f}}–{{2:.{0}f}})".format(decimals)
    return template.format(point, low, high)


def formatted_performance_table(metrics: pd.DataFrame, method_ids: Sequence[str]) -> pd.DataFrame:
    indexed = metrics.set_index("method_id", drop=False)
    rows: List[Dict[str, Any]] = []
    for method_id in method_ids:
        if method_id not in indexed.index:
            continue
        row = indexed.loc[method_id]
        display: Dict[str, Any] = {"Method": row["method_label"]}
        for metric in ("accuracy", "auc", "f1", "sensitivity", "specificity"):
            display[METRIC_LABELS[metric] + " (95% CI)"] = _format_ci(row[metric + "_point"], row[metric + "_ci_low"], row[metric + "_ci_high"])
        rows.append(display)
    return pd.DataFrame(rows)


def formatted_confusion_table(metrics: pd.DataFrame, method_ids: Sequence[str]) -> pd.DataFrame:
    indexed = metrics.set_index("method_id", drop=False)
    rows: List[Dict[str, Any]] = []
    for method_id in method_ids:
        if method_id in indexed.index:
            row = indexed.loc[method_id]
            rows.append({"Method": row["method_label"], "TP": int(row["tp"]), "TN": int(row["tn"]), "FP": int(row["fp"]), "FN": int(row["fn"])})
    return pd.DataFrame(rows)


def formatted_full_conventional_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Supplementary Table ST1: performance intervals plus one patient-level matrix."""
    performance = formatted_performance_table(metrics, BASELINE_ORDER)
    confusion = formatted_confusion_table(metrics, BASELINE_ORDER)
    if performance.empty:
        return performance
    return performance.merge(confusion, on="Method", how="left", validate="one_to_one")


def seed_stability_table(tl_runs: pd.DataFrame) -> pd.DataFrame:
    required = {"model_name", "seed", "test_accuracy", "test_precision", "test_sensitivity", "test_specificity", "test_f1", "test_auc"}
    if not required.issubset(tl_runs.columns):
        raise ValueError("TL runs lack required stability columns")
    rows: List[Dict[str, Any]] = []
    for model_name in TL_MODELS:
        group = tl_runs.loc[tl_runs["model_name"].astype(str) == model_name]
        row: Dict[str, Any] = {"Model": model_name, "n runs": int(len(group))}
        for label, source in (("ACC", "test_accuracy"), ("AUC", "test_auc"), ("F1S", "test_f1"), ("SEN", "test_sensitivity"), ("SPE", "test_specificity"), ("PRE", "test_precision")):
            values = pd.to_numeric(group[source], errors="raise")
            row[label + " median (IQR)"] = "{0:.3f} ({1:.3f})".format(float(values.median()), float(values.quantile(0.75) - values.quantile(0.25)))
        rows.append(row)
    return pd.DataFrame(rows)


def formatted_paired_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in comparisons.itertuples(index=False):
        values = row._asdict()
        rows.append({
            "Comparison": values["comparison"],
            "AUC difference (95% CI)": _format_ci(values["auc_difference"], values["auc_ci_low"], values["auc_ci_high"]),
            "Paired DeLong p": _format_p(values["delong_p_value"]),
            "ACC difference (95% CI)": _format_ci(values["accuracy_difference"], values["accuracy_ci_low"], values["accuracy_ci_high"]),
            "McNemar p": _format_p(values["mcnemar_p_value"]),
            "F1S difference (95% CI)": _format_ci(values["f1_difference"], values["f1_ci_low"], values["f1_ci_high"]),
            "SEN difference (95% CI)": _format_ci(values["sensitivity_difference"], values["sensitivity_ci_low"], values["sensitivity_ci_high"]),
            "SPE difference (95% CI)": _format_ci(values["specificity_difference"], values["specificity_ci_low"], values["specificity_ci_high"]),
        })
    return pd.DataFrame(rows)


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return "<0.001" if value < 0.001 else "{0:.3f}".format(value)


def calibration_table(patient_predictions: pd.DataFrame, method_ids: Sequence[str], settings: ReportingSettings) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method_id in method_ids:
        group = patient_predictions.loc[patient_predictions["method_id"] == method_id].sort_values("within_split_order")
        if group.empty or not bool(group["probability_available"].all()):
            continue
        probabilities = group["probability"].to_numpy(dtype=float)
        labels = group["observed_label"].to_numpy(dtype=int)
        intercept, slope = _calibration_intercept_slope(labels, probabilities)
        rows.append({
            "Method": group["method_label"].iloc[0], "Brier score": float(brier_score_loss(labels, probabilities)),
            "Calibration intercept": intercept, "Calibration slope": slope, "n patients": int(len(group)),
        })
    return pd.DataFrame(rows)


def _calibration_intercept_slope(labels: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
    logits = np.log(np.clip(probabilities, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probabilities, 1e-6, 1.0))
    try:
        try:
            model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
            model.fit(logits.reshape(-1, 1), labels)
        except ValueError:
            model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            model.fit(logits.reshape(-1, 1), labels)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def decision_curve_table(patient_predictions: pd.DataFrame, method_ids: Sequence[str], settings: ReportingSettings) -> pd.DataFrame:
    thresholds = np.arange(settings.dca_threshold_start, settings.dca_threshold_end + settings.dca_threshold_step / 2.0, settings.dca_threshold_step)
    rows: List[Dict[str, Any]] = []
    prevalence: Optional[float] = None
    for method_id in method_ids:
        group = patient_predictions.loc[patient_predictions["method_id"] == method_id].sort_values("within_split_order")
        if group.empty or not bool(group["probability_available"].all()):
            continue
        labels = group["observed_label"].to_numpy(dtype=int)
        probabilities = group["probability"].to_numpy(dtype=float)
        prevalence = float(labels.mean())
        for threshold in thresholds:
            predicted = probabilities >= threshold
            tp = int(np.sum((predicted == 1) & (labels == 1)))
            fp = int(np.sum((predicted == 1) & (labels == 0)))
            benefit = tp / len(labels) - fp / len(labels) * threshold / (1.0 - threshold)
            rows.append({"method_id": method_id, "Method": group["method_label"].iloc[0], "threshold_probability": float(threshold), "net_benefit": float(benefit)})
    if prevalence is None:
        return pd.DataFrame(columns=["method_id", "Method", "threshold_probability", "net_benefit"])
    for threshold in thresholds:
        rows.append({"method_id": "Treat all", "Method": "Treat all", "threshold_probability": float(threshold), "net_benefit": float(prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold))})
        rows.append({"method_id": "Treat none", "Method": "Treat none", "threshold_probability": float(threshold), "net_benefit": 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Publication-style figures: same visual language as the submitted notebooks
# ---------------------------------------------------------------------------

def _paper_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
        "axes.labelsize": 20, "axes.labelweight": "bold", "xtick.labelsize": 20,
        "ytick.labelsize": 20, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.linewidth": 0.8,
    })


def _model_order(available: Iterable[str]) -> List[str]:
    observed = list(dict.fromkeys(str(item) for item in available))
    return [model for model in TL_MODELS if model in observed] + [model for model in observed if model not in TL_MODELS]


def plot_tl_roc(patient_predictions: pd.DataFrame, output: Path) -> None:
    _paper_style()
    methods = _model_order(patient_predictions.loc[patient_predictions["method_type"] == "transfer_learning", "method_id"].unique())
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H"]
    line_styles = ["-", "--", "-.", ":"]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figure, axis = plt.subplots(figsize=(14, 8))
    for index, method_id in enumerate(methods):
        group = patient_predictions.loc[patient_predictions["method_id"] == method_id].sort_values("within_split_order")
        labels = group["observed_label"].to_numpy(dtype=int)
        probabilities = group["probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        axis.plot(fpr, tpr, label="{0} (AUC = {1:.2f})".format(method_id, auc), color=colours[index % len(colours)], linestyle=line_styles[index % len(line_styles)], marker=markers[index % len(markers)], markevery=max(1, len(fpr) // 8), linewidth=1.5, solid_capstyle="round")
    axis.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random classification")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False Positive Rate", labelpad=15)
    axis.set_ylabel("True Positive Rate", labelpad=15)
    ticks = axis.get_yticks()
    axis.set_yticks([tick for tick in ticks if tick != 0.0])
    axis.grid(True, linewidth=0.8, alpha=0.7)
    axis.legend(title="Models", bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=18, title_fontsize=18)
    figure.tight_layout(pad=3.0)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_dca(dca: pd.DataFrame, output: Path, title: str = "") -> None:
    _paper_style()
    if dca.empty:
        raise ValueError("Decision-curve data are empty")
    methods = [method for method in dca["Method"].unique() if method not in ("Treat all", "Treat none")]
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H"]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fixed_colours = {"Ensemble": "blue", "Reference CNN": "darkgrey", "Clinical reader": "red"}
    figure, axis = plt.subplots(figsize=(14, 8))
    for index, method in enumerate(methods):
        group = dca.loc[dca["Method"] == method]
        group = group.loc[group["net_benefit"] >= 0.0]
        axis.plot(group["threshold_probability"] * 100.0, group["net_benefit"], label=method, marker=markers[index % len(markers)], markevery=(index % 10, 10), color=fixed_colours.get(method, colours[index % len(colours)]), linewidth=2.0, solid_capstyle="round")
    for method, style in (("Treat all", "--"), ("Treat none", "-.")):
        group = dca.loc[dca["Method"] == method]
        group = group.loc[group["net_benefit"] >= 0.0]
        axis.plot(group["threshold_probability"] * 100.0, group["net_benefit"], linestyle=style, linewidth=1.2, color="black", alpha=0.55, label=method)
    axis.set_xlim(0, 100)
    non_negative = dca.loc[dca["net_benefit"] >= 0.0, "net_benefit"]
    maximum = float(non_negative.max()) if not non_negative.empty else 0.30
    axis.set_ylim(0.0, max(0.35, math.ceil(maximum * 20.0) / 20.0 + 0.05))
    axis.set_xlabel("Threshold Probability (%)", labelpad=15)
    axis.set_ylabel("Net Benefit", labelpad=15)
    axis.grid(True)
    axis.legend(title="Models", bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=18, title_fontsize=18)
    if title:
        axis.set_title(title, fontsize=18, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_seed_stability(tl_runs: pd.DataFrame, output: Path) -> None:
    _paper_style()
    model_order = _model_order(tl_runs["model_name"].astype(str).unique())
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for axis, metric_column, title in zip(axes, ("test_accuracy", "test_auc"), ("Accuracy", "AUC")):
        series = [pd.to_numeric(tl_runs.loc[tl_runs["model_name"].astype(str) == model, metric_column], errors="raise").to_numpy(dtype=float) for model in model_order]
        box = axis.boxplot(series, patch_artist=True, showfliers=True, medianprops={"color": "red", "linewidth": 2}, boxprops={"facecolor": "white", "edgecolor": "blue"}, whiskerprops={"color": "blue", "linewidth": 1.5}, capprops={"color": "blue", "linewidth": 1.5}, flierprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "blue"})
        axis.set_xticks(range(1, len(model_order) + 1))
        axis.set_xticklabels(model_order)
        axis.set_title(title, fontsize=20, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylim(-0.02, 1.05)
        axis.tick_params(axis="x", labelrotation=60, labelsize=12)
        axis.grid(True, linestyle="--", color="grey", alpha=0.5)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.5)
    axes[0].set_ylabel("Value", labelpad=15)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_calibration(patient_predictions: pd.DataFrame, method_ids: Sequence[str], settings: ReportingSettings, output: Path) -> None:
    _paper_style()
    figure, axis = plt.subplots(figsize=(14, 8))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted = 0
    for index, method_id in enumerate(method_ids):
        group = patient_predictions.loc[patient_predictions["method_id"] == method_id].sort_values("within_split_order")
        if group.empty or not bool(group["probability_available"].all()):
            continue
        observed, predicted = calibration_curve(group["observed_label"], group["probability"], n_bins=settings.calibration_bins, strategy="uniform")
        axis.plot(predicted, observed, marker="o", linewidth=2, label=group["method_label"].iloc[0], color=colours[plotted % len(colours)])
        plotted += 1
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.2, label="Perfect calibration")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean Predicted Probability", labelpad=15)
    axis.set_ylabel("Observed Proportion Positive", labelpad=15)
    axis.grid(True, alpha=0.7)
    axis.legend(title="Models", bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=16, title_fontsize=18)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# One complete paper-order output package
# ---------------------------------------------------------------------------

def _write_table_csv(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)


def _write_excel(tables: Mapping[str, pd.DataFrame], output: Path) -> None:
    with pd.ExcelWriter(str(output), engine="openpyxl", mode="w") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
            worksheet = writer.book[name[:31]]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)


def captions_and_text(ensemble_metadata: Mapping[str, str], clinical_reader_present: bool) -> str:
    configuration = "{0} {1}".format(ensemble_metadata.get("pool_name", "[development-selected pool]"), ensemble_metadata.get("rule_id", "[development-selected rule]")).replace("_", " ")
    reader = " The clinical reader is included as a descriptive comparator." if clinical_reader_present else ""
    return """# Results text and captions — generated reporting package

## Transfer learning

**N2.** Table T1 presents a comparison of all pre-trained models using ACC, AUC, F1S, SEN, and SPE. Values were calculated from seed-aggregated patient-level predictions in the independent test cohort and are reported with 95% bootstrap confidence intervals.

**N3.** Table T2 presents TP, TN, FP, and FN for all pre-trained models. Each count was calculated once per test patient from the seed-aggregated predictions.

**N4.** Figure F1 presents receiver operating characteristic curves for all pre-trained models, derived from seed-aggregated patient-level predictions in the independent test cohort.

**N5.** Figure F2 illustrates decision-curve analysis through net-benefit curves for all pre-trained models. Each curve represents the net benefit of using the respective model for classifying significant CAD, compared with treat-all and treat-none strategies.

## Ensemble learning

**N7.** Using development-set predictions, the {configuration} configuration was selected before independent test evaluation as the final ensemble. Its constituent architectures and development-only selection record are retained in the raw ensemble workbook's Study_lock sheet.

**N8.** Table T3 presents the final ensemble, its constituent models, the reference CNN when supplied, and the two prespecified conventional models based on the complete 17-segment MBF pattern, with and without clinical variables.{reader}

**N9.** Table T4 shows TP, TN, FP, and FN for the principal comparison approaches in the independent test cohort.

**N11.** Figure F3 shows decision-curve analysis of the final probabilistic comparison models, alongside treat-all and treat-none strategies.

## Captions

**Table T1.** Comparison of pre-trained model performance for classifying significant CAD. Values were calculated from seed-aggregated patient-level predictions in the independent test cohort and are presented with 95% bootstrap confidence intervals. Abbreviations: ACC, accuracy; AUC, area under the receiver operating characteristic curve; F1S, F1-score; SEN, sensitivity; SPE, specificity.

**Table T2.** Confusion-matrix results for the pre-trained models. Counts were calculated from seed-aggregated patient-level predictions in the independent test cohort. Abbreviations: FN, false negative; FP, false positive; TN, true negative; TP, true positive.

**Figure F1.** Receiver operating characteristic curves for the pre-trained models. Each curve was derived from seed-aggregated patient-level predictions in the independent test cohort. The diagonal line represents random classification.

**Figure F2.** Net-benefit curves for all pre-trained models across threshold probabilities from 0% to 100%. Curves were derived from seed-aggregated patient-level predictions in the independent test cohort and are compared with treat-all and treat-none strategies.

**Table T3.** Performance comparison of final prespecified approaches for classifying significant CAD in the independent test cohort. The ensemble configuration was selected using development data before independent test evaluation. Values are presented with 95% bootstrap confidence intervals.

**Table T4.** Confusion-matrix results for the final comparison approaches. Counts were calculated once per patient in the independent test cohort.

**Figure F3.** Decision-curve analysis of final probabilistic models for classifying significant CAD in the independent test cohort, alongside treat-all and treat-none strategies.

**Supplementary Table ST1.** Performance of all six prespecified conventional quantitative-PET and clinical comparators for classifying significant CAD in the independent test cohort.

**Supplementary Table ST2.** Stability of transfer-learning performance across 100 random seeds. Values are reported as median (interquartile range) across repeated training runs and describe stochastic training variability; they were not treated as independent patient-level observations for clinical inference.

**Supplementary Figure SF1.** Distribution of accuracy and AUC across 100 random seeds for each transfer-learning architecture. The distributions describe stability across stochastic model initialisations.

**Supplementary Table ST3.** Patient-level paired comparisons between the final ensemble and prespecified approaches. Confidence intervals for differences were obtained by patient-level bootstrap resampling. Paired DeLong testing was used for AUC comparisons, and McNemar testing was used for paired binary classifications.

**Supplementary Table ST4.** Calibration measures for probabilistic models in the independent test cohort. Brier score, calibration intercept, and calibration slope are reported for models that generated patient-level probabilities.
""".format(configuration=configuration, reader=reader)


def run_results_publish(
    sources: SourceData,
    output_dir: Path,
    settings: ReportingSettings,
    principal_baselines: Sequence[str] = ("segmental17_mbf_logistic", "clinical_plus_segmental17_logistic"),
    make_calibration_figure: bool = False,
) -> Mapping[str, pd.DataFrame]:
    """Generate paper-order tables, figures, and provenance from validated inputs."""
    if sources.ensemble_patient_predictions is None:
        raise ValueError("Final ensemble raw workbook is required before generating final paper tables")
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = [sources.tl_patient_predictions, sources.ensemble_patient_predictions, sources.baseline_patient_predictions]
    if sources.reference_cnn_patient_predictions is not None:
        frames.append(sources.reference_cnn_patient_predictions)
    if sources.clinical_reader_patient_predictions is not None:
        frames.append(sources.clinical_reader_patient_predictions)
    patient_predictions = pd.concat(frames, ignore_index=True, sort=False)
    if patient_predictions.duplicated(["method_id", "within_split_order"]).any():
        raise ValueError("Reporting inputs have duplicate method/patient rows")
    metrics = patient_metrics_ci(patient_predictions, settings)

    ensemble_constituents = [name for name in sources.ensemble_metadata.get("constituent_models", "").split(";") if name]
    if len(ensemble_constituents) not in (3, 5) or not set(ensemble_constituents).issubset(set(TL_MODELS)):
        raise ValueError("Raw ensemble workbook must identify three or five valid TL constituents")
    principal = ["Reference CNN"] if sources.reference_cnn_patient_predictions is not None else []
    if sources.clinical_reader_patient_predictions is not None:
        principal.append("Clinical reader")
    principal.extend(["Ensemble"] + ensemble_constituents + list(principal_baselines))
    main_confusion = [method for method in (["Reference CNN"] if sources.reference_cnn_patient_predictions is not None else []) + (["Clinical reader"] if sources.clinical_reader_patient_predictions is not None else []) + ["Ensemble"] + list(principal_baselines) if method in set(metrics["method_id"])]

    all_tl = list(TL_MODELS)
    all_baselines = list(BASELINE_ORDER)
    comparison_targets = []
    if sources.reference_cnn_patient_predictions is not None:
        comparison_targets.append("Reference CNN")
    comparison_targets.extend(ensemble_constituents + all_baselines)
    if sources.clinical_reader_patient_predictions is not None:
        comparison_targets.append("Clinical reader")
    comparisons = paired_comparison_table(patient_predictions, "Ensemble", comparison_targets, settings)

    calibration_ids = ["Ensemble"] + ensemble_constituents
    if sources.reference_cnn_patient_predictions is not None:
        calibration_ids.append("Reference CNN")
    calibration_ids.extend([baseline for baseline in all_baselines if baseline != "global_mbf_lt_2_3_rule"])

    tables: Dict[str, pd.DataFrame] = {
        "T1_TL_performance": formatted_performance_table(metrics, all_tl),
        "T2_TL_confusion": formatted_confusion_table(metrics, all_tl),
        "T3_principal_comparison": formatted_performance_table(metrics, principal),
        "T4_principal_confusion": formatted_confusion_table(metrics, main_confusion),
        "ST1_all_conventional": formatted_full_conventional_table(metrics),
        "ST2_TL_seed_stability": seed_stability_table(sources.tl_runs),
        "ST3_paired_comparisons": formatted_paired_table(comparisons),
        "ST4_calibration": calibration_table(patient_predictions, calibration_ids, settings),
        "patient_level_metrics_numeric": metrics,
        "patient_level_predictions_secure": patient_predictions,
        "paired_comparisons_numeric": comparisons,
    }
    for name, table in tables.items():
        _write_table_csv(table, tables_dir / (name + ".csv"))
    _write_excel(tables, output_dir / "results_publish_tables.xlsx")

    plot_tl_roc(patient_predictions, figures_dir / "F1_TL_ROC.png")
    dca_all_tl = decision_curve_table(patient_predictions, all_tl, settings)
    plot_dca(dca_all_tl, figures_dir / "F2_TL_DCA.png")
    final_dca_ids = ["Ensemble"]
    if sources.reference_cnn_patient_predictions is not None:
        final_dca_ids.append("Reference CNN")
    if ensemble_constituents:
        final_dca_ids.append(ensemble_constituents[0])
    final_dca_ids.extend(list(principal_baselines))
    dca_final = decision_curve_table(patient_predictions, final_dca_ids, settings)
    plot_dca(dca_final, figures_dir / "F3_final_comparison_DCA.png")
    plot_seed_stability(sources.tl_runs, figures_dir / "SF1_TL_seed_stability.png")
    if make_calibration_figure:
        plot_calibration(patient_predictions, calibration_ids, settings, figures_dir / "SF2_calibration.png")

    (output_dir / "captions_and_results_text.md").write_text(captions_and_text(sources.ensemble_metadata, sources.clinical_reader_patient_predictions is not None), encoding="utf-8")
    manifest = {
        "analysis": "EJPH-D-26-00179 paper-order reporting package",
        "settings": asdict(settings),
        "source_modified": False,
        "statistical_boundary": "100 seed runs are descriptive stability only. Patient-level CIs and paired tests use one seed-mean probability per test patient/method.",
        "excluded_legacy_procedures": ["best run", "Borda Count", "test-set ensemble selection", "run-level inferential tests"],
        "ensemble_metadata": dict(sources.ensemble_metadata),
        "principal_baselines": list(principal_baselines),
        "clinical_reader_included": sources.clinical_reader_patient_predictions is not None,
        "reference_cnn_included": sources.reference_cnn_patient_predictions is not None,
        "deliverables": {
            "tables_workbook": "results_publish_tables.xlsx",
            "figures": ["F1_TL_ROC.png", "F2_TL_DCA.png", "F3_final_comparison_DCA.png", "SF1_TL_seed_stability.png"],
        },
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tables
