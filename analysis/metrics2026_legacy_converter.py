"""Convert final Experiment2026 raw results into the legacy metrics.xlsx layout.

This is a visualisation-compatibility adapter only. It never changes the audited
``metrics2026.xlsx`` source workbook and it must not be used to recreate legacy
selection, best-run, Borda-count, or run-level inferential analyses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

EXPECTED_MODELS: Tuple[str, ...] = (
    "VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "InceptionV3",
    "InceptionResNetV2", "DenseNet169", "DenseNet201", "MobileNetV2", "Xception",
)
REFERENCE_CNN_MODEL = "ReferenceCNN_4Conv_v2_55"
LEGACY_COLUMNS: Tuple[str, ...] = (
    "model_name", "elapsed_time", "input_size", "batch_size", "epochs",
    "epochs_partial_1", "epochs_partial_2", "freeze_fe", "is_gray", "optimizer",
    "early_stopping", "class_weights",
    "train_accuracy", "train_precision", "train_recall", "train_f1score",
    "train_confusion_matrix", "train_predicts",
    "val_accuracy", "val_precision", "val_recall", "val_f1score", "val_auc",
    "val_specificity", "val_confusion_matrix", "val_predicts", "val_roc_curve_fpr",
    "val_roc_curve_tpr", "val_roc_curve_th",
    "test_accuracy", "test_precision", "test_recall", "test_f1score", "test_auc",
    "test_specificity", "test_confusion_matrix", "test_predicts", "test_roc_curve_fpr",
    "test_roc_curve_tpr", "test_roc_curve_th", "tag",
)


def _parse_probability_csv(value: Any, expected_length: int) -> np.ndarray:
    text = str(value).strip().strip("[](){}")
    if not text or text.lower() == "nan":
        raise ValueError("Probability list is empty")
    values = np.asarray([float(part.strip()) for part in text.split(",") if part.strip()], dtype=float)
    if len(values) != expected_length:
        raise ValueError("Expected {0} probabilities; found {1}".format(expected_length, len(values)))
    if not np.isfinite(values).all() or not np.logical_and(values >= 0.0, values <= 1.0).all():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return values


def _legacy_probability_text(values: Sequence[float]) -> str:
    """Use the old four-decimal text style so legacy plotting cells parse normally."""
    return ",".join("{0:.4f}".format(float(value)) for value in values)


def _legacy_roc_text(labels: np.ndarray, probabilities: np.ndarray) -> Tuple[str, str, str]:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    return (
        ",".join("{0:.4f}".format(float(value)) for value in fpr),
        ",".join("{0:.4f}".format(float(value)) for value in tpr),
        ",".join("{0:.4f}".format(float(value)) for value in thresholds),
    )


def _legacy_confusion_text(row: Mapping[str, Any], prefix: str) -> str:
    matrix = np.asarray([
        [int(row["{0}_tn".format(prefix)]), int(row["{0}_fp".format(prefix)])],
        [int(row["{0}_fn".format(prefix)]), int(row["{0}_tp".format(prefix)])],
    ])
    return str(matrix)


def _required_run_columns() -> set:
    common = {
        "model_name", "seed", "elapsed_seconds", "input_size", "batch_size", "optimizer",
        "class_weights_enabled", "phase1_requested_epochs", "phase2_requested_epochs", "phase3_requested_epochs",
    }
    for prefix in ("train", "validation", "test"):
        common.update({
            "{0}_accuracy".format(prefix), "{0}_precision".format(prefix),
            "{0}_sensitivity".format(prefix), "{0}_f1".format(prefix),
            "{0}_auc".format(prefix), "{0}_specificity".format(prefix),
            "{0}_tn".format(prefix), "{0}_fp".format(prefix),
            "{0}_fn".format(prefix), "{0}_tp".format(prefix),
            "{0}_probabilities".format(prefix),
        })
    return common


def read_and_validate_metrics2026(source_workbook: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """Read a complete final 11×100 Experiment2026 workbook without modifying it."""
    if not source_workbook.exists():
        raise FileNotFoundError("Source workbook does not exist: {0}".format(source_workbook))
    sheets = pd.read_excel(str(source_workbook), sheet_name=None, engine="openpyxl")
    expected_sheets = {"Run_results", "Patient_manifest", "Study_lock"}
    if set(sheets) != expected_sheets:
        raise ValueError("metrics2026 workbook must contain exactly {0}; found {1}".format(sorted(expected_sheets), sorted(sheets)))
    runs = sheets["Run_results"].copy()
    missing_columns = sorted(_required_run_columns() - set(runs.columns))
    if missing_columns:
        raise ValueError("Run_results is missing required columns: {0}".format(missing_columns))
    if len(runs) != len(EXPECTED_MODELS) * 100:
        raise ValueError("Expected final workbook with exactly 1,100 runs; found {0}".format(len(runs)))
    if set(runs["model_name"].astype(str)) != set(EXPECTED_MODELS):
        raise ValueError("Run_results does not contain the expected 11 final architectures")
    if runs.duplicated(["model_name", "seed"]).any():
        raise ValueError("Run_results contains duplicate model/seed rows")
    for model_name in EXPECTED_MODELS:
        seeds = set(pd.to_numeric(runs.loc[runs["model_name"].astype(str) == model_name, "seed"], errors="raise").astype(int))
        if seeds != set(range(1, 101)):
            raise ValueError("{0} must contain exactly seeds 1-100".format(model_name))

    manifest = sheets["Patient_manifest"].copy()
    required_manifest = {"split", "within_split_order", "observed_label"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError("Patient_manifest is missing required columns: {0}".format(sorted(required_manifest - set(manifest.columns))))
    labels_by_split: Dict[str, np.ndarray] = {}
    expected_counts = {"train": (61, 36, 25), "validation": (31, 20, 11), "test": (46, 26, 20)}
    for split, (expected_length, expected_zero, expected_one) in expected_counts.items():
        subset = manifest.loc[manifest["split"].astype(str) == split].copy()
        subset["within_split_order"] = pd.to_numeric(subset["within_split_order"], errors="raise").astype(int)
        subset["observed_label"] = pd.to_numeric(subset["observed_label"], errors="raise").astype(int)
        subset = subset.sort_values("within_split_order").reset_index(drop=True)
        labels = subset["observed_label"].to_numpy(dtype=int)
        if len(labels) != expected_length or subset["within_split_order"].tolist() != list(range(1, expected_length + 1)):
            raise ValueError("Patient_manifest has an invalid {0} order or length".format(split))
        if (int((labels == 0).sum()), int((labels == 1).sum())) != (expected_zero, expected_one):
            raise ValueError("Patient_manifest has invalid {0} class counts".format(split))
        labels_by_split[split] = labels
    return runs.sort_values(["model_name", "seed"]).reset_index(drop=True), labels_by_split, sheets["Study_lock"].copy()


def _legacy_row(
    row: Mapping[str, Any],
    labels_by_split: Mapping[str, np.ndarray],
    freeze_feature_extractor: bool = True,
) -> Dict[str, Any]:
    train_probabilities = _parse_probability_csv(row["train_probabilities"], len(labels_by_split["train"]))
    validation_probabilities = _parse_probability_csv(row["validation_probabilities"], len(labels_by_split["validation"]))
    test_probabilities = _parse_probability_csv(row["test_probabilities"], len(labels_by_split["test"]))
    val_fpr, val_tpr, val_thresholds = _legacy_roc_text(labels_by_split["validation"], validation_probabilities)
    test_fpr, test_tpr, test_thresholds = _legacy_roc_text(labels_by_split["test"], test_probabilities)
    return {
        "model_name": str(row["model_name"]),
        "elapsed_time": float(row["elapsed_seconds"]),
        "input_size": int(row["input_size"]),
        "batch_size": int(row["batch_size"]),
        "epochs": int(row["phase1_requested_epochs"]),
        "epochs_partial_1": int(row["phase2_requested_epochs"]),
        "epochs_partial_2": int(row["phase3_requested_epochs"]),
        "freeze_fe": bool(freeze_feature_extractor),
        "is_gray": False,
        "optimizer": str(row["optimizer"]),
        "early_stopping": True,
        "class_weights": bool(row["class_weights_enabled"]),
        "train_accuracy": float(row["train_accuracy"]),
        "train_precision": float(row["train_precision"]),
        "train_recall": float(row["train_sensitivity"]),
        "train_f1score": float(row["train_f1"]),
        "train_confusion_matrix": _legacy_confusion_text(row, "train"),
        "train_predicts": _legacy_probability_text(train_probabilities),
        "val_accuracy": float(row["validation_accuracy"]),
        "val_precision": float(row["validation_precision"]),
        "val_recall": float(row["validation_sensitivity"]),
        "val_f1score": float(row["validation_f1"]),
        "val_auc": float(row["validation_auc"]),
        "val_specificity": float(row["validation_specificity"]),
        "val_confusion_matrix": _legacy_confusion_text(row, "validation"),
        "val_predicts": _legacy_probability_text(validation_probabilities),
        "val_roc_curve_fpr": val_fpr,
        "val_roc_curve_tpr": val_tpr,
        "val_roc_curve_th": val_thresholds,
        "test_accuracy": float(row["test_accuracy"]),
        "test_precision": float(row["test_precision"]),
        "test_recall": float(row["test_sensitivity"]),
        "test_f1score": float(row["test_f1"]),
        "test_auc": float(row["test_auc"]),
        "test_specificity": float(row["test_specificity"]),
        "test_confusion_matrix": _legacy_confusion_text(row, "test"),
        "test_predicts": _legacy_probability_text(test_probabilities),
        "test_roc_curve_fpr": test_fpr,
        "test_roc_curve_tpr": test_tpr,
        "test_roc_curve_th": test_thresholds,
        # The original output's tag was used as the run identifier; the new locked
        # seed is placed there to retain direct old-notebook grouping compatibility.
        "tag": str(int(row["seed"])),
    }


def convert_metrics2026_to_legacy(source_workbook: Path, output_workbook: Path) -> pd.DataFrame:
    """Write a separate old-layout workbook with one legacy-style sheet named ``sheet1``."""
    source_workbook = source_workbook.resolve()
    output_workbook = output_workbook.resolve()
    if source_workbook == output_workbook:
        raise ValueError("Output must be a separate file; never overwrite metrics2026.xlsx")
    runs, labels_by_split, _ = read_and_validate_metrics2026(source_workbook)
    converted = pd.DataFrame([_legacy_row(row._asdict(), labels_by_split) for row in runs.itertuples(index=False)])
    if tuple(converted.columns) != LEGACY_COLUMNS:
        raise RuntimeError("Converted workbook columns differ from the fixed legacy layout")
    if len(converted) != 1100 or converted.duplicated(["model_name", "tag"]).any():
        raise RuntimeError("Converted workbook lacks one unique legacy row per model/seed")
    conversion_lock = pd.DataFrame([
        ("source_workbook", str(source_workbook)),
        ("source_design", "Final Experiment2026 workbook: 11 architectures x 100 seeds"),
        ("output_purpose", "Visualisation compatibility with legacy metrics.xlsx plotting cells"),
        ("source_modified", "false"),
        ("prediction_precision", "Legacy-compatible sheet1 prediction strings are rounded to four decimals; source metrics2026.xlsx retains audited full precision"),
        ("statistical_boundary", "Do not reuse old best-run, test-set-selection, Borda-count, or run-level inferential cells"),
        ("run_identifier_mapping", "legacy tag = new seed"),
    ], columns=["item", "value"])
    output_workbook.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(output_workbook), engine="openpyxl", mode="w") as writer:
        converted.to_excel(writer, sheet_name="sheet1", index=False)
        conversion_lock.to_excel(writer, sheet_name="Conversion_lock", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
    return converted


def read_and_validate_reference_cnn_metrics2026(source_workbook: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """Read the complete seed-varied reference-CNN workbook without modifying it."""
    if not source_workbook.exists():
        raise FileNotFoundError("Reference-CNN source workbook does not exist: {0}".format(source_workbook))
    sheets = pd.read_excel(str(source_workbook), sheet_name=None, engine="openpyxl")
    expected_sheets = {"Run_results", "Patient_manifest", "Study_lock"}
    if set(sheets) != expected_sheets:
        raise ValueError(
            "Reference-CNN workbook must contain exactly {0}; found {1}".format(
                sorted(expected_sheets), sorted(sheets)
            )
        )
    study_lock = sheets["Study_lock"].copy()
    if not {"item", "value"}.issubset(study_lock.columns):
        raise ValueError("Reference-CNN Study_lock must contain item and value columns")
    lock_values = dict(zip(study_lock["item"].astype(str), study_lock["value"].astype(str)))
    if lock_values.get("random_seed_policy") != "seed_varied_1_to_100":
        raise ValueError(
            "Reference-CNN workbook is not the required seed-varied 1-100 analysis. "
            "Do not use fixed-seed audit repeats for stability or paper reporting."
        )

    runs = sheets["Run_results"].copy()
    missing_columns = sorted(_required_run_columns() - set(runs.columns))
    if missing_columns:
        raise ValueError("Reference-CNN Run_results is missing required columns: {0}".format(missing_columns))
    if len(runs) != 100:
        raise ValueError("Reference-CNN workbook must contain exactly 100 runs; found {0}".format(len(runs)))
    if set(runs["model_name"].astype(str)) != {REFERENCE_CNN_MODEL}:
        raise ValueError("Reference-CNN Run_results must contain only {0}".format(REFERENCE_CNN_MODEL))
    if runs.duplicated("seed").any():
        raise ValueError("Reference-CNN Run_results contains duplicate seeds")
    if set(pd.to_numeric(runs["seed"], errors="raise").astype(int)) != set(range(1, 101)):
        raise ValueError("Reference-CNN workbook must contain exactly seeds 1-100")

    manifest = sheets["Patient_manifest"].copy()
    required_manifest = {"split", "within_split_order", "observed_label"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError(
            "Reference-CNN Patient_manifest is missing required columns: {0}".format(
                sorted(required_manifest - set(manifest.columns))
            )
        )
    labels_by_split: Dict[str, np.ndarray] = {}
    expected_counts = {"train": (61, 36, 25), "validation": (31, 20, 11), "test": (46, 26, 20)}
    for split, (expected_length, expected_zero, expected_one) in expected_counts.items():
        subset = manifest.loc[manifest["split"].astype(str) == split].copy()
        subset["within_split_order"] = pd.to_numeric(subset["within_split_order"], errors="raise").astype(int)
        subset["observed_label"] = pd.to_numeric(subset["observed_label"], errors="raise").astype(int)
        subset = subset.sort_values("within_split_order").reset_index(drop=True)
        labels = subset["observed_label"].to_numpy(dtype=int)
        if len(labels) != expected_length or subset["within_split_order"].tolist() != list(range(1, expected_length + 1)):
            raise ValueError("Reference-CNN Patient_manifest has an invalid {0} order or length".format(split))
        if (int((labels == 0).sum()), int((labels == 1).sum())) != (expected_zero, expected_one):
            raise ValueError("Reference-CNN Patient_manifest has invalid {0} class counts".format(split))
        labels_by_split[split] = labels
    return runs.sort_values("seed").reset_index(drop=True), labels_by_split, study_lock


def convert_reference_cnn_metrics2026_to_legacy(source_workbook: Path, output_workbook: Path) -> pd.DataFrame:
    """Write a separate old-layout sheet for the complete 100-seed reference-CNN workbook."""
    source_workbook = source_workbook.resolve()
    output_workbook = output_workbook.resolve()
    if source_workbook == output_workbook:
        raise ValueError("Output must be a separate file; never overwrite CNN_metrics2026.xlsx")
    runs, labels_by_split, _ = read_and_validate_reference_cnn_metrics2026(source_workbook)
    converted = pd.DataFrame([
        _legacy_row(row._asdict(), labels_by_split, freeze_feature_extractor=False)
        for row in runs.itertuples(index=False)
    ])
    if tuple(converted.columns) != LEGACY_COLUMNS:
        raise RuntimeError("Converted reference-CNN columns differ from the fixed legacy layout")
    if len(converted) != 100 or converted.duplicated(["model_name", "tag"]).any():
        raise RuntimeError("Converted reference-CNN workbook lacks one unique legacy row per seed")
    conversion_lock = pd.DataFrame([
        ("source_workbook", str(source_workbook)),
        ("source_design", "Seed-varied retained four-convolution reference CNN: one architecture x 100 prespecified random seeds"),
        ("output_purpose", "Visualisation compatibility with legacy metrics.xlsx plotting cells"),
        ("source_modified", "false"),
        ("prediction_precision", "Legacy-compatible sheet1 prediction strings are rounded to four decimals; source CNN_metrics2026.xlsx retains audited full precision"),
        ("statistical_boundary", "Do not use the compatibility copy for best-run, test-set selection, or run-level inferential analyses"),
        ("run_identifier_mapping", "legacy tag = new seed"),
    ], columns=["item", "value"])
    output_workbook.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(output_workbook), engine="openpyxl", mode="w") as writer:
        converted.to_excel(writer, sheet_name="sheet1", index=False)
        conversion_lock.to_excel(writer, sheet_name="Conversion_lock", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
    return converted
