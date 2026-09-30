"""Adapt the submitted legacy ``metrics.xlsx`` for an ensemble-rule reproduction check.

This is deliberately a narrow verification adapter, not a final-analysis converter.  The
legacy file stores 12 architectures x 100 *row-ordered* runs in one sheet.  Its ``tag``
values are not unique for several architectures, while the historic ensemble notebook
matched runs by their position within each architecture's 100 rows.  This adapter
therefore maps that position to ``seed`` 1--100 and retains the source tag separately.

The submitted manuscript reported 11 architectures.  The legacy workbook additionally
contains EfficientNetB0, which is excluded from the reproduction workbook because it
was not part of the manuscript's 11-architecture TL/EL comparison.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


MANUSCRIPT_MODELS: Tuple[str, ...] = (
    "VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "InceptionV3",
    "InceptionResNetV2", "DenseNet169", "DenseNet201", "MobileNetV2", "Xception",
)
LEGACY_EXCLUDED_MODELS: Tuple[str, ...] = ("EfficientNetB0",)
REQUIRED_LEGACY_COLUMNS: Tuple[str, ...] = (
    "model_name", "elapsed_time", "input_size", "batch_size", "optimizer",
    "class_weights", "train_accuracy", "train_precision", "train_recall",
    "train_f1score", "train_confusion_matrix", "test_accuracy", "test_precision",
    "test_recall", "test_f1score", "test_confusion_matrix", "predicts",
    "test_specificity", "tag",
)
METRIC_COLUMNS: Tuple[str, ...] = (
    "accuracy", "precision", "sensitivity", "specificity", "f1", "auc",
    "tn", "fp", "fn", "tp", "probabilities", "binary_predictions",
)


def _canonical_columns() -> List[str]:
    columns = [
        "model_name", "seed", "legacy_run_position", "legacy_tag", "legacy_source_row",
        "elapsed_seconds", "input_size", "batch_size", "optimizer", "learning_rate",
        "loss", "dropout_rate", "threshold", "class_weights_enabled", "class_weight_0",
        "class_weight_1", "augmentation", "preprocessing",
    ]
    for phase in (1, 2, 3):
        columns.extend([
            "phase{0}_requested_epochs".format(phase),
            "phase{0}_actual_epochs".format(phase),
            "phase{0}_total_params".format(phase),
            "phase{0}_trainable_params".format(phase),
            "phase{0}_non_trainable_params".format(phase),
            "phase{0}_selected_layers".format(phase),
        ])
    for split in ("train", "validation", "test"):
        columns.extend(["{0}_{1}".format(split, metric) for metric in METRIC_COLUMNS])
    return columns


CANONICAL_COLUMNS: Tuple[str, ...] = tuple(_canonical_columns())


def read_legacy_workbook(path: Path) -> pd.DataFrame:
    """Read and validate the submitted one-sheet legacy metrics workbook."""
    if not path.exists():
        raise FileNotFoundError("Legacy metrics workbook not found: {0}".format(path))
    sheets = pd.read_excel(str(path), sheet_name=None, engine="openpyxl")
    if len(sheets) != 1:
        raise ValueError("Legacy metrics workbook must contain exactly one worksheet")
    runs = next(iter(sheets.values())).copy()
    missing = sorted(set(REQUIRED_LEGACY_COLUMNS) - set(runs.columns))
    if missing:
        raise ValueError("Legacy metrics workbook lacks columns: {0}".format(missing))
    runs["model_name"] = runs["model_name"].astype(str)
    expected = set(MANUSCRIPT_MODELS) | set(LEGACY_EXCLUDED_MODELS)
    observed = set(runs["model_name"])
    if observed != expected or len(runs) != 1200:
        raise ValueError(
            "Expected the submitted 1,200-row legacy workbook with 11 manuscript models plus EfficientNetB0; "
            "found {0} rows and models {1}".format(len(runs), sorted(observed))
        )
    counts = runs.groupby("model_name", sort=False).size()
    if not counts.eq(100).all():
        raise ValueError("Every legacy architecture must contain exactly 100 rows")
    return runs.reset_index(drop=True)


def read_test_labels(path: Path) -> np.ndarray:
    """Load the ordered 46-label historic test vector."""
    if not path.exists():
        raise FileNotFoundError("Legacy test-label file not found: {0}".format(path))
    labels = np.loadtxt(str(path), dtype=np.int64, ndmin=1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(labels) != 46 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Legacy test-label file must contain exactly 46 binary labels")
    if (int((labels == 0).sum()), int((labels == 1).sum())) != (26, 20):
        raise ValueError("Legacy test labels must have class 0=26 and class 1=20")
    return labels


def parse_probability_csv(value: Any) -> np.ndarray:
    """Parse one legacy 46-value, four-decimal test-probability string."""
    text = str(value).strip().strip("[](){} ")
    values = [part.strip() for part in text.split(",") if part.strip()]
    probabilities = np.asarray([float(part) for part in values], dtype=float)
    if len(probabilities) != 46 or not np.isfinite(probabilities).all() or not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all():
        raise ValueError("Every legacy prediction string must contain 46 finite probabilities in [0, 1]")
    return probabilities


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Mapping[str, Any]:
    predictions = np.round(probabilities).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "probabilities": ",".join(format(float(value), ".4f") for value in probabilities),
        "binary_predictions": ",".join(str(int(value)) for value in predictions),
    }


def _parse_confusion_matrix(value: Any) -> Tuple[int, int, int, int]:
    """Parse the historical ``[[TN FP] [FN TP]]`` string representation."""
    values = [int(number) for number in re.findall(r"-?\d+", str(value))]
    if len(values) != 4 or any(number < 0 for number in values):
        raise ValueError("Legacy confusion matrix must contain TN, FP, FN and TP")
    return values[0], values[1], values[2], values[3]


def _legacy_test_values(source: Mapping[str, Any], recomputed: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep source metrics while using stored probabilities for rule reproduction.

    Four source rows contain a probability serialized as exactly ``0.5000``. The
    old scalar metrics were calculated before four-decimal rounding, so their hard
    classifications cannot always be reconstructed from the saved strings. We retain
    those historical values here; the later ensemble check operates only on the saved
    probability strings, just as the historical ensemble notebook did.
    """
    tn, fp, fn, tp = _parse_confusion_matrix(source["test_confusion_matrix"])
    values = {
        "accuracy": float(source["test_accuracy"]),
        "precision": float(source["test_precision"]),
        "sensitivity": float(source["test_recall"]),
        "specificity": float(source["test_specificity"]),
        "f1": float(source["test_f1score"]),
        "auc": float(recomputed["auc"]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "probabilities": str(recomputed["probabilities"]),
        "binary_predictions": str(recomputed["binary_predictions"]),
    }
    return values


def _source_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _legacy_train_values(source: Mapping[str, Any]) -> Mapping[str, Any]:
    """Map source train metrics while retaining their historical 92-patient semantics."""
    return {
        "accuracy": float(source["train_accuracy"]),
        "precision": float(source["train_precision"]),
        "sensitivity": float(source["train_recall"]),
        "f1": float(source["train_f1score"]),
    }


def convert_legacy_metrics2025(legacy_runs: pd.DataFrame, test_labels: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a current three-sheet workbook representation of old TL test predictions.

    ``seed`` is deliberately the one-based row position per architecture.  It does
    **not** reproduce the non-unique old ``tag`` field.  This preserves the same
    pairing used by the old ensemble notebook's reset-index logic.
    """
    labels = np.asarray(test_labels, dtype=int).reshape(-1)
    if len(labels) != 46 or (int((labels == 0).sum()), int((labels == 1).sum())) != (26, 20):
        raise ValueError("Expected the ordered 46-patient historic test label vector")
    missing = sorted(set(REQUIRED_LEGACY_COLUMNS) - set(legacy_runs.columns))
    if missing:
        raise ValueError("Legacy run table lacks columns: {0}".format(missing))

    rows: List[Dict[str, Any]] = []
    for model_name in MANUSCRIPT_MODELS:
        model_runs = legacy_runs.loc[legacy_runs["model_name"].astype(str) == model_name].copy()
        if len(model_runs) != 100:
            raise ValueError("Legacy source must contain exactly 100 rows for {0}".format(model_name))
        # Preserve original source row order: that is how old ensemble runs were paired.
        for run_position, source_tuple in enumerate(model_runs.itertuples(index=True), start=1):
            source = source_tuple._asdict()
            probabilities = parse_probability_csv(source["predicts"])
            recomputed_metrics = _binary_metrics(labels, probabilities)
            legacy_test_metrics = _legacy_test_values(source, recomputed_metrics)
            row: Dict[str, Any] = OrderedDict((column, None) for column in CANONICAL_COLUMNS)
            row.update({
                "model_name": model_name,
                "seed": int(run_position),
                "legacy_run_position": int(run_position),
                "legacy_tag": str(source["tag"]),
                "legacy_source_row": int(source["Index"]) + 2,
                "elapsed_seconds": float(source["elapsed_time"]),
                "input_size": int(source["input_size"]),
                "batch_size": int(source["batch_size"]),
                "optimizer": str(source["optimizer"]),
                "threshold": 0.50,
                "class_weights_enabled": _source_bool(source["class_weights"]),
                "phase1_requested_epochs": int(source.get("epochs", 0)),
                "phase2_requested_epochs": int(source.get("epochs_partial_1", 0)),
                "phase3_requested_epochs": int(source.get("epochs_partial_2", 0)),
                "augmentation": "not recorded in legacy metrics.xlsx",
                "preprocessing": "not recorded in legacy metrics.xlsx",
            })
            for metric_name, value in _legacy_train_values(source).items():
                row["train_{0}".format(metric_name)] = value
            # Legacy training predictions and validation data were not saved in this file.
            for metric_name, value in legacy_test_metrics.items():
                row["test_{0}".format(metric_name)] = value
            rows.append(row)

    converted = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    converted = converted.sort_values(["model_name", "seed"]).reset_index(drop=True)
    if len(converted) != 1100 or converted.duplicated(["model_name", "seed"]).any():
        raise RuntimeError("Converted workbook must contain exactly 11 architectures x 100 position-matched rows")

    manifest = pd.DataFrame([
        {
            "split": "test",
            "within_split_order": index,
            "file_name": "legacy_test_order_{0:03d}.jpg".format(index),
            "relative_path": "not recorded in legacy metrics.xlsx",
            "observed_label": int(label),
        }
        for index, label in enumerate(labels, start=1)
    ])
    lock_rows = [
        ("study", "Legacy 2025 ensemble-rule reproduction check; not a final revision analysis input"),
        ("source_workbook", "metrics.xlsx (submitted legacy one-sheet workbook)"),
        ("source_shape", "1,200 rows: 12 architectures x 100 source rows"),
        ("retained_models", ";".join(MANUSCRIPT_MODELS)),
        ("excluded_source_model", "EfficientNetB0; present in source but not in the submitted manuscript's 11-architecture TL/EL comparison"),
        ("run_identifier", "seed = one-based row position within each architecture; reproduces legacy ensemble reset-index matching"),
        ("legacy_tag", "Preserved row-wise as legacy_tag only; source tags are non-unique and must not be used to pair runs"),
        ("test_patient_identity", "Original metrics.xlsx did not retain file names. Synthetic ordered identifiers preserve the source prediction/label order only."),
        ("test_label_source", "ordered 46-value historic ica_lables.txt; class 0=26, class 1=20"),
        ("prediction_precision", "exact four-decimal probability strings preserved from source predicts column"),
        ("source_metric_preservation", "Historical ACC, PRE, SEN, SPE, F1 and confusion-matrix values are copied from legacy metrics.xlsx. AUC was not saved and is recomputed from stored probabilities."),
        ("saved_prediction_precision", "Rules operate on exact four-decimal probability strings saved in legacy predicts, as the historical ensemble notebook did."),
        ("rounding_limit", "Four source rows contain 0.5000 after serialization; their original full-precision threshold side is unavailable. This affects stored individual-model hard metrics only, not the direct saved-prediction rule reproduction."),
        ("allowed_use", "Regression check that current combination-rule implementation reproduces historical Top-5 Max Rule behavior; not for final manuscript inference"),
    ]
    study_lock = pd.DataFrame(lock_rows, columns=["item", "value"])
    return converted, manifest, study_lock


def write_metrics2025(output_path: Path, run_results: pd.DataFrame, manifest: pd.DataFrame, study_lock: pd.DataFrame) -> None:
    """Write the three-sheet legacy reproduction workbook without modifying source data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(output_path), engine="openpyxl", mode="w") as writer:
        run_results.to_excel(writer, sheet_name="Run_results", index=False)
        manifest.to_excel(writer, sheet_name="Patient_manifest", index=False)
        study_lock.to_excel(writer, sheet_name="Study_lock", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)


def convert_workbooks(legacy_workbook: Path, test_labels_file: Path, output_workbook: Path) -> Mapping[str, Any]:
    """Perform the complete conversion and return a concise provenance summary."""
    legacy_runs = read_legacy_workbook(legacy_workbook)
    labels = read_test_labels(test_labels_file)
    runs, manifest, lock = convert_legacy_metrics2025(legacy_runs, labels)
    write_metrics2025(output_workbook, runs, manifest, lock)
    return {
        "output": str(output_workbook),
        "source_rows": int(len(legacy_runs)),
        "output_rows": int(len(runs)),
        "models": int(runs["model_name"].nunique()),
        "seeds_per_model": 100,
        "test_patients": int(len(manifest)),
        "excluded_source_model": ";".join(LEGACY_EXCLUDED_MODELS),
    }
