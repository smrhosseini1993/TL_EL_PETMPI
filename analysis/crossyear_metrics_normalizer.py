"""Normalize final 2026 TL results to the internal 2025-comparison workbook contract.

The output is deliberately an internal comparison derivative.  It never replaces the
full-precision ``metrics2026.xlsx`` that is used for the final revision analysis.

The shared contract uses the 2025 adapter's 72 core Run_results columns and one
test-only 46-patient manifest with order-based synthetic identifiers. Two explicit
internal modes are supported:

* ``full_precision`` preserves actual 2026 test probabilities and metrics. Use this
  for a meaningful comparison of the old and new data through the same code.
* ``four_decimal`` emulates the historical saved 2025 prediction representation.
  Use only to test historical-rule sensitivity: rounding can create probability ties
  and materially change AUC even when hard classifications are unchanged.

Original 2026 full-precision test outputs and full source manifests/locks are retained
as extra columns/sheets in either mode for auditability.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from analysis.legacy_metrics2025_adapter import CANONICAL_COLUMNS, MANUSCRIPT_MODELS, parse_probability_csv


REQUIRED_RUN_COLUMNS: Tuple[str, ...] = (
    "model_name", "seed", "test_probabilities", "test_accuracy", "test_precision",
    "test_sensitivity", "test_specificity", "test_f1", "test_auc", "test_tn",
    "test_fp", "test_fn", "test_tp",
)
REQUIRED_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "split", "within_split_order", "file_name", "relative_path", "observed_label",
)
TEST_METRICS: Tuple[str, ...] = (
    "accuracy", "precision", "sensitivity", "specificity", "f1", "auc", "tn", "fp", "fn", "tp",
)


def _read_source_workbook(path: Path) -> Mapping[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError("2026 source workbook not found: {0}".format(path))
    sheets = pd.read_excel(str(path), sheet_name=None, engine="openpyxl")
    required_sheets = {"Run_results", "Patient_manifest", "Study_lock"}
    missing = sorted(required_sheets - set(sheets))
    if missing:
        raise ValueError("2026 source workbook lacks sheets: {0}".format(missing))
    missing_columns = sorted(set(REQUIRED_RUN_COLUMNS) - set(sheets["Run_results"].columns))
    if missing_columns:
        raise ValueError("2026 source Run_results lacks columns: {0}".format(missing_columns))
    missing_manifest = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(sheets["Patient_manifest"].columns))
    if missing_manifest:
        raise ValueError("2026 source Patient_manifest lacks columns: {0}".format(missing_manifest))
    return sheets


def _test_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    test = manifest.loc[manifest["split"].astype(str) == "test"].copy()
    test["within_split_order"] = pd.to_numeric(test["within_split_order"], errors="raise").astype(int)
    test["observed_label"] = pd.to_numeric(test["observed_label"], errors="raise").astype(int)
    test = test.sort_values("within_split_order").reset_index(drop=True)
    if len(test) != 46 or test["within_split_order"].tolist() != list(range(1, 47)):
        raise ValueError("2026 source manifest must contain 46 ordered test patients")
    if (int((test["observed_label"] == 0).sum()), int((test["observed_label"] == 1).sum())) != (26, 20):
        raise ValueError("2026 source test labels must have class 0=26 and class 1=20")
    return test


def _normalized_test_metrics(labels: Sequence[int], probabilities: np.ndarray) -> Mapping[str, Any]:
    """Use historical NumPy rounding after four-decimal serialization."""
    labels_array = np.asarray(labels, dtype=int).reshape(-1)
    predictions = np.round(probabilities).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels_array, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels_array, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "probabilities": ",".join(format(float(value), ".4f") for value in probabilities),
        "binary_predictions": ",".join(str(int(value)) for value in predictions),
    }


def _source_value(value: Any) -> Any:
    """Preserve Excel scalars without coercing deliberately blank source fields."""
    return value if not pd.isna(value) else None


def normalize_run_results(
    source_runs: pd.DataFrame,
    labels: Sequence[int],
    probability_representation: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create the exact metrics2025 core columns plus audited 2026-only values."""
    if probability_representation not in ("full_precision", "four_decimal"):
        raise ValueError("probability_representation must be full_precision or four_decimal")
    runs = source_runs.copy()
    runs["model_name"] = runs["model_name"].astype(str)
    runs["seed"] = pd.to_numeric(runs["seed"], errors="raise").astype(int)
    if len(runs) != 1100 or set(runs["model_name"]) != set(MANUSCRIPT_MODELS):
        raise ValueError("2026 source must contain exactly 11 manuscript architectures x 100 seeds")
    if runs.duplicated(["model_name", "seed"]).any():
        raise ValueError("2026 source contains duplicate model/seed rows")
    for model_name in MANUSCRIPT_MODELS:
        if set(runs.loc[runs["model_name"] == model_name, "seed"]) != set(range(1, 101)):
            raise ValueError("2026 source lacks seeds 1-100 for {0}".format(model_name))

    converted_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    labels_array = np.asarray(labels, dtype=int).reshape(-1)
    for source_tuple in runs.itertuples(index=True):
        source = source_tuple._asdict()
        full_precision = parse_probability_csv(source["test_probabilities"])
        if probability_representation == "four_decimal":
            comparison_probabilities = np.asarray([float(format(value, ".4f")) for value in full_precision], dtype=float)
            comparison_metrics = _normalized_test_metrics(labels_array, comparison_probabilities)
        else:
            comparison_probabilities = full_precision
            comparison_metrics = {
                "accuracy": float(source["test_accuracy"]),
                "precision": float(source["test_precision"]),
                "sensitivity": float(source["test_sensitivity"]),
                "specificity": float(source["test_specificity"]),
                "f1": float(source["test_f1"]),
                "auc": float(source["test_auc"]),
                "tn": int(source["test_tn"]), "fp": int(source["test_fp"]),
                "fn": int(source["test_fn"]), "tp": int(source["test_tp"]),
                "probabilities": str(source["test_probabilities"]),
                "binary_predictions": str(source["test_binary_predictions"]),
            }

        row: Dict[str, Any] = OrderedDict((column, None) for column in CANONICAL_COLUMNS)
        for column in CANONICAL_COLUMNS:
            if column in source:
                row[column] = _source_value(source[column])
        # These three fields exist only to make the core contract column-identical
        # to metrics2025. Unlike 2025, the 2026 seed identifiers are genuinely unique.
        row["legacy_run_position"] = int(source["seed"])
        row["legacy_tag"] = "not_applicable_2026_seed_{0}".format(int(source["seed"]))
        row["legacy_source_row"] = None
        for metric_name, value in comparison_metrics.items():
            row["test_{0}".format(metric_name)] = value

        # Preserve all affected original 2026 values rather than silently overwriting them.
        row["source2026_test_probabilities_full_precision"] = str(source["test_probabilities"])
        row["source2026_test_binary_predictions_full_precision"] = str(source["test_binary_predictions"])
        for metric_name in TEST_METRICS:
            row["source2026_test_{0}".format(metric_name)] = _source_value(source["test_{0}".format(metric_name)])
        row["comparison_test_probability_representation"] = probability_representation
        converted_rows.append(row)

        differences = {
            metric: float(comparison_metrics[metric]) - float(source["test_{0}".format(metric)])
            for metric in TEST_METRICS
        }
        audit_rows.append({
            "model_name": str(source["model_name"]),
            "seed": int(source["seed"]),
            "n_literal_0_5000_after_normalization": int(np.sum(comparison_probabilities == 0.50)),
            "any_test_metric_difference": bool(any(abs(value) > 1e-12 for value in differences.values())),
            **{"difference_test_{0}".format(metric): value for metric, value in differences.items()},
        })

    output = pd.DataFrame(converted_rows)
    expected_core = list(CANONICAL_COLUMNS)
    extras = [column for column in output.columns if column not in expected_core]
    output = output[expected_core + extras].sort_values(["model_name", "seed"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values(["model_name", "seed"]).reset_index(drop=True)
    return output, audit


def normalize_manifest(source_manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a 2025-compatible test-only manifest while preserving source identities."""
    test = _test_manifest(source_manifest)
    rows = []
    for source in test.itertuples(index=False):
        rows.append({
            "split": "test",
            "within_split_order": int(source.within_split_order),
            "file_name": "legacy_test_order_{0:03d}.jpg".format(int(source.within_split_order)),
            "relative_path": "comparison-normalized; source2026 path retained in source2026_relative_path",
            "observed_label": int(source.observed_label),
            "source2026_file_name": str(source.file_name),
            "source2026_relative_path": str(source.relative_path),
        })
    return pd.DataFrame(rows), source_manifest.copy()


def comparison_study_lock(
    source_lock: pd.DataFrame,
    audit: pd.DataFrame,
    probability_representation: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create comparison provenance and preserve the source lock as a separate sheet."""
    difference_count = int(audit["any_test_metric_difference"].sum())
    literal_half_rows = int((audit["n_literal_0_5000_after_normalization"] > 0).sum())
    rows = [
        ("study", "Internal 2025-versus-2026 TL comparison normalization; not a manuscript reporting input"),
        ("source_workbook", "metrics2026.xlsx"),
        ("source_design", "2026 fixed-split 11 architecture x 100 seed transfer-learning workbook"),
        ("core_schema", "Run_results first 72 columns match metrics2025.xlsx exactly, including 2025-only provenance placeholders"),
        ("test_manifest", "46 test rows only; generic legacy_test_order_001..046 IDs align with metrics2025 order. Original 2026 identities retained as source2026_* columns."),
        ("test_probability_representation", probability_representation),
        ("test_probability_normalization", "Full-precision source values retained in test_probabilities for actual comparison." if probability_representation == "full_precision" else "test_probabilities serialized to four decimals to emulate the historical saved 2025 representation. Original full-precision strings retained in source2026_test_probabilities_full_precision."),
        ("test_metrics", "Original full-precision 2026 test metrics retained in core test_* columns." if probability_representation == "full_precision" else "test ACC, PRE, SEN, SPE, F1, AUC, TN, FP, FN, TP recomputed from the normalized four-decimal vectors using the historical np.round hard-call convention. Original values retained in source2026_test_* columns."),
        ("rounding_audit", "{0} of 1,100 rows have a normalized test metric differing from the full-precision source; {1} rows contain literal 0.5000 after four-decimal normalization.".format(difference_count, literal_half_rows)),
        ("allowed_use", "Internal, like-for-like comparison with metrics2025 and legacy ensemble-rule checks. Do not substitute for full-precision metrics2026.xlsx in final revision reporting."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"]), source_lock.copy()


def write_normalized_workbook(
    output_path: Path,
    run_results: pd.DataFrame,
    manifest: pd.DataFrame,
    study_lock: pd.DataFrame,
    audit: pd.DataFrame,
    source_full_manifest: pd.DataFrame,
    source_full_lock: pd.DataFrame,
) -> None:
    """Write an internal comparison workbook without modifying source metrics2026."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(output_path), engine="openpyxl", mode="w") as writer:
        run_results.to_excel(writer, sheet_name="Run_results", index=False)
        manifest.to_excel(writer, sheet_name="Patient_manifest", index=False)
        study_lock.to_excel(writer, sheet_name="Study_lock", index=False)
        audit.to_excel(writer, sheet_name="Normalization_audit", index=False)
        source_full_manifest.to_excel(writer, sheet_name="Source2026_full_manifest", index=False)
        source_full_lock.to_excel(writer, sheet_name="Source2026_study_lock", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)


def normalize_workbook(
    source_path: Path,
    output_path: Path,
    probability_representation: str = "full_precision",
) -> Mapping[str, Any]:
    """Perform an auditable 2026-to-2025 internal comparison normalization."""
    source = _read_source_workbook(source_path)
    test = _test_manifest(source["Patient_manifest"])
    run_results, audit = normalize_run_results(
        source["Run_results"],
        test["observed_label"].to_numpy(dtype=int),
        probability_representation,
    )
    manifest, full_manifest = normalize_manifest(source["Patient_manifest"])
    study_lock, full_lock = comparison_study_lock(source["Study_lock"], audit, probability_representation)
    write_normalized_workbook(output_path, run_results, manifest, study_lock, audit, full_manifest, full_lock)
    return {
        "output": str(output_path),
        "probability_representation": probability_representation,
        "run_rows": int(len(run_results)),
        "core_columns": int(len(CANONICAL_COLUMNS)),
        "extra_columns": int(len(run_results.columns) - len(CANONICAL_COLUMNS)),
        "metric_difference_rows": int(audit["any_test_metric_difference"].sum()),
        "literal_half_rows": int((audit["n_literal_0_5000_after_normalization"] > 0).sum()),
    }
