#!/usr/bin/env python3
"""Deterministic structural tests for the paper-order reporting workflow.

The fixture validates source schema alignment and output generation only. It is
not scientific analysis and it never represents study results.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.results_publish_core import (
    BASELINE_ORDER,
    TL_MODELS,
    ReportingSettings,
    load_all_sources,
    paired_delong,
    run_results_publish,
)


def probability_csv(labels: np.ndarray, offset: float = 0.0) -> str:
    values = np.where(labels == 1, 0.82 + offset, 0.18 + offset)
    values = np.clip(values, 0.01, 0.99)
    return ",".join(format(float(value), ".17g") for value in values)


def metric_columns(labels: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn = int(np.sum((labels == 0) & (prediction == 0)))
    fp = int(np.sum((labels == 0) & (prediction == 1)))
    fn = int(np.sum((labels == 1) & (prediction == 0)))
    tp = int(np.sum((labels == 1) & (prediction == 1)))
    return {
        "test_accuracy": float((prediction == labels).mean()),
        "test_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "test_sensitivity": float(tp / (tp + fn)),
        "test_specificity": float(tn / (tn + fp)),
        "test_f1": 1.0,
        "test_auc": 1.0,
        "test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp,
    }


def test_manifest(labels: np.ndarray, patient_field: str = "file_name") -> pd.DataFrame:
    return pd.DataFrame([
        {"split": "test", "within_split_order": index + 1, patient_field: "test_{0:03d}.jpg".format(index), "observed_label": int(label)}
        for index, label in enumerate(labels)
    ])


def build_tl_workbook(path: Path, labels: np.ndarray) -> None:
    rows = []
    for model_index, model in enumerate(TL_MODELS):
        for seed in range(1, 101):
            probability = np.clip(np.where(labels == 1, 0.80, 0.20) + model_index * 0.001 + (seed % 5) * 0.0001, 0.01, 0.99)
            row = {"model_name": model, "seed": seed, "test_probabilities": ",".join(format(float(value), ".17g") for value in probability)}
            row.update(metric_columns(labels, probability))
            rows.append(row)
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Run_results", index=False)
        test_manifest(labels).to_excel(writer, sheet_name="Patient_manifest", index=False)
        pd.DataFrame([{"item": "fixture", "value": "test"}]).to_excel(writer, sheet_name="Study_lock", index=False)


def build_ensemble_workbook(path: Path, labels: np.ndarray) -> None:
    rows = []
    for seed in range(1, 101):
        probability = np.clip(np.where(labels == 1, 0.88, 0.12) + (seed % 4) * 0.0001, 0.01, 0.99)
        row = {
            "seed": seed, "pool_name": "Top-5", "rule_id": "max", "rule_name": "Max Rule",
            "constituent_models": ";".join(TL_MODELS[:5]),
            "test_probabilities": ",".join(format(float(value), ".17g") for value in probability),
        }
        row.update(metric_columns(labels, probability))
        rows.append(row)
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Run_results", index=False)
        test_manifest(labels).to_excel(writer, sheet_name="Patient_manifest", index=False)
        pd.DataFrame([{"item": "fixture", "value": "test"}]).to_excel(writer, sheet_name="Study_lock", index=False)


def build_baseline_workbook(path: Path, labels: np.ndarray) -> None:
    rows = []
    for baseline_index, baseline_id in enumerate(BASELINE_ORDER):
        probability = np.clip(np.where(labels == 1, 0.72, 0.28) + baseline_index * 0.001, 0.01, 0.99)
        for order, (label, score) in enumerate(zip(labels, probability), start=1):
            rows.append({
                "baseline_id": baseline_id, "split": "test", "within_split_order": order,
                "observed_label": int(label), "score": float(score), "probability": float(score),
                "binary_prediction": int(score >= 0.5),
            })
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        pd.DataFrame([{"baseline_id": value} for value in BASELINE_ORDER]).to_excel(writer, sheet_name="Baseline_results", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Patient_predictions", index=False)
        test_manifest(labels, patient_field="clinical_id").assign(clinical_id=lambda frame: np.arange(1, len(frame) + 1)).to_excel(writer, sheet_name="Patient_manifest", index=False)
        pd.DataFrame([{"item": "fixture", "value": "test"}]).to_excel(writer, sheet_name="Study_lock", index=False)


def build_cnn_workbook(path: Path, labels: np.ndarray) -> None:
    rows = []
    for seed in range(1, 101):
        probability = np.clip(np.where(labels == 1, 0.76, 0.24) + (seed % 7) * 0.0001, 0.01, 0.99)
        rows.append({
            "model_name": "ReferenceCNN_4Conv_v2_55", "seed": seed,
            "test_probabilities": ",".join(format(float(value), ".17g") for value in probability),
        })
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Run_results", index=False)
        test_manifest(labels).to_excel(writer, sheet_name="Patient_manifest", index=False)
        pd.DataFrame([{"item": "fixture", "value": "reference-cnn-test"}]).to_excel(writer, sheet_name="Study_lock", index=False)


def test_complete_reporting_package() -> None:
    labels = np.asarray([0] * 26 + [1] * 20, dtype=int)
    settings = ReportingSettings(bootstrap_iterations=25, random_seed=42)
    with tempfile.TemporaryDirectory(prefix="results_publish_") as temporary:
        root = Path(temporary)
        tl = root / "metrics2026.xlsx"
        ensemble = root / "R1_final_ensemble_raw_runs.xlsx"
        baseline = root / "baseline_metrics2026.xlsx"
        cnn = root / "reference_cnn_runs.xlsx"
        build_tl_workbook(tl, labels)
        build_ensemble_workbook(ensemble, labels)
        build_baseline_workbook(baseline, labels)
        build_cnn_workbook(cnn, labels)
        sources = load_all_sources(
            tl_workbook=tl, baseline_workbook=baseline, ensemble_workbook=ensemble,
            reference_cnn_workbook=cnn, clinical_reader_file=None, settings=settings,
        )
        assert len(sources.tl_runs) == 1100
        assert len(sources.tl_patient_predictions) == 11 * 46
        assert len(sources.ensemble_runs) == 100
        assert len(sources.ensemble_patient_predictions) == 46
        assert len(sources.baseline_patient_predictions) == 6 * 46
        assert len(sources.reference_cnn_patient_predictions) == 46
        assert sources.reference_cnn_patient_predictions["binary_prediction"].isin([0, 1]).all()
        output = root / "output"
        tables = run_results_publish(sources, output, settings)
        assert {"T1_TL_performance", "T2_TL_confusion", "T3_principal_comparison", "T4_principal_confusion", "ST1_all_conventional", "ST2_TL_seed_stability", "ST3_paired_comparisons", "ST4_calibration"}.issubset(tables)
        assert len(tables["T1_TL_performance"]) == 11
        assert len(tables["ST1_all_conventional"]) == 6
        assert {"TP", "TN", "FP", "FN"}.issubset(tables["ST1_all_conventional"].columns)
        assert (output / "results_publish_tables.xlsx").exists()
        assert (output / "captions_and_results_text.md").exists()
        assert (output / "analysis_manifest.json").exists()
        for name in ("F1_TL_ROC.png", "F2_TL_DCA.png", "F3_final_comparison_DCA.png", "SF1_TL_seed_stability.png"):
            assert (output / "figures" / name).exists(), name


def test_delong_equal_scores() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    difference, z_value, p_value = paired_delong(labels, scores, scores)
    assert abs(difference) < 1e-12
    assert z_value == 0.0
    assert p_value == 1.0


def main() -> None:
    test_complete_reporting_package()
    test_delong_equal_scores()
    print("Results-publish workflow structural tests passed.")


if __name__ == "__main__":
    main()
