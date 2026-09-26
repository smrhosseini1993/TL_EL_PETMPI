#!/usr/bin/env python3
"""Synthetic no-PET validation for the R1 TL Sections 0–6 analysis functions."""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.r1_tl_results import (
    AnalysisSettings,
    R1Results,
    aggregate_patient_probabilities,
    calibration_summary,
    decision_curve_data,
    final_confusion_table,
    integrity_table,
    patient_metric_table,
    phase_audit_table,
    load_r1_results,
    plot_calibration,
    plot_confusion_tradeoff,
    plot_dca_overview,
    plot_dca_uncertainty,
    plot_phase_epochs,
    plot_roc,
    plot_runtime_by_model,
    plot_sensitivity_specificity,
    plot_stability_boxplots,
    runtime_table,
    seed_confusion_table,
    seed_stability_main_table,
    validate_r1_results,
)

MODELS = ("VGG16", "Xception")
SEEDS = tuple(range(1, 101))


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= 0.5).astype(int)
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    tp = int(((predicted == 1) & (labels == 1)).sum())
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / max(1, tp + fp),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "auc": 0.85,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def synthetic_results() -> R1Results:
    rng = np.random.default_rng(42)
    labels = np.asarray([0] * 26 + [1] * 20)
    patient_ids = [f"synthetic_{index:03d}" for index in range(46)]
    status_rows, metric_rows, prediction_rows, phase_rows = [], [], [], []
    for model_index, model_name in enumerate(MODELS):
        for seed in SEEDS:
            base = np.where(labels == 1, 0.70 + model_index * 0.02, 0.25 - model_index * 0.01)
            probabilities = np.clip(base + rng.normal(0, 0.10, size=46), 0.01, 0.99)
            metrics = binary_metrics(labels, probabilities)
            status_rows.append({"model_name": model_name, "seed": seed, "batch_name": "synthetic", "status": "completed"})
            metric_row = {
                "model_name": model_name, "seed": seed, "protocol_id": "synthetic", "head_learning_rate": 3e-5,
                "fine_tune_learning_rate": 1e-5, "dropout_rate": 0.30, "input_size": 256, "batch_size": 5,
                "phase1_epochs": 30, "phase2_epochs": 30, "phase3_epochs": 30, "early_stopping_patience": 4,
                "threshold": 0.5, "optimizer": "Adam", "loss": "binary_crossentropy", "class_weights": 0,
                "augmentation": "none", "elapsed_seconds": 40 + model_index * 15 + seed / 100,
            }
            for split in ("train", "validation", "test"):
                for name, value in metrics.items():
                    metric_row[f"{split}_{name}"] = value
            metric_rows.append(metric_row)
            for patient_id, label, probability in zip(patient_ids, labels, probabilities):
                prediction_rows.append({"patient_id": patient_id, "model_name": model_name, "seed": seed, "observed_label": int(label), "probability": float(probability)})
            for phase, trainable in ((1, 1000), (2, 2000), (3, 3000)):
                phase_rows.append({"model_name": model_name, "seed": seed, "phase": phase, "total_params": 5000, "trainable_params": trainable, "non_trainable_params": 5000 - trainable, "selected_backbone_layers": phase * 2, "learning_rate": 3e-5 if phase == 1 else 1e-5, "requested_epochs": 30, "actual_epochs": 5 + phase})
    return R1Results(
        run_status=pd.DataFrame(status_rows),
        run_metrics=pd.DataFrame(metric_rows),
        test_predictions=pd.DataFrame(prediction_rows),
        phase_parameters=pd.DataFrame(phase_rows),
        batches=pd.DataFrame([{"batch_name": "synthetic", "manifest_json": json.dumps({"split_counts": {"train": 61, "validation": 31, "test": 46}, "split_class_counts": {"train": {"0": 36, "1": 25}, "validation": {"0": 20, "1": 11}, "test": {"0": 26, "1": 20}}}), "created_at": "synthetic"}]),
        metadata={},
    )


def test_sqlite_loader() -> None:
    """Exercise the read-only SQLite path, including named metadata columns."""
    with tempfile.TemporaryDirectory(prefix="r1_tl_loader_test_") as temporary:
        output = Path(temporary)
        database = output / "r1_final_tl_runs.sqlite"
        connection = sqlite3.connect(str(database))
        try:
            connection.executescript(
                """
                CREATE TABLE metadata (key TEXT, value TEXT);
                CREATE TABLE batches (batch_name TEXT, manifest_json TEXT, created_at TEXT);
                CREATE TABLE run_status (model_name TEXT, seed INTEGER, status TEXT);
                CREATE TABLE run_metrics (model_name TEXT, seed INTEGER);
                CREATE TABLE test_predictions (patient_id TEXT, model_name TEXT, seed INTEGER, observed_label INTEGER, probability REAL);
                CREATE TABLE phase_parameters (model_name TEXT, seed INTEGER, phase INTEGER);
                """
            )
            connection.execute("INSERT INTO metadata VALUES (?, ?)", ("loader_test", json.dumps({"ok": True})))
            connection.execute("INSERT INTO batches VALUES (?, ?, ?)", ("synthetic", "{}", "synthetic"))
            statuses, metrics, predictions, phases = [], [], [], []
            for model_name in (
                "VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "InceptionV3",
                "InceptionResNetV2", "DenseNet169", "DenseNet201", "MobileNetV2", "Xception",
            ):
                for seed in range(1, 101):
                    statuses.append((model_name, seed, "completed"))
                    metrics.append((model_name, seed))
                    phases.extend((model_name, seed, phase) for phase in (1, 2, 3))
                    predictions.extend((f"patient_{patient:03d}", model_name, seed, patient % 2, 0.5) for patient in range(46))
            connection.executemany("INSERT INTO run_status VALUES (?, ?, ?)", statuses)
            connection.executemany("INSERT INTO run_metrics VALUES (?, ?)", metrics)
            connection.executemany("INSERT INTO phase_parameters VALUES (?, ?, ?)", phases)
            connection.executemany("INSERT INTO test_predictions VALUES (?, ?, ?, ?, ?)", predictions)
            connection.commit()
        finally:
            connection.close()
        loaded = load_r1_results(output)
        assert loaded.metadata["loader_test"] == {"ok": True}
        assert len(loaded.run_metrics) == 1100


def main() -> None:
    test_sqlite_loader()
    result = synthetic_results()
    validate_r1_results(result, expected_models=MODELS, expected_seeds=SEEDS)
    settings = AnalysisSettings(metric_bootstrap_iterations=100, dca_bootstrap_iterations=100, calibration_bins=5)
    aggregated = aggregate_patient_probabilities(result.test_predictions, settings)
    assert len(aggregated) == len(MODELS) * 46
    assert aggregated["n_seeds"].eq(100).all()
    metric_table = patient_metric_table(aggregated, settings)
    assert len(metric_table) == len(MODELS)
    assert metric_table.filter(regex="_point$").apply(lambda column: column.between(0, 1).all()).all()
    assert len(final_confusion_table(aggregated, settings)) == len(MODELS)
    assert len(calibration_summary(aggregated, settings)) == len(MODELS)
    dca = decision_curve_data(aggregated, settings)
    assert set(MODELS).issubset(set(dca["model_name"]))
    assert len(integrity_table(result)) > 5
    assert len(seed_stability_main_table(result.run_metrics)) == len(MODELS)
    assert len(seed_confusion_table(result.run_metrics)) == len(MODELS)
    assert len(runtime_table(result.run_metrics)) == len(MODELS)
    assert len(phase_audit_table(result.phase_parameters)) == len(MODELS) * 3
    with tempfile.TemporaryDirectory(prefix="r1_tl_analysis_test_") as temporary:
        figures = Path(temporary)
        plot_stability_boxplots(result.run_metrics, figures / "stability.png")
        plot_sensitivity_specificity(result.run_metrics, figures / "tradeoff.png")
        plot_confusion_tradeoff(result.run_metrics, figures / "errors.png")
        plot_runtime_by_model(result.run_metrics, figures / "runtime.png")
        plot_phase_epochs(result.phase_parameters, figures / "epochs.png")
        plot_roc(aggregated, figures / "roc.png")
        plot_calibration(aggregated, settings, figures / "calibration.png")
        plot_dca_overview(dca, figures / "dca.png")
        plot_dca_uncertainty(dca, figures / "dca_uncertainty.png")
        assert len(list(figures.glob("*.png"))) == 9
        assert all(path.stat().st_size > 1000 for path in figures.glob("*.png"))
    print("Synthetic R1 TL Sections 0–6 analysis test passed.")


if __name__ == "__main__":
    main()
