#!/usr/bin/env python3
"""Synthetic integration test for the compact final-run SQLite and workbook outputs.

No PET data, model, TensorFlow fit, or real patient identifier is used. The test creates
five completed synthetic preflight seeds, validates safe resume behaviour, and exports
a workbook with the expected sheets.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from final_results_store import (
    connect_results_database,
    initialise_results_store,
    mark_run_started,
    record_completed_run,
    run_is_complete,
    validate_results_database,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def protocol() -> dict:
    return {
        "protocol_id": "hlr3e-05_flr1e-05_do30",
        "head_learning_rate": 3e-5,
        "fine_tune_learning_rate": 1e-5,
        "dropout_rate": 0.30,
        "input_size": 256,
        "batch_size": 5,
        "phase1_epochs": 30,
        "phase2_epochs": 30,
        "phase3_epochs": 30,
        "early_stopping_patience": 4,
        "threshold": 0.5,
        "optimizer": "Adam",
        "loss": "binary_crossentropy",
        "class_weights": False,
        "augmentation": "none",
    }


def metrics() -> dict:
    return {
        "accuracy": 1.0,
        "precision": 1.0,
        "sensitivity": 1.0,
        "specificity": 1.0,
        "f1": 1.0,
        "auc": 1.0,
        "tn": 26,
        "fp": 0,
        "fn": 0,
        "tp": 20,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="r1_final_store_test_") as temp:
        results_dir = Path(temp) / "R1_final_TL"
        batch = {
            "batch_name": "seeds_001_005",
            "preflight": True,
            "models": ["Xception"],
            "seeds": [1, 2, 3, 4, 5],
            "protocol": protocol(),
            "split_counts": {"train": 61, "validation": 31, "test": 46},
            "split_class_counts": {"train": {"0": 36, "1": 25}, "validation": {"0": 20, "1": 11}, "test": {"0": 26, "1": 20}},
            "runtime": {"git_commit": "synthetic", "tensorflow_version": "synthetic", "python_version": "synthetic"},
        }
        signature = {
            "protocol_file_sha256": "synthetic_protocol",
            "selection_provenance_sha256": "synthetic_provenance",
            "split_manifest_sha256": "synthetic_manifest",
            "protocol": protocol(),
            "split_counts": batch["split_counts"],
            "split_class_counts": batch["split_class_counts"],
            "code_commit": "synthetic",
            "tensorflow_version": "synthetic",
            "python_version": "synthetic",
        }
        labels = np.asarray([0] * 26 + [1] * 20)
        probabilities = np.asarray([0.10] * 26 + [0.90] * 20)
        test_rows = pd.DataFrame(
            {
                "patient_id": [f"synthetic_test_{index:03d}" for index in range(46)],
                "observed_label": labels,
                "probability": probabilities,
                "threshold": [0.5] * 46,
                "binary_prediction": labels,
            }
        )
        phase_rows = [
            {"phase": 1, "total_params": 1000, "trainable_params": 100, "non_trainable_params": 900, "selected_backbone_layers": 0, "learning_rate": 3e-5, "requested_epochs": 30, "actual_epochs": 8},
            {"phase": 2, "total_params": 1000, "trainable_params": 300, "non_trainable_params": 700, "selected_backbone_layers": 5, "learning_rate": 1e-5, "requested_epochs": 30, "actual_epochs": 7},
            {"phase": 3, "total_params": 1000, "trainable_params": 600, "non_trainable_params": 400, "selected_backbone_layers": 11, "learning_rate": 1e-5, "requested_epochs": 30, "actual_epochs": 6},
        ]
        connection = connect_results_database(results_dir)
        try:
            initialise_results_store(connection, batch_name="seeds_001_005", batch_manifest=batch, study_signature=signature)
            for seed in range(1, 6):
                mark_run_started(connection, model_name="Xception", seed=seed, batch_name="seeds_001_005")
                record_completed_run(
                    connection,
                    model_name="Xception",
                    seed=seed,
                    batch_name="seeds_001_005",
                    protocol=protocol(),
                    elapsed_seconds=12.3,
                    split_metrics={"train": metrics(), "validation": metrics(), "test": metrics()},
                    test_prediction_rows=test_rows,
                    phase_rows=phase_rows,
                )
            assert run_is_complete(connection, "Xception", 1, expected_test_patients=46)
            validation = validate_results_database(connection, models=["Xception"], seeds=[1, 2, 3, 4, 5], expected_test_patients=46)
            assert validation["summary"]["valid"], validation["summary"]
            assert validation["summary"]["complete_runs"] == 5
        finally:
            connection.close()

        validator_command = [
            sys.executable,
            "scripts/validate_final_results_store.py",
            "--results-dir",
            str(results_dir),
            "--models",
            "Xception",
            "--seeds",
            "1-5",
        ]
        validator = subprocess.run(validator_command, cwd=REPO_ROOT, text=True, capture_output=True)
        if validator.returncode != 0:
            raise RuntimeError(f"Structural validator failed:\n{validator.stdout}\n{validator.stderr}")

        workbook = results_dir / "synthetic_final.xlsx"
        command = [
            sys.executable,
            "scripts/export_final_results_workbook.py",
            "--results-dir",
            str(results_dir),
            "--output",
            str(workbook),
            "--models",
            "Xception",
            "--seeds",
            "1-5",
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Workbook export failed:\n{completed.stdout}\n{completed.stderr}")
        workbook_sheets = pd.ExcelFile(workbook).sheet_names
        expected_sheets = {
            "Run_metrics_1100",
            "Test_predictions",
            "Seed_coverage",
            "Protocol_and_split",
            "Phase_parameters",
            "Run_status",
            "Batches",
        }
        assert expected_sheets.issubset(workbook_sheets), workbook_sheets
        assert len(pd.read_excel(workbook, sheet_name="Run_metrics_1100")) == 5
        assert len(pd.read_excel(workbook, sheet_name="Test_predictions")) == 230
        print("Compact SQLite output and workbook-export test passed.")


if __name__ == "__main__":
    main()
