#!/usr/bin/env python3
"""Synthetic structural tests for the metrics2026 legacy-compatibility converter."""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.metrics2026_legacy_converter import EXPECTED_MODELS, LEGACY_COLUMNS, convert_metrics2026_to_legacy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probability_text(size: int, labels: np.ndarray) -> str:
    probabilities = np.where(labels == 1, 0.90, 0.10)
    assert len(probabilities) == size
    return ",".join(format(float(value), ".17g") for value in probabilities)


def metric_values(labels: np.ndarray) -> dict:
    probabilities = np.where(labels == 1, 0.90, 0.10)
    predicted = (probabilities >= 0.50).astype(int)
    tn = int(((labels == 0) & (predicted == 0)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    tp = int(((labels == 1) & (predicted == 1)).sum())
    return {
        "accuracy": 1.0, "precision": 1.0, "sensitivity": 1.0, "f1": 1.0,
        "auc": 1.0, "specificity": 1.0, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "probabilities": probability_text(len(labels), labels),
    }


def build_source_workbook(path: Path) -> None:
    labels = {
        "train": np.asarray([0] * 36 + [1] * 25),
        "validation": np.asarray([0] * 20 + [1] * 11),
        "test": np.asarray([0] * 26 + [1] * 20),
    }
    rows = []
    for model_name in EXPECTED_MODELS:
        for seed in range(1, 101):
            row = {
                "model_name": model_name, "seed": seed, "elapsed_seconds": 1.25,
                "input_size": 128, "batch_size": 10, "optimizer": "Adam",
                "class_weights_enabled": True,
                "phase1_requested_epochs": 100, "phase2_requested_epochs": 100, "phase3_requested_epochs": 100,
            }
            for split, values in labels.items():
                row.update({"{0}_{1}".format(split, key): value for key, value in metric_values(values).items()})
            rows.append(row)
    manifest_rows = []
    for split, values in labels.items():
        for order, label in enumerate(values, start=1):
            manifest_rows.append({
                "split": split, "within_split_order": order,
                "file_name": "{0}_{1:03d}.jpg".format(split, order), "observed_label": int(label),
            })
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Run_results", index=False)
        pd.DataFrame(manifest_rows).to_excel(writer, sheet_name="Patient_manifest", index=False)
        pd.DataFrame([{"item": "synthetic", "value": "test"}]).to_excel(writer, sheet_name="Study_lock", index=False)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="metrics2026_legacy_converter_") as temporary:
        directory = Path(temporary)
        source = directory / "metrics2026.xlsx"
        output = directory / "metrics_legacy_compatible.xlsx"
        build_source_workbook(source)
        before = sha256(source)
        converted = convert_metrics2026_to_legacy(source, output)
        after = sha256(source)
        assert before == after, "Source workbook was modified"
        assert len(converted) == 1100
        assert tuple(converted.columns) == LEGACY_COLUMNS
        assert converted.duplicated(["model_name", "tag"]).sum() == 0
        assert converted["tag"].astype(str).isin([str(seed) for seed in range(1, 101)]).all()
        assert converted["test_predicts"].str.split(",").str.len().eq(46).all()
        assert converted["val_predicts"].str.split(",").str.len().eq(31).all()
        assert converted["train_predicts"].str.split(",").str.len().eq(61).all()
        sheets = pd.read_excel(output, sheet_name=None, engine="openpyxl")
        assert set(sheets) == {"sheet1", "Conversion_lock"}
        assert len(sheets["sheet1"]) == 1100
        assert tuple(sheets["sheet1"].columns) == LEGACY_COLUMNS
        assert "statistical_boundary" in set(sheets["Conversion_lock"]["item"])
    print("metrics2026 legacy-compatibility converter test passed.")


if __name__ == "__main__":
    main()
