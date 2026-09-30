#!/usr/bin/env python3
"""Structural tests for the narrow legacy ensemble-rule reproduction adapter."""
from __future__ import print_function

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.legacy_metrics2025_adapter import (
    LEGACY_EXCLUDED_MODELS,
    MANUSCRIPT_MODELS,
    convert_legacy_metrics2025,
)


def source_row(model_name, index, labels):
    probabilities = np.where(labels == 1, 0.90, 0.10).astype(float)
    predictions = np.round(probabilities).astype(int)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    return {
        "model_name": model_name, "elapsed_time": 1.0 + index, "input_size": 128,
        "batch_size": 5, "optimizer": "Adam", "class_weights": True,
        "train_accuracy": 0.90, "train_precision": 0.90, "train_recall": 0.90,
        "train_f1score": 0.90, "train_confusion_matrix": "[[1 0] [0 1]]",
        "test_accuracy": 1.0, "test_precision": 1.0, "test_recall": 1.0,
        "test_f1score": 1.0, "test_confusion_matrix": "[[26 0] [0 20]]",
        "predicts": ",".join("{0:.4f}".format(value) for value in probabilities),
        "test_specificity": 1.0, "tag": (index % 5) + 1,
    }


def main():
    labels = np.asarray([0] * 26 + [1] * 20, dtype=int)
    rows = []
    for model_name in list(MANUSCRIPT_MODELS) + list(LEGACY_EXCLUDED_MODELS):
        rows.extend(source_row(model_name, index, labels) for index in range(100))
    runs, manifest, lock = convert_legacy_metrics2025(pd.DataFrame(rows), labels)
    assert len(runs) == 1100
    assert set(runs["model_name"]) == set(MANUSCRIPT_MODELS)
    assert set(runs["seed"]) == set(range(1, 101))
    assert runs["legacy_tag"].astype(str).iloc[0] == "1"
    assert runs["test_probabilities"].str.split(",").str.len().eq(46).all()
    assert len(manifest) == 46
    assert set(lock["item"]).issuperset({"run_identifier", "legacy_tag", "allowed_use"})
    print("Legacy metrics2025 adapter structural tests passed.")


if __name__ == "__main__":
    main()
