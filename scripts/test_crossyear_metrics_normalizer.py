#!/usr/bin/env python3
"""Structural tests for internal 2025-versus-2026 comparison normalization."""
from __future__ import print_function

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.crossyear_metrics_normalizer import normalize_manifest, normalize_run_results
from analysis.legacy_metrics2025_adapter import CANONICAL_COLUMNS, MANUSCRIPT_MODELS


def build_source_runs(labels):
    rows = []
    for model_index, model_name in enumerate(MANUSCRIPT_MODELS):
        for seed in range(1, 101):
            probabilities = np.where(labels == 1, 0.90009, 0.10001).astype(float)
            probabilities[0] = 0.50004 if seed == 1 and model_index == 0 else probabilities[0]
            binary = np.round(probabilities).astype(int)
            tn = int(((labels == 0) & (binary == 0)).sum())
            fp = int(((labels == 0) & (binary == 1)).sum())
            fn = int(((labels == 1) & (binary == 0)).sum())
            tp = int(((labels == 1) & (binary == 1)).sum())
            rows.append({
                "model_name": model_name, "seed": seed, "elapsed_seconds": 1.0,
                "input_size": 128, "batch_size": 10, "optimizer": "Adam",
                "test_probabilities": ",".join(format(float(value), ".17g") for value in probabilities),
                "test_binary_predictions": ",".join(str(int(value)) for value in binary),
                "test_accuracy": (tn + tp) / 46.0, "test_precision": tp / max(tp + fp, 1),
                "test_sensitivity": tp / 20.0, "test_specificity": tn / 26.0,
                "test_f1": 2 * tp / max(2 * tp + fp + fn, 1), "test_auc": 1.0,
                "test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp,
            })
    return pd.DataFrame(rows)


def main():
    labels = np.asarray([0] * 26 + [1] * 20, dtype=int)
    source = build_source_runs(labels)
    normalized, audit = normalize_run_results(source, labels, "four_decimal")
    assert list(normalized.columns[:len(CANONICAL_COLUMNS)]) == list(CANONICAL_COLUMNS)
    assert len(normalized) == 1100
    assert normalized["test_probabilities"].str.split(",").str.len().eq(46).all()
    assert normalized["test_probabilities"].str.split(",").str[0].eq("0.5000").sum() == 1
    assert normalized["source2026_test_probabilities_full_precision"].notna().all()
    assert len(audit) == 1100

    full_precision, full_audit = normalize_run_results(source, labels, "full_precision")
    source_row = source.loc[(source["model_name"] == "VGG16") & (source["seed"] == 1)].iloc[0]
    normalized_row = full_precision.loc[(full_precision["model_name"] == "VGG16") & (full_precision["seed"] == 1)].iloc[0]
    assert normalized_row["test_probabilities"] == source_row["test_probabilities"]
    assert not full_audit["any_test_metric_difference"].any()

    source_manifest = pd.DataFrame([
        {"split": "test", "within_split_order": index, "file_name": "source_{0}.jpg".format(index), "relative_path": "data/test/source_{0}.jpg".format(index), "observed_label": int(label)}
        for index, label in enumerate(labels, start=1)
    ])
    normalized_manifest, source_copy = normalize_manifest(source_manifest)
    assert len(normalized_manifest) == 46
    assert normalized_manifest["file_name"].iloc[0] == "legacy_test_order_001.jpg"
    assert normalized_manifest["source2026_file_name"].iloc[-1] == "source_46.jpg"
    assert source_copy.equals(source_manifest)
    print("Cross-year metrics normalizer structural tests passed.")


if __name__ == "__main__":
    main()
