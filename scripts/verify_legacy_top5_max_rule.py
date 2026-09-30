#!/usr/bin/env python3
"""Verify current Top-5 Max Rule behavior against the old row-order algorithm.

This utility is a regression check only. It compares the new shared ensemble code
against a direct implementation of the legacy notebook's algorithm:

1. reset each selected architecture's source rows to positions 1--100;
2. pair the same position across architectures;
3. take the maximum of the saved 46 class-1 probabilities;
4. threshold at >= 0.50; and
5. calculate the 100 descriptive metric rows.

It intentionally does *not* compare the reconstructed four-decimal source to claims
that depended on unavailable full-precision historical probabilities.
"""
from __future__ import print_function

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.legacy_metrics2025_adapter import MANUSCRIPT_MODELS, parse_probability_csv
from analysis.r1_ensemble_core import (
    create_run_matched_ensemble_predictions,
    load_historic_tl_workbook,
    raw_ensemble_run_results,
    raw_ensemble_study_lock,
    run_stability_summary,
    validate_manual_configuration,
)

TOP5_MODELS = (
    "InceptionResNetV2", "DenseNet201", "Xception", "MobileNetV2", "VGG19",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Top-5 Max Rule parity with the submitted legacy row-position logic.")
    parser.add_argument("--legacy-workbook", type=Path, required=True, help="Original one-sheet metrics.xlsx")
    parser.add_argument("--test-labels", type=Path, required=True, help="Ordered 46-value old ica_lables.txt")
    parser.add_argument("--metrics2025", type=Path, required=True, help="Converted three-sheet metrics2025.xlsx")
    parser.add_argument("--output-workbook", type=Path, required=True, help="New raw 100-run parity-proof workbook")
    return parser.parse_args()


def direct_legacy_top5_max(legacy_runs, labels):
    """Execute the old notebook's reset-index, row-position matching literally."""
    rows = []
    for position in range(100):
        probability_rows = []
        for model_name in TOP5_MODELS:
            model_runs = legacy_runs.loc[legacy_runs["model_name"].astype(str) == model_name].reset_index(drop=True)
            probability_rows.append(parse_probability_csv(model_runs.iloc[position]["predicts"]))
        score = np.vstack(probability_rows).max(axis=0)
        binary = (score >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, binary, labels=[0, 1]).ravel()
        rows.append({
            "seed": position + 1,
            "probability": score,
            "binary_prediction": binary,
            "accuracy": float(accuracy_score(labels, binary)),
            "precision": float(precision_score(labels, binary, zero_division=0)),
            "sensitivity": float(recall_score(labels, binary, zero_division=0)),
            "specificity": float(tn / (tn + fp)),
            "f1": float(f1_score(labels, binary, zero_division=0)),
            "auc": float(roc_auc_score(labels, score)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    return rows


def assert_exact_parity(direct_rows, ensemble_predictions, ensemble_metrics):
    metric_columns = ("accuracy", "precision", "sensitivity", "specificity", "f1", "auc", "tn", "fp", "fn", "tp")
    for direct in direct_rows:
        seed = int(direct["seed"])
        probabilities = ensemble_predictions.loc[ensemble_predictions["seed"].astype(int) == seed].sort_values("within_split_order")
        metrics = ensemble_metrics.loc[ensemble_metrics["seed"].astype(int) == seed].iloc[0]
        if not np.array_equal(probabilities["probability"].to_numpy(dtype=float), direct["probability"]):
            raise AssertionError("Probability-vector parity failed for source row position {0}".format(seed))
        if not np.array_equal(probabilities["binary_prediction"].to_numpy(dtype=int), direct["binary_prediction"]):
            raise AssertionError("Binary-call parity failed for source row position {0}".format(seed))
        for column in metric_columns:
            if not np.isclose(float(metrics[column]), float(direct[column]), rtol=0.0, atol=1e-15):
                raise AssertionError("Metric parity failed for seed {0}, {1}".format(seed, column))


def summary_table(summary):
    rows = []
    for metric in ("ACC", "PRE", "SEN", "SPE", "F1S", "AUC"):
        rows.append({
            "metric": metric,
            "median": float(summary[metric + "_median"]),
            "iqr": float(summary[metric + "_iqr"]),
            "display_2dp": "{0:.2f} ({1:.2f})".format(float(summary[metric + "_median"]), float(summary[metric + "_iqr"])),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    legacy_sheets = pd.read_excel(str(args.legacy_workbook), sheet_name=None, engine="openpyxl")
    if len(legacy_sheets) != 1:
        raise ValueError("Original legacy workbook must contain exactly one worksheet")
    legacy_runs = next(iter(legacy_sheets.values())).copy()
    labels = np.loadtxt(str(args.test_labels), dtype=int, ndmin=1).reshape(-1)
    if len(labels) != 46 or (int((labels == 0).sum()), int((labels == 1).sum())) != (26, 20):
        raise ValueError("Expected the ordered historic test vector with 26 class-0 and 20 class-1 labels")

    direct_rows = direct_legacy_top5_max(legacy_runs, labels)
    configuration = validate_manual_configuration(
        top3_models=TOP5_MODELS[:3],
        top5_models=TOP5_MODELS,
        selected_pool="Top-5",
        selected_rule="max",
    )
    predictions, manifest, _ = load_historic_tl_workbook(args.metrics2025, MANUSCRIPT_MODELS)
    ensemble_predictions, ensemble_metrics = create_run_matched_ensemble_predictions(predictions, configuration)
    assert_exact_parity(direct_rows, ensemble_predictions, ensemble_metrics)
    summary = run_stability_summary(ensemble_metrics).iloc[0]

    raw_runs = raw_ensemble_run_results(ensemble_predictions, ensemble_metrics, configuration)
    study_lock = raw_ensemble_study_lock(configuration, args.metrics2025)
    extra_lock = pd.DataFrame([
        {
            "item": "parity_verification",
            "value": "Passed: all 100 probability vectors, binary calls, six metrics, and TP/TN/FP/FN counts exactly equal a direct reconstruction of the historical notebook's row-position Top-5 Max Rule algorithm.",
        },
        {
            "item": "rounding_scope",
            "value": "The comparison uses the exact four-decimal stored probability strings from legacy metrics.xlsx. It validates rule parity, not unavailable full-precision historical outputs.",
        },
    ])
    study_lock = pd.concat([study_lock, extra_lock], ignore_index=True)
    args.output_workbook.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(args.output_workbook), engine="openpyxl", mode="w") as writer:
        raw_runs.to_excel(writer, sheet_name="Run_results", index=False)
        manifest.to_excel(writer, sheet_name="Patient_manifest", index=False)
        study_lock.to_excel(writer, sheet_name="Study_lock", index=False)
        summary_table(summary).to_excel(writer, sheet_name="Parity_summary", index=False)

    print("PASS: new shared rule code exactly reproduces all 100 legacy row-position Top-5 Max outputs.")
    print("Selected constituent models:", "; ".join(TOP5_MODELS))
    print("\nReproduced median (IQR) from saved four-decimal legacy probabilities:")
    for row in summary_table(summary).itertuples(index=False):
        print("{0}: {1}".format(row.metric, row.display_2dp))
    print("\nProof workbook:", args.output_workbook)


if __name__ == "__main__":
    main()
