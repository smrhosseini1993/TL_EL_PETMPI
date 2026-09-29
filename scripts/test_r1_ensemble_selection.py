#!/usr/bin/env python3
"""Synthetic structural tests for the R1 development-only ensemble workflow."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.r1_ensemble_core import (
    RULE_ORDER,
    architecture_oof_ranking,
    candidate_pools,
    configuration_ranking,
    selection_lock,
    validate_oof_predictions,
    write_json,
)

MODELS = (
    "VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "InceptionV3",
    "InceptionResNetV2", "DenseNet169", "DenseNet201", "MobileNetV2", "Xception",
)


def oof_fixture() -> pd.DataFrame:
    """Small deterministic fixture; it is used only to validate code structure."""
    labels = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
    rows = []
    for model_index, model_name in enumerate(MODELS):
        for patient_index, label in enumerate(labels):
            base = 0.15 if label == 0 else 0.85
            # The architecture-specific offset preserves [0, 1] values while ensuring
            # every candidate rule receives correctly aligned probabilities.
            offset = (model_index - 5) * 0.002
            rows.append({
                "patient_id": "fixture_{0:03d}".format(patient_index),
                "fold": patient_index % 5 + 1,
                "model_name": model_name,
                "observed_label": label,
                "probability": base + offset,
            })
    return pd.DataFrame(rows)


def test_core_selection() -> None:
    oof = validate_oof_predictions(oof_fixture(), expected_models=MODELS)
    assert len(oof) == len(MODELS) * 10
    ranking = architecture_oof_ranking(oof)
    assert len(ranking) == len(MODELS)
    assert ranking["development_rank"].tolist() == list(range(1, len(MODELS) + 1))
    pools = candidate_pools(ranking)
    assert len(pools["Top-3"]) == 3
    assert len(pools["Top-5"]) == 5
    configuration_table, predictions, weights = configuration_ranking(oof, pools)
    assert len(configuration_table) == 10
    assert set(configuration_table["rule_id"]) == set(RULE_ORDER)
    assert set(configuration_table["pool_name"]) == {"Top-3", "Top-5"}
    assert len(predictions) == 10 * 10
    assert predictions.groupby("configuration_id")["patient_id"].nunique().eq(10).all()
    weighted_rows = configuration_table.loc[configuration_table["rule_id"] == "weighted_sum"]
    assert weighted_rows["weights_source"].eq("pooled development OOF AUC").all()
    for configuration_id, values in weights.items():
        if configuration_id.endswith("weighted_sum"):
            assert abs(sum(values.values()) - 1.0) < 1e-12

    with tempfile.TemporaryDirectory(prefix="r1_ensemble_selection_") as temporary:
        output = Path(temporary)
        oof_path = output / "oof_predictions_all.csv"
        manifest_path = output / "cv_run_manifest.json"
        oof.to_csv(oof_path, index=False)
        write_json(manifest_path, {"test_data_accessed": False, "partial_technical_preflight": False})
        lock = selection_lock(ranking, configuration_table, pools, weights, oof_path, manifest_path)
        assert lock["test_data_accessed"] is False
        assert lock["n_candidate_configurations"] == 10
        assert lock["selected_configuration"]["configuration_id"] == configuration_table.iloc[0]["configuration_id"]
        assert len(lock["selected_configuration"]["constituent_models"]) in (3, 5)


def test_cv_dry_run() -> None:
    with tempfile.TemporaryDirectory(prefix="r1_ensemble_cv_dry_") as temporary:
        temporary_root = Path(temporary)
        training = temporary_root / "data" / "training"
        training.mkdir(parents=True)
        for index in range(92):
            (training / "patient_{0:03d}.jpg".format(index)).touch()
        labels = [0] * 56 + [1] * 36
        (training / "ica_lables.txt").write_text("\n".join(str(value) for value in labels) + "\n", encoding="utf-8")
        output_dir = temporary_root / "secure_output"
        command = [
            sys.executable,
            str(REPO_ROOT / "Experiment2026" / "ensemble_selection_cv_2026.py"),
            "--root", str(temporary_root),
            "--output-dir", str(output_dir),
            "--models", "VGG16",
            "--allow-partial",
            "--dry-run",
        ]
        completed = subprocess.run(command, cwd=str(REPO_ROOT / "Experiment2026"), check=True, text=True, capture_output=True)
        assert "Test data is deliberately not loaded" in completed.stdout
        manifest_path = output_dir / "manifests" / "cv_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["test_data_accessed"] is False
        assert manifest["partial_technical_preflight"] is True
        fold_manifest = pd.read_csv(output_dir / "manifests" / "development_fold_manifest.csv")
        assert len(fold_manifest) == 92
        assert fold_manifest["patient_id"].nunique() == 92
        assert fold_manifest["fold"].nunique() == 5


def test_notebook_content() -> None:
    notebook = json.loads((REPO_ROOT / "notebooks" / "R1_ensemble_configuration_selection.ipynb").read_text(encoding="utf-8"))
    content = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Borda Count" in content  # Explicitly documents its exclusion.
    assert "Borda Count is deliberately excluded" in content
    assert "10 configurations" in content
    assert "WRITE_SELECTION_LOCK" in content
    assert "metrics2026.xlsx" not in content
    assert "test cohort" in content


def main() -> None:
    test_core_selection()
    test_cv_dry_run()
    test_notebook_content()
    print("R1 development-only ensemble-selection tests passed.")


if __name__ == "__main__":
    main()
