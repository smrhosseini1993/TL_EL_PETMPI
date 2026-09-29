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
EXPERIMENT_DIRECTORY = REPO_ROOT / "Experiment2026"
if str(EXPERIMENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

from analysis.r1_ensemble_core import (
    RULE_ORDER,
    architecture_oof_ranking,
    candidate_pools,
    configuration_ranking,
    create_run_matched_ensemble_predictions,
    load_historic_tl_workbook,
    patient_level_summary,
    raw_ensemble_run_results,
    raw_ensemble_study_lock,
    run_stability_summary,
    selection_lock,
    validate_oof_predictions,
    validate_manual_configuration,
    write_json,
)
from ensemble_selection_cv_2026 import write_cv_results_workbook

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


def test_cv_results_workbook() -> None:
    with tempfile.TemporaryDirectory(prefix="r1_ensemble_cv_workbook_") as temporary:
        output = Path(temporary)
        workbook = write_cv_results_workbook(
            output_dir=output,
            run_manifest={"test_data_accessed": False, "partial_technical_preflight": False},
            fold_manifest=pd.DataFrame({"patient_id": ["fixture_001"], "fold": [1], "observed_label": [0]}),
            all_predictions=pd.DataFrame({"patient_id": ["fixture_001"], "model_name": [MODELS[0]], "fold": [1], "observed_label": [0], "probability": [0.1]}),
            fold_metrics=pd.DataFrame({"model_name": [MODELS[0]], "fold": [1], "auc": [0.8]}),
            phase_parameters=pd.DataFrame({"model_name": [MODELS[0]], "fold": [1], "phase": [1]}),
        )
        sheets = pd.read_excel(workbook, sheet_name=None)
        assert {"Study_lock", "Fold_manifest", "OOF_predictions", "Fold_metrics", "Phase_parameters"}.issubset(sheets)
        assert "Architecture_ranking" not in sheets
        assert "Candidate_pools" not in sheets


def test_final_historic_split_utilities() -> None:
    top5 = list(MODELS[:5])
    configuration = validate_manual_configuration(
        top3_models=top5[:3],
        top5_models=top5,
        selected_pool="Top-5",
        selected_rule="max",
    )
    labels = np.asarray([0] * 26 + [1] * 20)
    rows = []
    for model_index, model_name in enumerate(top5):
        for seed in range(1, 101):
            for patient_index, label in enumerate(labels):
                base = 0.18 if label == 0 else 0.82
                probability = np.clip(base + model_index * 0.01 + (seed % 5) * 0.002, 0.01, 0.99)
                rows.append({
                    "patient_id": "test_{0:03d}".format(patient_index),
                    "observed_label": int(label),
                    "model_name": model_name,
                    "seed": seed,
                    "probability": float(probability),
                })
    ensemble_predictions, run_metrics = create_run_matched_ensemble_predictions(pd.DataFrame(rows), configuration)
    assert len(ensemble_predictions) == 100 * 46
    assert len(run_metrics) == 100
    assert ensemble_predictions.groupby("seed").size().eq(46).all()
    stability = run_stability_summary(run_metrics)
    assert stability.loc[0, "n_runs"] == 100
    patient_predictions, patient_metrics = patient_level_summary(
        ensemble_predictions, bootstrap_iterations=100, random_seed=42
    )
    assert len(patient_predictions) == 46
    assert patient_metrics.loc[0, "n_test_patients"] == 46
    assert patient_metrics.filter(regex="_point$").apply(lambda column: column.between(0, 1).all()).all()
    raw_runs = raw_ensemble_run_results(ensemble_predictions, run_metrics, configuration)
    assert len(raw_runs) == 100
    assert raw_runs["test_probabilities"].str.split(",").str.len().eq(46).all()
    raw_lock = raw_ensemble_study_lock(configuration, Path("synthetic_metrics2026.xlsx"))
    assert set(raw_lock["item"]).issuperset({"selected_pool", "selected_rule", "n_ensemble_runs"})

    with tempfile.TemporaryDirectory(prefix="r1_historic_workbook_") as temporary:
        temporary_path = Path(temporary)
        workbook = temporary_path / "metrics2026.xlsx"
        probability_csv = ",".join(["0.10"] * 26 + ["0.90"] * 20)
        run_rows = [
            {"model_name": model_name, "seed": seed, "test_probabilities": probability_csv}
            for model_name in MODELS for seed in range(1, 101)
        ]
        manifest_rows = [
            {"split": "test", "within_split_order": index + 1, "file_name": "test_{0:03d}.jpg".format(index), "observed_label": int(label)}
            for index, label in enumerate(labels)
        ]
        with pd.ExcelWriter(workbook, engine="openpyxl", mode="w") as writer:
            pd.DataFrame(run_rows).to_excel(writer, sheet_name="Run_results", index=False)
            pd.DataFrame(manifest_rows).to_excel(writer, sheet_name="Patient_manifest", index=False)
            pd.DataFrame([{"item": "synthetic", "value": "test"}]).to_excel(writer, sheet_name="Study_lock", index=False)
        loaded_predictions, loaded_manifest, loaded_lock = load_historic_tl_workbook(workbook, MODELS)
        assert len(loaded_predictions) == len(MODELS) * 100 * 46
        assert len(loaded_manifest) == 46
        assert len(loaded_lock) == 1


def test_notebook_content() -> None:
    notebook = json.loads((REPO_ROOT / "notebooks" / "R1_ensemble_configuration_selection.ipynb").read_text(encoding="utf-8"))
    content = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Borda Count" in content  # Explicitly documents its exclusion.
    assert "Borda Count is excluded" in content
    assert "10 configurations" in content
    assert "CV_WORKBOOK" in content
    assert "configuration_table" in content
    assert "TOP3_MODELS" in content
    assert "This notebook has not written a file" in content
    final_notebook = json.loads((REPO_ROOT / "notebooks" / "R1_final_ensemble_analysis.ipynb").read_text(encoding="utf-8"))
    final_content = "".join("".join(cell.get("source", [])) for cell in final_notebook["cells"])
    assert "metrics2026.xlsx" in final_content
    assert "TOP3_MODELS" in final_content
    assert "TOP5_MODELS" in final_content
    assert "SELECTED_RULE" in final_content
    assert "RUN_ANALYSIS = False" in final_content
    assert "R1_final_ensemble_raw_runs.xlsx" in final_content


def main() -> None:
    test_core_selection()
    test_cv_dry_run()
    test_cv_results_workbook()
    test_final_historic_split_utilities()
    test_notebook_content()
    print("R1 development-only ensemble-selection tests passed.")


if __name__ == "__main__":
    main()
