#!/usr/bin/env python3
"""Development-only five-fold CV for revised ensemble-configuration selection.

This runner deliberately reads only ``root/data/training`` (92 development patients).
It uses the retained final Experiment2026 transfer-learning procedure, creates one
out-of-fold prediction per development patient and architecture, and never opens,
loads, or infers the 46-patient independent test set. Its outputs are secure and
must never be committed to Git.
"""
from __future__ import print_function

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

import experiments2026 as final_protocol


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIRECTORY.parent
OUTPUT_SUBDIRECTORIES = (
    "manifests",
    "fold_records",
    "predictions/folds",
    "predictions/by_model",
    "summaries",
)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run development-only 5-fold CV for final-protocol ensemble selection. It never loads test data."
    )
    parser.add_argument("--root", type=Path, required=True, help="Secure dataparent directory containing data/training.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New secure directory for CV manifests, OOF predictions, and summaries.")
    parser.add_argument(
        "--models",
        default=",".join(final_protocol.SUPPORTED_MODELS),
        help="Comma-separated architecture names. The complete run must include all 11 architectures.",
    )
    parser.add_argument("--base-seed", type=int, default=20260929, help="Prespecified base seed used to derive one seed per model/fold.")
    parser.add_argument("--cv-seed", type=int, default=42, help="Prespecified stratified-fold assignment seed.")
    parser.add_argument("--n-folds", type=int, default=5, help="Five folds are required for the final ensemble-selection run.")
    parser.add_argument("--allow-partial", action="store_true", help="Permit a model subset for a technical preflight only; it cannot create a final ranking.")
    parser.add_argument("--resume", action="store_true", help="Resume only a compatible interrupted run by reusing validated completed fold records.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output directory before starting. Never use to mix runs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data, fold assignments, and output plan without constructing a model.")
    return parser.parse_args()


def selected_models(argument: str) -> List[str]:
    models = parse_csv(argument)
    if not models:
        raise ValueError("At least one architecture must be requested")
    unknown = sorted(set(models) - set(final_protocol.SUPPORTED_MODELS))
    if unknown:
        raise ValueError("Unsupported architecture(s): {0}".format(", ".join(unknown)))
    if len(models) != len(set(models)):
        raise ValueError("Architecture names must not be duplicated")
    return models


def final_protocol_description() -> Dict[str, Any]:
    return {
        "code_file": "Experiment2026/experiments2026.py",
        "input_size": 128,
        "batch_size": 10,
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "loss": "binary_crossentropy",
        "dropout_rate": 0.50,
        "class_weights": True,
        "augmentation": "none",
        "red_green_shift": False,
        "phase1_epochs": 100,
        "phase2_epochs": 100,
        "phase3_epochs": 100,
        "early_stopping": "validation binary accuracy; patience=4; restore_best_weights=True; one callback reused across legacy phases",
        "fine_tuning": "literal legacy schedule: phase 1 freezes backbone; phase 2 marks layer -2 trainable; phase 3 additionally marks layer -3 trainable",
        "preprocessing": "architecture-specific official Keras preprocess_input after resize; no /255 scaling",
        "dataset_shuffle": "none; preserves the literal legacy tf.data order",
        "binary_threshold": 0.50,
    }


def json_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_output_directory(output_dir: Path, manifest: Mapping[str, Any], overwrite: bool, resume: bool) -> None:
    manifest_path = output_dir / "manifests" / "cv_run_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite and resume:
            raise ValueError("Use either --overwrite or --resume, never both")
        if overwrite:
            shutil.rmtree(str(output_dir))
        elif resume:
            if not manifest_path.exists():
                raise FileNotFoundError("Cannot resume: existing output directory lacks {0}".format(manifest_path))
            existing = read_json(manifest_path)
            if not json_equal(existing, manifest):
                raise RuntimeError("Cannot resume: the existing CV run manifest does not match the requested locked plan")
        else:
            raise FileExistsError(
                "Output directory {0} is not empty. Use a new directory, --resume for the identical plan, or --overwrite to delete it.".format(output_dir)
            )
    for relative in OUTPUT_SUBDIRECTORIES:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)


def load_development_data(root: Path) -> Tuple[List[Path], np.ndarray]:
    development_dir = root / "data" / "training"
    paths = sorted(development_dir.glob("*.jpg"))
    labels = final_protocol.load_binary_labels(development_dir / "ica_lables.txt")
    if len(paths) != 92 or len(labels) != 92:
        raise ValueError("Expected exactly 92 development JPEGs/labels; found {0} JPEGs and {1} labels.".format(len(paths), len(labels)))
    observed = (int((labels == 0).sum()), int((labels == 1).sum()))
    if observed != (56, 36):
        raise ValueError("Expected 92 development labels with class 0=56 and class 1=36; found class 0={0}, class 1={1}.".format(*observed))
    return paths, labels


def build_fold_manifest(paths: Sequence[Path], labels: np.ndarray, splitter: StratifiedKFold) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold, (_, validation_indices) in enumerate(splitter.split(paths, labels), start=1):
        for index in validation_indices:
            rows.append({
                "patient_id": paths[index].name,
                "fold": fold,
                "observed_label": int(labels[index]),
            })
    manifest = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    if len(manifest) != len(paths) or manifest["patient_id"].duplicated().any():
        raise RuntimeError("The fold manifest must contain every development patient exactly once")
    return manifest


def fold_seed(base_seed: int, model_index: int, fold: int) -> int:
    """Prespecified, distinct deterministic seed for each architecture/fold fit."""
    return int(base_seed + model_index * 10 + fold)


def record_path(output_dir: Path, model_name: str, fold: int) -> Path:
    return output_dir / "fold_records" / "{0}_fold_{1}.json".format(model_name, fold)


def prediction_path(output_dir: Path, model_name: str, fold: int) -> Path:
    return output_dir / "predictions" / "folds" / "oof_{0}_fold_{1}.csv".format(model_name, fold)


def load_completed_fold(output_dir: Path, model_name: str, fold: int, expected_patient_ids: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    prediction_file = prediction_path(output_dir, model_name, fold)
    record_file = record_path(output_dir, model_name, fold)
    if not prediction_file.exists() or not record_file.exists():
        raise FileNotFoundError("A completed fold requires both prediction and record files")
    predictions = pd.read_csv(prediction_file)
    required = {"patient_id", "fold", "model_name", "observed_label", "probability", "binary_prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError("Completed fold prediction file has an invalid schema: {0}".format(prediction_file))
    if set(predictions["patient_id"].astype(str)) != set(expected_patient_ids):
        raise ValueError("Completed fold prediction patients differ from the locked fold assignment")
    if predictions["model_name"].astype(str).nunique() != 1 or str(predictions["model_name"].iloc[0]) != model_name:
        raise ValueError("Completed fold prediction file has the wrong architecture")
    if predictions["fold"].nunique() != 1 or int(predictions["fold"].iloc[0]) != fold:
        raise ValueError("Completed fold prediction file has the wrong fold")
    if predictions["patient_id"].duplicated().any() or not predictions["probability"].between(0.0, 1.0).all():
        raise ValueError("Completed fold prediction file is incomplete or invalid")
    return predictions.sort_values("patient_id").reset_index(drop=True), read_json(record_file)


def train_one_fold(
    model_name: str,
    seed: int,
    fold: int,
    train_paths: Sequence[Path],
    train_labels: np.ndarray,
    validation_paths: Sequence[Path],
    validation_labels: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit one final-protocol CV fold and return its OOF predictions/provenance."""
    final_protocol.set_seed(seed)
    tf.keras.backend.clear_session()
    train_dataset = final_protocol.make_dataset(
        train_paths, train_labels, model_name, 128, 10, seed, training=True, red_green_shift=False
    )
    validation_dataset = final_protocol.make_dataset(
        validation_paths, validation_labels, model_name, 128, 10, seed, training=False, red_green_shift=False
    )
    class_weights = final_protocol.make_class_weights(train_labels)
    model, base_model = final_protocol.make_model(model_name, 128)
    early_stopping = final_protocol.make_early_stopping()
    started = time.time()
    phase_1 = final_protocol.run_phase(
        model, base_model, train_dataset, validation_dataset, class_weights, early_stopping,
        phase=1, phase_epochs=100, initial_epoch=0,
    )
    phase_2 = final_protocol.run_phase(
        model, base_model, train_dataset, validation_dataset, class_weights, early_stopping,
        phase=2, phase_epochs=100, initial_epoch=100,
    )
    phase_3 = final_protocol.run_phase(
        model, base_model, train_dataset, validation_dataset, class_weights, early_stopping,
        phase=3, phase_epochs=100, initial_epoch=200,
    )
    probabilities = model.predict(validation_dataset, verbose=0).reshape(-1)
    metrics = final_protocol.calculate_metrics(validation_labels, probabilities)
    predictions = pd.DataFrame({
        "patient_id": [path.name for path in validation_paths],
        "fold": int(fold),
        "model_name": model_name,
        "seed": int(seed),
        "observed_label": np.asarray(validation_labels, dtype=int),
        "probability": np.asarray(probabilities, dtype=float),
        "binary_prediction": np.round(probabilities).astype(int),
    }).sort_values("patient_id").reset_index(drop=True)
    record = {
        "model_name": model_name,
        "fold": int(fold),
        "seed": int(seed),
        "n_train": int(len(train_labels)),
        "n_validation": int(len(validation_labels)),
        "train_class_0": int((train_labels == 0).sum()),
        "train_class_1": int((train_labels == 1).sum()),
        "validation_class_0": int((validation_labels == 0).sum()),
        "validation_class_1": int((validation_labels == 1).sum()),
        "class_weight_0": float(class_weights[0]),
        "class_weight_1": float(class_weights[1]),
        "elapsed_seconds": float(time.time() - started),
        "phase_parameters": [phase_1, phase_2, phase_3],
        "validation_metrics": {key: value for key, value in metrics.items() if key not in ("probabilities", "binary_predictions")},
    }
    tf.keras.backend.clear_session()
    gc.collect()
    return predictions, record


def collect_complete_outputs(output_dir: Path, models: Sequence[str], n_folds: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_predictions: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []
    for model_name in models:
        model_predictions: List[pd.DataFrame] = []
        for fold in range(1, n_folds + 1):
            prediction_file = prediction_path(output_dir, model_name, fold)
            record_file = record_path(output_dir, model_name, fold)
            if not prediction_file.exists() or not record_file.exists():
                raise RuntimeError("Cannot finalize CV outputs: missing completed fold {0} / {1}".format(model_name, fold))
            predictions = pd.read_csv(prediction_file)
            record = read_json(record_file)
            model_predictions.append(predictions)
            all_predictions.append(predictions)
            metric_row = dict(record["validation_metrics"])
            metric_row.update({key: record[key] for key in (
                "model_name", "fold", "seed", "n_train", "n_validation", "train_class_0", "train_class_1",
                "validation_class_0", "validation_class_1", "class_weight_0", "class_weight_1", "elapsed_seconds",
            )})
            fold_rows.append(metric_row)
            for phase in record["phase_parameters"]:
                phase_row = dict(phase)
                phase_row.update({"model_name": model_name, "fold": int(fold), "seed": int(record["seed"])})
                phase_rows.append(phase_row)
        combined = pd.concat(model_predictions, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
        if len(combined) != 92 or combined["patient_id"].duplicated().any():
            raise RuntimeError("Architecture {0} does not have exactly one OOF prediction for every development patient".format(model_name))
        combined.to_csv(output_dir / "predictions" / "by_model" / "oof_{0}.csv".format(model_name), index=False)
    return (
        pd.concat(all_predictions, ignore_index=True).sort_values(["model_name", "patient_id"]).reset_index(drop=True),
        pd.DataFrame(fold_rows).sort_values(["model_name", "fold"]).reset_index(drop=True),
        pd.DataFrame(phase_rows).sort_values(["model_name", "fold", "phase"]).reset_index(drop=True),
    )


def architecture_summary(all_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, group in all_predictions.groupby("model_name", sort=True):
        metrics = final_protocol.calculate_metrics(group["observed_label"], group["probability"])
        rows.append({
            "model_name": model_name,
            "n_oof_patients": int(len(group)),
            "n_folds": int(group["fold"].nunique()),
            "pooled_oof_auc": metrics["auc"],
            "pooled_oof_accuracy": metrics["accuracy"],
            "pooled_oof_precision": metrics["precision"],
            "pooled_oof_sensitivity": metrics["sensitivity"],
            "pooled_oof_specificity": metrics["specificity"],
            "pooled_oof_f1": metrics["f1"],
            "pooled_oof_tn": metrics["tn"],
            "pooled_oof_fp": metrics["fp"],
            "pooled_oof_fn": metrics["fn"],
            "pooled_oof_tp": metrics["tp"],
        })
    summary = pd.DataFrame(rows).sort_values(["pooled_oof_auc", "model_name"], ascending=[False, True]).reset_index(drop=True)
    summary.insert(0, "development_rank", np.arange(1, len(summary) + 1, dtype=int))
    return summary


def write_cv_results_workbook(
    output_dir: Path,
    run_manifest: Mapping[str, Any],
    fold_manifest: pd.DataFrame,
    all_predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    phase_parameters: pd.DataFrame,
    ranking: pd.DataFrame,
    full_run: bool,
) -> Path:
    """Write one readable secure CV workbook after all requested fits complete."""
    output_file = output_dir / "ensemble_selection_cv_results.xlsx"
    lock_rows = [{"item": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, bool)) else value} for key, value in run_manifest.items()]
    if full_run:
        pools = {
            "Top-3": ranking.loc[ranking["development_rank"] <= 3, ["development_rank", "model_name"]],
            "Top-5": ranking.loc[ranking["development_rank"] <= 5, ["development_rank", "model_name"]],
        }
        pool_rows: List[Dict[str, Any]] = []
        for pool_name, pool in pools.items():
            for pool_rank, row in enumerate(pool.sort_values("development_rank").itertuples(index=False), start=1):
                pool_rows.append({
                    "pool_name": pool_name,
                    "pool_rank": pool_rank,
                    "development_rank": int(row.development_rank),
                    "model_name": str(row.model_name),
                })
        candidate_pools = pd.DataFrame(pool_rows)
    else:
        candidate_pools = pd.DataFrame(columns=["pool_name", "pool_rank", "development_rank", "model_name"])
    with pd.ExcelWriter(str(output_file), engine="openpyxl", mode="w") as writer:
        pd.DataFrame(lock_rows).to_excel(writer, sheet_name="Study_lock", index=False)
        fold_manifest.to_excel(writer, sheet_name="Fold_manifest", index=False)
        ranking.to_excel(writer, sheet_name="Architecture_ranking", index=False)
        candidate_pools.to_excel(writer, sheet_name="Candidate_pools", index=False)
        all_predictions.to_excel(writer, sheet_name="OOF_predictions", index=False)
        fold_metrics.to_excel(writer, sheet_name="Fold_metrics", index=False)
        phase_parameters.to_excel(writer, sheet_name="Phase_parameters", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
    return output_file


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.output_dir = args.output_dir.resolve()
    models = selected_models(args.models)
    full_run = set(models) == set(final_protocol.SUPPORTED_MODELS) and len(models) == len(final_protocol.SUPPORTED_MODELS)
    if not full_run and not args.allow_partial:
        raise ValueError("A model subset is permitted only for a technical preflight; add --allow-partial explicitly")
    if args.n_folds != 5 and not args.allow_partial:
        raise ValueError("The final ensemble-selection run requires exactly five stratified folds")
    if args.n_folds < 2:
        raise ValueError("At least two folds are required")
    paths, labels = load_development_data(args.root)
    if args.n_folds > int(np.bincount(labels).min()):
        raise ValueError("The requested fold count exceeds the minority-class count")
    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.cv_seed)
    fold_manifest = build_fold_manifest(paths, labels, splitter)
    manifest = {
        "study": "EJPH-D-26-00179 development-only ensemble configuration selection",
        "purpose": "rank architectures and select one ensemble configuration using development OOF predictions only",
        "test_data_accessed": False,
        "root_data_access": "data/training only; data/test is never opened by this runner",
        "n_development_patients": int(len(paths)),
        "development_class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "models": models,
        "n_folds": int(args.n_folds),
        "cv_seed": int(args.cv_seed),
        "base_seed": int(args.base_seed),
        "partial_technical_preflight": bool(not full_run),
        "final_protocol": final_protocol_description(),
    }
    prepare_output_directory(args.output_dir, manifest, args.overwrite, args.resume)
    fold_manifest.to_csv(args.output_dir / "manifests" / "development_fold_manifest.csv", index=False)
    planned_fits = len(models) * args.n_folds
    print("Development-only cohort: n=92; class 0={0}; class 1={1}".format(int((labels == 0).sum()), int((labels == 1).sum())))
    print("Architectures: {0}; folds: {1}; planned fits: {2}".format(len(models), args.n_folds, planned_fits))
    print("Locked final protocol: 128 input | batch 10 | Adam 0.0003 | dropout 0.50 | class weights | no augmentation")
    print("Test data is deliberately not loaded by this runner.")
    if args.dry_run:
        print("Dry run complete: the development cohort, fold assignments, and run manifest were validated; no model was constructed.")
        return

    split_indices = list(splitter.split(paths, labels))
    for model_index, model_name in enumerate(models):
        for fold, (train_indices, validation_indices) in enumerate(split_indices, start=1):
            expected_ids = [paths[index].name for index in validation_indices]
            prediction_file = prediction_path(args.output_dir, model_name, fold)
            record_file = record_path(args.output_dir, model_name, fold)
            if args.resume and prediction_file.exists() and record_file.exists():
                load_completed_fold(args.output_dir, model_name, fold, expected_ids)
                print("Skipping validated completed fold: {0} | fold {1}".format(model_name, fold))
                continue
            if prediction_file.exists() or record_file.exists():
                raise RuntimeError("Found a partial fold output for {0} fold {1}. Remove both files only after investigation; do not overwrite silently.".format(model_name, fold))
            seed = fold_seed(args.base_seed, model_index, fold)
            print("\n=== {0} | fold {1}/{2} | seed {3} ===".format(model_name, fold, args.n_folds, seed))
            predictions, record = train_one_fold(
                model_name=model_name,
                seed=seed,
                fold=fold,
                train_paths=[paths[index] for index in train_indices],
                train_labels=labels[train_indices],
                validation_paths=[paths[index] for index in validation_indices],
                validation_labels=labels[validation_indices],
            )
            predictions.to_csv(prediction_file, index=False)
            write_json(record_file, record)
            print("Fold {0}: AUC={1:.4f}, ACC={2:.4f}; saved secure OOF predictions.".format(
                fold, float(record["validation_metrics"]["auc"]), float(record["validation_metrics"]["accuracy"])
            ))

    all_predictions, fold_metrics, phase_parameters = collect_complete_outputs(args.output_dir, models, args.n_folds)
    all_predictions.to_csv(args.output_dir / "predictions" / "oof_predictions_all.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "summaries" / "cv_fold_metrics.csv", index=False)
    phase_parameters.to_csv(args.output_dir / "summaries" / "cv_phase_parameters.csv", index=False)
    ranking = architecture_summary(all_predictions)
    ranking.to_csv(args.output_dir / "summaries" / "architecture_oof_ranking.csv", index=False)
    workbook = write_cv_results_workbook(
        output_dir=args.output_dir,
        run_manifest=manifest,
        fold_manifest=fold_manifest,
        all_predictions=all_predictions,
        fold_metrics=fold_metrics,
        phase_parameters=phase_parameters,
        ranking=ranking,
        full_run=full_run,
    )
    if full_run:
        print("\nCompleted all 55 development-only fits.")
        print("Saved architecture ranking: {0}".format(args.output_dir / "summaries" / "architecture_oof_ranking.csv"))
        print("Saved CV results workbook: {0}".format(workbook))
        print("Next step: use this workbook in R1_ensemble_configuration_selection.ipynb.")
    else:
        print("\nPartial technical preflight complete. Do not use partial output for ensemble selection.")


if __name__ == "__main__":
    main()
