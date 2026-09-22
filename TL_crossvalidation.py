#!/usr/bin/env python3
"""Development-only hyperparameter selection for the R1 PET TL reanalysis.

This runner deliberately loads only root/data/training (92 development patients).
It never reads root/data/test. It evaluates the prespecified common 3×3×3 protocol
grid across the eleven backbones and saves patient-level out-of-fold predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from tl_reanalysis_core import (
    SUPPORTED_MODELS,
    Protocol,
    assert_binary_labels,
    binary_metrics,
    build_dataset,
    build_model,
    image_paths,
    load_labels,
    manifest_json,
    prediction_table,
    protocol_grid,
    run_three_phase_training,
    runtime_metadata,
    set_global_seed,
    verify_paths_and_labels,
)

REPO_ROOT = Path(__file__).resolve().parent


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R1 development-only stratified CV protocol selection. Never loads test data."
    )
    parser.add_argument("--root", type=Path, required=True, help="Secure project root containing data/training.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Secure output directory for manifests, OOF predictions, and selection summaries.",
    )
    parser.add_argument(
        "--models",
        default=",".join(SUPPORTED_MODELS),
        help="Comma-separated model names. Default: all 11 prespecified architectures.",
    )
    parser.add_argument(
        "--protocol-ids",
        default="all",
        help="Comma-separated protocol IDs or 'all'. Partial runs require --allow-partial and cannot select a protocol.",
    )
    parser.add_argument("--seed", type=int, default=43, help="Fixed training seed for every CV fit.")
    parser.add_argument("--cv-seed", type=int, default=42, help="Fixed StratifiedKFold split seed.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of stratified CV folds; final protocol requires 5.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a model/protocol subset for code preflight. It never writes selected_protocol.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data, folds, and plan without fitting models.")
    parser.add_argument("--overwrite", action="store_true", help="Allow reuse of a non-empty output directory.")
    return parser.parse_args()


def select_models(argument: str) -> List[str]:
    models = parse_csv(argument)
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Allowed: {', '.join(SUPPORTED_MODELS)}")
    if not models:
        raise ValueError("At least one model must be selected")
    return models


def select_protocols(argument: str) -> List[Protocol]:
    grid = protocol_grid()
    by_id = {protocol.protocol_id: protocol for protocol in grid}
    if argument == "all":
        return grid
    identifiers = parse_csv(argument)
    unknown = sorted(set(identifiers) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown protocol ID(s): {unknown}")
    return [by_id[identifier] for identifier in identifiers]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} is not empty. Use a new directory or --overwrite; "
            "never mix protocol-search runs."
        )
    for relative in ("manifests", "predictions", "summaries", "phase_parameters", "histories"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def build_fold_manifest(paths: Sequence[Path], labels: np.ndarray, splitter: StratifiedKFold) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for fold, (_, validation_index) in enumerate(splitter.split(paths, labels), start=1):
        for index in validation_index:
            rows.append({"patient_id": paths[index].name, "fold": fold, "observed_label": int(labels[index])})
    manifest = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    if len(manifest) != len(paths) or manifest["patient_id"].duplicated().any():
        raise RuntimeError("Fold manifest must contain each development patient exactly once")
    return manifest


def ranked_selection(common_summary: pd.DataFrame) -> pd.Series:
    best_auc = common_summary["mean_pooled_oof_auc"].max()
    # Exact numerical tie only; no arbitrary 'within x AUC' margin is used.
    tied = common_summary.loc[
        common_summary["mean_pooled_oof_auc"].round(12) == round(float(best_auc), 12)
    ].copy()
    tied["dropout_priority"] = (tied["dropout_rate"] - 0.50).abs()
    return tied.sort_values(
        ["fine_tune_learning_rate", "head_learning_rate", "dropout_priority"],
        ascending=[True, True, True],
    ).iloc[0]


def main() -> None:
    args = parse_args()
    models = select_models(args.models)
    protocols = select_protocols(args.protocol_ids)
    full_requested_grid = set(models) == set(SUPPORTED_MODELS) and len(protocols) == len(protocol_grid())
    if not full_requested_grid and not args.allow_partial:
        raise ValueError(
            "A partial model/protocol run is only for code preflight. Pass --allow-partial explicitly; "
            "it cannot select the locked common protocol."
        )
    if args.n_folds != 5 and not args.allow_partial:
        raise ValueError("The locked R1 protocol uses exactly 5 folds. Use --allow-partial only for code preflight.")

    development_dir = args.root / "data" / "training"
    label_file = development_dir / "ica_lables.txt"
    paths = image_paths(development_dir)
    labels = load_labels(label_file)
    verify_paths_and_labels(paths, labels, "Development cohort")
    labels = assert_binary_labels(labels, "Development cohort")

    min_class_count = int(np.bincount(labels).min())
    if args.n_folds > min_class_count:
        raise ValueError(f"n_folds={args.n_folds} exceeds minority-class count={min_class_count}")

    prepare_output_dir(args.output_dir, args.overwrite)
    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.cv_seed)
    fold_manifest = build_fold_manifest(paths, labels, splitter)
    fold_manifest.to_csv(args.output_dir / "manifests" / "cv_fold_manifest.csv", index=False)

    run_manifest = {
        "purpose": "development_only_common_protocol_selection",
        "test_data_accessed": False,
        "development_image_dir": str(development_dir),
        "development_label_file": str(label_file),
        "n_development_patients": int(len(paths)),
        "class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "models": models,
        "protocols": [protocol.as_dict() for protocol in protocols],
        "training_seed": args.seed,
        "cv_seed": args.cv_seed,
        "n_folds": args.n_folds,
        "partial_preflight": bool(not full_requested_grid),
        "runtime": runtime_metadata(REPO_ROOT),
    }
    manifest_json(args.output_dir / "manifests" / "cv_run_manifest.json", run_manifest)

    planned_fits = len(models) * len(protocols) * args.n_folds
    print(f"Development patients: {len(paths)}; class 0={int((labels == 0).sum())}; class 1={int((labels == 1).sum())}")
    print(f"Models: {len(models)}; protocols: {len(protocols)}; folds: {args.n_folds}; planned fits: {planned_fits}")
    print("Test data is deliberately not loaded by this runner.")
    if args.dry_run:
        print("Dry run complete: manifests and fold assignments were written; no model was constructed.")
        return

    all_predictions: List[pd.DataFrame] = []
    all_runs: List[Dict[str, object]] = []
    all_parameters: List[Dict[str, object]] = []
    all_histories: List[Dict[str, object]] = []

    for model_name in models:
        for protocol in protocols:
            print(f"\n=== {model_name} | {protocol.protocol_id} ===")
            protocol_predictions: List[pd.DataFrame] = []
            for fold, (train_index, validation_index) in enumerate(splitter.split(paths, labels), start=1):
                tf.keras.backend.clear_session()
                set_global_seed(args.seed)
                train_paths = [str(paths[index]) for index in train_index]
                validation_paths = [str(paths[index]) for index in validation_index]
                train_labels = labels[train_index]
                validation_labels = labels[validation_index]

                train_dataset = build_dataset(
                    train_paths, train_labels, model_name, protocol.input_size, protocol.batch_size, training=True
                )
                validation_dataset = build_dataset(
                    validation_paths, validation_labels, model_name, protocol.input_size, protocol.batch_size, training=False
                )
                model, base_model = build_model(model_name, protocol, imagenet_weights=True)
                histories, phase_parameters = run_three_phase_training(
                    model, base_model, model_name, protocol, train_dataset, validation_dataset
                )
                probabilities = model.predict(validation_dataset, verbose=0).reshape(-1)
                metrics = binary_metrics(validation_labels, probabilities, protocol.threshold)
                table = prediction_table(
                    [Path(path).name for path in validation_paths],
                    validation_labels,
                    probabilities,
                    split="development_oof",
                    model_name=model_name,
                    protocol=protocol,
                    seed=args.seed,
                    fold=fold,
                )
                protocol_predictions.append(table)
                all_predictions.append(table)

                all_runs.append(
                    {
                        "model_name": model_name,
                        "protocol_id": protocol.protocol_id,
                        "fold": fold,
                        "seed": args.seed,
                        "n_train": int(len(train_labels)),
                        "n_validation": int(len(validation_labels)),
                        "train_positive": int((train_labels == 1).sum()),
                        "validation_positive": int((validation_labels == 1).sum()),
                        **protocol.as_dict(),
                        **metrics,
                    }
                )
                for phase_record in phase_parameters:
                    all_parameters.append({"model_name": model_name, "protocol_id": protocol.protocol_id, "fold": fold, **phase_record})
                all_histories.append(
                    {
                        "model_name": model_name,
                        "protocol_id": protocol.protocol_id,
                        "fold": fold,
                        "seed": args.seed,
                        "history": histories,
                    }
                )
                print(f"Fold {fold}: AUC={metrics['auc']:.4f}, ACC={metrics['accuracy']:.4f}")

            combined = pd.concat(protocol_predictions, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
            if len(combined) != len(paths) or combined["patient_id"].duplicated().any():
                raise RuntimeError(f"OOF predictions incomplete for {model_name} {protocol.protocol_id}")
            combined.to_csv(args.output_dir / "predictions" / f"oof_{model_name}_{protocol.protocol_id}.csv", index=False)
            tf.keras.backend.clear_session()

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(args.output_dir / "predictions" / "oof_predictions_all.csv", index=False)
    pd.DataFrame(all_runs).to_csv(args.output_dir / "summaries" / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(all_parameters).to_csv(args.output_dir / "phase_parameters" / "cv_phase_parameters.csv", index=False)
    manifest_json(args.output_dir / "histories" / "cv_phase_histories.json", {"runs": all_histories})

    architecture_rows: List[Dict[str, object]] = []
    for (model_name, protocol_id), group in predictions_df.groupby(["model_name", "protocol_id"], sort=True):
        protocol = next(item for item in protocols if item.protocol_id == protocol_id)
        metrics = binary_metrics(group["observed_label"], group["probability"], protocol.threshold)
        architecture_rows.append(
            {
                "model_name": model_name,
                "protocol_id": protocol_id,
                **protocol.as_dict(),
                "n_oof_patients": int(len(group)),
                "pooled_oof_auc": metrics["auc"],
                "pooled_oof_accuracy": metrics["accuracy"],
                "pooled_oof_sensitivity": metrics["sensitivity"],
                "pooled_oof_specificity": metrics["specificity"],
            }
        )
    architecture_summary = pd.DataFrame(architecture_rows).sort_values(["protocol_id", "model_name"])
    architecture_summary.to_csv(args.output_dir / "summaries" / "architecture_protocol_oof_summary.csv", index=False)

    common_summary = (
        architecture_summary.groupby(
            ["protocol_id", "head_learning_rate", "fine_tune_learning_rate", "dropout_rate"], as_index=False
        )
        .agg(
            mean_pooled_oof_auc=("pooled_oof_auc", "mean"),
            sd_pooled_oof_auc=("pooled_oof_auc", "std"),
            n_architectures=("model_name", "nunique"),
        )
        .sort_values("mean_pooled_oof_auc", ascending=False)
        .reset_index(drop=True)
    )
    common_summary.to_csv(args.output_dir / "summaries" / "common_protocol_summary.csv", index=False)

    if full_requested_grid:
        incomplete = common_summary.loc[common_summary["n_architectures"] != len(SUPPORTED_MODELS)]
        if not incomplete.empty:
            raise RuntimeError(
                "Cannot select a common protocol until every candidate has pooled OOF results "
                f"for all {len(SUPPORTED_MODELS)} architectures."
            )
        selected = ranked_selection(common_summary)
        selected_protocol = next(protocol for protocol in protocols if protocol.protocol_id == selected["protocol_id"])
        manifest_json(args.output_dir / "selected_protocol.json", selected_protocol.as_dict())
        manifest_json(
            args.output_dir / "selection_provenance.json",
            {
                "selection_metric": "mean_pooled_out_of_fold_auc_across_11_architectures",
                "tie_rule": "lower fine-tune learning rate, then lower head learning rate, then dropout closest to 0.50",
                "selected_protocol": selected_protocol.as_dict(),
                "selected_summary": selected.to_dict(),
                "test_data_accessed": False,
                "cv_manifest": str(args.output_dir / "manifests" / "cv_run_manifest.json"),
            },
        )
        print(f"\nSelected locked protocol: {selected_protocol.protocol_id}")
        print(f"Saved: {args.output_dir / 'selected_protocol.json'}")
    else:
        print("\nPartial preflight completed. No selected_protocol.json was written.")


if __name__ == "__main__":
    main()
