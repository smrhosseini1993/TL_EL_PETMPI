#!/usr/bin/env python3
"""Locked historic 61/31/46 TL runner for the R1 PET reanalysis.

The runner accepts only the protocol selected by the complete development-only CV
search. It writes all completed runs to one resumable SQLite database after each model /
seed fit. It does not select architectures, thresholds, or best seeds.
"""
from __future__ import annotations

import argparse
import json
import traceback
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from final_results_store import (
    connect_results_database,
    database_path,
    initialise_results_store,
    mark_run_failed,
    mark_run_started,
    record_completed_run,
    run_is_complete,
    sha256_file,
)
from tl_reanalysis_core import (
    SUPPORTED_MODELS,
    binary_metrics,
    build_dataset,
    build_model,
    load_protocol,
    manifest_json,
    prediction_table,
    run_three_phase_training,
    runtime_metadata,
    set_global_seed,
)

REPO_ROOT = Path(__file__).resolve().parent
REQUIRED_SPLIT_COLUMNS = {"patient_id", "relative_path", "split", "observed_label"}
EXPECTED_SPLIT_SIZES = {"train": 61, "validation": 31, "test": 46}
# These are the published historic split class counts: 25/36, 11/20, and 20/26.
EXPECTED_SPLIT_CLASS_COUNTS = {
    "train": {0: 36, 1: 25},
    "validation": {0: 20, 1: 11},
    "test": {0: 26, 1: 20},
}


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seed_specification(value: str) -> List[int]:
    seeds: List[int] = []
    for token in parse_csv(value):
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid seed range: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    unique = sorted(set(seeds))
    if not unique or min(unique) < 1:
        raise ValueError("Seeds must be positive integers")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R1 locked historic 61/31/46 TL benchmark runner.")
    parser.add_argument("--root", type=Path, required=True, help="Secure project root containing image files.")
    parser.add_argument("--split-manifest", type=Path, required=True, help="Verified secure 61/31/46 split CSV.")
    parser.add_argument("--protocol-file", type=Path, required=True, help="selected_protocol.json from the complete CV search.")
    parser.add_argument(
        "--selection-provenance",
        type=Path,
        help="selection_provenance.json from the same complete CV output directory. Defaults beside --protocol-file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Secure R1_final_TL directory containing the one durable SQLite database and batch manifests.",
    )
    parser.add_argument(
        "--batch-name",
        required=True,
        help="Immutable batch identifier, e.g. seeds_001_005 or seeds_006_100.",
    )
    parser.add_argument(
        "--models",
        default=",".join(SUPPORTED_MODELS),
        help="Comma-separated models. Default: all 11 prespecified architectures.",
    )
    parser.add_argument("--seeds", required=True, help="Predeclared seeds, e.g. 1-5 or 6-100.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Mark the seeds 1-5 technical acceptance batch; no performance-based change is allowed afterward.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs, manifests, and database compatibility without fitting models.")
    return parser.parse_args()


def select_models(argument: str) -> List[str]:
    models = parse_csv(argument)
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Allowed: {', '.join(SUPPORTED_MODELS)}")
    if not models:
        raise ValueError("At least one model must be selected")
    return models


def validate_batch_contract(batch_name: str, seeds: Sequence[int], preflight: bool) -> None:
    expected_preflight_seeds = list(range(1, 6))
    expected_production_seeds = list(range(6, 101))
    if preflight:
        if list(seeds) != expected_preflight_seeds:
            raise ValueError("The technical preflight must use exactly seeds 1-5.")
        if batch_name != "seeds_001_005":
            raise ValueError("The technical preflight batch name must be exactly 'seeds_001_005'.")
    else:
        if list(seeds) != expected_production_seeds:
            raise ValueError("The production batch must use exactly seeds 6-100.")
        if batch_name != "seeds_006_100":
            raise ValueError("The production batch name must be exactly 'seeds_006_100'.")


def load_split_manifest(root: Path, manifest_file: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_file, dtype={"patient_id": str, "relative_path": str})
    missing = REQUIRED_SPLIT_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest missing required columns: {sorted(missing)}")
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate patient_id values")
    if manifest["relative_path"].duplicated().any():
        raise ValueError("Split manifest contains duplicate relative_path values")
    if set(manifest["split"].unique()) != set(EXPECTED_SPLIT_SIZES):
        raise ValueError("Split manifest must contain exactly train, validation, and test splits")
    manifest["observed_label"] = manifest["observed_label"].astype(int)
    if not np.isin(manifest["observed_label"], [0, 1]).all():
        raise ValueError("Split manifest labels must be binary 0/1")

    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        group = manifest.loc[manifest["split"] == split]
        if len(group) != expected_size:
            raise ValueError(f"Split '{split}' has {len(group)} patients; expected {expected_size}")
        observed_counts = {label: int((group["observed_label"] == label).sum()) for label in (0, 1)}
        if observed_counts != EXPECTED_SPLIT_CLASS_COUNTS[split]:
            raise ValueError(
                f"Split '{split}' has class counts {observed_counts}; "
                f"expected {EXPECTED_SPLIT_CLASS_COUNTS[split]} for the historic 61/31/46 split"
            )
    for relative_path in manifest["relative_path"]:
        if not (root / relative_path).is_file():
            raise FileNotFoundError(f"Manifest file does not exist: {root / relative_path}")
    return manifest.sort_values(["split", "patient_id"]).reset_index(drop=True)


def split_records(root: Path, manifest: pd.DataFrame, split: str) -> tuple[List[str], List[str], np.ndarray]:
    group = manifest.loc[manifest["split"] == split].sort_values("patient_id")
    paths = [str(root / path) for path in group["relative_path"]]
    identifiers = group["patient_id"].astype(str).tolist()
    labels = group["observed_label"].astype(int).to_numpy()
    return paths, identifiers, labels


def verify_protocol_provenance(protocol_file: Path, provenance_file: Path, protocol_id: str) -> Mapping[str, object]:
    """Reject ad hoc protocol JSON files that were not locked by a full CV run."""
    if protocol_file.name != "selected_protocol.json":
        raise ValueError("--protocol-file must be the selected_protocol.json written by TL_crossvalidation.py")
    if not provenance_file.is_file():
        raise FileNotFoundError(f"Missing CV selection provenance: {provenance_file}")
    with provenance_file.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if provenance.get("test_data_accessed") is not False:
        raise ValueError("Protocol provenance must document test_data_accessed=false")
    selected = provenance.get("selected_protocol", {})
    if selected.get("protocol_id") != protocol_id:
        raise ValueError("selected_protocol.json does not match selection_provenance.json")
    return provenance


def batch_manifest(
    *,
    batch_name: str,
    preflight: bool,
    models: Sequence[str],
    seeds: Sequence[int],
    protocol: Mapping[str, object],
    root: Path,
    split_manifest: Path,
    protocol_file: Path,
    provenance_file: Path,
    manifest: pd.DataFrame,
) -> Dict[str, object]:
    return {
        "batch_name": batch_name,
        "purpose": "technical_acceptance_preflight_included_in_final_100_seeds" if preflight else "remaining_final_locked_stability_runs",
        "preflight": bool(preflight),
        "models": list(models),
        "seeds": list(seeds),
        "protocol": dict(protocol),
        "secure_root": str(root),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "protocol_file": str(protocol_file),
        "protocol_file_sha256": sha256_file(protocol_file),
        "selection_provenance": str(provenance_file),
        "selection_provenance_sha256": sha256_file(provenance_file),
        "split_counts": {split: int((manifest["split"] == split).sum()) for split in EXPECTED_SPLIT_SIZES},
        "split_class_counts": {
            split: {str(label): int(((manifest["split"] == split) & (manifest["observed_label"] == label)).sum()) for label in (0, 1)}
            for split in EXPECTED_SPLIT_SIZES
        },
        "test_data_accessed": True,
        "selection_performed": False,
        "best_seed_selection_performed": False,
        "runtime": runtime_metadata(REPO_ROOT),
    }


def study_signature(batch: Mapping[str, object]) -> Dict[str, object]:
    """Fields which must never differ between the two final-run batches."""
    runtime = batch["runtime"]
    return {
        "protocol_file_sha256": batch["protocol_file_sha256"],
        "selection_provenance_sha256": batch["selection_provenance_sha256"],
        "split_manifest_sha256": batch["split_manifest_sha256"],
        "protocol": batch["protocol"],
        "split_counts": batch["split_counts"],
        "split_class_counts": batch["split_class_counts"],
        "code_commit": runtime["git_commit"],
        "tensorflow_version": runtime["tensorflow_version"],
        "python_version": runtime["python_version"],
    }


def compact_phase_rows(phase_parameters: Sequence[Mapping[str, object]], histories: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    histories_by_phase = {int(item["phase"]): item for item in histories}
    rows: List[Dict[str, object]] = []
    for item in phase_parameters:
        phase = int(item["phase"])
        history = histories_by_phase[phase]
        rows.append(
            {
                **dict(item),
                "learning_rate": float(history["learning_rate"]),
                "requested_epochs": int(history["requested_epochs"]),
                "actual_epochs": int(history["actual_epochs"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    models = select_models(args.models)
    seeds = parse_seed_specification(args.seeds)
    validate_batch_contract(args.batch_name, seeds, args.preflight)

    protocol_object = load_protocol(args.protocol_file)
    protocol = protocol_object.as_dict()
    provenance_file = args.selection_provenance or (args.protocol_file.parent / "selection_provenance.json")
    verify_protocol_provenance(args.protocol_file, provenance_file, protocol_object.protocol_id)
    manifest = load_split_manifest(args.root, args.split_manifest)
    train_paths, train_ids, train_labels = split_records(args.root, manifest, "train")
    validation_paths, validation_ids, validation_labels = split_records(args.root, manifest, "validation")
    test_paths, test_ids, test_labels = split_records(args.root, manifest, "test")

    current_batch_manifest = batch_manifest(
        batch_name=args.batch_name,
        preflight=args.preflight,
        models=models,
        seeds=seeds,
        protocol=protocol,
        root=args.root,
        split_manifest=args.split_manifest,
        protocol_file=args.protocol_file,
        provenance_file=provenance_file,
        manifest=manifest,
    )
    connection = connect_results_database(args.results_dir)
    try:
        initialise_results_store(
            connection,
            batch_name=args.batch_name,
            batch_manifest=current_batch_manifest,
            study_signature=study_signature(current_batch_manifest),
        )
        manifest_json(args.results_dir / f"run_manifest_{args.batch_name}.json", current_batch_manifest)

        planned_fits = len(models) * len(seeds)
        print(f"Locked split: train={len(train_paths)}, validation={len(validation_paths)}, test={len(test_paths)}")
        print(f"Models: {len(models)}; seeds: {len(seeds)}; planned fits: {planned_fits}; preflight={args.preflight}")
        print(f"Protocol: {protocol_object.protocol_id}")
        print(f"Results database: {database_path(args.results_dir)}")
        if args.dry_run:
            print("Dry run complete: manifest and database compatibility were validated; no model was constructed.")
            return

        for model_name in models:
            for seed in seeds:
                if run_is_complete(connection, model_name, seed, expected_test_patients=len(test_ids)):
                    print(f"Skipping completed run: {model_name} | seed {seed}")
                    continue
                print(f"\n=== {model_name} | {protocol_object.protocol_id} | seed {seed} ===")
                started = time.perf_counter()
                mark_run_started(connection, model_name=model_name, seed=seed, batch_name=args.batch_name)
                try:
                    tf.keras.backend.clear_session()
                    set_global_seed(seed)
                    train_dataset = build_dataset(train_paths, train_labels, model_name, protocol_object.input_size, protocol_object.batch_size, training=True)
                    train_evaluation_dataset = build_dataset(train_paths, train_labels, model_name, protocol_object.input_size, protocol_object.batch_size, training=False)
                    validation_dataset = build_dataset(validation_paths, validation_labels, model_name, protocol_object.input_size, protocol_object.batch_size, training=False)
                    test_dataset = build_dataset(test_paths, test_labels, model_name, protocol_object.input_size, protocol_object.batch_size, training=False)
                    model, base_model = build_model(model_name, protocol_object, imagenet_weights=True)
                    histories, phase_parameters = run_three_phase_training(
                        model, base_model, model_name, protocol_object, train_dataset, validation_dataset
                    )
                    elapsed_seconds = time.perf_counter() - started

                    train_probabilities = model.predict(train_evaluation_dataset, verbose=0).reshape(-1)
                    validation_probabilities = model.predict(validation_dataset, verbose=0).reshape(-1)
                    test_probabilities = model.predict(test_dataset, verbose=0).reshape(-1)
                    split_metrics = {
                        "train": binary_metrics(train_labels, train_probabilities, protocol_object.threshold),
                        "validation": binary_metrics(validation_labels, validation_probabilities, protocol_object.threshold),
                        "test": binary_metrics(test_labels, test_probabilities, protocol_object.threshold),
                    }
                    test_table = prediction_table(
                        test_ids,
                        test_labels,
                        test_probabilities,
                        split="test",
                        model_name=model_name,
                        protocol=protocol_object,
                        seed=seed,
                    )
                    record_completed_run(
                        connection,
                        model_name=model_name,
                        seed=seed,
                        batch_name=args.batch_name,
                        protocol=protocol,
                        elapsed_seconds=elapsed_seconds,
                        split_metrics=split_metrics,
                        test_prediction_rows=test_table,
                        phase_rows=compact_phase_rows(phase_parameters, histories),
                    )
                    print(f"Completed: {model_name} | seed {seed} | elapsed={elapsed_seconds:.1f}s")
                    del model, base_model
                    tf.keras.backend.clear_session()
                except Exception as error:
                    elapsed_seconds = time.perf_counter() - started
                    mark_run_failed(
                        connection,
                        model_name=model_name,
                        seed=seed,
                        elapsed_seconds=elapsed_seconds,
                        error_message=traceback.format_exc(),
                    )
                    tf.keras.backend.clear_session()
                    raise RuntimeError(f"Run failed: {model_name} | seed {seed}; error recorded in SQLite") from error
        print("\nCompleted locked fixed-split batch.")
        print("Runs are stored in the single SQLite results file; test performance is not used to change settings.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
