#!/usr/bin/env python3
"""Locked historic 61/31/46 TL runner for the R1 PET reanalysis.

This runner accepts only a selected-protocol JSON file created by TL_crossvalidation.py.
It does not select hyperparameters, architectures, thresholds, or best seeds. It writes
patient-level full-precision predictions for every seed and split.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

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
    parser.add_argument("--split-manifest", type=Path, required=True, help="Secure CSV defining the locked 61/31/46 split.")
    parser.add_argument("--protocol-file", type=Path, required=True, help="selected_protocol.json from a complete CV search.")
    parser.add_argument(
        "--selection-provenance",
        type=Path,
        help="selection_provenance.json from the same complete CV output directory. Defaults beside --protocol-file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Secure empty directory for this run batch.")
    parser.add_argument(
        "--models",
        default=",".join(SUPPORTED_MODELS),
        help="Comma-separated models. Default: all 11 prespecified architectures.",
    )
    parser.add_argument("--seeds", default="1-100", help="Predeclared seeds, e.g. 1-5 for technical preflight or 6-100 after it passes.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Mark a technical preflight. This label does not permit any test-result-based settings change.",
    )
    parser.add_argument("--save-models", action="store_true", help="Save model weights by model/protocol/seed; off by default to limit storage.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifests and run plan without fitting models.")
    return parser.parse_args()


def select_models(argument: str) -> List[str]:
    models = parse_csv(argument)
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Allowed: {', '.join(SUPPORTED_MODELS)}")
    if not models:
        raise ValueError("At least one model must be selected")
    return models


def load_split_manifest(root: Path, manifest_file: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_file)
    missing = REQUIRED_SPLIT_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest missing required columns: {sorted(missing)}")
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate patient_id values")
    if manifest["relative_path"].duplicated().any():
        raise ValueError("Split manifest contains duplicate relative_path values")
    if set(manifest["split"].unique()) != set(EXPECTED_SPLIT_SIZES):
        raise ValueError("Split manifest must contain exactly train, validation, and test splits")
    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        observed_size = int((manifest["split"] == split).sum())
        if observed_size != expected_size:
            raise ValueError(f"Split '{split}' has {observed_size} patients; expected {expected_size}")
    labels = manifest["observed_label"].astype(int).to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Split manifest labels must be binary 0/1")
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


def make_output_dirs(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory {output_dir} is not empty. Use a distinct preflight/production directory; "
            "never append an incompatible batch."
        )
    for relative in ("manifests", "predictions", "summaries", "phase_parameters", "histories", "models"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def verify_protocol_provenance(protocol_file: Path, provenance_file: Path, protocol_id: str) -> None:
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


def main() -> None:
    args = parse_args()
    models = select_models(args.models)
    seeds = parse_seed_specification(args.seeds)
    if args.preflight and len(seeds) > 5:
        raise ValueError("Technical preflight may include at most five seeds. Use a separate production batch after review.")
    if not args.preflight and set(seeds) == set(range(1, 6)):
        raise ValueError("Use --preflight for seeds 1-5 so the batch is clearly labelled.")

    protocol = load_protocol(args.protocol_file)
    provenance_file = args.selection_provenance or (args.protocol_file.parent / "selection_provenance.json")
    verify_protocol_provenance(args.protocol_file, provenance_file, protocol.protocol_id)
    manifest = load_split_manifest(args.root, args.split_manifest)
    make_output_dirs(args.output_dir)
    train_paths, train_ids, train_labels = split_records(args.root, manifest, "train")
    validation_paths, validation_ids, validation_labels = split_records(args.root, manifest, "validation")
    test_paths, test_ids, test_labels = split_records(args.root, manifest, "test")

    run_manifest = {
        "purpose": "locked_historic_61_31_46_final_tl_runs",
        "preflight": bool(args.preflight),
        "models": models,
        "seeds": seeds,
        "protocol_file": str(args.protocol_file),
        "selection_provenance": str(provenance_file),
        "protocol": protocol.as_dict(),
        "split_manifest": str(args.split_manifest),
        "split_counts": {split: int((manifest["split"] == split).sum()) for split in EXPECTED_SPLIT_SIZES},
        "split_class_counts": {
            split: {"0": int(((manifest["split"] == split) & (manifest["observed_label"] == 0)).sum()), "1": int(((manifest["split"] == split) & (manifest["observed_label"] == 1)).sum())}
            for split in EXPECTED_SPLIT_SIZES
        },
        "test_data_accessed": True,
        "selection_performed": False,
        "best_seed_selection_performed": False,
        "runtime": runtime_metadata(REPO_ROOT),
    }
    manifest_json(args.output_dir / "manifests" / "fixed_split_run_manifest.json", run_manifest)
    manifest.to_csv(args.output_dir / "manifests" / "locked_split_manifest_copy.csv", index=False)

    planned_fits = len(models) * len(seeds)
    print(f"Locked split: train={len(train_paths)}, validation={len(validation_paths)}, test={len(test_paths)}")
    print(f"Models: {len(models)}; seeds: {len(seeds)}; planned fits: {planned_fits}; preflight={args.preflight}")
    print(f"Protocol: {protocol.protocol_id}")
    if args.dry_run:
        print("Dry run complete: manifests were written; no model was constructed.")
        return

    all_prediction_tables: List[pd.DataFrame] = []
    all_summary_rows: List[Dict[str, object]] = []
    all_parameter_rows: List[Dict[str, object]] = []

    for model_name in models:
        for seed in seeds:
            print(f"\n=== {model_name} | {protocol.protocol_id} | seed {seed} ===")
            tf.keras.backend.clear_session()
            set_global_seed(seed)
            train_dataset = build_dataset(train_paths, train_labels, model_name, protocol.input_size, protocol.batch_size, training=True)
            train_evaluation_dataset = build_dataset(train_paths, train_labels, model_name, protocol.input_size, protocol.batch_size, training=False)
            validation_dataset = build_dataset(validation_paths, validation_labels, model_name, protocol.input_size, protocol.batch_size, training=False)
            test_dataset = build_dataset(test_paths, test_labels, model_name, protocol.input_size, protocol.batch_size, training=False)

            started = time.perf_counter()
            model, base_model = build_model(model_name, protocol, imagenet_weights=True)
            histories, phase_parameters = run_three_phase_training(
                model, base_model, model_name, protocol, train_dataset, validation_dataset
            )
            elapsed_seconds = time.perf_counter() - started

            train_probabilities = model.predict(train_evaluation_dataset, verbose=0).reshape(-1)
            validation_probabilities = model.predict(validation_dataset, verbose=0).reshape(-1)
            test_probabilities = model.predict(test_dataset, verbose=0).reshape(-1)
            train_metrics = binary_metrics(train_labels, train_probabilities, protocol.threshold)
            validation_metrics = binary_metrics(validation_labels, validation_probabilities, protocol.threshold)
            test_metrics = binary_metrics(test_labels, test_probabilities, protocol.threshold)

            tables = [
                prediction_table(train_ids, train_labels, train_probabilities, split="train", model_name=model_name, protocol=protocol, seed=seed),
                prediction_table(validation_ids, validation_labels, validation_probabilities, split="validation", model_name=model_name, protocol=protocol, seed=seed),
                prediction_table(test_ids, test_labels, test_probabilities, split="test", model_name=model_name, protocol=protocol, seed=seed),
            ]
            for table in tables:
                all_prediction_tables.append(table)
                split = str(table["split"].iloc[0])
                table.to_csv(args.output_dir / "predictions" / f"{split}_{model_name}_{protocol.protocol_id}_seed{seed:03d}.csv", index=False)

            summary_row: Dict[str, object] = {
                "model_name": model_name,
                "protocol_id": protocol.protocol_id,
                "seed": seed,
                "elapsed_seconds": elapsed_seconds,
                **protocol.as_dict(),
            }
            for prefix, metrics in (("train", train_metrics), ("validation", validation_metrics), ("test", test_metrics)):
                summary_row.update({f"{prefix}_{metric}": value for metric, value in metrics.items()})
            all_summary_rows.append(summary_row)
            for phase_record in phase_parameters:
                all_parameter_rows.append({"model_name": model_name, "protocol_id": protocol.protocol_id, "seed": seed, **phase_record})
            manifest_json(
                args.output_dir / "histories" / f"history_{model_name}_{protocol.protocol_id}_seed{seed:03d}.json",
                {
                    "model_name": model_name,
                    "protocol": protocol.as_dict(),
                    "seed": seed,
                    "elapsed_seconds": elapsed_seconds,
                    "phase_histories": histories,
                },
            )
            if args.save_models:
                model.save(args.output_dir / "models" / f"model_{model_name}_{protocol.protocol_id}_seed{seed:03d}.keras")
            print(f"Test AUC={test_metrics['auc']:.4f}; test ACC={test_metrics['accuracy']:.4f}; elapsed={elapsed_seconds:.1f}s")
            tf.keras.backend.clear_session()

    pd.concat(all_prediction_tables, ignore_index=True).to_csv(
        args.output_dir / "predictions" / "patient_level_predictions_all.csv", index=False
    )
    pd.DataFrame(all_summary_rows).to_csv(args.output_dir / "summaries" / "run_level_metrics_descriptive_only.csv", index=False)
    pd.DataFrame(all_parameter_rows).to_csv(args.output_dir / "phase_parameters" / "fixed_split_phase_parameters.csv", index=False)
    print("\nCompleted locked fixed-split batch.")
    print("Run-level metrics are descriptive stability outputs only; no best-seed or run-level inferential test is produced.")


if __name__ == "__main__":
    main()
