#!/usr/bin/env python3
"""Build a reviewable 61/31/46 split manifest from secure image directories.

The script does not infer the historic split from alphabetical order.  Supply the
original secure patient-order file(s), one patient/image ID per line, to preserve the
published partition.  The generated CSV must be inspected before fitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a locked R1 61/31/46 split CSV from secure ID lists.")
    parser.add_argument("--root", type=Path, required=True, help="Secure root containing all image files.")
    parser.add_argument("--train-ids", type=Path, required=True, help="61 original training IDs, one per line.")
    parser.add_argument("--validation-ids", type=Path, required=True, help="31 original validation IDs, one per line.")
    parser.add_argument("--test-ids", type=Path, required=True, help="46 original test IDs, one per line.")
    parser.add_argument("--label-table", type=Path, required=True, help="CSV with patient_id and observed_label columns.")
    parser.add_argument("--image-directory", type=Path, required=True, help="Secure image directory relative to root, e.g. data/training.")
    parser.add_argument("--test-image-directory", type=Path, required=True, help="Secure test-image directory relative to root, e.g. data/test.")
    parser.add_argument("--output", type=Path, required=True, help="Output secure manifest CSV.")
    return parser.parse_args()


def load_ids(path: Path, expected: int) -> List[str]:
    identifiers = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(identifiers) != expected:
        raise ValueError(f"{path} contains {len(identifiers)} IDs; expected {expected}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path} contains duplicate IDs")
    return identifiers


def main() -> None:
    args = parse_args()
    train_ids = load_ids(args.train_ids, 61)
    validation_ids = load_ids(args.validation_ids, 31)
    test_ids = load_ids(args.test_ids, 46)
    all_ids = train_ids + validation_ids + test_ids
    if len(set(all_ids)) != 138:
        raise ValueError("Patient IDs overlap between train, validation, and test lists")

    labels = pd.read_csv(args.label_table, dtype={"patient_id": str})
    if not {"patient_id", "observed_label"}.issubset(labels.columns):
        raise ValueError("label-table must contain patient_id and observed_label columns")
    if labels["patient_id"].duplicated().any():
        raise ValueError("label-table contains duplicate patient_id values")
    label_map = labels.set_index("patient_id")["observed_label"].astype(int).to_dict()

    rows = []
    for identifier, split, directory in (
        *((identifier, "train", args.image_directory) for identifier in train_ids),
        *((identifier, "validation", args.image_directory) for identifier in validation_ids),
        *((identifier, "test", args.test_image_directory) for identifier in test_ids),
    ):
        if identifier not in label_map:
            raise ValueError(f"No label found for {identifier}")
        candidates = sorted((args.root / directory).glob(f"{identifier}.*"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one image for {identifier} in {args.root / directory}; found {len(candidates)}")
        rows.append(
            {
                "patient_id": identifier,
                "relative_path": str(candidates[0].relative_to(args.root)),
                "split": split,
                "observed_label": label_map[identifier],
            }
        )
    manifest = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {args.output}; counts: {manifest['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
