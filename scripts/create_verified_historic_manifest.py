#!/usr/bin/env python3
"""Create the secure historic 61/31/46 manifest from the verified legacy ordering.

Use only after independently confirming that the old fixed-validation code used sorted
training JPEG names, the first 61 for training, the last 31 for validation, and all
sorted test JPEGs for testing. The script refuses any mismatch from the verified
published split sizes and class counts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

EXPECTED = {
    "train": {"size": 61, "class_0": 36, "class_1": 25},
    "validation": {"size": 31, "class_0": 20, "class_1": 11},
    "test": {"size": 46, "class_0": 26, "class_1": 20},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a verified secure historic 61/31/46 split manifest.")
    parser.add_argument("--root", type=Path, required=True, help="Secure root containing data/training and data/test.")
    parser.add_argument("--output", type=Path, required=True, help="Secure output CSV outside the Git working tree.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manifest only after deliberate review.")
    parser.add_argument("--dry-run", action="store_true", help="Print the verified manifest summary without writing a file.")
    return parser.parse_args()


def read_labels(path: Path, expected_count: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    labels = pd.read_csv(path, header=None).iloc[:, 0].astype(int).to_numpy()
    if len(labels) != expected_count:
        raise ValueError(f"{path} has {len(labels)} labels; expected {expected_count}")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path} contains labels other than 0 and 1")
    return labels


def append_rows(rows: List[dict], *, root: Path, paths: List[Path], labels: np.ndarray, split: str) -> None:
    expected = EXPECTED[split]
    if len(paths) != expected["size"]:
        raise ValueError(f"{split} has {len(paths)} images; expected {expected['size']}")
    observed = {0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
    if observed != {0: expected["class_0"], 1: expected["class_1"]}:
        raise ValueError(
            f"{split} class counts are {observed}; expected "
            f"{{0: {expected['class_0']}, 1: {expected['class_1']}}}"
        )
    for path, label in zip(paths, labels):
        rows.append(
            {
                # The source dataset exposes image filenames rather than a separately
                # exportable clinical patient ID. The filename is the stable local image key.
                "patient_id": path.name,
                "relative_path": str(path.relative_to(root)),
                "split": split,
                "observed_label": int(label),
            }
        )


def main() -> None:
    args = parse_args()
    training_dir = args.root / "data" / "training"
    test_dir = args.root / "data" / "test"
    training_paths = sorted(training_dir.glob("*.jpg"))
    test_paths = sorted(test_dir.glob("*.jpg"))
    training_labels = read_labels(training_dir / "ica_lables.txt", expected_count=92)
    test_labels = read_labels(test_dir / "ica_lables.txt", expected_count=46)
    if len(training_paths) != 92:
        raise ValueError(f"Found {len(training_paths)} training JPEGs; expected 92")
    if len(test_paths) != 46:
        raise ValueError(f"Found {len(test_paths)} test JPEGs; expected 46")

    rows: List[dict] = []
    append_rows(rows, root=args.root, paths=training_paths[:61], labels=training_labels[:61], split="train")
    append_rows(rows, root=args.root, paths=training_paths[61:], labels=training_labels[61:], split="validation")
    append_rows(rows, root=args.root, paths=test_paths, labels=test_labels, split="test")
    manifest = pd.DataFrame(rows)
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Stable image keys overlap across splits; manifest cannot be created safely")
    if manifest["relative_path"].duplicated().any():
        raise ValueError("Image paths overlap across splits; manifest cannot be created safely")

    print("Verified historic manifest:")
    for split in ("train", "validation", "test"):
        group = manifest.loc[manifest["split"] == split]
        print(
            f"{split}: n={len(group)}, class 0={int((group['observed_label'] == 0).sum())}, "
            f"class 1={int((group['observed_label'] == 1).sum())}"
        )
    print(f"Training boundary: {training_paths[60].name} | {training_paths[61].name}")
    if args.dry_run:
        print("Dry run complete: no manifest was written.")
        return
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {args.output}. Use --overwrite only after review.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote secure manifest: {args.output}")


if __name__ == "__main__":
    main()
