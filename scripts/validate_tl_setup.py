#!/usr/bin/env python3
"""Fast preflight validation for R1 TL inputs before any GPU fitting.

This utility performs no TensorFlow import and no model construction. It validates the
secure 92-patient development directory and, optionally, the locked 61/31/46 manifest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SPLIT_SIZES = {"train": 61, "validation": 31, "test": 46}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate secure R1 TL input files before GPU execution.")
    parser.add_argument("--root", type=Path, required=True, help="Secure project root containing data/training.")
    parser.add_argument("--split-manifest", type=Path, help="Optional verified fixed 61/31/46 manifest CSV.")
    return parser.parse_args()


def read_binary_labels(path: Path) -> np.ndarray:
    labels = pd.read_csv(path, header=None).iloc[:, 0].astype(int).to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path} contains non-binary labels")
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{path} must contain both classes")
    return labels


def main() -> None:
    args = parse_args()
    development_dir = args.root / "data" / "training"
    images = sorted(development_dir.glob("*.jpg"))
    labels = read_binary_labels(development_dir / "ica_lables.txt")
    if len(images) != 92:
        raise ValueError(f"Expected 92 development JPEGs; found {len(images)} in {development_dir}")
    if len(labels) != 92:
        raise ValueError(f"Expected 92 development labels; found {len(labels)}")
    print(f"Development cohort validated: JPEGs={len(images)}, labels={len(labels)}, class 0={(labels == 0).sum()}, class 1={(labels == 1).sum()}")

    if args.split_manifest:
        manifest = pd.read_csv(args.split_manifest)
        required = {"patient_id", "relative_path", "split", "observed_label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
        if manifest["patient_id"].duplicated().any() or manifest["relative_path"].duplicated().any():
            raise ValueError("Split manifest has duplicate patient IDs or image paths")
        for split, expected in EXPECTED_SPLIT_SIZES.items():
            observed = int((manifest["split"] == split).sum())
            if observed != expected:
                raise ValueError(f"Split '{split}' has {observed} rows; expected {expected}")
        if len(manifest) != 138:
            raise ValueError(f"Expected 138 manifest rows; found {len(manifest)}")
        missing_files = [path for path in manifest["relative_path"] if not (args.root / path).is_file()]
        if missing_files:
            raise FileNotFoundError(f"Manifest references missing image(s), first: {missing_files[0]}")
        print("Locked split manifest validated: train=61, validation=31, test=46; no duplicate IDs/paths; all images exist.")


if __name__ == "__main__":
    main()
