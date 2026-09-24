#!/usr/bin/env python3
"""Synthetic test for the verified historic 61/31/46 manifest generator."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="historic_manifest_test_") as temp:
        root = Path(temp) / "secure_project"
        training = root / "data" / "training"
        testing = root / "data" / "test"
        training.mkdir(parents=True)
        testing.mkdir(parents=True)
        for index in range(1, 93):
            (training / f"cropped_{index:03d}_smoothed_polar_map.jpg").touch()
        for index in range(1, 47):
            (testing / f"test_{index:03d}_smoothed_polar_map.jpg").touch()
        training_labels = [0] * 36 + [1] * 25 + [0] * 20 + [1] * 11
        test_labels = [0] * 26 + [1] * 20
        (training / "ica_lables.txt").write_text("\n".join(map(str, training_labels)) + "\n", encoding="utf-8")
        (testing / "ica_lables.txt").write_text("\n".join(map(str, test_labels)) + "\n", encoding="utf-8")
        output = Path(temp) / "secure_config" / "locked_historic_61_31_46.csv"
        command = [
            sys.executable,
            "scripts/create_verified_historic_manifest.py",
            "--root",
            str(root),
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Generator failed:\n{completed.stdout}\n{completed.stderr}")
        manifest = pd.read_csv(output)
        assert len(manifest) == 138
        assert manifest["patient_id"].is_unique
        assert manifest["relative_path"].is_unique
        expected = {
            "train": {0: 36, 1: 25},
            "validation": {0: 20, 1: 11},
            "test": {0: 26, 1: 20},
        }
        for split, expected_counts in expected.items():
            group = manifest.loc[manifest["split"] == split]
            observed = {label: int((group["observed_label"] == label).sum()) for label in (0, 1)}
            assert observed == expected_counts, (split, observed)
        print("Verified historic-manifest generator test passed.")


if __name__ == "__main__":
    main()
