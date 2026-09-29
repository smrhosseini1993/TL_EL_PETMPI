#!/usr/bin/env python3
"""Structural tests for the retained four-convolution reference-CNN runner."""
from __future__ import print_function

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Experiment2026 import reference_cnn2026 as cnn


def write_jpeg(path: Path) -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(str(path), format="JPEG")


def create_secure_root(root: Path) -> None:
    training = root / "data" / "training"
    testing = root / "data" / "test"
    training.mkdir(parents=True)
    testing.mkdir(parents=True)
    for index in range(1, 93):
        write_jpeg(training / "cropped_{0:03d}_smoothed_polar_map.jpg".format(index))
    for index in range(1, 47):
        write_jpeg(testing / "cropped_{0:03d}_smoothed_polar_map.jpg".format(index))
    np.savetxt(str(training / "ica_lables.txt"), np.asarray([0] * 36 + [1] * 25 + [0] * 20 + [1] * 11), fmt="%d")
    np.savetxt(str(testing / "ica_lables.txt"), np.asarray([0] * 26 + [1] * 20), fmt="%d")


def test_model_and_metrics() -> None:
    model = cnn.make_model(cnn.INPUT_SIZE)
    assert int(model.count_params()) == 124289
    cnn.compile_model(model)
    values = cnn.calculate_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert values["accuracy"] == 1.0
    assert values["auc"] == 1.0
    assert values["probabilities"].count(",") == 3
    assert values["binary_predictions"] == "0,0,1,1"


def test_dry_run_and_workbook_integrity() -> None:
    with tempfile.TemporaryDirectory(prefix="reference_cnn2026_") as temporary:
        root = Path(temporary)
        secure_root = root / "dataparent"
        output = root / "CNN_metrics2026.xlsx"
        create_secure_root(secure_root)

        command = [
            sys.executable, str(REPO_ROOT / "Experiment2026" / "reference_cnn2026.py"),
            "--root", str(secure_root), "--output-file", str(output), "--seed", "1", "--dry-run",
        ]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert "Locked source-CNN split verified" in result.stdout
        assert "Dry run complete" in result.stdout
        assert not output.exists()

        split_data = cnn.read_split_paths_and_labels(secure_root)
        manifest = cnn.patient_manifest(secure_root, split_data)
        lock = cnn.study_lock_rows()
        row = {
            "model_name": cnn.MODEL_NAME, "seed": 1, "test_probabilities": ",".join(["0.1"] * 46),
            "train_probabilities": ",".join(["0.1"] * 61), "validation_probabilities": ",".join(["0.1"] * 31),
        }
        cnn.append_run_to_workbook(output, row, manifest, lock)
        assert cnn.existing_run(output, 1)
        sheets = pd.read_excel(output, sheet_name=None, engine="openpyxl")
        assert set(sheets) == {"Run_results", "Patient_manifest", "Study_lock"}
        assert len(sheets["Run_results"]) == 1
        try:
            cnn.append_run_to_workbook(output, row, manifest, lock)
            raise AssertionError("Duplicate seed did not raise an error")
        except ValueError as error:
            assert "Duplicate completed reference-CNN seed" in str(error)


def main() -> None:
    test_model_and_metrics()
    test_dry_run_and_workbook_integrity()
    print("Reference-CNN runner structural tests passed.")


if __name__ == "__main__":
    main()
