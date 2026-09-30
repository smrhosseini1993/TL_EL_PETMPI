#!/usr/bin/env python3
"""Structural tests for the internal preprocessing-only diagnostic runner."""
from __future__ import print_function

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Experiment2026 import preprocessing_only_ablation as diagnostic


def write_jpeg(path: Path) -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :, 0] = 100
    pixels[:, :, 1] = 50
    pixels[:, :, 2] = 25
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


def test_diagnostic_lock() -> None:
    lock = diagnostic.diagnostic_study_lock_rows()
    values = dict(zip(lock["item"], lock["value"]))
    assert "INTERNAL DIAGNOSTIC ONLY" in values["study"]
    assert "batch_size=5" in values["legacy_conditions_retained"]
    assert "red/green" in values["legacy_conditions_retained"]
    assert "unseeded" in values["legacy_conditions_retained"]
    assert "preprocess_input" in values["preprocessing_only_correction"]
    assert "manuscript" in values["allowed_use"].lower()


def test_training_colour_shift_precedes_official_preprocessing() -> None:
    with tempfile.TemporaryDirectory(prefix="preprocessing_only_pixels_") as temporary:
        image_path = Path(temporary) / "one.jpg"
        write_jpeg(image_path)
        dataset = diagnostic.make_legacy_condition_dataset(
            [image_path], [0], "VGG16", 8, 1, training=True,
        )
        observed_image, _ = next(iter(dataset))
        decoded = diagnostic.tf.io.decode_jpeg(diagnostic.tf.io.read_file(str(image_path)), channels=3)
        decoded = diagnostic.tf.image.resize(decoded, (8, 8))
        decoded = diagnostic.tf.cast(decoded, diagnostic.tf.float32)
        red, green, blue = diagnostic.tf.unstack(decoded, axis=-1)
        legacy_adjusted = diagnostic.tf.stack([
            diagnostic.tf.clip_by_value(red * 1.1, 0.0, 255.0),
            diagnostic.tf.clip_by_value(green * 1.1, 0.0, 255.0),
            blue,
        ], axis=-1)
        expected = diagnostic.core.MODEL_PREPROCESSORS["VGG16"](legacy_adjusted)
        assert np.allclose(observed_image.numpy()[0], expected.numpy(), rtol=0.0, atol=1e-6)


def test_dry_run() -> None:
    with tempfile.TemporaryDirectory(prefix="preprocessing_only_ablation_") as temporary:
        root = Path(temporary)
        secure_root = root / "dataparent"
        output = root / "metrics_preprocessing_only.xlsx"
        create_secure_root(secure_root)
        command = [
            sys.executable,
            str(REPO_ROOT / "Experiment2026" / "preprocessing_only_ablation.py"),
            "--root", str(secure_root),
            "--output-file", str(output),
            "--model-name", "VGG16",
            "--repeat", "1",
            "--dry-run",
        ]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert "INTERNAL DIAGNOSTIC ONLY" in result.stdout
        assert "batch=5" in result.stdout
        assert "red/green x1.1" in result.stdout
        assert "Only learning-pipeline correction" in result.stdout
        assert "Dry run complete" in result.stdout
        assert not output.exists()


def main() -> None:
    test_diagnostic_lock()
    test_training_colour_shift_precedes_official_preprocessing()
    test_dry_run()
    print("Preprocessing-only diagnostic structural tests passed.")


if __name__ == "__main__":
    main()
