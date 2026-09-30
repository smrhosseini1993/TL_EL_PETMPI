#!/usr/bin/env python3
"""Seed-varied 100-run reanalysis of the historic four-convolution reference CNN.

This runner retains the architecture and fixed training settings in the supplied
``polar_map_classifier_v2_55_AUC.py`` and ``polar_map_classifier_v2_55_ACC.py``
reference implementations.  Those source files differ only in the metric displayed
while fitting; neither uses that metric for early stopping or model selection.

Intentional 2026 output changes:
* one prespecified random seed per run (1--100), applied before model construction
  and fitting to support a valid repeated-run stability analysis;
* full-precision ordered train/validation/test probabilities in one audited workbook;
* separate 61-patient train and 31-patient validation metrics, while retaining the
  source code's ``validation_split=1/3`` fit behaviour; and
* no test-set selection, best-run reporting, or run-level inferential testing.

The output workbook is secure local data and must never be committed to Git.
"""
from __future__ import print_function

import argparse
import gc
import os
import random
import time
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras import layers, regularizers

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIRECTORY.parent / "dataparent"
DEFAULT_OUTPUT_FILE = SCRIPT_DIRECTORY / "CNN_metrics2026.xlsx"
MODEL_NAME = "ReferenceCNN_4Conv_v2_55"
INPUT_SIZE = 256
BATCH_SIZE = 20
EPOCHS = 35
LEARNING_RATE = 0.005
LEARNING_RATE_DECAY = 1e-8
MOMENTUM = 0.9
CLASS_WEIGHTS = {0: 1.0, 1: 3.0}
RANDOM_SEED_POLICY = "seed_varied_1_to_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one seed of the historic four-convolution PET polar-map reference CNN."
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="dataparent directory containing data/training and data/test."
    )
    parser.add_argument(
        "--output-file", type=Path, default=DEFAULT_OUTPUT_FILE,
        help="Secure local workbook written by this runner (default: CNN_metrics2026.xlsx)."
    )
    parser.add_argument("--seed", type=int, required=True, help="Prespecified random seed (1-100) for one reference-CNN run.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE, help="Locked source-code resize dimension (256).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Locked source-code mini-batch size (20).")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Locked source-code epoch limit (35).")
    parser.add_argument("--resume", action="store_true", help="Skip the seed if it already exists in the workbook.")
    parser.add_argument("--dry-run", action="store_true", help="Validate secure data and arguments without loading images or writing a workbook.")
    return parser.parse_args()


def set_run_random_state(seed: int) -> None:
    """Apply one prespecified seed before constructing and fitting the CNN.

    The architecture, data split, preprocessing, optimizer, class weights, batch
    size, and epoch count remain literal to Jarmo's source. Only the random-state
    policy changes from its fixed constants (NumPy=1, TensorFlow=2) so 100 runs
    represent distinct stochastic training realizations rather than GPU noise.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass


def load_binary_labels(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError("Label file does not exist: {0}".format(path))
    values = np.loadtxt(str(path), dtype=np.int64, ndmin=1)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Labels in {0} must be binary 0/1.".format(path))
    return values


def read_split_paths_and_labels(root: Path) -> Dict[str, Dict[str, Any]]:
    """Validate the historic alphabetical 61/31/46 partition without loading images."""
    training_directory = root / "data" / "training"
    test_directory = root / "data" / "test"
    development_paths = sorted(training_directory.glob("*.jpg"))
    test_paths = sorted(test_directory.glob("*.jpg"))
    development_labels = load_binary_labels(training_directory / "ica_lables.txt")
    test_labels = load_binary_labels(test_directory / "ica_lables.txt")

    if len(development_paths) != 92 or len(development_labels) != 92:
        raise ValueError(
            "Expected 92 development JPEGs/labels; found {0} JPEGs and {1} labels.".format(
                len(development_paths), len(development_labels)
            )
        )
    if len(test_paths) != 46 or len(test_labels) != 46:
        raise ValueError(
            "Expected 46 test JPEGs/labels; found {0} JPEGs and {1} labels.".format(
                len(test_paths), len(test_labels)
            )
        )

    result = {
        "train": {"paths": development_paths[:61], "labels": development_labels[:61]},
        "validation": {"paths": development_paths[61:], "labels": development_labels[61:]},
        "test": {"paths": test_paths, "labels": test_labels},
        "development": {"paths": development_paths, "labels": development_labels},
    }
    expected_counts = {"train": (61, 36, 25), "validation": (31, 20, 11), "test": (46, 26, 20)}
    for split_name, (expected_total, expected_zero, expected_one) in expected_counts.items():
        labels = result[split_name]["labels"]
        observed = (len(labels), int((labels == 0).sum()), int((labels == 1).sum()))
        if observed != (expected_total, expected_zero, expected_one):
            raise ValueError(
                "Historic {0} split mismatch; expected n/0/1={1}, observed={2}.".format(
                    split_name, (expected_total, expected_zero, expected_one), observed
                )
            )
    return result


def _load_one_legacy_image(path: Path, input_size: int) -> np.ndarray:
    """Retain the source's skimage resize -> uint8 -> divide-by-255 sequence."""
    with Image.open(str(path)) as image:
        source = np.asarray(image)
    resized = resize(source, (input_size, input_size, 3))
    requantized = (255.0 * resized).astype(np.uint8)
    # Keeping the uint8/int division also retains the source code's float64
    # NumPy array before Keras performs its standard input cast.
    return requantized / 255


def load_images(paths: Sequence[Path], input_size: int) -> np.ndarray:
    # Do not specify a dtype: Jarmo's ``uint8 / 255`` path yielded a float64
    # NumPy input array before Keras' normal input casting.
    images = np.asarray([_load_one_legacy_image(path, input_size) for path in paths])
    expected_shape = (len(paths), input_size, input_size, 3)
    if images.shape != expected_shape:
        raise RuntimeError("Unexpected loaded image shape: expected {0}, observed {1}.".format(expected_shape, images.shape))
    return images


def make_model(input_size: int) -> keras.Model:
    """Build the supplied four-convolution source architecture without modifications."""
    model = keras.Sequential(name=MODEL_NAME)
    model.add(layers.Conv2D(12, kernel_size=(3, 3), strides=(2, 2), activation="relu", input_shape=(input_size, input_size, 3), padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def compile_model(model: keras.Model) -> None:
    """Implement the source SGD momentum and negligible inverse-time LR decay."""
    schedule = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=1,
        decay_rate=LEARNING_RATE_DECAY,
        staircase=False,
    )
    optimizer = keras.optimizers.SGD(learning_rate=schedule, momentum=MOMENTUM)
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=optimizer,
        # The supplied AUC and ACC variants differed only at this display-only point.
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )


def full_precision_csv(values: Sequence[float]) -> str:
    return ",".join(format(float(value), ".17g") for value in values)


def binary_csv(probabilities: Sequence[float]) -> str:
    return ",".join(str(int(float(value) > 0.50)) for value in probabilities)


def calculate_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=int).reshape(-1)
    y_probability = np.asarray(probabilities, dtype=float).reshape(-1)
    # Keras' source-era predict_classes implementation used ``probability > 0.5``.
    y_prediction = (y_probability > 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "precision": float(precision_score(y_true, y_prediction, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_prediction, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "probabilities": full_precision_csv(y_probability),
        "binary_predictions": binary_csv(y_probability),
    }


def prefixed_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {"{0}_{1}".format(prefix, key): value for key, value in metrics.items()}


def patient_manifest(root: Path, split_data: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        for order, (path, label) in enumerate(zip(split_data[split_name]["paths"], split_data[split_name]["labels"]), start=1):
            rows.append({
                "split": split_name,
                "within_split_order": order,
                "file_name": path.name,
                "relative_path": str(path.relative_to(root)),
                "observed_label": int(label),
            })
    return pd.DataFrame(rows)


def study_lock_rows() -> pd.DataFrame:
    rows = [
        ("study", "EJPH-D-26-00179 seed-varied reference CNN 100-run reanalysis"),
        ("code_file", "reference_cnn2026.py"),
        ("source_implementations", "polar_map_classifier_v2_55_AUC.py and polar_map_classifier_v2_55_ACC.py; source difference was display metric only, not architecture or training"),
        ("random_seed_policy", RANDOM_SEED_POLICY),
        ("random_seed_scope", "Per-run seed 1-100 is applied to Python, NumPy, TensorFlow, and Keras before model construction and fitting. This intentionally replaces the source's fixed NumPy=1 and TensorFlow=2 random-state calls for repeated-run stability analysis only."),
        ("model_name", MODEL_NAME),
        ("split", "Historic alphabetical fixed split: first 61 development patients=train; next 31=validation; separate 46=test"),
        ("split_class_counts", "train: 36 label-0 / 25 label-1; validation: 20 / 11; test: 26 / 20"),
        ("input", "256 x 256 x 3 RGB JPEG; source code uses int(1024/4)"),
        ("image_processing", "Literal source path: np.array(Image.open) -> skimage resize -> multiply by 255 -> uint8 -> divide by 255"),
        ("architecture", "Conv2D(12, 3x3, stride 2, ReLU) -> MaxPool -> Conv2D(16, 3x3, stride 2, ReLU) -> MaxPool -> Conv2D(32, 3x3, stride 2, ReLU) -> MaxPool -> Conv2D(64, 3x3, stride 2, ReLU, L2=0.1) -> MaxPool -> Flatten -> Dense(512, ReLU) -> Dense(128, ReLU) -> sigmoid"),
        ("total_parameters", "124289 for the locked 256 x 256 source architecture"),
        ("optimizer", "SGD: learning_rate=0.005, inverse-time decay=1e-8 per optimizer update, momentum=0.9"),
        ("batch_size", "20"),
        ("epochs", "35; no early stopping"),
        ("loss", "binary cross-entropy"),
        ("class_weights", "Literal source weights: label-0=1, label-1=3"),
        ("fit_validation", "Literal source behavior: model.fit on 92 development images with validation_split=1/3; first 61 rows fit, final 31 rows validate; training rows shuffled by Keras each epoch"),
        ("threshold", "Literal source-era Keras predict_classes threshold: probability >0.50"),
        ("run_identifier", "Run_results seed values 1-100 are the prespecified random seeds applied before model construction and fitting"),
        ("probability_storage", "Full-precision comma-separated train/validation/test probabilities stored once per run; Patient_manifest fixes within-split order"),
        ("analysis_scope", "100 seed rows describe stochastic-training stability only. Patient-level analysis must aggregate probabilities by patient across seeds; no best-run selection or run-level inference is permitted."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.reset_index(drop=True).fillna("").astype(str).equals(right.reset_index(drop=True).fillna("").astype(str))


def _read_existing_workbook(output_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not output_file.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sheets = pd.read_excel(str(output_file), sheet_name=None, engine="openpyxl")
    expected = {"Run_results", "Patient_manifest", "Study_lock"}
    if set(sheets) != expected:
        raise ValueError("Existing workbook must contain exactly {0}; found {1}.".format(sorted(expected), sorted(sheets)))
    return sheets["Run_results"], sheets["Patient_manifest"], sheets["Study_lock"]


def existing_run(output_file: Path, seed: int) -> bool:
    if not output_file.exists():
        return False
    runs, _, _ = _read_existing_workbook(output_file)
    return not runs.empty and bool((pd.to_numeric(runs["seed"], errors="raise").astype(int) == int(seed)).any())


def validate_existing_workbook(output_file: Path, manifest: pd.DataFrame, lock: pd.DataFrame) -> pd.DataFrame:
    """Reject any existing workbook that cannot belong to this locked analysis."""
    runs, existing_manifest, existing_lock = _read_existing_workbook(output_file)
    if not existing_manifest.empty and not _same_frame(existing_manifest, manifest):
        raise ValueError("Existing Patient_manifest differs from this locked secure split. Refusing to mix outputs.")
    if not existing_lock.empty and not _same_frame(existing_lock, lock):
        if {"item", "value"}.issubset(existing_lock.columns):
            existing_values = dict(zip(existing_lock["item"].astype(str), existing_lock["value"].astype(str)))
            if existing_values.get("random_seed_policy") != RANDOM_SEED_POLICY:
                raise ValueError(
                    "Existing CNN_metrics2026.xlsx is a fixed-seed audit or uses an unknown seed policy. "
                    "Archive it outside the active output path before starting the seed-varied 1-100 analysis."
                )
        raise ValueError("Existing Study_lock differs from this reference-CNN protocol. Refusing to mix outputs.")
    return runs


def append_run_to_workbook(output_file: Path, run_row: Mapping[str, Any], manifest: pd.DataFrame, lock: pd.DataFrame) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    runs = validate_existing_workbook(output_file, manifest, lock)

    new_row = pd.DataFrame([dict(run_row)])
    if runs.empty:
        runs = new_row
    else:
        if (pd.to_numeric(runs["seed"], errors="raise").astype(int) == int(run_row["seed"])).any():
            raise ValueError("Duplicate completed reference-CNN seed: {0}".format(run_row["seed"]))
        runs = pd.concat([runs, new_row], ignore_index=True, sort=False)
    runs = runs.sort_values("seed").reset_index(drop=True)

    temporary = output_file.with_name(".{0}.temporary.xlsx".format(output_file.stem))
    with pd.ExcelWriter(str(temporary), engine="openpyxl", mode="w") as writer:
        runs.to_excel(writer, sheet_name="Run_results", index=False)
        manifest.to_excel(writer, sheet_name="Patient_manifest", index=False)
        lock.to_excel(writer, sheet_name="Study_lock", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                cell.font = font
    temporary.replace(output_file)


def main() -> None:
    args = parse_args()
    if args.seed < 1 or args.seed > 100:
        raise ValueError("Reference-CNN seed must be within 1-100.")
    if args.input_size != INPUT_SIZE:
        raise ValueError("This retained source CNN is locked to --input-size {0}.".format(INPUT_SIZE))
    if args.batch_size != BATCH_SIZE:
        raise ValueError("This retained source CNN is locked to --batch-size {0}.".format(BATCH_SIZE))
    if args.epochs != EPOCHS:
        raise ValueError("This retained source CNN is locked to --epochs {0}.".format(EPOCHS))

    args.root = args.root.resolve()
    args.output_file = args.output_file.resolve()
    split_data = read_split_paths_and_labels(args.root)
    manifest = patient_manifest(args.root, split_data)
    lock = study_lock_rows()

    print("Locked source-CNN split verified: train=61 (0=36, 1=25); validation=31 (0=20, 1=11); test=46 (0=26, 1=20)")
    print("Model={0}; seed={1}; output={2}".format(MODEL_NAME, args.seed, args.output_file))
    print("Protocol: input=256; batch=20; epochs=35; SGD lr=0.005, momentum=0.9, decay=1e-8; class weights={0:1, 1:3}")
    if args.output_file.exists():
        validate_existing_workbook(args.output_file, manifest, lock)
    if args.dry_run:
        print("Dry run complete: secure split and locked arguments were validated; no images were loaded, model built, or workbook written.")
        return
    if args.resume and existing_run(args.output_file, args.seed):
        print("Skipping completed reference-CNN seed {0}.".format(args.seed))
        return

    tf.keras.backend.clear_session()
    set_run_random_state(args.seed)
    development_images = load_images(split_data["development"]["paths"], args.input_size)
    test_images = load_images(split_data["test"]["paths"], args.input_size)
    development_labels = np.asarray(split_data["development"]["labels"], dtype=np.int64)
    test_labels = np.asarray(split_data["test"]["labels"], dtype=np.int64)

    model = make_model(args.input_size)
    if int(model.count_params()) != 124289:
        raise RuntimeError("Reference CNN parameter count changed unexpectedly: {0}".format(model.count_params()))
    compile_model(model)
    started = time.time()
    history = model.fit(
        development_images,
        development_labels,
        validation_split=1.0 / 3.0,
        batch_size=args.batch_size,
        class_weight=CLASS_WEIGHTS,
        epochs=args.epochs,
        verbose=2,
        shuffle=True,
    )
    elapsed_seconds = time.time() - started

    development_probabilities = model.predict(development_images, batch_size=args.batch_size, verbose=0).reshape(-1)
    test_probabilities = model.predict(test_images, batch_size=args.batch_size, verbose=0).reshape(-1)
    train_probabilities = development_probabilities[:61]
    validation_probabilities = development_probabilities[61:]

    run_row: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "seed": int(args.seed),
        "elapsed_seconds": float(elapsed_seconds),
        "input_size": int(args.input_size),
        "batch_size": int(args.batch_size),
        "optimizer": "SGD",
        "learning_rate": LEARNING_RATE,
        "learning_rate_decay": LEARNING_RATE_DECAY,
        "momentum": MOMENTUM,
        "loss": "binary_crossentropy",
        "dropout_rate": 0.0,
        "threshold": 0.50,
        "class_weights_enabled": True,
        "class_weight_0": CLASS_WEIGHTS[0],
        "class_weight_1": CLASS_WEIGHTS[1],
        "augmentation": "none",
        "preprocessing": "legacy skimage resize -> uint8 requantization -> /255",
        "phase1_requested_epochs": EPOCHS,
        "phase1_actual_epochs": int(len(history.history.get("loss", []))),
        "phase1_total_params": int(model.count_params()),
        "phase1_trainable_params": int(sum(keras.backend.count_params(weight) for weight in model.trainable_weights)),
        "phase1_non_trainable_params": int(sum(keras.backend.count_params(weight) for weight in model.non_trainable_weights)),
        "phase1_selected_layers": "all layers trainable from initialization",
        "phase2_requested_epochs": 0,
        "phase2_actual_epochs": 0,
        "phase2_total_params": int(model.count_params()),
        "phase2_trainable_params": int(sum(keras.backend.count_params(weight) for weight in model.trainable_weights)),
        "phase2_non_trainable_params": int(sum(keras.backend.count_params(weight) for weight in model.non_trainable_weights)),
        "phase2_selected_layers": "not applicable",
        "phase3_requested_epochs": 0,
        "phase3_actual_epochs": 0,
        "phase3_total_params": int(model.count_params()),
        "phase3_trainable_params": int(sum(keras.backend.count_params(weight) for weight in model.trainable_weights)),
        "phase3_non_trainable_params": int(sum(keras.backend.count_params(weight) for weight in model.non_trainable_weights)),
        "phase3_selected_layers": "not applicable",
        "validation_split_fraction": 1.0 / 3.0,
        "fit_development_patients": 92,
        "fit_train_patients": 61,
        "fit_validation_patients": 31,
    }
    run_row.update(prefixed_metrics("train", calculate_metrics(split_data["train"]["labels"], train_probabilities)))
    run_row.update(prefixed_metrics("validation", calculate_metrics(split_data["validation"]["labels"], validation_probabilities)))
    run_row.update(prefixed_metrics("test", calculate_metrics(test_labels, test_probabilities)))
    append_run_to_workbook(args.output_file, run_row, manifest, lock)
    print("Completed reference CNN, seed {0}; wrote one row to {1}".format(args.seed, args.output_file))
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
