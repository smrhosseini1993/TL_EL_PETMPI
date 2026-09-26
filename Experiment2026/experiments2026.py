#!/usr/bin/env python3
"""2026 fixed-split transfer-learning experiment for PET polar maps.

This is a deliberately compact successor to experiments_validation.py. It retains the
legacy 61/31/46 split, the 11 architectures, classifier head, class weighting, Adam
learning rate, dropout, and layer -2/-3 fine-tuning schedule requested for the 2026
rerun. It corrects only the input preprocessing and augmentation paths, and writes one
secure Excel workbook with all run-level results plus the fixed patient-order manifest.

The output workbook contains patient-linked probabilities. It must remain on approved
secure storage and must never be committed to Git.
"""
from __future__ import print_function

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import densenet, inception_resnet_v2, inception_v3, mobilenet_v2, resnet, vgg16, vgg19, xception
from tensorflow.keras.applications.densenet import DenseNet169, DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIRECTORY.parent / "dataparent"
DEFAULT_OUTPUT_FILE = SCRIPT_DIRECTORY / "metrics2026.xlsx"

SUPPORTED_MODELS: Tuple[str, ...] = (
    "VGG16",
    "VGG19",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "InceptionV3",
    "InceptionResNetV2",
    "DenseNet169",
    "DenseNet201",
    "MobileNetV2",
    "Xception",
)

MODEL_FACTORIES: Mapping[str, Any] = {
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "InceptionV3": InceptionV3,
    "InceptionResNetV2": InceptionResNetV2,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "MobileNetV2": MobileNetV2,
    "Xception": Xception,
}

MODEL_PREPROCESSORS: Mapping[str, Any] = {
    "VGG16": vgg16.preprocess_input,
    "VGG19": vgg19.preprocess_input,
    "ResNet50": resnet.preprocess_input,
    "ResNet101": resnet.preprocess_input,
    "ResNet152": resnet.preprocess_input,
    "InceptionV3": inception_v3.preprocess_input,
    "InceptionResNetV2": inception_resnet_v2.preprocess_input,
    "DenseNet169": densenet.preprocess_input,
    "DenseNet201": densenet.preprocess_input,
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "Xception": xception.preprocess_input,
}

PREPROCESSING_LABELS: Mapping[str, str] = {
    "VGG16": "keras.applications.vgg16.preprocess_input (RGB-to-BGR, ImageNet mean-centering)",
    "VGG19": "keras.applications.vgg19.preprocess_input (RGB-to-BGR, ImageNet mean-centering)",
    "ResNet50": "keras.applications.resnet.preprocess_input (RGB-to-BGR, ImageNet mean-centering)",
    "ResNet101": "keras.applications.resnet.preprocess_input (RGB-to-BGR, ImageNet mean-centering)",
    "ResNet152": "keras.applications.resnet.preprocess_input (RGB-to-BGR, ImageNet mean-centering)",
    "InceptionV3": "keras.applications.inception_v3.preprocess_input (0-255 to -1 to +1)",
    "InceptionResNetV2": "keras.applications.inception_resnet_v2.preprocess_input (0-255 to -1 to +1)",
    "DenseNet169": "keras.applications.densenet.preprocess_input (ImageNet RGB normalization)",
    "DenseNet201": "keras.applications.densenet.preprocess_input (ImageNet RGB normalization)",
    "MobileNetV2": "keras.applications.mobilenet_v2.preprocess_input (0-255 to -1 to +1)",
    "Xception": "keras.applications.xception.preprocess_input (0-255 to -1 to +1)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fixed-split 2026 PET TL model/seed and append it to one Excel workbook.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="dataparent directory containing data/training and data/test.")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Only generated result file: metrics2026.xlsx.")
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, required=True, help="One ImageNet architecture.")
    parser.add_argument("--seed", type=int, required=True, help="Prespecified integer seed for this run.")
    parser.add_argument("--input-size", type=int, default=128, help="Fixed retained legacy input size (default: 128).")
    parser.add_argument("--batch-size", type=int, default=5, help="Fixed batch size (default: 5).")
    parser.add_argument("--phase1-epochs", type=int, default=100, help="Maximum head-training epochs (default: 100).")
    parser.add_argument("--phase2-epochs", type=int, default=100, help="Maximum legacy layer -2 fine-tuning epochs (default: 100).")
    parser.add_argument("--phase3-epochs", type=int, default=100, help="Maximum legacy layer -2/-3 fine-tuning epochs (default: 100).")
    parser.add_argument("--resume", action="store_true", help="Skip this model/seed if it is already present in the one workbook.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input data and locked split without constructing a model or writing an output file.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def load_binary_labels(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError("Label file does not exist: {0}".format(path))
    values = np.loadtxt(str(path), dtype=np.int64, ndmin=1)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Labels in {0} must be binary 0/1.".format(path))
    return values


def read_split_data(root: Path) -> Dict[str, Dict[str, Any]]:
    training_directory = root / "data" / "training"
    test_directory = root / "data" / "test"
    training_paths = sorted(training_directory.glob("*.jpg"))
    test_paths = sorted(test_directory.glob("*.jpg"))
    training_labels = load_binary_labels(training_directory / "ica_lables.txt")
    test_labels = load_binary_labels(test_directory / "ica_lables.txt")

    if len(training_paths) != 92 or len(training_labels) != 92:
        raise ValueError("Expected exactly 92 development JPEGs/labels; found {0} JPEGs and {1} labels.".format(len(training_paths), len(training_labels)))
    if len(test_paths) != 46 or len(test_labels) != 46:
        raise ValueError("Expected exactly 46 test JPEGs/labels; found {0} JPEGs and {1} labels.".format(len(test_paths), len(test_labels)))

    data = {
        "train": {"paths": training_paths[:61], "labels": training_labels[:61]},
        "validation": {"paths": training_paths[61:], "labels": training_labels[61:]},
        "test": {"paths": test_paths, "labels": test_labels},
    }
    expected_counts = {"train": (61, 36, 25), "validation": (31, 20, 11), "test": (46, 26, 20)}
    for split, (expected_total, expected_zero, expected_one) in expected_counts.items():
        labels = data[split]["labels"]
        observed = (len(labels), int((labels == 0).sum()), int((labels == 1).sum()))
        if observed != (expected_total, expected_zero, expected_one):
            raise ValueError("Historic {0} split mismatch; expected n/0/1={1}, observed={2}.".format(split, (expected_total, expected_zero, expected_one), observed))
    return data


def patient_manifest(root: Path, split_data: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        paths = split_data[split]["paths"]
        labels = split_data[split]["labels"]
        for index, (path, label) in enumerate(zip(paths, labels), start=1):
            rows.append({
                "split": split,
                "within_split_order": index,
                "file_name": path.name,
                "relative_path": str(path.relative_to(root)),
                "observed_label": int(label),
            })
    return pd.DataFrame(rows)


def make_dataset(paths: Sequence[Path], labels: Sequence[int], model_name: str, input_size: int, batch_size: int, seed: int, training: bool) -> tf.data.Dataset:
    preprocess_input = MODEL_PREPROCESSORS[model_name]
    path_text = [str(path) for path in paths]
    label_values = np.asarray(labels, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((path_text, label_values))

    def preprocess(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (input_size, input_size), antialias=True)
        image = tf.cast(image, tf.float32)
        # Deliberately no /255, no red/green change, no MixUp, and no geometric/color augmentation.
        image = preprocess_input(image)
        return image, label

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # Legacy experiments_validation.py passed an ordered tf.data.Dataset to model.fit.
    # Its shuffle=True argument does not shuffle a Dataset input, so preserve that exact
    # order here rather than introducing a new per-epoch training-data shuffle.
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def make_model(model_name: str, input_size: int) -> Tuple[keras.Model, keras.Model]:
    base_model = MODEL_FACTORIES[model_name](include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3))
    base_model.trainable = False
    model = keras.Sequential([
        base_model,
        layers.Flatten(name="legacy_flatten"),
        layers.Dense(1024, activation="relu", name="legacy_dense_1024"),
        layers.Dropout(0.50, name="legacy_dropout_1"),
        layers.Dense(512, activation="relu", name="legacy_dense_512"),
        layers.Dropout(0.50, name="legacy_dropout_2"),
        layers.Dense(256, activation="relu", name="legacy_dense_256"),
        layers.Dropout(0.50, name="legacy_dropout_3"),
        layers.Dense(1, activation="sigmoid", name="cad_probability"),
    ], name="{0}_legacy_head_2026".format(model_name))
    return model, base_model


def configure_legacy_fine_tuning(base_model: keras.Model, phase: int) -> List[str]:
    """Retain the literal legacy layer -2 then -3 trainability sequence."""
    if phase == 1:
        base_model.trainable = False
        return []
    selected = [-2] if phase == 2 else [-3]
    selected_names: List[str] = []
    for index in selected:
        layer = base_model.layers[index]
        layer.trainable = True
        selected_names.append(layer.name)
    return selected_names


def parameter_counts(model: keras.Model) -> Dict[str, int]:
    trainable = int(sum(keras.backend.count_params(weight) for weight in model.trainable_weights))
    non_trainable = int(sum(keras.backend.count_params(weight) for weight in model.non_trainable_weights))
    return {"total_params": trainable + non_trainable, "trainable_params": trainable, "non_trainable_params": non_trainable}


def compile_model(model: keras.Model) -> None:
    # Recompilation is required after changing trainable flags in Keras.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )


def make_class_weights(train_labels: np.ndarray) -> Dict[int, float]:
    count_zero = int((train_labels == 0).sum())
    count_one = int((train_labels == 1).sum())
    total = len(train_labels)
    return {0: total / (2.0 * count_zero), 1: total / (2.0 * count_one)}


def make_early_stopping() -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", mode="max", patience=4, restore_best_weights=True, verbose=1)


def run_phase(model: keras.Model, base_model: keras.Model, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, class_weights: Mapping[int, float], phase: int, phase_epochs: int, initial_epoch: int) -> Dict[str, Any]:
    selected_layers = configure_legacy_fine_tuning(base_model, phase)
    # The initial compile and the subsequent inner-layer trainability changes reproduce
    # the legacy experiments_validation.py behavior exactly. No revised fine-tuning
    # strategy is introduced in this retained-legacy run.
    if phase == 1:
        compile_model(model)
    counts = parameter_counts(model)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=initial_epoch + phase_epochs,
        initial_epoch=initial_epoch,
        callbacks=[make_early_stopping()],
        class_weight=dict(class_weights),
        shuffle=False,
        verbose=2,
    )
    return {
        "phase": phase,
        "selected_layers": ";".join(selected_layers),
        "actual_epochs": len(history.history.get("loss", [])),
        "requested_epochs": phase_epochs,
        "total_params": counts["total_params"],
        "trainable_params": counts["trainable_params"],
        "non_trainable_params": counts["non_trainable_params"],
    }


def full_precision_csv(values: Sequence[float]) -> str:
    return ",".join(format(float(value), ".17g") for value in values)


def binary_csv(probabilities: Sequence[float], threshold: float) -> str:
    return ",".join(str(int(float(value) >= threshold)) for value in probabilities)


def calculate_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.50) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=int).reshape(-1)
    y_prob = np.asarray(probabilities, dtype=float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "probabilities": full_precision_csv(y_prob),
        "binary_predictions": binary_csv(y_prob, threshold),
    }


def prefixed_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in metrics.items():
        result["{0}_{1}".format(prefix, key)] = value
    return result


def study_lock_rows() -> pd.DataFrame:
    rows = [
        ("study", "EJPH-D-26-00179 2026 fixed-split TL rerun"),
        ("code_file", "experiments2026.py"),
        ("architectures", ";".join(SUPPORTED_MODELS)),
        ("split", "Historic alphabetical fixed split: first 61 development patients=train; next 31=validation; separate 46=test"),
        ("split_class_counts", "train: 36 label-0 / 25 label-1; validation: 20 / 11; test: 26 / 20"),
        ("input", "128 x 128 x 3 RGB JPEG"),
        ("preprocessing", "Architecture-specific official Keras preprocess_input; raw decoded/resized pixels remain 0-255 before that function"),
        ("augmentation", "None: no red/green channel shift, no MixUp, no geometric or colour augmentation"),
        ("classifier_head", "Flatten -> Dense(1024, ReLU) -> Dropout(0.50) -> Dense(512, ReLU) -> Dropout(0.50) -> Dense(256, ReLU) -> Dropout(0.50) -> sigmoid"),
        ("optimizer", "Adam with learning_rate=0.0003"),
        ("loss", "Binary cross-entropy"),
        ("class_weights", "Enabled; calculated from the 61 training labels in every run"),
        ("threshold", "0.50"),
        ("early_stopping", "validation binary accuracy; patience=4; restore_best_weights=True; fresh callback in each phase"),
        ("fine_tuning", "Literal legacy schedule: phase 1 freezes backbone; phase 2 marks backbone layer -2 trainable; phase 3 additionally marks backbone layer -3 trainable; no revised fine-tuning strategy was introduced."),
        ("seed", "Each Run_results row contains one prespecified integer seed; seed is applied to Python, NumPy and TensorFlow."),
        ("probability_storage", "Full precision comma-separated train/validation/test probabilities stored once per run; Patient_manifest fixes their within-split order."),
        ("analysis_scope", "Run-level distributions are descriptive stability only. Patient-level analysis must aggregate predictions by patient across seeds using Patient_manifest."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def read_existing_workbook(output_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not output_file.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sheets = pd.read_excel(str(output_file), sheet_name=None, engine="openpyxl")
    expected = {"Run_results", "Patient_manifest", "Study_lock"}
    observed = set(sheets)
    if observed != expected:
        raise ValueError("Existing workbook must contain exactly {0}; found {1}.".format(sorted(expected), sorted(observed)))
    return sheets["Run_results"], sheets["Patient_manifest"], sheets["Study_lock"]


def same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.reset_index(drop=True).fillna("").astype(str).equals(right.reset_index(drop=True).fillna("").astype(str))


def append_run_to_workbook(output_file: Path, run_row: Mapping[str, Any], manifest: pd.DataFrame) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    runs, existing_manifest, existing_lock = read_existing_workbook(output_file)
    expected_lock = study_lock_rows()
    if not existing_manifest.empty and not same_frame(existing_manifest, manifest):
        raise ValueError("Existing Patient_manifest differs from the current secure data/split. Refusing to mix results.")
    if not existing_lock.empty and not same_frame(existing_lock, expected_lock):
        raise ValueError("Existing Study_lock differs from this code's locked protocol. Refusing to mix results.")
    new_row = pd.DataFrame([dict(run_row)])
    if not runs.empty:
        duplicate = (runs["model_name"].astype(str) == str(run_row["model_name"])) & (runs["seed"].astype(int) == int(run_row["seed"]))
        if duplicate.any():
            raise ValueError("Duplicate completed model/seed in workbook: {0} / {1}.".format(run_row["model_name"], run_row["seed"]))
        runs = pd.concat([runs, new_row], ignore_index=True, sort=False)
    else:
        runs = new_row
    runs = runs.sort_values(["model_name", "seed"]).reset_index(drop=True)
    with pd.ExcelWriter(str(output_file), engine="openpyxl", mode="w") as writer:
        runs.to_excel(writer, sheet_name="Run_results", index=False)
        manifest.to_excel(writer, sheet_name="Patient_manifest", index=False)
        expected_lock.to_excel(writer, sheet_name="Study_lock", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)


def existing_run(output_file: Path, model_name: str, seed: int) -> bool:
    if not output_file.exists():
        return False
    runs, _, _ = read_existing_workbook(output_file)
    if runs.empty:
        return False
    return bool(((runs["model_name"].astype(str) == model_name) & (runs["seed"].astype(int) == seed)).any())


def main() -> None:
    args = parse_args()
    if args.input_size != 128:
        raise ValueError("This retained-legacy 2026 script is locked to --input-size 128.")
    if args.batch_size != 5:
        raise ValueError("This retained-legacy 2026 script is locked to --batch-size 5.")
    if min(args.phase1_epochs, args.phase2_epochs, args.phase3_epochs) <= 0:
        raise ValueError("All three phase epoch limits must be positive.")

    args.root = args.root.resolve()
    args.output_file = args.output_file.resolve()
    split_data = read_split_data(args.root)
    manifest = patient_manifest(args.root, split_data)

    print("Locked fixed split verified: train=61 (0=36, 1=25); validation=31 (0=20, 1=11); test=46 (0=26, 1=20)")
    print("Model={0}; seed={1}; output={2}".format(args.model_name, args.seed, args.output_file))
    print("Preprocessing={0}".format(PREPROCESSING_LABELS[args.model_name]))
    print("Augmentation=None; class weights=enabled; Adam LR=0.0003; dropout=0.50; input=128")

    if args.dry_run:
        print("Dry run complete: secure split, manifest order and locked arguments were validated; no model and no workbook were created.")
        return
    if args.resume and existing_run(args.output_file, args.model_name, args.seed):
        print("Skipping completed workbook row for {0}, seed {1}.".format(args.model_name, args.seed))
        return

    set_seed(args.seed)
    tf.keras.backend.clear_session()
    train_dataset = make_dataset(split_data["train"]["paths"], split_data["train"]["labels"], args.model_name, args.input_size, args.batch_size, args.seed, training=True)
    train_evaluation_dataset = make_dataset(split_data["train"]["paths"], split_data["train"]["labels"], args.model_name, args.input_size, args.batch_size, args.seed, training=False)
    validation_dataset = make_dataset(split_data["validation"]["paths"], split_data["validation"]["labels"], args.model_name, args.input_size, args.batch_size, args.seed, training=False)
    test_dataset = make_dataset(split_data["test"]["paths"], split_data["test"]["labels"], args.model_name, args.input_size, args.batch_size, args.seed, training=False)

    class_weights = make_class_weights(split_data["train"]["labels"])
    model, base_model = make_model(args.model_name, args.input_size)
    started = time.time()
    phase_1 = run_phase(model, base_model, train_dataset, validation_dataset, class_weights, phase=1, phase_epochs=args.phase1_epochs, initial_epoch=0)
    phase_2 = run_phase(model, base_model, train_dataset, validation_dataset, class_weights, phase=2, phase_epochs=args.phase2_epochs, initial_epoch=args.phase1_epochs)
    phase_3 = run_phase(model, base_model, train_dataset, validation_dataset, class_weights, phase=3, phase_epochs=args.phase3_epochs, initial_epoch=args.phase1_epochs + args.phase2_epochs)
    elapsed_seconds = time.time() - started

    train_probabilities = model.predict(train_evaluation_dataset, verbose=0).reshape(-1)
    validation_probabilities = model.predict(validation_dataset, verbose=0).reshape(-1)
    test_probabilities = model.predict(test_dataset, verbose=0).reshape(-1)
    run_row: Dict[str, Any] = {
        "model_name": args.model_name,
        "seed": int(args.seed),
        "elapsed_seconds": float(elapsed_seconds),
        "input_size": int(args.input_size),
        "batch_size": int(args.batch_size),
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "loss": "binary_crossentropy",
        "dropout_rate": 0.50,
        "threshold": 0.50,
        "class_weights_enabled": True,
        "class_weight_0": float(class_weights[0]),
        "class_weight_1": float(class_weights[1]),
        "augmentation": "none",
        "preprocessing": PREPROCESSING_LABELS[args.model_name],
        "phase1_requested_epochs": phase_1["requested_epochs"],
        "phase1_actual_epochs": phase_1["actual_epochs"],
        "phase1_total_params": phase_1["total_params"],
        "phase1_trainable_params": phase_1["trainable_params"],
        "phase1_non_trainable_params": phase_1["non_trainable_params"],
        "phase1_selected_layers": phase_1["selected_layers"],
        "phase2_requested_epochs": phase_2["requested_epochs"],
        "phase2_actual_epochs": phase_2["actual_epochs"],
        "phase2_total_params": phase_2["total_params"],
        "phase2_trainable_params": phase_2["trainable_params"],
        "phase2_non_trainable_params": phase_2["non_trainable_params"],
        "phase2_selected_layers": phase_2["selected_layers"],
        "phase3_requested_epochs": phase_3["requested_epochs"],
        "phase3_actual_epochs": phase_3["actual_epochs"],
        "phase3_total_params": phase_3["total_params"],
        "phase3_trainable_params": phase_3["trainable_params"],
        "phase3_non_trainable_params": phase_3["non_trainable_params"],
        "phase3_selected_layers": phase_3["selected_layers"],
    }
    run_row.update(prefixed_metrics("train", calculate_metrics(split_data["train"]["labels"], train_probabilities)))
    run_row.update(prefixed_metrics("validation", calculate_metrics(split_data["validation"]["labels"], validation_probabilities)))
    run_row.update(prefixed_metrics("test", calculate_metrics(split_data["test"]["labels"], test_probabilities)))
    append_run_to_workbook(args.output_file, run_row, manifest)
    print("Completed {0}, seed {1}; wrote one row to {2}".format(args.model_name, args.seed, args.output_file))
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
