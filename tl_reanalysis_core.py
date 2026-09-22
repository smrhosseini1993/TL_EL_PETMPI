"""Shared, non-sensitive utilities for the R1 PET polar-map TL reanalysis.

This module deliberately contains no patient data, paths, labels, or experiment results.
It keeps the two runners consistent with the predeclared R1 protocol.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

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

MODEL_FACTORIES: Mapping[str, Callable[..., keras.Model]] = {
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

MODEL_PREPROCESSORS: Mapping[str, Callable[[tf.Tensor], tf.Tensor]] = {
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

# Native terminal macro-stage prefixes.  Phase 2 unfreezes the first tuple; Phase 3
# unfreezes both tuples.  The resolver verifies every prefix against the local Keras
# installation before a fit starts.  Prefixes are kept as code, not inferred after
# seeing results, so the protocol is fixed and reviewable.
FINE_TUNE_STAGE_PREFIXES: Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "VGG16": (("block5_",), ("block4_",)),
    "VGG19": (("block5_",), ("block4_",)),
    "ResNet50": (("conv5_",), ("conv4_",)),
    "ResNet101": (("conv5_",), ("conv4_",)),
    "ResNet152": (("conv5_",), ("conv4_",)),
    # Inception internals use generic conv2d_<n> names.  Native stage output markers
    # below are resolved by _layers_after_marker rather than guessed conv prefixes.
    "InceptionV3": (("__after_mixed8__",), ("__after_mixed7__",)),
    "InceptionResNetV2": (("__after_mixed_7a__",), ("__after_mixed_6a__",)),
    "DenseNet169": (("conv5_",), ("conv4_", "conv5_")),
    "DenseNet201": (("conv5_",), ("conv4_", "conv5_")),
    "MobileNetV2": (("block_16_", "Conv_1", "out_relu"), ("block_13_", "block_14_", "block_15_", "block_16_", "Conv_1", "out_relu")),
    "Xception": (("block14_",), ("block13_", "block14_")),
}


@dataclass(frozen=True)
class Protocol:
    """One common, development-selected TL protocol."""

    protocol_id: str
    head_learning_rate: float
    fine_tune_learning_rate: float
    dropout_rate: float
    input_size: int = 256
    batch_size: int = 5
    phase1_epochs: int = 30
    phase2_epochs: int = 30
    phase3_epochs: int = 30
    early_stopping_patience: int = 4
    threshold: float = 0.50
    optimizer: str = "Adam"
    loss: str = "binary_crossentropy"
    class_weights: bool = False
    augmentation: str = "none"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Protocol":
        required = {
            "protocol_id",
            "head_learning_rate",
            "fine_tune_learning_rate",
            "dropout_rate",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"Protocol is missing required fields: {missing}")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Protocol contains unsupported fields: {unknown}")
        return cls(**{key: values[key] for key in allowed if key in values})

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def require_supported_model(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Allowed: {', '.join(SUPPORTED_MODELS)}")


def set_global_seed(seed: int) -> None:
    """Set all relevant pseudo-random seeds before constructing a model."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Determinism is best-effort because availability differs by TensorFlow build.
        pass


def make_preprocess_fn(model_name: str, input_size: int) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """Decode RGB JPEGs and apply the official Keras preprocessing for model_name."""
    require_supported_model(model_name)
    preprocess_input = MODEL_PREPROCESSORS[model_name]

    def preprocess_image(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (input_size, input_size), antialias=True)
        image = tf.cast(image, tf.float32)  # keep 0–255 scale for Keras preprocess_input
        image = preprocess_input(image)
        return image, label

    return preprocess_image


def build_dataset(
    paths: Sequence[str],
    labels: Sequence[float],
    model_name: str,
    input_size: int,
    batch_size: int,
    *,
    training: bool,
) -> tf.data.Dataset:
    """Create a deterministic, non-augmented dataset for one split."""
    if len(paths) != len(labels):
        raise ValueError(f"Image/label mismatch: {len(paths)} paths, {len(labels)} labels")
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels, dtype=np.float32)))
    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    dataset = dataset.map(make_preprocess_fn(model_name, input_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return dataset


def _backbone_output(base_model: keras.Model, inputs: keras.KerasTensor) -> keras.KerasTensor:
    # training=False holds BatchNormalization statistics in inference mode during all
    # phases; this is required for stable small-batch fine-tuning.
    return base_model(inputs, training=False)


def build_model(model_name: str, protocol: Protocol, *, imagenet_weights: bool = True) -> Tuple[keras.Model, keras.Model]:
    """Build the retained legacy head on an ImageNet Keras backbone."""
    require_supported_model(model_name)
    factory = MODEL_FACTORIES[model_name]
    weights = "imagenet" if imagenet_weights else None
    inputs = keras.Input(shape=(protocol.input_size, protocol.input_size, 3), name="pet_polar_map")
    base_model = factory(include_top=False, weights=weights, input_tensor=inputs)
    features = _backbone_output(base_model, inputs)
    x = layers.Flatten(name="legacy_flatten")(features)
    x = layers.Dense(1024, activation="relu", name="legacy_dense_1024")(x)
    x = layers.Dropout(protocol.dropout_rate, name="legacy_dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="legacy_dense_512")(x)
    x = layers.Dropout(protocol.dropout_rate, name="legacy_dropout_2")(x)
    x = layers.Dense(256, activation="relu", name="legacy_dense_256")(x)
    x = layers.Dropout(protocol.dropout_rate, name="legacy_dropout_3")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="cad_probability")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{model_name}_legacy_head")
    return model, base_model


def set_backbone_trainability(base_model: keras.Model, model_name: str, phase: int) -> List[str]:
    """Apply the fixed phase-specific terminal-stage registry and return trainable layers.

    Phase 1 freezes the entire backbone.  Phase 2 unfreezes its terminal stage.
    Phase 3 unfreezes its terminal two stages.  BatchNormalization layers always
    remain non-trainable because the backbone is called with training=False.
    """
    if phase not in (1, 2, 3):
        raise ValueError("phase must be 1, 2, or 3")
    require_supported_model(model_name)
    phase2_prefixes, phase3_added_prefixes = FINE_TUNE_STAGE_PREFIXES[model_name]
    trainable_prefixes: Tuple[str, ...] = ()
    if phase == 2:
        trainable_prefixes = phase2_prefixes
    elif phase == 3:
        trainable_prefixes = tuple(dict.fromkeys(phase2_prefixes + phase3_added_prefixes))

    base_model.trainable = True
    if model_name == "InceptionV3" and phase > 1:
        marker = "mixed8" if phase == 2 else "mixed7"
        selected_names = _layers_after_marker(base_model, marker)
        selected_name_set = set(selected_names)
    elif model_name == "InceptionResNetV2" and phase > 1:
        marker = "mixed_7a" if phase == 2 else "mixed_6a"
        selected_names = _layers_after_marker(base_model, marker)
        selected_name_set = set(selected_names)
    else:
        selected_name_set = {
            layer.name for layer in base_model.layers if any(layer.name.startswith(prefix) for prefix in trainable_prefixes)
        }

    for layer in base_model.layers:
        selected = layer.name in selected_name_set
        layer.trainable = selected and not isinstance(layer, layers.BatchNormalization)

    if phase == 1:
        base_model.trainable = False

    if phase > 1:
        selected_names = [layer.name for layer in base_model.layers if layer.trainable]
        if not selected_names:
            raise RuntimeError(
                f"Fine-tuning registry for {model_name} phase {phase} selected no trainable layers. "
                f"Check local Keras layer names and FINE_TUNE_STAGE_PREFIXES."
            )
        return selected_names
    return []


def _layers_after_marker(base_model: keras.Model, marker: str) -> List[str]:
    """Select all flattened Keras layers following a native Inception stage marker."""
    marker_indices = [index for index, layer in enumerate(base_model.layers) if layer.name == marker]
    if len(marker_indices) != 1:
        raise RuntimeError(f"Expected one '{marker}' stage marker; found {len(marker_indices)}")
    return [layer.name for layer in base_model.layers[marker_indices[0] + 1 :]]


def compile_model(model: keras.Model, learning_rate: float) -> None:
    """Compile with a new Adam optimizer after each trainability change."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )


def build_early_stopping(protocol: Protocol) -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy",
        mode="max",
        patience=protocol.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )


def parameter_counts(model: keras.Model) -> Dict[str, int]:
    trainable = int(sum(keras.backend.count_params(weight) for weight in model.trainable_weights))
    non_trainable = int(sum(keras.backend.count_params(weight) for weight in model.non_trainable_weights))
    return {
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }


def run_three_phase_training(
    model: keras.Model,
    base_model: keras.Model,
    model_name: str,
    protocol: Protocol,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    *,
    verbose: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]]]:
    """Run the predeclared three phases and collect reproducible histories/counts."""
    phases = (
        (1, protocol.phase1_epochs, protocol.head_learning_rate),
        (2, protocol.phase2_epochs, protocol.fine_tune_learning_rate),
        (3, protocol.phase3_epochs, protocol.fine_tune_learning_rate),
    )
    phase_histories: List[Dict[str, Any]] = []
    phase_parameters: List[Dict[str, int]] = []
    cumulative_epoch = 0

    for phase, phase_epochs, learning_rate in phases:
        selected_layers = set_backbone_trainability(base_model, model_name, phase)
        compile_model(model, learning_rate)
        counts = parameter_counts(model)
        counts.update({"phase": phase, "selected_backbone_layers": len(selected_layers)})
        phase_parameters.append(counts)

        callbacks = [build_early_stopping(protocol)]
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=cumulative_epoch + phase_epochs,
            initial_epoch=cumulative_epoch,
            callbacks=callbacks,
            shuffle=False,  # training datasets are shuffled explicitly in build_dataset
            verbose=verbose,
        )
        ran_epochs = len(history.history.get("loss", []))
        phase_histories.append(
            {
                "phase": phase,
                "learning_rate": learning_rate,
                "requested_epochs": phase_epochs,
                "actual_epochs": ran_epochs,
                "selected_backbone_layers": selected_layers,
                "history": {key: [float(value) for value in values] for key, values in history.history.items()},
            }
        )
        cumulative_epoch += phase_epochs

    return phase_histories, phase_parameters


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> Dict[str, Any]:
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prediction_table(
    identifiers: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    split: str,
    model_name: str,
    protocol: Protocol,
    seed: int,
    fold: int | None = None,
) -> pd.DataFrame:
    probabilities_array = np.asarray(probabilities, dtype=float).reshape(-1)
    labels_array = np.asarray(labels, dtype=int).reshape(-1)
    if not (len(identifiers) == len(labels_array) == len(probabilities_array)):
        raise ValueError("Prediction records have inconsistent lengths")
    table = pd.DataFrame(
        {
            "patient_id": list(identifiers),
            "split": split,
            "model_name": model_name,
            "protocol_id": protocol.protocol_id,
            "seed": seed,
            "observed_label": labels_array,
            "probability": probabilities_array,
            "threshold": protocol.threshold,
            "binary_prediction": (probabilities_array >= protocol.threshold).astype(int),
        }
    )
    if fold is not None:
        table.insert(3, "fold", fold)
    return table


def assert_binary_labels(labels: Sequence[float], context: str) -> np.ndarray:
    values = np.asarray(labels, dtype=int).reshape(-1)
    if not np.isin(values, [0, 1]).all():
        invalid = sorted(set(values.tolist()) - {0, 1})
        raise ValueError(f"{context} labels must be 0/1; found {invalid}")
    if len(np.unique(values)) != 2:
        raise ValueError(f"{context} must contain both classes for AUC/stratification")
    return values


def load_labels(label_file: Path) -> np.ndarray:
    values = pd.read_csv(label_file, header=None).iloc[:, 0].to_numpy()
    return assert_binary_labels(values, str(label_file))


def image_paths(image_dir: Path) -> List[Path]:
    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg files found in {image_dir}")
    return paths


def verify_paths_and_labels(paths: Sequence[Path], labels: Sequence[int], context: str) -> None:
    if len(paths) != len(labels):
        raise ValueError(f"{context}: {len(paths)} JPEGs but {len(labels)} labels")


def manifest_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def load_protocol(path: Path) -> Protocol:
    with path.open("r", encoding="utf-8") as handle:
        return Protocol.from_mapping(json.load(handle))


def protocol_grid() -> List[Protocol]:
    """Return the prespecified 3×3×3 common protocol grid."""
    protocols: List[Protocol] = []
    for head_lr in (3e-5, 1e-4, 3e-4):
        for fine_lr in (1e-6, 3e-6, 1e-5):
            for dropout in (0.30, 0.50, 0.60):
                protocol_id = f"hlr{head_lr:.0e}_flr{fine_lr:.0e}_do{int(dropout * 100):02d}"
                protocols.append(
                    Protocol(
                        protocol_id=protocol_id,
                        head_learning_rate=head_lr,
                        fine_tune_learning_rate=fine_lr,
                        dropout_rate=dropout,
                    )
                )
    return protocols


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def runtime_metadata(repo_root: Path) -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__ if hasattr(keras, "__version__") else "bundled-with-tensorflow",
        "git_commit": git_commit(repo_root),
        "augmentation": "none",
        "class_weights": False,
        "input_channels": 3,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
