#!/usr/bin/env python3
"""Internal diagnostic: isolate the effect of corrected input preprocessing.

This runner deliberately retains the archived legacy learning conditions:
- historic 61/31/46 alphabetical split;
- 128x128 RGB images;
- batch size 5;
- training-only 10% red/green channel increase;
- frozen ImageNet backbone and literal legacy three-phase calls;
- Adam (0.0003), class weights, dropout 0.50, and early stopping;
- no explicit Python/NumPy/TensorFlow random seed, matching the legacy launcher.

The only intentional learning-pipeline difference from experiments_validation.py is
that each ImageNet backbone receives its official architecture-specific Keras
``preprocess_input`` transform instead of a universal /255 scaling. This is an
internal ablation diagnostic only. Its workbook must never be used for manuscript
results, model selection, ensembles, or clinical inference.
"""
from __future__ import print_function

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from Experiment2026 import experiments2026 as core
except ImportError:
    # Direct execution from Experiment2026/ on the hospital server.
    import experiments2026 as core

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIRECTORY.parent / "dataparent"
DEFAULT_OUTPUT_FILE = SCRIPT_DIRECTORY / "metrics_preprocessing_only.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Internal legacy-condition preprocessing-only PET TL diagnostic."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="dataparent directory containing data/training and data/test.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Diagnostic workbook; never a manuscript-analysis input.",
    )
    parser.add_argument(
        "--model-name",
        choices=core.SUPPORTED_MODELS,
        required=True,
        help="One ImageNet architecture.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        required=True,
        help="Positive bookkeeping repeat ID. It is deliberately not used as a random seed.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=128,
        help="Locked legacy input size (128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        choices=(5,),
        help="Locked legacy mini-batch size (5).",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=100,
        help="Maximum legacy head-training epochs (100).",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=100,
        help="Maximum legacy layer -2 phase epochs (100).",
    )
    parser.add_argument(
        "--phase3-epochs",
        type=int,
        default=100,
        help="Maximum legacy layer -2/-3 phase epochs (100).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip this model/repeat if already present in the diagnostic workbook.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the split and diagnostic lock without constructing a model or writing a workbook.",
    )
    return parser.parse_args()


def make_legacy_condition_dataset(
    paths: Sequence[Path],
    labels: Sequence[int],
    model_name: str,
    input_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    """Reproduce the legacy red/green input alteration, then apply only the fix."""
    preprocess_input = core.MODEL_PREPROCESSORS[model_name]
    path_text = [str(path) for path in paths]
    label_values = np.asarray(labels, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((path_text, label_values))

    def preprocess(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (input_size, input_size))
        image = tf.cast(image, tf.float32)
        if training:
            red, green, blue = tf.unstack(image, axis=-1)
            green = tf.clip_by_value(green * 1.1, 0.0, 255.0)
            red = tf.clip_by_value(red * 1.1, 0.0, 255.0)
            image = tf.stack([red, green, blue], axis=-1)
        # This line is the only intentional learning-pipeline correction versus 2025.
        image = preprocess_input(image)
        return image, label

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # As in the legacy tf.data path, model.fit(shuffle=True) does not reshuffle examples.
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if not training:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def diagnostic_study_lock_rows() -> pd.DataFrame:
    rows = [
        ("study", "INTERNAL DIAGNOSTIC ONLY: preprocessing-only ablation; never manuscript input"),
        ("code_file", "preprocessing_only_ablation.py"),
        ("architectures", ";".join(core.SUPPORTED_MODELS)),
        ("split", "Historic alphabetical fixed split: first 61 development patients=train; next 31=validation; separate 46=test"),
        ("split_class_counts", "train: 36 label-0 / 25 label-1; validation: 20 / 11; test: 26 / 20"),
        ("input", "128 x 128 x 3 RGB JPEG"),
        ("legacy_conditions_retained", "batch_size=5; training-only red/green channels x1.1 with clipping; frozen backbone behavior; unseeded repeats"),
        ("preprocessing_only_correction", "Architecture-specific official Keras preprocess_input replaces the legacy universal /255 scaling"),
        ("augmentation", "Legacy training-only red/green channel x1.1 retained; no MixUp, geometric augmentation, or stochastic colour augmentation"),
        ("classifier_head", "Flatten -> Dense(1024, ReLU) -> Dropout(0.50) -> Dense(512, ReLU) -> Dropout(0.50) -> Dense(256, ReLU) -> Dropout(0.50) -> sigmoid"),
        ("optimizer", "Adam with learning_rate=0.0003"),
        ("batch_size", "5 images per mini-batch"),
        ("loss", "Binary cross-entropy"),
        ("class_weights", "Enabled; calculated from the 61 training labels in every run"),
        ("threshold", "Legacy np.round(probability) conversion; equivalent to a 0.50 threshold except for an exact 0.50 tie"),
        ("early_stopping", "validation binary accuracy; patience=4; restore_best_weights=True; one callback reused across the three literal legacy phases"),
        ("fine_tuning", "Literal legacy schedule: phase 1 freezes backbone; phase 2 marks backbone layer -2 trainable; phase 3 additionally marks backbone layer -3 trainable"),
        ("repeat_identifier", "Run_results.seed stores a bookkeeping repeat ID only; no explicit random seed is applied, to retain legacy unseeded behavior"),
        ("allowed_use", "Internal attribution diagnostic only. Do not use for manuscript results, ensemble selection, clinical comparison, or statistical inference."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def main() -> None:
    args = parse_args()
    if args.input_size != 128:
        raise ValueError("This preprocessing-only diagnostic is locked to --input-size 128.")
    if args.repeat < 1:
        raise ValueError("--repeat must be a positive integer.")
    if min(args.phase1_epochs, args.phase2_epochs, args.phase3_epochs) <= 0:
        raise ValueError("All three phase epoch limits must be positive.")

    args.root = args.root.resolve()
    args.output_file = args.output_file.resolve()
    split_data = core.read_split_data(args.root)
    manifest = core.patient_manifest(args.root, split_data)
    study_lock = diagnostic_study_lock_rows()

    print("INTERNAL DIAGNOSTIC ONLY — never manuscript input")
    print("Locked split verified: train=61 (0=36, 1=25); validation=31 (0=20, 1=11); test=46 (0=26, 1=20)")
    print("Model={0}; repeat={1}; output={2}".format(args.model_name, args.repeat, args.output_file))
    print("Legacy conditions: input=128; batch=5; red/green x1.1 on training images; no explicit seed")
    print("Only learning-pipeline correction: {0}".format(core.PREPROCESSING_LABELS[args.model_name]))

    if args.dry_run:
        print("Dry run complete: secure split and diagnostic lock were validated; no model and no workbook were created.")
        return
    if args.resume and core.existing_run(args.output_file, args.model_name, args.repeat):
        print("Skipping completed diagnostic row for {0}, repeat {1}.".format(args.model_name, args.repeat))
        return

    # Do not call core.set_seed: legacy runs did not pass --det, so they were unseeded.
    tf.keras.backend.clear_session()
    train_dataset = make_legacy_condition_dataset(
        split_data["train"]["paths"], split_data["train"]["labels"], args.model_name,
        args.input_size, args.batch_size, training=True,
    )
    validation_dataset = make_legacy_condition_dataset(
        split_data["validation"]["paths"], split_data["validation"]["labels"], args.model_name,
        args.input_size, args.batch_size, training=False,
    )
    test_dataset = make_legacy_condition_dataset(
        split_data["test"]["paths"], split_data["test"]["labels"], args.model_name,
        args.input_size, args.batch_size, training=False,
    )

    class_weights = core.make_class_weights(split_data["train"]["labels"])
    model, base_model = core.make_model(args.model_name, args.input_size)
    early_stopping_callback = core.make_early_stopping()
    started = time.time()
    phase_1 = core.run_phase(model, base_model, train_dataset, validation_dataset, class_weights, early_stopping_callback, phase=1, phase_epochs=args.phase1_epochs, initial_epoch=0)
    phase_2 = core.run_phase(model, base_model, train_dataset, validation_dataset, class_weights, early_stopping_callback, phase=2, phase_epochs=args.phase2_epochs, initial_epoch=args.phase1_epochs)
    phase_3 = core.run_phase(model, base_model, train_dataset, validation_dataset, class_weights, early_stopping_callback, phase=3, phase_epochs=args.phase3_epochs, initial_epoch=args.phase1_epochs + args.phase2_epochs)
    elapsed_seconds = time.time() - started

    # The legacy script also measured train performance on red/green-shifted training images.
    train_probabilities = model.predict(train_dataset, verbose=0).reshape(-1)
    validation_probabilities = model.predict(validation_dataset, verbose=0).reshape(-1)
    test_probabilities = model.predict(test_dataset, verbose=0).reshape(-1)
    run_row: Dict[str, Any] = {
        "model_name": args.model_name,
        "seed": int(args.repeat),
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
        "augmentation": "legacy_red_green_x1.1_train_only",
        "preprocessing": core.PREPROCESSING_LABELS[args.model_name],
        "random_seed_policy": "legacy_unseeded_repeat; seed column is repeat identifier only",
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
    run_row.update(core.prefixed_metrics("train", core.calculate_metrics(split_data["train"]["labels"], train_probabilities)))
    run_row.update(core.prefixed_metrics("validation", core.calculate_metrics(split_data["validation"]["labels"], validation_probabilities)))
    run_row.update(core.prefixed_metrics("test", core.calculate_metrics(split_data["test"]["labels"], test_probabilities)))
    core.append_run_to_workbook(args.output_file, run_row, manifest, study_lock)
    print("Completed preprocessing-only diagnostic: {0}, repeat {1}; wrote one row to {2}".format(args.model_name, args.repeat, args.output_file))
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
