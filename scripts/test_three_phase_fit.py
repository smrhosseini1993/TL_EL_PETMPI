#!/usr/bin/env python3
"""Synthetic CPU smoke test for the complete three-phase training implementation.

It uses generated tensors only, no patient files, labels, or ImageNet downloads.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from tl_reanalysis_core import Protocol, build_model, run_three_phase_training, set_global_seed


def main() -> None:
    set_global_seed(43)
    protocol = Protocol(
        protocol_id="synthetic_smoke_test",
        head_learning_rate=1e-4,
        fine_tune_learning_rate=3e-6,
        dropout_rate=0.5,
        phase1_epochs=1,
        phase2_epochs=1,
        phase3_epochs=1,
        early_stopping_patience=1,
    )
    images = np.random.uniform(0, 255, size=(8, 256, 256, 3)).astype("float32")
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype="float32")
    train_dataset = tf.data.Dataset.from_tensor_slices((images[:6], labels[:6])).batch(2)
    validation_dataset = tf.data.Dataset.from_tensor_slices((images[6:], labels[6:])).batch(2)
    model, base_model = build_model("MobileNetV2", protocol, imagenet_weights=False)
    histories, phase_parameters = run_three_phase_training(
        model, base_model, "MobileNetV2", protocol, train_dataset, validation_dataset, verbose=0
    )
    assert [record["phase"] for record in histories] == [1, 2, 3]
    assert all(record["actual_epochs"] >= 1 for record in histories)
    assert phase_parameters[0]["trainable_params"] < phase_parameters[1]["trainable_params"]
    assert phase_parameters[1]["trainable_params"] <= phase_parameters[2]["trainable_params"]
    probabilities = model.predict(validation_dataset, verbose=0)
    assert probabilities.shape == (2, 1)
    assert np.isfinite(probabilities).all()
    print("Synthetic three-phase MobileNetV2 fit passed.")


if __name__ == "__main__":
    main()
