#!/usr/bin/env python3
"""CPU-only validation of all R1 Keras backbone registries.

It builds each full model without downloading ImageNet weights, validates that phases 2
and 3 select trainable native backbone layers, checks parameter-count accounting, and
verifies each official preprocessing function returns finite tensors.
"""
from __future__ import annotations

import gc

import numpy as np
import tensorflow as tf

from tl_reanalysis_core import (
    SUPPORTED_MODELS,
    Protocol,
    build_model,
    parameter_counts,
    set_backbone_trainability,
)


def main() -> None:
    protocol = Protocol(
        protocol_id="registry_validation",
        head_learning_rate=1e-4,
        fine_tune_learning_rate=3e-6,
        dropout_rate=0.5,
    )
    sample = tf.ones((1, 256, 256, 3), dtype=tf.float32) * 127.5
    records = []
    for model_name in SUPPORTED_MODELS:
        tf.keras.backend.clear_session()
        model, base_model = build_model(model_name, protocol, imagenet_weights=False)
        phase1 = set_backbone_trainability(base_model, model_name, 1)
        phase1_counts = parameter_counts(model)
        phase2 = set_backbone_trainability(base_model, model_name, 2)
        phase2_counts = parameter_counts(model)
        phase3 = set_backbone_trainability(base_model, model_name, 3)
        phase3_counts = parameter_counts(model)
        if not phase2 or not phase3:
            raise RuntimeError(f"{model_name}: phase registry selected no layers")
        if phase2_counts["trainable_params"] <= phase1_counts["trainable_params"]:
            raise RuntimeError(f"{model_name}: phase 2 did not increase trainable parameter count")
        if phase3_counts["trainable_params"] < phase2_counts["trainable_params"]:
            raise RuntimeError(f"{model_name}: phase 3 reduced trainable parameter count")
        probabilities = model(sample, training=False).numpy()
        if probabilities.shape != (1, 1) or not np.isfinite(probabilities).all():
            raise RuntimeError(f"{model_name}: invalid forward output")
        records.append(
            {
                "model": model_name,
                "total": phase1_counts["total_params"],
                "p1_trainable": phase1_counts["trainable_params"],
                "p2_trainable": phase2_counts["trainable_params"],
                "p3_trainable": phase3_counts["trainable_params"],
                "p2_layers": len(phase2),
                "p3_layers": len(phase3),
            }
        )
        print(records[-1])
        del model, base_model
        gc.collect()
        tf.keras.backend.clear_session()
    print(f"Validated {len(records)} backbone registries.")


if __name__ == "__main__":
    main()
