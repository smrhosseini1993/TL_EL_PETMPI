#!/bin/bash
# Development-ranking workflow: 11 architectures × 10 repeated 5-fold CV runs.

set -euo pipefail

models=(
    "DenseNet169"
    "DenseNet201"
    "InceptionResNetV2"
    "InceptionV3"
    "MobileNetV2"
    "ResNet50"
    "ResNet101"
    "ResNet152"
    "VGG16"
    "VGG19"
    "Xception"
)

for model in "${models[@]}"; do
    for run in $(seq 1 10); do
        echo "Running ${model}: CV repetition ${run}/10"
        python TL_crossvalidation.py \
            --model_name "$model" \
            --input_size 128 \
            --batch_size 5 \
            --training_intervals 3 \
            --interval_epochs 30 \
            --early_stopping \
            --optimizer Adam \
            --class_weights \
            --tag "$run" \
            --seed "$run"
    done
done
