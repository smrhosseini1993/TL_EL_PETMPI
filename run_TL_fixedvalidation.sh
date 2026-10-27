#!/bin/bash
# Fixed-split transfer-learning benchmark: 11 architectures × 100 independent runs.

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
    for run in $(seq 1 100); do
        echo "Running ${model}: run ${run}/100"
        python TL_fixedvalidation.py \
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
