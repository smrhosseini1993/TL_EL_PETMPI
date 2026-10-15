#!/bin/bash

# Transfer Learning with Validation Split (2:1 train/val)
# Uses: TL_fixedvalidation.py
# Modify the 'models' array below to select which models to train
# Adjust the counter limit below to change the number of runs per model.

echo "Running Experiments..."

# Eleven architectures evaluated in the manuscript.
models=(
    "DenseNet169"
    "DenseNet201"
    "InceptionResNetV2"
    "InceptionV3"
    "MobileNetV2"
    "ResNet101"
    "ResNet152"
    "ResNet50"
    "VGG16"
    "VGG19"
    "Xception"
)

for model in "${models[@]}"
do
    echo "###################################################################################################################"
    counter=1
    until [ $counter -gt 100 ]
    do
        echo $counter
        python TL_fixedvalidation.py --model_name $model --input_size 128 --batch_size 5 --epochs 30 --partial_epochs 30 --partial_epochs_2 30 --freeze_fe --early_stopping --optimizer Adam --class_weights --tag $counter
        ((counter++))
    done
    echo "-------------------------------------------------------------------------------------"
done

read -p "Press any key to continue..."
