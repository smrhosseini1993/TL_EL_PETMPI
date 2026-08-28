#!/bin/bash

# Transfer Learning with Validation Split (2:1 train/val)
# Uses: TL_fixedvalidation.py
# Modify the 'models' array below to select which models to train
# Adjust the counter limit (line 39) to change number of runs per model
# Use 256×256 to match the published Teuho et al. reference-CNN polar-map input size.

echo "Running Experiments..."

# All available models from your code
# models=(
#     "DenseNet201"
#     "DenseNet169"
#     "ResNet50"
#     "ResNet101"
#     "ResNet152"
#     "InceptionV3"
#     "InceptionResNetV2"
#     "VGG16"
#     "VGG19"
#     "Xception"
#     "EfficientNetB0"
#     "EfficientNetB1"
#     "EfficientNetB2"
#     "EfficientNetB3"
#     "EfficientNetB4"
#     "EfficientNetB5"
#     "EfficientNetB6"
#     "EfficientNetB7"
#     "NASNetLarge"
#     "NASNetMobile"
#     "MobileNetV2"
#     "EfficientNetV2B0"
#     "EfficientNetV2B1"
#     "EfficientNetV2B2"
# )

# Choose the models you want to run
models=("VGG19")

for model in "${models[@]}"
do
    echo "###################################################################################################################"
    counter=1
    until [ $counter -gt 80 ]
    do
        echo $counter
        python TL_fixedvalidation.py --model_name $model --input_size 256 --batch_size 10 --epochs 100 --partial_epochs 100 --partial_epochs_2 100 --freeze_fe --early_stopping --optimizer Adam --class_weights --tag $counter
        ((counter++))
    done
    echo "-------------------------------------------------------------------------------------"
done

read -p "Press any key to continue..."
