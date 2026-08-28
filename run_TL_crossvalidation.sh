#!/bin/bash

# Transfer Learning with 5-Fold Cross-Validation
# Uses: TL_crossvalidation.py
# Modify the 'models' arrays below to select which models to train
# Adjust the counter limit (line 14) to change number of runs per model
# Use 256×256 to match the published Teuho et al. reference-CNN polar-map input size.

echo "Running Experiments..."

models=("ResNet50")

for model in "${models[@]}"
do
    echo "###################################################################################################################"
    counter=1
    until [ $counter -gt 10 ]
    do
        echo $counter
        python TL_crossvalidation.py --model_name $model --input_size 256 --batch_size 5 --epochs 100 --partial_epochs 100 --partial_epochs_2 100 --freeze_fe --early_stopping --optimizer Adam --class_weights --tag $counter
        ((counter++))
    done
    echo "-------------------------------------------------------------------------------------"
done

read -p "Press any key to continue..."
