import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import datetime
import sys
import logging
from time import time
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# region arg
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name',
    dest='model_name',
    help='Keras Model Name',
    default="Xception",
    type=str
)
parser.add_argument(
    '--input_size',
    dest='input_size',
    help='Image size',
    default=128,
    type=int
)
parser.add_argument(
    '--batch_size',
    dest='batch_size',
    help='Batch Size',
    default=5,
    type=int
)
parser.add_argument(
    '--epochs',
    dest='epochs',
    help='Epochs count',
    default=30,
    type=int
)
parser.add_argument(
    '--optimizer',
    dest='optimizer',
    help='Keras Optimizer Name',
    default="Adam",
    type=str
)
parser.add_argument(
    '--early_stopping',
    dest='early_stopping',
    help='Use Early Stopping?',
    action='store_true'
)
parser.add_argument(
    '--class_weights',
    dest='class_weights',
    help='Use Class Weights?',
    action='store_true'
)
parser.add_argument(
    '--tag',
    dest='tag',
    help='custom tag',
    default="tag",
    type=str
)
parser.add_argument(
    '--seed',
    dest='seed',
    help='Random seed recorded with this run',
    required=True,
    type=int
)
parser.add_argument(
    '--training_intervals',
    dest='training_intervals',
    help='Number of sequential classifier-head training intervals',
    default=3,
    type=int
)
parser.add_argument(
    '--interval_epochs',
    dest='interval_epochs',
    help='Maximum epochs in each classifier-head training interval',
    default=30,
    type=int
)
args = parser.parse_args()
# endregion
from tensorflow import keras

import datetime
import glob
import logging
import sys
from time import time

import pandas as pd

from PIL import Image
from numpy import genfromtxt
from skimage.transform import resize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow.data import Dataset
from tensorflow.keras.applications.densenet import DenseNet169, DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception

# The project root is the directory containing this script and the following structure:
#   ROOT/
#   ├── data/
#   │   ├── training/
#   │   │   ├── *.jpg (training images)
#   │   │   └── ica_lables.txt (training labels)
#   │   └── test/
#   │       ├── *.jpg (test images)
#   │       └── ica_lables.txt (test labels)
#   ├── results/
#   │   └── models/
#   ├── reports/
#   └── log/
ROOT = os.path.dirname(os.path.abspath(__file__))


class Log(object):
    def __init__(self):
        self.orgstdout = sys.stdout
        self.log = open(
            os.path.join(ROOT, f"log/log - {str(datetime.datetime.now()).replace(':', '-')}.txt"),
            "a",
        )

    def write(self, msg):
        self.orgstdout.write(msg)
        self.log.write(msg)

    def flush(self):
        self.orgstdout.flush()
        self.log.flush()


# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model(str_model_name):
    models = {
        "DenseNet201": DenseNet201,
        "DenseNet169": DenseNet169,
        "ResNet50": ResNet50,
        "ResNet101": ResNet101,
        "ResNet152": ResNet152,
        "InceptionV3": InceptionV3,
        "InceptionResNetV2": InceptionResNetV2,
        "VGG16": VGG16,
        "VGG19": VGG19,
        "Xception": Xception,
        "MobileNetV2": MobileNetV2,
    }
    return models[str_model_name]


def get_optimizer(str_optimize_name):
    optimizers = {
        "Adam": tf.keras.optimizers.Adam,
        "Adagrad": tf.keras.optimizers.Adagrad,
        "SGD": tf.keras.optimizers.SGD,
        "RMSprop": tf.keras.optimizers.RMSprop,
        "Ftrl": tf.keras.optimizers.Ftrl,
    }
    return optimizers[str_optimize_name](learning_rate=0.0001)


def normalize_model_name(name):
    return name


def get_preprocessor(image_size):
    def preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    return preprocess_image


if __name__ == '__main__':
    for relative_path in ('log', 'results', 'results/models', 'reports'):
        os.makedirs(os.path.join(ROOT, relative_path), exist_ok=True)
    #########################################
    # Create a log file from console output #
    #########################################
    sys.stdout = Log()

    #######################
    # get input arguments #
    #######################
    print("version", tf.__version__)
    print(tf.config.list_physical_devices("GPU"))

    base_model_class = get_model(args.model_name)
    input_size = args.input_size
    batch_size = args.batch_size
    epochs = args.epochs
    training_intervals = args.training_intervals
    interval_epochs = args.interval_epochs
    epochs = interval_epochs
    opt = get_optimizer(args.optimizer)
    early_stopping = args.early_stopping
    use_class_weights = args.class_weights
    tag = args.tag
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"Experiment Configurations: {args.__dict__}")

    ########
    # Main #
    ########
    
    #############################
    # Data Preparation Required #
    #############################
    # Expected data directory structure:
    #   ROOT/data/training/
    #       - *.jpg files (training images, 92 total)
    #       - ica_lables.txt (labels file, one label per line, 0 or 1)
    #   ROOT/data/test/
    #       - *.jpg files (test images, 46 total)
    #       - ica_lables.txt (labels file, one label per line, 0 or 1)
    #
    # This script performs 5-fold cross-validation on the training data (92 images).
    # Each fold uses ~73 images for training and ~19 images for validation.
    # The held-out test set is not read by this cross-validation script.
    #
    # Images are sorted alphabetically by filename before splitting into folds.
    
    print("Reading labels...")
    # Load only training data for cross-validation
    train_labels_file = os.path.join(ROOT, 'data', 'training', 'ica_lables.txt')
    train_labels = genfromtxt(train_labels_file, delimiter='\n', dtype=None)
    train_labels = np.array(train_labels, np.float32)

    # Set the polar map resizing parameters here
    x_y_size_images = input_size

    # Load training images
    print("Reading Training Images...")
    images_pattern = os.path.join(ROOT, 'data', 'training', '*.jpg')
    image_paths = sorted(glob.glob(images_pattern))
    
    # Setup 5-fold cross-validation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize storage for cross-validation results
    all_val_metrics = []
    
    # For each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f"\n==== Processing Fold {fold+1}/{n_folds} ====")
        
        # Split the data for this fold
        fold_train_paths = [image_paths[i] for i in train_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_paths = [image_paths[i] for i in val_idx]  
        fold_val_labels = train_labels[val_idx]
        
        print(f"Training samples: {len(fold_train_paths)}, Validation samples: {len(fold_val_paths)}")
        
        # Create TF datasets for training data
        train_ds_raw = tf.data.Dataset.from_tensor_slices((fold_train_paths, fold_train_labels))
        train_ds_raw = train_ds_raw.map(
            get_preprocessor(image_size=x_y_size_images),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds_raw.batch(batch_size, drop_remainder=True)
        train_ds_raw = train_ds_raw.batch(batch_size)
        
        # Create TF datasets for validation data
        val_dataset = tf.data.Dataset.from_tensor_slices((fold_val_paths, fold_val_labels))
        val_dataset = val_dataset.map(
            get_preprocessor(x_y_size_images),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        ###
        # class weights
        ###
        if use_class_weights:
            weight_for_0 = (1 / (fold_train_labels.shape[0] - np.sum(fold_train_labels))) * (fold_train_labels.shape[0] / 2.0)
            weight_for_1 = (1 / np.sum(fold_train_labels)) * (fold_train_labels.shape[0] / 2.0)
            class_weights = {
                0: weight_for_0,
                1: weight_for_1
            }
        else:
            class_weights = None

        ###
        # Load / build the model
        ###
        print(f"Loading Model for fold {fold+1}...")
        base_model = base_model_class(
            include_top=False,
            weights="imagenet",
            input_shape=(x_y_size_images, x_y_size_images, 3),
        )

        # The pretrained ImageNet base remains frozen; only the classifier head is optimized.
        base_model.trainable = False

        # build model
        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification
            ]
        )

        # Optimizer and metrics 
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=opt,
            metrics=[tf.keras.metrics.AUC(), 'binary_accuracy']
        )


        # Train the frozen-base classifier head in sequential intervals.
        print(f"Training model for fold {fold + 1}...")
        tik = time()
        for interval in range(training_intervals):
            start_epoch = interval * interval_epochs
            end_epoch = (interval + 1) * interval_epochs
            print(f"Classifier-head training interval {interval + 1}/{training_intervals}")
            interval_callbacks = []
            if early_stopping:
                interval_callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_binary_accuracy', patience=4, restore_best_weights=True,
                    )
                )
            model.fit(
                train_ds_raw,
                initial_epoch=start_epoch,
                epochs=end_epoch,
                verbose=2,
                shuffle=True,
                callbacks=interval_callbacks,
                validation_data=val_dataset,
                class_weight=class_weights,
            )
        toc = time()
        training_time = toc - tik

        # Evaluate on validation set
        print(f"Evaluating model for fold {fold+1} on validation data...")
        val_predicts = model.predict(val_dataset)
        val_accuracy = accuracy_score(fold_val_labels, np.round(val_predicts))
        val_precision = precision_score(fold_val_labels, np.round(val_predicts))
        val_recall = recall_score(fold_val_labels, np.round(val_predicts))  # Same as sensitivity
        val_sensitivity = val_recall  # Adding sensitivity explicitly (same as recall)
        val_f1score = f1_score(fold_val_labels, np.round(val_predicts))
        val_confusion_matrix = confusion_matrix(fold_val_labels, np.round(val_predicts))
        val_roc_auc = roc_auc_score(fold_val_labels, val_predicts)
        val_roc = roc_curve(fold_val_labels, val_predicts)
        
        # Calculate specificity
        tn, fp, fn, tp = val_confusion_matrix.ravel()
        val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store validation metrics for this fold
        fold_results = {
            "fold": fold + 1,
            "model_name": base_model_class.__name__,
            "elapsed_time": training_time,
            "input_size": input_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "training_intervals": training_intervals,
            "interval_epochs": interval_epochs,
            "backbone_trainable": False,
            "optimizer": args.optimizer,
            "early_stopping": early_stopping,
            "class_weights": use_class_weights,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_sensitivity": val_sensitivity,  # Added sensitivity
            "val_f1score": val_f1score,
            "val_confusion_matrix": str(val_confusion_matrix),
            "val_specificity": val_specificity,
            "val_auc": val_roc_auc,
            "val_predicts": ",".join([f"{n[0]:.4f}" for n in val_predicts.tolist()]),
            "val_roc_curve_fpr": ",".join([f"{n:.4f}" for n in val_roc[0].tolist()]),
            "val_roc_curve_tpr": ",".join([f"{n:.4f}" for n in val_roc[1].tolist()]),
            "val_roc_curve_th": ",".join([f"{n:.4f}" for n in val_roc[2].tolist()]),
            "tag": tag,
            "seed": seed,
        }
        all_val_metrics.append(fold_results)
        
        # Print fold results
        print(f"\nFold {fold+1} validation results:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Sensitivity/Recall: {val_sensitivity:.4f}")  # Make it clearer they're the same
        print(f"Specificity: {val_specificity:.4f}")
        print(f"F1 Score: {val_f1score:.4f}")
        print(f"AUC: {val_roc_auc:.4f}")
        print(f"Confusion Matrix: {val_confusion_matrix}")
        
        # Clear memory between folds
        tf.keras.backend.clear_session()
    
    # After all folds are processed, save results
    cv_results_df = pd.DataFrame(all_val_metrics)
    
    # Calculate average metrics across all folds
    metrics_cols = ['val_accuracy', 'val_precision', 'val_sensitivity', 
                    'val_f1score', 'val_auc', 'val_specificity', 'elapsed_time']
    avg_metrics = cv_results_df[metrics_cols].mean().to_dict()
    std_metrics = cv_results_df[metrics_cols].std().to_dict()
    
    # Print summary of cross-validation
    print("\n===== Cross-Validation Summary =====")
    print(f"Model: {base_model_class.__name__}")
    print(f"Accuracy: {avg_metrics['val_accuracy']:.4f} ± {std_metrics['val_accuracy']:.4f}")
    print(f"Precision: {avg_metrics['val_precision']:.4f} ± {std_metrics['val_precision']:.4f}")
    print(f"Sensitivity: {avg_metrics['val_sensitivity']:.4f} ± {std_metrics['val_sensitivity']:.4f}")
    print(f"Specificity: {avg_metrics['val_specificity']:.4f} ± {std_metrics['val_specificity']:.4f}")
    print(f"F1 Score: {avg_metrics['val_f1score']:.4f} ± {std_metrics['val_f1score']:.4f}")
    print(f"AUC: {avg_metrics['val_auc']:.4f} ± {std_metrics['val_auc']:.4f}")
    
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Add run timestamp to identify different runs
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for result in all_val_metrics:
        result['run_timestamp'] = run_timestamp
    
    # Single Excel file for all results
    results_file = os.path.join(reports_dir, 'all_cv_fold_results.xlsx')
    
    try:
        # If file exists, load it and append new results
        if os.path.exists(results_file):
            existing_results = pd.read_excel(results_file)
            updated_results = pd.concat([existing_results, cv_results_df], ignore_index=True)
            updated_results.to_excel(results_file, index=False)
            print(f"Appended new results to: {results_file}")
            print(f"File now contains {len(updated_results)} rows")
        else:
            # Create new file
            cv_results_df.to_excel(results_file, index=False)
            print(f"Created new results file: {results_file}")
    except Exception as e:
        # If there's an error, save to a backup file
        backup_file = os.path.join(reports_dir, f'cv_results_backup_{run_timestamp}.xlsx')
        cv_results_df.to_excel(backup_file, index=False)
        print(f"Error saving to main file: {e}")
        print(f"Saved backup to: {backup_file}")
    
    print("\nCross-validation complete! All fold results saved to:")
    print(f"➤ {results_file}")
