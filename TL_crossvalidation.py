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
    help='Image size; default 256 matches the published Teuho et al. reference-CNN polar-map representation',
    default=256,
    type=int
)
parser.add_argument(
    '--freeze_fe',
    dest='freeze_fe',
    help='Freeze Feature Extraction?',
    action='store_true'
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
    default=35,
    type=int
)
parser.add_argument(
    '--partial_epochs',
    dest='partial_epochs',
    help='Partial fine tuning phase epochs count',
    default=5,
    type=int
)
parser.add_argument(
    '--partial_epochs_2',
    dest='partial_epochs_2',
    help='Partial fine tuning phase epochs count',
    default=2,
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
    '--is_gray',
    dest='is_gray',
    help='Convert images to  Gray Scale?',
    action='store_true'
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
    '--det',
    dest='det',
    help='Deterministic experiments?',
    action='store_true'
)
args = parser.parse_args()
det = args.det
# endregion
if det:
    seed_number = 43
    os.environ['PYTHONHASHSEED'] = str(seed_number)

    import random

    random.seed(seed_number)

    import numpy as np

    np.random.seed(seed_number)

    import tensorflow as tf

    tf.random.set_seed(seed_number)

    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed_number)
else:
    import random
    import numpy as np
    import tensorflow as tf
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
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
)
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,

)

# Set this to the project root directory containing the following structure:
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
ROOT = '/path/to/project/root'


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
        "EfficientNetB0": EfficientNetB0,
        "EfficientNetB1": EfficientNetB1,
        "EfficientNetB2": EfficientNetB2,
        "EfficientNetB3": EfficientNetB3,
        "EfficientNetB4": EfficientNetB4,
        "EfficientNetB5": EfficientNetB5,
        "EfficientNetB6": EfficientNetB6,
        "EfficientNetB7": EfficientNetB7,
        "NASNetLarge": NASNetLarge,
        "NASNetMobile": NASNetMobile,
        "MobileNetV2": MobileNetV2,
        "EfficientNetV2B0": EfficientNetV2B0,
        "EfficientNetV2B1": EfficientNetV2B1,
        "EfficientNetV2B2": EfficientNetV2B2,
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
    freeze_fe = args.freeze_fe
    batch_size = args.batch_size
    epochs = args.epochs
    partial_epochs = args.partial_epochs
    partial_epochs_2 = args.partial_epochs_2
    opt = get_optimizer(args.optimizer)
    is_gray = args.is_gray
    early_stopping = args.early_stopping
    use_class_weights = args.class_weights
    tag = args.tag

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
    # Test set (46 images) is used for final evaluation only.
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

        if freeze_fe:
            # Use pretrained model as it is, check that layers are frozen
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

        # Setup callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_binary_accuracy',
            patience=4,
            restore_best_weights=True,
        )
        callbacks = []
        if early_stopping:
            callbacks.append(early_stopping_callback)

        # Train the model
        print(f"Training Model for fold {fold+1}...")
        tik = time()
        
        # Initial training phase
        history = model.fit(
            train_ds_raw,
            epochs=epochs,
            verbose=2,
            shuffle=True,
            callbacks=callbacks,
            class_weight=class_weights,
            validation_data=val_dataset,
        )

        # Partial fine-tuning phase 1 if enabled
        if partial_epochs > 0:
            model.layers[0].layers[-2].trainable = True
            print("Partial fine-tuning phase 1")
            history = model.fit(
                train_ds_raw,
                epochs=partial_epochs + epochs,
                initial_epoch=epochs,
                verbose=2,
                shuffle=True,
                callbacks=callbacks,
                validation_data=val_dataset,
                class_weight=class_weights,
            )
            
        # Partial fine-tuning phase 2 if enabled
        if partial_epochs_2 > 0:
            model.layers[0].layers[-3].trainable = True
            print("Partial fine-tuning phase 2")
            history = model.fit(
                train_ds_raw,
                epochs=partial_epochs_2 + partial_epochs + epochs,
                initial_epoch=partial_epochs + epochs,
                verbose=2,
                shuffle=True,
                callbacks=callbacks,
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
            "epochs_partial_1": partial_epochs,
            "epochs_partial_2": partial_epochs_2,
            "freeze_fe": freeze_fe,
            "is_gray": is_gray,
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
