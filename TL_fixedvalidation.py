import os
import argparse

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
    default=299,
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
    return optimizers[str_optimize_name](learning_rate=0.0003)


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
    # The script automatically splits the training folder into:
    #   - First 61 images (2/3) for training
    #   - Last 31 images (1/3) for validation
    #
    # Images are sorted alphabetically by filename before splitting.
    
    print("Reading labels...")
    # Load the ICA label data

    train_labels_file = os.path.join(ROOT, 'data', 'training', 'ica_lables.txt')
    train_labels = genfromtxt(train_labels_file, delimiter='\n', dtype=None)
    train_labels = np.array(train_labels, np.float32)

    test_labels_file = os.path.join(ROOT, 'data', 'test', 'ica_lables.txt')
    test_labels = genfromtxt(test_labels_file, delimiter='\n', dtype=None)
    test_labels = np.array(test_labels, np.float32)

    # Set the polar map resizing parameters here
    x_y_size_images = input_size

    # Load training images, resize and check result
    print("Reading Training Images...")

    images_pattern = os.path.join(ROOT, 'data', 'training', '*.jpg')
    image_paths = sorted(glob.glob(images_pattern))
    
    # Split into training (first 2/3) and validation (last 1/3) like Teuho et al.
    train_count = int(len(image_paths) * 2/3)  # First 2/3 for training (61 images)
    val_count = len(image_paths) - train_count  # Last 1/3 for validation (31 images)
    
    print(f"Total training folder images: {len(image_paths)}")
    print(f"Using first {train_count} for training and last {val_count} for validation")
    
    train_image_paths = image_paths[:train_count]
    train_image_labels = train_labels[:train_count]
    
    val_image_paths = image_paths[train_count:]
    val_image_labels = train_labels[train_count:]
    
    # Create training dataset
    train_ds_raw = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
    train_ds_raw = train_ds_raw.map(
        get_preprocessor(image_size=x_y_size_images),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds_raw.batch(batch_size, drop_remainder=True)
    train_ds_raw = train_ds_raw.batch(batch_size)

    # Create validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((val_image_paths, val_image_labels))
    val_ds = val_ds.map(
        get_preprocessor(image_size=x_y_size_images),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Load test images, resize and check result
    print("Reading Testing Images...")

    images_pattern = os.path.join(ROOT, 'data', 'test', '*.jpg')
    test_image_paths = sorted(glob.glob(images_pattern))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
    test_dataset = test_dataset.map(get_preprocessor(x_y_size_images), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    ###
    # class weights
    ###
    if use_class_weights:
        # Calculate weights based on training set only
        weight_for_0 = (1 / (train_image_labels.shape[0] - np.sum(train_image_labels))) * (train_image_labels.shape[0] / 2.0)
        weight_for_1 = (1 / np.sum(train_image_labels)) * (train_image_labels.shape[0] / 2.0)
        class_weights = {
            0: weight_for_0,
            1: weight_for_1
        }

    ###
    # Load / build the model
    ###

    print("Loading Model...")
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
    model.summary()

    # Optimizer and metrics
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=opt,
        metrics=[tf.keras.metrics.AUC(), 'binary_accuracy']
    )

    # Save initial model
    model_file_name = os.path.join(ROOT, 'results', 'models', f'{normalize_model_name(base_model_class.__name__)}.h5')
    model.save(model_file_name)

    print("Training Model...")
    # Setup callbacks
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        patience=4,
        restore_best_weights=True,
    )
    callbacks = []
    if early_stopping:
        callbacks.append(early_stopping_callback)

    # Use val_ds for validation instead of test_dataset
    tik = time()
    polar_train = model.fit(
        train_ds_raw,
        epochs=epochs,
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
        class_weight=None if not use_class_weights else class_weights,
        validation_data=val_ds,  # Use validation data instead of test data
    )

    if partial_epochs > 0:
        model.layers[0].layers[-2].trainable = True

        print("freeze fe: False")
        model.fit(
            train_ds_raw,
            epochs=partial_epochs + epochs,
            initial_epoch=epochs,
            verbose=2,
            shuffle=True,
            callbacks=callbacks,
            validation_data=val_ds,  # Use validation data instead of test data
            class_weight=None if not use_class_weights else class_weights,
        )

    if partial_epochs_2 > 0:
        model.layers[0].layers[-3].trainable = True

        print("freeze fe: False")
        model.fit(
            train_ds_raw,
            epochs=partial_epochs_2 + partial_epochs + epochs,
            initial_epoch=partial_epochs + epochs,
            verbose=2,
            shuffle=True,
            callbacks=callbacks,
            validation_data=val_ds,  # Use validation data instead of test data
            class_weight=None if not use_class_weights else class_weights,
        )

    toc = time()

    ##############################
    # Evaluate model performance #
    ##############################

    print("Evaluating Model...")
    # Evaluate training performance
    train_eval = model.evaluate(train_ds)
    print('Training accuracy:', train_eval[1])
    
    # Evaluate validation performance
    val_eval = model.evaluate(val_ds)
    print('Validation accuracy:', val_eval[1])

    # Evaluate test performance
    test_eval = model.evaluate(test_dataset)
    print('Test accuracy:', test_eval[1])

    ##############################
    # Calculate detailed metrics #
    ##############################

    # Get predictions for training set
    train_predicts = model.predict(train_ds_raw)
    train_accuracy = accuracy_score(train_image_labels, np.round(train_predicts))
    train_precision = precision_score(train_image_labels, np.round(train_predicts))
    train_recall = recall_score(train_image_labels, np.round(train_predicts))
    train_f1score = f1_score(train_image_labels, np.round(train_predicts))
    train_confusion_matrix = confusion_matrix(train_image_labels, np.round(train_predicts))
    train_roc_auc = roc_auc_score(train_image_labels, train_predicts)
    train_predicts_df = pd.DataFrame(train_predicts, columns=['predict'])

    # Get metrics for validation set
    val_predicts = model.predict(val_ds)
    val_accuracy = accuracy_score(val_image_labels, np.round(val_predicts))
    val_precision = precision_score(val_image_labels, np.round(val_predicts))
    val_recall = recall_score(val_image_labels, np.round(val_predicts))
    val_f1score = f1_score(val_image_labels, np.round(val_predicts))
    val_confusion_matrix = confusion_matrix(val_image_labels, np.round(val_predicts))
    val_roc_auc = roc_auc_score(val_image_labels, val_predicts)
    val_roc = roc_curve(val_image_labels, val_predicts)
    val_predicts_df = pd.DataFrame(val_predicts, columns=['predict'])
    
    # Calculate validation specificity
    val_tn, val_fp, val_fn, val_tp = val_confusion_matrix.ravel()
    val_specificity = val_tn / (val_tn + val_fp)

    # Get predictions for test set
    test_predicts = model.predict(test_dataset)
    test_accuracy = accuracy_score(test_labels, np.round(test_predicts))
    test_precision = precision_score(test_labels, np.round(test_predicts))
    test_recall = recall_score(test_labels, np.round(test_predicts))
    test_f1score = f1_score(test_labels, np.round(test_predicts))
    test_confusion_matrix = confusion_matrix(test_labels, np.round(test_predicts))
    test_roc_auc = roc_auc_score(test_labels, test_predicts)
    test_roc = roc_curve(test_labels, test_predicts)
    test_predicts_df = pd.DataFrame(test_predicts, columns=['predict'])
    
    # Calculate test specificity
    test_tn, test_fp, test_fn, test_tp = test_confusion_matrix.ravel()
    test_specificity = test_tn / (test_tn + test_fp)

    ##############################
    # Save results to Excel file #
    ##############################

    # Save all metrics to the metrics.xlsx file
    result_file_path = os.path.join(ROOT, 'reports', 'metrics.xlsx')
    previous_metrics_df = None
    if os.path.exists(result_file_path):
        previous_metrics_df = pd.read_excel(result_file_path)

    # Include all metrics in results
    results_dict = {
        "model_name": base_model_class.__name__,
        "elapsed_time": toc - tik,
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
        
        # Training metrics
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1score": train_f1score,
        "train_confusion_matrix": str(train_confusion_matrix),
        "train_predicts": ",".join([f"{n[0]:.4f}" for n in train_predicts.tolist()]),
        
        # Validation metrics
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall, 
        "val_f1score": val_f1score,
        "val_auc": val_roc_auc,
        "val_specificity": val_specificity,
        "val_confusion_matrix": str(val_confusion_matrix),
        "val_predicts": ",".join([f"{n[0]:.4f}" for n in val_predicts.tolist()]),
        "val_roc_curve_fpr": ",".join([f"{n:.4f}" for n in val_roc[0].tolist()]),
        "val_roc_curve_tpr": ",".join([f"{n:.4f}" for n in val_roc[1].tolist()]),
        "val_roc_curve_th": ",".join([f"{n:.4f}" for n in val_roc[2].tolist()]),
        
        # Test metrics
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1score": test_f1score,
        "test_auc": test_roc_auc,
        "test_specificity": test_specificity,
        "test_confusion_matrix": str(test_confusion_matrix),
        "test_predicts": ",".join([f"{n[0]:.4f}" for n in test_predicts.tolist()]),
        "test_roc_curve_fpr": ",".join([f"{n:.4f}" for n in test_roc[0].tolist()]),
        "test_roc_curve_tpr": ",".join([f"{n:.4f}" for n in test_roc[1].tolist()]),
        "test_roc_curve_th": ",".join([f"{n:.4f}" for n in test_roc[2].tolist()]),
        
        "tag": tag,
    }
    
    print(results_dict)

    # Save to metrics.xlsx
    results_tmp_df = pd.DataFrame([results_dict])

    if previous_metrics_df is not None:
        previous_metrics_df = pd.concat([previous_metrics_df, results_tmp_df], ignore_index=True)
    else:
        previous_metrics_df = results_tmp_df

    with pd.ExcelWriter(result_file_path) as writer:
        previous_metrics_df.to_excel(writer, sheet_name='sheet1', index=False)

    # Save predictions to separate files
    labels_file_name_template = f"{results_dict['model_name']}-{results_dict['input_size']}-{results_dict['batch_size']}-{results_dict['epochs']}"
    print(f"Saving predictions for experiment {labels_file_name_template}")

    results_root = "results"
    
    # Save training predictions
    file_name = os.path.join(ROOT, results_root, f"train-{labels_file_name_template}.xlsx")
    with pd.ExcelWriter(file_name) as writer:
        train_predicts_df.to_excel(writer, sheet_name='sheet1', index=False)
    
    # Save validation predictions
    file_name = os.path.join(ROOT, results_root, f"val-{labels_file_name_template}.xlsx")
    with pd.ExcelWriter(file_name) as writer:
        val_predicts_df.to_excel(writer, sheet_name='sheet1', index=False)

    # Save test predictions
    file_name = os.path.join(ROOT, results_root, f"test-{labels_file_name_template}.xlsx")
    with pd.ExcelWriter(file_name) as writer:
        test_predicts_df.to_excel(writer, sheet_name='sheet1', index=False)

    # Save model file
    model_file_name = os.path.join(ROOT, 'results', 'models', f'{normalize_model_name(base_model_class.__name__)}.h5')
    model.save(model_file_name)

    print(f"Done Saving Experiment for Model {base_model_class.__name__}")