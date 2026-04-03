"""
Helper functions for preprocessing, evaluation, plotting, and inference.
"""

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.models import load_model


CLASS_NAMES = [str(i) for i in range(10)]


def validate_dataset_structure(dataset_path: Path):
    """
    Ensure the dataset contains folders named 0 to 9.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    missing_folders = [str(i) for i in range(10) if not (dataset_path / str(i)).exists()]
    if missing_folders:
        raise FileNotFoundError(
            "Dataset is missing class folders: " + ", ".join(missing_folders)
        )


def load_image(image_path: Path, image_size: Tuple[int, int] = (64, 64)):
    """
    Read, resize, and normalize an image.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype("float32") / 255.0
    return image


def preprocess_frame(frame, image_size: Tuple[int, int] = (64, 64)):
    """
    Preprocess a webcam frame for model inference.
    """

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, image_size)
    normalized_frame = resized_frame.astype("float32") / 255.0
    batch = np.expand_dims(normalized_frame, axis=0)
    return batch


def save_training_plot(history, output_path="training_plot.png"):
    """
    Save training accuracy and loss graphs.
    """

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png"):
    """
    Plot and save the confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def load_trained_model(model_path: str):
    """
    Load a saved Keras model.
    """

    return load_model(model_path)
