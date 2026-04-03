"""
Dataset loading utilities for ASL digit recognition.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from utils import load_image, validate_dataset_structure


def load_dataset(
    dataset_dir: str,
    image_size: Tuple[int, int] = (64, 64),
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load images from a folder-based dataset and split them into train/test sets.

    Expected dataset structure:
        dataset/
            0/
            1/
            ...
            9/
    """

    dataset_path = Path(dataset_dir)
    validate_dataset_structure(dataset_path)

    images = []
    labels = []

    for class_index in range(10):
        class_dir = dataset_path / str(class_index)
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file():
                image = load_image(image_path, image_size=image_size)
                if image is not None:
                    images.append(image)
                    labels.append(class_index)

    if not images:
        raise ValueError(
            "No valid images were found. Please check your dataset path and image files."
        )

    x = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    y_train_encoded = to_categorical(y_train, num_classes=10)
    y_test_encoded = to_categorical(y_test, num_classes=10)

    return x_train, x_test, y_train_encoded, y_test_encoded, y_train, y_test
