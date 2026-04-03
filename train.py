"""
Training pipeline for ASL digit recognition.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from data_loader import load_dataset
from model import build_cnn_model
from utils import plot_confusion_matrix, save_training_plot


def train_model(
    dataset_dir: str,
    model_output_path: str = "asl_digit_model.h5",
    epochs: int = 10,
    batch_size: int = 32,
):
    """
    Train the CNN model and save results.
    """

    x_train, x_test, y_train, y_test, y_train_labels, y_test_labels = load_dataset(
        dataset_dir
    )

    model = build_cnn_model(input_shape=x_train.shape[1:], num_classes=10)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    output_dir = Path(model_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test_labels, predicted_labels)

    model.save(model_output_path)

    save_training_plot(history, output_path=str(output_dir / "training_plot.png"))
    plot_confusion_matrix(
        y_test_labels,
        predicted_labels,
        output_path=str(output_dir / "confusion_matrix.png"),
    )

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "accuracy_score": float(accuracy),
        "model_path": str(Path(model_output_path).resolve()),
        "training_plot_path": str((output_dir / "training_plot.png").resolve()),
        "confusion_matrix_path": str((output_dir / "confusion_matrix.png").resolve()),
        "num_train_samples": int(len(y_train_labels)),
        "num_test_samples": int(len(y_test_labels)),
    }

    return model, history, results


def parse_arguments():
    """
    Parse command-line arguments for training.
    """

    parser = argparse.ArgumentParser(
        description="Train a CNN model on a folder-based ASL digit dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset directory containing folders 0 to 9",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="asl_digit_model.h5",
        help="Path where the trained model will be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    return parser.parse_args()


def main():
    """
    Command-line entry point for training.
    """

    args = parse_arguments()
    _, _, results = train_model(
        dataset_dir=args.dataset,
        model_output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("Training complete.")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
