"""
Command-line entry point for real-time ASL digit recognition.
"""

import argparse

from predict import DEFAULT_MODEL_PATH, run_realtime_prediction


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time ASL hand sign recognition using a webcam"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained Keras model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index for OpenCV (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    run_realtime_prediction(model_path=args.model, camera_index=args.camera)


if __name__ == "__main__":
    main()
