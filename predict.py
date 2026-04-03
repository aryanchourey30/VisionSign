"""
Production-ready real-time webcam inference for ASL recognition.
"""

from collections import Counter, deque
from pathlib import Path
from time import perf_counter
from typing import Deque, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model


DEFAULT_MODEL_PATH = "realmodel.h5"
ROI_TOP_LEFT = (320, 100)
ROI_BOTTOM_RIGHT = (520, 300)
PREDICTION_BUFFER_SIZE = 5

# This mapping assumes the trained model predicts 25 static ASL classes: A-Y.
ASL_LABELS = [chr(code) for code in range(ord("A"), ord("Y") + 1)]


def load_trained_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained Keras model once before the webcam loop starts.
    """

    resolved_path = Path(model_path)
    if not resolved_path.exists():
        candidate_paths = sorted(Path.cwd().glob("*.h5"))
        if len(candidate_paths) == 1:
            resolved_path = candidate_paths[0]

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {resolved_path}. "
            "Pass --model with the correct file path."
        )

    model = load_model(resolved_path, compile=False)
    return model


def preprocess_image(roi):
    """
    Convert ROI to grayscale, resize it to 28x28, normalize, and reshape it.
    """

    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (28, 28))
    normalized = resized.astype("float32") / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped


def predict_gesture(model, roi):
    """
    Run inference on the ROI and return the predicted label and confidence.
    """

    input_tensor = preprocess_image(roi)
    probabilities = model.predict(input_tensor, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    if predicted_index >= len(ASL_LABELS):
        return f"Class {predicted_index}", confidence

    return ASL_LABELS[predicted_index], confidence


def update_prediction_buffer(
    prediction_buffer: Deque[Tuple[str, float]], prediction: str, confidence: float
):
    """
    Update the rolling prediction buffer and return a stable prediction.
    """

    prediction_buffer.append((prediction, confidence))

    label_counts = Counter(label for label, _ in prediction_buffer)
    stable_label, _ = max(
        label_counts.items(),
        key=lambda item: (
            item[1],
            np.mean([conf for label, conf in prediction_buffer if label == item[0]]),
        ),
    )

    matching_confidences = [
        conf for label, conf in prediction_buffer if label == stable_label
    ]
    stable_confidence = float(np.mean(matching_confidences))

    return stable_label, stable_confidence


def extract_roi(frame, top_left, bottom_right):
    """
    Extract the fixed hand region from the frame.
    """

    x1, y1 = top_left
    x2, y2 = bottom_right
    return frame[y1:y2, x1:x2]


def draw_prediction_panel(frame, prediction: str, confidence: float, fps: float):
    """
    Draw the prediction UI panel, confidence, FPS, and ROI box.
    """

    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)

    panel_top_left = (15, 15)
    panel_bottom_right = (315, 130)
    cv2.rectangle(frame, panel_top_left, panel_bottom_right, (30, 30, 30), -1)
    cv2.rectangle(frame, panel_top_left, panel_bottom_right, (0, 255, 255), 2)

    cv2.putText(
        frame,
        f"Predicted: {prediction}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (30, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (180, 255, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Place hand inside ROI",
        (ROI_TOP_LEFT[0] - 5, ROI_TOP_LEFT[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def run_realtime_prediction(model_path: str = DEFAULT_MODEL_PATH, camera_index: int = 0):
    """
    Run the real-time ASL recognition loop.
    """

    model = load_trained_model(model_path)
    output_classes = int(model.output_shape[-1])

    if output_classes != len(ASL_LABELS):
        print(
            f"Warning: model outputs {output_classes} classes, "
            f"but the ASL label map expects {len(ASL_LABELS)} classes."
        )

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not access the webcam. Please check your camera.")

    prediction_buffer: Deque[Tuple[str, float]] = deque(maxlen=PREDICTION_BUFFER_SIZE)
    previous_time = perf_counter()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            roi = extract_roi(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT)

            if roi.size == 0:
                print("ROI is empty. Check ROI coordinates and webcam resolution.")
                break

            prediction, confidence = predict_gesture(model, roi)
            stable_prediction, stable_confidence = update_prediction_buffer(
                prediction_buffer, prediction, confidence
            )

            current_time = perf_counter()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            draw_prediction_panel(frame, stable_prediction, stable_confidence, fps)
            cv2.imshow("Real-Time ASL Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as error:
        print(f"Application error: {error}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
