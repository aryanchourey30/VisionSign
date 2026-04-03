# VisionSign

VisionSign is a Python-based real-time hand sign recognition project built with TensorFlow/Keras and OpenCV. It opens your webcam, extracts a fixed hand region, runs a trained model on each frame, and shows the predicted class with a confidence score on screen.

The repository already includes a trained model file, `realmodel.h5`, so anyone who forks or clones the project can run the live demo immediately after installing dependencies.

## What This Repo Includes

- Real-time webcam inference with OpenCV
- A bundled trained model: `realmodel.h5`
- Prediction smoothing to reduce frame-to-frame flicker
- A simple CNN training pipeline for folder-based datasets
- Utility functions for plotting training curves and confusion matrices

## Current Model Details

The included `realmodel.h5` file was verified locally and expects:

- Input shape: `28 x 28 x 1`
- Output classes: `25`

The live prediction code currently maps those 25 outputs to static ASL alphabet classes `A` through `Y`.

## Project Structure

```text
VisionSign/
|-- main.py
|-- predict.py
|-- train.py
|-- data_loader.py
|-- model.py
|-- utils.py
|-- requirements.txt
|-- realmodel.h5
`-- README.md
```

## Quick Start

### 1. Clone Your Fork

```bash
git clone https://github.com/<your-username>/VisionSign.git
cd VisionSign
```

### 2. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv venv
venv\Scripts\activate.bat
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Real-Time Demo

Use the bundled model:

```bash
python main.py
```

Use a different model file:

```bash
python main.py --model path/to/model.h5
```

Use a different webcam index:

```bash
python main.py --camera 1
```

## How the Live Prediction Flow Works

1. The webcam opens and shows a live video feed.
2. A fixed ROI box appears on the frame.
3. You place your hand sign inside that box.
4. The ROI is converted to grayscale, resized to `28x28`, normalized, and reshaped.
5. The model predicts the class.
6. A short rolling buffer stabilizes the displayed prediction.
7. Press `q` to close the app.

## Train a New Model

The repo also includes a training pipeline for datasets organized like this:

```text
dataset/
|-- 0/
|-- 1/
|-- 2/
|-- 3/
|-- 4/
|-- 5/
|-- 6/
|-- 7/
|-- 8/
`-- 9/
```

Run training with:

```bash
python train.py --dataset path/to/dataset --output trained_model.h5 --epochs 10 --batch-size 32
```

Training outputs include:

- The saved model file
- `training_plot.png`
- `confusion_matrix.png`

## Important Note About Training vs Inference

There is currently a mismatch between the generic training pipeline and the bundled live inference model:

- `train.py` builds a `64 x 64 x 3` model for digit folders `0-9`
- `realmodel.h5` expects `28 x 28 x 1` input and outputs `25` classes

So:

- The included `realmodel.h5` works with `main.py`
- A model trained directly with the current `train.py` will not be a drop-in replacement for `realmodel.h5` unless you also update the inference preprocessing and label mapping

If you want, this repo can be extended later so training and inference use the exact same label set and image shape.

## Requirements

- Python 3
- Webcam
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

Install all Python packages with:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### Webcam does not open

- Make sure no other app is using the camera
- Try `python main.py --camera 1`
- Check camera permissions in your operating system

### Model file not found

- Make sure `realmodel.h5` is present in the project root
- Or pass an explicit model path with `--model`

### Poor predictions

- Keep your hand fully inside the ROI box
- Use a plain background when possible
- Improve lighting and avoid heavy motion blur

## Suggested Fork Workflow

After forking, a new user can usually get started with:

```bash
git clone https://github.com/<their-username>/VisionSign.git
cd VisionSign
python -m venv venv
pip install -r requirements.txt
python main.py
```

## License

No license file is included yet. If you want others to reuse or modify this project clearly, add a license such as MIT.
