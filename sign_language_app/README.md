# Sign Language App

This folder contains a real-time ASL recognition desktop app built with MediaPipe, OpenCV, Tkinter, and a pluggable classifier.

## Features
- Two-panel UI with live camera feed and reference guide.
- Real-time hand landmark extraction (21 keypoints).
- Background-thread classification on every second frame.
- Temporal smoothing across recent predictions to reduce flicker.
- Sentence builder with confidence gating and hold-time debouncing.
- Text-to-speech for confirmed sentence.
- Practice mode with score and streak tracking.
- Pluggable model inference: RandomForest or CNN (1D Conv).

## Run The Application

### 1. Install dependencies
```bash
pip install -r sign_language_app/requirements.txt
```

### 2. Start from repository root (recommended)
```bash
cd /Users/saillesh/Desktop/ALS/ASL-Recognition-
python -m sign_language_app.main
```

Quick command (run the program):
```bash
cd /Users/saillesh/Desktop/ALS/ASL-Recognition- && \
PYTHONPATH=/Users/saillesh/Desktop/ALS/ASL-Recognition- \
/Users/saillesh/Desktop/ALS/.venv/bin/python -m sign_language_app.main
```

### 3. Start from outside repository root
If your terminal is at `/Users/saillesh/Desktop/ALS` (or any other folder), set `PYTHONPATH`:

```bash
PYTHONPATH=/Users/saillesh/Desktop/ALS/ASL-Recognition- \
python -m sign_language_app.main
```

### 4. Use the project virtual environment explicitly (optional)
```bash
PYTHONPATH=/Users/saillesh/Desktop/ALS/ASL-Recognition- \
/Users/saillesh/Desktop/ALS/.venv/bin/python -m sign_language_app.main
```

## Web App

The project also includes a lightweight browser-based version with the same landmark pipeline and CNN inference.

### Install web dependencies
```bash
pip install -r sign_language_app/requirements.txt
```

### Start the web app
```bash
cd /Users/saillesh/Desktop/ALS/ASL-Recognition-
PYTHONPATH=/Users/saillesh/Desktop/ALS/ASL-Recognition- \
/Users/saillesh/Desktop/ALS/.venv/bin/python -m sign_language_app.web.server
```

### Open in browser
Visit:
```text
http://127.0.0.1:8000
```

The browser requests camera access locally, sends captured frames to the backend, and shows live predictions in a minimal Apple-like interface.

### Why this matters
`sign_language_app` is a Python package under the repository root. If you run from a different current working directory without `PYTHONPATH`, Python cannot resolve the package and raises `ModuleNotFoundError: No module named 'sign_language_app'`.

## System Design

### Architecture diagram
```text
				+---------------------------+
				|      Tkinter App UI       |
				| main.py + ui/camera_panel |
				+------------+--------------+
							 |
							 v
				+---------------------------+
				|      Gesture Engine       |
				| OpenCV + MediaPipe (21pt) |
				+------------+--------------+
							 |
							 v
				+---------------------------+
				| Landmark Normalization    |
				| wrist-relative + max-norm |
				+------------+--------------+
							 |
							 v
				+---------------------------+
				|  Inference Worker Thread  |
				| bounded in/out queues     |
				+------------+--------------+
							 |
				+------------+-------------+
				|                          |
				v                          v
   +---------------------------+   +---------------------------+
   | RandomForest Classifier   |   | CNN Classifier            |
   | classifier.py             |   | cnn/classifier.py         |
   +------------+--------------+   +------------+--------------+
				\                          /
				 \                        /
				  +----------+-----------+
							 |
							 v
				+---------------------------+
				| U/V/R geometric calibration|
				+------------+--------------+
							 |
							 v
				+---------------------------+
				| Temporal smoothing (7-win)|
				+------------+--------------+
							 |
							 v
				+---------------------------+
				| SentenceBuilder + TTS     |
				| threshold + hold debounce |
				+------------+--------------+
							 |
							 v
				+---------------------------+
				| UI update + Practice mode |
				+---------------------------+

Training path:
data/*.csv -> trainer.py / cnn/trainer.py -> asl_model.pkl (+ asl_model.keras for CNN)
```

### High-level architecture
- UI Layer (`main.py`, `ui/*`): Tkinter desktop application, camera panel, reference panel, practice panel.
- Vision Layer (`gesture_engine.py`): webcam capture, MediaPipe hand tracking, 21-landmark extraction, wrist-relative normalization.
- Inference Layer (`classifier.py`, `cnn/classifier.py`): model loading and prediction (RF or CNN), with U/V/R geometric calibration.
- Interaction Layer (`sentence_builder.py`): confidence thresholding, hold-time debouncing, sentence state, clear/confirm behavior, TTS.
- Training Layer (`trainer.py`, `cnn/trainer.py`): dataset loading, split strategy, model training, artifact export.

### Runtime data flow
1. Camera frame is captured with OpenCV in `GestureEngine.read()`.
2. MediaPipe returns hand landmarks (21 points).
3. Landmarks are normalized (wrist translation + max-norm scaling) into feature vectors.
4. `CameraPanel` sends features to a background classification thread.
5. `ASLClassifier` routes prediction to either:
	 - RandomForest model path, or
	 - CNN module via `CNNClassifier`.
6. Top predictions are temporally smoothed across a short rolling window.
7. `SentenceBuilder` applies confidence threshold and hold-time debounce before committing labels.
8. UI updates detected label, confidence bars, reference highlight, and sentence text.

### Concurrency model
- Main Tkinter thread: rendering, UI updates, controls.
- Worker thread: model inference (`_classification_worker`) to avoid blocking UI.
- Bounded queues: decouple frame production and prediction consumption.

### Model artifacts
- `sign_language_app/models/asl_model.pkl`
	- RandomForest payload, or
	- CNN wrapper metadata (`model_type=cnn1d`, classes, input shape, model path).
- `sign_language_app/models/asl_model.keras`
	- Saved Keras CNN model used by `CNNClassifier`.

### Normalization contract (critical)
Training and inference both use the same landmark normalization:
- translate all points relative to wrist landmark,
- divide by max landmark norm.

Keeping this contract consistent is required for stable real-time predictions.

## Train a model
1. Collect samples using `LandmarkCollector` in `sign_language_app/trainer.py`.
2. Save rows in `data/landmarks.csv` as: `label,f0,f1,...,f41`.
3. Train:
```bash
python -m sign_language_app.trainer
```
4. Model is saved to `sign_language_app/models/asl_model.pkl`.

### Train Option A: Landmark-based 1D-CNN
This mode treats each sample as a landmark tensor (`21x2` or `21x3`) and applies a small Conv1D network across landmark index.

Key setup:
- Train/test split: `80/20` (`test_size=0.2`, `stratify=True`, `random_state=42`)
- Validation split (inside training set): `0.1`
- Epochs: configurable via `--epochs` (default `80`) with early stopping (`patience=8`)
- Split strategy: `random` (default) or `group-similarity` to reduce near-duplicate leakage

Command:
```bash
python -m sign_language_app.trainer \
	--dataset-csv data/landmarks_from_public_kaggle.csv \
	--model-output sign_language_app/models/asl_model.pkl \
	--model-type cnn1d \
	--split-strategy group-similarity \
	--epochs 80 \
	--batch-size 64 \
	--learning-rate 0.001
```

Notes:
- The `.pkl` output is a wrapper with metadata plus a `.keras` model path.
- The app classifier auto-detects this wrapper and runs CNN inference if TensorFlow is installed.
- CSV feature width must be `42` (`21x2`) or `63` (`21x3`).

## Use Kaggle ASL Fingerspelling data
This trainer now supports converting the Kaggle competition files used in:
`https://www.kaggle.com/code/gusthema/asl-fingerspelling-recognition-w-tensorflow`

Expected Kaggle folder layout:
- `<kaggle_root>/train.csv`
- `<kaggle_root>/train_landmarks/*.parquet`

Convert and train in one step:
```bash
python -m sign_language_app.trainer \
	--kaggle-root /path/to/asl-fingerspelling \
	--kaggle-output-csv data/kaggle_landmarks.csv \
	--model-output sign_language_app/models/asl_model.pkl
```

Convert only (no training):
```bash
python -m sign_language_app.trainer \
	--kaggle-root /path/to/asl-fingerspelling \
	--build-only
```

Note: This converter extracts a static hand-pose feature per sequence (median hand landmarks), so it is useful as a bootstrap dataset for the current static classifier.

## Notes
- If `asl_model.pkl` is not present, the app falls back to a minimal heuristic recognizer for a few core letters.
- Reference icons and `asl_chart.png` are auto-generated at startup if missing.
