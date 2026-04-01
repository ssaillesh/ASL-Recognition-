# Sign Language App

This folder contains a real-time ASL recognition desktop app built with MediaPipe, OpenCV, Tkinter, and a pluggable classifier.

## Features
- Two-panel UI with live camera feed and reference guide.
- Real-time hand landmark extraction (21 keypoints).
- Background-thread classification on every second frame.
- Confidence threshold at 80% with 0.8s debouncing.
- Sentence builder with SPACE, CLEAR, and CONFIRM gestures.
- Text-to-speech for confirmed sentence.
- Practice mode with score and streak tracking.
- Data collector and trainer for a RandomForest model.

## Run
```bash
pip install -r sign_language_app/requirements.txt
python -m sign_language_app.main
```

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
- If `asl_model.pkl` is not present, the app falls back to heuristic recognition for core commands and several words/letters.
- Reference icons and `asl_chart.png` are auto-generated at startup if missing.
