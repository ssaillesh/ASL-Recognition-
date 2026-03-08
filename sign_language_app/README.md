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

## Notes
- If `asl_model.pkl` is not present, the app falls back to heuristic recognition for core commands and several words/letters.
- Reference icons and `asl_chart.png` are auto-generated at startup if missing.
