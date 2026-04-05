# Interactive Hand Gesture Recognition
Real-time ASL-focused hand gesture recognition desktop application that detects hand landmarks from webcam input, predicts gestures, builds sentences, and supports text-to-speech output.

## Features
- Real-time webcam hand tracking using MediaPipe with 21 landmark points.
- Live ASL gesture prediction with top-3 confidence display.
- Debounced gesture confirmation (0.8s hold) to reduce flickering output.
- Sentence builder with special gestures for `SPACE`, `CLEAR`, and `CONFIRM`.
- Text-to-speech playback for confirmed sentence output.
- Two-panel Tkinter UI: camera feed + searchable ASL reference guide.
- Practice Mode with random prompts, score, and streak tracking.
- Optional training pipeline for a custom `RandomForest` classifier.
- Kaggle ASL Fingerspelling conversion support from parquet landmark files.

## Getting Started
### Prerequisites
- macOS, Linux, or Windows with webcam access.
- Python 3.10+.
- `pip` (or virtual environment tooling such as `venv`).
- Optional for Kaggle workflow: Kaggle account + API token (`kaggle.json`).

### Installation
1. Clone the repository.
```bash
git clone <your-repo-url>
cd Interactive-Hand-Gesture-Recognition
```
2. Create and activate a virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install project dependencies.
```bash
pip install -r sign_language_app/requirements.txt
```
4. (Recommended) Keep MediaPipe pinned to the tested version.
```bash
pip install 'mediapipe==0.10.14'
```

## Usage
Run the desktop application:
```bash
source /Users/saillesh/Desktop/ALS/.venv/bin/activate
cd /Users/saillesh/Desktop/ALS/ASL-Recognition-
python -m sign_language_app.main
```

The application opens a Tkinter window and keeps running until you close the window.

Train a model from an existing landmark CSV (`label,f0...f41`):
```bash
.venv/bin/python -m sign_language_app.trainer \
	--dataset-csv data/landmarks.csv \
	--model-output sign_language_app/models/asl_model.pkl
```

Convert Kaggle ASL Fingerspelling data and train in one command:
```bash
.venv/bin/python -m sign_language_app.trainer \
	--kaggle-root /path/to/asl-fingerspelling \
	--kaggle-output-csv data/kaggle_landmarks.csv \
	--model-output sign_language_app/models/asl_model.pkl
```

Expected Kaggle structure:
- `/path/to/asl-fingerspelling/train.csv`
- `/path/to/asl-fingerspelling/train_landmarks/*.parquet`

After training, restart the app; it will load `sign_language_app/models/asl_model.pkl` automatically.

## Contributing
Contributions are welcome.
- Open an issue describing the bug/feature before large changes.
- Create a feature branch and keep commits focused.
- Include reproduction steps and test notes in pull requests.
- If you add new gestures or labels, update both training and UI docs.

## License
No `LICENSE` file is currently present in this repository.
If you plan to open-source this project, add a license file (commonly `MIT`) and update this section to link it.

## Contact
Project maintainer: repository owner (`saillesh`).
For support or collaboration, please open an issue in this repository.
