# CNN Module for ASL Recognition

This module provides a complete 1D convolutional neural network (CNN) implementation for real-time hand gesture recognition using MediaPipe hand landmarks.

## Structure

```
sign_language_app/cnn/
├── __init__.py          # Module exports
├── trainer.py           # CNN model training pipeline
└── classifier.py        # CNN model inference and predictions
```

## Components

### `trainer.py` — CNN Training Pipeline

Handles all CNN model training with the following features:

- **Architecture**: 3× Conv1D layers + BatchNormalization + GlobalAveragePooling + Dense layers
- **Input**: Hand landmarks (21 joints × 2 coordinates = 42 features)
- **Output**: 26-class gesture predictions (A-Z)
- **Key Functions**:
  - `train_cnn_model()`: Main training function with configurable hyperparameters
  - Supports split strategies: `random` (stratified) or `group-similarity` (leakage-resistant)
  - Early stopping with patience=8 to prevent overfitting

**Usage**:
```bash
python -m sign_language_app.trainer \
  --dataset-csv data/landmarks_from_public_kaggle.csv \
  --model-output sign_language_app/models/asl_model.pkl \
  --model-type cnn1d \
  --epochs 25 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --split-strategy random
```

**Options**:
- `--epochs`: Number of training epochs (default: 80)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Adam optimizer learning rate (default: 0.001)
- `--split-strategy`: `random` or `group-similarity` (default: `random`)

### `classifier.py` — CNN Inference

Loads trained CNN models and performs real-time gesture predictions.

- **Key Class**: `CNNClassifier`
  - `__init__(model_path)`: Initialize with .keras or .pkl wrapper path
  - `predict(feature_vector)`: Return (label, confidence, top3_predictions)
  - `is_loaded()`: Check if model is ready

**Example**:
```python
from sign_language_app.cnn.classifier import CNNClassifier

classifier = CNNClassifier('sign_language_app/models/asl_model.pkl')
label, confidence, top3 = classifier.predict(landmark_features)
print(f"Predicted: {label} ({confidence:.2%})")
```

### `__init__.py` — Module Exports

Exposes the public API:
```python
from sign_language_app.cnn import CNNClassifier, train_cnn_model
```

## Integration with Main App

The CNN module is automatically integrated into the main application:

1. **Training**: When `--model-type cnn1d` is used, the trainer imports and calls `sign_language_app.cnn.trainer.train_cnn_model()`
2. **Inference**: The main `ASLClassifier` detects CNN models and delegates to `CNNClassifier` for predictions
3. **U/V/R Calibration**: Geometric disambiguation is still applied after CNN predictions

## Model Artifacts

When training completes, two files are created:

- **`asl_model.keras`**: The trained TensorFlow/Keras model (can be loaded directly)
- **`asl_model.pkl`**: Wrapper metadata including:
  - Model type (`"cnn1d"`)
  - Path to `.keras` file
  - Class labels (A-Z)
  - Input shape (21, 2)
  - Training metadata (epochs, learning rate, test accuracy, etc.)
  - Split strategy used

## Performance

### Typical Metrics (Random Stratified Split)
- Test Accuracy: **99.81%**
- Macro-averaged F1: **0.994**
- Early stopping: ~15-16 epochs (from 25 requested)

### Realistic Metrics (Group-Similarity Leakage-Safe Split)
- Test Accuracy: **~92-93%**
- Macro-averaged F1: **~0.92**
- More conservative but honest evaluation

## Data Normalization

The CNN module uses **exact runtime normalization**:
1. Translate landmarks by wrist position (landmark 0)
2. Divide by maximum Euclidean norm across landmarks
3. Produces normalized hand-centric features invariant to translation and scale

This matches `gesture_engine._normalized_landmarks()` to ensure consistency between training and inference.

## Dependencies

- TensorFlow/Keras (2.14+)
  - Install: `tensorflow` (Linux/Windows) or `tensorflow-macos` (Apple Silicon)
- NumPy
- scikit-learn (for split strategies)

## Advantages over RandomForest

1. **End-to-End Learning**: Learns hierarchical spatial patterns directly from landmarks
2. **Temporal Awareness**: Conv1D layers capture ordinal relationships between hand joints
3. **Better Scaling**: Performance improves more gracefully with larger datasets
4. **Fewer Hyperparameters**: No feature engineering needed
