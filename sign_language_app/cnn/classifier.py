"""CNN-based classifier for real-time ASL gesture recognition."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np


class CNNClassifier:
    """Loads and infers with a trained 1D CNN model for hand gesture recognition."""

    def __init__(self, model_path: str) -> None:
        """
        Initialize the CNN classifier.
        
        Args:
            model_path: Path to the .keras model file or .pkl wrapper.
        """
        self.model = None
        self.classes: List[str] = []
        self.input_shape: Tuple[int, int] | None = None
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load the trained TensorFlow/Keras CNN model."""
        if not model_path or not os.path.exists(model_path):
            return

        # Handle both .keras and .pkl wrapper paths
        if model_path.endswith(".pkl"):
            import pickle
            try:
                with open(model_path, "rb") as f:
                    payload = pickle.load(f)
                if not isinstance(payload, dict):
                    return
                model_file = str(payload.get("model_path", "")).strip()
                if not model_file or not os.path.isabs(model_file):
                    if model_file and not os.path.isabs(model_file):
                        model_file = os.path.join(os.path.dirname(model_path), model_file)
                if not os.path.exists(model_file):
                    return
                self.classes = [str(v) for v in payload.get("classes", [])]
                shape = payload.get("input_shape", [21, 2])
                if isinstance(shape, list) and len(shape) == 2:
                    self.input_shape = (int(shape[0]), int(shape[1]))
                model_path = model_file
            except Exception:
                return

        # Load the actual .keras model
        if not model_path.endswith(".keras"):
            return

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
        except ImportError:
            pass
        except Exception:
            self.model = None

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict the gesture from a normalized hand landmark feature vector.
        
        Args:
            feature_vector: 42-dim (21x2) or 63-dim (21x3) normalized hand landmark vector.
            
        Returns:
            Tuple of (predicted_label, confidence, top3_predictions)
        """
        if self.model is None:
            return ("UNKNOWN", 0.0, [("UNKNOWN", 0.0)])

        channels = 2
        if self.input_shape is not None:
            channels = int(self.input_shape[1])

        if channels <= 0 or len(feature_vector) % channels != 0:
            return ("UNKNOWN", 0.0, [("UNKNOWN", 0.0)])

        # Reshape for CNN: (1, 21, channels)
        sample = feature_vector.reshape(1, len(feature_vector) // channels, channels)
        probabilities = self.model.predict(sample, verbose=0)[0]
        indices = np.argsort(probabilities)[::-1][:3]

        classes = self.classes if self.classes else [str(i) for i in range(len(probabilities))]
        top3: List[Tuple[str, float]] = [
            (str(classes[idx]), float(probabilities[idx])) for idx in indices
        ]

        label, confidence = top3[0]
        return (label, confidence, top3)

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
