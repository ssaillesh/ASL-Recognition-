"""CNN-based recognition module for ASL hand gestures.

Provides 1D convolutional neural network training and inference for real-time
hand gesture recognition using MediaPipe landmarks.
"""

from sign_language_app.cnn.classifier import CNNClassifier
from sign_language_app.cnn.trainer import train_cnn_model

__all__ = ["CNNClassifier", "train_cnn_model"]
