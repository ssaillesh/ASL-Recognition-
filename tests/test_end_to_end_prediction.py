from __future__ import annotations

import csv
import os
import unittest

import numpy as np

from sign_language_app.classifier import ASLClassifier
from sign_language_app.preprocessing import infer_landmark_channels, normalize_landmark_tensor


MODEL_PATH = os.path.join("sign_language_app", "models", "asl_model.pkl")
DATASET_PATH = os.path.join("data", "landmarks_from_public_kaggle.csv")


class TestEndToEndPrediction(unittest.TestCase):
    def test_classifier_pipeline_output_contract(self) -> None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(DATASET_PATH):
            self.skipTest("model or dataset missing")

        with open(DATASET_PATH, "r", newline="") as f:
            row = next(csv.reader(f))

        feature = np.asarray([float(v) for v in row[1:]], dtype=np.float32)
        channels = infer_landmark_channels(feature.shape[0])
        tensor = feature.reshape(21, channels)
        norm = normalize_landmark_tensor(tensor).flatten()
        points = [tuple(p.tolist()) for p in normalize_landmark_tensor(tensor)]

        clf = ASLClassifier(MODEL_PATH)
        pred = clf.predict(norm, points)

        self.assertIsInstance(pred.label, str)
        self.assertGreaterEqual(float(pred.confidence), 0.0)
        self.assertLessEqual(float(pred.confidence), 1.0)
        self.assertTrue(len(pred.top3) >= 1)


if __name__ == "__main__":
    unittest.main()
