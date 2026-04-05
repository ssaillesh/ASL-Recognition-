from __future__ import annotations

import csv
import os
import unittest

import numpy as np

from sign_language_app.cnn.classifier import CNNClassifier
from sign_language_app.preprocessing import infer_landmark_channels, normalize_landmark_tensor


MODEL_PATH = os.path.join("sign_language_app", "models", "asl_model.pkl")
DATASET_PATH = os.path.join("data", "landmarks_from_public_kaggle.csv")


def _load(path: str, limit: int = 200) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    vectors = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            labels.append(row[0])
            vectors.append([float(v) for v in row[1:]])
    return np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=str)


class TestModelInference(unittest.TestCase):
    def test_offline_accuracy_is_reasonable(self) -> None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(DATASET_PATH):
            self.skipTest("model or dataset missing")

        clf = CNNClassifier(MODEL_PATH)
        if not clf.is_loaded():
            self.skipTest("cnn model not loaded")

        X, y = _load(DATASET_PATH, limit=300)
        channels = infer_landmark_channels(X.shape[1])
        X = X.reshape((-1, 21, channels)).astype(np.float32)
        X = np.array([normalize_landmark_tensor(s).flatten() for s in X], dtype=np.float32)

        preds = []
        for sample in X:
            label, _conf, _top3 = clf.predict(sample)
            preds.append(label)

        acc = float(np.mean(np.asarray(preds) == y))
        self.assertGreaterEqual(acc, 0.8)


if __name__ == "__main__":
    unittest.main()
