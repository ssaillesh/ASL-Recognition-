from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from typing import List, Tuple

import numpy as np

from sign_language_app.cnn.classifier import CNNClassifier
from sign_language_app.preprocessing import infer_landmark_channels, normalize_landmark_tensor


def load_csv_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    labels: List[str] = []
    vectors: List[List[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(str(row[0]))
            vectors.append([float(v) for v in row[1:]])
    return np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=str)


def preprocess(X: np.ndarray) -> np.ndarray:
    channels = infer_landmark_channels(X.shape[1])
    tensor = X.reshape((-1, 21, channels)).astype(np.float32)
    tensor = np.array([normalize_landmark_tensor(s) for s in tensor], dtype=np.float32)
    return tensor.reshape((tensor.shape[0], -1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline CNN validator for ASL landmark model")
    parser.add_argument("--model", default=os.path.join("sign_language_app", "models", "asl_model.pkl"))
    parser.add_argument("--dataset", default=os.path.join("data", "landmarks_from_public_kaggle.csv"))
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--show-misclassified", type=int, default=25)
    args = parser.parse_args()

    clf = CNNClassifier(args.model)
    if not clf.is_loaded():
        raise RuntimeError(f"Unable to load CNN model from {args.model}")

    X, y = load_csv_dataset(args.dataset)
    X = preprocess(X)

    limit = min(len(y), args.max_samples)
    X = X[:limit]
    y = y[:limit]

    correct = 0
    wrong: List[Tuple[int, str, str, float]] = []
    confusion = Counter()

    for i, (vec, true_label) in enumerate(zip(X, y)):
        pred_label, conf, _top3 = clf.predict(vec)
        if pred_label == true_label:
            correct += 1
        else:
            wrong.append((i, true_label, pred_label, float(conf)))
            confusion[(true_label, pred_label)] += 1

    accuracy = correct / float(limit) if limit else 0.0
    print(f"[offline] evaluated={limit}")
    print(f"[offline] accuracy={accuracy:.4f}")
    print(f"[offline] misclassified={len(wrong)}")

    if wrong:
        print("[offline] sample misclassifications:")
        for i, true_label, pred_label, conf in wrong[: args.show_misclassified]:
            print(f"  idx={i} true={true_label} pred={pred_label} conf={conf:.3f}")

        print("[offline] top confusion pairs:")
        for (true_label, pred_label), count in confusion.most_common(15):
            print(f"  {true_label} -> {pred_label}: {count}")


if __name__ == "__main__":
    main()
