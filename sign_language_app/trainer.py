from __future__ import annotations

import csv
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


@dataclass
class TrainingConfig:
    dataset_csv: str
    model_output: str


class LandmarkCollector:
    def __init__(self, camera_index: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    def _normalize(self, points: List[Tuple[float, float]]) -> np.ndarray:
        wrist_x, wrist_y = points[0]
        shifted = np.array([[x - wrist_x, y - wrist_y] for x, y in points], dtype=np.float32)
        max_norm = np.max(np.linalg.norm(shifted, axis=1))
        if max_norm > 0:
            shifted /= max_norm
        return shifted.flatten()

    def capture_samples(self, label: str, count: int, output_csv: str) -> None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        captured = 0

        with open(output_csv, "a", newline="") as handle:
            writer = csv.writer(handle)
            while captured < count:
                ok, frame = self.cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)

                multi_hand_landmarks = getattr(result, "multi_hand_landmarks", None)
                if multi_hand_landmarks:
                    lm = multi_hand_landmarks[0]
                    points = [(p.x, p.y) for p in lm.landmark]
                    features = self._normalize(points)
                    writer.writerow([label] + features.tolist())
                    captured += 1

                cv2.putText(
                    frame,
                    f"Collecting {label}: {captured}/{count}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


def train_model(config: TrainingConfig) -> None:
    labels = []
    vectors = []

    with open(config.dataset_csv, "r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            labels.append(row[0])
            vectors.append([float(v) for v in row[1:]])

    X = np.array(vectors, dtype=np.float32)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(config.model_output), exist_ok=True)
    with open(config.model_output, "wb") as handle:
        pickle.dump({"model": model, "label_encoder": None}, handle)

    print(f"Saved model to {config.model_output}")


if __name__ == "__main__":
    default_csv = os.path.join("data", "landmarks.csv")
    default_model = os.path.join("models", "asl_model.pkl")
    train_model(TrainingConfig(dataset_csv=default_csv, model_output=default_model))
