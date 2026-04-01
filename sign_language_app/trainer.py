from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.cluster import MiniBatchKMeans

from sign_language_app.classifier import WORD_LABELS


POSE_SELECTED_IDS = [13, 15, 17, 19, 21, 14, 16, 18, 20, 22]


def kaggle_selected_columns() -> List[str]:
    cols: List[str] = []
    for axis in ("x", "y", "z"):
        for side in ("right", "left"):
            for idx in range(21):
                cols.append(f"{axis}_{side}_hand_{idx}")
        for idx in POSE_SELECTED_IDS:
            cols.append(f"{axis}_pose_{idx}")
    return cols


@dataclass
class TrainingConfig:
    dataset_csv: str
    model_output: str


def _load_csv_dataset(dataset_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    labels = []
    vectors = []

    with open(dataset_csv, "r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            labels.append(row[0])
            vectors.append([float(v) for v in row[1:]])

    X = np.array(vectors, dtype=np.float32)
    y = np.array(labels)
    return X, y


def _infer_landmark_channels(feature_count: int) -> int:
    if feature_count == 42:
        return 2
    if feature_count == 63:
        return 3
    raise ValueError(
        f"Unsupported feature width {feature_count}. Expected 42 (21x2) or 63 (21x3)."
    )


def _normalize_landmark_tensor(sample: np.ndarray) -> np.ndarray:
    """Match runtime normalization: wrist-relative, scaled by max landmark norm."""
    if sample.shape[0] != 21:
        return sample

    wrist = sample[0:1, :]
    translated = sample - wrist
    norms = np.linalg.norm(translated, axis=1)
    max_norm = float(np.max(norms)) if norms.size else 0.0
    if max_norm > 0.0:
        translated = translated / max_norm
    return translated.astype(np.float32)


def _build_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    split_strategy: str,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split_strategy == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test

    if split_strategy != "group-similarity":
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    # Build pseudo-groups by clustering each class; this prevents very similar samples
    # from being split between train/test and reduces optimistic leakage.
    groups = np.zeros(len(y), dtype=np.int32)
    offset = 0
    labels = np.unique(y)
    for label in labels:
        idx = np.where(y == label)[0]
        k = max(5, min(40, len(idx) // 20))
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=256, n_init="auto")
        g = km.fit_predict(X[idx])
        groups[idx] = g + offset
        offset += k

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


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


def _normalize_points(points: np.ndarray) -> Optional[np.ndarray]:
    if points.shape != (21, 2):
        return None

    if not np.isfinite(points).all():
        return None

    wrist_x, wrist_y = points[0]
    translated = points - np.array([wrist_x, wrist_y], dtype=np.float32)
    max_norm = np.max(np.linalg.norm(translated, axis=1))
    if max_norm <= 0:
        return None

    translated = translated / max_norm
    return translated.flatten().astype(np.float32)


def _sort_landmark_columns(columns: Iterable[str]) -> List[str]:
    def _index(name: str) -> int:
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else 10_000

    return sorted(columns, key=_index)


def _hand_axis_columns(columns: Sequence[str], axis: str, side: str) -> List[str]:
    strict_prefix = f"{axis}_{side}_hand_"
    strict = [col for col in columns if col.startswith(strict_prefix)]
    if strict:
        return _sort_landmark_columns(strict)

    # Fallback for alternate naming conventions.
    loose = [col for col in columns if col.startswith(f"{axis}_") and f"_{side}_hand_" in col]
    return _sort_landmark_columns(loose)


def _extract_sequence_feature(sequence_df: pd.DataFrame) -> Optional[np.ndarray]:
    cols = list(sequence_df.columns)

    best_points: Optional[np.ndarray] = None
    best_score = -1

    for side in ("right", "left"):
        x_cols = _hand_axis_columns(cols, "x", side)
        y_cols = _hand_axis_columns(cols, "y", side)

        if len(x_cols) < 21 or len(y_cols) < 21:
            continue

        x_cols = x_cols[:21]
        y_cols = y_cols[:21]

        x_vals = sequence_df[x_cols].to_numpy(dtype=np.float32)
        y_vals = sequence_df[y_cols].to_numpy(dtype=np.float32)

        valid_score = int(np.isfinite(x_vals).sum() + np.isfinite(y_vals).sum())
        if valid_score <= best_score:
            continue

        x_med = np.nanmedian(x_vals, axis=0)
        y_med = np.nanmedian(y_vals, axis=0)
        points = np.stack([x_med, y_med], axis=1)

        best_points = points
        best_score = valid_score

    if best_points is None:
        return None

    return _normalize_points(best_points)


def _map_phrase_to_label(phrase: str) -> Optional[str]:
    cleaned = phrase.strip().upper()

    if len(cleaned) == 1 and "A" <= cleaned <= "Z":
        return cleaned

    if cleaned in WORD_LABELS:
        return cleaned

    return None


def build_dataset_from_kaggle(
    kaggle_root: str,
    output_csv: str,
    max_samples_per_label: int = 600,
) -> None:
    train_csv = os.path.join(kaggle_root, "train.csv")
    landmarks_dir = os.path.join(kaggle_root, "train_landmarks")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv at {train_csv}")
    if not os.path.isdir(landmarks_dir):
        raise FileNotFoundError(f"Missing train_landmarks directory at {landmarks_dir}")

    meta = pd.read_csv(train_csv)
    if "sequence_id" not in meta.columns or "phrase" not in meta.columns:
        raise ValueError("train.csv must contain sequence_id and phrase columns")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    label_counts: Counter[str] = Counter()
    written = 0
    expected_cols = set(kaggle_selected_columns())

    with open(output_csv, "w", newline="") as handle:
        writer = csv.writer(handle)

        for row in meta.itertuples(index=False):
            label = _map_phrase_to_label(str(row.phrase))
            if not label:
                continue
            if label_counts[label] >= max_samples_per_label:
                continue

            sequence_id = str(row.sequence_id)
            parquet_path = os.path.join(landmarks_dir, f"{sequence_id}.parquet")
            if not os.path.exists(parquet_path):
                continue

            # Use the exact Kaggle-style selected columns when available.
            parquet_cols = list(pq.read_schema(parquet_path).names)
            use_cols = [c for c in parquet_cols if c in expected_cols]
            seq_df = pd.read_parquet(parquet_path, columns=use_cols if use_cols else None)
            feature = _extract_sequence_feature(seq_df)
            if feature is None:
                continue

            writer.writerow([label] + feature.tolist())
            label_counts[label] += 1
            written += 1

            if written % 500 == 0:
                print(f"Converted {written} samples...")

    print(f"Saved converted dataset to {output_csv}")
    print(f"Total samples: {written}")
    print(f"Per-label counts: {dict(sorted(label_counts.items()))}")


def train_model(config: TrainingConfig, split_strategy: str = "random") -> None:
    X, y = _load_csv_dataset(config.dataset_csv)

    X_train, X_test, y_train, y_test = _build_train_test_split(
        X,
        y,
        split_strategy=split_strategy,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(config.model_output), exist_ok=True)
    with open(config.model_output, "wb") as handle:
        pickle.dump({"model": model, "label_encoder": None}, handle)

    print(f"Saved model to {config.model_output}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL model from landmark CSV or Kaggle Fingerspelling data")
    parser.add_argument("--dataset-csv", default=os.path.join("data", "landmarks.csv"))
    parser.add_argument("--model-output", default=os.path.join("models", "asl_model.pkl"))
    parser.add_argument("--model-type", choices=["rf", "cnn1d"], default="rf")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--split-strategy", choices=["random", "group-similarity"], default="random")
    parser.add_argument("--kaggle-root", default="", help="Path containing train.csv and train_landmarks/")
    parser.add_argument("--kaggle-output-csv", default=os.path.join("data", "kaggle_landmarks.csv"))
    parser.add_argument("--max-samples-per-label", type=int, default=600)
    parser.add_argument("--build-only", action="store_true", help="Only convert Kaggle dataset without training")
    args = parser.parse_args()

    dataset_csv = args.dataset_csv

    if args.kaggle_root:
        build_dataset_from_kaggle(
            kaggle_root=args.kaggle_root,
            output_csv=args.kaggle_output_csv,
            max_samples_per_label=args.max_samples_per_label,
        )
        dataset_csv = args.kaggle_output_csv

    if not args.build_only:
        config = TrainingConfig(dataset_csv=dataset_csv, model_output=args.model_output)
        if args.model_type == "cnn1d":
            from sign_language_app.cnn.trainer import train_cnn_model
            train_cnn_model(
                config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                split_strategy=args.split_strategy,
            )
        else:
            train_model(config, split_strategy=args.split_strategy)
