from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def infer_landmark_channels(feature_count: int) -> int:
    if feature_count == 42:
        return 2
    if feature_count == 63:
        return 3
    raise ValueError(
        f"Unsupported feature width {feature_count}. Expected 42 (21x2) or 63 (21x3)."
    )


def normalize_landmark_tensor(sample: np.ndarray) -> np.ndarray:
    """Normalize a 21xC landmark tensor using wrist-relative max-norm scaling."""
    if sample.shape[0] != 21:
        return sample.astype(np.float32)

    wrist = sample[0:1, :]
    translated = sample - wrist
    norms = np.linalg.norm(translated, axis=1)
    max_norm = float(np.max(norms)) if norms.size else 0.0
    if max_norm > 0.0:
        translated = translated / max_norm
    return translated.astype(np.float32)


def normalize_landmarks_xy(points: Sequence[Point]) -> np.ndarray:
    """Normalize 2D landmarks and return flattened 42-dim feature vector."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (21, 2):
        return np.zeros((42,), dtype=np.float32)
    normalized = normalize_landmark_tensor(arr)
    return normalized.flatten().astype(np.float32)


def feature_stats(feature_vector: np.ndarray) -> Dict[str, float]:
    vec = np.asarray(feature_vector, dtype=np.float32)
    if vec.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
        "mean": float(np.mean(vec)),
        "std": float(np.std(vec)),
    }
