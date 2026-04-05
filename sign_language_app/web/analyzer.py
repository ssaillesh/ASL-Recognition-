from __future__ import annotations

from collections import deque
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import cv2
import numpy as np

from sign_language_app.classifier import ASLClassifier, PredictionResult
from sign_language_app.preprocessing import normalize_landmarks_xy

Point = Tuple[float, float]


class WebGestureAnalyzer:
    def __init__(self, model_path: str) -> None:
        self.classifier = ASLClassifier(model_path)
        self.mp_hands = cast(Any, import_module("mediapipe.python.solutions.hands"))
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmark_history = deque(maxlen=12)

    def close(self) -> None:
        self.hands.close()

    @staticmethod
    def _distance(points: Sequence[Point], a: int, b: int) -> float:
        ax, ay = points[a]
        bx, by = points[b]
        return float(np.hypot(ax - bx, ay - by))

    @staticmethod
    def _finger_extended(points: Sequence[Point], tip: int, pip: int, mcp: int) -> bool:
        return points[tip][1] < points[pip][1] < points[mcp][1]

    def _motion_jz_override(self) -> Optional[PredictionResult]:
        if len(self._landmark_history) < 8:
            return None

        trace = [np.asarray(sample[8], dtype=np.float32) for sample in self._landmark_history if sample is not None]
        if len(trace) < 8:
            return None

        trajectory = np.asarray(trace, dtype=np.float32)
        deltas = np.diff(trajectory, axis=0)
        if deltas.size == 0:
            return None

        path_len = float(np.sum(np.linalg.norm(deltas, axis=1)))
        net = trajectory[-1] - trajectory[0]
        dx = deltas[:, 0]

        sign_changes = int(np.sum(np.sign(dx[:-1]) != np.sign(dx[1:]))) if len(dx) > 2 else 0

        if path_len > 0.14 and sign_changes >= 2 and abs(float(net[0])) > 0.03:
            return PredictionResult(label="Z", confidence=0.86, top3=[("Z", 0.86), ("UNKNOWN", 0.14)])

        if path_len > 0.12 and float(net[1]) > 0.04 and abs(float(net[0])) > 0.01:
            return PredictionResult(label="J", confidence=0.84, top3=[("J", 0.84), ("UNKNOWN", 0.16)])

        return None

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode image payload")
        return cv2.flip(frame, 1)

    def predict_image(self, image_bytes: bytes) -> Dict[str, object]:
        frame = self._decode_image(image_bytes)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        landmarks: Optional[List[Point]] = None
        feature_vector = None

        multi_hand_landmarks = getattr(result, "multi_hand_landmarks", None)
        if multi_hand_landmarks:
            hand_landmarks = multi_hand_landmarks[0]
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            self._landmark_history.append(landmarks)
            feature_vector = normalize_landmarks_xy(landmarks)

        prediction = PredictionResult(label="UNKNOWN", confidence=0.0, top3=[("UNKNOWN", 0.0)])
        if feature_vector is not None and landmarks is not None:
            prediction = self.classifier.predict(feature_vector, landmarks)
            override = self._motion_jz_override()
            if override is not None and override.confidence >= prediction.confidence:
                prediction = override

        return {
            "label": prediction.label,
            "confidence": float(prediction.confidence),
            "top3": prediction.top3,
            "landmarks_detected": bool(landmarks),
            "feature_count": int(feature_vector.shape[0]) if feature_vector is not None else 0,
        }
