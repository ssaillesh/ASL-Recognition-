from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


ALPHABET_LABELS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
WORD_LABELS: List[str] = []
SPECIAL_LABELS: List[str] = []
ALL_LABELS = ALPHABET_LABELS + WORD_LABELS + SPECIAL_LABELS


@dataclass
class PredictionResult:
    label: str
    confidence: float
    top3: List[Tuple[str, float]]


class ASLClassifier:
    """Loads a trained model (RandomForest or CNN) and evaluates gestures."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.model_type = "rf"
        self.cnn_classifier = None
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            return

        with open(self.model_path, "rb") as handle:
            payload = pickle.load(handle)

        if isinstance(payload, dict):
            model_type = str(payload.get("model_type", "")).lower()
            if model_type == "cnn1d":
                # Use the dedicated CNN classifier module
                self.model_type = "cnn1d"
                try:
                    from sign_language_app.cnn.classifier import CNNClassifier
                    self.cnn_classifier = CNNClassifier(self.model_path)
                except ImportError:
                    pass
            else:
                self.model = payload.get("model")
                self.label_encoder = payload.get("label_encoder")
        else:
            self.model = payload

    @staticmethod
    def _distance(points: Sequence[Tuple[float, float]], a: int, b: int) -> float:
        ax, ay = points[a]
        bx, by = points[b]
        return float(np.hypot(ax - bx, ay - by))

    @staticmethod
    def _finger_extended(points: Sequence[Tuple[float, float]], tip: int, pip: int, mcp: int) -> bool:
        tip_y = points[tip][1]
        pip_y = points[pip][1]
        mcp_y = points[mcp][1]
        return tip_y < pip_y < mcp_y

    def _heuristic_predict(self, points: Sequence[Tuple[float, float]]) -> PredictionResult:
        index_ext = self._finger_extended(points, 8, 6, 5)
        middle_ext = self._finger_extended(points, 12, 10, 9)
        ring_ext = self._finger_extended(points, 16, 14, 13)
        pinky_ext = self._finger_extended(points, 20, 18, 17)

        pinch_index_thumb = self._distance(points, 8, 4) < 0.08
        scored: Dict[str, float] = {}

        if pinch_index_thumb and not middle_ext and not ring_ext and not pinky_ext:
            scored["A"] = 0.8
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            scored["D"] = 0.8
        if index_ext and middle_ext and ring_ext and pinky_ext:
            scored["B"] = 0.78

        if not scored:
            return PredictionResult(label="UNKNOWN", confidence=0.0, top3=[("UNKNOWN", 0.0)])

        top3 = sorted(scored.items(), key=lambda item: item[1], reverse=True)[:3]
        label, conf = top3[0]
        return PredictionResult(label=label, confidence=float(conf), top3=[(k, float(v)) for k, v in top3])

    def _calibrate_uvr(
        self,
        points: Sequence[Tuple[float, float]],
        top3: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Resolve common U/V/R confusions using finger spread/cross geometry."""
        labels = [label for label, _ in top3]
        if not any(label in {"U", "V", "R"} for label in labels):
            return top3

        index_ext = self._finger_extended(points, 8, 6, 5)
        middle_ext = self._finger_extended(points, 12, 10, 9)
        ring_ext = self._finger_extended(points, 16, 14, 13)
        pinky_ext = self._finger_extended(points, 20, 18, 17)

        # Only calibrate when the classic two-finger pattern is present.
        if not (index_ext and middle_ext and not ring_ext and not pinky_ext):
            return top3

        tip_gap = self._distance(points, 8, 12)
        pip_gap = self._distance(points, 6, 10)

        # Detect a cross by observing index/middle x-order flip from PIP to TIP.
        pip_order = points[6][0] - points[10][0]
        tip_order = points[8][0] - points[12][0]
        crossed = pip_order * tip_order < 0

        scores = {"U": 0.0, "V": 0.0, "R": 0.0}
        if crossed:
            scores["R"] += 1.0

        if tip_gap > max(0.06, pip_gap * 1.1):
            scores["V"] += 0.9
        if tip_gap < min(0.055, pip_gap * 0.9):
            scores["U"] += 0.9

        # If there is no strong geometric signal, keep model ranking unchanged.
        best_label = max(scores, key=scores.get)
        if scores[best_label] < 0.75:
            return top3

        base = dict(top3)
        for label in ("U", "V", "R"):
            if label not in base:
                base[label] = 0.0
            base[label] += scores[label] * 0.35

        calibrated = sorted(base.items(), key=lambda item: item[1], reverse=True)[:3]
        return [(label, float(score)) for label, score in calibrated]

    def predict(self, feature_vector: np.ndarray, points: Sequence[Tuple[float, float]]) -> PredictionResult:
        if self.model is None and self.cnn_classifier is None:
            return self._heuristic_predict(points)

        if self.model_type == "cnn1d" and self.cnn_classifier is not None:
            # Use the dedicated CNN classifier
            label, confidence, top3_model = self.cnn_classifier.predict(feature_vector)
            
            # Apply U/V/R calibration
            top3_model = self._calibrate_uvr(points, top3_model)

            if top3_model[0][1] < 0.58:
                return PredictionResult(label="UNKNOWN", confidence=float(top3_model[0][1]), top3=top3_model)

            label, confidence = top3_model[0]
            return PredictionResult(label=label, confidence=confidence, top3=top3_model)

        if self.model is None:
            return self._heuristic_predict(points)

        # RandomForest path
        probabilities = self.model.predict_proba([feature_vector])[0]
        indices = np.argsort(probabilities)[::-1][:3]

        labels = self.model.classes_
        if self.label_encoder is not None:
            labels = self.label_encoder.inverse_transform(labels.astype(int))

        top3_model: List[Tuple[str, float]] = [(str(labels[idx]), float(probabilities[idx])) for idx in indices]
        top3_model = self._calibrate_uvr(points, top3_model)

        # Reject unstable low-confidence outputs instead of forcing a wrong letter.
        if top3_model[0][1] < 0.58:
            return PredictionResult(label="UNKNOWN", confidence=float(top3_model[0][1]), top3=top3_model)
        
        label, confidence = top3_model[0]
        return PredictionResult(label=label, confidence=confidence, top3=top3_model)
