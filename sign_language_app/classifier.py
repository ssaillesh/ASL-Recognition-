from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


ALPHABET_LABELS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
WORD_LABELS = [
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "PLEASE",
    "SORRY",
    "HELP",
    "I LOVE YOU",
    "GOOD",
    "BAD",
]
SPECIAL_LABELS = ["SPACE", "CLEAR", "CONFIRM"]
ALL_LABELS = ALPHABET_LABELS + WORD_LABELS + SPECIAL_LABELS


@dataclass
class PredictionResult:
    label: str
    confidence: float
    top3: List[Tuple[str, float]]


class ASLClassifier:
    """Loads a trained model if available and falls back to simple geometric heuristics."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            return

        with open(self.model_path, "rb") as handle:
            payload = pickle.load(handle)

        if isinstance(payload, dict):
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
        pinch_middle_thumb = self._distance(points, 12, 4) < 0.08
        thumb_up = points[4][1] < points[3][1] < points[2][1] and points[4][1] < points[8][1]

        open_palm = index_ext and middle_ext and ring_ext and pinky_ext
        fist = not index_ext and not middle_ext and not ring_ext and not pinky_ext

        scored: Dict[str, float] = {
            "SPACE": 0.9 if open_palm else 0.05,
            "CLEAR": 0.92 if fist else 0.05,
            "CONFIRM": 0.92 if thumb_up else 0.05,
            "HELLO": 0.75 if open_palm else 0.05,
            "YES": 0.72 if fist else 0.05,
            "NO": 0.7 if index_ext and middle_ext and not ring_ext and not pinky_ext else 0.05,
            "PLEASE": 0.72 if pinch_middle_thumb else 0.05,
            "HELP": 0.7 if pinch_index_thumb and middle_ext else 0.05,
            "I LOVE YOU": 0.78 if index_ext and pinky_ext and thumb_up else 0.05,
            "GOOD": 0.7 if open_palm and not thumb_up else 0.05,
            "BAD": 0.66 if open_palm and thumb_up else 0.05,
            "SORRY": 0.68 if fist and thumb_up else 0.05,
            "THANK YOU": 0.7 if pinch_index_thumb and open_palm else 0.05,
        }

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

    def predict(self, feature_vector: np.ndarray, points: Sequence[Tuple[float, float]]) -> PredictionResult:
        if self.model is None:
            return self._heuristic_predict(points)

        # Get model predictions
        probabilities = self.model.predict_proba([feature_vector])[0]
        indices = np.argsort(probabilities)[::-1][:3]

        labels = self.model.classes_
        if self.label_encoder is not None:
            labels = self.label_encoder.inverse_transform(labels.astype(int))

        top3_model: List[Tuple[str, float]] = [(str(labels[idx]), float(probabilities[idx])) for idx in indices]
        
        # If model confidence is too low, try heuristic as well
        if top3_model[0][1] < 0.5:
            heuristic_result = self._heuristic_predict(points)
            # If heuristic has better confidence, use it
            if heuristic_result.confidence >= top3_model[0][1]:
                return heuristic_result
        
        label, confidence = top3_model[0]
        return PredictionResult(label=label, confidence=confidence, top3=top3_model)
