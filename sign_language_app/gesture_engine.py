from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import mediapipe as mp
import numpy as np

from sign_language_app.preprocessing import normalize_landmarks_xy

Point = Tuple[float, float]


class GestureEngine:
    """Captures webcam frames, extracts landmarks, and builds normalized features."""

    def __init__(self, camera_index: int = 0, frame_width: int = 960, frame_height: int = 540) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.mp_hands = cast(Any, mp.solutions.hands)
        self.mp_draw = cast(Any, mp.solutions.drawing_utils)
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.prev_time = time.time()
        self.fps = 0.0

    def close(self) -> None:
        self.cap.release()
        self.hands.close()

    def _normalized_landmarks(self, points: List[Point]) -> np.ndarray:
        return normalize_landmarks_xy(points)

    def _fingertip_colors(self, idx: int) -> Tuple[int, int, int]:
        colors = {
            4: (0, 255, 255),
            8: (0, 255, 0),
            12: (255, 255, 0),
            16: (255, 0, 255),
            20: (0, 128, 255),
        }
        return colors.get(idx, (0, 255, 255))

    def _draw_landmarks(self, frame: np.ndarray, landmarks: List[Point], mp_landmarks) -> None:
        self.mp_draw.draw_landmarks(frame, mp_landmarks, list(self.mp_hands.HAND_CONNECTIONS))
        h, w, _ = frame.shape
        for idx in (4, 8, 12, 16, 20):
            x = int(landmarks[idx][0] * w)
            y = int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 8, self._fingertip_colors(idx), -1)

    def read(self) -> Optional[Dict[str, object]]:
        success, frame = self.cap.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        landmarks: Optional[List[Point]] = None
        feature_vector = None

        multi_hand_landmarks = getattr(result, "multi_hand_landmarks", None)
        if multi_hand_landmarks:
            hand_landmarks = cast(Any, multi_hand_landmarks[0])
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            feature_vector = self._normalized_landmarks(landmarks)
            self._draw_landmarks(frame, landmarks, hand_landmarks)

        now = time.time()
        delta = now - self.prev_time
        if delta > 0:
            self.fps = 1.0 / delta
        self.prev_time = now

        return {
            "frame": frame,
            "landmarks": landmarks,
            "feature_vector": feature_vector,
            "fps": self.fps,
        }
