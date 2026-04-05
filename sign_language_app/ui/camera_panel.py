from __future__ import annotations

from collections import deque
import logging
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Optional, cast

import cv2
import numpy as np
from PIL import Image, ImageTk

from sign_language_app.classifier import ASLClassifier, PredictionResult
from sign_language_app.gesture_engine import GestureEngine
from sign_language_app.preprocessing import feature_stats
from sign_language_app.sentence_builder import SentenceBuilder


LOGGER = logging.getLogger(__name__)


class CameraPanel(ttk.Frame):
    def __init__(self, parent: tk.Misc, model_path: str, on_label=None) -> None:
        super().__init__(parent)
        self.on_label = on_label
        self.engine = GestureEngine()
        self.classifier = ASLClassifier(model_path)
        self.builder = SentenceBuilder(confidence_threshold=0.62, hold_seconds=0.5)
        self.debug_runtime = os.environ.get("ASL_DEBUG_RT", "0") == "1"
        self.save_uncertain = os.environ.get("ASL_SAVE_UNCERTAIN_FRAMES", "0") == "1"
        self._last_debug_frame = -999

        if self.save_uncertain:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.debug_frame_dir = os.path.join(root_dir, "debug_frames")
            os.makedirs(self.debug_frame_dir, exist_ok=True)
        else:
            self.debug_frame_dir = ""

        self.classify_every = 2
        self.frame_index = 0

        self._latest_image = None
        self._current_prediction = PredictionResult(label="", confidence=0.0, top3=[])
        self._last_label: Optional[str] = None
        self._prediction_window = deque(maxlen=7)
        self._landmark_history = deque(maxlen=12)

        self.in_queue: queue.Queue = queue.Queue(maxsize=2)
        self.out_queue: queue.Queue = queue.Queue(maxsize=2)
        self.worker = threading.Thread(target=self._classification_worker, daemon=True)
        self.worker.start()

        self.camera_label = ttk.Label(self)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.gesture_var = tk.StringVar(value="Detected: --")
        self.sentence_var = tk.StringVar(value="Sentence: ")
        ttk.Label(bottom, textvariable=self.gesture_var, font=("Helvetica", 22, "bold")).pack(anchor="w")
        ttk.Label(bottom, textvariable=self.sentence_var, font=("Helvetica", 16)).pack(anchor="w", pady=(4, 0))

        meter = ttk.Frame(self)
        meter.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.bars = []
        self.bar_vars = []
        for _idx in range(3):
            label_var = tk.StringVar(value="--")
            bar = ttk.Progressbar(meter, orient=tk.HORIZONTAL, maximum=100)
            ttk.Label(meter, textvariable=label_var).pack(anchor="w")
            bar.pack(fill=tk.X, pady=(0, 5))
            self.bar_vars.append(label_var)
            self.bars.append(bar)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Button(controls, text="Clear Sentence", command=self.builder.clear).pack(side=tk.LEFT)

        self.after(10, self._render_loop)

    def _classification_worker(self) -> None:
        while True:
            feature_vector, landmarks = self.in_queue.get()
            prediction = self.classifier.predict(feature_vector, landmarks)
            if self.out_queue.full():
                _ = self.out_queue.get_nowait()
            self.out_queue.put(prediction)

    def _push_for_classification(self, feature_vector, landmarks) -> None:
        if feature_vector is None or landmarks is None:
            return
        if self.in_queue.full():
            return
        self.in_queue.put((feature_vector, landmarks))

    def _poll_prediction(self) -> None:
        try:
            while True:
                self._current_prediction = self.out_queue.get_nowait()
                self._prediction_window.append(self._current_prediction)
        except queue.Empty:
            return

    def _smoothed_prediction(self) -> PredictionResult:
        if not self._prediction_window:
            return self._current_prediction

        score_by_label = {}
        top3_by_label = {}
        for pred in self._prediction_window:
            if not pred.label:
                continue
            score_by_label[pred.label] = score_by_label.get(pred.label, 0.0) + float(pred.confidence)
            for name, score in pred.top3:
                top3_by_label[name] = top3_by_label.get(name, 0.0) + float(score)

        if not score_by_label:
            return self._current_prediction

        ranked = sorted(score_by_label.items(), key=lambda item: item[1], reverse=True)
        label, score_sum = ranked[0]
        confidence = score_sum / max(1, len(self._prediction_window))

        ranked_top3 = sorted(top3_by_label.items(), key=lambda item: item[1], reverse=True)[:3]
        normalized_top3 = [(name, score / max(1, len(self._prediction_window))) for name, score in ranked_top3]
        return PredictionResult(label=label, confidence=confidence, top3=normalized_top3)

    def _motion_jz_override(self) -> Optional[PredictionResult]:
        if len(self._landmark_history) < 8:
            return None

        points = [np.asarray(sample[8], dtype=np.float32) for sample in self._landmark_history if sample is not None and len(sample) > 8]
        if len(points) < 8:
            return None

        traj = np.asarray(points, dtype=np.float32)
        diffs = np.diff(traj, axis=0)
        if diffs.size == 0:
            return None

        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
        net = traj[-1] - traj[0]
        dx = diffs[:, 0]
        dy = diffs[:, 1]

        sign_changes = int(np.sum(np.sign(dx[:-1]) != np.sign(dx[1:]))) if len(dx) > 2 else 0
        vertical_drop = float(net[1])
        horizontal_move = float(abs(net[0]))

        # Z: require stronger zig-zag evidence to avoid accidental overrides from jitter.
        if path_len > 0.18 and sign_changes >= 3 and horizontal_move > 0.05 and float(np.max(np.abs(dx))) > 0.012:
            return PredictionResult(label="Z", confidence=0.86, top3=[("Z", 0.86), ("UNKNOWN", 0.14)])

        # J: downward hook-like motion. Keep the threshold conservative to reduce false positives.
        if path_len > 0.16 and vertical_drop > 0.06 and horizontal_move > 0.015:
            return PredictionResult(label="J", confidence=0.84, top3=[("J", 0.84), ("UNKNOWN", 0.16)])

        return None

    def _draw_overlay(self, frame, fps: float) -> None:
        p = self._current_prediction
        label = p.label or "--"
        confidence_text = f"{int(p.confidence * 100)}%"

        cv2.rectangle(frame, (10, 10), (480, 82), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {label}", (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(frame, f"Conf: {confidence_text}", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 80), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    def _update_confidence_meter(self, prediction: PredictionResult) -> None:
        entries = prediction.top3 + [("--", 0.0)] * 3
        for idx in range(3):
            name, score = entries[idx]
            self.bar_vars[idx].set(f"{idx + 1}. {name} ({int(score * 100)}%)")
            self.bars[idx]["value"] = int(score * 100)

    def _render_loop(self) -> None:
        payload = self.engine.read()
        if payload is None:
            self.after(10, self._render_loop)
            return

        frame = cast(np.ndarray, payload["frame"])
        landmarks = payload["landmarks"]
        feature_vector = payload["feature_vector"]

        if landmarks is not None:
            self._landmark_history.append(landmarks)

        self.frame_index += 1
        if self.frame_index % self.classify_every == 0:
            self._push_for_classification(feature_vector, landmarks)

        self._poll_prediction()
        stable_prediction = self._smoothed_prediction()

        motion_override = self._motion_jz_override()
        allow_motion_override = stable_prediction.label in {"J", "Z"} or stable_prediction.confidence < 0.68
        if (
            motion_override is not None
            and allow_motion_override
            and motion_override.confidence >= stable_prediction.confidence
        ):
            stable_prediction = motion_override

        state = self.builder.update(stable_prediction.label, stable_prediction.confidence)

        if self.debug_runtime and feature_vector is not None and self.frame_index - self._last_debug_frame >= 10:
            stats = feature_stats(np.asarray(feature_vector, dtype=np.float32))
            raw = self._current_prediction
            LOGGER.info(
                "rt frame=%d raw=%s(%.3f) smooth=%s(%.3f) f[min=%.4f max=%.4f mean=%.4f std=%.4f]",
                self.frame_index,
                raw.label,
                float(raw.confidence),
                stable_prediction.label,
                float(stable_prediction.confidence),
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["std"],
            )
            self._last_debug_frame = self.frame_index

        if self.save_uncertain and stable_prediction.confidence < 0.5 and self.frame_index % 15 == 0:
            path = os.path.join(self.debug_frame_dir, f"frame_{self.frame_index:06d}.jpg")
            cv2.imwrite(path, frame)

        if self.on_label and stable_prediction.label != self._last_label:
            self.on_label(stable_prediction.label)
            self._last_label = stable_prediction.label

        self.gesture_var.set(
            f"Detected: {stable_prediction.label or '--'} ({int(stable_prediction.confidence * 100)}%)"
        )
        self.sentence_var.set(f"Sentence: {state.sentence}")
        self._update_confidence_meter(stable_prediction)

        fps = cast(float, payload["fps"])
        self._draw_overlay(frame, fps)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = image.resize((900, 500))
        self._latest_image = ImageTk.PhotoImage(image=image)
        self.camera_label.configure(image=self._latest_image)

        self.after(10, self._render_loop)

    def shutdown(self) -> None:
        self.engine.close()

    @property
    def current_prediction(self) -> PredictionResult:
        return self._current_prediction
