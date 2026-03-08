from __future__ import annotations

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
from sign_language_app.sentence_builder import SentenceBuilder


class CameraPanel(ttk.Frame):
    def __init__(self, parent: tk.Misc, model_path: str, on_label=None) -> None:
        super().__init__(parent)
        self.on_label = on_label
        self.engine = GestureEngine()
        self.classifier = ASLClassifier(model_path)
        self.builder = SentenceBuilder(confidence_threshold=0.8, hold_seconds=0.8)

        self.classify_every = 2
        self.frame_index = 0

        self._latest_image = None
        self._current_prediction = PredictionResult(label="", confidence=0.0, top3=[])
        self._last_label: Optional[str] = None

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
        except queue.Empty:
            return

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

        self.frame_index += 1
        if self.frame_index % self.classify_every == 0:
            self._push_for_classification(feature_vector, landmarks)

        self._poll_prediction()

        state = self.builder.update(self._current_prediction.label, self._current_prediction.confidence)

        if self.on_label and self._current_prediction.label != self._last_label:
            self.on_label(self._current_prediction.label)
            self._last_label = self._current_prediction.label

        self.gesture_var.set(
            f"Detected: {self._current_prediction.label or '--'} ({int(self._current_prediction.confidence * 100)}%)"
        )
        self.sentence_var.set(f"Sentence: {state.sentence}")
        self._update_confidence_meter(self._current_prediction)

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
