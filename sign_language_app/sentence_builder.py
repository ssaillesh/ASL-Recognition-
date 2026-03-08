from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import pyttsx3  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency at runtime
    pyttsx3 = None


@dataclass
class SentenceState:
    confirmed_label: str
    sentence: str


class SentenceBuilder:
    def __init__(self, confidence_threshold: float = 0.8, hold_seconds: float = 0.8) -> None:
        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds

        self.current_candidate = ""
        self.candidate_since = 0.0
        self.last_confirmed = ""

        self.sentence = ""
        self.clear_started_at = 0.0
        self.tts = pyttsx3.init() if pyttsx3 else None
        if self.tts:
            self.tts.setProperty("rate", 155)

    def _append_token(self, token: str) -> None:
        if token == "SPACE":
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "
            return
        if len(token) == 1 and token.isalpha():
            self.sentence += token
        elif token not in {"CLEAR", "CONFIRM"}:
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "
            self.sentence += token
            self.sentence += " "

    def _speak(self) -> None:
        text = self.sentence.strip()
        if not text or self.tts is None:
            return
        self.tts.say(text)
        self.tts.runAndWait()

    def update(self, label: str, confidence: float, now: float | None = None) -> SentenceState:
        now = now or time.time()
        confirmed = ""

        if label == "CLEAR":
            if self.clear_started_at == 0.0:
                self.clear_started_at = now
            if now - self.clear_started_at >= 2.0:
                self.sentence = ""
                self.current_candidate = ""
                self.last_confirmed = ""
                self.clear_started_at = 0.0
                confirmed = "CLEAR"
            return SentenceState(confirmed_label=confirmed, sentence=self.sentence)

        self.clear_started_at = 0.0

        if label == "CONFIRM" and confidence >= self.confidence_threshold:
            self._speak()
            return SentenceState(confirmed_label="CONFIRM", sentence=self.sentence)

        if confidence < self.confidence_threshold:
            self.current_candidate = ""
            self.candidate_since = 0.0
            return SentenceState(confirmed_label="", sentence=self.sentence)

        if label != self.current_candidate:
            self.current_candidate = label
            self.candidate_since = now
            return SentenceState(confirmed_label="", sentence=self.sentence)

        if now - self.candidate_since >= self.hold_seconds:
            if label != self.last_confirmed:
                self._append_token(label)
                self.last_confirmed = label
                confirmed = label
            self.current_candidate = ""
            self.candidate_since = 0.0

        return SentenceState(confirmed_label=confirmed, sentence=self.sentence)

    def clear(self) -> None:
        self.sentence = ""
        self.current_candidate = ""
        self.candidate_since = 0.0
        self.last_confirmed = ""
