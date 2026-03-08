from __future__ import annotations

import random
import tkinter as tk
from tkinter import ttk

from sign_language_app.classifier import ALPHABET_LABELS, WORD_LABELS


class PracticeModePanel(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.mode_var = tk.StringVar(value="alphabet")
        self.target = "A"
        self.streak = 0
        self.score = 0

        header = ttk.Label(self, text="Practice Mode", font=("Helvetica", 18, "bold"))
        header.pack(anchor="w", padx=10, pady=(10, 4))

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=10, pady=6)
        ttk.Radiobutton(controls, text="Alphabet", variable=self.mode_var, value="alphabet", command=self.next_prompt).pack(side=tk.LEFT)
        ttk.Radiobutton(controls, text="Common Words", variable=self.mode_var, value="words", command=self.next_prompt).pack(side=tk.LEFT, padx=10)

        self.target_label = ttk.Label(self, text="", font=("Helvetica", 34, "bold"))
        self.target_label.pack(pady=14)

        self.feedback = ttk.Label(self, text="Show the target sign to begin.", font=("Helvetica", 16))
        self.feedback.pack(pady=4)

        self.stats = ttk.Label(self, text="Score: 0 | Streak: 0", font=("Helvetica", 14))
        self.stats.pack(pady=4)

        ttk.Button(self, text="Next Prompt", command=self.next_prompt).pack(pady=8)
        self.next_prompt()

    def next_prompt(self) -> None:
        pool = ALPHABET_LABELS if self.mode_var.get() == "alphabet" else WORD_LABELS
        self.target = random.choice(pool)
        self.target_label.configure(text=self.target)
        self.feedback.configure(text="Show this sign and hold it steady.")

    def update_prediction(self, label: str, confidence: float) -> None:
        if not label:
            return
        if label == self.target and confidence >= 0.8:
            self.score += 1
            self.streak += 1
            self.feedback.configure(text="Correct")
            self.next_prompt()
        else:
            self.streak = 0
            self.feedback.configure(text="Try Again")

        self.stats.configure(text=f"Score: {self.score} | Streak: {self.streak}")
