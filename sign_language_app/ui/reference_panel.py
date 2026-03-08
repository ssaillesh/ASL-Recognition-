from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

from sign_language_app.classifier import ALPHABET_LABELS, WORD_LABELS


class ReferencePanel(ttk.Frame):
    def __init__(self, parent: tk.Misc, assets_root: str) -> None:
        super().__init__(parent)
        self.assets_root = assets_root
        self._cards: Dict[str, ttk.Frame] = {}
        self._thumbnails: Dict[str, ImageTk.PhotoImage] = {}

        self.search_var = tk.StringVar()

        search_row = ttk.Frame(self)
        search_row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(search_row, text="Search:").pack(side=tk.LEFT)
        search = ttk.Entry(search_row, textvariable=self.search_var)
        search.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.search_var.trace_add("write", self._apply_filter)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        self.alpha_container, self.alpha_frame = self._make_scroll_tab(self.tabs)
        self.word_container, self.word_frame = self._make_scroll_tab(self.tabs)

        self.tabs.add(self.alpha_container, text="Alphabet")
        self.tabs.add(self.word_container, text="Words")

        self._populate(self.alpha_frame, ALPHABET_LABELS)
        self._populate(self.word_frame, WORD_LABELS)

    def _make_scroll_tab(self, parent: ttk.Notebook) -> Tuple[ttk.Frame, ttk.Frame]:
        container = ttk.Frame(parent)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        return container, inner

    def _image_for(self, label: str) -> ImageTk.PhotoImage:
        filename = f"{label}.png" if len(label) == 1 else f"{label.replace(' ', '_')}.png"
        path = os.path.join(self.assets_root, "gesture_icons", filename)

        if os.path.exists(path):
            img = Image.open(path).resize((84, 84))
        else:
            img = Image.new("RGB", (84, 84), color=(33, 38, 45))
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle((2, 2, 82, 82), radius=10, outline=(230, 210, 80), width=2)
            short = label if len(label) <= 8 else label[:8]
            draw.text((10, 33), short, fill=(255, 255, 255), font=ImageFont.load_default())

        return ImageTk.PhotoImage(img)

    def _populate(self, parent: ttk.Frame, labels: Iterable[str]) -> None:
        for label in labels:
            card = ttk.Frame(parent, padding=8)
            card.pack(fill=tk.X, padx=6, pady=5)

            thumb = self._image_for(label)
            self._thumbnails[label] = thumb
            ttk.Label(card, image=thumb).pack(side=tk.LEFT)

            text_col = ttk.Frame(card)
            text_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            ttk.Label(text_col, text=label, font=("Helvetica", 13, "bold")).pack(anchor="w")
            tip = "Hold steady and keep fingers visible."
            ttk.Label(text_col, text=tip, foreground="#cccccc").pack(anchor="w")

            self._cards[label] = card

    def _apply_filter(self, *_args) -> None:
        query = self.search_var.get().strip().lower()
        for label, card in self._cards.items():
            visible = not query or query in label.lower()
            if visible:
                card.pack(fill=tk.X, padx=6, pady=5)
            else:
                card.pack_forget()

    def highlight(self, label: str) -> None:
        for key, card in self._cards.items():
            if key == label:
                card.configure(style="Highlight.TFrame")
            else:
                card.configure(style="TFrame")
