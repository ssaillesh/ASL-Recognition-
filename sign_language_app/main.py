from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import ttk

from sign_language_app.assets_bootstrap import ensure_assets
from sign_language_app.ui.camera_panel import CameraPanel
from sign_language_app.ui.practice_mode import PracticeModePanel
from sign_language_app.ui.reference_panel import ReferencePanel


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sign Language Recognition")
        self.geometry("1460x860")
        self.configure(bg="#0f1115")

        self._configure_styles()

        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(root_dir, "models", "asl_model.pkl")
        assets_root = os.path.join(root_dir, "assets")
        ensure_assets(assets_root)

        main_split = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_split.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main_split)
        right = ttk.Notebook(main_split)

        main_split.add(left, weight=3)
        main_split.add(right, weight=2)

        self.reference_panel = ReferencePanel(right, assets_root=assets_root)
        self.practice_panel = PracticeModePanel(right)

        right.add(self.reference_panel, text="Reference Guide")
        right.add(self.practice_panel, text="Practice Mode")

        self.camera_panel = CameraPanel(left, model_path=model_path, on_label=self._on_detected_label)
        self.camera_panel.pack(fill=tk.BOTH, expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#12161c")
        style.configure("TLabel", background="#12161c", foreground="#f2f2f2")
        style.configure("TButton", font=("Helvetica", 12, "bold"))
        style.configure("Highlight.TFrame", background="#2c3e0f")

    def _on_detected_label(self, label: str) -> None:
        self.reference_panel.highlight(label)
        if label:
            prediction = self.camera_panel.current_prediction
            self.practice_panel.update_prediction(label, prediction.confidence)

    def _on_close(self) -> None:
        self.camera_panel.shutdown()
        self.destroy()


def main() -> None:
    if os.environ.get("ASL_DEBUG_RT", "0") == "1" or os.environ.get("ASL_DEBUG_CNN", "0") == "1":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
