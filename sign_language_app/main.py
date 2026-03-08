from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

from sign_language_app.assets_bootstrap import ensure_assets
from sign_language_app.ui.camera_panel import CameraPanel
from sign_language_app.ui.practice_mode import PracticeModePanel
from sign_language_app.ui.reference_panel import ReferencePanel


class WelcomeDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.title("Welcome")
        self.geometry("680x360")
        self.configure(bg="#101316")
        self.transient(parent)
        self.grab_set()

        title = tk.Label(
            self,
            text="ASL Recognition App",
            font=("Helvetica", 26, "bold"),
            fg="#f8f8f8",
            bg="#101316",
        )
        title.pack(pady=(24, 10))

        body = (
            "1. Keep one hand visible and centered in frame.\n"
            "2. Hold a gesture for about 0.8 seconds to confirm.\n"
            "3. Open palm inserts SPACE. Closed fist for 2 seconds clears sentence.\n"
            "4. Thumbs up triggers text-to-speech for the current sentence.\n"
            "5. Press Quit to close the app cleanly."
        )
        tk.Label(self, text=body, justify=tk.LEFT, font=("Helvetica", 14), fg="#e3e3e3", bg="#101316").pack(padx=24, pady=8)

        tk.Button(self, text="Start", font=("Helvetica", 14, "bold"), command=self.destroy, bg="#f0cd4c", fg="#111111").pack(pady=18)


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
    app = App()
    dialog = WelcomeDialog(app)
    app.wait_window(dialog)
    app.mainloop()


if __name__ == "__main__":
    main()
