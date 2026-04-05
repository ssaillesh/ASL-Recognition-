from __future__ import annotations

from collections import deque
import unittest

from sign_language_app.classifier import ASLClassifier
from sign_language_app.ui.camera_panel import CameraPanel


class TestLetterCalibration(unittest.TestCase):
    def _blank_points(self):
        return [(0.5, 0.5) for _ in range(21)]

    def test_calibrate_ab_prefers_a_for_closed_fist(self) -> None:
        clf = ASLClassifier.__new__(ASLClassifier)
        points = self._blank_points()
        # Fold non-thumb fingers and keep thumb close to wrist.
        for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
            points[tip] = (0.5, 0.62)
            points[pip] = (0.5, 0.56)
            points[mcp] = (0.5, 0.52)
        points[3] = (0.45, 0.52)
        points[4] = (0.47, 0.55)

        top3 = [("T", 0.91), ("A", 0.09), ("B", 0.01)]
        calibrated = ASLClassifier._calibrate_ab(clf, points, top3)

        self.assertEqual(calibrated[0][0], "A")

    def test_calibrate_ab_prefers_b_for_open_palm(self) -> None:
        clf = ASLClassifier.__new__(ASLClassifier)
        points = self._blank_points()
        for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
            points[tip] = (0.5, 0.30)
            points[pip] = (0.5, 0.44)
            points[mcp] = (0.5, 0.54)
        points[3] = (0.44, 0.56)
        points[4] = (0.46, 0.58)

        top3 = [("D", 0.91), ("B", 0.05), ("A", 0.04)]
        calibrated = ASLClassifier._calibrate_ab(clf, points, top3)

        self.assertEqual(calibrated[0][0], "B")

    def test_motion_override_detects_z_like_path(self) -> None:
        panel = object.__new__(CameraPanel)
        panel._landmark_history = deque(maxlen=12)
        for x, y in ((0.10, 0.10), (0.18, 0.10), (0.26, 0.16), (0.18, 0.22), (0.10, 0.28), (0.18, 0.34), (0.28, 0.34), (0.36, 0.34)):
            pts = [(0.5, 0.5) for _ in range(21)]
            pts[8] = (x, y)
            panel._landmark_history.append(pts)

        result = CameraPanel._motion_jz_override(panel)
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "Z")

    def test_motion_override_detects_j_like_path(self) -> None:
        panel = object.__new__(CameraPanel)
        panel._landmark_history = deque(maxlen=12)
        for x, y in ((0.20, 0.10), (0.20, 0.18), (0.20, 0.26), (0.18, 0.34), (0.16, 0.42), (0.14, 0.48), (0.12, 0.54), (0.10, 0.60)):
            pts = [(0.5, 0.5) for _ in range(21)]
            pts[8] = (x, y)
            panel._landmark_history.append(pts)

        result = CameraPanel._motion_jz_override(panel)
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "J")


if __name__ == "__main__":
    unittest.main()
