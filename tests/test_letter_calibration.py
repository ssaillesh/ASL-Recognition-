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
        points[3] = (0.47, 0.53)
        points[4] = (0.49, 0.51)

        top3 = [("T", 0.41), ("A", 0.39), ("B", 0.01)]
        calibrated = ASLClassifier._calibrate_ab(clf, points, top3)

        self.assertEqual(calibrated[0][0], "A")

    def test_calibrate_ab_prefers_b_for_open_palm(self) -> None:
        clf = ASLClassifier.__new__(ASLClassifier)
        points = self._blank_points()
        for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
            points[tip] = (0.5, 0.30)
            points[pip] = (0.5, 0.44)
            points[mcp] = (0.5, 0.54)
        points[3] = (0.47, 0.53)
        points[4] = (0.49, 0.51)

        top3 = [("D", 0.43), ("B", 0.40), ("A", 0.04)]
        calibrated = ASLClassifier._calibrate_ab(clf, points, top3)

        self.assertEqual(calibrated[0][0], "B")

    def test_calibrate_ah_prefers_h_for_two_fingers(self) -> None:
        clf = ASLClassifier.__new__(ASLClassifier)
        points = self._blank_points()
        points[5] = (0.42, 0.54)
        points[6] = (0.42, 0.42)
        points[8] = (0.42, 0.28)
        points[9] = (0.52, 0.54)
        points[10] = (0.52, 0.42)
        points[12] = (0.52, 0.28)
        points[13] = (0.56, 0.58)
        points[14] = (0.56, 0.62)
        points[16] = (0.56, 0.66)
        points[17] = (0.60, 0.58)
        points[18] = (0.60, 0.62)
        points[20] = (0.60, 0.66)
        points[3] = (0.46, 0.53)
        points[4] = (0.49, 0.51)

        top3 = [("A", 0.84), ("H", 0.16), ("B", 0.02)]
        calibrated = ASLClassifier._calibrate_ah(clf, points, top3)

        self.assertEqual(calibrated[0][0], "H")

    def test_calibrate_oz_prefers_o_for_closed_circle(self) -> None:
        clf = ASLClassifier.__new__(ASLClassifier)
        points = self._blank_points()
        for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
            offset = (tip - 8) * 0.01
            points[tip] = (0.50 + offset, 0.64)
            points[pip] = (0.50 + offset, 0.58)
            points[mcp] = (0.50 + offset, 0.52)
        points[3] = (0.47, 0.53)
        points[4] = (0.49, 0.51)

        top3 = [("Z", 0.81), ("O", 0.19), ("A", 0.01)]
        calibrated = ASLClassifier._calibrate_oz(clf, points, top3)

        self.assertEqual(calibrated[0][0], "O")

    def test_motion_override_detects_z_like_path(self) -> None:
        panel = object.__new__(CameraPanel)
        panel._landmark_history = deque(maxlen=12)
        for x, y in (
            (0.10, 0.10),
            (0.21, 0.10),
            (0.30, 0.15),
            (0.20, 0.20),
            (0.10, 0.26),
            (0.22, 0.30),
            (0.33, 0.34),
            (0.20, 0.38),
            (0.18, 0.42),
        ):
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
