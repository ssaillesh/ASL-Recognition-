from __future__ import annotations

import unittest

import numpy as np

from sign_language_app.preprocessing import normalize_landmark_tensor, normalize_landmarks_xy


class TestPreprocessing(unittest.TestCase):
    def test_normalize_tensor_shape_and_origin(self) -> None:
        points = np.array([[0.5 + i * 0.01, 0.4 + i * 0.01] for i in range(21)], dtype=np.float32)
        normalized = normalize_landmark_tensor(points)

        self.assertEqual(normalized.shape, (21, 2))
        self.assertAlmostEqual(float(normalized[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[0, 1]), 0.0, places=6)
        self.assertLessEqual(float(np.max(np.linalg.norm(normalized, axis=1))), 1.0 + 1e-6)

    def test_normalize_xy_flattens_to_42(self) -> None:
        points = [(0.2 + i * 0.01, 0.3 + i * 0.02) for i in range(21)]
        vec = normalize_landmarks_xy(points)

        self.assertEqual(vec.shape, (42,))
        self.assertTrue(np.isfinite(vec).all())


if __name__ == "__main__":
    unittest.main()
