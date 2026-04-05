from __future__ import annotations

import os
import pickle
import unittest


MODEL_PATH = os.path.join("sign_language_app", "models", "asl_model.pkl")


class TestLabelMapping(unittest.TestCase):
    def test_wrapper_has_consistent_class_mapping(self) -> None:
        if not os.path.exists(MODEL_PATH):
            self.skipTest("model wrapper not found")

        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or payload.get("model_type") != "cnn1d":
            self.skipTest("not a cnn1d wrapper")

        classes = payload.get("classes", [])
        self.assertEqual(len(classes), 26)

        class_to_idx = payload.get("class_to_idx", {})
        idx_to_class = payload.get("idx_to_class", {})
        if class_to_idx and idx_to_class:
            self.assertEqual(set(class_to_idx.keys()), set(classes))
            for label, idx in class_to_idx.items():
                self.assertEqual(str(idx_to_class.get(int(idx))), str(label))


if __name__ == "__main__":
    unittest.main()
