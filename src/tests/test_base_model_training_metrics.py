import sys
import unittest
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault(
    "sklearn.metrics",
    types.SimpleNamespace(f1_score=lambda *args, **kwargs: 0.0),
)

from base_model_training.metrics import compute_macro_f1_from_span_lists


class BaseModelTrainingMetricsTests(unittest.TestCase):
    def test_compute_macro_f1_from_span_lists_matches_per_label_average(self):
        gold = [
            [
                {"start": 0, "end": 5, "label": "Person"},
                {"start": 10, "end": 15, "label": "Location"},
            ]
        ]
        pred = [
            [
                {"start": 0, "end": 5, "label": "Person"},
                {"start": 20, "end": 25, "label": "Location"},
            ]
        ]

        score = compute_macro_f1_from_span_lists(pred, gold, ["Person", "Location"])

        self.assertAlmostEqual(score, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
