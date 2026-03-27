import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.review_model_predictions import enrich_rows


class ReviewModelPredictionsTests(unittest.TestCase):
    def test_enrich_rows_sorts_worst_first(self):
        rows = [
            {"text": "John", "spans": [{"start": 0, "end": 4, "label": "Person"}]},
            {"text": "Rio", "spans": [{"start": 0, "end": 3, "label": "Location"}]},
        ]
        preds = [
            [],
            [{"start": 0, "end": 3, "label": "Location"}],
        ]

        enriched = enrich_rows(rows, preds)

        self.assertEqual(enriched[0]["text"], "John")
        self.assertEqual(enriched[0]["_review"]["fn"], 1)
        self.assertEqual(enriched[1]["_review"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
