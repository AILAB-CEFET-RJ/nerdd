import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.review_gliner2_predictions import _normalize_entity_types, predict_entities_for_texts


class _StubModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def extract_entities(self, text, entity_types):
        return self.outputs[text]


class ReviewGLiNER2PredictionsTests(unittest.TestCase):
    def test_normalize_entity_types_maps_supported_labels(self):
        self.assertEqual(
            _normalize_entity_types(["Person", "Location", "Organization"]),
            ["person", "location", "organization"],
        )

    def test_predict_entities_for_texts_normalizes_offsets(self):
        model = _StubModel(
            {
                "Ivete Sangalo": [{"text": "Ivete Sangalo", "label": "person"}],
                "Xuxa Meneguel": [{"text": "Xuxa Meneguel", "label": "person"}],
            }
        )
        predicted = predict_entities_for_texts(model, ["Ivete Sangalo", "Xuxa Meneguel"], ["person"])
        self.assertEqual(predicted[0][0]["start"], 0)
        self.assertEqual(predicted[0][0]["end"], 13)
        self.assertEqual(predicted[1][0]["text"], "Xuxa Meneguel")


if __name__ == "__main__":
    unittest.main()
