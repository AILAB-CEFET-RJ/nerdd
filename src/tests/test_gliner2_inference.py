import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gliner2_inference import attach_missing_offsets, normalize_gliner2_entities, predict_entities_for_text


class _StubModel:
    def __init__(self, entities):
        self._entities = entities

    def extract_entities(self, text, entity_types):
        return self._entities


class GLiNER2InferenceTests(unittest.TestCase):
    def test_normalize_gliner2_entities_maps_labels_and_scores(self):
        entities = [
            {"text": "Ivete Sangalo", "label": "person", "confidence": 0.8},
            {"text": "TV Globo", "label": "organization", "score": 0.7},
        ]
        normalized = normalize_gliner2_entities(entities)
        self.assertEqual(normalized[0]["label"], "Organization")
        self.assertEqual(normalized[1]["label"], "Person")
        self.assertIn("score", normalized[0])
        self.assertIn("score", normalized[1])

    def test_attach_missing_offsets_finds_entities_in_text(self):
        entities = [
            {"text": "Ivete Sangalo", "label": "Person"},
            {"text": "Xuxa Meneguel", "label": "Person"},
        ]
        attached = attach_missing_offsets("Ivete Sangalo, Xuxa Meneguel", entities)
        self.assertEqual(attached[0]["start"], 0)
        self.assertEqual(attached[0]["end"], 13)
        self.assertEqual(attached[1]["start"], 15)
        self.assertEqual(attached[1]["end"], 28)

    def test_predict_entities_for_text_filters_missing_offsets(self):
        model = _StubModel(
            [
                {"text": "Ivete Sangalo", "label": "person"},
                {"text": "inexistente", "label": "person"},
            ]
        )
        predicted = predict_entities_for_text(model, "Ivete Sangalo", ["person"])
        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0]["text"], "Ivete Sangalo")


if __name__ == "__main__":
    unittest.main()
