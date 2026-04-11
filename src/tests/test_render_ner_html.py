import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.render_ner_html import get_spans


class RenderNerHtmlTests(unittest.TestCase):
    def test_get_spans_supports_review_seed_entities(self):
        row = {
            "text": "Rua Alpha com Joao",
            "review_seed_entities": [
                {"text": "Rua Alpha", "label": "Location", "start": 0, "end": 9},
                {"text": "Joao", "label": "Person", "start": 14, "end": 18},
            ],
        }
        spans = get_spans(row, span_field="review_seed_entities")
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["label"], "Location")

    def test_get_spans_supports_nested_adjudication_entities_final(self):
        row = {
            "text": "Rua Alpha com Joao",
            "adjudication": {
                "entities_final": [
                    {"text": "Rua Alpha", "label": "Location", "start": 0, "end": 9},
                ]
            },
        }
        spans = get_spans(row, span_field="adjudication.entities_final")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["text"], "Rua Alpha")


if __name__ == "__main__":
    unittest.main()
