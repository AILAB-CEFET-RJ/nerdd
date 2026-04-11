import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.render_ner_html import DEFAULT_SCORE_FIELDS, get_spans, parse_args, render_text_with_spans


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

    def test_render_text_with_spans_shows_score_when_configured(self):
        text = "Rua Alpha com Joao"
        spans = [
            {"text": "Rua Alpha", "label": "Location", "start": 0, "end": 9, "score_context_boosted": 0.91234},
        ]
        html = render_text_with_spans(text, spans, {"Location": "#0B6E4F"}, score_fields=["score_context_boosted"])
        self.assertIn("0.912", html)
        self.assertIn("score_context_boosted", html)

    def test_parse_args_uses_default_score_fields(self):
        with patch.object(sys, "argv", ["render_ner_html.py", "--input", "in.jsonl", "--output", "out.html"]):
            args = parse_args()
        self.assertEqual(args.score_fields, ",".join(DEFAULT_SCORE_FIELDS))


if __name__ == "__main__":
    unittest.main()
