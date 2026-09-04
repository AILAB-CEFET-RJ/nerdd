import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mine_train_oof_errors import (
    _build_error_row,
    _build_source_row,
    _extract_error_tags,
    _filter_spans_by_score,
)


class MineTrainOofErrorsTests(unittest.TestCase):
    def test_extract_error_tags_marks_boundary_truncation(self):
        text = "Rua Armenia sao goncalo"
        gold = [{"start": 0, "end": 12, "label": "Location"}]
        pred = [{"start": 4, "end": 12, "label": "Location"}]
        tags = _extract_error_tags(text, gold, pred)
        self.assertIn("boundary_truncation", tags)

    def test_build_error_row_returns_none_on_exact_match(self):
        sample = {"text": "Mesquita", "sample_id": "a", "row_index_0based": 0, "row_index_1based": 1}
        spans = [{"start": 0, "end": 8, "label": "Location"}]
        self.assertIsNone(_build_error_row(source_row=sample, text=sample["text"], gold_spans=spans, pred_spans=spans))

    def test_filter_spans_by_score_keeps_missing_scores(self):
        spans = [
            {"start": 0, "end": 1, "label": "Location", "score": 0.3},
            {"start": 2, "end": 3, "label": "Location", "score": 0.7},
            {"start": 4, "end": 5, "label": "Location"},
        ]
        kept = _filter_spans_by_score(spans, 0.6)
        self.assertEqual([span["start"] for span in kept], [2, 4])

    def test_build_source_row_preserves_original_index_and_metadata(self):
        row = {
            "text": "relato",
            "spans": [],
            "assunto": "Teste",
            "_editor_row_index": 7,
        }
        source = _build_source_row(row, 3, {})
        self.assertEqual(source["row_index_0based"], 3)
        self.assertEqual(source["row_index_1based"], 4)
        self.assertEqual(source["sample_id"], "sample_3")
        self.assertEqual(source["source_fields"], {"assunto": "Teste"})
        self.assertEqual(source["metadata_match"]["status"], "unmatched")


if __name__ == "__main__":
    unittest.main()
