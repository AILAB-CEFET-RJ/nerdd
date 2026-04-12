import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.summarize_oof_error_cases import _labels_in_row, _row_matches, _summarize


class SummarizeOofErrorCasesTests(unittest.TestCase):
    def test_row_matches_filters_by_tag_and_label(self):
        row = {
            "error_tags": ["has_fn", "boundary_truncation"],
            "gold_spans": [{"label": "Location"}],
            "pred_spans": [{"label": "Location"}],
        }
        self.assertTrue(_row_matches(row, {"has_fn"}, {"Location"}))
        self.assertFalse(_row_matches(row, {"has_fp"}, {"Location"}))
        self.assertFalse(_row_matches(row, {"has_fn"}, {"Organization"}))

    def test_summarize_counts_tags_and_labels(self):
        rows = [
            {
                "sample_id": "a",
                "text": "Rua Armenia",
                "error_tags": ["has_fn", "boundary_truncation"],
                "gold_spans": [{"label": "Location"}],
                "pred_spans": [{"label": "Location"}],
            },
            {
                "sample_id": "b",
                "text": "CV chatuba",
                "error_tags": ["has_fp"],
                "gold_spans": [{"label": "Organization"}],
                "pred_spans": [{"label": "Organization"}],
            },
        ]
        summary = _summarize(rows, sample_limit=5)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["error_tag_counts"]["has_fn"], 1)
        self.assertEqual(summary["label_counts"]["Location"], 1)
        self.assertEqual(summary["label_counts"]["Organization"], 1)
        self.assertIn("has_fn::Location", summary["tag_label_counts"])


if __name__ == "__main__":
    unittest.main()
