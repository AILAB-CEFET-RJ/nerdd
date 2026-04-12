import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mine_train_oof_errors import _build_error_row, _extract_error_tags


class MineTrainOofErrorsTests(unittest.TestCase):
    def test_extract_error_tags_marks_boundary_truncation(self):
        text = "Rua Armenia sao goncalo"
        gold = [{"start": 0, "end": 12, "label": "Location"}]
        pred = [{"start": 4, "end": 12, "label": "Location"}]
        tags = _extract_error_tags(text, gold, pred)
        self.assertIn("boundary_truncation", tags)

    def test_build_error_row_returns_none_on_exact_match(self):
        sample = {"text": "Mesquita", "sample_id": "a"}
        spans = [{"start": 0, "end": 8, "label": "Location"}]
        self.assertIsNone(_build_error_row(1, sample, spans, spans))


if __name__ == "__main__":
    unittest.main()
