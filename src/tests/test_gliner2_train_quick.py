import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gliner2_training.train_quick import _convert_rows_to_gliner2_jsonl, _normalize_label


class GLiNER2TrainQuickTests(unittest.TestCase):
    def test_normalize_label_maps_project_labels(self):
        self.assertEqual(_normalize_label("Person"), "person")
        self.assertEqual(_normalize_label("Location"), "location")
        self.assertEqual(_normalize_label("Organization"), "organization")

    def test_convert_rows_to_gliner2_jsonl_converts_spans_to_entities(self):
        rows = [
            {
                "text": "Ivete Sangalo em Salvador",
                "spans": [
                    {"start": 0, "end": 13, "label": "Person"},
                    {"start": 17, "end": 25, "label": "Location"},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "train.jsonl"
            _convert_rows_to_gliner2_jsonl(rows, target)
            content = target.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(content), 1)
            payload = json.loads(content[0])
            self.assertEqual(payload["output"]["entities"]["person"], ["Ivete Sangalo"])
            self.assertEqual(payload["output"]["entities"]["location"], ["Salvador"])


if __name__ == "__main__":
    unittest.main()
