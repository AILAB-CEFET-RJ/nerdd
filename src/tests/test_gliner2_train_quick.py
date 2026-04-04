import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gliner2_training.train_quick import (
    _convert_entity_rows_to_spans,
    _convert_rows_to_gliner2_jsonl,
    _merge_training_rows,
    _normalize_label,
)


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

    def test_convert_entity_rows_to_spans_converts_adjudicated_entities(self):
        rows = [
            {
                "text": "Ivete Sangalo em Salvador",
                "entities": [
                    {"text": "Ivete Sangalo", "label": "Person", "start": 0, "end": 13},
                    {"text": "Salvador", "label": "Location", "start": 17, "end": 25},
                ],
            }
        ]
        converted = _convert_entity_rows_to_spans(rows)
        self.assertEqual(
            converted,
            [
                {
                    "text": "Ivete Sangalo em Salvador",
                    "spans": [
                        {"start": 0, "end": 13, "label": "Person"},
                        {"start": 17, "end": 25, "label": "Location"},
                    ],
                }
            ],
        )

    def test_merge_training_rows_prefers_supervised_on_duplicate_text(self):
        supervised = [{"text": "A", "spans": [{"start": 0, "end": 1, "label": "Person"}]}]
        pseudolabels = [
            {"text": "A", "spans": [{"start": 0, "end": 1, "label": "Location"}]},
            {"text": "B", "spans": [{"start": 0, "end": 1, "label": "Organization"}]},
        ]
        merged = _merge_training_rows(
            supervised,
            pseudolabels,
            train_mode="supervised_plus_pseudolabels",
            deduplicate_by_text=True,
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["spans"][0]["label"], "Person")
        self.assertEqual(merged[1]["text"], "B")

if __name__ == "__main__":
    unittest.main()
