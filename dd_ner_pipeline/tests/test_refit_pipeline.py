import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.refit_pipeline import (
    extract_text,
    normalize_entities,
    prepare_training_records,
    split_train_val,
)


class RefitPipelineTests(unittest.TestCase):
    def test_extract_text_priority(self):
        record = {"relato": "abc", "text": "xyz"}
        text, source = extract_text(record, text_keys=("text", "relato"))
        self.assertEqual(text, "xyz")
        self.assertEqual(source, "text")

    def test_normalize_entities_filters_invalid(self):
        record = {
            "entities": [
                {"start": 0, "end": 3, "label": "Person"},
                {"start": "x", "end": 4, "label": "Person"},
                {"start": 2, "end": 1, "label": "Person"},
                {"start": 0, "end": 4, "label": ""},
                {"start": 0, "end": 4, "label": "Misc"},
            ]
        }
        entities, source, counters = normalize_entities(record, allowed_labels={"Person"})
        self.assertEqual(source, "entities")
        self.assertEqual(len(entities), 1)
        self.assertEqual(counters["invalid_span"], 2)
        self.assertEqual(counters["invalid_label"], 1)
        self.assertEqual(counters["filtered_by_label"], 1)

    def test_prepare_training_records(self):
        rows = [
            {"relato": "abc", "entities": [{"start": 0, "end": 3, "label": "Person"}]},
            {"relato": "", "entities": [{"start": 0, "end": 1, "label": "Person"}]},
            {"relato": "x", "entities": []},
        ]
        prepared, counters = prepare_training_records(rows, allowed_labels={"Person"})
        self.assertEqual(len(prepared), 1)
        self.assertEqual(counters["input_records"], 3)
        self.assertEqual(counters["missing_text"], 1)
        self.assertEqual(counters["dropped_no_entities"], 1)
        self.assertEqual(counters["kept_records"], 1)

    def test_split_train_val(self):
        records = [{"id": i} for i in range(10)]
        train, val = split_train_val(records, val_ratio=0.2, seed=42)
        self.assertEqual(len(train) + len(val), 10)
        self.assertGreaterEqual(len(val), 1)
        train2, val2 = split_train_val(records, val_ratio=0.2, seed=42)
        self.assertEqual(train, train2)
        self.assertEqual(val, val2)


if __name__ == "__main__":
    unittest.main()
