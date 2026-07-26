import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.serve_ner_annotation_editor import save_dataset, validate_and_clean_records


class ServeNerAnnotationEditorTests(unittest.TestCase):
    def test_validate_removes_editor_index_when_not_original(self):
        payload = {
            "records": [
                {
                    "text": "Rua A",
                    "spans": [{"start": 0, "end": 5, "label": "Location"}],
                    "_editor_row_index": 0,
                }
            ]
        }

        records = validate_and_clean_records(
            payload,
            expected_records=1,
            original_has_editor_index=[False],
        )

        self.assertNotIn("_editor_row_index", records[0])

    def test_validate_preserves_original_editor_index(self):
        payload = {
            "records": [
                {
                    "text": "Rua A",
                    "spans": [{"start": 0, "end": 5, "label": "Location"}],
                    "_editor_row_index": 10,
                }
            ]
        }

        records = validate_and_clean_records(
            payload,
            expected_records=1,
            original_has_editor_index=[True],
        )

        self.assertEqual(records[0]["_editor_row_index"], 10)

    def test_validate_rejects_invalid_offsets(self):
        payload = {
            "records": [
                {
                    "text": "Rua A",
                    "spans": [{"start": 0, "end": 99, "label": "Location"}],
                }
            ]
        }

        with self.assertRaises(ValueError):
            validate_and_clean_records(
                payload,
                expected_records=1,
                original_has_editor_index=[False],
            )

    def test_save_dataset_creates_backup_and_replaces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            path.write_text('[{"text": "old", "spans": []}]\n', encoding="utf-8")

            backup = save_dataset(path, [{"text": "new", "spans": []}])

            self.assertTrue(backup.exists())
            self.assertEqual(json.loads(backup.read_text(encoding="utf-8"))[0]["text"], "old")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))[0]["text"], "new")


if __name__ == "__main__":
    unittest.main()
