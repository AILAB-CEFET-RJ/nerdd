import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl


class IoUtilsTests(unittest.TestCase):
    def test_load_jsonl_accepts_json_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.json"
            path.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
            self.assertEqual(load_jsonl(path), [{"a": 1}, {"a": 2}])

    def test_load_jsonl_accepts_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
            self.assertEqual(load_jsonl(path), [{"a": 1}, {"a": 2}])

    def test_load_jsonl_accepts_single_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "row.json"
            path.write_text(json.dumps({"a": 1}), encoding="utf-8")
            self.assertEqual(load_jsonl(path), [{"a": 1}])


if __name__ == "__main__":
    unittest.main()
