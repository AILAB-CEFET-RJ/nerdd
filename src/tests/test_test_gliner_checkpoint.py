import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.test_gliner_checkpoint import _load_rows, _resolve_model_path


class _Args:
    def __init__(self, *, text=None, file=None):
        self.text = text or []
        self.file = file


class TestTestGlinerCheckpoint(unittest.TestCase):
    def test_load_rows_merges_cli_and_file_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "samples.txt"
            input_path.write_text("Ivete Sangalo\n\nXuxa Meneguel\n", encoding="utf-8")
            args = _Args(text=["Anitta"], file=str(input_path))
            self.assertEqual(
                _load_rows(args),
                [
                    {"text": "Anitta", "record_score": None},
                    {"text": "Ivete Sangalo", "record_score": None},
                    {"text": "Xuxa Meneguel", "record_score": None},
                ],
            )

    def test_load_rows_reads_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "samples.jsonl"
            input_path.write_text(
                '{"text":"Ivete Sangalo","record_score":0.75}\n{"relato":"Xuxa Meneguel","record_score_context_boosted":0.8}\n{"foo":"bar"}\n',
                encoding="utf-8",
            )
            args = _Args(file=str(input_path))
            self.assertEqual(
                _load_rows(args),
                [
                    {"text": "Ivete Sangalo", "record_score": 0.75},
                    {"text": "Xuxa Meneguel", "record_score": 0.8},
                ],
            )

    def test_load_rows_rejects_missing_inputs(self):
        with self.assertRaises(ValueError):
            _load_rows(_Args())

    def test_resolve_model_path_preserves_hf_repo_ids(self):
        self.assertEqual(_resolve_model_path("birdred/glinerdd"), "birdred/glinerdd")


if __name__ == "__main__":
    unittest.main()
