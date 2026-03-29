import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.test_gliner_checkpoint import _load_texts


class _Args:
    def __init__(self, *, text=None, file=None):
        self.text = text or []
        self.file = file


class TestTestGlinerCheckpoint(unittest.TestCase):
    def test_load_texts_merges_cli_and_file_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "samples.txt"
            input_path.write_text("Ivete Sangalo\n\nXuxa Meneguel\n", encoding="utf-8")
            args = _Args(text=["Anitta"], file=str(input_path))
            self.assertEqual(_load_texts(args), ["Anitta", "Ivete Sangalo", "Xuxa Meneguel"])

    def test_load_texts_rejects_missing_inputs(self):
        with self.assertRaises(ValueError):
            _load_texts(_Args())


if __name__ == "__main__":
    unittest.main()
