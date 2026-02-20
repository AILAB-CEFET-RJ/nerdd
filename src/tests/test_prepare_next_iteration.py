import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.prepare_next_iteration_pipeline import _load_records, process_records


class PrepareNextIterationTests(unittest.TestCase):
    def test_process_records_projection_and_required(self):
        rows = [
            {"assunto": "A", "relato": "texto 1", "bairroLocal": "Centro"},
            {"assunto": "B", "relato": "   ", "bairroLocal": "Tijuca"},
        ]
        out, counters = process_records(
            rows,
            keep_fields=["assunto", "relato", "bairroLocal"],
            required_fields=["relato"],
            fill_missing_with="",
            coerce_non_string="stringify",
            drop_empty_relato=False,
            deduplicate_by=[],
            source_path="test.jsonl",
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(counters["input_rows"], 2)
        self.assertEqual(counters["dropped_missing_required"], 1)

    def test_process_records_deduplicate(self):
        rows = [
            {"assunto": "A", "relato": "texto 1"},
            {"assunto": "A", "relato": "texto 1"},
        ]
        out, counters = process_records(
            rows,
            keep_fields=["assunto", "relato"],
            required_fields=["relato"],
            fill_missing_with="",
            coerce_non_string="stringify",
            drop_empty_relato=False,
            deduplicate_by=["assunto", "relato"],
            source_path="test.jsonl",
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(counters["dropped_duplicate"], 1)

    def test_process_records_coerce_error(self):
        rows = [{"assunto": {"x": 1}, "relato": "ok"}]
        with self.assertRaises(ValueError):
            process_records(
                rows,
                keep_fields=["assunto", "relato"],
                required_fields=["relato"],
                fill_missing_with="",
                coerce_non_string="error",
                drop_empty_relato=False,
                deduplicate_by=[],
                source_path="test.jsonl",
            )

    def test_load_records_jsonl_error_with_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.jsonl"
            p.write_text('{"a":1}\n{"b":\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _load_records(str(p), allow_json=False)
            self.assertIn("line 2", str(ctx.exception))

    def test_load_records_allow_json_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "list.json"
            p.write_text(json.dumps([{"a": 1}, {"b": 2}]), encoding="utf-8")
            rows = _load_records(str(p), allow_json=True)
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
