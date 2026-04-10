import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.expand_location_spans_with_markers import (
    compile_prefix_regex,
    read_json_or_jsonl_with_format,
    transform_rows,
)


class TestExpandLocationSpansWithMarkers(unittest.TestCase):
    def setUp(self):
        self.prefix_re = compile_prefix_regex(
            ["rua", "trav", "trv", "tr", "avenida", "av"],
            ["dr", "dra", "prof"],
        )

    def test_expands_simple_rua_prefix(self):
        text = "Venda de carga roubada na rua judite guerra pavuna pedido urgente."
        row = {
            "text": text,
            "spans": [
                {"start": text.index("judite"), "end": text.index("pavuna") - 1, "label": "Location"},
                {"start": text.index("pavuna"), "end": text.index("pavuna") + len("pavuna"), "label": "Location"},
            ],
        }
        out, summary = transform_rows([row], prefix_re=self.prefix_re)
        span = out[0]["spans"][0]
        self.assertEqual(text[span["start"] : span["end"]], "rua judite guerra")
        self.assertEqual(summary["expanded_spans"], 1)

    def test_expands_abbreviated_trv_with_title(self):
        text = "Tráfico de drogas trv . Francisco Macieira e Trav Dr . Lopes indivíduos."
        row = {
            "text": text,
            "spans": [
                {"start": text.index("Francisco"), "end": text.index("Macieira") + len("Macieira"), "label": "Location"},
                {"start": text.index("Lopes"), "end": text.index("Lopes") + len("Lopes"), "label": "Location"},
            ],
        }
        out, summary = transform_rows([row], prefix_re=self.prefix_re)
        first, second = out[0]["spans"]
        self.assertEqual(text[first["start"] : first["end"]], "trv . Francisco Macieira")
        self.assertEqual(text[second["start"] : second["end"]], "Trav Dr . Lopes")
        self.assertEqual(summary["expanded_spans"], 2)

    def test_leaves_existing_marker_span_unchanged(self):
        text = "Tem um carro parado na tr Miguel pinto em frente ao número 394."
        start = text.index("tr Miguel pinto")
        row = {
            "text": text,
            "spans": [{"start": start, "end": start + len("tr Miguel pinto"), "label": "Location"}],
        }
        out, summary = transform_rows([row], prefix_re=self.prefix_re)
        span = out[0]["spans"][0]
        self.assertEqual(text[span["start"] : span["end"]], "tr Miguel pinto")
        self.assertEqual(summary.get("expanded_spans", 0), 0)

    def test_preserves_json_array_format_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.json"
            path.write_text(json.dumps([{"text": "abc", "spans": []}], ensure_ascii=False), encoding="utf-8")
            rows, fmt = read_json_or_jsonl_with_format(str(path))
            self.assertEqual(fmt, "json")
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
