import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.trim_span_whitespace import transform_rows


class TrimSpanWhitespaceTests(unittest.TestCase):
    def test_trims_span_boundaries_and_preserves_other_fields(self):
        rows = [
            {
                "text": "Rua Alpha fica aqui",
                "assunto": "Roubos",
                "spans": [{"start": 0, "end": 10, "label": "Location", "text": "Rua Alpha "}],
            }
        ]

        transformed, summary = transform_rows(rows)

        self.assertEqual(transformed[0]["assunto"], "Roubos")
        self.assertEqual(transformed[0]["spans"][0]["start"], 0)
        self.assertEqual(transformed[0]["spans"][0]["end"], 9)
        self.assertEqual(transformed[0]["spans"][0]["text"], "Rua Alpha")
        self.assertEqual(summary["trimmed_spans"], 1)
        self.assertEqual(summary["rows_changed"], 1)

    def test_drops_span_that_becomes_empty_after_trim(self):
        rows = [{"text": "   ", "spans": [{"start": 0, "end": 3, "label": "Location"}]}]

        transformed, summary = transform_rows(rows)

        self.assertEqual(transformed[0]["spans"], [])
        self.assertEqual(summary["dropped_empty_after_trim"], 1)


if __name__ == "__main__":
    unittest.main()
