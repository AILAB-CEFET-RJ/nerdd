import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.reannotate_legacy_calibration import build_revision, clean_spans, synchronize_identical_text_rows


class ReannotateLegacyCalibrationTests(unittest.TestCase):
    def test_revision_removes_currently_disallowed_generic_and_platform_spans(self):
        text = "a polícia usa Gmail e a prefeitura falou com o tio"
        def span(mention, label):
            start = text.index(mention)
            return {"start": start, "end": start + len(mention), "label": label}

        rows = [
            {
                "text": text,
                "spans": [
                    span("polícia", "Organization"),
                    span("Gmail", "Organization"),
                    span("prefeitura", "Organization"),
                    span("tio", "Person"),
                ],
            }
        ]

        revised, audit = build_revision(rows)

        self.assertEqual(revised[0]["spans"], [])
        self.assertEqual(len(audit), 1)

    def test_clean_spans_trims_outer_punctuation_and_removes_nested_span(self):
        record = {
            "text": "[rua Alfa]",
            "spans": [
                {"start": 0, "end": 10, "label": "Location"},
                {"start": 1, "end": 9, "label": "Location"},
            ],
        }

        cleaned = clean_spans(record)

        self.assertEqual(
            cleaned,
            [{"start": 1, "end": 9, "label": "Location"}],
        )

    def test_identical_reports_receive_the_same_revised_spans(self):
        rows = [
            {"text": "Rua Alfa", "spans": [{"start": 0, "end": 8, "label": "Location"}]},
            {"text": "Rua Alfa", "spans": []},
        ]

        synchronize_identical_text_rows(rows)

        self.assertEqual(rows[0]["spans"], rows[1]["spans"])
        self.assertEqual(rows[1]["spans"][0]["label"], "Location")


if __name__ == "__main__":
    unittest.main()
