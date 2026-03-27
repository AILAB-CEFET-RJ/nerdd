import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_gliner_predictions import _parse_csv, build_summary


class CompareGlinerPredictionsTests(unittest.TestCase):
    def test_parse_csv(self):
        self.assertEqual(_parse_csv("Person,Location,Organization"), ["Person", "Location", "Organization"])
        self.assertEqual(_parse_csv(""), [])

    def test_build_summary_counts_labels(self):
        rows = [
            {
                "baseline_entities": [{"label": "Person"}, {"label": "Location"}],
                "model_entities": [{"label": "Person"}],
            },
            {
                "baseline_entities": [{"label": "Organization"}],
                "model_entities": [{"label": "Organization"}, {"label": "Location"}],
            },
        ]
        summary = build_summary(rows)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["baseline_total_spans"], 3)
        self.assertEqual(summary["model_total_spans"], 3)
        self.assertEqual(summary["baseline_label_counts"]["Person"], 1)
        self.assertEqual(summary["model_label_counts"]["Location"], 1)


if __name__ == "__main__":
    unittest.main()
