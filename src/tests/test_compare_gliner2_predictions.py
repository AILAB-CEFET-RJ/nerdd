import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_gliner2_predictions import build_summary, map_gliner2_labels


class CompareGLiNER2PredictionsTests(unittest.TestCase):
    def test_map_gliner2_labels_normalizes_common_labels(self):
        entities = [
            {"text": "Ivete Sangalo", "label": "person", "start": 0, "end": 14},
            {"text": "TV Globo", "label": "organization", "start": 20, "end": 28},
        ]
        mapped = map_gliner2_labels(entities)
        self.assertEqual({item["label"] for item in mapped}, {"Person", "Organization"})

    def test_build_summary_counts_baseline_base_and_adapter(self):
        rows = [
            {
                "baseline_entities": [{"label": "Person"}, {"label": "Location"}],
                "gliner2_base_entities": [{"label": "Person"}],
                "gliner2_adapter_entities": [{"label": "Person"}, {"label": "Organization"}],
            }
        ]
        summary = build_summary(rows)
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["baseline_total_spans"], 2)
        self.assertEqual(summary["gliner2_base_total_spans"], 1)
        self.assertEqual(summary["gliner2_adapter_total_spans"], 2)


if __name__ == "__main__":
    unittest.main()
