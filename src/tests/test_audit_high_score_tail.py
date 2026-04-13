import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_high_score_tail import summarize_rows


class AuditHighScoreTailTests(unittest.TestCase):
    def test_summarize_rows_reports_overconfidence_gap(self):
        rows = [
            {"label": "Location", "target": 1, "score_raw": 0.95, "score_calibrated": 0.85},
            {"label": "Location", "target": 0, "score_raw": 0.92, "score_calibrated": 0.75},
            {"label": "Organization", "target": 0, "score_raw": 0.81, "score_calibrated": 0.60},
            {"label": "Person", "target": 1, "score_raw": 0.70, "score_calibrated": 0.68},
        ]

        summary = summarize_rows(rows, thresholds=[0.8, 0.9])

        overall_08_raw = summary["overall"]["raw"][0]
        self.assertEqual(overall_08_raw["count"], 3)
        self.assertAlmostEqual(overall_08_raw["score_mean"], (0.95 + 0.92 + 0.81) / 3)
        self.assertAlmostEqual(overall_08_raw["accuracy"], 1 / 3)
        self.assertGreater(overall_08_raw["overconfidence_gap"], 0.0)

        overall_09_cal = summary["overall"]["calibrated"][1]
        self.assertEqual(overall_09_cal["count"], 0)
        self.assertIsNone(overall_09_cal["overconfidence_gap"])

        location_09_raw = summary["per_label"]["Location"]["raw"][1]
        self.assertEqual(location_09_raw["count"], 2)
        self.assertAlmostEqual(location_09_raw["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
