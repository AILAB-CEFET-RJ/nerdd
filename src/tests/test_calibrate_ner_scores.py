import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.calibrate_ner_scores import (
    build_prediction_calibration_rows,
    reliability_rows,
    threshold_rows,
)


class CalibrateNerScoresTests(unittest.TestCase):
    def test_build_prediction_rows_classifies_outcomes(self):
        rows = [
            {
                "row_index_1based": 1,
                "text": "7 BPM Rio",
                "gold_spans": [
                    {"start": 0, "end": 5, "label": "Organization"},
                    {"start": 6, "end": 9, "label": "Location"},
                ],
                "pred_spans": [
                    {"start": 0, "end": 5, "label": "Organization", "score": 0.9},
                    {"start": 2, "end": 5, "label": "Organization", "score": 0.8},
                    {"start": 6, "end": 9, "label": "Organization", "score": 0.7},
                    {"start": 0, "end": 5, "label": "Person", "score": 0.6},
                    {"start": 6, "end": 8, "label": "Location", "score": 0.5},
                ],
            }
        ]

        calibration_rows, support = build_prediction_calibration_rows(
            rows,
            {"Organization", "Location", "Person"},
        )

        self.assertEqual(support, {"Organization": 1, "Location": 1})
        self.assertEqual([row["outcome"] for row in calibration_rows], [
            "exact",
            "boundary_mismatch",
            "label_confusion",
            "label_confusion",
            "boundary_mismatch",
        ])
        self.assertEqual([row["target"] for row in calibration_rows], [1, 0, 0, 0, 0])

    def test_reliability_rows_bins_scores_by_label_and_overall(self):
        rows = [
            {"label": "Organization", "score": 0.1, "target": 0},
            {"label": "Organization", "score": 0.8, "target": 1},
            {"label": "Person", "score": 0.9, "target": 0},
        ]

        bins = reliability_rows(rows, ["Organization", "Person"], bins=2)
        all_high = [row for row in bins if row["label"] == "ALL" and row["bin_index"] == 1][0]

        self.assertEqual(all_high["count"], 2)
        self.assertEqual(all_high["correct"], 1)
        self.assertAlmostEqual(all_high["precision"], 0.5)
        self.assertAlmostEqual(all_high["score_mean"], 0.85)

    def test_threshold_rows_computes_precision_recall_f1(self):
        rows = [
            {"label": "Organization", "score": 0.95, "target": 1},
            {"label": "Organization", "score": 0.90, "target": 0},
            {"label": "Organization", "score": 0.70, "target": 1},
        ]

        report = threshold_rows(
            rows,
            ["Organization"],
            [0.8],
            {"Organization": 3},
        )
        org = [row for row in report if row["label"] == "Organization"][0]

        self.assertEqual(org["predicted"], 2)
        self.assertEqual(org["correct"], 1)
        self.assertEqual(org["false_positive"], 1)
        self.assertAlmostEqual(org["precision"], 0.5)
        self.assertAlmostEqual(org["recall"], 1 / 3)
        self.assertAlmostEqual(org["f1"], 0.4)


if __name__ == "__main__":
    unittest.main()
