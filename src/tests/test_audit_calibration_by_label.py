import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_calibration_by_label import load_rows, summarize_rows


class AuditCalibrationByLabelTests(unittest.TestCase):
    def test_summarize_rows_separates_positive_and_negative_by_label(self):
        rows = [
            {"label": "Organization", "target": 1, "score_raw": 0.9, "score_calibrated": 0.8},
            {"label": "Organization", "target": 0, "score_raw": 0.85, "score_calibrated": 0.6},
            {"label": "Person", "target": 0, "score_raw": 0.4, "score_calibrated": 0.3},
            {"label": "Person", "target": 1, "score_raw": 0.7, "score_calibrated": 0.75},
        ]

        summary = summarize_rows(rows, high_score_threshold=0.8)

        self.assertEqual(summary["rows_total"], 4)
        self.assertEqual(summary["per_label"]["Organization"]["raw"]["negatives"]["count"], 1)
        self.assertEqual(summary["per_label"]["Organization"]["raw"]["negatives"]["share_ge_threshold"], 1.0)
        self.assertEqual(summary["per_label"]["Organization"]["calibrated"]["negatives"]["share_ge_threshold"], 0.0)

    def test_load_rows_without_calibrator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "calibration.csv"
            csv_path.write_text(
                "RowId,Label,Entidade_Referencia,Entidade_Predita,Start,End,Score,Validacao\n"
                "1,Organization,,abc,0,3,0.9,0\n"
                "1,Person,abc,abc,0,3,0.8,1\n",
                encoding="utf-8",
            )
            rows = load_rows(
                str(csv_path),
                score_col="Score",
                label_col="Validacao",
                class_col="Label",
                allowed_labels={"Organization", "Person"},
                calibrator=None,
            )

            self.assertEqual(len(rows), 2)
            self.assertIsNone(rows[0]["score_calibrated"])
            self.assertEqual({row["label"] for row in rows}, {"Organization", "Person"})


if __name__ == "__main__":
    unittest.main()
