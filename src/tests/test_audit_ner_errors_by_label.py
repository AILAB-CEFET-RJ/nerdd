import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_ner_errors_by_label import build_label_error_report


class AuditNerErrorsByLabelTests(unittest.TestCase):
    def test_detects_missed_and_spurious_target_label(self):
        rows = [
            {
                "row_index_1based": 1,
                "text": "SEAP e loja",
                "gold_spans": [{"start": 0, "end": 4, "label": "Organization"}],
                "pred_spans": [{"start": 7, "end": 11, "label": "Organization"}],
            }
        ]

        errors, review_rows = build_label_error_report(rows, "Organization")

        self.assertEqual(len(errors), 2)
        self.assertEqual([item["direction"] for item in errors], ["FN", "FP"])
        self.assertEqual(errors[0]["error_type"], "missed")
        self.assertEqual(errors[1]["error_type"], "spurious")
        self.assertEqual(review_rows[0]["row_index_1based"], 1)

    def test_detects_label_confusion_for_target_label(self):
        rows = [
            {
                "row_index_1based": 1,
                "text": "7 BPM",
                "gold_spans": [{"start": 0, "end": 5, "label": "Organization"}],
                "pred_spans": [{"start": 0, "end": 5, "label": "Location"}],
            }
        ]

        errors, _ = build_label_error_report(rows, "Organization")

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["direction"], "FN")
        self.assertEqual(errors[0]["error_type"], "label_confusion")
        self.assertEqual(errors[0]["overlapping_spans"][0]["label"], "Location")

    def test_detects_boundary_mismatch_for_target_label(self):
        rows = [
            {
                "row_index_1based": 1,
                "text": "27 BPM",
                "gold_spans": [{"start": 0, "end": 6, "label": "Organization"}],
                "pred_spans": [{"start": 3, "end": 6, "label": "Organization"}],
            }
        ]

        errors, _ = build_label_error_report(rows, "Organization")

        self.assertEqual(len(errors), 2)
        self.assertTrue(all(item["error_type"] == "boundary_mismatch" for item in errors))

    def test_exact_match_is_not_an_error(self):
        rows = [
            {
                "row_index_1based": 1,
                "text": "BOPE",
                "gold_spans": [{"start": 0, "end": 4, "label": "Organization"}],
                "pred_spans": [{"start": 0, "end": 4, "label": "Organization"}],
            }
        ]

        errors, review_rows = build_label_error_report(rows, "Organization")

        self.assertEqual(errors, [])
        self.assertEqual(review_rows, [])


if __name__ == "__main__":
    unittest.main()
