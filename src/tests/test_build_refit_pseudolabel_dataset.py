import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_refit_pseudolabel_dataset import build_refit_pseudolabel_dataset


class BuildRefitPseudolabelDatasetTests(unittest.TestCase):
    def test_top_n_limits_projected_rows_before_conversion(self):
        rows = [
            {
                "source_id": "tip-1",
                "text": "Ivete Sangalo em Salvador",
                "adjudication": {
                    "decision": "accept",
                    "review_confidence": "high",
                    "entities_final": [
                        {"text": "Ivete Sangalo", "label": "Person", "start": 0, "end": 13},
                        {"text": "Salvador", "label": "Location", "start": 17, "end": 25},
                    ],
                },
                "model": "gpt-5",
            },
            {
                "source_id": "tip-2",
                "text": "Rua A em Nilópolis",
                "adjudication": {
                    "decision": "accept_with_edits",
                    "review_confidence": "medium",
                    "entities_final": [
                        {"text": "Rua A", "label": "Location", "start": 0, "end": 5},
                        {"text": "Nilópolis", "label": "Location", "start": 9, "end": 18},
                    ],
                },
                "model": "gpt-5",
            },
            {
                "source_id": "tip-3",
                "text": "Texto rejeitado",
                "adjudication": {
                    "decision": "reject",
                    "review_confidence": "low",
                    "entities_final": [],
                },
                "model": "gpt-5",
            },
        ]

        emitted, summary = build_refit_pseudolabel_dataset(
            rows,
            allowed_decisions={"accept", "accept_with_edits"},
            allowed_labels={"Person", "Location", "Organization"},
            include_source_payload=False,
            top_n=2,
        )

        self.assertEqual([row["source_id"] for row in emitted], ["tip-1", "tip-2"])
        self.assertEqual(summary["records_total"], 3)
        self.assertEqual(summary["records_selected"], 2)
        self.assertEqual(summary["records_emitted"], 2)
        self.assertEqual(summary["top_n"], 2)
        self.assertEqual(summary["summary"]["selected_input_records"], 2)
        self.assertEqual(summary["summary"]["kept_records"], 2)
        self.assertEqual(summary["summary"]["label_counts"]["Location"], 3)
        self.assertEqual(summary["summary"]["label_counts"]["Person"], 1)

    def test_top_n_zero_keeps_all_rows(self):
        rows = [
            {
                "source_id": "tip-1",
                "text": "Rua A em Nilópolis",
                "adjudication": {
                    "decision": "accept",
                    "review_confidence": "high",
                    "entities_final": [
                        {"text": "Rua A", "label": "Location", "start": 0, "end": 5},
                        {"text": "Nilópolis", "label": "Location", "start": 9, "end": 18},
                    ],
                },
                "model": "gpt-5",
            }
        ]

        emitted, summary = build_refit_pseudolabel_dataset(
            rows,
            allowed_decisions={"accept", "accept_with_edits"},
            allowed_labels={"Person", "Location", "Organization"},
            include_source_payload=False,
            top_n=0,
        )

        self.assertEqual(len(emitted), 1)
        self.assertEqual(summary["records_selected"], 1)
        self.assertEqual(summary["top_n"], 0)

    def test_top_n_negative_raises(self):
        with self.assertRaises(ValueError):
            build_refit_pseudolabel_dataset(
                [],
                allowed_decisions={"accept", "accept_with_edits"},
                allowed_labels={"Person", "Location", "Organization"},
                include_source_payload=False,
                top_n=-1,
            )


if __name__ == "__main__":
    unittest.main()
