import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_location_only_pseudolabels import audit_candidates


def _entity(text, mention, label, score):
    start = text.index(mention)
    return {
        "start": start,
        "end": start + len(mention),
        "text": mention,
        "label": label,
        "score_calibrated": score,
    }


class AuditLocationOnlyPseudolabelsTests(unittest.TestCase):
    def test_audit_flags_credible_omitted_non_target_by_source_id(self):
        text = "Rua Alfa perto de Joao"
        candidates = [
            {
                "source_id": "row-1",
                "text": text,
                "entities": [_entity(text, "Rua Alfa", "Location", 0.98)],
                "_pseudolabel": {"record_score_location": 0.97},
            }
        ]
        predictions = [
            {
                "source_id": "row-1",
                "text": text,
                "entities": [
                    _entity(text, "Rua Alfa", "Location", 0.98),
                    _entity(text, "Joao", "Person", 0.75),
                ],
            }
        ]

        rows, review, summary = audit_candidates(
            candidates,
            predictions,
            target_label="Location",
            omitted_labels={"Person", "Organization"},
            score_fields=["score_calibrated", "score"],
            credible_score_threshold=0.6,
            id_fields=["source_id"],
        )

        self.assertEqual(rows[0]["match_status"], "source_id")
        self.assertEqual(rows[0]["risk_level"], "credible_omitted_non_target")
        self.assertEqual(rows[0]["credible_omitted_labels"], "Person")
        self.assertEqual(summary["rows_with_credible_omitted_non_target"], 1)
        self.assertEqual(len(review), 1)

    def test_audit_uses_normalized_text_and_keeps_low_score_risk_separate(self):
        candidates = [{"text": "Tráfico  na Praça Alfa", "entities": []}]
        prediction_text = "trafico na praca alfa"
        predictions = [
            {
                "text": prediction_text,
                "entities": [_entity(prediction_text, "trafico", "Organization", 0.3)],
            }
        ]

        rows, _review, summary = audit_candidates(
            candidates,
            predictions,
            target_label="Location",
            omitted_labels={"Organization"},
            score_fields=["score_calibrated"],
            credible_score_threshold=0.6,
            id_fields=["source_id"],
        )

        self.assertEqual(rows[0]["match_status"], "normalized_text")
        self.assertEqual(rows[0]["risk_level"], "only_low_score_non_target")
        self.assertEqual(summary["rows_with_low_score_non_target"], 1)


if __name__ == "__main__":
    unittest.main()
