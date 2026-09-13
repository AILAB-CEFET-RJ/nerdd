import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_pseudolabel_completeness import audit_completeness


def entity(text, mention, label, score):
    start = text.index(mention)
    return {
        "start": start,
        "end": start + len(mention),
        "text": mention,
        "label": label,
        "score_calibrated": score,
    }


class AuditPseudolabelCompletenessTests(unittest.TestCase):
    def test_classifies_retained_and_credible_omitted_entities_of_any_label(self):
        text = "Rua Alfa com Joao do CV"
        selected = [{"source_id": "one", "text": text, "entities": [entity(text, "Joao", "Person", 0.95)]}]
        predictions = [
            {
                "source_id": "one",
                "text": text,
                "entities": [
                    entity(text, "Joao", "Person", 0.95),
                    entity(text, "Rua Alfa", "Location", 0.96),
                    entity(text, "CV", "Organization", 0.88),
                ],
            }
        ]

        entity_rows, record_rows, review_rows, summary = audit_completeness(
            selected,
            predictions,
            score_fields=["score_calibrated"],
            credible_score_threshold=0.9,
            id_fields=["source_id"],
        )

        self.assertEqual(summary["matched_rows"], 1)
        self.assertEqual(summary["entity_status_counts"]["retained_exact"], 1)
        self.assertEqual(summary["entity_status_counts"]["omitted_exact"], 2)
        self.assertEqual(record_rows[0]["credible_omitted_count"], 1)
        self.assertEqual(record_rows[0]["credible_omitted_labels"], "Location")
        self.assertEqual(len(review_rows), 1)
        self.assertEqual({row["full_label"] for row in entity_rows if row["credible_omission"]}, {"Location"})

    def test_marks_boundary_overlap_as_not_retained(self):
        text = "Rua Alfa"
        selected = [{"text": text, "entities": [{"start": 4, "end": 8, "label": "Location"}]}]
        predictions = [{"text": text, "entities": [entity(text, "Rua Alfa", "Location", 0.95)]}]

        entity_rows, record_rows, _review_rows, summary = audit_completeness(
            selected,
            predictions,
            score_fields=["score_calibrated"],
            credible_score_threshold=0.9,
            id_fields=["source_id"],
        )

        self.assertEqual(entity_rows[0]["status"], "overlap_not_retained_same_label")
        self.assertEqual(record_rows[0]["risk_level"], "credible_omission")
        self.assertEqual(summary["entity_status_counts"]["overlap_not_retained_same_label"], 1)

    def test_matches_equivalent_duplicate_prediction_texts(self):
        text = "Rua Alfa"
        selected = [{"text": text, "entities": [entity(text, "Rua Alfa", "Location", 0.95)]}]
        predictions = [
            {"text": text, "entities": [entity(text, "Rua Alfa", "Location", 0.95)]},
            {"text": text, "entities": [entity(text, "Rua Alfa", "Location", 0.95)]},
        ]

        _entity_rows, record_rows, _review_rows, summary = audit_completeness(
            selected,
            predictions,
            score_fields=["score_calibrated"],
            credible_score_threshold=0.9,
            id_fields=["source_id"],
        )

        self.assertEqual(record_rows[0]["match_status"], "normalized_text_equivalent_duplicate")
        self.assertEqual(summary["matched_rows"], 1)
