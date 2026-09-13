import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.filter_pseudolabels_by_completeness import build_completeness_gate


class FilterPseudolabelsByCompletenessTests(unittest.TestCase):
    def test_keeps_only_unambiguous_rows_without_label_specific_risks(self):
        selected = [{"text": "safe"}, {"text": "organization risk"}, {"text": "ambiguous"}]
        records = [
            {"selected_index_1based": "1", "match_status": "source_id"},
            {"selected_index_1based": "2", "match_status": "normalized_text"},
            {"selected_index_1based": "3", "match_status": "ambiguous_text"},
        ]
        entities = [
            {
                "selected_index_1based": "2",
                "full_label": "Organization",
                "full_mention": "CV",
                "score": "0.91",
                "status": "omitted_exact",
            },
            {
                "selected_index_1based": "1",
                "full_label": "Location",
                "full_mention": "Rua Alfa",
                "score": "0.94",
                "status": "omitted_exact",
            },
        ]

        kept, decisions, summary = build_completeness_gate(
            selected,
            records,
            entities,
            label_min_scores={"Location": 0.95, "Organization": 0.90},
            allowed_match_statuses={"source_id", "normalized_text"},
        )

        self.assertEqual([row["text"] for row in kept], ["safe"])
        self.assertEqual(summary["decision_counts"], {"ineligible_match": 1, "kept": 1, "risky_omission": 1})
        self.assertEqual(summary["risky_entities_by_label"], {"Organization": 1})
        self.assertEqual(decisions[1]["risky_omitted_labels"], "Organization")
        self.assertEqual(kept[0]["_pseudolabel_completeness_gate"]["match_status"], "source_id")

    def test_excludes_rows_without_audit_records(self):
        kept, decisions, summary = build_completeness_gate(
            [{"text": "not audited"}],
            [],
            [],
            label_min_scores={"Person": 0.8},
            allowed_match_statuses={"source_id"},
        )

        self.assertEqual(kept, [])
        self.assertEqual(decisions[0]["decision"], "missing_audit_record")
        self.assertEqual(summary["decision_counts"], {"missing_audit_record": 1})
