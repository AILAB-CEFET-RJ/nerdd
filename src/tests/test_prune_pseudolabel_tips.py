import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prune_pseudolabel_tips import prune_rows


class PrunePseudolabelTipsTests(unittest.TestCase):
    def test_prunes_by_score_and_cap(self):
        rows = [
            {
                "relato": "abc",
                "entities": [
                    {"label": "Organization", "start": 0, "end": 3, "score_context_boosted": 0.9},
                    {"label": "Organization", "start": 4, "end": 7, "score_context_boosted": 0.8},
                    {"label": "Person", "start": 8, "end": 11, "score_context_boosted": 0.2},
                ],
            }
        ]

        cleaned_rows, summary = prune_rows(
            rows,
            min_entity_score=0.5,
            max_entities_per_tip=1,
            score_fields=["score_context_boosted"],
            allowed_labels=set(),
            drop_tips_over_max=False,
            drop_empty_tips=False,
        )

        self.assertEqual(len(cleaned_rows), 1)
        self.assertEqual(len(cleaned_rows[0]["entities"]), 1)
        self.assertEqual(cleaned_rows[0]["entities"][0]["score_context_boosted"], 0.9)
        self.assertEqual(summary["dropped_by_score"], 1)
        self.assertEqual(summary["pruned_by_cap"], 1)

    def test_drop_empty_tips_and_filter_labels(self):
        rows = [
            {
                "relato": "a",
                "entities": [
                    {"label": "Organization", "start": 0, "end": 1, "score_context_boosted": 0.7},
                ],
            },
            {
                "relato": "b",
                "entities": [
                    {"label": "Person", "start": 0, "end": 1, "score_context_boosted": 0.8},
                ],
            },
        ]

        cleaned_rows, summary = prune_rows(
            rows,
            min_entity_score=0.0,
            max_entities_per_tip=0,
            score_fields=["score_context_boosted"],
            allowed_labels={"Person"},
            drop_tips_over_max=False,
            drop_empty_tips=True,
        )

        self.assertEqual(len(cleaned_rows), 1)
        self.assertEqual(cleaned_rows[0]["entities"][0]["label"], "Person")
        self.assertEqual(summary["dropped_by_label"], 1)
        self.assertEqual(summary["dropped_empty_tips"], 1)

    def test_drop_tip_over_max(self):
        rows = [
            {
                "relato": "a",
                "entities": [
                    {"label": "Person", "start": 0, "end": 1, "score_context_boosted": 0.9},
                    {"label": "Person", "start": 2, "end": 3, "score_context_boosted": 0.8},
                ],
            }
        ]

        cleaned_rows, summary = prune_rows(
            rows,
            min_entity_score=0.0,
            max_entities_per_tip=1,
            score_fields=["score_context_boosted"],
            allowed_labels=set(),
            drop_tips_over_max=True,
            drop_empty_tips=True,
        )

        self.assertEqual(cleaned_rows, [])
        self.assertEqual(summary["dropped_tip_over_cap"], 1)
        self.assertEqual(summary["dropped_empty_tips"], 1)


if __name__ == "__main__":
    unittest.main()
