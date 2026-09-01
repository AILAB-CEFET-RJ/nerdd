import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rank_pseudolabel_candidates import rank_rows


class RankPseudolabelCandidatesTests(unittest.TestCase):
    def test_rank_rows_orders_by_requested_score(self):
        rows = [
            {"id": "a", "record_score_location": 0.80, "entities": [{"label": "Location"}], "text": "aaaa"},
            {"id": "b", "record_score_location": 0.95, "entities": [{"label": "Location"}], "text": "bb"},
            {"id": "c", "record_score_location": 0.90, "entities": [{"label": "Location"}], "text": "c"},
        ]
        ranked, counters = rank_rows(
            rows,
            score_fields=["record_score_location"],
            min_score=0.0,
            required_labels=set(),
            label_field="label",
        )
        self.assertEqual([row["id"] for row in ranked], ["b", "c", "a"])
        self.assertEqual(ranked[0]["_pseudolabel_selection"]["rank"], 1)
        self.assertEqual(ranked[0]["_pseudolabel_selection"]["score_field"], "record_score_location")
        self.assertEqual(counters["rows_after_filters"], 3)

    def test_rank_rows_respects_min_score(self):
        rows = [
            {"id": "a", "record_score_location": 0.79, "entities": [{"label": "Location"}]},
            {"id": "b", "record_score_location": 0.80, "entities": [{"label": "Location"}]},
        ]
        ranked, counters = rank_rows(
            rows,
            score_fields=["record_score_location"],
            min_score=0.80,
            required_labels=set(),
            label_field="label",
        )
        self.assertEqual([row["id"] for row in ranked], ["b"])
        self.assertEqual(counters["dropped_min_score"], 1)

    def test_rank_rows_respects_required_labels(self):
        rows = [
            {"id": "a", "record_score_location": 0.90, "entities": [{"label": "Person"}]},
            {"id": "b", "record_score_location": 0.80, "entities": [{"label": "Location"}]},
        ]
        ranked, counters = rank_rows(
            rows,
            score_fields=["record_score_location"],
            min_score=0.0,
            required_labels={"Location"},
            label_field="label",
        )
        self.assertEqual([row["id"] for row in ranked], ["b"])
        self.assertEqual(counters["dropped_required_labels"], 1)

    def test_rank_rows_uses_fallback_score_fields(self):
        rows = [
            {"id": "a", "record_score": 0.80, "entities": [{"label": "Location"}]},
            {"id": "b", "record_score_location": 0.90, "entities": [{"label": "Location"}]},
        ]
        ranked, _ = rank_rows(
            rows,
            score_fields=["record_score_location", "record_score"],
            min_score=0.0,
            required_labels={"Location"},
            label_field="label",
        )
        self.assertEqual([row["id"] for row in ranked], ["b", "a"])
        self.assertEqual(ranked[1]["_pseudolabel_selection"]["score_field"], "record_score")


if __name__ == "__main__":
    unittest.main()
