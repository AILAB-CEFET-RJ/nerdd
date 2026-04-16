import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.select_top_utility_candidates import _sort_rows


class SelectTopUtilityCandidatesTests(unittest.TestCase):
    def test_sort_rows_uses_requested_field_and_assigns_rank(self):
        rows = [
            {"source_id": "a", "adjudication_priority_score": 0.91, "novelty_pool_adjusted_priority_score": 0.93, "review_seed_entities": [{"label": "Location", "text": "A"}], "text": "A"},
            {"source_id": "b", "adjudication_priority_score": 0.95, "novelty_pool_adjusted_priority_score": 0.94, "review_seed_entities": [{"label": "Location", "text": "B"}], "text": "BBBB"},
            {"source_id": "c", "adjudication_priority_score": 0.90, "novelty_pool_adjusted_priority_score": 0.97, "review_seed_entities": [{"label": "Location", "text": "C"}], "text": "CC"},
        ]
        ranked = _sort_rows(rows, "novelty_pool_adjusted_priority_score", float("-inf"))
        self.assertEqual([row["source_id"] for row in ranked], ["c", "b", "a"])
        self.assertEqual(ranked[0]["_utility_selection"]["rank"], 1)
        self.assertEqual(ranked[1]["_utility_selection"]["ranking_field"], "novelty_pool_adjusted_priority_score")

    def test_sort_rows_respects_min_score(self):
        rows = [
            {"source_id": "a", "adjudication_priority_score": 0.91, "text": "A"},
            {"source_id": "b", "adjudication_priority_score": 0.89, "text": "B"},
        ]
        ranked = _sort_rows(rows, "adjudication_priority_score", 0.9)
        self.assertEqual([row["source_id"] for row in ranked], ["a"])


if __name__ == "__main__":
    unittest.main()
