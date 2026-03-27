import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rank_pseudolabel_candidates import rank_candidates


class RankPseudolabelCandidatesTests(unittest.TestCase):
    def test_clean_candidate_ranks_above_dense_noisy_candidate(self):
        rows = [
            {
                "relato": "Ocorrencia na Rua Igoa com Joao e Mercado Central.",
                "record_score": 0.92,
                "entities": [
                    {"start": 15, "end": 24, "label": "Location", "score_context_boosted": 0.94},
                    {"start": 29, "end": 33, "label": "Person", "score_context_boosted": 0.88},
                    {"start": 36, "end": 51, "label": "Organization", "score_context_boosted": 0.87},
                ],
            },
            {
                "relato": "A B C D E F G H I J K L M N O P Q",
                "record_score": 0.91,
                "entities": [
                    {"start": i, "end": i + 1, "label": "Organization", "score_context_boosted": 0.61}
                    for i in range(0, 20, 2)
                ],
            },
        ]

        ranked, counters = rank_candidates(
            rows,
            record_score_fields=["record_score"],
            entity_score_fields=["score_context_boosted"],
            label_field="label",
            min_record_score=0.0,
            max_entities=0,
            max_short_span_ratio=1.0,
            short_span_max_chars=3,
            high_entity_score_threshold=0.8,
            low_entity_score_threshold=0.6,
            required_labels=set(),
        )

        self.assertEqual(counters["rows_kept"], 2)
        self.assertEqual(ranked[0]["relato"], rows[0]["relato"])
        self.assertGreater(
            ranked[0]["_candidate_rank"]["candidate_quality_score"],
            ranked[1]["_candidate_rank"]["candidate_quality_score"],
        )

    def test_filters_by_required_label_and_max_entities(self):
        rows = [
            {
                "relato": "Joao foi visto no local.",
                "record_score": 0.8,
                "entities": [{"start": 0, "end": 4, "label": "Person", "score_context_boosted": 0.91}],
            },
            {
                "relato": "Rua Alfa com duas referencias.",
                "record_score": 0.82,
                "entities": [
                    {"start": 0, "end": 8, "label": "Location", "score_context_boosted": 0.88},
                    {"start": 13, "end": 17, "label": "Location", "score_context_boosted": 0.87},
                    {"start": 18, "end": 28, "label": "Person", "score_context_boosted": 0.86},
                ],
            },
        ]

        ranked, counters = rank_candidates(
            rows,
            record_score_fields=["record_score"],
            entity_score_fields=["score_context_boosted"],
            label_field="label",
            min_record_score=0.0,
            max_entities=2,
            max_short_span_ratio=1.0,
            short_span_max_chars=3,
            high_entity_score_threshold=0.8,
            low_entity_score_threshold=0.6,
            required_labels={"Location"},
        )

        self.assertEqual(counters["rows_kept"], 0)
        self.assertEqual(counters["dropped_required_labels"], 1)
        self.assertEqual(counters["dropped_entity_count"], 1)


if __name__ == "__main__":
    unittest.main()
