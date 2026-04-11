import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rank_pseudolabel_candidates import build_candidate, rank_candidates


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
            min_entities=0,
            max_entities=0,
            min_text_length=0,
            max_low_score_share=1.0,
            max_location_ratio=1.0,
            max_short_span_ratio=1.0,
            drop_generic_entity_texts=False,
            drop_list_like_person_dumps=False,
            drop_political_copypasta=False,
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
            min_entities=0,
            max_entities=2,
            min_text_length=0,
            max_low_score_share=1.0,
            max_location_ratio=1.0,
            max_short_span_ratio=1.0,
            drop_generic_entity_texts=False,
            drop_list_like_person_dumps=False,
            drop_political_copypasta=False,
            short_span_max_chars=3,
            high_entity_score_threshold=0.8,
            low_entity_score_threshold=0.6,
            required_labels={"Location"},
        )

        self.assertEqual(counters["rows_kept"], 0)
        self.assertEqual(counters["dropped_required_labels"], 1)
        self.assertEqual(counters["dropped_entity_count"], 1)

    def test_short_location_abbreviation_is_not_counted_as_suspicious_short_span(self):
        row = {
            "relato": "Ocorrencia em RJ e na Rua Alfa.",
            "record_score": 0.8,
            "entities": [
                {"start": 14, "end": 16, "label": "Location", "text": "RJ", "score_context_boosted": 0.91},
                {"start": 22, "end": 30, "label": "Location", "text": "Rua Alfa", "score_context_boosted": 0.92},
            ],
        }

        candidate = build_candidate(
            row,
            row_index=1,
            record_score_fields=["record_score"],
            entity_score_fields=["score_context_boosted"],
            label_field="label",
            short_span_max_chars=3,
            high_entity_score_threshold=0.8,
            low_entity_score_threshold=0.6,
        )

        self.assertEqual(candidate["_candidate_rank"]["short_span_ratio"], 0.0)

    def test_short_location_marker_standalone_remains_suspicious(self):
        row = {
            "relato": "Ocorrencia na tr e depois em RJ.",
            "record_score": 0.8,
            "entities": [
                {"start": 14, "end": 16, "label": "Location", "text": "tr", "score_context_boosted": 0.91},
                {"start": 29, "end": 31, "label": "Location", "text": "RJ", "score_context_boosted": 0.92},
            ],
        }

        candidate = build_candidate(
            row,
            row_index=1,
            record_score_fields=["record_score"],
            entity_score_fields=["score_context_boosted"],
            label_field="label",
            short_span_max_chars=3,
            high_entity_score_threshold=0.8,
            low_entity_score_threshold=0.6,
        )

        self.assertAlmostEqual(candidate["_candidate_rank"]["short_span_ratio"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
