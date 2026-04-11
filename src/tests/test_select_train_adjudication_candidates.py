import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.select_train_annotation_cases import compute_trainability_score, row_passes_filters


class SelectTrainAdjudicationCandidatesTests(unittest.TestCase):
    class Args:
        max_text_length = 900
        min_seed_entities = 1
        max_seed_entities = 4
        max_union_entities = 8
        max_gliner2_noise_proxy = 0.6
        min_agreement_ratio = 0.15
        max_agreement_ratio = 0.8
        require_agreed_or_baseline_seed = True
        penalize_generic_seeds = True

    def test_filters_out_rows_without_stable_seed_origin(self):
        row = {
            "text": "texto simples",
            "metadata": {
                "agreement_ratio": 0.4,
                "gliner2_noise_proxy": 0.2,
                "entity_count_agreed": 0,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 1,
            },
            "review_seed_entities": [{"text": "centro", "label": "Location", "seed_origin": "gliner2_location_metadata_match"}],
        }
        ok, reasons = row_passes_filters(row, self.Args())
        self.assertFalse(ok)
        self.assertIn("no_stable_seed_origin", reasons)

    def test_prefers_clearer_more_stable_rows(self):
        strong = {
            "text": "Rua A em Niteroi, Drogaria Pacheco e Joao.",
            "record_score": 0.95,
            "candidate_quality_score": 0.92,
            "metadata": {
                "agreement_ratio": 0.45,
                "gliner2_noise_proxy": 0.15,
                "baseline_coverage_proxy": 0.7,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 1,
            },
            "review_seed_entities": [
                {"text": "Niteroi", "label": "Location", "seed_origin": "agreed_exact"},
                {"text": "Drogaria Pacheco", "label": "Organization", "seed_origin": "baseline_high_score"},
                {"text": "Joao", "label": "Person", "seed_origin": "agreed_exact"},
            ],
        }
        weak = {
            "text": "homens na rua",
            "record_score": 0.88,
            "candidate_quality_score": 0.8,
            "metadata": {
                "agreement_ratio": 0.18,
                "gliner2_noise_proxy": 0.55,
                "baseline_coverage_proxy": 0.2,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 2,
                "entity_count_gliner2_only": 2,
            },
            "review_seed_entities": [
                {"text": "homens", "label": "Person", "seed_origin": "baseline_high_score"},
            ],
        }
        strong_score, _, _ = compute_trainability_score(strong, self.Args())
        weak_score, _, _ = compute_trainability_score(weak, self.Args())
        self.assertGreater(strong_score, weak_score)


if __name__ == "__main__":
    unittest.main()
