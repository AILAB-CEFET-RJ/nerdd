import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.retrieve_similar_train_annotation_cases import (
    _feature_dict,
    _has_degenerate_merged_span,
    _has_obvious_truncation,
    _mean_feature_vector,
    _selector_filter_args,
    _similarity,
)


class RetrieveSimilarTrainAnnotationCasesTests(unittest.TestCase):
    def test_similarity_prefers_structurally_similar_location_case(self):
        positive = {
            "text": "Barricadas em engenheiro Belford, rua major augusto cesar, comendador Luiz de matos",
            "metadata": {
                "agreement_ratio": 0.35,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
                "entity_count_baseline": 2,
            },
            "review_seed_entities": [
                {"text": "engenheiro Belford", "label": "Location"},
                {"text": "comendador Luiz de matos", "label": "Location"},
            ],
            "adjudication_priority_score": 0.82,
        }
        similar = {
            "text": "Colocaram barricada rua comendador Luis de matos, engenheiro Belford, são João de Meriti",
            "metadata": {
                "agreement_ratio": 0.30,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
                "entity_count_baseline": 2,
            },
            "review_seed_entities": [
                {"text": "engenheiro Belford", "label": "Location"},
                {"text": "são João de Meriti", "label": "Location"},
            ],
            "adjudication_priority_score": 0.72,
        }
        dissimilar = {
            "text": "Na rua da mata na Vila verde/travessa do canal no final do valão/Terreirão rua 1 / macega rua 2 /cachopa/dioneia",
            "metadata": {
                "agreement_ratio": 0.28,
                "entity_count_agreed": 3,
                "entity_count_baseline_only": 2,
                "entity_count_gliner2_only": 1,
                "entity_count_baseline": 5,
            },
            "review_seed_entities": [
                {"text": "Vila verde", "label": "Location"},
                {"text": "final do valão", "label": "Location"},
                {"text": "macega", "label": "Location"},
                {"text": "cachopa", "label": "Location"},
                {"text": "dioneia", "label": "Location"},
            ],
            "adjudication_priority_score": 0.32,
        }

        proto = _mean_feature_vector([positive], person_only_short_text_max_length=80)
        similar_score = _similarity(_feature_dict(similar, person_only_short_text_max_length=80), proto)
        dissimilar_score = _similarity(_feature_dict(dissimilar, person_only_short_text_max_length=80), proto)
        self.assertGreater(similar_score, dissimilar_score)

    def test_feature_dict_tracks_multi_token_location_ratio(self):
        row = {
            "text": "Rua Sergipe com rua Amazonas localidade Coreia em Mesquita",
            "metadata": {},
            "review_seed_entities": [
                {"text": "Amazonas", "label": "Location"},
                {"text": "Coreia", "label": "Location"},
                {"text": "Mesquita", "label": "Location"},
                {"text": "Rua Sergipe", "label": "Location"},
            ],
        }
        vec = _feature_dict(row, person_only_short_text_max_length=80)
        self.assertAlmostEqual(vec["multi_token_location_ratio"], 0.25)

    def test_detects_degenerate_merged_span(self):
        seeds = [
            {"text": "caixa da água Mesquita", "label": "Location"},
            {"text": "Mesquita", "label": "Location"},
        ]
        self.assertTrue(_has_degenerate_merged_span(seeds))

    def test_detects_obvious_truncation(self):
        self.assertTrue(_has_obvious_truncation([{"text": "Petropolis", "label": "Location"}]))

    def test_selector_filter_args_enable_hard_gates(self):
        args = _selector_filter_args(
            type(
                "Args",
                (),
                {
                    "min_text_length": 20,
                    "max_text_length": 900,
                    "min_seed_entities": 1,
                    "max_seed_entities": 6,
                    "max_union_entities": 16,
                    "max_baseline_entities": 28,
                    "max_gliner2_noise_proxy": 0.7,
                    "max_person_seed_ratio": 0.9,
                    "min_agreement_ratio": 0.0,
                    "max_agreement_ratio": 0.9,
                    "require_agreed_or_baseline_seed": True,
                    "penalize_generic_seeds": True,
                    "drop_list_like_person_dumps": True,
                    "drop_person_only_short_texts": True,
                    "person_only_short_text_max_length": 80,
                    "require_location_seed": True,
                    "require_domain_context": True,
                },
            )()
        )
        self.assertTrue(args.require_location_seed)
        self.assertTrue(args.drop_list_like_person_dumps)


if __name__ == "__main__":
    unittest.main()
