import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.retrieve_similar_train_annotation_cases import _feature_dict, _mean_feature_vector, _similarity


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


if __name__ == "__main__":
    unittest.main()
