import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_adjudication_cases import (
    _extract_location_metadata_terms,
    _matches_location_metadata,
    build_review_seed_entities,
    match_entities,
    normalize_baseline_entities,
    normalize_entity_text,
)


class TestBuildLlmAdjudicationInput(unittest.TestCase):
    def test_normalize_entity_text_removes_case_accents_and_spacing(self):
        self.assertEqual(normalize_entity_text("  São   João Mériti, "), "sao joao meriti")

    def test_match_entities_finds_exact_soft_and_conflict(self):
        baseline = [
            {"text": "São João Meriti", "label": "Location", "text_norm": "sao joao meriti", "ner_score": 0.9},
            {"text": "Rua Sao Pedro", "label": "Location", "text_norm": "rua sao pedro", "ner_score": 0.88},
            {"text": "Carlos", "label": "Person", "text_norm": "carlos", "ner_score": 0.95},
        ]
        gliner2 = [
            {"text": "são joão meriti", "label": "Location", "text_norm": "sao joao meriti"},
            {"text": "sao pedro", "label": "Location", "text_norm": "sao pedro"},
            {"text": "Carlos", "label": "Organization", "text_norm": "carlos"},
        ]
        matched = match_entities(baseline, gliner2, soft_match_min_chars=4)
        self.assertEqual(len(matched["agreed_entities"]), 2)
        self.assertEqual(matched["agreed_entities"][0]["match_type"], "exact")
        self.assertEqual(matched["agreed_entities"][1]["match_type"], "soft")
        self.assertEqual(len(matched["conflicts"]), 1)
        self.assertEqual(len(matched["baseline_only_entities"]), 1)
        self.assertEqual(len(matched["gliner2_only_entities"]), 1)

    def test_build_review_seed_entities_keeps_agreements_and_high_score_baseline_only(self):
        agreed = [
            {
                "match_type": "exact",
                "baseline_entity": {"text": "Ivete Sangalo", "label": "Person", "text_norm": "ivete sangalo", "ner_score": 0.99},
            }
        ]
        baseline_only = [
            {"text": "Trindade", "label": "Location", "text_norm": "trindade", "ner_score": 0.91},
            {"text": "P2", "label": "Organization", "text_norm": "p2", "ner_score": 0.55},
        ]
        seeded = build_review_seed_entities(
            agreed_entities=agreed,
            baseline_only_entities=baseline_only,
            gliner2_only_entities=[],
            location_metadata_terms={"trindade"},
            baseline_seed_score_threshold=0.80,
            gliner2_location_min_chars=5,
        )
        self.assertEqual([item["text"] for item in seeded], ["Trindade", "Ivete Sangalo"])

    def test_extract_location_metadata_terms_and_promote_matching_gliner2_locations(self):
        row = {
            "bairroLocal": "Trindade",
            "cidadeLocal": "Sao Gonçalo",
            "pontodeReferenciaLocal": "Rua Cuiabá esquina com Uruguaiana",
        }
        terms = _extract_location_metadata_terms(row)
        self.assertIn("trindade", terms)
        self.assertIn("sao goncalo", terms)
        gliner2_only = [
            {"text": "Cuiabá", "label": "Location", "text_norm": "cuiaba"},
            {"text": "Trindade", "label": "Location", "text_norm": "trindade"},
            {"text": "polícia", "label": "Organization", "text_norm": "policia"},
        ]
        seeded = build_review_seed_entities(
            agreed_entities=[],
            baseline_only_entities=[],
            gliner2_only_entities=gliner2_only,
            location_metadata_terms=terms,
            baseline_seed_score_threshold=0.80,
            gliner2_location_min_chars=5,
        )
        self.assertEqual([item["text"] for item in seeded], ["Cuiabá", "Trindade"])

    def test_matches_location_metadata_requires_token_boundaries(self):
        terms = {"pavuna", "rua cuiaba esquina com uruguaiana"}
        self.assertTrue(
            _matches_location_metadata(
                {"text": "Cuiabá", "label": "Location", "text_norm": "cuiaba"},
                terms,
                min_chars=5,
            )
        )
        self.assertFalse(
            _matches_location_metadata(
                {"text": "AVUNA", "label": "Location", "text_norm": "avuna"},
                terms,
                min_chars=5,
            )
        )

    def test_build_review_seed_entities_filters_noisy_location_seeds(self):
        seeded = build_review_seed_entities(
            agreed_entities=[],
            baseline_only_entities=[
                {"text": "Gonçalo", "label": "Location", "text_norm": "goncalo", "ner_score": 0.95},
                {"text": "bairro", "label": "Location", "text_norm": "bairro", "ner_score": 0.95},
                {"text": "Rua Piauí", "label": "Location", "text_norm": "rua piaui", "ner_score": 0.95},
                {"text": "Trindade", "label": "Location", "text_norm": "trindade", "ner_score": 0.95},
            ],
            gliner2_only_entities=[
                {"text": "paulo", "label": "Location", "text_norm": "paulo"},
                {"text": "Cuiabá", "label": "Location", "text_norm": "cuiaba"},
                {"text": "da vala", "label": "Location", "text_norm": "da vala"},
            ],
            location_metadata_terms={"trindade", "rua cuiaba esquina com uruguaiana", "rua avenida paulo damasceno"},
            baseline_seed_score_threshold=0.80,
            gliner2_location_min_chars=5,
        )
        self.assertEqual([item["text"] for item in seeded], ["Cuiabá", "Trindade"])

    def test_normalize_baseline_entities_drops_fragmentary_and_generic_noise(self):
        normalized = normalize_baseline_entities(
            [
                {"text": ".", "label": "Location", "score": 0.9},
                {"text": "PROXIMO", "label": "Person", "score": 0.9},
                {"text": "Escadao", "label": "Location", "score": 0.9},
                {"text": "ÃO JOÃO DE MERITI", "label": "Location", "score": 0.99},
                {"text": "Mesquita", "label": "Location", "score": 0.99},
            ],
            {"Person", "Location", "Organization"},
        )
        self.assertEqual([item["text"] for item in normalized], ["Mesquita"])


if __name__ == "__main__":
    unittest.main()
