import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_llm_adjudication_input import (
    build_review_seed_entities,
    match_entities,
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
            {"text": "Rua Piauí", "label": "Location", "text_norm": "rua piaui", "ner_score": 0.91},
            {"text": "P2", "label": "Organization", "text_norm": "p2", "ner_score": 0.55},
        ]
        seeded = build_review_seed_entities(
            agreed_entities=agreed,
            baseline_only_entities=baseline_only,
            baseline_seed_score_threshold=0.80,
        )
        self.assertEqual([item["text"] for item in seeded], ["Rua Piauí", "Ivete Sangalo"])


if __name__ == "__main__":
    unittest.main()
