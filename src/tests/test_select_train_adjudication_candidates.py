import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.select_train_annotation_cases import compute_trainability_score, row_passes_filters


class SelectTrainAdjudicationCandidatesTests(unittest.TestCase):
    class Args:
        min_text_length = 20
        max_text_length = 900
        min_seed_entities = 1
        max_seed_entities = 4
        max_union_entities = 8
        max_baseline_entities = 12
        max_gliner2_noise_proxy = 0.6
        max_person_seed_ratio = 0.6
        min_agreement_ratio = 0.15
        max_agreement_ratio = 0.8
        require_agreed_or_baseline_seed = True
        penalize_generic_seeds = True
        drop_list_like_person_dumps = True
        drop_person_only_short_texts = True
        person_only_short_text_max_length = 80
        require_location_seed = True
        require_domain_context = True

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

    def test_filters_list_like_person_dump(self):
        row = {
            "text": "Ivete Sangalo, Xuxa Meneguel, Anita thainara carvalho, são taradas",
            "metadata": {
                "agreement_ratio": 0.4,
                "gliner2_noise_proxy": 0.2,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "Ivete Sangalo", "label": "Person", "seed_origin": "agreed_exact"},
                {"text": "Xuxa Meneguel", "label": "Person", "seed_origin": "agreed_exact"},
                {"text": "Anita thainara carvalho", "label": "Person", "seed_origin": "baseline_high_score"},
            ],
        }
        ok, reasons = row_passes_filters(row, self.Args())
        self.assertFalse(ok)
        self.assertIn("list_like_person_dump", reasons)

    def test_filters_person_only_short_text(self):
        row = {
            "text": "Lucas de aparecida Gonçalves",
            "metadata": {
                "agreement_ratio": 0.4,
                "gliner2_noise_proxy": 0.2,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 0,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "Lucas de aparecida Gonçalves", "label": "Person", "seed_origin": "agreed_exact"},
            ],
        }
        ok, reasons = row_passes_filters(row, self.Args())
        self.assertFalse(ok)
        self.assertIn("person_only_short_text", reasons)

    def test_filters_rows_without_location_seed(self):
        row = {
            "text": "Trafico com Joao e Mateus na regiao",
            "metadata": {
                "agreement_ratio": 0.4,
                "gliner2_noise_proxy": 0.2,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "Joao", "label": "Person", "seed_origin": "agreed_exact"},
            ],
        }
        ok, reasons = row_passes_filters(row, self.Args())
        self.assertFalse(ok)
        self.assertIn("missing_location_seed", reasons)

    def test_filters_rows_without_domain_context(self):
        row = {
            "text": "Centro Mesquita",
            "metadata": {
                "agreement_ratio": 0.4,
                "gliner2_noise_proxy": 0.2,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 0,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "Mesquita", "label": "Location", "seed_origin": "agreed_exact"},
            ],
        }
        ok, reasons = row_passes_filters(row, self.Args())
        self.assertFalse(ok)
        self.assertIn("missing_domain_context", reasons)

    def test_filters_overdense_baseline_case(self):
        row = {
            "text": "Tráfico em muitas ruas do bairro",
            "metadata": {
                "agreement_ratio": 0.4,
                "gliner2_noise_proxy": 0.2,
                "entity_count_baseline": 18,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 4,
                "entity_count_gliner2_only": 1,
            },
            "review_seed_entities": [
                {"text": "bairro alfa", "label": "Location", "seed_origin": "agreed_exact"},
                {"text": "rua beta", "label": "Location", "seed_origin": "baseline_high_score"},
            ],
        }
        ok, reasons = row_passes_filters(row, self.Args())
        self.assertFalse(ok)
        self.assertIn("too_many_baseline_entities", reasons)

    def test_filters_person_heavy_seed_set(self):
        row = {
            "text": "Trafico com Joao, Pedro e Carlos na favela Coreia",
            "metadata": {
                "agreement_ratio": 0.4,
                "gliner2_noise_proxy": 0.2,
                "entity_count_baseline": 5,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "Joao", "label": "Person", "seed_origin": "agreed_exact"},
                {"text": "Pedro", "label": "Person", "seed_origin": "agreed_exact"},
                {"text": "Carlos", "label": "Person", "seed_origin": "baseline_high_score"},
                {"text": "Coreia", "label": "Location", "seed_origin": "agreed_exact"},
            ],
        }
        ok, reasons = row_passes_filters(row, self.Args())
        self.assertFalse(ok)
        self.assertIn("person_seed_ratio_too_high", reasons)


if __name__ == "__main__":
    unittest.main()
