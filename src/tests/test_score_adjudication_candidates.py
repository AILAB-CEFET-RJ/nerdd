import sys
import tempfile
import unittest
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.score_adjudication_candidates import _build_train_location_inventory, compute_adjudication_priority


class ScoreAdjudicationCandidatesTests(unittest.TestCase):
    def test_prefers_midband_domain_aligned_case_over_trivial_high_confidence_case(self):
        useful = {
            "text": "Trafico de drogas na Rua Alpha, bairro Centro, Mesquita.",
            "record_score": 0.58,
            "metadata": {
                "agreement_ratio": 0.42,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 1,
            },
            "review_seed_entities": [
                {"text": "Rua Alpha", "label": "Location", "seed_origin": "agreed_exact"},
                {"text": "Centro", "label": "Location", "seed_origin": "baseline_high_score"},
            ],
        }
        trivial = {
            "text": "Rua Alpha Mesquita.",
            "record_score": 0.99,
            "metadata": {
                "agreement_ratio": 0.96,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 0,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "Rua Alpha", "label": "Location", "seed_origin": "agreed_exact"},
                {"text": "Mesquita", "label": "Location", "seed_origin": "agreed_exact"},
            ],
        }
        useful_score, useful_components, _, useful_reasons, _ = compute_adjudication_priority(
            useful, person_only_short_text_max_length=80
        )
        trivial_score, trivial_components, _, _, _ = compute_adjudication_priority(
            trivial, person_only_short_text_max_length=80
        )
        self.assertGreater(useful_score, trivial_score)
        self.assertGreater(useful_components["record_score_midband_score"], trivial_components["record_score_midband_score"])
        self.assertIn("domain_aligned", useful_reasons)

    def test_penalizes_list_like_person_dump(self):
        row = {
            "text": "Ivete Sangalo, Xuxa Meneguel, Anita thainara carvalho, são taradas",
            "record_score": 0.71,
            "metadata": {
                "agreement_ratio": 0.35,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 1,
            },
            "review_seed_entities": [
                {"text": "Ivete Sangalo", "label": "Person", "seed_origin": "agreed_exact"},
                {"text": "Xuxa Meneguel", "label": "Person", "seed_origin": "agreed_exact"},
                {"text": "Anita thainara carvalho", "label": "Person", "seed_origin": "baseline_high_score"},
            ],
        }
        score, _, penalties, reasons, _ = compute_adjudication_priority(row, person_only_short_text_max_length=80)
        self.assertLess(score, 0.0)
        self.assertGreater(penalties["list_like_person_penalty"], 0.0)
        self.assertIn("list_like_person_penalty", reasons)

    def test_penalizes_dense_location_directory_style_case(self):
        compact = {
            "text": "Colocaram barricada rua comendador Luis de matos, engenheiro Belford, são João de Meriti.",
            "record_score": 0.62,
            "metadata": {
                "agreement_ratio": 0.31,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "engenheiro Belford", "label": "Location", "seed_origin": "agreed_exact"},
                {"text": "são João de Meriti", "label": "Location", "seed_origin": "baseline_high_score"},
            ],
        }
        dense = {
            "text": "Na rua da mata na Vila verde/travessa do canal no final do valão/Terreirão rua 1 / macega rua 2 /cachopa/dioneia",
            "record_score": 0.60,
            "metadata": {
                "agreement_ratio": 0.28,
                "entity_count_agreed": 3,
                "entity_count_baseline_only": 2,
                "entity_count_gliner2_only": 1,
            },
            "review_seed_entities": [
                {"text": "Vila verde", "label": "Location", "seed_origin": "agreed_exact"},
                {"text": "final do valão", "label": "Location", "seed_origin": "baseline_high_score"},
                {"text": "macega", "label": "Location", "seed_origin": "baseline_high_score"},
                {"text": "cachopa", "label": "Location", "seed_origin": "baseline_high_score"},
                {"text": "dioneia", "label": "Location", "seed_origin": "baseline_high_score"},
            ],
        }
        compact_score, _, compact_penalties, compact_reasons, _ = compute_adjudication_priority(
            compact, person_only_short_text_max_length=80
        )
        dense_score, _, dense_penalties, dense_reasons, _ = compute_adjudication_priority(
            dense, person_only_short_text_max_length=80
        )
        self.assertGreater(compact_score, dense_score)
        self.assertEqual(compact_penalties["separator_density_penalty"], 0.0)
        self.assertGreater(dense_penalties["separator_density_penalty"], 0.0)
        self.assertIn("location_expansion_risk_penalty", dense_reasons)
        self.assertIn("small_fix_profile", compact_reasons)

    def test_prefers_canonical_small_address_fix_over_mixed_label_case(self):
        canonical = {
            "text": "Carro roubado rua ramalho Monteiro boassu são Gonçalo",
            "record_score": 1e-12,
            "metadata": {
                "agreement_ratio": 0.22,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 2,
                "entity_count_gliner2_only": 1,
            },
            "review_seed_entities": [
                {"text": "boassu", "label": "Location", "seed_origin": "agreed_exact"},
                {"text": "são Gonçalo", "label": "Location", "seed_origin": "baseline_high_score"},
                {"text": "Gonçalo", "label": "Location", "seed_origin": "gliner2_location_metadata_match"},
            ],
        }
        mixed = {
            "text": "Mateus Lima facção comando vermelho morro caixa da água Juscelino",
            "record_score": 0.4,
            "metadata": {
                "agreement_ratio": 0.35,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 2,
                "entity_count_gliner2_only": 1,
            },
            "review_seed_entities": [
                {"text": "Mateus Lima", "label": "Person", "seed_origin": "agreed_exact"},
                {"text": "comando vermelho", "label": "Organization", "seed_origin": "baseline_high_score"},
                {"text": "morro caixa da água", "label": "Location", "seed_origin": "baseline_high_score"},
                {"text": "Juscelino", "label": "Location", "seed_origin": "baseline_high_score"},
            ],
        }
        canonical_score, canonical_components, canonical_penalties, canonical_reasons, _ = compute_adjudication_priority(
            canonical, person_only_short_text_max_length=80
        )
        mixed_score, _, mixed_penalties, _, _ = compute_adjudication_priority(
            mixed, person_only_short_text_max_length=80
        )
        self.assertGreater(canonical_score, mixed_score)
        self.assertGreater(canonical_components["canonical_address_score"], 0.0)
        self.assertIn("canonical_address_profile", canonical_reasons)
        self.assertEqual(canonical_penalties["mixed_label_risk_penalty"], 0.0)
        self.assertGreater(mixed_penalties["mixed_label_risk_penalty"], 0.0)

    def test_compact_mixed_location_org_case_avoids_mixed_label_penalty(self):
        compact_mixed = {
            "text": "Milícia de cabucu toma queimados",
            "record_score": 1e-13,
            "metadata": {
                "agreement_ratio": 0.24,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 2,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "cabucu", "label": "Location", "seed_origin": "agreed_exact"},
                {"text": "toma queimados", "label": "Location", "seed_origin": "baseline_high_score"},
                {"text": "Milícia", "label": "Organization", "seed_origin": "baseline_high_score"},
            ],
        }
        _, _, penalties, _, _ = compute_adjudication_priority(compact_mixed, person_only_short_text_max_length=80)
        self.assertEqual(penalties["mixed_label_risk_penalty"], 0.0)

    def test_novelty_features_reward_unseen_location_seeds(self):
        row = {
            "text": "Barricadas na comunidade Az de Ouro em Olinda",
            "record_score": 0.6,
            "metadata": {
                "agreement_ratio": 0.35,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "comunidade Az de Ouro", "label": "Location", "seed_origin": "baseline_high_score"},
                {"text": "Olinda", "label": "Location", "seed_origin": "agreed_exact"},
            ],
        }
        novelty_context = {
            "location_texts": {"mesquita", "olinda"},
            "location_frequencies": {"mesquita": 3, "olinda": 1},
        }
        pool_context = {
            "location_frequencies": {"comunidade az de ouro": 4, "olinda": 5},
        }

        score, components, _, reasons, _ = compute_adjudication_priority(
            row,
            person_only_short_text_max_length=80,
            novelty_context=novelty_context,
            pool_context=pool_context,
        )

        self.assertGreater(score, 0.0)
        self.assertAlmostEqual(components["toponym_novelty_ratio"], 0.5, places=6)
        self.assertGreater(components["toponym_rarity_score"], 0.0)
        if score >= 0.9:
            self.assertGreater(components["novelty_adjusted_priority_score"], score)
        else:
            self.assertEqual(components["novelty_adjusted_priority_score"], score)
        self.assertIn("novel_toponyms", reasons)
        self.assertGreater(components["pool_toponym_frequency_score"], 0.0)
        self.assertEqual(components["exoticity_penalty"], 0.0)
        if score >= 0.9:
            self.assertGreaterEqual(components["novelty_pool_adjusted_priority_score"], score)
        else:
            self.assertEqual(components["novelty_pool_adjusted_priority_score"], score)
        self.assertIn("recurring_pool_toponyms", reasons)

    def test_novelty_features_default_to_empty_without_context(self):
        row = {
            "text": "Barricadas na comunidade Az de Ouro em Olinda",
            "record_score": 0.5,
            "metadata": {
                "agreement_ratio": 0.4,
                "entity_count_agreed": 1,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "comunidade Az de Ouro", "label": "Location", "seed_origin": "baseline_high_score"},
            ],
        }

        _, components, _, reasons, _ = compute_adjudication_priority(
            row,
            person_only_short_text_max_length=80,
        )

        self.assertNotIn("toponym_novelty_ratio", components)
        self.assertNotIn("novel_toponyms", reasons)

    def test_exoticity_penalty_applies_when_novel_but_not_recurrent_in_pool(self):
        row = {
            "text": "Barricadas na localidade Boca da Paris",
            "record_score": 0.6,
            "metadata": {
                "agreement_ratio": 0.35,
                "entity_count_agreed": 2,
                "entity_count_baseline_only": 1,
                "entity_count_gliner2_only": 0,
            },
            "review_seed_entities": [
                {"text": "Boca da Paris", "label": "Location", "seed_origin": "baseline_high_score"},
                {"text": "Paris", "label": "Location", "seed_origin": "gliner2_location_metadata_match"},
            ],
        }
        novelty_context = {
            "location_texts": {"mesquita"},
            "location_frequencies": {"mesquita": 3},
        }
        pool_context = {
            "location_frequencies": {"boca da paris": 0, "paris": 0},
        }

        score, components, _, reasons, _ = compute_adjudication_priority(
            row,
            person_only_short_text_max_length=80,
            novelty_context=novelty_context,
            pool_context=pool_context,
        )

        self.assertGreaterEqual(components["toponym_novelty_ratio"], 0.5)
        self.assertEqual(components["pool_toponym_frequency_score"], 0.0)
        self.assertEqual(components["exoticity_penalty"], 0.03)
        if score >= 0.9:
            self.assertLess(components["novelty_pool_adjusted_priority_score"], score + (0.03 * components["novelty_score"]))
        else:
            self.assertEqual(components["novelty_pool_adjusted_priority_score"], score)
        self.assertIn("exoticity_penalty", reasons)

    def test_train_inventory_recovers_location_text_from_offsets(self):
        rows = [
            {
                "text": "Tiros na rua Alpha em Mesquita",
                "spans": [
                    {"start": 9, "end": 19, "label": "Location"},
                    {"start": 22, "end": 30, "label": "Location"},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            inventory = _build_train_location_inventory(str(path))

        self.assertIn("rua alpha", inventory["location_texts"])
        self.assertIn("mesquita", inventory["location_texts"])
        self.assertEqual(inventory["location_frequencies"]["rua alpha"], 1)
        self.assertEqual(inventory["location_frequencies"]["mesquita"], 1)


if __name__ == "__main__":
    unittest.main()
