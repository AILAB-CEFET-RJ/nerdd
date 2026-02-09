import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.config import ContextBoostConfig
from pseudolabelling.context_boost import apply_context_boost_to_record, normalize_text


class ContextBoostTests(unittest.TestCase):
    def _base_config(self):
        return ContextBoostConfig(
            write_trace_fields=True,
            write_legacy_fields=False,
            fallback_score_fields=["score_ts", "score_iso"],
        )

    def test_normalize_text_removes_accents_and_case(self):
        self.assertEqual(normalize_text("TijucÁ"), "tijuca")
        self.assertEqual(normalize_text("  Rua   José  "), "rua jose")

    def test_all_entities_boost_when_metadata_matches_text(self):
        config = self._base_config()
        config.boost_scope = "all-entities"
        record = {
            "relato": "Tráfico na rua Jose Marques, no bairro Centro.",
            "bairroLocal": "Centro",
            "logradouroLocal": "Rua José Marques",
            "entities": [
                {"text": "Jose Marques", "label": "Person", "score": 0.5},
                {"text": "Centro", "label": "Location", "score": 0.6},
            ],
        }
        boosted, stats = apply_context_boost_to_record(record, config)
        self.assertEqual(stats["boosted_entities"], 2)
        self.assertAlmostEqual(boosted["entities"][0]["score_context_boosted"], 0.6, places=6)
        self.assertAlmostEqual(boosted["entities"][1]["score_context_boosted"], 0.72, places=6)

    def test_location_only_scope(self):
        config = self._base_config()
        config.boost_scope = "location-only"
        record = {
            "relato": "Acontece em Centro",
            "bairroLocal": "Centro",
            "entities": [
                {"text": "Centro", "label": "Location", "score": 0.5},
                {"text": "Joao", "label": "Person", "score": 0.5},
            ],
        }
        boosted, stats = apply_context_boost_to_record(record, config)
        self.assertEqual(stats["boosted_entities"], 1)
        self.assertAlmostEqual(boosted["entities"][0]["score_context_boosted"], 0.6, places=6)
        self.assertAlmostEqual(boosted["entities"][1]["score_context_boosted"], 0.5, places=6)

    def test_matched_only_scope(self):
        config = self._base_config()
        config.boost_scope = "matched-only"
        record = {
            "relato": "Acontece em Centro e Tijuca",
            "bairroLocal": "Centro",
            "entities": [
                {"text": "Centro", "label": "Location", "score": 0.5},
                {"text": "Tijuca", "label": "Location", "score": 0.5},
            ],
        }
        boosted, stats = apply_context_boost_to_record(record, config)
        self.assertEqual(stats["boosted_entities"], 1)
        self.assertAlmostEqual(boosted["entities"][0]["score_context_boosted"], 0.6, places=6)
        self.assertAlmostEqual(boosted["entities"][1]["score_context_boosted"], 0.5, places=6)

    def test_score_fallback_precedence(self):
        config = self._base_config()
        config.base_score_field = "score"
        config.boost_scope = "all-entities"
        record = {
            "relato": "Centro",
            "bairroLocal": "Centro",
            "entities": [
                {"text": "Centro", "label": "Location", "score_ts": 0.7},
            ],
        }
        boosted, _stats = apply_context_boost_to_record(record, config)
        self.assertAlmostEqual(boosted["entities"][0]["score_context_boosted"], 0.84, places=6)
        self.assertEqual(boosted["entities"][0]["_context_boost_score_source"], "score_ts")


if __name__ == "__main__":
    unittest.main()
