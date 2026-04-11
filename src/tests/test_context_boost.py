import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.config import ContextBoostConfig
from pseudolabelling.context_boost import (
    _build_boosted_entity_audit_rows,
    apply_context_boost_to_record,
    normalize_text,
    _entity_matches_metadata,
)


class ContextBoostTests(unittest.TestCase):
    def _base_config(self):
        return ContextBoostConfig(
            write_trace_fields=True,
            write_legacy_fields=False,
            fallback_score_fields=["score_calibrated", "score_ts", "score_iso"],
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

    def test_location_matched_only_scope(self):
        config = self._base_config()
        config.boost_scope = "location-matched-only"
        record = {
            "relato": "Acontece em Centro com apoio da Guarda Centro",
            "bairroLocal": "Centro",
            "entities": [
                {"text": "Centro", "label": "Location", "score": 0.5},
                {"text": "Guarda Centro", "label": "Organization", "score": 0.5},
            ],
        }
        boosted, stats = apply_context_boost_to_record(record, config)
        self.assertEqual(stats["boosted_entities"], 1)
        self.assertAlmostEqual(boosted["entities"][0]["score_context_boosted"], 0.6, places=6)
        self.assertAlmostEqual(boosted["entities"][1]["score_context_boosted"], 0.5, places=6)
        self.assertEqual(boosted["entities"][0]["_context_boost_reason"], "location-entity-overlaps-metadata")
        self.assertEqual(boosted["entities"][1]["_context_boost_reason"], "no-boost")

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

    def test_entity_metadata_match_rejects_trivial_substrings(self):
        metadata_values = [("cidadeLocal", "Rio de Janeiro"), ("bairroLocal", "Centro")]
        self.assertFalse(_entity_matches_metadata({"text": "de"}, metadata_values))
        self.assertFalse(_entity_matches_metadata({"text": "."}, metadata_values))
        self.assertFalse(_entity_matches_metadata({"text": "na"}, metadata_values))

    def test_entity_metadata_match_accepts_informative_overlap(self):
        metadata_values = [("logradouroLocal", "Estrada do Galeao"), ("bairroLocal", "Ilha do Governador")]
        self.assertFalse(_entity_matches_metadata({"text": "Galeao", "score_calibrated": 0.9}, metadata_values))
        self.assertTrue(_entity_matches_metadata({"text": "Ilha do Governador", "score_calibrated": 0.9}, metadata_values))
        self.assertTrue(_entity_matches_metadata({"text": "Estrada do Galeao", "score_calibrated": 0.9}, metadata_values))

    def test_entity_metadata_match_requires_minimum_score(self):
        metadata_values = [("logradouroLocal", "Rua Faia")]
        self.assertFalse(_entity_matches_metadata({"text": "Rua Faia", "score_calibrated": 0.2}, metadata_values))
        self.assertTrue(_entity_matches_metadata({"text": "Rua Faia", "score_calibrated": 0.8}, metadata_values))

    def test_entity_metadata_match_requires_stronger_overlap_than_single_generic_token(self):
        metadata_values = [("logradouroLocal", "Rua Faia")]
        self.assertFalse(_entity_matches_metadata({"text": "Rua", "score_calibrated": 0.9}, metadata_values))
        self.assertFalse(_entity_matches_metadata({"text": "Faia", "score_calibrated": 0.9}, metadata_values))
        self.assertTrue(_entity_matches_metadata({"text": "Rua Faia", "score_calibrated": 0.9}, metadata_values))

    def test_boost_uses_only_metadata_matched_in_text(self):
        config = self._base_config()
        config.boost_scope = "location-matched-only"
        record = {
            "relato": "Ocorrencia na Rua Faia.",
            "logradouroLocal": "Rua Faia",
            "cidadeLocal": "Rio de Janeiro",
            "entities": [
                {"text": "Rua Faia", "label": "Location", "score_calibrated": 0.9},
                {"text": "Rio de Janeiro", "label": "Location", "score_calibrated": 0.9},
            ],
        }
        boosted, stats = apply_context_boost_to_record(record, config)
        self.assertEqual(stats["boosted_entities"], 1)
        self.assertAlmostEqual(boosted["entities"][0]["score_context_boosted"], 1.0, places=6)
        self.assertAlmostEqual(boosted["entities"][1]["score_context_boosted"], 0.9, places=6)
        self.assertTrue(boosted["entities"][0]["_context_boost_applied"])
        self.assertFalse(boosted["entities"][1]["_context_boost_applied"])

    def test_build_boosted_entity_audit_rows_keeps_only_boosted_entities(self):
        config = self._base_config()
        config.boost_scope = "location-matched-only"
        record = {
            "sample_id": "tip-1",
            "relato": "Ocorrencia na Rua Faia com apoio da policia.",
            "logradouroLocal": "Rua Faia",
            "entities": [
                {"text": "Rua Faia", "label": "Location", "score_calibrated": 0.9, "start": 14, "end": 22},
                {"text": "policia", "label": "Organization", "score_calibrated": 0.9, "start": 36, "end": 43},
            ],
        }
        boosted, _ = apply_context_boost_to_record(record, config)
        rows = _build_boosted_entity_audit_rows(boosted, row_index=1, config=config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entity_text"], "Rua Faia")
        self.assertEqual(rows[0]["label"], "Location")
        self.assertEqual(rows[0]["boost_reason"], "location-entity-overlaps-metadata")


if __name__ == "__main__":
    unittest.main()
