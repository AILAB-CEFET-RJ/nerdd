import argparse
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.optimize_context_boost_factor import (
    _build_boost_record,
    _default_recommendation_entity_threshold,
    choose_recommendation,
    enrich_rows_from_metadata_sources,
    evaluate_boost_factors,
)


def _args(**overrides):
    values = {
        "target_label": "Location",
        "labels": "Person,Location,Organization",
        "base_score_field": "score",
        "boosted_score_field": "score_context_boosted_sim",
        "record_score_aggregation": "p75",
        "text_field_priority": "text,relato",
        "metadata_fields": "logradouroLocal,bairroLocal,cidadeLocal,pontodeReferenciaLocal",
        "boost_scope": "location-matched-only",
        "match_policy": "any-metadata-in-text",
        "location_labels": "Location",
        "dedupe_mode": "label_text",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class OptimizeContextBoostFactorTests(unittest.TestCase):
    def test_build_boost_record_promotes_source_fields_to_top_level(self):
        row = {
            "text": "fato na Rua Alfa",
            "pred_spans": [],
            "gold_spans": [],
            "source_fields": {"logradouroLocal": "Rua Alfa", "assunto": "Teste"},
        }
        record = _build_boost_record(row, base_score_field="score")
        self.assertEqual(record["logradouroLocal"], "Rua Alfa")
        self.assertEqual(record["assunto"], "Teste")
        self.assertEqual(record["entities"], [])

    def test_evaluate_boost_factors_counts_exact_promoted_location(self):
        rows = [
            {
                "row_index_0based": 0,
                "row_index_1based": 1,
                "sample_id": "sample_0",
                "fold": 1,
                "text": "trafico na Rua Alfa",
                "logradouroLocal": "Rua Alfa",
                "gold_spans": [{"start": 11, "end": 19, "label": "Location"}],
                "pred_spans": [
                    {"start": 11, "end": 19, "label": "Location", "text": "Rua Alfa", "score": 0.75}
                ],
            }
        ]
        metric_rows, promoted_entities, promoted_records = evaluate_boost_factors(
            rows,
            args=_args(),
            labels=["Person", "Location", "Organization"],
            boost_factors=[1.0, 1.1],
            entity_thresholds=[0.8],
            record_thresholds=[0.8],
        )
        boosted_row = [row for row in metric_rows if row["boost_factor"] == 1.1][0]
        self.assertEqual(boosted_row["promoted_entity_count"], 1)
        self.assertEqual(boosted_row["promoted_entity_exact_count"], 1)
        self.assertEqual(boosted_row["promoted_record_count"], 1)
        self.assertEqual(len(promoted_entities), 1)
        self.assertEqual(len(promoted_records), 1)

    def test_choose_recommendation_prefers_smallest_factor_on_tie(self):
        rows = [
            {
                "boost_factor": 1.1,
                "entity_threshold": 0.8,
                "record_threshold": 0.8,
                "promoted_entity_count": 2,
                "promoted_entity_exact_count": 2,
                "promoted_entity_precision": 1.0,
            },
            {
                "boost_factor": 1.2,
                "entity_threshold": 0.8,
                "record_threshold": 0.8,
                "promoted_entity_count": 2,
                "promoted_entity_exact_count": 2,
                "promoted_entity_precision": 1.0,
            },
        ]
        rec = choose_recommendation(
            rows,
            boost_factors=[1.0, 1.1, 1.2],
            entity_threshold=0.8,
            record_threshold=0.8,
            precision_floor=0.9,
        )
        self.assertEqual(rec["recommended_boost_factor"], 1.1)

    def test_default_recommendation_threshold_uses_highest_at_most_point_eight(self):
        self.assertEqual(_default_recommendation_entity_threshold([0.6, 0.8, 0.9]), 0.8)

    def test_enrich_rows_from_metadata_sources_matches_normalized_text(self):
        rows = [{"text": "Tráfico na Rua Alfa", "pred_spans": [], "gold_spans": []}]
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            source.write_text(
                '{"relato":"trafico na rua alfa","logradouroLocal":"Rua Alfa"}\n',
                encoding="utf-8",
            )
            enriched, stats = enrich_rows_from_metadata_sources(
                rows,
                source_paths=[source],
                source_text_fields=["relato", "text"],
                metadata_fields=["logradouroLocal"],
            )
        self.assertEqual(stats["oof_rows_metadata_matched"], 1)
        self.assertEqual(enriched[0]["logradouroLocal"], "Rua Alfa")
        self.assertEqual(enriched[0]["source_fields"]["logradouroLocal"], "Rua Alfa")
        self.assertEqual(enriched[0]["metadata_match"]["status"], "matched_unique_optimizer")


if __name__ == "__main__":
    unittest.main()
