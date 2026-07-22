import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.enrich_labeled_with_source_assunto import enrich_rows, validate_preserved_core
from tools.inventory_labeled_source_assuntos import build_source_indexes


class EnrichLabeledWithSourceAssuntoTests(unittest.TestCase):
    def test_enriches_unambiguous_match_and_preserves_core_fields(self):
        indexes, _ = build_source_indexes(
            [str(Path(__file__).parent / "fixtures_source.json")],
            source_text_field="auto",
        )
        rows = [
            {
                "text": "Denuncia na Rua Sao Joao , proximo ao mercado .",
                "spans": [{"start": 12, "end": 25, "label": "Location"}],
            }
        ]

        enriched, audit = enrich_rows(
            rows,
            labeled_input="labeled.json",
            indexes=indexes,
            labeled_text_field="auto",
            overwrite_existing_assunto=False,
        )

        validate_preserved_core(rows, enriched)
        self.assertEqual(enriched[0]["assunto"], "Roubos em Geral")
        self.assertEqual(enriched[0]["_source_assunto_match"]["status"], "matched_duplicate_same_assunto")
        self.assertEqual(enriched[0]["_source_assunto_match"]["selected_assunto"], "Roubos em Geral")
        self.assertEqual(audit[0]["selected_assunto"], "Roubos em Geral")

    def test_does_not_set_assunto_for_unmatched_row(self):
        indexes, _ = build_source_indexes(
            [str(Path(__file__).parent / "fixtures_source.json")],
            source_text_field="auto",
        )
        rows = [{"text": "Relato sem correspondencia", "spans": []}]

        enriched, audit = enrich_rows(
            rows,
            labeled_input="labeled.json",
            indexes=indexes,
            labeled_text_field="auto",
            overwrite_existing_assunto=False,
        )

        self.assertNotIn("assunto", enriched[0])
        self.assertEqual(enriched[0]["_source_assunto_match"]["status"], "unmatched")
        self.assertEqual(audit[0]["selected_assunto"], "")


if __name__ == "__main__":
    unittest.main()
