import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inventory_labeled_source_assuntos import (
    build_inventory_rows,
    build_source_indexes,
    match_key,
    summarize_inventory,
)


class InventoryLabeledSourceAssuntosTests(unittest.TestCase):
    def test_ascii_alnum_strategy_matches_tokenized_labeled_text(self):
        source_text = "Denúncia na Rua São João, próximo ao mercado."
        labeled_text = "Denuncia na Rua Sao Joao , proximo ao mercado ."

        self.assertNotEqual(match_key(source_text, "exact"), match_key(labeled_text, "exact"))
        self.assertEqual(
            match_key(source_text, "ascii_alnum_lower"),
            match_key(labeled_text, "ascii_alnum_lower"),
        )

    def test_inventory_recovers_assunto_and_marks_duplicate_same_assunto(self):
        indexes, source_counts = build_source_indexes(
            [str(Path(__file__).parent / "fixtures_source.json")],
            source_text_field="auto",
        )

        inventory = build_inventory_rows(
            [str(Path(__file__).parent / "fixtures_labeled.json")],
            indexes,
            labeled_text_field="auto",
        )
        summary = summarize_inventory(inventory, source_row_counts=source_counts)

        self.assertEqual(inventory[0]["match_status"], "matched_duplicate_same_assunto")
        self.assertEqual(inventory[0]["first_assunto"], "Roubos em Geral")
        self.assertEqual(inventory[0]["matched_partitions"], "sanitized")
        self.assertEqual(inventory[0]["first_source_partition"], "sanitized")
        self.assertEqual(inventory[0]["first_sanitization_status"], "kept")
        self.assertEqual(inventory[0]["first_sanitization_row_index_1based"], "10")
        self.assertEqual(inventory[0]["match_count"], 2)
        self.assertEqual(summary["match_status"], {"matched_duplicate_same_assunto": 1})
        self.assertEqual(summary["matched_partitions"], {"sanitized": 1})
        self.assertEqual(summary["unambiguous_assunto"], {"Roubos em Geral": 1})


if __name__ == "__main__":
    unittest.main()
