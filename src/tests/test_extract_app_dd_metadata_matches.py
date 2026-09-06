import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.extract_app_dd_metadata_matches import (  # noqa: E402
    extract_metadata_after_labeled_text,
    find_app_matches,
    normalize_text,
    repair_mojibake,
)


class ExtractAppDdMetadataMatchesTests(unittest.TestCase):
    def test_repair_mojibake(self):
        self.assertEqual(repair_mojibake("TrÃ¡fico de drogas"), "Tráfico de drogas")

    def test_normalize_text_removes_case_accents_and_punctuation(self):
        self.assertEqual(normalize_text("Rua Antônio João, RJ"), "ruaantoniojoaorj")

    def test_extract_metadata_after_split_relato(self):
        app_row = [
            "1",
            "hash",
            "0",
            "0",
            "TrÃ¡fico de drogas",
            "Na esquina da rua",
            " podem ser vistos traficantes",
            "",
            "Nova IguaÃ§u",
            "0",
            "Rua Rita de CÃ¡ssia com Rua E",
            "",
            "",
            "Posse",
            "",
            "Esquina",
            "",
            "42581.497800925928",
            "0",
            "1",
        ]
        metadata, status = extract_metadata_after_labeled_text(
            app_row,
            "Na esquina da rua podem ser vistos traficantes",
        )
        self.assertEqual(status, "ok")
        self.assertEqual(metadata["cidadeLocal"], "Nova Iguaçu")
        self.assertEqual(metadata["logradouroLocal"], "Rua Rita de Cássia com Rua E")
        self.assertEqual(metadata["bairroLocal"], "Posse")
        self.assertEqual(metadata["pontodeReferenciaLocal"], "Esquina")

    def test_find_app_matches_prefers_exact_relato(self):
        exact = {"relatoalvo": [2]}
        full = ["relatoalvometadata", "outrorelatoalvo", "relatoalvo"]
        strategy, matches = find_app_matches(
            labeled_text="Relato alvo",
            exact_relatos=exact,
            full_texts=full,
        )
        self.assertEqual(strategy, "exact_relato")
        self.assertEqual(matches, [2])


if __name__ == "__main__":
    unittest.main()
