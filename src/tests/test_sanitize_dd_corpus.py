import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sanitize_dd_corpus import classify_row


class SanitizeDdCorpusTests(unittest.TestCase):
    def classify(self, relato, *, seen_exact=None, seen_normalized=None):
        if seen_exact is None:
            seen_exact = set()
        if seen_normalized is None:
            seen_normalized = set()
        return classify_row(
            {"relato": relato},
            row_index=1,
            seen_exact=seen_exact,
            seen_normalized=seen_normalized,
            max_relato_chars=5000,
            min_flag_short_chars=20,
        )

    def test_drops_name_list_dump(self):
        status, reasons, _ = self.classify("Ivete Sangalo, Xuxa Meneguel, Anita thainara carvalho, são taradas")
        self.assertEqual(status, "dropped_safe")
        self.assertIn("name_list_dump", reasons)

    def test_drops_short_low_context_text(self):
        status, reasons, _ = self.classify("Lucas de aparecida Gonçalves")
        self.assertEqual(status, "dropped_safe")
        self.assertIn("short_low_context", reasons)

    def test_keeps_short_domain_like_report(self):
        status, reasons, _ = self.classify("Barricadas em Eng Belford")
        self.assertNotEqual(status, "dropped_safe")
        self.assertNotIn("short_low_context", reasons)
        self.assertNotIn("name_list_dump", reasons)


if __name__ == "__main__":
    unittest.main()
