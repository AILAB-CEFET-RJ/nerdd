import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_tokenizers import build_summary


class CompareTokenizersTests(unittest.TestCase):
    def test_build_summary_counts_differences_and_unk_rows(self):
        comparisons = [
            {
                "fast": {"unk_count": 1},
                "slow": {"unk_count": 0},
                "delta": {"token_count": 2, "unk_count": 1, "tokens_equal": False},
            },
            {
                "fast": {"unk_count": 0},
                "slow": {"unk_count": 1},
                "delta": {"token_count": 0, "unk_count": -1, "tokens_equal": True},
            },
        ]
        summary = build_summary(comparisons)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["rows_with_token_differences"], 1)
        self.assertEqual(summary["rows_with_fast_unk"], 1)
        self.assertEqual(summary["rows_with_slow_unk"], 1)


if __name__ == "__main__":
    unittest.main()
