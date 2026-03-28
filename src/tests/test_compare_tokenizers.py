import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_tokenizers import _tokenize, build_summary


class CompareTokenizersTests(unittest.TestCase):
    def test_tokenize_exposes_per_token_rows(self):
        class StubTokenizer:
            is_fast = True
            unk_token = "[UNK]"

            def __call__(self, text, add_special_tokens, truncation, return_offsets_mapping):
                return {"input_ids": [101, 102], "offset_mapping": [(0, 2), (3, 5)]}

            def convert_ids_to_tokens(self, input_ids):
                return ["ab", "[UNK]"]

        result = _tokenize(StubTokenizer(), "ab cd")
        self.assertEqual(result["token_rows"][0]["token"], "ab")
        self.assertEqual(result["token_rows"][0]["input_id"], 101)
        self.assertEqual(result["token_rows"][0]["start"], 0)
        self.assertEqual(result["token_rows"][0]["end"], 2)
        self.assertEqual(result["token_rows"][0]["text"], "ab")
        self.assertEqual(result["token_rows"][1]["token"], "[UNK]")
        self.assertEqual(result["unk_count"], 1)

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
