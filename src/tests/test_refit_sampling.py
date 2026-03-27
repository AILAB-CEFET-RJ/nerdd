import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.refit_pipeline import sample_pseudolabel_rows


class RefitSamplingTests(unittest.TestCase):
    def test_ratio_sampling_is_deterministic_and_non_empty(self):
        rows = [{"text": f"row_{i}"} for i in range(10)]
        sampled_a, counts_a = sample_pseudolabel_rows(rows, sample_ratio=0.3, max_records=0, seed=42)
        sampled_b, counts_b = sample_pseudolabel_rows(rows, sample_ratio=0.3, max_records=0, seed=42)

        self.assertEqual(sampled_a, sampled_b)
        self.assertEqual(len(sampled_a), 3)
        self.assertEqual(counts_a["dropped_by_ratio"], 7)
        self.assertEqual(counts_a, counts_b)

    def test_cap_applies_after_ratio(self):
        rows = [{"text": f"row_{i}"} for i in range(20)]
        sampled, counts = sample_pseudolabel_rows(rows, sample_ratio=0.5, max_records=4, seed=7)

        self.assertEqual(len(sampled), 4)
        self.assertEqual(counts["input_records"], 20)
        self.assertEqual(counts["dropped_by_ratio"], 10)
        self.assertEqual(counts["dropped_by_cap"], 6)
        self.assertEqual(counts["kept_records"], 4)


if __name__ == "__main__":
    unittest.main()
