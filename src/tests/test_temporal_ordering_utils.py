import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sample_large_corpus import sample_rows
from tools.split_dataset_for_calibration import build_random_split
from tools.split_large_corpus_into_chunks import maybe_shuffle_rows


class TemporalOrderingUtilsTests(unittest.TestCase):
    def test_sample_rows_can_shuffle_output_order(self):
        rows = [{"id": i} for i in range(20)]
        sampled_preserved, indices_preserved = sample_rows(rows, sample_size=5, seed=42, preserve_input_order=True)
        sampled_shuffled, indices_shuffled = sample_rows(rows, sample_size=5, seed=42, preserve_input_order=False)

        self.assertEqual(sorted(indices_preserved), sorted(indices_shuffled))
        self.assertEqual(indices_preserved, sorted(indices_preserved))
        self.assertNotEqual(indices_shuffled, sorted(indices_shuffled))
        self.assertEqual([row["id"] for row in sampled_preserved], indices_preserved)
        self.assertEqual([row["id"] for row in sampled_shuffled], indices_shuffled)

    def test_chunk_shuffle_changes_row_order_reproducibly(self):
        rows = [{"id": i} for i in range(10)]
        shuffled_a = maybe_shuffle_rows(rows, shuffle_first=True, seed=7)
        shuffled_b = maybe_shuffle_rows(rows, shuffle_first=True, seed=7)

        self.assertEqual(shuffled_a, shuffled_b)
        self.assertNotEqual([row["id"] for row in shuffled_a], list(range(10)))

    def test_calibration_split_can_shuffle_output(self):
        rows = [{"spans": [{"label": "Person"}], "id": i} for i in range(10)]
        calibration_rows, final_rows, summary = build_random_split(
            rows,
            ratio=0.3,
            seed=42,
            label_field="label",
            shuffle_output=True,
        )

        self.assertTrue(summary["shuffle_output"])
        self.assertNotEqual([row["id"] for row in calibration_rows], sorted(row["id"] for row in calibration_rows))
        self.assertEqual(len(calibration_rows) + len(final_rows), 10)
