import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.reshuffle_train_test_split import remove_exact_duplicates_across_inputs, reshuffle_train_test


class ReshuffleTrainTestSplitTests(unittest.TestCase):
    def test_preserves_sizes_and_is_deterministic(self):
        train_rows = [{"id": f"train_{idx}"} for idx in range(4)]
        test_rows = [{"id": f"test_{idx}"} for idx in range(2)]

        train_a, test_a, order_a = reshuffle_train_test(train_rows, test_rows, seed=42)
        train_b, test_b, order_b = reshuffle_train_test(train_rows, test_rows, seed=42)

        self.assertEqual(len(train_a), 4)
        self.assertEqual(len(test_a), 2)
        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)
        self.assertEqual(order_a, order_b)
        combined_ids = {row["id"] for row in train_a + test_a}
        self.assertEqual(combined_ids, {row["id"] for row in train_rows + test_rows})

    def test_remove_duplicates_across_inputs_keeps_train_copy(self):
        shared = {"id": "shared", "text": "same"}
        train_rows = [{"id": "train_only"}, shared]
        test_rows = [shared, {"id": "test_only"}]

        dedup_train, dedup_test, dropped = remove_exact_duplicates_across_inputs(train_rows, test_rows)

        self.assertEqual(dedup_train, train_rows)
        self.assertEqual(dedup_test, [{"id": "test_only"}])
        self.assertEqual(dropped, 1)


if __name__ == "__main__":
    unittest.main()
