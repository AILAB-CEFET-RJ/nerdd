import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.reshuffle_train_test_split import reshuffle_train_test


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


if __name__ == "__main__":
    unittest.main()
