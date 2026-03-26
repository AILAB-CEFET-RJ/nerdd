import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import filter_dense_rows


class InspectDenseTipsTests(unittest.TestCase):
    def test_filter_dense_rows_applies_threshold_and_sorts(self):
        rows = [
            {
                "relato": "a" * 10,
                "entities": [{"label": "Location"}] * 3,
            },
            {
                "relato": "b" * 30,
                "entities": [{"label": "Location"}] * 5,
            },
            {
                "relato": "c" * 20,
                "entities": [{"label": "Person"}] * 5,
            },
        ]
        dense_rows = filter_dense_rows(rows, min_entities=4, label_field="label")
        self.assertEqual(len(dense_rows), 2)
        self.assertEqual(dense_rows[0]["_dense_tip"]["entity_count"], 5)
        self.assertEqual(dense_rows[0]["relato"], "b" * 30)
        self.assertEqual(dense_rows[1]["relato"], "c" * 20)


if __name__ == "__main__":
    unittest.main()
