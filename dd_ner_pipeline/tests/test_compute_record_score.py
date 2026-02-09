import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.compute_record_score import compute_record_score


class ComputeRecordScoreTests(unittest.TestCase):
    def test_mean_aggregation(self):
        record = {"entities": [{"score": 0.2}, {"score": 0.6}]}
        score, valid, invalid, empty = compute_record_score(
            record,
            score_field="score",
            entity_key="entities",
            aggregation="mean",
            empty_entities_policy="zero",
        )
        self.assertAlmostEqual(score, 0.4, places=6)
        self.assertEqual(valid, 2)
        self.assertEqual(invalid, 0)
        self.assertFalse(empty)

    def test_max_aggregation(self):
        record = {"entities": [{"score": 0.2}, {"score": 0.6}]}
        score, _, _, _ = compute_record_score(
            record,
            score_field="score",
            entity_key="entities",
            aggregation="max",
            empty_entities_policy="zero",
        )
        self.assertAlmostEqual(score, 0.6, places=6)

    def test_median_aggregation(self):
        record = {"entities": [{"score": 0.1}, {"score": 0.9}, {"score": 0.5}]}
        score, _, _, _ = compute_record_score(
            record,
            score_field="score",
            entity_key="entities",
            aggregation="median",
            empty_entities_policy="zero",
        )
        self.assertAlmostEqual(score, 0.5, places=6)

    def test_p75_aggregation(self):
        record = {"entities": [{"score": 0.1}, {"score": 0.2}, {"score": 0.9}, {"score": 0.95}]}
        score, _, _, _ = compute_record_score(
            record,
            score_field="score",
            entity_key="entities",
            aggregation="p75",
            empty_entities_policy="zero",
        )
        self.assertAlmostEqual(score, 0.9, places=6)

    def test_empty_policy_null(self):
        record = {"entities": []}
        score, valid, invalid, empty = compute_record_score(
            record,
            score_field="score",
            entity_key="entities",
            aggregation="mean",
            empty_entities_policy="null",
        )
        self.assertIsNone(score)
        self.assertEqual(valid, 0)
        self.assertEqual(invalid, 0)
        self.assertTrue(empty)

    def test_empty_policy_error(self):
        record = {"entities": [{"score": "n/a"}]}
        with self.assertRaises(ValueError):
            compute_record_score(
                record,
                score_field="score",
                entity_key="entities",
                aggregation="mean",
                empty_entities_policy="error",
            )

    def test_ner_fallback(self):
        record = {"ner": [{"score": 0.8}]}
        score, _, _, _ = compute_record_score(
            record,
            score_field="score",
            entity_key="entities",
            aggregation="mean",
            empty_entities_policy="zero",
        )
        self.assertAlmostEqual(score, 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
