import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.split_by_score import split_records


class SplitByScoreTests(unittest.TestCase):
    def test_operator_ge(self):
        rows = [{"id": 1, "s": 0.5}, {"id": 2, "s": 0.7}]
        kept, discarded, summary = split_records(
            rows=rows,
            score_field="s",
            threshold=0.6,
            operator="ge",
            fallback_score_field="",
            missing_policy="discard",
            trace_key="_split",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(discarded), 1)
        self.assertEqual(kept[0]["id"], 2)
        self.assertEqual(summary["kept_count"], 1)

    def test_operator_lt(self):
        rows = [{"id": 1, "s": 0.5}, {"id": 2, "s": 0.7}]
        kept, discarded, _summary = split_records(
            rows=rows,
            score_field="s",
            threshold=0.6,
            operator="lt",
            fallback_score_field="",
            missing_policy="discard",
            trace_key="_split",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["id"], 1)
        self.assertEqual(len(discarded), 1)

    def test_fallback_field(self):
        rows = [{"id": 1, "fallback": 0.9}]
        kept, discarded, _summary = split_records(
            rows=rows,
            score_field="missing",
            threshold=0.8,
            operator="ge",
            fallback_score_field="fallback",
            missing_policy="discard",
            trace_key="_split",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(discarded), 0)
        self.assertEqual(kept[0]["_split"]["score_source"], "fallback")

    def test_missing_policy_zero(self):
        rows = [{"id": 1}, {"id": 2, "s": 0.2}]
        kept, discarded, summary = split_records(
            rows=rows,
            score_field="s",
            threshold=0.1,
            operator="ge",
            fallback_score_field="",
            missing_policy="zero",
            trace_key="_split",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(discarded), 1)
        self.assertEqual(summary["missing_scores"], 0)

    def test_missing_policy_discard(self):
        rows = [{"id": 1}, {"id": 2, "s": 0.2}]
        kept, discarded, summary = split_records(
            rows=rows,
            score_field="s",
            threshold=0.1,
            operator="ge",
            fallback_score_field="",
            missing_policy="discard",
            trace_key="_split",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(discarded), 1)
        self.assertEqual(summary["missing_scores"], 1)
        self.assertEqual(discarded[0]["_split"]["missing_reason"], "missing_score_fields")

    def test_missing_policy_error(self):
        rows = [{"id": 1}]
        with self.assertRaises(ValueError):
            split_records(
                rows=rows,
                score_field="s",
                threshold=0.1,
                operator="ge",
                fallback_score_field="",
                missing_policy="error",
                trace_key="_split",
            )


if __name__ == "__main__":
    unittest.main()
