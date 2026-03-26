import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_chunk_quality import infer_audit_dir, summarize_kept_rows


class EvaluateChunkQualityTests(unittest.TestCase):
    def test_infer_audit_dir_from_run_dir_name(self):
        run_dir = Path("artifacts/pseudolabelling/multi_with_negatives_chunk02_50k_t037_cuda_v12")
        audit_dir = infer_audit_dir(run_dir)
        self.assertEqual(
            audit_dir,
            Path("artifacts/pseudolabelling/context_boost_audit_v12_chunk02"),
        )

    def test_summarize_kept_rows_collects_core_metrics(self):
        rows = [
            {
                "relato": "texto a",
                "_split": {"score_value": 0.41},
                "entities": [
                    {"label": "Location", "score_context_boosted": 0.91},
                    {"label": "Person", "score_context_boosted": 0.32},
                ],
            },
            {
                "relato": "texto a",
                "_split": {"score_value": 0.52},
                "entities": [
                    {"label": "Location", "score_context_boosted": 0.81},
                ],
            },
        ]
        summary = summarize_kept_rows(rows, strong_score_threshold=0.8, weak_score_threshold=0.5)
        self.assertEqual(summary["kept_rows"], 2)
        self.assertEqual(summary["unique_texts"], 1)
        self.assertEqual(summary["duplicate_text_rows"], 1)
        self.assertAlmostEqual(summary["duplicate_text_rate"], 0.5)
        self.assertAlmostEqual(summary["avg_entities_per_kept"], 1.5)
        self.assertAlmostEqual(summary["avg_strong_entities_per_kept"], 1.0)
        self.assertAlmostEqual(summary["avg_weak_entities_per_kept"], 0.5)
        self.assertAlmostEqual(summary["avg_record_score_in_kept"], 0.465)
        self.assertEqual(summary["label_counts"]["Location"], 2)
        self.assertEqual(summary["label_counts"]["Person"], 1)


if __name__ == "__main__":
    unittest.main()
