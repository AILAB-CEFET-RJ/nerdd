import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.evaluate_refit_pipeline import (
    compute_span_metrics,
    format_classification_report,
    load_gt_jsonl_strict,
)


class EvaluateRefitPipelineTests(unittest.TestCase):
    def test_load_gt_jsonl_strict_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.jsonl"
            row = {"text": "John in Rio", "spans": [{"start": 0, "end": 4, "label": "Person"}]}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            rows = load_gt_jsonl_strict(str(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["text"], "John in Rio")

    def test_load_gt_jsonl_strict_invalid_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.jsonl"
            row = {"text": "", "spans": []}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_gt_jsonl_strict(str(path))

    def test_compute_span_metrics(self):
        gold = [[{"start": 0, "end": 4, "label": "Person"}]]
        pred = [[{"start": 0, "end": 4, "label": "Person"}]]
        metrics = compute_span_metrics(gold, pred, labels=["Person"])
        self.assertAlmostEqual(metrics["micro"]["f1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0, places=6)
        report = format_classification_report(metrics)
        self.assertIn("macro f1", report)

    def test_compute_span_metrics_with_errors(self):
        gold = [[{"start": 0, "end": 4, "label": "Person"}]]
        pred = [[{"start": 5, "end": 8, "label": "Person"}]]
        metrics = compute_span_metrics(gold, pred, labels=["Person"])
        self.assertAlmostEqual(metrics["micro"]["f1"], 0.0, places=6)
        self.assertEqual(metrics["per_label_errors"]["Person"]["fp"], 1)
        self.assertEqual(metrics["per_label_errors"]["Person"]["fn"], 1)


if __name__ == "__main__":
    unittest.main()
