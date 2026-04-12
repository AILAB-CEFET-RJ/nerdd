import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_leave_one_out_refits import _metric, cmd_build, cmd_summarize


class PrepareLeaveOneOutRefitsTests(unittest.TestCase):
    def test_metric_reads_nested_value(self):
        summary = {"test_metrics": {"micro": {"f1": 0.8}, "macro_f1": 0.7}}
        self.assertEqual(_metric(summary, "test_metrics.micro.f1"), 0.8)
        self.assertEqual(_metric(summary, "test_metrics.macro_f1"), 0.7)
        self.assertIsNone(_metric(summary, "missing.key"))

    def test_build_and_summarize_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_jsonl = root / "input.jsonl"
            input_rows = [
                {"source_id": "a", "text": "x"},
                {"source_id": "b", "text": "y"},
            ]
            with input_jsonl.open("w", encoding="utf-8") as handle:
                for row in input_rows:
                    handle.write(json.dumps(row) + "\n")

            output_dir = root / "loo"
            build_summary = root / "build_summary.json"
            cmd_build(
                type(
                    "Args",
                    (),
                    {
                        "input_jsonl": str(input_jsonl),
                        "output_dir": str(output_dir),
                        "summary_json": str(build_summary),
                    },
                )()
            )
            self.assertTrue((output_dir / "without__a.jsonl").exists())
            self.assertTrue((output_dir / "without__b.jsonl").exists())

            baseline_summary = root / "baseline.json"
            baseline_summary.write_text(json.dumps({"test_metrics": {"micro": {"f1": 0.8}, "macro_f1": 0.7}}), encoding="utf-8")
            full_summary = root / "full.json"
            full_summary.write_text(json.dumps({"test_metrics": {"micro": {"f1": 0.82}, "macro_f1": 0.72}}), encoding="utf-8")

            exp_dir = root / "runs"
            (exp_dir / "without__a").mkdir(parents=True)
            (exp_dir / "without__a" / "quick_summary.json").write_text(
                json.dumps({"test_metrics": {"micro": {"f1": 0.83}, "macro_f1": 0.73}}),
                encoding="utf-8",
            )
            (exp_dir / "without__b").mkdir(parents=True)
            (exp_dir / "without__b" / "quick_summary.json").write_text(
                json.dumps({"test_metrics": {"micro": {"f1": 0.81}, "macro_f1": 0.71}}),
                encoding="utf-8",
            )

            summary_json = root / "summary.json"
            cmd_summarize(
                type(
                    "Args",
                    (),
                    {
                        "baseline_summary": str(baseline_summary),
                        "full_summary": str(full_summary),
                        "experiment_dir": str(exp_dir),
                        "summary_json": str(summary_json),
                    },
                )()
            )

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(len(summary["experiments"]), 2)
            self.assertEqual(summary["experiments"][0]["dropped_source_id"], "a")
            self.assertAlmostEqual(summary["experiments"][0]["delta_vs_full_micro"], 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
