import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_refit_regressions import run_audit


class AuditRefitRegressionsTests(unittest.TestCase):
    def test_audit_detects_boundary_and_missing_regressions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gold_path = root / "gold.json"
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            out_dir = root / "out"

            gold_rows = [
                {
                    "text": "delegacia de Araruama",
                    "spans": [{"start": 0, "end": 21, "label": "Location"}],
                },
                {
                    "text": "Joao mora em Niteroi",
                    "spans": [
                        {"start": 0, "end": 4, "label": "Person"},
                        {"start": 13, "end": 20, "label": "Location"},
                    ],
                },
            ]
            gold_path.write_text(json.dumps(gold_rows, ensure_ascii=False), encoding="utf-8")

            baseline_rows = [
                {
                    "text": "delegacia de Araruama",
                    "entities": [{"start": 0, "end": 21, "label": "Location", "text": "delegacia de Araruama"}],
                },
                {
                    "text": "Joao mora em Niteroi",
                    "entities": [
                        {"start": 0, "end": 4, "label": "Person", "text": "Joao"},
                        {"start": 13, "end": 20, "label": "Location", "text": "Niteroi"},
                    ],
                },
            ]
            baseline_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in baseline_rows) + "\n",
                encoding="utf-8",
            )

            candidate_rows = [
                {
                    "text": "delegacia de Araruama",
                    "entities": [{"start": 13, "end": 21, "label": "Location", "text": "Araruama"}],
                },
                {
                    "text": "Joao mora em Niteroi",
                    "entities": [{"start": 0, "end": 4, "label": "Person", "text": "Joao"}],
                },
            ]
            candidate_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in candidate_rows) + "\n",
                encoding="utf-8",
            )

            class Args:
                gold = str(gold_path)
                baseline_pred = str(baseline_path)
                candidate_pred = str(candidate_path)
                output_dir = str(out_dir)
                title = "test"
                log_level = "INFO"

            run_audit(Args())

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            regressions = [
                json.loads(line)
                for line in (out_dir / "regressions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(summary["outcomes"]["loss"], 2)
            self.assertEqual(summary["loss_reason_counts"]["boundary_or_partial"], 1)
            self.assertEqual(summary["loss_reason_counts"]["missing_entity"], 1)
            self.assertAlmostEqual(summary["micro_f1_delta"], -0.6, places=6)
            loss_reason_sets = [set(row["_audit"]["loss_reasons"]) for row in regressions]
            self.assertIn({"missing_entity"}, loss_reason_sets)
            self.assertIn({"boundary_or_partial"}, loss_reason_sets)


if __name__ == "__main__":
    unittest.main()
