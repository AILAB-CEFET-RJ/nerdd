import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.iterative_orchestrator import IterativeCycleConfig, run_iterative_cycle


class IterativeOrchestratorTests(unittest.TestCase):
    def test_runs_pipeline_steps_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "iter_run"
            config = IterativeCycleConfig(
                run_dir=str(run_dir),
                use_calibration=False,
                evaluate_refit=True,
                eval_gt_jsonl="../data/dd_corpus_small_test_filtered.json",
                prepare_next_iteration=True,
            )

            with patch("pseudolabelling.iterative_orchestrator._run_corpus_prediction") as p_pred, patch(
                "pseudolabelling.iterative_orchestrator.run_context_boost"
            ) as p_boost, patch(
                "pseudolabelling.iterative_orchestrator.run_compute_record_score"
            ) as p_score, patch("pseudolabelling.iterative_orchestrator.run_split") as p_split, patch(
                "pseudolabelling.iterative_orchestrator.run_refit"
            ) as p_refit, patch(
                "pseudolabelling.iterative_orchestrator.run_evaluate_refit"
            ) as p_eval, patch(
                "pseudolabelling.iterative_orchestrator.run_prepare_next_iteration"
            ) as p_prepare:
                run_iterative_cycle(config, script_path=__file__)

            self.assertEqual(p_pred.call_count, 1)
            self.assertEqual(p_boost.call_count, 1)
            self.assertEqual(p_score.call_count, 1)
            self.assertEqual(p_split.call_count, 1)
            self.assertEqual(p_refit.call_count, 1)
            self.assertEqual(p_eval.call_count, 1)
            self.assertEqual(p_prepare.call_count, 1)

            summary_path = run_dir / "cycle_summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["use_calibration"], False)
            self.assertEqual(payload["config"]["evaluate_refit"], True)
            self.assertEqual(payload["config"]["prepare_next_iteration"], True)


if __name__ == "__main__":
    unittest.main()
