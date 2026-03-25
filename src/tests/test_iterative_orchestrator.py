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
                prediction_model_max_length=384,
                prediction_map_location="cuda",
                refit_mode="supervised_only",
                refit_pseudolabel_path="../artifacts/accumulated/kept_acc_01.jsonl",
                use_calibration=False,
                evaluate_refit=True,
                eval_gt_jsonl="../data/dd_corpus_small_test_final.json",
                eval_model_max_length=384,
                eval_map_location="cuda",
                prepare_next_iteration=True,
            )

            def write_fake_eval_artifacts(eval_cfg, script_path):
                out_dir = Path(eval_cfg["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                if out_dir.name == "07_eval_base":
                    metrics = {
                        "micro": {"f1": 0.40},
                        "macro_f1": 0.35,
                        "labels": ["Person", "Location"],
                        "per_label": {
                            "Person": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 10},
                            "Location": {"precision": 0.6, "recall": 0.3, "f1": 0.40, "support": 20},
                        },
                    }
                else:
                    metrics = {
                        "micro": {"f1": 0.45},
                        "macro_f1": 0.38,
                        "labels": ["Person", "Location"],
                        "per_label": {
                            "Person": {"precision": 0.55, "recall": 0.45, "f1": 0.49, "support": 10},
                            "Location": {"precision": 0.62, "recall": 0.34, "f1": 0.44, "support": 20},
                        },
                    }
                (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

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
                p_eval.side_effect = write_fake_eval_artifacts
                run_iterative_cycle(config, script_path=__file__)

            self.assertEqual(p_pred.call_count, 1)
            self.assertEqual(p_boost.call_count, 1)
            self.assertEqual(p_score.call_count, 1)
            self.assertEqual(p_split.call_count, 1)
            self.assertEqual(p_refit.call_count, 1)
            self.assertEqual(p_eval.call_count, 2)
            self.assertEqual(p_prepare.call_count, 1)
            self.assertEqual(p_refit.call_args.args[0].refit_mode, "supervised_only")
            self.assertEqual(
                p_refit.call_args.args[0].pseudolabel_path,
                "../artifacts/accumulated/kept_acc_01.jsonl",
            )
            self.assertEqual(p_pred.call_args.args[0].model_max_length, 384)
            self.assertEqual(p_pred.call_args.args[0].map_location, "cuda")
            self.assertEqual(p_boost.call_args.args[0].boost_scope, "location-matched-only")
            self.assertEqual(p_eval.call_args_list[0].args[0]["model_max_length"], 384)
            self.assertEqual(p_eval.call_args_list[1].args[0]["model_max_length"], 384)
            self.assertEqual(p_eval.call_args_list[0].args[0]["map_location"], "cuda")
            self.assertEqual(p_eval.call_args_list[1].args[0]["map_location"], "cuda")

            summary_path = run_dir / "cycle_summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["refit_mode"], "supervised_only")
            self.assertEqual(payload["config"]["prediction_map_location"], "cuda")
            self.assertEqual(payload["config"]["prediction_model_max_length"], 384)
            self.assertEqual(
                payload["config"]["refit_pseudolabel_path"],
                "../artifacts/accumulated/kept_acc_01.jsonl",
            )
            self.assertEqual(payload["config"]["use_calibration"], False)
            self.assertEqual(payload["config"]["evaluate_refit"], True)
            self.assertEqual(payload["config"]["eval_model_max_length"], 384)
            self.assertEqual(payload["config"]["eval_map_location"], "cuda")
            self.assertEqual(payload["config"]["prepare_next_iteration"], True)
            self.assertTrue((run_dir / "09_base_vs_refit_comparison.json").exists())
            comparison_payload = json.loads((run_dir / "09_base_vs_refit_comparison.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(comparison_payload["micro_f1"]["delta"], 0.05)
            self.assertEqual(Path(payload["artifacts"]["base_eval_dir"]).name, "07_eval_base")
            self.assertEqual(Path(payload["artifacts"]["refit_eval_dir"]).name, "08_eval_refit")
            self.assertEqual(Path(payload["artifacts"]["base_vs_refit_comparison"]).name, "09_base_vs_refit_comparison.json")


if __name__ == "__main__":
    unittest.main()
