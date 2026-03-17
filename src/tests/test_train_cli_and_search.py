import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.cli import build_config, parse_args
from base_model_training.search import generate_trial_params


class TrainCliAndSearchTests(unittest.TestCase):
    def test_parse_args_builds_split_learning_rate_lists(self):
        argv = [
            "train_nested_kfold.py",
            "--backbone-lr-values",
            "5e-6,1e-5",
            "--ner-lr-values",
            "2e-5,3e-5",
            "--weight-decay-values",
            "0.01,0.05",
            "--train-sampling",
            "weighted",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        config = build_config(args)
        self.assertEqual(config.backbone_lr_values, [5e-6, 1e-5])
        self.assertEqual(config.ner_lr_values, [2e-5, 3e-5])
        self.assertEqual(config.weight_decay_values, [0.01, 0.05])
        self.assertEqual(config.train_sampling, "weighted")

    def test_removed_lr_values_argument_is_rejected(self):
        argv = ["train_nested_kfold.py", "--lr-values", "1e-5"]

        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_generate_trial_params_uses_three_way_product(self):
        class _Config:
            backbone_lr_values = [5e-6, 1e-5]
            ner_lr_values = [2e-5, 3e-5]
            weight_decay_values = [0.01, 0.05]
            search_mode = "grid"
            num_trials = 10

        candidates = generate_trial_params(_Config(), rng=None)

        self.assertEqual(len(candidates), 8)
        self.assertIn((5e-6, 2e-5, 0.01), candidates)
        self.assertIn((1e-5, 3e-5, 0.05), candidates)


if __name__ == "__main__":
    unittest.main()
