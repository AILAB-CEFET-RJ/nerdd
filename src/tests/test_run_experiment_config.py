import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_experiment_config import output_dir_for_repeat, repeat_specs


class RunExperimentConfigTests(unittest.TestCase):
    def test_repeat_specs_expand_from_seed_start(self):
        specs = repeat_specs({"n_repeats": 3, "seed_start": 53}, default_seed=42)

        self.assertEqual([spec["seed"] for spec in specs], [53, 54, 55])
        self.assertEqual(specs[0]["repeat_index_1based"], 1)
        self.assertEqual(specs[2]["n_repeats"], 3)

    def test_repeat_specs_fall_back_to_config_seed(self):
        specs = repeat_specs({}, default_seed=42)

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["seed"], 42)

    def test_output_dir_for_repeat_adds_repeat_subdir_only_for_multiple_repeats(self):
        single = {
            "repeat_index_1based": 1,
            "n_repeats": 1,
            "seed": 42,
        }
        repeated = {
            "repeat_index_1based": 2,
            "n_repeats": 3,
            "seed": 54,
        }

        self.assertEqual(output_dir_for_repeat("artifacts/run", single), "artifacts/run")
        self.assertEqual(
            output_dir_for_repeat("artifacts/run", repeated),
            "artifacts/run/repeat_02_seed54",
        )


if __name__ == "__main__":
    unittest.main()
