import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_configs import (
    build_dataclass_config_from_experiment,
    load_dataclass_config,
    select_experiment,
)


@dataclass
class DummyConfig:
    train_path: str = "train.json"
    thresholds: list[float] | None = None
    tokenization_strategy: str = "whitespace"


class ExperimentConfigTests(unittest.TestCase):
    def test_select_experiment_requires_id_for_multiple_configs(self):
        experiments = [{"experiment_id": "a"}, {"experiment_id": "b"}]

        with self.assertRaises(ValueError):
            select_experiment(experiments)

        self.assertEqual(select_experiment(experiments, experiment_id="b")["experiment_id"], "b")

    def test_build_dataclass_config_applies_defaults_and_normalizes_thresholds(self):
        config, metadata = build_dataclass_config_from_experiment(
            {
                "experiment_id": "exp1",
                "entrypoint": "base_model_training.train_quick",
                "thresholds": [0.5, "0.6"],
                "tokenization_strategy": "regex",
            },
            config_class=DummyConfig,
            required_entrypoint="base_model_training.train_quick",
            defaults=DummyConfig(thresholds=[0.7]),
        )

        self.assertEqual(metadata["experiment_id"], "exp1")
        self.assertEqual(config.train_path, "train.json")
        self.assertEqual(config.thresholds, [0.5, 0.6])
        self.assertEqual(config.tokenization_strategy, "regex")

    def test_build_dataclass_config_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            build_dataclass_config_from_experiment(
                {"unknown": True},
                config_class=DummyConfig,
            )

    def test_load_dataclass_config_accepts_single_object_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"train_path": "custom.json"}), encoding="utf-8")

            config, _metadata = load_dataclass_config(path, config_class=DummyConfig)

        self.assertEqual(config.train_path, "custom.json")


if __name__ == "__main__":
    unittest.main()
