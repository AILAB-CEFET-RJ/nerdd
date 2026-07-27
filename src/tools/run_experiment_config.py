#!/usr/bin/env python3
"""Run experiments declared in JSON config files."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_configs import (
    load_experiment_config_file,
    select_experiment,
)


LOGGER = logging.getLogger(__name__)


ENTRYPOINTS = {"base_model_training.train_quick"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JSON-configured experiments.")
    parser.add_argument("--config-json", required=True, help="JSON experiment config file.")
    parser.add_argument("--experiment-id", default="", help="Experiment id to run.")
    parser.add_argument("--all", action="store_true", help="Run every experiment in the config file.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def repeat_specs(metadata: dict, default_seed: int) -> list[dict]:
    n_repeats = int(metadata.get("n_repeats", 1) or 1)
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1.")
    seed_start = metadata.get("seed_start")
    if seed_start is None:
        seed_start = default_seed
    seed_start = int(seed_start)
    return [
        {
            "repeat_index_0based": idx,
            "repeat_index_1based": idx + 1,
            "n_repeats": n_repeats,
            "seed_start": seed_start,
            "seed": seed_start + idx,
        }
        for idx in range(n_repeats)
    ]


def output_dir_for_repeat(output_dir: str, spec: dict) -> str:
    if spec["n_repeats"] == 1:
        return output_dir
    suffix = f"repeat_{spec['repeat_index_1based']:02d}_seed{spec['seed']}"
    return str(Path(output_dir) / suffix)


def _run_train_quick(raw_config: dict) -> None:
    from base_model_training.train_quick import QuickTrainConfig, run_quick_experiment
    from experiment_configs import build_dataclass_config_from_experiment

    config, metadata = build_dataclass_config_from_experiment(
        raw_config,
        config_class=QuickTrainConfig,
        required_entrypoint="base_model_training.train_quick",
        defaults=QuickTrainConfig(thresholds=[0.6]),
    )
    experiment_id = metadata.get("experiment_id", "")
    if experiment_id:
        LOGGER.info("Running experiment: %s", experiment_id)
    for spec in repeat_specs(metadata, config.seed):
        repeat_config = replace(
            config,
            seed=spec["seed"],
            output_dir=output_dir_for_repeat(config.output_dir, spec),
        )
        repeat_metadata = {**metadata, "repeat": spec}
        if spec["n_repeats"] > 1:
            LOGGER.info(
                "Running repeat %s/%s with seed %s",
                spec["repeat_index_1based"],
                spec["n_repeats"],
                spec["seed"],
            )
        run_quick_experiment(
            repeat_config,
            script_path=str(Path(__file__).resolve().parents[1] / "base_model_training" / "train_quick.py"),
            experiment_metadata=repeat_metadata,
        )


def run_one(raw_config: dict) -> None:
    entrypoint = raw_config.get("entrypoint", "base_model_training.train_quick")
    if entrypoint not in ENTRYPOINTS:
        raise ValueError(f"Unsupported entrypoint: {entrypoint}")
    if entrypoint == "base_model_training.train_quick":
        _run_train_quick(raw_config)
        return
    raise ValueError(f"Unsupported entrypoint: {entrypoint}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    experiments = load_experiment_config_file(args.config_json)
    if args.all and args.experiment_id:
        raise ValueError("Use either --all or --experiment-id, not both.")
    selected = experiments if args.all else [select_experiment(experiments, experiment_id=args.experiment_id)]
    for raw_config in selected:
        run_one(raw_config)


if __name__ == "__main__":
    main()
