"""Nested KFold training CLI entrypoint."""

import logging

from base_model_training.cli import build_config, parse_args
from base_model_training.cv import run_experiment


def main():
    """Script entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_config(args)
    run_experiment(config, script_path=__file__)


if __name__ == "__main__":
    main()
