import logging

from pseudolabelling.prepare_next_iteration_cli import build_config, parse_args
from pseudolabelling.prepare_next_iteration_pipeline import run_prepare_next_iteration


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_config(args)
    run_prepare_next_iteration(config, script_path=__file__)


if __name__ == "__main__":
    main()
