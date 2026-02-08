import logging

from gliner_train.eval_cli import build_config, parse_args
from gliner_train.evaluate import run_evaluation


def main():
    """Script entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_config(args)
    run_evaluation(config, script_path=__file__)


if __name__ == "__main__":
    main()
