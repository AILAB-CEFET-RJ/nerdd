import logging

from pseudolabelling.evaluate_refit_cli import build_config, parse_args
from pseudolabelling.evaluate_refit_pipeline import run_evaluate_refit


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_config(args)
    run_evaluate_refit(config, script_path=__file__)


if __name__ == "__main__":
    main()
