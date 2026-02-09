import logging

from pseudolabelling.cli import build_config, parse_args
from pseudolabelling.pipeline import run_corpus_prediction


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_config(args)
    run_corpus_prediction(config, script_path=__file__)


if __name__ == "__main__":
    main()
