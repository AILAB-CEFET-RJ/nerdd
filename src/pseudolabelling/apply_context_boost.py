import logging

from pseudolabelling.context_boost import run_context_boost
from pseudolabelling.context_boost_cli import build_config, parse_args


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_config(args)
    run_context_boost(config, script_path=__file__)


if __name__ == "__main__":
    main()
