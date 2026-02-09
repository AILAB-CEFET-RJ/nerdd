import logging

from pseudolabelling.iterative_orchestrator import build_config, parse_args, run_iterative_cycle


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_config(args)
    run_iterative_cycle(config, script_path=__file__)


if __name__ == "__main__":
    main()
