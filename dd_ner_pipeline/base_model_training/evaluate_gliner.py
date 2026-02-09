import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gliner_train.eval_cli import build_config, parse_args


def main():
    """Script entrypoint."""
    args = parse_args()
    from gliner_train.evaluate import run_evaluation

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_config(args)
    run_evaluation(config, script_path=__file__)


if __name__ == "__main__":
    main()
