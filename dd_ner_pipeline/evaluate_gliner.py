"""Deprecated wrapper.

Use `gliner_train/evaluate_gliner.py` instead.
"""

import logging

from gliner_train.evaluate_gliner import main as _new_main


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.warning("Deprecated script: use 'gliner_train/evaluate_gliner.py'.")
    _new_main()


if __name__ == "__main__":
    main()
