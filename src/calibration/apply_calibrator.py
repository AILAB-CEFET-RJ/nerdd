import argparse
import json
import logging
from pathlib import Path

try:
    from calibration.serialization import apply_calibrator_to_score, load_calibrator
except ImportError:  # pragma: no cover
    from serialization import apply_calibrator_to_score, load_calibrator

LOGGER = logging.getLogger(__name__)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a fitted calibrator artifact to entity scores in a JSONL corpus.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--calibrator-path", required=True)
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--output-score-field", default="score_calibrated")
    parser.add_argument("--preserve-original-score-field", default="score_original")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    rows = load_jsonl(args.input_jsonl)
    calibrator = load_calibrator(args.calibrator_path)

    entities_calibrated = 0
    for row in rows:
        for entity in row.get("entities", []):
            if args.score_field not in entity:
                continue
            label = str(entity.get(args.label_field, ""))
            raw_score = entity.get(args.score_field)
            if args.preserve_original_score_field and args.preserve_original_score_field not in entity:
                entity[args.preserve_original_score_field] = raw_score
            entity[args.output_score_field] = apply_calibrator_to_score(raw_score, label, calibrator)
            entities_calibrated += 1

    save_jsonl(args.output_jsonl, rows)
    LOGGER.info("Applied calibrator to %s entities", entities_calibrated)
    LOGGER.info("Saved calibrated JSONL to: %s", args.output_jsonl)


if __name__ == "__main__":
    main()
