#!/usr/bin/env python3
"""Apply a saved NER score calibrator to entity predictions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl, save_jsonl
from base_model_training.paths import resolve_path
from calibration.ner_score_calibrator import apply_calibrator_to_score

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a per-label NER score calibrator to prediction JSONL.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--output-score-field", default="score_calibrated")
    parser.add_argument("--entity-key", default="entities")
    parser.add_argument("--fallback-entity-key", default="ner")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def calibrated_rows(
    rows: list[dict[str, Any]],
    *,
    calibrator: dict[str, Any],
    entity_key: str,
    fallback_entity_key: str,
    score_field: str,
    output_score_field: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    stats = {
        "rows_total": len(rows),
        "entity_lists_found": 0,
        "entities_total": 0,
        "entities_calibrated": 0,
        "entities_without_score": 0,
    }
    for row in rows:
        enriched = dict(row)
        key = entity_key if isinstance(enriched.get(entity_key), list) else fallback_entity_key
        entities = enriched.get(key)
        if not isinstance(entities, list):
            output.append(enriched)
            continue
        stats["entity_lists_found"] += 1
        new_entities = []
        for entity in entities:
            new_entity = dict(entity)
            stats["entities_total"] += 1
            if score_field not in new_entity:
                stats["entities_without_score"] += 1
            else:
                new_entity[output_score_field] = apply_calibrator_to_score(
                    new_entity[score_field],
                    str(new_entity.get("label", "")),
                    calibrator,
                )
                stats["entities_calibrated"] += 1
            new_entities.append(new_entity)
        enriched[key] = new_entities
        output.append(enriched)
    return output, stats


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    script_dir = Path(__file__).resolve().parent
    input_path = resolve_path(script_dir, args.input_jsonl)
    output_path = resolve_path(script_dir, args.output_jsonl)
    calibrator_path = resolve_path(script_dir, args.calibrator)

    rows = load_jsonl(str(input_path))
    calibrator = load_json(calibrator_path)
    output, stats = calibrated_rows(
        rows,
        calibrator=calibrator,
        entity_key=args.entity_key,
        fallback_entity_key=args.fallback_entity_key,
        score_field=args.score_field,
        output_score_field=args.output_score_field,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(str(output_path), output)
    LOGGER.info("Saved calibrated predictions: %s", output_path)
    LOGGER.info("Calibration stats: %s", stats)


if __name__ == "__main__":
    main()
