#!/usr/bin/env python3
"""Generate explicit GLiNER2 predictions for existing candidate rows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.paths import resolve_repo_artifact_path
from gliner2_inference import predict_entities_for_text
from gliner2_loader import load_gliner2_model
from tools.inspect_dense_tips import get_text, read_json_or_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)
DEFAULT_ENTITY_TYPES = ["person", "location", "organization"]


def _parse_csv(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def build_summary(rows: list[dict], counters: Counter, args: argparse.Namespace) -> dict:
    return {
        "input": str(resolve_repo_artifact_path(__file__, args.input)),
        "output_jsonl": str(resolve_repo_artifact_path(__file__, args.output_jsonl)),
        "records_total": counters["records_total"],
        "records_emitted": len(rows),
        "gliner2_model": args.model_path,
        "adapter_dir": args.adapter_dir,
        "entity_types": _parse_csv(args.entity_types) or list(DEFAULT_ENTITY_TYPES),
        "summary": {
            "gliner2_entities_total": counters["gliner2_entities_total"],
            "avg_gliner2_entities_per_record": (
                counters["gliner2_entities_total"] / len(rows) if rows else 0.0
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GLiNER2 over an existing JSON/JSONL artifact and append gliner2_entities."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL with canonical text already present.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with appended gliner2_entities.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--model-path", default="fastino/gliner2-base-v1", help="GLiNER2 base model id or local path.")
    parser.add_argument("--adapter-dir", default="", help="Optional GLiNER2 adapter directory.")
    parser.add_argument("--entity-types", default="person,location,organization", help="Comma-separated GLiNER2 entity types.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional maximum number of records to process.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    rows = read_json_or_jsonl(str(resolve_repo_artifact_path(__file__, args.input)))
    if args.max_records > 0:
        rows = rows[: args.max_records]

    entity_types = _parse_csv(args.entity_types) or list(DEFAULT_ENTITY_TYPES)
    model = load_gliner2_model(
        str(resolve_repo_artifact_path(__file__, args.model_path)),
        adapter_dir=str(resolve_repo_artifact_path(__file__, args.adapter_dir)) if args.adapter_dir else "",
        logger=LOGGER,
        context="explicit prediction",
    )

    enriched_rows = []
    counters = Counter()
    for idx, row in enumerate(rows, start=1):
        text = get_text(row)
        if not text:
            raise ValueError(f"Missing canonical text for row {idx}.")
        gliner2_entities = predict_entities_for_text(model, text, entity_types)
        enriched = dict(row)
        enriched["gliner2_entities"] = gliner2_entities
        enriched["_gliner2_prediction_meta"] = {
            "row_index_1based": idx,
            "gliner2_entity_count": len(gliner2_entities),
            "model_path": args.model_path,
            "adapter_dir": args.adapter_dir,
            "entity_types": entity_types,
        }
        enriched_rows.append(enriched)
        counters["records_total"] += 1
        counters["gliner2_entities_total"] += len(gliner2_entities)

    write_jsonl(str(resolve_repo_artifact_path(__file__, args.output_jsonl)), enriched_rows)
    LOGGER.info("Saved GLiNER2 predictions JSONL: %s", args.output_jsonl)

    if args.summary_json:
        summary_path = resolve_repo_artifact_path(__file__, args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(build_summary(enriched_rows, counters, args), indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Saved GLiNER2 prediction summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()
