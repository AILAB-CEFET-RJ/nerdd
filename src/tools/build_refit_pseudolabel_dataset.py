#!/usr/bin/env python3
"""Project LLM adjudication outputs into a refit-ready pseudolabel JSONL.

This utility converts the output of `run_llm_adjudication.py` into the minimal
record format consumed by `pseudolabelling.refit_model`:

- `text`
- `entities` with `text`, `label`, `start`, `end`

Only rows with configured adjudication decisions are kept. The output is meant
to be passed through `--pseudolabel-path`, while the supervised corpus remains
separate and is supplied through `--supervised-train-path`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)

DEFAULT_ALLOWED_DECISIONS = ("accept", "accept_with_edits")
DEFAULT_ALLOWED_LABELS = ("Person", "Location", "Organization")


def _parse_csv(raw_value: str) -> list[str]:
    values = [piece.strip() for piece in str(raw_value or "").split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one CSV value must be provided.")
    return values


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_text(row: dict) -> str:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text
    source = row.get("_source")
    if isinstance(source, dict):
        nested = source.get("text")
        if isinstance(nested, str) and nested.strip():
            return nested
    raise ValueError("Missing canonical text field in adjudication row.")


def _normalize_entity(entity: dict, *, text: str, allowed_labels: set[str]) -> dict:
    if not isinstance(entity, dict):
        raise ValueError("Entity must be an object.")

    entity_text = str(entity.get("text", "")).strip()
    label = str(entity.get("label", "")).strip()
    start = _safe_int(entity.get("start"))
    end = _safe_int(entity.get("end"))

    if not entity_text:
        raise ValueError("Entity text is required.")
    if label not in allowed_labels:
        raise ValueError(f"Unsupported entity label: {label!r}")
    if start is None or end is None or end <= start:
        raise ValueError(f"Invalid entity offsets for {entity_text!r}: start={start} end={end}")
    if end > len(text):
        raise ValueError(f"Entity offsets out of bounds for {entity_text!r}: end={end} len(text)={len(text)}")
    span_text = text[start:end]
    if span_text != entity_text:
        raise ValueError(
            f"Entity text/offset mismatch for {entity_text!r}: text[{start}:{end}]={span_text!r}"
        )

    return {
        "text": entity_text,
        "label": label,
        "start": start,
        "end": end,
    }


def _project_record(
    row: dict,
    *,
    allowed_decisions: set[str],
    allowed_labels: set[str],
    include_source_payload: bool,
) -> tuple[dict | None, Counter]:
    counters = Counter()

    adjudication = row.get("adjudication")
    if not isinstance(adjudication, dict):
        counters["dropped_missing_adjudication"] += 1
        return None, counters

    decision = str(adjudication.get("decision", "")).strip()
    counters[f"decision_{decision or 'missing'}"] += 1
    if decision not in allowed_decisions:
        counters["dropped_by_decision"] += 1
        return None, counters

    text = _require_text(row)
    entities_final = adjudication.get("entities_final")
    if not isinstance(entities_final, list):
        raise ValueError("adjudication.entities_final must be a list.")
    if not entities_final:
        counters["dropped_empty_entities"] += 1
        return None, counters

    entities = [_normalize_entity(entity, text=text, allowed_labels=allowed_labels) for entity in entities_final]

    projected = {
        "source_id": row.get("source_id"),
        "text": text,
        "entities": entities,
        "_pseudolabel_meta": {
            "decision": decision,
            "review_confidence": adjudication.get("review_confidence", ""),
            "model": row.get("model", ""),
        },
    }
    if include_source_payload and isinstance(row.get("_source"), dict):
        projected["_source"] = row["_source"]

    counters["kept_records"] += 1
    counters["kept_entities_total"] += len(entities)
    return projected, counters


def build_refit_pseudolabel_dataset(
    rows: list[dict],
    *,
    allowed_decisions: set[str],
    allowed_labels: set[str],
    include_source_payload: bool,
    top_n: int = 0,
) -> tuple[list[dict], dict]:
    if top_n < 0:
        raise ValueError("--top-n must be >= 0.")

    selected_rows = rows[:top_n] if top_n else rows
    emitted = []
    counters = Counter(
        {
            "input_records": len(rows),
            "selected_input_records": len(selected_rows),
        }
    )
    label_counts = Counter()

    for index, row in enumerate(selected_rows, start=1):
        try:
            projected, row_counts = _project_record(
                row,
                allowed_decisions=allowed_decisions,
                allowed_labels=allowed_labels,
                include_source_payload=include_source_payload,
            )
        except Exception as exc:
            source_id = row.get("source_id", f"row_{index}")
            raise RuntimeError(f"Failed to project adjudication row {source_id!r}: {exc}") from exc
        counters.update(row_counts)
        if projected is None:
            continue
        emitted.append(projected)
        label_counts.update(entity["label"] for entity in projected["entities"])

    summary = {
        "records_total": len(rows),
        "records_selected": len(selected_rows),
        "records_emitted": len(emitted),
        "top_n": top_n,
        "allowed_decisions": sorted(allowed_decisions),
        "allowed_labels": sorted(allowed_labels),
        "summary": {
            **dict(counters),
            "label_counts": dict(label_counts),
            "avg_entities_per_record": (sum(len(row["entities"]) for row in emitted) / len(emitted)) if emitted else 0.0,
        },
    }
    return emitted, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LLM adjudication outputs into a refit-ready pseudolabel dataset."
    )
    parser.add_argument("--input", required=True, help="Input adjudication JSONL from run_llm_adjudication.py.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path with text + entities records.")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary JSON path with counts and label distribution.",
    )
    parser.add_argument(
        "--allowed-decisions",
        default=",".join(DEFAULT_ALLOWED_DECISIONS),
        help="Comma-separated adjudication decisions to keep.",
    )
    parser.add_argument(
        "--allowed-labels",
        default=",".join(DEFAULT_ALLOWED_LABELS),
        help="Comma-separated label whitelist for emitted entities.",
    )
    parser.add_argument(
        "--include-source-payload",
        action="store_true",
        help="Preserve the original _source payload in each emitted record for auditability.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="If > 0, keep only the first N adjudicated rows from the input before projection.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    allowed_decisions = set(_parse_csv(args.allowed_decisions))
    allowed_labels = set(_parse_csv(args.allowed_labels))

    rows = read_json_or_jsonl(args.input)
    emitted, summary = build_refit_pseudolabel_dataset(
        rows,
        allowed_decisions=allowed_decisions,
        allowed_labels=allowed_labels,
        include_source_payload=args.include_source_payload,
        top_n=args.top_n,
    )

    write_jsonl(args.output_jsonl, emitted)
    LOGGER.info("Saved refit pseudolabel dataset: %s", args.output_jsonl)

    summary = {
        "input": str(Path(args.input).resolve()),
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
        **summary,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Saved summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()
