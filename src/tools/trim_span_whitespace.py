#!/usr/bin/env python3
"""Trim leading/trailing whitespace from NER span boundaries in JSON/JSONL corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TEXT_FIELDS = ("text", "relato", "texto", "description", "descricao")
SPAN_FIELDS = ("spans", "entities", "ner")


def parse_jsonl(text: str, path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Invalid JSONL in {path} at line {line_no}: expected object.")
        rows.append(row)
    return rows


def read_json_or_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path)
    payload = source.read_text(encoding="utf-8")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return parse_jsonl(payload, source), "jsonl"

    if isinstance(parsed, list):
        if not all(isinstance(row, dict) for row in parsed):
            raise ValueError(f"JSON list input must contain only objects: {source}")
        return parsed, "json"
    if isinstance(parsed, dict):
        return [parsed], "json"
    raise ValueError(f"Unsupported input format: {source}")


def get_text(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            return value
    return ""


def iter_span_lists(row: dict[str, Any]):
    for field in SPAN_FIELDS:
        spans = row.get(field)
        if isinstance(spans, list):
            yield field, spans


def trim_span(text: str, span: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    start = span.get("start")
    end = span.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return dict(span), False, False
    if start < 0 or end > len(text) or end <= start:
        return dict(span), False, False

    new_start = start
    new_end = end
    while new_start < new_end and text[new_start].isspace():
        new_start += 1
    while new_end > new_start and text[new_end - 1].isspace():
        new_end -= 1

    if new_start == start and new_end == end:
        return dict(span), False, False
    if new_end <= new_start:
        return dict(span), False, True

    trimmed = dict(span)
    trimmed["start"] = new_start
    trimmed["end"] = new_end
    if "text" in trimmed:
        trimmed["text"] = text[new_start:new_end]
    return trimmed, True, False


def make_change_record(
    row_idx: int,
    span_idx: int,
    span_field: str,
    label: str,
    before_start: int,
    before_end: int,
    after_start: int | None,
    after_end: int | None,
    before_text: str,
    after_text: str,
    action: str,
) -> dict[str, Any]:
    return {
        "row_index_1based": row_idx,
        "span_index_0based": span_idx,
        "span_field": span_field,
        "label": label,
        "action": action,
        "before_start": before_start,
        "before_end": before_end,
        "after_start": after_start,
        "after_end": after_end,
        "before": before_text,
        "after": after_text,
    }


def transform_rows(
    rows: list[dict[str, Any]], *, include_changes: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transformed = []
    stats = Counter()
    examples = []
    changes = []

    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            transformed.append(row)
            continue
        text = get_text(row)
        new_row = dict(row)
        row_changed = False
        for span_field, spans in iter_span_lists(row):
            new_spans = []
            for span_idx, span in enumerate(spans):
                if not isinstance(span, dict):
                    new_spans.append(span)
                    continue
                new_span, changed, became_empty = trim_span(text, span)
                if became_empty:
                    stats["dropped_empty_after_trim"] += 1
                    label = str(span.get("label", ""))
                    change = make_change_record(
                        row_idx,
                        span_idx,
                        span_field,
                        label,
                        span["start"],
                        span["end"],
                        None,
                        None,
                        text[span["start"] : span["end"]],
                        "",
                        "drop_empty_after_trim",
                    )
                    if len(examples) < 20:
                        examples.append(change)
                    if include_changes:
                        changes.append(change)
                    continue
                new_spans.append(new_span)
                if changed:
                    row_changed = True
                    stats["trimmed_spans"] += 1
                    label = str(span.get("label", ""))
                    if label:
                        stats[f"trimmed_label::{label}"] += 1
                    change = make_change_record(
                        row_idx,
                        span_idx,
                        span_field,
                        label,
                        span["start"],
                        span["end"],
                        new_span["start"],
                        new_span["end"],
                        text[span["start"] : span["end"]],
                        text[new_span["start"] : new_span["end"]],
                        "trim",
                    )
                    if len(examples) < 20:
                        examples.append(change)
                    if include_changes:
                        changes.append(change)
            new_row[span_field] = new_spans
        if row_changed:
            stats["rows_changed"] += 1
        transformed.append(new_row)

    stats["rows_total"] = len(rows)
    summary = {**dict(stats), "examples": examples}
    if include_changes:
        summary["changes"] = changes
    return transformed, summary


def write_rows(path: str | Path, rows: list[dict[str, Any]], fmt: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    target.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim leading/trailing whitespace from span start/end offsets."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL corpus.")
    parser.add_argument("--output", default="", help="Output path. Required unless --inplace or --dry-run.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write transformed corpus.")
    parser.add_argument("--verbose", action="store_true", help="Include every span change in the report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inplace and args.output:
        raise ValueError("Use either --inplace or --output, not both.")

    rows, fmt = read_json_or_jsonl(args.input)
    transformed, summary = transform_rows(rows, include_changes=args.verbose)
    summary["input"] = str(Path(args.input).resolve())
    output_path = args.input if args.inplace else args.output
    summary["output"] = str(Path(output_path).resolve()) if output_path else ""

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_json:
        write_json(args.summary_json, summary)
    if args.dry_run:
        return
    if not output_path:
        raise ValueError("Provide --output, or use --inplace / --dry-run.")
    write_rows(output_path, transformed, fmt)


if __name__ == "__main__":
    main()
