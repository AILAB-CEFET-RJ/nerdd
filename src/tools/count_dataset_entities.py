#!/usr/bin/env python3
"""Count rows and entity-label frequencies in JSON and JSONL corpora."""

import argparse
import json
from collections import Counter
from pathlib import Path


def _parse_jsonl(text):
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def read_json_or_jsonl(path):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    text = source.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _parse_jsonl(text)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Unsupported JSON format.")


def iter_spans(row):
    for key in ("spans", "entities", "ner"):
        spans = row.get(key)
        if isinstance(spans, list):
            yield from spans
            return


def count_rows_and_entities(rows, label_field):
    label_counts = Counter()
    total_entities = 0

    for row in rows:
        spans = list(iter_spans(row))
        total_entities += len(spans)
        for span in spans:
            label = span.get(label_field, "UNKNOWN")
            label_counts[str(label)] += 1

    return {
        "rows": len(rows),
        "entities": total_entities,
        "labels": label_counts,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count rows and entity labels in JSON/JSONL corpora."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more input JSON/JSONL files.",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        help="Label field name inside each span/entity object.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for raw_path in args.inputs:
        rows = read_json_or_jsonl(raw_path)
        stats = count_rows_and_entities(rows, label_field=args.label_field)

        print(raw_path)
        print(f"  relatos: {stats['rows']}")
        print(f"  entidades: {stats['entities']}")
        for label, count in sorted(stats["labels"].items()):
            print(f"  {label}: {count}")
        print()


if __name__ == "__main__":
    main()
