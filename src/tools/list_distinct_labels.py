#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path


def _parse_jsonl(text):
    rows = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc
    return rows


def read_json_or_jsonl(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
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
    for key in ("entities", "ner", "spans"):
        spans = row.get(key)
        if isinstance(spans, list):
            for span in spans:
                yield span


def collect_labels(rows, field):
    counts = Counter()
    for row in rows:
        for span in iter_spans(row):
            label = span.get(field)
            if isinstance(label, str) and label.strip():
                counts[label] += 1
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Print distinct label values and counts from JSON/JSONL.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file")
    parser.add_argument("--field", default="label", help="Label field name inside each span object")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    counts = collect_labels(rows, args.field)
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if not ordered:
        print("(no labels found)")
        return
    print("label,count")
    for label, count in ordered:
        print(f"{label},{count}")


if __name__ == "__main__":
    main()

