#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.render_ner_html import render_html


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


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_spans(row):
    for key in ("entities", "spans", "ner"):
        spans = row.get(key)
        if isinstance(spans, list):
            return spans
    return []


def get_text(row):
    for key in ("relato", "text", "texto", "description", "descricao"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def count_labels(spans, label_field):
    counts = Counter()
    for span in spans:
        counts[str(span.get(label_field, "UNKNOWN"))] += 1
    return counts


def summarize_rows(rows, label_field):
    label_counts = Counter()
    entity_counts = []
    for row in rows:
        spans = get_spans(row)
        entity_counts.append(len(spans))
        label_counts.update(count_labels(spans, label_field))

    return {
        "tips": len(rows),
        "avg_entities_per_tip": (sum(entity_counts) / len(entity_counts)) if entity_counts else 0.0,
        "max_entities_per_tip": max(entity_counts) if entity_counts else 0,
        "label_counts": dict(label_counts),
    }


def annotate_rows(rows, label_field):
    annotated = []
    for idx, row in enumerate(rows, start=1):
        spans = get_spans(row)
        label_counts = count_labels(spans, label_field)
        enriched = dict(row)
        enriched["_dense_tip"] = {
            "row_index_1based": idx,
            "entity_count": len(spans),
            "label_counts": dict(label_counts),
            "text_length": len(get_text(row)),
        }
        annotated.append(enriched)
    return annotated


def filter_dense_rows(rows, min_entities, label_field):
    annotated = annotate_rows(rows, label_field=label_field)
    kept = [row for row in annotated if row["_dense_tip"]["entity_count"] >= min_entities]
    kept.sort(
        key=lambda row: (
            -row["_dense_tip"]["entity_count"],
            -row["_dense_tip"]["text_length"],
        )
    )
    return kept


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and inspect tips with many entities."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file.")
    parser.add_argument(
        "--min-entities",
        type=int,
        required=True,
        help="Minimum number of entities for a tip to be selected.",
    )
    parser.add_argument(
        "--max-tips",
        type=int,
        default=0,
        help="Optional maximum number of selected tips to keep in outputs (0 = all).",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        help="Label field name inside each entity/span object.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="Optional JSONL output with selected reports.",
    )
    parser.add_argument(
        "--output-html",
        default="",
        help="Optional HTML output for visual inspection.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary JSON output.",
    )
    parser.add_argument(
        "--title",
        default="Dense Tips Viewer",
        help="HTML title when --output-html is used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    dense_rows = filter_dense_rows(rows, min_entities=args.min_entities, label_field=args.label_field)

    if args.max_tips > 0:
        dense_rows = dense_rows[: args.max_tips]

    summary = summarize_rows(dense_rows, label_field=args.label_field)
    summary["input"] = str(Path(args.input).resolve())
    summary["min_entities"] = args.min_entities

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, dense_rows)
        print(f"Saved JSONL: {args.output_jsonl}")

    if args.output_html:
        render_html(dense_rows, output_path=args.output_html, title=args.title, max_reports=0)
        print(f"Saved HTML: {args.output_html}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved summary JSON: {args.summary_json}")

    print(f"Selected tips: {summary['tips']}")
    print(f"Average entities/tip: {summary['avg_entities_per_tip']:.2f}")
    print(f"Max entities/tip: {summary['max_entities_per_tip']}")
    for label, count in sorted(summary["label_counts"].items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
