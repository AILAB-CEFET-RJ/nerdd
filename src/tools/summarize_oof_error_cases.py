#!/usr/bin/env python3
"""Summarize and filter OOF error cases exported by mine_train_oof_errors.py."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl
from base_model_training.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OOF error cases by label and error tag.")
    parser.add_argument("--input", required=True, help="Input OOF error cases JSONL.")
    parser.add_argument("--output-jsonl", default="", help="Optional filtered JSONL output.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--require-tag", action="append", default=[], help="Keep only rows containing this error tag. Repeatable.")
    parser.add_argument("--require-label", action="append", default=[], help="Keep only rows touching this label in gold or pred spans. Repeatable.")
    parser.add_argument("--sample-limit", type=int, default=10, help="Number of example rows to include in the summary.")
    return parser.parse_args()


def _labels_in_row(row: dict) -> set[str]:
    labels = set()
    for field in ("gold_spans", "pred_spans", "fn_spans", "fp_spans"):
        for span in row.get(field, []) or []:
            label = span.get("label")
            if label:
                labels.add(str(label))
    return labels


def _row_matches(row: dict, required_tags: set[str], required_labels: set[str]) -> bool:
    row_tags = {str(tag) for tag in row.get("error_tags", [])}
    row_labels = _labels_in_row(row)
    if required_tags and not required_tags.issubset(row_tags):
        return False
    if required_labels and row_labels.isdisjoint(required_labels):
        return False
    return True


def _summarize(rows: list[dict], sample_limit: int) -> dict:
    tag_counts = Counter()
    label_counts = Counter()
    tag_label_counts = Counter()
    samples = []

    for row in rows:
        tags = [str(tag) for tag in row.get("error_tags", [])]
        labels = sorted(_labels_in_row(row))
        tag_counts.update(tags)
        label_counts.update(labels)
        for tag in tags:
            for label in labels:
                tag_label_counts[(tag, label)] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "error_tags": tags,
                    "labels": labels,
                    "text": row.get("text", "")[:280],
                }
            )

    return {
        "rows": len(rows),
        "error_tag_counts": dict(tag_counts),
        "label_counts": dict(label_counts),
        "tag_label_counts": {
            f"{tag}::{label}": count for (tag, label), count in sorted(tag_label_counts.items())
        },
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = resolve_path(script_dir, args.input)
    output_jsonl = resolve_path(script_dir, args.output_jsonl) if args.output_jsonl else None
    summary_json = resolve_path(script_dir, args.summary_json) if args.summary_json else None

    rows = list(load_jsonl(str(input_path)))
    required_tags = {str(tag) for tag in args.require_tag}
    required_labels = {str(label) for label in args.require_label}
    filtered = [row for row in rows if _row_matches(row, required_tags, required_labels)]

    if output_jsonl:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for row in filtered:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(input_path),
        "rows_total": len(rows),
        "rows_filtered": len(filtered),
        "required_tags": sorted(required_tags),
        "required_labels": sorted(required_labels),
        "filtered_summary": _summarize(filtered, args.sample_limit),
    }

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
