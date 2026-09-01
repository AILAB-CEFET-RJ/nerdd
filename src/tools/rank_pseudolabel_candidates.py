#!/usr/bin/env python3
"""Select top-K pseudolabel candidates by a record-level score field."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import get_spans, get_text, read_json_or_jsonl, write_jsonl
from tools.render_ner_html import render_html


def _parse_csv(raw_value: str) -> list[str]:
    return [piece.strip() for piece in str(raw_value or "").split(",") if piece.strip()]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_score(row: dict[str, Any], score_fields: list[str]) -> tuple[float | None, str]:
    for field in score_fields:
        score = _safe_float(row.get(field))
        if score is not None:
            return score, field
    return None, ""


def _label_counts(row: dict[str, Any], label_field: str) -> Counter:
    counts = Counter()
    for entity in get_spans(row):
        label = str(entity.get(label_field, "")).strip()
        if label:
            counts[label] += 1
    return counts


def _has_required_label(row: dict[str, Any], required_labels: set[str], label_field: str) -> bool:
    if not required_labels:
        return True
    counts = _label_counts(row, label_field)
    return any(counts.get(label, 0) > 0 for label in required_labels)


def rank_rows(
    rows: list[dict[str, Any]],
    *,
    score_fields: list[str],
    min_score: float,
    required_labels: set[str],
    label_field: str,
) -> tuple[list[dict[str, Any]], Counter]:
    counters = Counter()
    ranked = []
    for row_index, row in enumerate(rows, start=1):
        counters["rows_total"] += 1
        score, score_field = _pick_score(row, score_fields)
        if score is None:
            counters["dropped_missing_score"] += 1
            continue
        if score < min_score:
            counters["dropped_min_score"] += 1
            continue
        if not _has_required_label(row, required_labels, label_field):
            counters["dropped_required_labels"] += 1
            continue

        selected = dict(row)
        counts = _label_counts(selected, label_field)
        selected["_pseudolabel_selection"] = {
            "source_row_index_1based": row_index,
            "rank": None,
            "score": score,
            "score_field": score_field,
            "text_length": len(get_text(selected)),
            "entity_count": sum(counts.values()),
            "label_counts": dict(counts),
        }
        ranked.append(selected)

    ranked.sort(
        key=lambda row: (
            -row["_pseudolabel_selection"]["score"],
            row["_pseudolabel_selection"]["entity_count"],
            row["_pseudolabel_selection"]["text_length"],
            row["_pseudolabel_selection"]["source_row_index_1based"],
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["_pseudolabel_selection"]["rank"] = rank

    counters["rows_after_filters"] = len(ranked)
    return ranked, counters


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "source_row_index_1based",
        "score",
        "score_field",
        "entity_count",
        "location_count",
        "organization_count",
        "person_count",
        "text_length",
        "text_preview",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            meta = row["_pseudolabel_selection"]
            counts = meta["label_counts"]
            writer.writerow(
                {
                    "rank": meta["rank"],
                    "source_row_index_1based": meta["source_row_index_1based"],
                    "score": f"{meta['score']:.8f}",
                    "score_field": meta["score_field"],
                    "entity_count": meta["entity_count"],
                    "location_count": counts.get("Location", 0),
                    "organization_count": counts.get("Organization", 0),
                    "person_count": counts.get("Person", 0),
                    "text_length": meta["text_length"],
                    "text_preview": get_text(row).replace("\n", " ")[:180],
                }
            )


def build_summary(
    *,
    rows_total: int,
    ranked_rows: list[dict[str, Any]],
    emitted_rows: list[dict[str, Any]],
    counters: Counter,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_scores = [row["_pseudolabel_selection"]["score"] for row in emitted_rows]
    label_counts = Counter()
    for row in emitted_rows:
        label_counts.update(row["_pseudolabel_selection"]["label_counts"])

    return {
        "input": str(Path(args.input).resolve()),
        "outputs": {
            "jsonl": args.output_jsonl or None,
            "csv": args.output_csv or None,
            "html": args.output_html or None,
        },
        "config": {
            "score_fields": _parse_csv(args.score_fields),
            "min_score": args.min_score,
            "top_n": args.top_n,
            "required_labels": _parse_csv(args.required_labels),
            "label_field": args.label_field,
        },
        "summary": {
            "rows_total": rows_total,
            "rows_after_filters": len(ranked_rows),
            "rows_emitted": len(emitted_rows),
            "dropped_missing_score": int(counters["dropped_missing_score"]),
            "dropped_min_score": int(counters["dropped_min_score"]),
            "dropped_required_labels": int(counters["dropped_required_labels"]),
            "selected_score_mean": (sum(selected_scores) / len(selected_scores)) if selected_scores else None,
            "selected_score_min": min(selected_scores) if selected_scores else None,
            "selected_score_max": max(selected_scores) if selected_scores else None,
            "selected_label_counts": dict(sorted(label_counts.items())),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select top-K pseudolabel candidates by record-level score.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL with scored pseudolabel candidates.")
    parser.add_argument("--output-jsonl", default="", help="Optional JSONL output with selected rows.")
    parser.add_argument("--output-csv", default="", help="Optional CSV output with compact selected-row metadata.")
    parser.add_argument("--output-html", default="", help="Optional HTML output for visual review.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--title", default="Top Pseudolabel Candidates", help="HTML title when --output-html is used.")
    parser.add_argument(
        "--score-fields",
        default="record_score,record_score_location,record_score_context_boosted,score_relato",
        help="Comma-separated record-level score fields to try in order.",
    )
    parser.add_argument("--min-score", type=float, default=float("-inf"), help="Minimum score required.")
    parser.add_argument("--top-n", type=int, default=100, help="Maximum rows to emit. Use 0 for all rows after filters.")
    parser.add_argument("--required-labels", default="", help="Optional comma-separated labels required in each row.")
    parser.add_argument("--label-field", default="label", help="Entity label field.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    ranked_rows, counters = rank_rows(
        rows,
        score_fields=_parse_csv(args.score_fields),
        min_score=args.min_score,
        required_labels=set(_parse_csv(args.required_labels)),
        label_field=args.label_field,
    )
    emitted_rows = ranked_rows if args.top_n <= 0 else ranked_rows[: args.top_n]

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, emitted_rows)
        print(f"Saved JSONL: {args.output_jsonl}")

    if args.output_csv:
        write_csv(args.output_csv, emitted_rows)
        print(f"Saved CSV: {args.output_csv}")

    if args.output_html:
        render_html(emitted_rows, output_path=args.output_html, title=args.title, max_reports=0)
        print(f"Saved HTML: {args.output_html}")

    summary = build_summary(
        rows_total=len(rows),
        ranked_rows=ranked_rows,
        emitted_rows=emitted_rows,
        counters=counters,
        args=args,
    )
    if args.summary_json:
        target = Path(args.summary_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved summary JSON: {args.summary_json}")

    print(f"Rows after filters: {summary['summary']['rows_after_filters']}/{summary['summary']['rows_total']}")
    print(f"Rows emitted: {summary['summary']['rows_emitted']}")
    if summary["summary"]["selected_score_min"] is not None:
        print(
            "Selected score range: "
            f"{summary['summary']['selected_score_min']:.6f} - "
            f"{summary['summary']['selected_score_max']:.6f}"
        )


if __name__ == "__main__":
    main()
