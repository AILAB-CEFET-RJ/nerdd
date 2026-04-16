#!/usr/bin/env python3
"""Select top utility-scored adjudication candidates from a scored JSONL corpus."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import get_text, read_json_or_jsonl, write_jsonl
from tools.review_adjudication_cases import render_adjudication_review
from tools.select_train_annotation_cases import _review_seed_entities, _safe_float


DEFAULT_RANKING_FIELD = "adjudication_priority_score"
DEFAULT_HTML_LAYERS = "baseline_entities,gliner2_entities,review_seed_entities"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select top-K utility-scored candidates from a scored adjudication corpus.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL with utility-scored adjudication candidates.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with the selected top-K rows.")
    parser.add_argument("--output-csv", default="", help="Optional CSV output with compact metadata.")
    parser.add_argument("--output-html", default="", help="Optional HTML output for visual review.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--ranking-field", default=DEFAULT_RANKING_FIELD, help="Numeric field used to rank rows.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of rows to emit.")
    parser.add_argument("--min-score", type=float, default=float("-inf"), help="Optional minimum ranking-field value.")
    parser.add_argument("--title", default="Top Utility Candidates", help="HTML title when --output-html is used.")
    parser.add_argument("--html-layers", default=DEFAULT_HTML_LAYERS, help="Comma-separated layers for HTML review.")
    parser.add_argument("--score-fields", default="ner_score,confidence,score_calibrated,score", help="Comma-separated score fields for HTML/entity lists.")
    return parser.parse_args()


def _parse_csv(raw_value: str) -> list[str]:
    return [piece.strip() for piece in str(raw_value or "").split(",") if piece.strip()]


def _top_level_score(row: dict, key: str):
    return _safe_float(row.get(key), None)


def _sort_rows(rows: list[dict], ranking_field: str, min_score: float) -> list[dict]:
    filtered = []
    for row in rows:
        score = _top_level_score(row, ranking_field)
        if score is None or score < min_score:
            continue
        enriched = dict(row)
        enriched["_selected_utility_score"] = score
        filtered.append(enriched)
    filtered.sort(
        key=lambda row: (
            -float(row["_selected_utility_score"]),
            -float(_safe_float(row.get("adjudication_priority_score"), -1e9)),
            len(_review_seed_entities(row)),
            len(get_text(row)),
        )
    )
    for idx, row in enumerate(filtered, start=1):
        row["_utility_selection"] = {
            "rank": idx,
            "ranking_field": ranking_field,
            "ranking_score": row["_selected_utility_score"],
            "seed_count": len(_review_seed_entities(row)),
            "text_length": len(get_text(row)),
        }
    return filtered


def _write_csv(path: str, rows: list[dict], ranking_field: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "source_id",
        "ranking_field",
        "ranking_score",
        "adjudication_priority_score",
        "novelty_adjusted_priority_score",
        "novelty_pool_adjusted_priority_score",
        "toponym_novelty_ratio",
        "pool_toponym_frequency_score",
        "seed_count",
        "text_length",
        "text_preview",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            meta = row["_utility_selection"]
            writer.writerow(
                {
                    "rank": meta["rank"],
                    "source_id": row.get("source_id", ""),
                    "ranking_field": ranking_field,
                    "ranking_score": meta["ranking_score"],
                    "adjudication_priority_score": row.get("adjudication_priority_score"),
                    "novelty_adjusted_priority_score": row.get("novelty_adjusted_priority_score"),
                    "novelty_pool_adjusted_priority_score": row.get("novelty_pool_adjusted_priority_score"),
                    "toponym_novelty_ratio": row.get("toponym_novelty_ratio"),
                    "pool_toponym_frequency_score": row.get("pool_toponym_frequency_score"),
                    "seed_count": meta["seed_count"],
                    "text_length": meta["text_length"],
                    "text_preview": get_text(row).replace("\n", " ")[:160],
                }
            )


def _build_summary(rows: list[dict], *, input_path: str, ranking_field: str, top_n: int, min_score: float, total_rows: int) -> dict:
    ranking_scores = [float(row["_utility_selection"]["ranking_score"]) for row in rows]
    return {
        "input": str(Path(input_path).resolve()),
        "rows_total": total_rows,
        "rows_selected": len(rows),
        "ranking_field": ranking_field,
        "top_n": top_n,
        "min_score": min_score,
        "selected_summary": {
            "avg_ranking_score": (sum(ranking_scores) / len(ranking_scores)) if ranking_scores else 0.0,
            "max_ranking_score": max(ranking_scores) if ranking_scores else 0.0,
            "min_ranking_score": min(ranking_scores) if ranking_scores else 0.0,
            "avg_seed_count": (sum(len(_review_seed_entities(row)) for row in rows) / len(rows)) if rows else 0.0,
            "avg_text_length": (sum(len(get_text(row)) for row in rows) / len(rows)) if rows else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    ranked = _sort_rows(rows, args.ranking_field, args.min_score)
    if args.top_n > 0:
        ranked = ranked[: args.top_n]

    write_jsonl(args.output_jsonl, ranked)
    print(f"Saved JSONL: {args.output_jsonl}")

    if args.output_csv:
        _write_csv(args.output_csv, ranked, args.ranking_field)
        print(f"Saved CSV: {args.output_csv}")

    if args.output_html:
        render_adjudication_review(
            ranked,
            output_path=args.output_html,
            title=args.title,
            layers=_parse_csv(args.html_layers),
            score_fields=_parse_csv(args.score_fields),
        )
        print(f"Saved HTML: {args.output_html}")

    if args.summary_json:
        summary = _build_summary(
            ranked,
            input_path=args.input,
            ranking_field=args.ranking_field,
            top_n=args.top_n,
            min_score=args.min_score,
            total_rows=len(rows),
        )
        target = Path(args.summary_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved summary JSON: {args.summary_json}")

    print(f"Selected {len(ranked)}/{len(rows)} rows by {args.ranking_field}")


if __name__ == "__main__":
    main()
