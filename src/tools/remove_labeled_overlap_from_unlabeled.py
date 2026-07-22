#!/usr/bin/env python3
"""Remove labeled train/test/calibration overlaps from an unlabeled JSONL pool.

This is stricter than the original large-corpus sanitization: it can ignore
punctuation and accents so tokenization differences do not leak labeled records
into the pseudolabelling candidate pool.

Example:
python3 src/tools/remove_labeled_overlap_from_unlabeled.py \
  --input-jsonl data/large_sanitized/large_sanitized.jsonl \
  --labeled-input data/dd_corpus_small_train.json \
  --labeled-input data/dd_corpus_small_test.json \
  --labeled-input data/dd_corpus_small_calibration.json \
  --output-jsonl data/large_sanitized/large_sanitized_no_labeled_overlap.jsonl \
  --removed-jsonl data/large_sanitized/large_sanitized_labeled_overlap_removed.jsonl \
  --summary-json data/large_sanitized/large_sanitized_no_labeled_overlap_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inventory_labeled_source_assuntos import (  # noqa: E402
    MATCH_STRATEGIES,
    get_text,
    match_key,
    read_json_or_jsonl,
)


DEFAULT_INPUT = "data/large_sanitized/large_sanitized.jsonl"


def parse_strategy_list(value: str) -> list[str]:
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    if not strategies:
        raise ValueError("At least one match strategy is required.")
    invalid = [strategy for strategy in strategies if strategy not in MATCH_STRATEGIES]
    if invalid:
        raise ValueError(f"Unsupported match strategies: {invalid}. Valid: {MATCH_STRATEGIES}")
    return strategies


def build_labeled_key_index(
    labeled_inputs: list[str],
    *,
    labeled_text_field: str,
    strategies: list[str],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, int]]:
    index: dict[str, dict[str, list[dict[str, Any]]]] = {
        strategy: defaultdict(list)
        for strategy in strategies
    }
    row_counts: dict[str, int] = {}

    for labeled_input in labeled_inputs:
        rows = read_json_or_jsonl(labeled_input)
        row_counts[labeled_input] = len(rows)
        for idx, row in enumerate(rows):
            text = get_text(row, labeled_text_field)
            if not text:
                continue
            ref = {
                "labeled_input": labeled_input,
                "labeled_row_index_0based": idx,
                "labeled_row_index_1based": idx + 1,
            }
            for strategy in strategies:
                key = match_key(text, strategy)
                if key:
                    index[strategy][key].append(ref)
    return index, row_counts


def first_overlap(
    text: str,
    key_index: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    strategies: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    for strategy in strategies:
        key = match_key(text, strategy)
        if not key:
            continue
        matches = key_index[strategy].get(key, [])
        if matches:
            return strategy, matches
    return "", []


def iter_jsonl(path: str | Path):
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {source} at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL in {source} at line {line_no}: expected object.")
            yield line_no, row


def open_jsonl_writer(path: str | Path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("w", encoding="utf-8")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def filter_unlabeled_pool(
    *,
    input_jsonl: str,
    output_jsonl: str,
    removed_jsonl: str,
    text_field: str,
    key_index: dict[str, dict[str, list[dict[str, Any]]]],
    strategies: list[str],
    preview_limit: int,
) -> dict[str, Any]:
    rows_in = 0
    rows_kept = 0
    rows_removed = 0
    removed_by_strategy: Counter = Counter()
    removed_by_labeled_input: Counter = Counter()
    removed_match_count: Counter = Counter()
    removed_preview: list[dict[str, Any]] = []

    removed_handle = open_jsonl_writer(removed_jsonl) if removed_jsonl else None
    try:
        with open_jsonl_writer(output_jsonl) as output_handle:
            for line_no, row in iter_jsonl(input_jsonl):
                rows_in += 1
                text = get_text(row, text_field)
                strategy, matches = first_overlap(text, key_index, strategies=strategies)
                if not matches:
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_kept += 1
                    continue

                rows_removed += 1
                removed_by_strategy[strategy] += 1
                removed_match_count[len(matches)] += 1
                for match in matches:
                    removed_by_labeled_input[str(match["labeled_input"])] += 1

                if len(removed_preview) < preview_limit:
                    removed_preview.append(
                        {
                            "input_row_index_1based": line_no,
                            "match_strategy": strategy,
                            "match_count": len(matches),
                            "matched_labeled_refs_preview": [
                                f"{match['labeled_input']}:{match['labeled_row_index_1based']}"
                                for match in matches[:10]
                            ],
                            "assunto": row.get("assunto", ""),
                            "relato_preview": text[:200],
                        }
                    )

                if removed_handle is not None:
                    removed = dict(row)
                    removed["_labeled_overlap_removal"] = {
                        "input_row_index_1based": line_no,
                        "match_strategy": strategy,
                        "match_count": len(matches),
                        "matched_labeled_refs_preview": [
                            {
                                "labeled_input": match["labeled_input"],
                                "labeled_row_index_1based": match["labeled_row_index_1based"],
                            }
                            for match in matches[:20]
                        ],
                    }
                    removed_handle.write(json.dumps(removed, ensure_ascii=False) + "\n")
    finally:
        if removed_handle is not None:
            removed_handle.close()

    return {
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "output_jsonl": str(Path(output_jsonl).resolve()),
        "removed_jsonl": str(Path(removed_jsonl).resolve()) if removed_jsonl else "",
        "rows_in": rows_in,
        "rows_kept": rows_kept,
        "rows_removed": rows_removed,
        "removal_rate": rows_removed / rows_in if rows_in else 0.0,
        "match_strategies": strategies,
        "removed_by_strategy": dict(removed_by_strategy),
        "removed_by_labeled_input_match_refs": dict(removed_by_labeled_input),
        "removed_match_count": {str(key): value for key, value in sorted(removed_match_count.items())},
        "removed_preview": removed_preview,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove labeled corpus overlaps from an unlabeled JSONL pool."
    )
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT, help="Input unlabeled JSONL pool.")
    parser.add_argument("--output-jsonl", required=True, help="Output filtered JSONL pool.")
    parser.add_argument(
        "--removed-jsonl",
        default="",
        help="Optional JSONL audit file with removed overlapping rows.",
    )
    parser.add_argument(
        "--labeled-input",
        action="append",
        required=True,
        help="Labeled JSON/JSONL corpus to exclude from the pool. Repeat for train/test/calibration.",
    )
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--text-field", default="auto", help="Input pool text field, or 'auto'.")
    parser.add_argument("--labeled-text-field", default="auto", help="Labeled text field, or 'auto'.")
    parser.add_argument(
        "--match-strategies",
        default=",".join(MATCH_STRATEGIES),
        help=f"Comma-separated strategies in priority order. Valid: {','.join(MATCH_STRATEGIES)}",
    )
    parser.add_argument("--preview-limit", type=int, default=20, help="Number of removed examples in summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = parse_strategy_list(args.match_strategies)
    key_index, labeled_row_counts = build_labeled_key_index(
        args.labeled_input,
        labeled_text_field=args.labeled_text_field,
        strategies=strategies,
    )
    summary = filter_unlabeled_pool(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        removed_jsonl=args.removed_jsonl,
        text_field=args.text_field,
        key_index=key_index,
        strategies=strategies,
        preview_limit=args.preview_limit,
    )
    summary["labeled_inputs"] = {
        "row_counts": labeled_row_counts,
        "total_rows": sum(labeled_row_counts.values()),
    }

    if args.summary_json:
        write_json(args.summary_json, summary)

    print(f"Input rows: {summary['rows_in']}")
    print(f"Kept rows: {summary['rows_kept']}")
    print(f"Removed overlaps: {summary['rows_removed']}")
    print(f"Output JSONL: {args.output_jsonl}")
    if args.removed_jsonl:
        print(f"Removed JSONL: {args.removed_jsonl}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
