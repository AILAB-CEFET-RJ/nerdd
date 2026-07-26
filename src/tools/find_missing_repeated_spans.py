#!/usr/bin/env python3
"""Find exact repeated entity mentions that may be missing annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEXT_FIELDS = ("text", "relato", "texto", "description", "descricao")
SPAN_FIELDS = ("spans", "entities", "ner")
DEFAULT_CONTEXT_CHARS = 80


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


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    payload = source.read_text(encoding="utf-8")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return parse_jsonl(payload, source)

    if isinstance(parsed, list):
        if not all(isinstance(row, dict) for row in parsed):
            raise ValueError(f"JSON list input must contain only objects: {source}")
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Unsupported input format: {source}")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_text(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            return value
    return ""


def get_spans(row: dict[str, Any]) -> list[Any]:
    for field in SPAN_FIELDS:
        spans = row.get(field)
        if isinstance(spans, list):
            return spans
    return []


def parse_labels(value: str) -> set[str]:
    return {label.strip() for label in value.split(",") if label.strip()}


def span_key(mention: str, label: str, *, ignore_case: bool) -> tuple[str, str]:
    if ignore_case:
        return mention.casefold(), label
    return mention, label


def find_all_exact(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    out = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            return out
        out.append((idx, idx + len(needle)))
        start = idx + 1


def has_valid_boundaries(text: str, start: int, end: int, needle: str) -> bool:
    if needle[0].isalnum() and start > 0 and text[start - 1].isalnum():
        return False
    if needle[-1].isalnum() and end < len(text) and text[end].isalnum():
        return False
    return True


def make_context(text: str, start: int, end: int, context_chars: int) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return f"{prefix}{text[left:right]}{suffix}"


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def valid_span(text: str, span: dict[str, Any]) -> bool:
    start = span.get("start")
    end = span.get("end")
    label = span.get("label")
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and isinstance(label, str)
        and bool(label.strip())
        and 0 <= start < end <= len(text)
    )


def covered_by_same_label(
    start: int,
    end: int,
    label: str,
    intervals_by_label: dict[str, list[tuple[int, int]]],
) -> bool:
    return any(
        annotated_start <= start and end <= annotated_end
        for annotated_start, annotated_end in intervals_by_label.get(label, [])
    )


def overlaps_same_label(
    start: int,
    end: int,
    label: str,
    intervals_by_label: dict[str, list[tuple[int, int]]],
) -> bool:
    return any(
        max(start, annotated_start) < min(end, annotated_end)
        for annotated_start, annotated_end in intervals_by_label.get(label, [])
    )


def overlaps_any_span(
    start: int,
    end: int,
    intervals: list[tuple[int, int]],
) -> bool:
    return any(
        max(start, annotated_start) < min(end, annotated_end)
        for annotated_start, annotated_end in intervals
    )


def build_mention_inventory(
    rows: list[dict[str, Any]],
    *,
    labels: set[str] | None = None,
    ignore_case: bool = False,
    min_len: int = 1,
) -> dict[tuple[str, str], dict[str, Any]]:
    inventory: dict[tuple[str, str], dict[str, Any]] = {}

    for row_idx, row in enumerate(rows):
        text = get_text(row)
        for span_idx, span in enumerate(get_spans(row)):
            if not isinstance(span, dict) or not valid_span(text, span):
                continue
            label = span["label"].strip()
            if labels is not None and label not in labels:
                continue
            mention = text[span["start"] : span["end"]]
            if len(mention) < min_len:
                continue
            key = span_key(mention, label, ignore_case=ignore_case)
            if key not in inventory:
                inventory[key] = {
                    "mention": mention,
                    "label": label,
                    "annotated_count": 0,
                    "examples": [],
                }
            item = inventory[key]
            item["annotated_count"] += 1
            if len(item["examples"]) < 5:
                item["examples"].append(
                    {
                        "row_index_0based": row_idx,
                        "row_index_1based": row_idx + 1,
                        "span_index_0based": span_idx,
                        "start": span["start"],
                        "end": span["end"],
                        "text_sha1": text_sha1(text),
                    }
                )

    return inventory


def find_missing_repeated_spans(
    rows: list[dict[str, Any]],
    *,
    labels: set[str] | None = None,
    ignore_case: bool = False,
    min_len: int = 1,
    min_annotated_count: int = 1,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> dict[str, Any]:
    inventory = build_mention_inventory(
        rows,
        labels=labels,
        ignore_case=ignore_case,
        min_len=min_len,
    )
    inventory = {
        key: value
        for key, value in inventory.items()
        if value["annotated_count"] >= min_annotated_count
    }

    candidates = []
    conflicts = []
    label_counts = Counter()
    conflict_label_counts = Counter()

    for row_idx, row in enumerate(rows):
        text = get_text(row)
        search_text = text.casefold() if ignore_case else text
        annotated_by_label = set()
        annotated_by_offsets: dict[tuple[int, int], set[str]] = defaultdict(set)
        intervals_by_label: dict[str, list[tuple[int, int]]] = defaultdict(list)
        annotated_intervals = []

        for span in get_spans(row):
            if not isinstance(span, dict) or not valid_span(text, span):
                continue
            label = span["label"].strip()
            annotated_by_label.add((span["start"], span["end"], label))
            annotated_by_offsets[(span["start"], span["end"])].add(label)
            intervals_by_label[label].append((span["start"], span["end"]))
            annotated_intervals.append((span["start"], span["end"]))

        row_hash = text_sha1(text)
        text_preview = text[:160]
        for _, item in inventory.items():
            mention = item["mention"]
            label = item["label"]
            needle = mention.casefold() if ignore_case else mention
            for start, end in find_all_exact(search_text, needle):
                if not has_valid_boundaries(search_text, start, end, needle):
                    continue
                if (
                    (start, end, label) in annotated_by_label
                    or covered_by_same_label(start, end, label, intervals_by_label)
                    or overlaps_same_label(start, end, label, intervals_by_label)
                ):
                    continue
                occurrence = text[start:end]
                if (start, end) in annotated_by_offsets:
                    existing_labels = sorted(annotated_by_offsets[(start, end)])
                    conflicts.append(
                        {
                            "row_index_0based": row_idx,
                            "row_index_1based": row_idx + 1,
                            "text_sha1": row_hash,
                            "text_preview": text_preview,
                            "mention": occurrence,
                            "inventory_mention": mention,
                            "expected_label": label,
                            "existing_labels": existing_labels,
                            "start": start,
                            "end": end,
                            "context": make_context(text, start, end, context_chars),
                            "annotated_count_elsewhere": item["annotated_count"],
                        }
                    )
                    conflict_label_counts[label] += 1
                    continue
                if overlaps_any_span(start, end, annotated_intervals):
                    continue
                candidates.append(
                    {
                        "row_index_0based": row_idx,
                        "row_index_1based": row_idx + 1,
                        "text_sha1": row_hash,
                        "text_preview": text_preview,
                        "label": label,
                        "mention": occurrence,
                        "inventory_mention": mention,
                        "start": start,
                        "end": end,
                        "context": make_context(text, start, end, context_chars),
                        "annotated_count_elsewhere": item["annotated_count"],
                    }
                )
                label_counts[label] += 1

    candidates.sort(key=lambda x: (x["label"], x["mention"], x["row_index_0based"], x["start"]))
    conflicts.sort(
        key=lambda x: (
            x["expected_label"],
            x["mention"],
            x["row_index_0based"],
            x["start"],
        )
    )
    unique_spans = sorted(
        inventory.values(),
        key=lambda x: (x["label"], x["mention"]),
    )

    return {
        "records_total": len(rows),
        "unique_annotated_spans": len(inventory),
        "missing_candidates_total": len(candidates),
        "conflicts_total": len(conflicts),
        "settings": {
            "labels": sorted(labels) if labels is not None else [],
            "ignore_case": ignore_case,
            "min_len": min_len,
            "min_annotated_count": min_annotated_count,
            "context_chars": context_chars,
        },
        "missing_candidates_by_label": dict(sorted(label_counts.items())),
        "conflicts_by_expected_label": dict(sorted(conflict_label_counts.items())),
        "unique_spans": unique_spans,
        "candidates": candidates,
        "conflicts": conflicts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory annotated entity mentions and report exact unannotated "
            "occurrences of those same mention+label pairs."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL annotated corpus.")
    parser.add_argument("--output", required=True, help="Output JSON report.")
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated label filter, e.g. Person,Location,Organization.",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Use case-insensitive exact string matching.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Ignore annotated mentions shorter than this many characters.",
    )
    parser.add_argument(
        "--min-annotated-count",
        type=int,
        default=1,
        help="Only search mentions annotated at least this many times.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_CONTEXT_CHARS,
        help="Characters to include before/after each candidate in context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = parse_labels(args.labels) if args.labels else None
    if args.min_len < 1:
        raise ValueError("--min-len must be >= 1.")
    if args.min_annotated_count < 1:
        raise ValueError("--min-annotated-count must be >= 1.")
    if args.context_chars < 0:
        raise ValueError("--context-chars must be >= 0.")

    rows = read_json_or_jsonl(args.input)
    report = find_missing_repeated_spans(
        rows,
        labels=labels,
        ignore_case=args.ignore_case,
        min_len=args.min_len,
        min_annotated_count=args.min_annotated_count,
        context_chars=args.context_chars,
    )
    report["input"] = str(Path(args.input).resolve())
    report["output"] = str(Path(args.output).resolve())
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "input": report["input"],
                "output": report["output"],
                "records_total": report["records_total"],
                "unique_annotated_spans": report["unique_annotated_spans"],
                "missing_candidates_total": report["missing_candidates_total"],
                "conflicts_total": report["conflicts_total"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
