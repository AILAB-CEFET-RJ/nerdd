#!/usr/bin/env python3
"""Expand Location spans to include preceding locative markers when applicable."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_MARKERS = [
    "rua",
    "r",
    "avenida",
    "av",
    "travessa",
    "trav",
    "trv",
    "tv",
    "tr",
    "estrada",
    "rodovia",
    "alameda",
    "praca",
    "praça",
]

DEFAULT_TITLES = [
    "dr",
    "dra",
    "prof",
    "profa",
    "dom",
    "sto",
    "sta",
    "pe",
    "min",
]


def _parse_jsonl(text: str) -> list[dict]:
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def read_json_or_jsonl_with_format(path: str) -> tuple[list[dict], str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))
    raw = source.read_text(encoding="utf-8")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return _parse_jsonl(raw), "jsonl"
    if isinstance(obj, list):
        return obj, "json"
    if isinstance(obj, dict):
        return [obj], "json"
    raise ValueError("Unsupported JSON format.")


def write_rows(path: str, rows: list[dict], fmt: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    target.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def compile_prefix_regex(markers: list[str], titles: list[str]) -> re.Pattern[str]:
    marker_alt = "|".join(re.escape(item) for item in markers)
    title_alt = "|".join(re.escape(item) for item in titles)
    return re.compile(
        rf"(?i)(?P<prefix>\b(?:{marker_alt})\b\s*\.?\s+(?:(?:{title_alt})\b\s*\.?\s+)*)$"
    )


def maybe_expand_location_span(text: str, span: dict, prefix_re: re.Pattern[str]) -> tuple[dict, bool, str]:
    if span.get("label") != "Location":
        return dict(span), False, ""
    start = span.get("start")
    end = span.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return dict(span), False, ""
    if start < 0 or end > len(text) or end <= start:
        return dict(span), False, ""
    existing_text = text[start:end]
    window_start = max(0, start - 40)
    prefix_window = text[window_start:start]
    match = prefix_re.search(prefix_window)
    if not match:
        return dict(span), False, ""
    prefix = match.group("prefix")
    new_start = start - len(prefix)
    if new_start < 0:
        return dict(span), False, ""
    expanded = dict(span)
    expanded["start"] = new_start
    expanded["end"] = end
    if "text" in expanded:
        expanded["text"] = text[new_start:end]
    return expanded, True, existing_text


def transform_rows(rows: list[dict], *, prefix_re: re.Pattern[str]) -> tuple[list[dict], dict]:
    out = []
    stats = Counter()
    examples = []
    for row_index, row in enumerate(rows, start=1):
        text = str(row.get("text", ""))
        spans = row.get("spans")
        if not isinstance(spans, list):
            out.append(dict(row))
            continue
        new_row = dict(row)
        new_spans = []
        row_changed = False
        for span in spans:
            if not isinstance(span, dict):
                new_spans.append(span)
                continue
            new_span, changed, old_text = maybe_expand_location_span(text, span, prefix_re)
            new_spans.append(new_span)
            if changed:
                row_changed = True
                stats["expanded_spans"] += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "row_index_1based": row_index,
                            "before": old_text,
                            "after": text[new_span["start"] : new_span["end"]],
                            "text": text[:220],
                        }
                    )
        new_row["spans"] = new_spans
        out.append(new_row)
        if row_changed:
            stats["rows_changed"] += 1
    stats["records_total"] = len(rows)
    stats["example_expansions"] = examples
    return out, dict(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand Location spans to include preceding locative markers like rua/travessa/av."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL corpus.")
    parser.add_argument("--output", default="", help="Output path. Defaults to '<input>.locfix.json'.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path.")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file.")
    parser.add_argument(
        "--markers",
        default=",".join(DEFAULT_MARKERS),
        help="Comma-separated list of locative markers to include.",
    )
    parser.add_argument(
        "--titles",
        default=",".join(DEFAULT_TITLES),
        help="Comma-separated list of optional abbreviated titles allowed between marker and name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, fmt = read_json_or_jsonl_with_format(args.input)
    markers = [item.strip() for item in str(args.markers).split(",") if item.strip()]
    titles = [item.strip() for item in str(args.titles).split(",") if item.strip()]
    prefix_re = compile_prefix_regex(markers, titles)
    transformed, summary = transform_rows(rows, prefix_re=prefix_re)

    output_path = args.input if args.inplace else (args.output or (str(Path(args.input)) + ".locfix"))
    write_rows(output_path, transformed, fmt)

    summary["input"] = str(Path(args.input).resolve())
    summary["output"] = str(Path(output_path).resolve())
    summary["markers"] = markers
    summary["titles"] = titles
    if args.summary_json:
        target = Path(args.summary_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
