#!/usr/bin/env python3
"""Inventory labeled records against source corpora and recover source assuntos.

The main use case is checking whether labeled train/test reports are present in the
original large unlabeled corpus, so topic metadata such as ``assunto`` can be
reconstructed for the labeled splits.

Example:
python3 src/tools/inventory_labeled_source_assuntos.py \
  --source-input data/deprecated/dd_corpus_large.json \
  --labeled-input data/dd_corpus_small_train.json \
  --labeled-input data/dd_corpus_small_test.json \
  --output-csv artifacts/inventories/labeled_source_assuntos.csv \
  --summary-json artifacts/inventories/labeled_source_assuntos_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEXT_FIELDS = ("text", "relato", "texto", "description", "descricao")
SPAN_FIELDS = ("spans", "entities", "ner")
SOURCE_METADATA_FIELDS = (
    "assunto",
    "logradouroLocal",
    "bairroLocal",
    "cidadeLocal",
    "pontodeReferenciaLocal",
)
MATCH_STRATEGIES = (
    "exact",
    "whitespace_lower",
    "alnum_lower",
    "ascii_alnum_lower",
)


@dataclass(frozen=True)
class SourceEntry:
    source_input: str
    source_row_index_0based: int
    text: str
    row: dict[str, Any]


def infer_source_partition(source_input: str, row: dict[str, Any]) -> str:
    sanitization = row.get("_sanitization")
    if isinstance(sanitization, dict):
        status = str(sanitization.get("status", "")).strip()
        if status == "kept":
            return "sanitized"
        if status == "flagged_review":
            return "flagged"
        if status == "dropped_safe":
            return "dropped"
        if status:
            return status

    name = Path(source_input).name.lower()
    if "sanitized" in name:
        return "sanitized"
    if "flagged" in name:
        return "flagged"
    if "dropped" in name:
        return "dropped"
    return "source"


def parse_jsonl(text: str, path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def get_text(row: dict[str, Any], text_field: str) -> str:
    if text_field != "auto":
        value = row.get(text_field)
        return value.strip() if isinstance(value, str) else ""

    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    for field in SPAN_FIELDS:
        spans = row.get(field)
        if isinstance(spans, list):
            return [span for span in spans if isinstance(span, dict)]
    return []


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def alnum_normalize(text: str, *, ascii_only: bool) -> str:
    if ascii_only:
        text = strip_accents(text)
    text = text.lower()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return collapse_whitespace(text)


def match_key(text: str, strategy: str) -> str:
    if strategy == "exact":
        return text.strip()
    if strategy == "whitespace_lower":
        return collapse_whitespace(text).lower()
    if strategy == "alnum_lower":
        return alnum_normalize(text, ascii_only=False)
    if strategy == "ascii_alnum_lower":
        return alnum_normalize(text, ascii_only=True)
    raise ValueError(f"Unknown match strategy: {strategy}")


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_source_indexes(
    source_inputs: list[str],
    *,
    source_text_field: str,
) -> tuple[dict[str, dict[str, list[SourceEntry]]], dict[str, int]]:
    indexes: dict[str, dict[str, list[SourceEntry]]] = {
        strategy: defaultdict(list)
        for strategy in MATCH_STRATEGIES
    }
    source_row_counts: dict[str, int] = {}

    for source_input in source_inputs:
        rows = read_json_or_jsonl(source_input)
        source_row_counts[source_input] = len(rows)
        for idx, row in enumerate(rows):
            text = get_text(row, source_text_field)
            if not text:
                continue
            entry = SourceEntry(
                source_input=source_input,
                source_row_index_0based=idx,
                text=text,
                row=row,
            )
            for strategy in MATCH_STRATEGIES:
                key = match_key(text, strategy)
                if key:
                    indexes[strategy][key].append(entry)
    return indexes, source_row_counts


def find_matches(
    text: str,
    indexes: dict[str, dict[str, list[SourceEntry]]],
) -> tuple[str, list[SourceEntry]]:
    for strategy in MATCH_STRATEGIES:
        key = match_key(text, strategy)
        if not key:
            continue
        matches = indexes[strategy].get(key, [])
        if matches:
            return strategy, matches
    return "", []


def label_counts_for_row(row: dict[str, Any]) -> Counter:
    counts: Counter = Counter()
    for span in get_spans(row):
        label = span.get("label")
        if label is not None:
            counts[str(label)] += 1
    return counts


def source_value(entry: SourceEntry | None, field: str) -> str:
    if entry is None:
        return ""
    value = entry.row.get(field)
    return "" if value is None else str(value)


def sanitization_value(entry: SourceEntry | None, field: str) -> str:
    if entry is None:
        return ""
    sanitization = entry.row.get("_sanitization")
    if not isinstance(sanitization, dict):
        return ""
    value = sanitization.get(field)
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return "" if value is None else str(value)


def match_status(matches: list[SourceEntry]) -> str:
    if not matches:
        return "unmatched"
    assuntos = {source_value(entry, "assunto") for entry in matches}
    if len(matches) == 1:
        return "matched_unique"
    if len(assuntos) == 1:
        return "matched_duplicate_same_assunto"
    return "matched_ambiguous_assunto"


def build_inventory_rows(
    labeled_inputs: list[str],
    indexes: dict[str, dict[str, list[SourceEntry]]],
    *,
    labeled_text_field: str,
) -> list[dict[str, Any]]:
    inventory_rows: list[dict[str, Any]] = []
    for labeled_input in labeled_inputs:
        labeled_rows = read_json_or_jsonl(labeled_input)
        for idx, row in enumerate(labeled_rows):
            text = get_text(row, labeled_text_field)
            strategy, matches = find_matches(text, indexes)
            first_match = matches[0] if matches else None
            label_counts = label_counts_for_row(row)
            assuntos = sorted({source_value(entry, "assunto") for entry in matches})
            partitions = sorted({infer_source_partition(entry.source_input, entry.row) for entry in matches})
            source_refs = [
                f"{entry.source_input}:{entry.source_row_index_0based + 1}"
                for entry in matches[:20]
            ]

            out: dict[str, Any] = {
                "labeled_input": labeled_input,
                "labeled_row_index_0based": idx,
                "labeled_row_index_1based": idx + 1,
                "labeled_id": row.get("id", ""),
                "text_sha1": text_sha1(text),
                "text_chars": len(text),
                "span_count": sum(label_counts.values()),
                "location_count": label_counts.get("Location", 0),
                "person_count": label_counts.get("Person", 0),
                "organization_count": label_counts.get("Organization", 0),
                "match_status": match_status(matches),
                "match_strategy": strategy,
                "match_count": len(matches),
                "matched_assuntos": "|".join(assuntos),
                "matched_partitions": "|".join(partitions),
                "matched_source_refs_preview": "|".join(source_refs),
                "first_source_input": first_match.source_input if first_match else "",
                "first_source_partition": infer_source_partition(first_match.source_input, first_match.row)
                if first_match
                else "",
                "first_source_row_index_0based": first_match.source_row_index_0based if first_match else "",
                "first_source_row_index_1based": first_match.source_row_index_0based + 1 if first_match else "",
                "first_sanitization_status": sanitization_value(first_match, "status"),
                "first_sanitization_reasons": sanitization_value(first_match, "reasons"),
                "first_sanitization_row_index_1based": sanitization_value(first_match, "row_index_1based"),
            }
            for field in SOURCE_METADATA_FIELDS:
                out[f"first_{field}"] = source_value(first_match, field)
            inventory_rows.append(out)
    return inventory_rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_inventory(
    inventory_rows: list[dict[str, Any]],
    *,
    source_row_counts: dict[str, int],
) -> dict[str, Any]:
    by_labeled_input: dict[str, Any] = {}
    for labeled_input in sorted({str(row["labeled_input"]) for row in inventory_rows}):
        subset = [row for row in inventory_rows if row["labeled_input"] == labeled_input]
        unambiguous_subset = [
            row
            for row in subset
            if row["match_status"] in {"matched_unique", "matched_duplicate_same_assunto"}
        ]
        by_labeled_input[labeled_input] = {
            "rows": len(subset),
            "match_status": dict(Counter(str(row["match_status"]) for row in subset)),
            "match_strategy": dict(Counter(str(row["match_strategy"]) for row in subset if row["match_strategy"])),
            "matched_partitions": dict(Counter(str(row["matched_partitions"]) for row in subset if row["matched_partitions"])),
            "first_source_partition": dict(
                Counter(str(row["first_source_partition"]) for row in subset if row["first_source_partition"])
            ),
            "unambiguous_assunto": dict(
                Counter(str(row["first_assunto"]) for row in unambiguous_subset if row["first_assunto"])
            ),
            "unambiguous_assunto_rows": len(unambiguous_subset),
        }

    unambiguous_rows = [
        row
        for row in inventory_rows
        if row["match_status"] in {"matched_unique", "matched_duplicate_same_assunto"}
    ]
    return {
        "source_row_counts": source_row_counts,
        "rows": len(inventory_rows),
        "match_status": dict(Counter(str(row["match_status"]) for row in inventory_rows)),
        "match_strategy": dict(Counter(str(row["match_strategy"]) for row in inventory_rows if row["match_strategy"])),
        "matched_partitions": dict(Counter(str(row["matched_partitions"]) for row in inventory_rows if row["matched_partitions"])),
        "first_source_partition": dict(
            Counter(str(row["first_source_partition"]) for row in inventory_rows if row["first_source_partition"])
        ),
        "unambiguous_assunto": dict(
            Counter(str(row["first_assunto"]) for row in unambiguous_rows if row["first_assunto"])
        ),
        "unambiguous_assunto_rows": len(unambiguous_rows),
        "by_labeled_input": by_labeled_input,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match labeled NER rows back to source corpora and recover source assunto metadata."
    )
    parser.add_argument(
        "--source-input",
        action="append",
        required=True,
        help="Source JSON/JSONL corpus. Repeat to search multiple corpora.",
    )
    parser.add_argument(
        "--labeled-input",
        action="append",
        required=True,
        help="Labeled JSON/JSONL corpus to inventory. Repeat for train/test/calibration.",
    )
    parser.add_argument("--output-csv", required=True, help="Output inventory CSV.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument(
        "--source-text-field",
        default="auto",
        help="Source text field, or 'auto' for common fields.",
    )
    parser.add_argument(
        "--labeled-text-field",
        default="auto",
        help="Labeled text field, or 'auto' for common fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexes, source_row_counts = build_source_indexes(
        args.source_input,
        source_text_field=args.source_text_field,
    )
    inventory_rows = build_inventory_rows(
        args.labeled_input,
        indexes,
        labeled_text_field=args.labeled_text_field,
    )
    write_csv(args.output_csv, inventory_rows)

    summary = summarize_inventory(inventory_rows, source_row_counts=source_row_counts)
    if args.summary_json:
        write_json(args.summary_json, summary)

    print(f"Source inputs: {len(args.source_input)}")
    print(f"Labeled inputs: {len(args.labeled_input)}")
    print(f"Inventory rows: {len(inventory_rows)}")
    print(f"Match status: {summary['match_status']}")
    print(f"Output CSV: {args.output_csv}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
