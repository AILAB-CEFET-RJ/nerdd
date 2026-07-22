#!/usr/bin/env python3
"""Add recovered source assunto metadata to a labeled NER corpus.

The script preserves the input records and spans, adding ``assunto`` only when the
source match is unambiguous:

- ``matched_unique``
- ``matched_duplicate_same_assunto``

For unmatched or ambiguous records, it writes an audit block but leaves ``assunto``
unset unless the input row already had one.

Example:
python3 src/tools/enrich_labeled_with_source_assunto.py \
  --source-input data/large_sanitized/large_sanitized.jsonl \
  --source-input data/large_sanitized/large_flagged.jsonl \
  --source-input data/large_sanitized/large_dropped.jsonl \
  --labeled-input data/annotations_corrected_2001.json \
  --output-json data/annotations_corrected_2001_with_assunto.json \
  --audit-csv artifacts/inventories/annotations_corrected_2001_assunto_audit.csv \
  --summary-json artifacts/inventories/annotations_corrected_2001_assunto_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inventory_labeled_source_assuntos import (  # noqa: E402
    MATCH_STRATEGIES,
    SourceEntry,
    build_source_indexes,
    find_matches,
    get_text,
    infer_source_partition,
    match_status,
    read_json_or_jsonl,
    sanitization_value,
    source_value,
    text_sha1,
)


UNAMBIGUOUS_STATUSES = {"matched_unique", "matched_duplicate_same_assunto"}


def source_ref(entry: SourceEntry) -> str:
    return f"{entry.source_input}:{entry.source_row_index_0based + 1}"


def match_payload(
    *,
    text: str,
    strategy: str,
    status: str,
    matches: list[SourceEntry],
    selected_assunto: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "strategy": strategy,
        "match_count": len(matches),
        "selected_assunto": selected_assunto,
        "matched_assuntos": sorted({source_value(entry, "assunto") for entry in matches if source_value(entry, "assunto")}),
        "source_partitions": sorted({infer_source_partition(entry.source_input, entry.row) for entry in matches}),
        "source_refs_preview": [source_ref(entry) for entry in matches[:20]],
        "text_sha1": text_sha1(text),
    }


def audit_row(
    *,
    labeled_input: str,
    row_index: int,
    text: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "labeled_input": labeled_input,
        "labeled_row_index_0based": row_index,
        "labeled_row_index_1based": row_index + 1,
        "text_sha1": text_sha1(text),
        "text_chars": len(text),
        "status": payload["status"],
        "strategy": payload["strategy"],
        "match_count": payload["match_count"],
        "selected_assunto": payload["selected_assunto"],
        "matched_assuntos": "|".join(payload["matched_assuntos"]),
        "source_partitions": "|".join(payload["source_partitions"]),
        "source_refs_preview": "|".join(payload["source_refs_preview"]),
    }


def selected_unambiguous_assunto(status: str, matches: list[SourceEntry]) -> str:
    if status not in UNAMBIGUOUS_STATUSES:
        return ""
    assuntos = sorted({source_value(entry, "assunto") for entry in matches if source_value(entry, "assunto")})
    if len(assuntos) != 1:
        return ""
    return assuntos[0]


def enrich_rows(
    rows: list[dict[str, Any]],
    *,
    labeled_input: str,
    indexes: dict[str, dict[str, list[SourceEntry]]],
    labeled_text_field: str,
    overwrite_existing_assunto: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    enriched_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        text = get_text(row, labeled_text_field)
        strategy, matches = find_matches(text, indexes)
        status = match_status(matches)
        selected_assunto = selected_unambiguous_assunto(status, matches)
        payload = match_payload(
            text=text,
            strategy=strategy,
            status=status,
            matches=matches,
            selected_assunto=selected_assunto,
        )

        enriched = dict(row)
        if selected_assunto and (overwrite_existing_assunto or "assunto" not in enriched):
            enriched["assunto"] = selected_assunto
        enriched["_source_assunto_match"] = payload

        enriched_rows.append(enriched)
        audit_rows.append(
            audit_row(
                labeled_input=labeled_input,
                row_index=idx,
                text=text,
                payload=payload,
            )
        )

    return enriched_rows, audit_rows


def summarize(enriched_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], source_row_counts: dict[str, int]) -> dict:
    status_counts = Counter(row["status"] for row in audit_rows)
    strategy_counts = Counter(row["strategy"] for row in audit_rows if row["strategy"])
    assunto_counts = Counter(row.get("assunto", "") for row in enriched_rows if row.get("assunto"))
    selected_assunto_counts = Counter(row["selected_assunto"] for row in audit_rows if row["selected_assunto"])
    source_partition_counts = Counter()
    for row in audit_rows:
        if row["source_partitions"]:
            source_partition_counts[row["source_partitions"]] += 1

    return {
        "rows": len(enriched_rows),
        "rows_with_assunto": sum(1 for row in enriched_rows if row.get("assunto")),
        "rows_without_assunto": sum(1 for row in enriched_rows if not row.get("assunto")),
        "match_status": dict(status_counts),
        "match_strategy": dict(strategy_counts),
        "assunto": dict(assunto_counts),
        "selected_assunto": dict(selected_assunto_counts),
        "source_partitions": dict(source_partition_counts),
        "source_row_counts": source_row_counts,
    }


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_preserved_core(original: list[dict[str, Any]], enriched: list[dict[str, Any]]) -> None:
    if len(original) != len(enriched):
        raise ValueError(f"Row count changed: {len(original)} != {len(enriched)}")
    for idx, (before, after) in enumerate(zip(original, enriched), start=1):
        if before.get("text") != after.get("text"):
            raise ValueError(f"text changed at row {idx}")
        if before.get("spans") != after.get("spans"):
            raise ValueError(f"spans changed at row {idx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add recovered source assunto metadata to a labeled JSON/JSONL NER corpus."
    )
    parser.add_argument(
        "--source-input",
        action="append",
        required=True,
        help="Source JSON/JSONL corpus. Repeat for sanitized/flagged/dropped.",
    )
    parser.add_argument("--labeled-input", required=True, help="Labeled JSON/JSONL corpus to enrich.")
    parser.add_argument("--output-json", required=True, help="Output enriched JSON array.")
    parser.add_argument("--audit-csv", default="", help="Optional per-row audit CSV.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--source-text-field", default="auto", help="Source text field, or 'auto'.")
    parser.add_argument("--labeled-text-field", default="auto", help="Labeled text field, or 'auto'.")
    parser.add_argument(
        "--overwrite-existing-assunto",
        action="store_true",
        help="Replace an existing assunto field when a conservative source match is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexes, source_row_counts = build_source_indexes(args.source_input, source_text_field=args.source_text_field)
    rows = read_json_or_jsonl(args.labeled_input)
    enriched_rows, audit_rows = enrich_rows(
        rows,
        labeled_input=args.labeled_input,
        indexes=indexes,
        labeled_text_field=args.labeled_text_field,
        overwrite_existing_assunto=args.overwrite_existing_assunto,
    )
    validate_preserved_core(rows, enriched_rows)
    summary = summarize(enriched_rows, audit_rows, source_row_counts)

    write_json(args.output_json, enriched_rows)
    if args.audit_csv:
        write_csv(args.audit_csv, audit_rows)
    if args.summary_json:
        write_json(args.summary_json, summary)

    print(f"Input rows: {len(rows)}")
    print(f"Rows with assunto: {summary['rows_with_assunto']}")
    print(f"Rows without assunto: {summary['rows_without_assunto']}")
    print(f"Match status: {summary['match_status']}")
    print(f"Output JSON: {args.output_json}")
    if args.audit_csv:
        print(f"Audit CSV: {args.audit_csv}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
