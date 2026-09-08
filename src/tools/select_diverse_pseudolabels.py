#!/usr/bin/env python3
"""Select a diverse top-k pseudolabel set from scored pseudolabel records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl
from tools.render_ner_html import render_html


def _parse_csv(raw_value: str) -> list[str]:
    return [piece.strip() for piece in str(raw_value or "").split(",") if piece.strip()]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_float(value: float) -> float | None:
    if math.isfinite(value):
        return value
    return None


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(value: Any) -> str:
    text = _strip_accents(str(value or "")).lower()
    text = " ".join(text.split())
    return text.strip(" \t\r\n.,;:!?()[]{}\"'")


def _text(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _get_path(row: dict[str, Any], field_path: str) -> Any:
    value: Any = row
    for part in field_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _score(row: dict[str, Any], fields: list[str]) -> tuple[float | None, str]:
    for field in fields:
        value = _safe_float(_get_path(row, field))
        if value is not None:
            return value, field
    pseudo = row.get("_pseudolabel")
    if isinstance(pseudo, dict):
        for field in fields:
            if "." in field:
                continue
            value = _safe_float(pseudo.get(field))
            if value is not None:
                return value, f"_pseudolabel.{field}"
    return None, ""


def _spans(row: dict[str, Any], span_keys: list[str]) -> list[dict[str, Any]]:
    for key in span_keys:
        value = row.get(key)
        if isinstance(value, list):
            return [span for span in value if isinstance(span, dict)]
    return []


def _span_label(span: dict[str, Any], label_field: str) -> str:
    return str(span.get(label_field, "")).strip()


def _span_text(row_text: str, span: dict[str, Any]) -> str:
    value = span.get("text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    try:
        return row_text[int(span["start"]) : int(span["end"])]
    except (KeyError, TypeError, ValueError):
        return ""


def _location_terms(
    row: dict[str, Any],
    *,
    text: str,
    span_keys: list[str],
    label_field: str,
    target_labels: set[str],
) -> list[str]:
    terms = []
    for span in _spans(row, span_keys):
        if target_labels and _span_label(span, label_field) not in target_labels:
            continue
        term = normalize_text(_span_text(text, span))
        if term:
            terms.append(term)
    return sorted(set(terms))


def _signature(terms: list[str], max_terms: int) -> str:
    if not terms:
        return ""
    return "|".join(sorted(terms)[:max_terms])


def _passes_caps(
    *,
    text_key: str,
    signature: str,
    terms: list[str],
    seen_texts: set[str],
    signature_counts: Counter,
    entity_counts: Counter,
    max_per_signature: int,
    max_per_entity: int,
) -> tuple[bool, str]:
    if text_key in seen_texts:
        return False, "duplicate_text"
    if max_per_signature > 0 and signature and signature_counts[signature] >= max_per_signature:
        return False, "signature_cap"
    if max_per_entity > 0:
        capped_terms = [term for term in terms if entity_counts[term] >= max_per_entity]
        if capped_terms:
            return False, "entity_cap"
    return True, ""


def select_diverse_rows(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    score_fields: list[str],
    min_score: float,
    text_fields: list[str],
    span_keys: list[str],
    label_field: str,
    target_labels: set[str],
    max_per_entity: int,
    max_per_signature: int,
    signature_max_terms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared = []
    counters = Counter()
    for source_index, row in enumerate(rows, start=1):
        counters["rows_total"] += 1
        score, score_field = _score(row, score_fields)
        if score is None:
            counters["dropped_missing_score"] += 1
            continue
        if score < min_score:
            counters["dropped_min_score"] += 1
            continue

        text = _text(row, text_fields)
        text_key = normalize_text(text)
        terms = _location_terms(
            row,
            text=text,
            span_keys=span_keys,
            label_field=label_field,
            target_labels=target_labels,
        )
        if not terms:
            counters["dropped_no_target_entities"] += 1
            continue

        prepared.append(
            {
                "source_index_1based": source_index,
                "row": row,
                "score": score,
                "score_field": score_field,
                "text": text,
                "text_key": text_key,
                "terms": terms,
                "signature": _signature(terms, signature_max_terms),
            }
        )

    prepared.sort(
        key=lambda item: (
            -item["score"],
            len(item["terms"]),
            len(item["text"]),
            item["source_index_1based"],
        )
    )

    selected = []
    audit_rows = []
    seen_texts: set[str] = set()
    entity_counts: Counter = Counter()
    signature_counts: Counter = Counter()

    for candidate in prepared:
        if top_n > 0 and len(selected) >= top_n:
            decision = "not_selected_after_top_n"
            reason = "top_n_reached"
            counters[reason] += 1
        else:
            passes, reason = _passes_caps(
                text_key=candidate["text_key"],
                signature=candidate["signature"],
                terms=candidate["terms"],
                seen_texts=seen_texts,
                signature_counts=signature_counts,
                entity_counts=entity_counts,
                max_per_signature=max_per_signature,
                max_per_entity=max_per_entity,
            )
            if passes:
                row = dict(candidate["row"])
                selection = dict(row.get("_pseudolabel_selection") or {})
                selection.update(
                    {
                        "diverse_rank": len(selected) + 1,
                        "source_row_index_1based": candidate["source_index_1based"],
                        "score": candidate["score"],
                        "score_field": candidate["score_field"],
                        "location_signature": candidate["signature"],
                        "unique_location_terms": candidate["terms"],
                    }
                )
                row["_pseudolabel_selection"] = selection
                selected.append(row)
                seen_texts.add(candidate["text_key"])
                signature_counts[candidate["signature"]] += 1
                entity_counts.update(candidate["terms"])
                decision = "selected"
                reason = ""
                counters["selected"] += 1
            else:
                decision = "rejected"
                counters[f"rejected_{reason}"] += 1

        audit_rows.append(
            {
                "source_row_index_1based": candidate["source_index_1based"],
                "decision": decision,
                "reason": reason,
                "score": f"{candidate['score']:.8f}",
                "score_field": candidate["score_field"],
                "unique_location_count": len(candidate["terms"]),
                "location_signature": candidate["signature"],
                "locations": " | ".join(candidate["terms"]),
                "text_preview": candidate["text"].replace("\n", " ")[:220],
            }
        )

    summary = {
        "rows_total": len(rows),
        "rows_after_score_and_label_filters": len(prepared),
        "rows_selected": len(selected),
        "counters": dict(counters),
        "selected_location_term_counts": dict(entity_counts.most_common()),
        "selected_signature_counts": dict(signature_counts.most_common()),
    }
    return selected, audit_rows, summary


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_row_index_1based",
        "decision",
        "reason",
        "score",
        "score_field",
        "unique_location_count",
        "location_signature",
        "locations",
        "text_preview",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a diverse top-k pseudolabel set.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL pseudolabel pool.")
    parser.add_argument("--output-jsonl", required=True, help="Selected pseudolabel JSONL output.")
    parser.add_argument("--summary-json", required=True, help="Selection summary JSON output.")
    parser.add_argument("--audit-csv", default="", help="Optional CSV with selected/rejected decisions.")
    parser.add_argument("--output-html", default="", help="Optional HTML review file for selected rows.")
    parser.add_argument("--title", default="Diverse Pseudolabel Selection", help="HTML title.")
    parser.add_argument("--top-n", type=int, default=500, help="Number of records to select. Use 0 for all passing caps.")
    parser.add_argument(
        "--score-fields",
        default="record_score_location,record_score,_pseudolabel.record_score_location",
        help="Comma-separated record-level score fields to try.",
    )
    parser.add_argument("--min-score", type=float, default=float("-inf"), help="Minimum record score.")
    parser.add_argument("--text-fields", default="text,relato,_inference_text", help="Comma-separated text fields.")
    parser.add_argument("--span-keys", default="spans,entities,ner", help="Comma-separated entity list keys.")
    parser.add_argument("--label-field", default="label", help="Span label field.")
    parser.add_argument("--target-labels", default="Location", help="Comma-separated labels used for diversity.")
    parser.add_argument("--max-per-entity", type=int, default=10, help="Maximum selected rows per normalized Location term. Use 0 to disable.")
    parser.add_argument("--max-per-signature", type=int, default=2, help="Maximum selected rows per normalized Location-set signature. Use 0 to disable.")
    parser.add_argument("--signature-max-terms", type=int, default=8, help="Maximum normalized terms used in a signature.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    selected, audit_rows, summary = select_diverse_rows(
        rows,
        top_n=args.top_n,
        score_fields=_parse_csv(args.score_fields),
        min_score=args.min_score,
        text_fields=_parse_csv(args.text_fields),
        span_keys=_parse_csv(args.span_keys),
        label_field=args.label_field,
        target_labels=set(_parse_csv(args.target_labels)),
        max_per_entity=args.max_per_entity,
        max_per_signature=args.max_per_signature,
        signature_max_terms=args.signature_max_terms,
    )

    write_jsonl(args.output_jsonl, selected)
    if args.audit_csv:
        write_csv(args.audit_csv, audit_rows)
    if args.output_html:
        render_html(selected, output_path=args.output_html, title=args.title, max_reports=0)

    output_summary = {
        "input": str(Path(args.input).resolve()),
        "outputs": {
            "jsonl": str(Path(args.output_jsonl).resolve()),
            "summary_json": str(Path(args.summary_json).resolve()),
            "audit_csv": str(Path(args.audit_csv).resolve()) if args.audit_csv else None,
            "html": str(Path(args.output_html).resolve()) if args.output_html else None,
        },
        "config": {
            "top_n": args.top_n,
            "score_fields": _parse_csv(args.score_fields),
            "min_score": _json_float(args.min_score),
            "text_fields": _parse_csv(args.text_fields),
            "span_keys": _parse_csv(args.span_keys),
            "label_field": args.label_field,
            "target_labels": _parse_csv(args.target_labels),
            "max_per_entity": args.max_per_entity,
            "max_per_signature": args.max_per_signature,
            "signature_max_terms": args.signature_max_terms,
        },
        "summary": summary,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(output_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Rows selected: {summary['rows_selected']}/{summary['rows_after_score_and_label_filters']}")
    print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Saved summary JSON: {args.summary_json}")
    if args.audit_csv:
        print(f"Saved audit CSV: {args.audit_csv}")
    if args.output_html:
        print(f"Saved HTML: {args.output_html}")


if __name__ == "__main__":
    main()
