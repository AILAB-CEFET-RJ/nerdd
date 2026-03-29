"""Summarize context-boost audit JSONL artifacts into compact statistics."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize context boost audit artifacts")
    parser.add_argument("--details-jsonl", required=True, help="Input 03_context_boost_details.jsonl")
    parser.add_argument("--summary-json", required=True, help="Output summary JSON")
    parser.add_argument("--rows-csv", required=True, help="Output CSV with one row per boosted entity")
    parser.add_argument("--top-n", type=int, default=20, help="How many top rows to keep in the summary preview")
    return parser.parse_args()


def build_summary(rows, top_n):
    summary = {
        "records_total": len(rows),
        "records_with_context_match": 0,
        "records_with_boosted_entities": 0,
        "entity_candidates_total": 0,
        "entity_boosts_total": 0,
        "boost_reason_counts": Counter(),
        "boosted_label_counts": Counter(),
        "matched_metadata_field_counts": Counter(),
        "top_boosted_records": [],
    }

    boosted_record_rows = []
    flat_boost_rows = []

    for row in rows:
        record_context_match = bool(row.get("record_context_match"))
        if record_context_match:
            summary["records_with_context_match"] += 1

        matched_fields = row.get("matched_metadata_fields") or []
        for field in matched_fields:
            summary["matched_metadata_field_counts"][str(field)] += 1

        entity_trace = row.get("entity_trace") or []
        summary["entity_candidates_total"] += len(entity_trace)

        boosted_entities = []
        for entity in entity_trace:
            if not entity.get("boost_applied"):
                continue
            boosted_entities.append(entity)
            summary["entity_boosts_total"] += 1
            summary["boost_reason_counts"][str(entity.get("boost_reason", ""))] += 1
            summary["boosted_label_counts"][str(entity.get("label", ""))] += 1
            flat_boost_rows.append(
                {
                    "row_index": row.get("row_index"),
                    "sample_id": row.get("sample_id"),
                    "id": row.get("id"),
                    "text_field_used": row.get("text_field_used"),
                    "matched_metadata_fields": ",".join(matched_fields),
                    "matched_metadata_values": " | ".join(row.get("matched_metadata_values") or []),
                    "entity_text": entity.get("text", ""),
                    "entity_label": entity.get("label", ""),
                    "score_before": entity.get("score_before"),
                    "score_after": entity.get("score_after"),
                    "boost_multiplier": entity.get("boost_multiplier"),
                    "score_source": entity.get("score_source", ""),
                    "entity_matches_metadata": entity.get("entity_matches_metadata"),
                    "boost_reason": entity.get("boost_reason", ""),
                }
            )

        if boosted_entities:
            summary["records_with_boosted_entities"] += 1
            max_after = max((_safe_float(item.get("score_after")) or 0.0) for item in boosted_entities)
            boosted_record_rows.append(
                {
                    "row_index": row.get("row_index"),
                    "sample_id": row.get("sample_id"),
                    "id": row.get("id"),
                    "boosted_entities": len(boosted_entities),
                    "matched_metadata_fields": matched_fields,
                    "matched_metadata_values": row.get("matched_metadata_values") or [],
                    "max_boosted_score_after": max_after,
                    "boosted_entity_preview": [
                        {
                            "text": item.get("text", ""),
                            "label": item.get("label", ""),
                            "score_after": item.get("score_after"),
                            "boost_reason": item.get("boost_reason", ""),
                        }
                        for item in boosted_entities[:5]
                    ],
                }
            )

    boosted_record_rows.sort(
        key=lambda item: (
            -(item.get("boosted_entities") or 0),
            -(_safe_float(item.get("max_boosted_score_after")) or 0.0),
            item.get("row_index") or 0,
        )
    )

    summary["boost_reason_counts"] = dict(summary["boost_reason_counts"])
    summary["boosted_label_counts"] = dict(summary["boosted_label_counts"])
    summary["matched_metadata_field_counts"] = dict(summary["matched_metadata_field_counts"])
    summary["top_boosted_records"] = boosted_record_rows[:top_n]
    return summary, flat_boost_rows


def write_rows_csv(path, rows):
    fieldnames = [
        "row_index",
        "sample_id",
        "id",
        "text_field_used",
        "matched_metadata_fields",
        "matched_metadata_values",
        "entity_text",
        "entity_label",
        "score_before",
        "score_after",
        "boost_multiplier",
        "score_source",
        "entity_matches_metadata",
        "boost_reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    details_path = Path(args.details_jsonl)
    summary_path = Path(args.summary_json)
    rows_csv_path = Path(args.rows_csv)

    if not details_path.exists():
        raise FileNotFoundError(f"Details JSONL not found: {details_path}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(str(details_path))
    summary, flat_boost_rows = build_summary(rows, top_n=args.top_n)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_rows_csv(rows_csv_path, flat_boost_rows)

    print(f"Saved summary JSON to: {summary_path}")
    print(f"Saved boosted-entity CSV to: {rows_csv_path}")
    print(f"Records with boosted entities: {summary['records_with_boosted_entities']}")
    print(f"Boosted entities total: {summary['entity_boosts_total']}")


if __name__ == "__main__":
    main()
