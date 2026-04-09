#!/usr/bin/env python3
"""Build a small diagnostic prompt-probe set from regression/win audits."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a 10-case diagnostic probe for train-annotation prompting.")
    parser.add_argument("--regressions-jsonl", required=True, help="Audit regressions.jsonl path")
    parser.add_argument("--wins-jsonl", required=True, help="Audit wins.jsonl path")
    parser.add_argument("--source-input", required=True, help="Original train-adjudication candidate JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Output prompt-probe JSONL")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON")
    parser.add_argument("--top-location-person", type=int, default=3)
    parser.add_argument("--top-location-org", type=int, default=2)
    parser.add_argument("--top-boundary", type=int, default=2)
    parser.add_argument("--top-spurious", type=int, default=2)
    parser.add_argument("--top-wins", type=int, default=1)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _span_overlap(left, right):
    return int(left["start"]) < int(right["end"]) and int(right["start"]) < int(left["end"])


def _normalize_entity_list(items):
    normalized = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
            label = str(item["label"])
        except Exception:
            continue
        if end <= start:
            continue
        normalized.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": str(item.get("text", "")),
            }
        )
    return normalized


def _infer_wrong_label_confusions(row):
    gold_spans = _normalize_entity_list(row.get("spans", []))
    candidate_spans = _normalize_entity_list(row.get("candidate_entities", []))
    confusions = Counter()
    for gold in gold_spans:
        overlapping = [cand for cand in candidate_spans if _span_overlap(cand, gold)]
        if not overlapping:
            continue
        if any(cand["label"] == gold["label"] for cand in overlapping):
            continue
        for cand in overlapping:
            confusions[f"{gold['label']}->{cand['label']}"] += 1
    return confusions


def _load_rows(path):
    return [row for row in read_json_or_jsonl(path) if isinstance(row, dict)]


def _best_rows(rows, predicate, limit):
    selected = []
    for row in rows:
        if not predicate(row):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _clean_probe_row(row, source_row, categories):
    payload = {
        "source_id": source_row.get("source_id", ""),
        "text": source_row.get("text") or source_row.get("_source", {}).get("text", ""),
        "review_seed_entities": source_row.get("review_seed_entities", []),
        "_probe_meta": {
            "categories": sorted(categories),
            "loss_reasons": row.get("_audit", {}).get("loss_reasons", {}),
            "delta_row_f1": row.get("_audit", {}).get("delta_row_f1"),
        },
    }
    if row.get("_audit", {}).get("win_reasons"):
        payload["_probe_meta"]["win_reasons"] = row["_audit"]["win_reasons"]
    return payload


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    regressions = _load_rows(args.regressions_jsonl)
    wins = _load_rows(args.wins_jsonl)
    source_rows = _load_rows(args.source_input)

    source_by_text = {}
    source_by_id = {}
    for row in source_rows:
        text = str(row.get("text") or row.get("_source", {}).get("text", "")).strip()
        if text and text not in source_by_text:
            source_by_text[text] = row
        source_id = str(row.get("source_id", "")).strip()
        if source_id and source_id not in source_by_id:
            source_by_id[source_id] = row

    selected = []
    selected_keys = set()
    category_counts = Counter()

    def add_rows(rows, category, limit):
        added = 0
        for row in rows:
            source_row = None
            source_id = str(row.get("source_id", "")).strip()
            if source_id:
                source_row = source_by_id.get(source_id)
            if source_row is None:
                source_row = source_by_text.get(str(row.get("text", "")).strip())
            if source_row is None:
                continue
            key = str(source_row.get("source_id") or source_row.get("text"))
            if key in selected_keys:
                # enrich existing categories if already selected
                for existing in selected:
                    existing_key = str(existing.get("source_id") or existing.get("text"))
                    if existing_key == key:
                        existing["_probe_meta"]["categories"] = sorted(
                            set(existing["_probe_meta"]["categories"]) | {category}
                        )
                continue
            selected.append(_clean_probe_row(row, source_row, {category}))
            selected_keys.add(key)
            category_counts[category] += 1
            added += 1
            if added >= limit:
                break

    regression_rows = list(regressions)
    add_rows(
        _best_rows(regression_rows, lambda row: _infer_wrong_label_confusions(row).get("Location->Person", 0) > 0, args.top_location_person),
        "location_to_person",
        args.top_location_person,
    )
    add_rows(
        _best_rows(regression_rows, lambda row: _infer_wrong_label_confusions(row).get("Location->Organization", 0) > 0, args.top_location_org),
        "location_to_organization",
        args.top_location_org,
    )
    add_rows(
        _best_rows(regression_rows, lambda row: row.get("_audit", {}).get("loss_reasons", {}).get("boundary_or_partial", 0) > 0, args.top_boundary),
        "boundary_or_partial",
        args.top_boundary,
    )
    add_rows(
        _best_rows(regression_rows, lambda row: row.get("_audit", {}).get("loss_reasons", {}).get("spurious_entity", 0) > 0, args.top_spurious),
        "spurious_entity",
        args.top_spurious,
    )
    add_rows(wins[: args.top_wins], "win_reference", args.top_wins)

    write_jsonl(args.output_jsonl, selected)
    LOGGER.info("Saved prompt probe JSONL: %s", args.output_jsonl)

    if args.summary_json:
        summary = {
            "regressions_input": len(regressions),
            "wins_input": len(wins),
            "selected_rows": len(selected),
            "category_counts": dict(category_counts),
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Saved summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()
