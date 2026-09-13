#!/usr/bin/env python3
"""Keep pseudolabel records whose full predictions pass a completeness gate.

The companion audit records every entity predicted for a selected report.  This
tool converts that audit into a reproducible record-level gate: a report is
kept only when it has an unambiguous match and no omitted entity reaches the
configured score floor for its label.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl
from tools.render_ner_html import render_html


DEFAULT_MATCH_STATUSES = {
    "source_id",
    "normalized_text",
    "normalized_text_equivalent_duplicate",
}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _parse_label_scores(value: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in _parse_csv(value):
        if "=" not in item:
            raise ValueError(f"Invalid label score: {item}. Expected LABEL=SCORE.")
        label, raw_score = item.split("=", 1)
        try:
            score = float(raw_score.strip())
        except ValueError as error:
            raise ValueError(f"Invalid label score: {item}. Expected LABEL=SCORE.") from error
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid label score: {item}. Expected LABEL=SCORE.")
        scores[label] = score
    if not scores:
        raise ValueError("At least one --label-min-scores entry is required.")
    return scores


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_completeness_gate(
    selected_rows: list[dict[str, Any]],
    record_audit_rows: list[dict[str, str]],
    entity_audit_rows: list[dict[str, str]],
    *,
    label_min_scores: dict[str, float],
    allowed_match_statuses: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Filter selected rows with a label-specific omitted-entity risk policy."""
    records_by_index: dict[int, dict[str, str]] = {}
    for record in record_audit_rows:
        index = _safe_int(record.get("selected_index_1based"))
        if index is not None:
            records_by_index[index] = record

    risky_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    risky_label_counts: Counter[str] = Counter()
    risky_status_counts: Counter[str] = Counter()
    for entity in entity_audit_rows:
        index = _safe_int(entity.get("selected_index_1based"))
        label = str(entity.get("full_label", "")).strip()
        score = _safe_float(entity.get("score"))
        status = str(entity.get("status", "")).strip()
        minimum = label_min_scores.get(label)
        if index is None or minimum is None or status == "retained_exact" or score is None or score < minimum:
            continue
        risk = {
            "label": label,
            "mention": entity.get("full_mention", ""),
            "score": score,
            "minimum_score": minimum,
            "status": status,
        }
        risky_by_index[index].append(risk)
        risky_label_counts[label] += 1
        risky_status_counts[status] += 1

    kept_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    for index, row in enumerate(selected_rows, start=1):
        audit = records_by_index.get(index)
        match_status = str(audit.get("match_status", "")) if audit else "missing_audit_record"
        risks = risky_by_index.get(index, [])
        if audit is None:
            decision = "missing_audit_record"
        elif match_status not in allowed_match_statuses:
            decision = "ineligible_match"
        elif risks:
            decision = "risky_omission"
        else:
            decision = "kept"
        decision_counts[decision] += 1
        decisions.append(
            {
                "selected_index_1based": index,
                "decision": decision,
                "match_status": match_status,
                "risky_omitted_entity_count": len(risks),
                "risky_omitted_labels": " | ".join(sorted({risk["label"] for risk in risks})),
                "max_risky_omitted_score": max((risk["score"] for risk in risks), default=None),
                "text_preview": str(row.get("text", row.get("relato", ""))).replace("\n", " ")[:260],
            }
        )
        if decision != "kept":
            continue
        cleaned = copy.deepcopy(row)
        cleaned["_pseudolabel_completeness_gate"] = {
            "selected_index_1based": index,
            "match_status": match_status,
            "label_min_scores": dict(label_min_scores),
            "risky_omitted_entities": [],
        }
        kept_rows.append(cleaned)

    summary = {
        "selected_rows": len(selected_rows),
        "record_audit_rows": len(record_audit_rows),
        "entity_audit_rows": len(entity_audit_rows),
        "kept_rows": len(kept_rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "risky_entities_by_label": dict(sorted(risky_label_counts.items())),
        "risky_entities_by_status": dict(sorted(risky_status_counts.items())),
        "label_min_scores": dict(label_min_scores),
        "allowed_match_statuses": sorted(allowed_match_statuses),
    }
    return kept_rows, decisions, summary


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "selected_index_1based", "decision", "match_status", "risky_omitted_entity_count",
        "risky_omitted_labels", "max_risky_omitted_score", "text_preview",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter pseudolabel records using a completeness audit.")
    parser.add_argument("--selected-jsonl", required=True, help="Selected pseudolabel JSONL before completeness gating.")
    parser.add_argument("--record-audit-csv", required=True, help="record_completeness_audit.csv from the audit tool.")
    parser.add_argument("--entity-audit-csv", required=True, help="entity_completeness_audit.csv from the audit tool.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--decisions-csv", default="", help="Optional row-level gate decision CSV.")
    parser.add_argument("--output-html", default="", help="Optional HTML review of retained rows.")
    parser.add_argument("--title", default="Completeness-gated pseudolabels")
    parser.add_argument(
        "--label-min-scores",
        required=True,
        help="Comma-separated omission risk floors, e.g. Person=0.80,Location=0.95,Organization=0.90.",
    )
    parser.add_argument(
        "--allowed-match-statuses",
        default=",".join(sorted(DEFAULT_MATCH_STATUSES)),
        help="Comma-separated audit match statuses eligible for retention.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_min_scores = _parse_label_scores(args.label_min_scores)
    allowed_match_statuses = set(_parse_csv(args.allowed_match_statuses))
    selected_rows = read_json_or_jsonl(args.selected_jsonl)
    kept_rows, decisions, summary = build_completeness_gate(
        selected_rows,
        _read_csv(args.record_audit_csv),
        _read_csv(args.entity_audit_csv),
        label_min_scores=label_min_scores,
        allowed_match_statuses=allowed_match_statuses,
    )
    write_jsonl(args.output_jsonl, kept_rows)
    summary_payload = {
        "inputs": {
            "selected_jsonl": str(Path(args.selected_jsonl).resolve()),
            "record_audit_csv": str(Path(args.record_audit_csv).resolve()),
            "entity_audit_csv": str(Path(args.entity_audit_csv).resolve()),
        },
        "config": vars(args),
        "summary": summary,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.decisions_csv:
        _write_csv(args.decisions_csv, decisions)
    if args.output_html:
        render_html(kept_rows, output_path=args.output_html, title=args.title, max_reports=0)

    print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Saved summary JSON: {args.summary_json}")
    if args.decisions_csv:
        print(f"Saved decisions CSV: {args.decisions_csv}")
    if args.output_html:
        print(f"Saved HTML: {args.output_html}")
    print(f"Rows kept: {summary['kept_rows']}/{summary['selected_rows']}")


if __name__ == "__main__":
    main()
