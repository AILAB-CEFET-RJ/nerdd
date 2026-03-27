#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl
from tools.render_ner_html import render_html

DEFAULT_SCORE_FIELDS = ("score_context_boosted", "score_calibrated", "score", "score_ts")
DEFAULT_ENTITY_KEYS = ("entities", "spans", "ner")


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_csv(value):
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_label_score_overrides(value):
    overrides = {}
    for item in _parse_csv(value):
        if "=" not in item:
            raise ValueError(f"Invalid label score override: {item}. Expected LABEL=SCORE.")
        label, raw_score = item.split("=", 1)
        label = label.strip()
        score = _safe_float(raw_score.strip())
        if not label or score is None:
            raise ValueError(f"Invalid label score override: {item}. Expected LABEL=SCORE.")
        overrides[label] = score
    return overrides


def _entity_key(row):
    for key in DEFAULT_ENTITY_KEYS:
        value = row.get(key)
        if isinstance(value, list):
            return key
    return "entities"


def _entity_score(entity, score_fields):
    for key in score_fields:
        score = _safe_float(entity.get(key))
        if score is not None:
            return score
    return None


def _entity_length(entity):
    start = entity.get("start")
    end = entity.get("end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return end - start
    text = entity.get("text")
    if isinstance(text, str):
        return len(text)
    return 0


def _rank_entities(entities, score_fields):
    ranked = []
    for index, entity in enumerate(entities):
        score = _entity_score(entity, score_fields)
        ranked.append(
            (
                -(score if score is not None else -1.0),
                -_entity_length(entity),
                entity.get("start") if isinstance(entity.get("start"), int) else 10**9,
                entity.get("end") if isinstance(entity.get("end"), int) else 10**9,
                str(entity.get("label", "")),
                index,
                entity,
            )
        )
    ranked.sort()
    return [item[-1] for item in ranked]


def prune_row(
    row,
    *,
    min_entity_score,
    label_min_scores,
    max_entities_per_tip,
    score_fields,
    allowed_labels,
    drop_tips_over_max,
):
    entity_key = _entity_key(row)
    entities = row.get(entity_key)
    if not isinstance(entities, list):
        entities = []

    stats = Counter()
    stats["entities_before"] = len(entities)
    working = []

    for entity in entities:
        label = entity.get("label")
        if allowed_labels and label not in allowed_labels:
            stats["dropped_by_label"] += 1
            continue

        score = _entity_score(entity, score_fields)
        if score is None:
            stats["missing_score"] += 1
            score = float("-inf")

        entity_min_score = label_min_scores.get(label, min_entity_score)
        if score < entity_min_score:
            stats["dropped_by_score"] += 1
            continue

        working.append(deepcopy(entity))

    if max_entities_per_tip > 0 and len(working) > max_entities_per_tip:
        if drop_tips_over_max:
            stats["dropped_tip_over_cap"] = 1
            working = []
        else:
            working = _rank_entities(working, score_fields)[:max_entities_per_tip]
            stats["pruned_by_cap"] = stats["entities_before"] - stats["dropped_by_score"] - stats["dropped_by_label"] - len(working)

    cleaned = deepcopy(row)
    cleaned[entity_key] = working
    cleaned["entities"] = working
    cleaned["_pseudolabel_prune"] = {
        "entity_key": entity_key,
        "entities_before": stats["entities_before"],
        "entities_after": len(working),
        "dropped_by_score": stats["dropped_by_score"],
        "dropped_by_label": stats["dropped_by_label"],
        "missing_score": stats["missing_score"],
        "pruned_by_cap": stats["pruned_by_cap"],
        "dropped_tip_over_cap": bool(stats["dropped_tip_over_cap"]),
        "min_entity_score": min_entity_score,
        "label_min_scores": dict(label_min_scores),
        "max_entities_per_tip": max_entities_per_tip,
        "score_fields": list(score_fields),
        "allowed_labels": sorted(allowed_labels) if allowed_labels else [],
    }
    stats["entities_after"] = len(working)
    return cleaned, stats


def prune_rows(
    rows,
    *,
    min_entity_score,
    label_min_scores,
    max_entities_per_tip,
    score_fields,
    allowed_labels,
    drop_tips_over_max,
    drop_empty_tips,
):
    cleaned_rows = []
    summary = Counter()

    for row in rows:
        cleaned, row_stats = prune_row(
            row,
            min_entity_score=min_entity_score,
            label_min_scores=label_min_scores,
            max_entities_per_tip=max_entities_per_tip,
            score_fields=score_fields,
            allowed_labels=allowed_labels,
            drop_tips_over_max=drop_tips_over_max,
        )
        summary.update(row_stats)
        summary["tips_total"] += 1
        if drop_empty_tips and not cleaned.get("entities"):
            summary["dropped_empty_tips"] += 1
            continue
        cleaned_rows.append(cleaned)

    summary["tips_kept"] = len(cleaned_rows)
    return cleaned_rows, summary


def build_summary(summary, args, input_path):
    entities_before = summary.get("entities_before", 0)
    entities_after = summary.get("entities_after", 0)
    tips_kept = summary.get("tips_kept", 0)
    tips_total = summary.get("tips_total", 0)
    return {
        "input": str(Path(input_path).resolve()),
        "tips_total": tips_total,
        "tips_kept": tips_kept,
        "tips_dropped_empty": summary.get("dropped_empty_tips", 0),
        "entities_before": entities_before,
        "entities_after": entities_after,
        "entities_removed": entities_before - entities_after,
        "avg_entities_per_tip_before": (entities_before / tips_total) if tips_total else 0.0,
        "avg_entities_per_tip_after": (entities_after / tips_kept) if tips_kept else 0.0,
        "dropped_by_score": summary.get("dropped_by_score", 0),
        "dropped_by_label": summary.get("dropped_by_label", 0),
        "missing_score": summary.get("missing_score", 0),
        "pruned_by_cap": summary.get("pruned_by_cap", 0),
        "dropped_tip_over_cap": summary.get("dropped_tip_over_cap", 0),
        "config": {
            "min_entity_score": args.min_entity_score,
            "label_min_scores": _parse_label_score_overrides(args.label_min_scores),
            "max_entities_per_tip": args.max_entities_per_tip,
            "drop_tips_over_max": args.drop_tips_over_max,
            "drop_empty_tips": args.drop_empty_tips,
            "score_fields": _parse_csv(args.score_fields),
            "allowed_labels": _parse_csv(args.allowed_labels),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune pseudolabel entities by score and per-tip density."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path.")
    parser.add_argument("--output-html", default="", help="Optional HTML output for cleaned rows.")
    parser.add_argument("--title", default="Pruned Pseudolabel Tips", help="HTML title when --output-html is used.")
    parser.add_argument("--min-entity-score", type=float, default=0.0, help="Minimum entity score to keep.")
    parser.add_argument(
        "--label-min-scores",
        default="",
        help="Optional comma-separated per-label score floors, e.g. Organization=0.8,Person=0.6",
    )
    parser.add_argument(
        "--max-entities-per-tip",
        type=int,
        default=0,
        help="Maximum number of entities to keep per tip (0 = unlimited).",
    )
    parser.add_argument(
        "--drop-tips-over-max",
        action="store_true",
        help="Drop entire tips that exceed --max-entities-per-tip instead of pruning to top-k.",
    )
    parser.add_argument(
        "--drop-empty-tips",
        action="store_true",
        help="Drop tips that end up with zero entities after pruning.",
    )
    parser.add_argument(
        "--score-fields",
        default=",".join(DEFAULT_SCORE_FIELDS),
        help="Comma-separated entity score fields to try in order.",
    )
    parser.add_argument(
        "--allowed-labels",
        default="",
        help="Optional comma-separated allowlist of labels to keep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    score_fields = _parse_csv(args.score_fields) or list(DEFAULT_SCORE_FIELDS)
    allowed_labels = set(_parse_csv(args.allowed_labels))
    label_min_scores = _parse_label_score_overrides(args.label_min_scores)

    cleaned_rows, counters = prune_rows(
        rows,
        min_entity_score=args.min_entity_score,
        label_min_scores=label_min_scores,
        max_entities_per_tip=args.max_entities_per_tip,
        score_fields=score_fields,
        allowed_labels=allowed_labels,
        drop_tips_over_max=args.drop_tips_over_max,
        drop_empty_tips=args.drop_empty_tips,
    )
    summary = build_summary(counters, args, args.input)

    write_jsonl(args.output_jsonl, cleaned_rows)
    print(f"Saved JSONL: {args.output_jsonl}")

    if args.output_html:
        render_html(cleaned_rows, output_path=args.output_html, title=args.title, max_reports=0)
        print(f"Saved HTML: {args.output_html}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved summary JSON: {args.summary_json}")

    print(f"Tips kept: {summary['tips_kept']}/{summary['tips_total']}")
    print(f"Entities before: {summary['entities_before']}")
    print(f"Entities after: {summary['entities_after']}")
    print(f"Entities removed: {summary['entities_removed']}")


if __name__ == "__main__":
    main()
