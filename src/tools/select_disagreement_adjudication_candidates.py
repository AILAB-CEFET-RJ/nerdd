#!/usr/bin/env python3
"""Select disagreement-focused candidates for Codex/GPT adjudication."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)

GENERIC_ORGS = {
    "polícia",
    "policia",
    "polícia militar",
    "policia militar",
    "prefeitura",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select disagreement-heavy adjudication candidates from prepare_adjudication_cases output."
    )
    parser.add_argument("--input", required=True, help="Input JSONL from prepare_adjudication_cases.py")
    parser.add_argument("--output-jsonl", required=True, help="Selected output JSONL")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--max-text-length", type=int, default=1200)
    parser.add_argument("--min-seed-entities", type=int, default=1)
    parser.add_argument("--max-seed-entities", type=int, default=3)
    parser.add_argument("--max-gliner2-noise-proxy", type=float, default=0.75)
    parser.add_argument("--max-agreement-ratio", type=float, default=0.34)
    parser.add_argument("--require-conflict-or-low-agreement", action="store_true")
    parser.add_argument("--penalize-generic-orgs", action="store_true")
    parser.add_argument("--max-union-entities", type=int, default=8)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata(row: dict) -> dict:
    return row.get("metadata") or {}


def _review_seed_entities(row: dict) -> list[dict]:
    seeds = row.get("review_seed_entities")
    return seeds if isinstance(seeds, list) else []


def _text(row: dict) -> str:
    text = row.get("text")
    if isinstance(text, str):
        return text
    source = row.get("_source")
    if isinstance(source, dict) and isinstance(source.get("text"), str):
        return source["text"]
    return ""


def _has_generic_org_seed(row: dict) -> bool:
    for ent in _review_seed_entities(row):
        if ent.get("label") != "Organization":
            continue
        if str(ent.get("text", "")).strip().lower() in GENERIC_ORGS:
            return True
    return False


def row_passes_filters(row: dict, args) -> tuple[bool, list[str]]:
    reasons = []
    text = _text(row)
    metadata = _metadata(row)
    seeds = _review_seed_entities(row)

    if not text.strip():
        reasons.append("missing_text")
    if len(text) > args.max_text_length:
        reasons.append("text_too_long")
    if len(seeds) < args.min_seed_entities:
        reasons.append("too_few_seed_entities")
    if len(seeds) > args.max_seed_entities:
        reasons.append("too_many_seed_entities")

    gliner2_noise_proxy = _safe_float(metadata.get("gliner2_noise_proxy"))
    if gliner2_noise_proxy > args.max_gliner2_noise_proxy:
        reasons.append("gliner2_noise_too_high")

    union_count = (
        int(metadata.get("entity_count_agreed", 0) or 0)
        + int(metadata.get("entity_count_baseline_only", 0) or 0)
        + int(metadata.get("entity_count_gliner2_only", 0) or 0)
    )
    if union_count > args.max_union_entities:
        reasons.append("too_many_union_entities")

    if args.require_conflict_or_low_agreement:
        conflicts = int(metadata.get("entity_count_conflicts", 0) or 0)
        agreement_ratio = _safe_float(metadata.get("agreement_ratio"))
        if conflicts <= 0 and agreement_ratio > args.max_agreement_ratio:
            reasons.append("no_conflict_and_agreement_too_high")

    return len(reasons) == 0, reasons


def compute_disagreement_score(row: dict, args) -> tuple[float, list[str], float]:
    metadata = _metadata(row)
    reasons = []

    conflicts = int(metadata.get("entity_count_conflicts", 0) or 0)
    baseline_only = int(metadata.get("entity_count_baseline_only", 0) or 0)
    gliner2_only = int(metadata.get("entity_count_gliner2_only", 0) or 0)
    agreement_ratio = _safe_float(metadata.get("agreement_ratio"))
    gliner2_noise_proxy = _safe_float(metadata.get("gliner2_noise_proxy"))
    seed_count = len(_review_seed_entities(row))

    score = 0.0
    score += 4.0 * conflicts
    score += 2.0 * min(baseline_only + gliner2_only, 4)
    score += 2.0 * (1.0 - agreement_ratio)
    score -= 2.0 * gliner2_noise_proxy
    score -= 0.75 * max(0, seed_count - 2)

    if conflicts > 0:
        reasons.append("has_conflict")
    if agreement_ratio <= args.max_agreement_ratio:
        reasons.append("low_agreement_ratio")
    if (baseline_only + gliner2_only) > 0:
        reasons.append("has_model_disagreement")
    if seed_count <= args.max_seed_entities:
        reasons.append("seed_entities_within_limit")

    generic_org_penalty = 0.0
    if args.penalize_generic_orgs and _has_generic_org_seed(row):
        generic_org_penalty = 1.5
        score -= generic_org_penalty
        reasons.append("generic_org_penalty")

    return score, reasons, generic_org_penalty


def build_summary(input_rows, kept_pool, selected_rows, dropped_counter):
    def _avg(key):
        values = [_safe_float((_metadata(row)).get(key), None) for row in selected_rows]
        values = [v for v in values if v is not None]
        return (sum(values) / len(values)) if values else 0.0

    label_counts = Counter()
    for row in selected_rows:
        for ent in _review_seed_entities(row):
            label_counts[str(ent.get("label", ""))] += 1

    conflict_rows = sum(1 for row in selected_rows if int(_metadata(row).get("entity_count_conflicts", 0) or 0) > 0)

    return {
        "input_rows": len(input_rows),
        "rows_after_filters": len(kept_pool),
        "rows_selected": len(selected_rows),
        "dropped_counts": dict(dropped_counter),
        "selected_summary": {
            "avg_agreement_ratio": _avg("agreement_ratio"),
            "avg_gliner2_noise_proxy": _avg("gliner2_noise_proxy"),
            "avg_seed_entities": (sum(len(_review_seed_entities(row)) for row in selected_rows) / len(selected_rows))
            if selected_rows else 0.0,
            "rows_with_conflicts": conflict_rows,
            "review_seed_label_counts": dict(label_counts),
        },
    }


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    rows = read_json_or_jsonl(args.input)
    dropped_counter = Counter()
    kept_pool = []

    for row in rows:
        ok, drop_reasons = row_passes_filters(row, args)
        if not ok:
            dropped_counter.update(drop_reasons)
            continue

        score, score_reasons, generic_org_penalty = compute_disagreement_score(row, args)
        enriched = dict(row)
        enriched["_disagreement_selection"] = {
            "score": score,
            "reasons": score_reasons,
            "generic_org_penalty": generic_org_penalty,
        }
        kept_pool.append(enriched)

    kept_pool.sort(
        key=lambda row: (
            -row["_disagreement_selection"]["score"],
            _safe_float((_metadata(row)).get("agreement_ratio")),
            -int((_metadata(row)).get("entity_count_conflicts", 0) or 0),
            len(_review_seed_entities(row)),
            str(row.get("source_id", "")),
        )
    )

    selected_rows = kept_pool[: args.top_n]
    write_jsonl(args.output_jsonl, selected_rows)
    LOGGER.info("Saved selected disagreement candidates: %s", args.output_jsonl)

    if args.summary_json:
        summary = build_summary(rows, kept_pool, selected_rows, dropped_counter)
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Saved summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()
