#!/usr/bin/env python3
"""Retrieve candidate records structurally similar to known-good train-annotation cases."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl
from tools.score_adjudication_candidates import _separator_count, compute_adjudication_priority
from tools.select_train_annotation_cases import (
    _generic_seed_count,
    _has_locative_markers,
    _has_narrative_markers,
    _metadata,
    _safe_float,
    _text,
    row_passes_filters,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank large-corpus adjudication candidates by similarity to known-good train-annotation cases."
    )
    parser.add_argument("--input", required=True, help="Candidate pool JSONL/JSON.")
    parser.add_argument("--positive-examples", required=True, help="JSONL/JSON with known-good prototype cases.")
    parser.add_argument(
        "--negative-examples",
        default="",
        help="Optional JSONL/JSON with known-bad prototype cases used as repulsion examples.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Ranked output JSONL.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--exclude-source-ids-from",
        action="append",
        default=[],
        help="Repeatable JSONL/JSON source whose source_id values will be excluded from retrieval output.",
    )
    parser.add_argument("--person-only-short-text-max-length", type=int, default=80)
    parser.add_argument("--min-text-length", type=int, default=20)
    parser.add_argument("--max-text-length", type=int, default=900)
    parser.add_argument("--min-seed-entities", type=int, default=1)
    parser.add_argument("--max-seed-entities", type=int, default=6)
    parser.add_argument("--max-union-entities", type=int, default=16)
    parser.add_argument("--max-baseline-entities", type=int, default=28)
    parser.add_argument("--max-gliner2-noise-proxy", type=float, default=0.7)
    parser.add_argument("--max-person-seed-ratio", type=float, default=0.9)
    parser.add_argument("--min-agreement-ratio", type=float, default=0.0)
    parser.add_argument("--max-agreement-ratio", type=float, default=0.9)
    parser.add_argument("--require-agreed-or-baseline-seed", action="store_true", default=True)
    parser.add_argument("--penalize-generic-seeds", action="store_true", default=True)
    parser.add_argument("--drop-list-like-person-dumps", action="store_true", default=True)
    parser.add_argument("--drop-person-only-short-texts", action="store_true", default=True)
    parser.add_argument("--require-location-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-domain-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _seed_entities(row: dict) -> list[dict]:
    seeds = row.get("review_seed_entities")
    if isinstance(seeds, list):
        return seeds
    adjudication = row.get("adjudication")
    if isinstance(adjudication, dict):
        entities_final = adjudication.get("entities_final")
        if isinstance(entities_final, list):
            return entities_final
    entities = row.get("entities")
    return entities if isinstance(entities, list) else []


def _tokenize(text: str) -> list[str]:
    return [tok for tok in str(text).strip().lower().split() if tok]


def _has_degenerate_merged_span(seeds: list[dict]) -> bool:
    texts = [str(ent.get("text", "")).strip() for ent in seeds if str(ent.get("text", "")).strip()]
    normalized = {_text.lower(): _text for _text in texts}
    token_sets = {text.lower(): set(_tokenize(text)) for text in texts}
    for text in texts:
        toks = _tokenize(text)
        if len(toks) < 2:
            continue
        token_set = set(toks)
        for other in texts:
            if text == other:
                continue
            other_toks = _tokenize(other)
            if len(other_toks) != 1:
                continue
            if set(other_toks).issubset(token_set):
                return True
    return False


def _has_obvious_truncation(seeds: list[dict]) -> bool:
    suspicious_singletons = {
        "petropolis",
        "delfim",
        "valadares",
        "janeiro",
        "rj",
        "belford",
        "amazonas",
    }
    for ent in seeds:
        text = str(ent.get("text", "")).strip().lower()
        if text in suspicious_singletons:
            return True
    return False


def _feature_dict(row: dict, *, person_only_short_text_max_length: int) -> dict[str, float]:
    text = _text(row)
    metadata = _metadata(row)
    seeds = _seed_entities(row)
    seed_count = len(seeds)
    label_counts = {"Location": 0, "Person": 0, "Organization": 0}
    token_lengths = []
    for ent in seeds:
        label = str(ent.get("label", "")).strip()
        if label in label_counts:
            label_counts[label] += 1
        mention = str(ent.get("text", "")).strip()
        if mention:
            token_lengths.append(len([tok for tok in mention.split() if tok]))

    union_count = (
        int(metadata.get("entity_count_agreed", 0) or 0)
        + int(metadata.get("entity_count_baseline_only", 0) or 0)
        + int(metadata.get("entity_count_gliner2_only", 0) or 0)
    )
    baseline_count = int(metadata.get("entity_count_baseline", 0) or 0)
    separator_count = _separator_count(text)
    agreement_ratio = _safe_float(metadata.get("agreement_ratio"))
    priority_score = _safe_float(row.get("adjudication_priority_score"))
    if priority_score == 0.0:
        priority_score, _, _, _ = compute_adjudication_priority(
            row, person_only_short_text_max_length=person_only_short_text_max_length
        )

    denom = max(seed_count, 1)
    avg_seed_tokens = (sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0
    multi_token_location_count = 0
    for ent in seeds:
        if str(ent.get("label", "")).strip() != "Location":
            continue
        mention = str(ent.get("text", "")).strip()
        if len([tok for tok in mention.split() if tok]) >= 2:
            multi_token_location_count += 1
    return {
        "text_length_norm": min(len(text), 500) / 500.0,
        "seed_count_norm": min(seed_count, 8) / 8.0,
        "location_ratio": label_counts["Location"] / denom,
        "person_ratio": label_counts["Person"] / denom,
        "organization_ratio": label_counts["Organization"] / denom,
        "agreement_ratio": min(max(agreement_ratio, 0.0), 1.0),
        "union_count_norm": min(union_count, 16) / 16.0,
        "baseline_count_norm": min(baseline_count, 16) / 16.0,
        "separator_norm": min(separator_count, 10) / 10.0,
        "generic_seed_ratio": _generic_seed_count({"review_seed_entities": seeds}) / denom,
        "has_narrative": 1.0 if _has_narrative_markers(text) else 0.0,
        "has_locative": 1.0 if _has_locative_markers(text) else 0.0,
        "avg_seed_tokens_norm": min(avg_seed_tokens, 4.0) / 4.0,
        "multi_token_location_ratio": multi_token_location_count / max(label_counts["Location"], 1),
        "priority_score": min(max(priority_score, 0.0), 1.0),
    }


def _mean_feature_vector(rows: list[dict], *, person_only_short_text_max_length: int) -> dict[str, float]:
    if not rows:
        raise ValueError("Prototype set must contain at least one row.")
    vectors = [_feature_dict(row, person_only_short_text_max_length=person_only_short_text_max_length) for row in rows]
    keys = list(vectors[0].keys())
    return {key: sum(vec[key] for vec in vectors) / len(vectors) for key in keys}


def _similarity(vec: dict[str, float], prototype: dict[str, float]) -> float:
    keys = prototype.keys()
    mean_abs_diff = sum(abs(vec[key] - prototype[key]) for key in keys) / len(list(keys))
    return max(0.0, 1.0 - mean_abs_diff)


def _load_excluded_source_ids(paths: list[str]) -> set[str]:
    excluded = set()
    for raw_path in paths:
        for row in read_json_or_jsonl(raw_path):
            source_id = str(row.get("source_id", "")).strip()
            if source_id:
                excluded.add(source_id)
    return excluded


def _build_summary(rows: list[dict], positive_count: int, negative_count: int, excluded_count: int) -> dict:
    return {
        "rows_total": len(rows),
        "positive_examples": positive_count,
        "negative_examples": negative_count,
        "excluded_source_ids": excluded_count,
        "top_scores_preview": [
            {
                "source_id": row.get("source_id"),
                "similarity_to_positive": row.get("similarity_to_positive"),
                "similarity_to_negative": row.get("similarity_to_negative"),
                "prototype_similarity_score": row.get("prototype_similarity_score"),
            }
            for row in rows[:10]
        ],
    }


def _selector_filter_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        min_seed_entities=args.min_seed_entities,
        max_seed_entities=args.max_seed_entities,
        max_union_entities=args.max_union_entities,
        max_baseline_entities=args.max_baseline_entities,
        max_gliner2_noise_proxy=args.max_gliner2_noise_proxy,
        max_person_seed_ratio=args.max_person_seed_ratio,
        min_agreement_ratio=args.min_agreement_ratio,
        max_agreement_ratio=args.max_agreement_ratio,
        require_agreed_or_baseline_seed=args.require_agreed_or_baseline_seed,
        penalize_generic_seeds=args.penalize_generic_seeds,
        drop_list_like_person_dumps=args.drop_list_like_person_dumps,
        drop_person_only_short_texts=args.drop_person_only_short_texts,
        person_only_short_text_max_length=args.person_only_short_text_max_length,
        require_location_seed=args.require_location_seed,
        require_domain_context=args.require_domain_context,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    candidates = read_json_or_jsonl(args.input)
    positives = read_json_or_jsonl(args.positive_examples)
    negatives = read_json_or_jsonl(args.negative_examples) if args.negative_examples else []
    excluded_source_ids = _load_excluded_source_ids(args.exclude_source_ids_from)

    positive_proto = _mean_feature_vector(
        positives, person_only_short_text_max_length=args.person_only_short_text_max_length
    )
    negative_proto = (
        _mean_feature_vector(negatives, person_only_short_text_max_length=args.person_only_short_text_max_length)
        if negatives
        else None
    )
    filter_args = _selector_filter_args(args)

    ranked = []
    for row in candidates:
        source_id = str(row.get("source_id", "")).strip()
        if source_id and source_id in excluded_source_ids:
            continue
        passes_filters, _ = row_passes_filters(row, filter_args)
        if not passes_filters:
            continue
        seeds = _seed_entities(row)
        vec = _feature_dict(row, person_only_short_text_max_length=args.person_only_short_text_max_length)
        seed_count = len(seeds)
        if seed_count < 2:
            continue
        record_score = _safe_float(row.get("record_score"), 0.0)
        if record_score <= 0.01:
            continue
        if _has_degenerate_merged_span(seeds):
            continue
        if _has_obvious_truncation(seeds):
            continue
        sim_pos = _similarity(vec, positive_proto)
        sim_neg = _similarity(vec, negative_proto) if negative_proto else 0.0
        penalties = 0.0
        if vec["priority_score"] <= 0.05:
            penalties += 0.20
        if vec["seed_count_norm"] <= 0.25:
            penalties += 0.10
        if vec["multi_token_location_ratio"] <= 0.25:
            penalties += 0.10
        if vec["location_ratio"] >= 0.95 and vec["seed_count_norm"] >= 0.625:
            penalties += 0.15
        final_score = sim_pos - (0.35 * sim_neg) + (0.15 * vec["priority_score"]) - penalties
        enriched = dict(row)
        enriched["prototype_similarity_score"] = round(final_score, 6)
        enriched["similarity_to_positive"] = round(sim_pos, 6)
        enriched["similarity_to_negative"] = round(sim_neg, 6)
        enriched["_prototype_similarity"] = {
            "features": vec,
            "penalties": round(penalties, 6),
            "positive_prototype": positive_proto,
            "negative_prototype": negative_proto,
        }
        ranked.append(enriched)

    ranked.sort(
        key=lambda row: (
            -_safe_float(row.get("prototype_similarity_score")),
            -_safe_float(row.get("similarity_to_positive")),
            -_safe_float(row.get("adjudication_priority_score")),
        )
    )

    if args.top_n > 0:
        ranked = ranked[: args.top_n]

    write_jsonl(args.output_jsonl, ranked)
    LOGGER.info("Wrote ranked prototype-similar rows: %s", args.output_jsonl)

    if args.summary_json:
        summary = _build_summary(
            ranked,
            positive_count=len(positives),
            negative_count=len(negatives),
            excluded_count=len(excluded_source_ids),
        )
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Wrote summary: %s", args.summary_json)


if __name__ == "__main__":
    main()
