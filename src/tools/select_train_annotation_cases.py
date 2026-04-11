#!/usr/bin/env python3
"""Select training-oriented LLM adjudication candidates from prepare_adjudication_cases output."""

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

GENERIC_ENTITY_TEXTS = {
    "moradores",
    "morador",
    "homens",
    "homem",
    "mulheres",
    "mulher",
    "elementos",
    "elemento",
    "traficantes",
    "traficante",
    "autoridades",
    "polícia",
    "policia",
    "aplicativo",
    "estado",
}

NARRATIVE_MARKERS = (
    "trafico",
    "tráfico",
    "roubo",
    "assalto",
    "arma",
    "armado",
    "drog",
    "morro",
    "favela",
    "bairro",
    "rua",
    "travessa",
    "avenida",
    "estrada",
    "comunidade",
    "milicia",
    "milícia",
    "policia",
    "polícia",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select train-annotation candidates from prepare_adjudication_cases output."
    )
    parser.add_argument("--input", required=True, help="Input JSONL from prepare_adjudication_cases.py")
    parser.add_argument("--output-jsonl", required=True, help="Selected output JSONL")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-text-length", type=int, default=20)
    parser.add_argument("--max-text-length", type=int, default=900)
    parser.add_argument("--min-seed-entities", type=int, default=1)
    parser.add_argument("--max-seed-entities", type=int, default=4)
    parser.add_argument("--max-union-entities", type=int, default=8)
    parser.add_argument("--max-gliner2-noise-proxy", type=float, default=0.6)
    parser.add_argument("--min-agreement-ratio", type=float, default=0.15)
    parser.add_argument("--max-agreement-ratio", type=float, default=0.8)
    parser.add_argument("--require-agreed-or-baseline-seed", action="store_true")
    parser.add_argument("--penalize-generic-seeds", action="store_true")
    parser.add_argument("--drop-list-like-person-dumps", action="store_true")
    parser.add_argument("--drop-person-only-short-texts", action="store_true")
    parser.add_argument("--person-only-short-text-max-length", type=int, default=80)
    parser.add_argument(
        "--ranking-field",
        default="adjudication_priority_score",
        help="Primary numeric field used to rank rows after filtering. Falls back to internal trainability score when missing.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata(row: dict) -> dict:
    return row.get("metadata") or {}


def _text(row: dict) -> str:
    text = row.get("text")
    if isinstance(text, str):
        return text
    source = row.get("_source")
    if isinstance(source, dict) and isinstance(source.get("text"), str):
        return source["text"]
    return ""


def _review_seed_entities(row: dict) -> list[dict]:
    seeds = row.get("review_seed_entities")
    return seeds if isinstance(seeds, list) else []


def _seed_label_counts(row: dict) -> Counter:
    counts = Counter()
    for ent in _review_seed_entities(row):
        label = str(ent.get("label", "")).strip()
        if label:
            counts[label] += 1
    return counts


def _generic_seed_count(row: dict) -> int:
    count = 0
    for ent in _review_seed_entities(row):
        text = str(ent.get("text", "")).strip().lower()
        if text in GENERIC_ENTITY_TEXTS:
            count += 1
    return count


def _seed_origin_counts(row: dict) -> Counter:
    counts = Counter()
    for ent in _review_seed_entities(row):
        origin = str(ent.get("seed_origin", "")).strip()
        if origin:
            counts[origin] += 1
    return counts


def _normalized_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _has_narrative_markers(text: str) -> bool:
    lowered = _normalized_text(text)
    return any(marker in lowered for marker in NARRATIVE_MARKERS)


def _is_list_like_person_dump(row: dict) -> bool:
    text = _text(row)
    if not isinstance(text, str):
        return False
    separators = text.count(",") + text.count(";")
    if separators < 2:
        return False
    label_counts = _seed_label_counts(row)
    person_count = int(label_counts.get("Person", 0))
    entity_total = sum(int(v) for v in label_counts.values())
    if person_count < 2 or entity_total == 0:
        return False
    if person_count / entity_total < 0.8:
        return False
    if _has_narrative_markers(text):
        return False
    return True


def _is_person_only_short_text(row: dict, max_length: int) -> bool:
    text = _text(row)
    if not text.strip() or len(text) > max_length:
        return False
    label_counts = _seed_label_counts(row)
    entity_total = sum(int(v) for v in label_counts.values())
    if entity_total == 0:
        return False
    if label_counts.get("Person", 0) != entity_total:
        return False
    if _has_narrative_markers(text):
        return False
    return True


def row_passes_filters(row: dict, args) -> tuple[bool, list[str]]:
    reasons = []
    text = _text(row)
    metadata = _metadata(row)
    seeds = _review_seed_entities(row)

    if not text.strip():
        reasons.append("missing_text")
    if len(text) < args.min_text_length:
        reasons.append("text_too_short")
    if len(text) > args.max_text_length:
        reasons.append("text_too_long")
    if len(seeds) < args.min_seed_entities:
        reasons.append("too_few_seed_entities")
    if len(seeds) > args.max_seed_entities:
        reasons.append("too_many_seed_entities")

    agreement_ratio = _safe_float(metadata.get("agreement_ratio"))
    if agreement_ratio < args.min_agreement_ratio:
        reasons.append("agreement_too_low")
    if agreement_ratio > args.max_agreement_ratio:
        reasons.append("agreement_too_high")

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

    if args.require_agreed_or_baseline_seed:
        seed_origins = _seed_origin_counts(row)
        if seed_origins.get("agreed_exact", 0) <= 0 and seed_origins.get("baseline_high_score", 0) <= 0:
            reasons.append("no_stable_seed_origin")
    if args.drop_list_like_person_dumps and _is_list_like_person_dump(row):
        reasons.append("list_like_person_dump")
    if args.drop_person_only_short_texts and _is_person_only_short_text(row, args.person_only_short_text_max_length):
        reasons.append("person_only_short_text")

    return len(reasons) == 0, reasons


def compute_trainability_score(row: dict, args) -> tuple[float, list[str], dict]:
    text = _text(row)
    metadata = _metadata(row)
    seed_count = len(_review_seed_entities(row))
    seed_origins = _seed_origin_counts(row)

    agreement_ratio = _safe_float(metadata.get("agreement_ratio"))
    gliner2_noise_proxy = _safe_float(metadata.get("gliner2_noise_proxy"))
    baseline_coverage_proxy = _safe_float(metadata.get("baseline_coverage_proxy"))
    record_score = _safe_float(row.get("record_score"), _safe_float(row.get("_source", {}).get("record_score")))
    candidate_quality_score = _safe_float(row.get("candidate_quality_score"), _safe_float(row.get("_candidate_rank", {}).get("candidate_quality_score")))
    generic_seed_count = _generic_seed_count(row)
    text_length = len(text)

    ideal_seed_count = 2.5
    seed_count_penalty = abs(seed_count - ideal_seed_count)
    text_length_penalty = max(0.0, (text_length - 450.0) / 450.0)

    score = 0.0
    score += 2.0 * candidate_quality_score
    score += 1.5 * record_score
    score += 1.5 * baseline_coverage_proxy
    score += 1.25 * agreement_ratio
    score -= 2.0 * gliner2_noise_proxy
    score += 0.75 * seed_origins.get("agreed_exact", 0)
    score += 0.4 * seed_origins.get("baseline_high_score", 0)
    score -= 0.6 * seed_count_penalty
    score -= 0.8 * text_length_penalty

    reasons = []
    if agreement_ratio >= 0.3:
        reasons.append("moderate_or_high_agreement")
    if baseline_coverage_proxy >= 0.5:
        reasons.append("baseline_coverage_ok")
    if seed_origins.get("agreed_exact", 0) > 0:
        reasons.append("has_agreed_seed")
    if seed_origins.get("baseline_high_score", 0) > 0:
        reasons.append("has_high_score_seed")
    if gliner2_noise_proxy <= 0.5:
        reasons.append("noise_controlled")
    if seed_count in {2, 3}:
        reasons.append("seed_count_preferred")

    penalties = {
        "generic_seed_penalty": 0.0,
        "text_length_penalty": text_length_penalty,
        "seed_count_penalty": seed_count_penalty,
    }
    if args.penalize_generic_seeds and generic_seed_count > 0:
        penalty = 1.0 * generic_seed_count
        penalties["generic_seed_penalty"] = penalty
        score -= penalty
        reasons.append("generic_seed_penalty")

    return score, reasons, penalties


def _ranking_score(row: dict, args) -> float:
    value = _safe_float(row.get(args.ranking_field), None)
    if value is not None:
        return value
    return _safe_float((row.get("_train_adjudication_selection") or {}).get("score"))


def build_summary(input_rows, kept_pool, selected_rows, dropped_counter, args):
    def _avg(key):
        values = [_safe_float((_metadata(row)).get(key), None) for row in selected_rows]
        values = [v for v in values if v is not None]
        return (sum(values) / len(values)) if values else 0.0

    label_counts = Counter()
    origin_counts = Counter()
    for row in selected_rows:
        for ent in _review_seed_entities(row):
            label_counts[str(ent.get("label", ""))] += 1
            if ent.get("seed_origin"):
                origin_counts[str(ent.get("seed_origin"))] += 1

    return {
        "input_rows": len(input_rows),
        "rows_after_filters": len(kept_pool),
        "rows_selected": len(selected_rows),
        "dropped_counts": dict(dropped_counter),
        "selected_summary": {
            "avg_agreement_ratio": _avg("agreement_ratio"),
            "avg_gliner2_noise_proxy": _avg("gliner2_noise_proxy"),
            "avg_seed_entities": (sum(len(_review_seed_entities(row)) for row in selected_rows) / len(selected_rows))
            if selected_rows
            else 0.0,
            "avg_ranking_score": (
                sum(_ranking_score(row, args) for row in selected_rows) / len(selected_rows)
            )
            if selected_rows
            else 0.0,
            "avg_adjudication_priority_score": (
                sum(_safe_float(row.get("adjudication_priority_score")) for row in selected_rows) / len(selected_rows)
            )
            if selected_rows
            else 0.0,
            "review_seed_label_counts": dict(label_counts),
            "review_seed_origin_counts": dict(origin_counts),
        },
        "filters": {
            "min_text_length": args.min_text_length,
            "max_text_length": args.max_text_length,
            "drop_list_like_person_dumps": args.drop_list_like_person_dumps,
            "drop_person_only_short_texts": args.drop_person_only_short_texts,
            "person_only_short_text_max_length": args.person_only_short_text_max_length,
            "ranking_field": args.ranking_field,
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

        score, score_reasons, penalties = compute_trainability_score(row, args)
        enriched = dict(row)
        enriched["_train_adjudication_selection"] = {
            "score": score,
            "reasons": score_reasons,
            "penalties": penalties,
        }
        kept_pool.append(enriched)

    kept_pool.sort(
        key=lambda row: (
            -_ranking_score(row, args),
            -_safe_float((_metadata(row)).get("baseline_coverage_proxy")),
            -_safe_float((_metadata(row)).get("agreement_ratio")),
            _safe_float((_metadata(row)).get("gliner2_noise_proxy")),
            len(_review_seed_entities(row)),
            str(row.get("source_id", "")),
        )
    )

    selected_rows = kept_pool[: args.top_n]
    write_jsonl(args.output_jsonl, selected_rows)
    LOGGER.info("Saved selected train-adjudication candidates: %s", args.output_jsonl)

    if args.summary_json:
        summary = build_summary(rows, kept_pool, selected_rows, dropped_counter, args)
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Saved summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()
