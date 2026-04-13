#!/usr/bin/env python3
"""Score adjudication candidates for expected training utility."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl
from tools.select_train_annotation_cases import (
    _generic_seed_count,
    _has_narrative_markers,
    _is_list_like_person_dump,
    _is_person_only_short_text,
    _metadata,
    _review_seed_entities,
    _safe_float,
    _seed_label_counts,
    _seed_origin_counts,
    _text,
)

LOGGER = logging.getLogger(__name__)

LOCATIVE_MARKERS = (
    "rua",
    "travessa",
    "trav",
    "avenida",
    "av ",
    "bairro",
    "morro",
    "favela",
    "comunidade",
    "praca",
    "praça",
    "estrada",
    "rodovia",
)

DOMAIN_NOISE_MARKERS = (
    "taradas",
    "tarado",
    "celebridade",
    "cantor",
    "cantora",
    "atriz",
    "ator",
    "bbb",
)


def _separator_count(text: str) -> int:
    raw = str(text or "")
    return raw.count("/") + raw.count(";") + raw.count(":") + raw.count(",")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score adjudication candidates for training utility before LLM review.")
    parser.add_argument("--input", required=True, help="Input JSONL from prepare_adjudication_cases.py")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with adjudication priority scores")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON")
    parser.add_argument("--person-only-short-text-max-length", type=int, default=80)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _normalized_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _has_locative_markers(text: str) -> bool:
    lowered = _normalized_text(text)
    return any(marker in lowered for marker in LOCATIVE_MARKERS)


def _has_domain_noise(text: str) -> bool:
    lowered = _normalized_text(text)
    return any(marker in lowered for marker in DOMAIN_NOISE_MARKERS)


def _has_intersection_pattern(text: str) -> bool:
    lowered = _normalized_text(text)
    return (" esquina com " in lowered) or (" com rua " in lowered) or (" entre a rua " in lowered)


def _has_address_number(text: str) -> bool:
    lowered = _normalized_text(text)
    return any(token in lowered for token in (" n ", " n°", " numero ", " nº ", " no ", " lote ", " quadra "))


def _has_street_marker_seed(seeds: list[dict]) -> bool:
    for seed in seeds:
        label = str(seed.get("label", ""))
        text = _normalized_text(seed.get("text", ""))
        if label == "Location" and any(text.startswith(marker + " ") for marker in LOCATIVE_MARKERS):
            return True
    return False


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _band_score(value: float, *, sweet_min: float, sweet_max: float, near_min: float, near_max: float) -> float:
    if sweet_min <= value <= sweet_max:
        return 1.0
    if near_min <= value <= near_max:
        return 0.6
    return 0.2


def compute_adjudication_priority(row: dict, *, person_only_short_text_max_length: int) -> tuple[float, dict, dict, list[str]]:
    text = _text(row)
    metadata = _metadata(row)
    seeds = _review_seed_entities(row)
    seed_label_counts = _seed_label_counts(row)
    seed_origin_counts = _seed_origin_counts(row)
    location_seed_count = int(seed_label_counts.get("Location", 0))
    generic_seed_count = _generic_seed_count(row)
    agreement_ratio = _safe_float(metadata.get("agreement_ratio"))
    record_score = _safe_float(row.get("record_score"), _safe_float(row.get("_source", {}).get("record_score")))
    union_count = (
        int(metadata.get("entity_count_agreed", 0) or 0)
        + int(metadata.get("entity_count_baseline_only", 0) or 0)
        + int(metadata.get("entity_count_gliner2_only", 0) or 0)
    )
    text_length = len(text)
    seed_count = len(seeds)
    person_seed_count = int(seed_label_counts.get("Person", 0))
    organization_seed_count = int(seed_label_counts.get("Organization", 0))
    separator_count = _separator_count(text)
    location_only_case = location_seed_count > 0 and person_seed_count == 0 and organization_seed_count == 0
    compact_mixed_case = (
        seed_count <= 4
        and union_count <= 6
        and location_seed_count >= 1
        and person_seed_count == 0
        and organization_seed_count <= 1
    )
    street_marker_seed = _has_street_marker_seed(seeds)
    intersection_pattern = _has_intersection_pattern(text)
    address_number_pattern = _has_address_number(text)

    domain_score = 0.0
    if location_seed_count > 0:
        domain_score += 0.4
    if _has_narrative_markers(text):
        domain_score += 0.2
    if _has_locative_markers(text):
        domain_score += 0.2
    if seed_origin_counts.get("agreed_exact", 0) > 0 or seed_origin_counts.get("baseline_high_score", 0) > 0:
        domain_score += 0.1
    if _has_domain_noise(text):
        domain_score -= 0.2
    if _is_list_like_person_dump(row):
        domain_score -= 0.4
    if _is_person_only_short_text(row, person_only_short_text_max_length):
        domain_score -= 0.4
    domain_score = _clamp(domain_score)

    disagreement_midband_score = _band_score(
        agreement_ratio,
        sweet_min=0.2,
        sweet_max=0.7,
        near_min=0.1,
        near_max=0.85,
    )
    record_score_midband_score = _band_score(
        record_score,
        sweet_min=0.35,
        sweet_max=0.80,
        near_min=0.20,
        near_max=0.92,
    )

    if location_seed_count >= 2:
        location_seed_score = 1.0
    elif location_seed_count == 1:
        location_seed_score = 0.6
    else:
        location_seed_score = 0.0

    adjudicability_score = 0.0
    if 1 <= seed_count <= 4:
        adjudicability_score += 0.5
    elif 5 <= seed_count <= 6:
        adjudicability_score += 0.2
    if 30 <= text_length <= 500:
        adjudicability_score += 0.3
    elif 20 <= text_length <= 700:
        adjudicability_score += 0.15
    if union_count <= 8:
        adjudicability_score += 0.2
    elif union_count <= 10:
        adjudicability_score += 0.1
    adjudicability_score = _clamp(adjudicability_score)

    micro_edit_score = 0.0
    if 1 <= seed_count <= 4:
        micro_edit_score += 0.4
    elif seed_count == 5:
        micro_edit_score += 0.15
    if 1 <= location_seed_count <= 3:
        micro_edit_score += 0.3
    elif location_seed_count == 4:
        micro_edit_score += 0.15
    if union_count <= 6:
        micro_edit_score += 0.2
    elif union_count <= 8:
        micro_edit_score += 0.1
    if separator_count <= 2:
        micro_edit_score += 0.1
    micro_edit_score = _clamp(micro_edit_score)

    canonical_address_score = 0.0
    if _has_locative_markers(text):
        canonical_address_score += 0.35
    if street_marker_seed:
        canonical_address_score += 0.25
    if intersection_pattern:
        canonical_address_score += 0.2
    if address_number_pattern:
        canonical_address_score += 0.1
    if location_only_case:
        canonical_address_score += 0.1
    canonical_address_score = _clamp(canonical_address_score)

    small_clean_edit_score = 0.0
    if seed_count <= 4:
        small_clean_edit_score += 0.3
    if union_count <= 6:
        small_clean_edit_score += 0.25
    if location_only_case:
        small_clean_edit_score += 0.2
    if person_seed_count == 0 and organization_seed_count <= 1:
        small_clean_edit_score += 0.15
    if generic_seed_count == 0:
        small_clean_edit_score += 0.1
    small_clean_edit_score = _clamp(small_clean_edit_score)

    penalties = {
        "generic_seed_penalty": 0.15 * generic_seed_count,
        "list_like_person_penalty": 0.8 if _is_list_like_person_dump(row) else 0.0,
        "person_only_short_penalty": 0.8 if _is_person_only_short_text(row, person_only_short_text_max_length) else 0.0,
        "domain_noise_penalty": 0.25 if _has_domain_noise(text) else 0.0,
        "separator_density_penalty": 0.25 if separator_count >= 6 else (0.1 if separator_count >= 4 else 0.0),
        "location_expansion_risk_penalty": (
            0.35
            if location_seed_count >= 5
            else (0.15 if location_seed_count == 4 and person_seed_count == 0 and organization_seed_count == 0 else 0.0)
        ),
        "seed_count_risk_penalty": 0.2 if seed_count >= 6 else 0.0,
        "mixed_label_risk_penalty": (
            0.28
            if (
                ((person_seed_count >= 1 or organization_seed_count >= 1) and location_seed_count >= 2)
                and not compact_mixed_case
            )
            else 0.0
        ),
    }

    score = 0.0
    score += 0.28 * domain_score
    score += 0.25 * disagreement_midband_score
    score += 0.10 * record_score_midband_score
    score += 0.15 * location_seed_score
    score += 0.10 * adjudicability_score
    score += 0.04 * micro_edit_score
    score += 0.08 * canonical_address_score
    score += 0.05 * small_clean_edit_score
    score -= sum(penalties.values())

    reasons = []
    if domain_score >= 0.6:
        reasons.append("domain_aligned")
    if disagreement_midband_score >= 1.0:
        reasons.append("agreement_in_midband")
    if record_score_midband_score >= 1.0:
        reasons.append("record_score_in_midband")
    if location_seed_score >= 0.6:
        reasons.append("has_location_seed")
    if adjudicability_score >= 0.8:
        reasons.append("easy_to_adjudicate")
    if micro_edit_score >= 0.8:
        reasons.append("small_fix_profile")
    if canonical_address_score >= 0.4:
        reasons.append("canonical_address_profile")
    if small_clean_edit_score >= 0.7:
        reasons.append("small_clean_edit_profile")
    if penalties["generic_seed_penalty"] > 0:
        reasons.append("generic_seed_penalty")
    if penalties["list_like_person_penalty"] > 0:
        reasons.append("list_like_person_penalty")
    if penalties["person_only_short_penalty"] > 0:
        reasons.append("person_only_short_penalty")
    if penalties["domain_noise_penalty"] > 0:
        reasons.append("domain_noise_penalty")
    if penalties["separator_density_penalty"] > 0:
        reasons.append("separator_density_penalty")
    if penalties["location_expansion_risk_penalty"] > 0:
        reasons.append("location_expansion_risk_penalty")
    if penalties["seed_count_risk_penalty"] > 0:
        reasons.append("seed_count_risk_penalty")

    components = {
        "domain_score": domain_score,
        "disagreement_midband_score": disagreement_midband_score,
        "record_score_midband_score": record_score_midband_score,
        "location_seed_score": location_seed_score,
        "adjudicability_score": adjudicability_score,
        "micro_edit_score": micro_edit_score,
        "canonical_address_score": canonical_address_score,
        "small_clean_edit_score": small_clean_edit_score,
    }
    diagnostics = {
        "text_length": text_length,
        "seed_count": seed_count,
        "location_seed_count": location_seed_count,
        "person_seed_count": person_seed_count,
        "generic_seed_count": generic_seed_count,
        "agreement_ratio": agreement_ratio,
        "record_score": record_score,
        "union_count": union_count,
        "separator_count": separator_count,
        "location_only_case": location_only_case,
        "compact_mixed_case": compact_mixed_case,
        "street_marker_seed": street_marker_seed,
        "intersection_pattern": intersection_pattern,
        "address_number_pattern": address_number_pattern,
    }
    return score, components, penalties, reasons


def build_summary(rows: list[dict]) -> dict:
    def _avg_top_level(key: str) -> float:
        values = [_safe_float(row.get(key), None) for row in rows]
        values = [value for value in values if value is not None]
        return (sum(values) / len(values)) if values else 0.0

    component_keys = (
        "domain_score",
        "disagreement_midband_score",
        "record_score_midband_score",
        "location_seed_score",
        "adjudicability_score",
        "micro_edit_score",
        "canonical_address_score",
        "small_clean_edit_score",
    )
    penalty_keys = (
        "generic_seed_penalty",
        "list_like_person_penalty",
        "person_only_short_penalty",
        "domain_noise_penalty",
        "separator_density_penalty",
        "location_expansion_risk_penalty",
        "seed_count_risk_penalty",
        "mixed_label_risk_penalty",
    )
    reason_counts = Counter()
    for row in rows:
        for reason in row.get("_adjudication_priority", {}).get("reasons", []):
            reason_counts[reason] += 1

    return {
        "rows_total": len(rows),
        "summary": {
            "avg_adjudication_priority_score": _avg_top_level("adjudication_priority_score"),
            "avg_domain_score": _avg_top_level("domain_score"),
            "avg_disagreement_midband_score": _avg_top_level("disagreement_midband_score"),
            "avg_record_score_midband_score": _avg_top_level("record_score_midband_score"),
            "avg_location_seed_score": _avg_top_level("location_seed_score"),
            "avg_adjudicability_score": _avg_top_level("adjudicability_score"),
            "reason_counts": dict(reason_counts),
            "penalty_averages": {
                key: (
                    sum(_safe_float((row.get("_adjudication_priority", {}).get("penalties", {})).get(key), 0.0) for row in rows)
                    / len(rows)
                )
                if rows
                else 0.0
                for key in penalty_keys
            },
            "component_averages": {
                key: (
                    sum(_safe_float((row.get("_adjudication_priority", {}).get("components", {})).get(key), 0.0) for row in rows)
                    / len(rows)
                )
                if rows
                else 0.0
                for key in component_keys
            },
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    rows = read_json_or_jsonl(args.input)
    output_rows = []
    for row in rows:
        score, components, penalties, reasons = compute_adjudication_priority(
            row,
            person_only_short_text_max_length=args.person_only_short_text_max_length,
        )
        enriched = dict(row)
        enriched["adjudication_priority_score"] = score
        enriched.update(components)
        enriched["_adjudication_priority"] = {
            "score": score,
            "components": components,
            "penalties": penalties,
            "reasons": reasons,
        }
        output_rows.append(enriched)

    write_jsonl(args.output_jsonl, output_rows)
    LOGGER.info("Saved adjudication-priority-scored rows: %s", args.output_jsonl)

    if args.summary_json:
        summary = build_summary(output_rows)
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Saved summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()
