#!/usr/bin/env python3
"""Build LLM adjudication inputs by merging baseline and GLiNER2 predictions.

This utility prepares a compact JSONL for downstream LLM-assisted adjudication.
It treats the existing predictions in the input rows as the baseline and augments
each record with GLiNER2 base predictions plus agreement/conflict metadata.

Typical use case:
- input: ranked pseudolabel candidates JSONL produced from the baseline pipeline
- process: run GLiNER2 base on the same texts, match entities conservatively
- output: JSONL containing baseline entities, GLiNER2 entities, agreements,
  conflicts, review seeds, and per-record summary metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.paths import resolve_repo_artifact_path
from tools.inspect_dense_tips import get_spans, get_text, read_json_or_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)

DEFAULT_LABELS = ["Person", "Location", "Organization"]
DEFAULT_ENTITY_TYPES = ["person", "location", "organization"]
DEFAULT_SOURCE_ID_FIELDS = ("sample_id", "id", "source_id")
DEFAULT_LOCATION_METADATA_FIELDS = ("logradouroLocal", "bairroLocal", "cidadeLocal", "pontodeReferenciaLocal")
LOCATION_GENERIC_TOKENS = {
    "bairro",
    "cidade",
    "centeo",
    "centro",
    "escadao",
    "estado",
    "hurgente",
    "local",
    "logradouro",
    "endereco",
    "endereço",
    "numero",
    "número",
    "quadra",
    "lote",
    "rua",
    "avenida",
    "travessa",
    "estrada",
    "rodovia",
    "praca",
    "proximo",
    "sj",
    "praça",
    "urgente",
}
GENERIC_ENTITY_TOKENS = {"proximo", "urgente", "hurgente", "local"}
LOCATION_CONNECTOR_TOKENS = {"a", "ao", "aos", "as", "e", "da", "das", "de", "do", "dos"}


def _parse_csv(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collapse_spaces(text: str) -> str:
    return " ".join(str(text).split())


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_entity_text(text: str) -> str:
    text = _collapse_spaces(str(text).strip().lower())
    text = _strip_accents(text)
    return text.strip(" \t\r\n.,;:!?()[]{}\"'")


def _normalized_tokens(text: str) -> list[str]:
    normalized = normalize_entity_text(text)
    if not normalized:
        return []
    return [token for token in normalized.split() if token]


def _extract_location_metadata_terms(row: dict) -> set[str]:
    terms: set[str] = set()
    for key in DEFAULT_LOCATION_METADATA_FIELDS:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        raw_parts = [value]
        raw_parts.extend(part for part in re.split(r"[\n,;()]+", value) if part and part.strip())
        for part in raw_parts:
            normalized = normalize_entity_text(part)
            if len(normalized) >= 4:
                terms.add(normalized)
    return terms


def _tokens_are_subsequence(needle_tokens: list[str], haystack_tokens: list[str]) -> bool:
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return False
    window = len(needle_tokens)
    for start in range(len(haystack_tokens) - window + 1):
        if haystack_tokens[start : start + window] == needle_tokens:
            return True
    return False


def _metadata_supports_entity(entity_norm: str, metadata_terms: set[str]) -> bool:
    entity_tokens = _normalized_tokens(entity_norm)
    if not entity_tokens:
        return False
    for term in metadata_terms:
        term_tokens = _normalized_tokens(term)
        if entity_norm == term:
            return True
        if _tokens_are_subsequence(entity_tokens, term_tokens):
            return True
        if _tokens_are_subsequence(term_tokens, entity_tokens):
            return True
    return False


def _is_viable_location_seed(entity: dict, *, location_metadata_terms: set[str], require_exact_metadata_for_single_token: bool) -> bool:
    entity_norm = str(entity.get("text_norm", ""))
    tokens = _normalized_tokens(entity_norm)
    if not tokens:
        return False
    if any(token in LOCATION_GENERIC_TOKENS for token in tokens):
        return False
    if tokens[0] in LOCATION_CONNECTOR_TOKENS:
        return False
    if len(tokens) == 1:
        if len(tokens[0]) < 6:
            return False
        if require_exact_metadata_for_single_token and entity_norm not in location_metadata_terms:
            return False
    return True


def _is_viable_entity(entity: dict) -> bool:
    text = str(entity.get("text", "")).strip()
    text_norm = str(entity.get("text_norm", ""))
    label = str(entity.get("label", "")).strip()
    tokens = _normalized_tokens(text_norm)

    if not text or not text_norm or not tokens:
        return False
    if len(text_norm) < 2:
        return False
    if len(tokens) == 1 and tokens[0] in GENERIC_ENTITY_TOKENS:
        return False
    if label == "Location":
        return _is_viable_location_seed(
            entity,
            location_metadata_terms=set(),
            require_exact_metadata_for_single_token=False,
        )
    return True


def _matches_location_metadata(entity: dict, metadata_terms: set[str], *, min_chars: int) -> bool:
    if entity.get("label") != "Location":
        return False
    entity_norm = str(entity.get("text_norm", ""))
    if len(entity_norm) < min_chars:
        return False
    return _metadata_supports_entity(entity_norm, metadata_terms)


def _entity_score(entity: dict) -> float | None:
    if not isinstance(entity, dict):
        return None
    for key in ("score_context_boosted", "score_calibrated", "score", "confidence", "probability", "score_ts"):
        score = _safe_float(entity.get(key))
        if score is not None:
            return score
    return None


def _normalize_entity(entity: dict, *, source: str) -> dict:
    text = str(entity.get("text", "")).strip()
    label = str(entity.get("label", "")).strip()
    normalized = {
        "text": text,
        "label": label,
        "source": source,
        "text_norm": normalize_entity_text(text),
    }
    start = entity.get("start")
    end = entity.get("end")
    if isinstance(start, int):
        normalized["start"] = start
    if isinstance(end, int):
        normalized["end"] = end
    ner_score = _entity_score(entity)
    if ner_score is not None:
        normalized["ner_score"] = ner_score
    return normalized


def _label_allowed(entity: dict, allowed_labels: set[str]) -> bool:
    return str(entity.get("label", "")).strip() in allowed_labels


def normalize_baseline_entities(entities: list[dict], allowed_labels: set[str]) -> list[dict]:
    normalized = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        normalized_entity = _normalize_entity(entity, source="baseline")
        if _label_allowed(normalized_entity, allowed_labels) and _is_viable_entity(normalized_entity):
            normalized.append(normalized_entity)
    return normalized


def normalize_gliner2_entities(entities: list[dict], allowed_labels: set[str]) -> list[dict]:
    normalized = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        normalized_entity = _normalize_entity(entity, source="gliner2")
        if _label_allowed(normalized_entity, allowed_labels) and _is_viable_entity(normalized_entity):
            normalized.append(normalized_entity)
    return normalized


def _is_soft_match(a: dict, b: dict, *, min_len: int) -> bool:
    if a.get("label") != b.get("label"):
        return False
    text_a = str(a.get("text_norm", ""))
    text_b = str(b.get("text_norm", ""))
    if not text_a or not text_b:
        return False
    if text_a == text_b:
        return False
    shorter = min(len(text_a), len(text_b))
    longer = max(len(text_a), len(text_b))
    if shorter < min_len:
        return False
    if shorter / max(longer, 1) < 0.6:
        return False
    return text_a in text_b or text_b in text_a


def match_entities(
    baseline_entities: list[dict],
    gliner2_entities: list[dict],
    *,
    soft_match_min_chars: int = 4,
) -> dict:
    baseline_used: set[int] = set()
    gliner2_used: set[int] = set()
    agreed = []
    conflicts = []

    exact_index = {}
    for index, entity in enumerate(gliner2_entities):
        key = (entity.get("label"), entity.get("text_norm"))
        exact_index.setdefault(key, []).append(index)

    for baseline_index, baseline_entity in enumerate(baseline_entities):
        key = (baseline_entity.get("label"), baseline_entity.get("text_norm"))
        candidates = exact_index.get(key, [])
        chosen = next((idx for idx in candidates if idx not in gliner2_used), None)
        if chosen is None:
            continue
        baseline_used.add(baseline_index)
        gliner2_used.add(chosen)
        agreed.append(
            {
                "baseline_index": baseline_index,
                "gliner2_index": chosen,
                "text": baseline_entity.get("text", ""),
                "label": baseline_entity.get("label", ""),
                "match_type": "exact",
                "consensus_score": 1.0,
                "baseline_entity": baseline_entity,
                "gliner2_entity": gliner2_entities[chosen],
            }
        )

    for baseline_index, baseline_entity in enumerate(baseline_entities):
        if baseline_index in baseline_used:
            continue
        chosen = None
        for gliner2_index, gliner2_entity in enumerate(gliner2_entities):
            if gliner2_index in gliner2_used:
                continue
            if _is_soft_match(baseline_entity, gliner2_entity, min_len=soft_match_min_chars):
                chosen = gliner2_index
                break
        if chosen is None:
            continue
        baseline_used.add(baseline_index)
        gliner2_used.add(chosen)
        agreed.append(
            {
                "baseline_index": baseline_index,
                "gliner2_index": chosen,
                "text": baseline_entity.get("text", ""),
                "label": baseline_entity.get("label", ""),
                "match_type": "soft",
                "consensus_score": 0.5,
                "baseline_entity": baseline_entity,
                "gliner2_entity": gliner2_entities[chosen],
            }
        )

    for baseline_index, baseline_entity in enumerate(baseline_entities):
        if baseline_index in baseline_used:
            continue
        for gliner2_index, gliner2_entity in enumerate(gliner2_entities):
            if gliner2_index in gliner2_used:
                continue
            if baseline_entity.get("text_norm") and baseline_entity.get("text_norm") == gliner2_entity.get("text_norm"):
                if baseline_entity.get("label") != gliner2_entity.get("label"):
                    conflicts.append(
                        {
                            "baseline_index": baseline_index,
                            "gliner2_index": gliner2_index,
                            "text": baseline_entity.get("text", ""),
                            "baseline_label": baseline_entity.get("label", ""),
                            "gliner2_label": gliner2_entity.get("label", ""),
                            "conflict_type": "label_mismatch",
                            "baseline_entity": baseline_entity,
                            "gliner2_entity": gliner2_entity,
                        }
                    )
                break

    baseline_only = [entity for index, entity in enumerate(baseline_entities) if index not in baseline_used]
    gliner2_only = [entity for index, entity in enumerate(gliner2_entities) if index not in gliner2_used]

    return {
        "agreed_entities": agreed,
        "baseline_only_entities": baseline_only,
        "gliner2_only_entities": gliner2_only,
        "conflicts": conflicts,
    }


def build_review_seed_entities(
    *,
    agreed_entities: list[dict],
    baseline_only_entities: list[dict],
    gliner2_only_entities: list[dict],
    location_metadata_terms: set[str],
    baseline_seed_score_threshold: float,
    gliner2_location_min_chars: int,
) -> list[dict]:
    seeded = []
    seen = set()
    for item in agreed_entities:
        entity = dict(item.get("baseline_entity", {}))
        if entity.get("label") == "Location" and not _is_viable_location_seed(
            entity,
            location_metadata_terms=location_metadata_terms,
            require_exact_metadata_for_single_token=False,
        ):
            continue
        entity["seed_origin"] = f"agreed_{item.get('match_type', 'exact')}"
        key = (entity.get("label"), entity.get("text_norm"))
        if key in seen:
            continue
        seen.add(key)
        seeded.append(entity)
    for entity in baseline_only_entities:
        ner_score = _safe_float(entity.get("ner_score"))
        if ner_score is None or ner_score < baseline_seed_score_threshold:
            continue
        if entity.get("label") == "Location" and not _is_viable_location_seed(
            entity,
            location_metadata_terms=location_metadata_terms,
            require_exact_metadata_for_single_token=True,
        ):
            continue
        seeded_entity = dict(entity)
        seeded_entity["seed_origin"] = "baseline_high_score"
        key = (seeded_entity.get("label"), seeded_entity.get("text_norm"))
        if key in seen:
            continue
        seen.add(key)
        seeded.append(seeded_entity)
    for entity in gliner2_only_entities:
        if not _matches_location_metadata(entity, location_metadata_terms, min_chars=gliner2_location_min_chars):
            continue
        if not _is_viable_location_seed(
            entity,
            location_metadata_terms=location_metadata_terms,
            require_exact_metadata_for_single_token=False,
        ):
            continue
        seeded_entity = dict(entity)
        seeded_entity["seed_origin"] = "gliner2_location_metadata_match"
        key = (seeded_entity.get("label"), seeded_entity.get("text_norm"))
        if key in seen:
            continue
        seen.add(key)
        seeded.append(seeded_entity)
    seeded.sort(key=lambda entity: (str(entity.get("label", "")), str(entity.get("text_norm", ""))))
    return seeded


def _resolve_source_id(row: dict, *, row_index: int) -> str:
    for key in DEFAULT_SOURCE_ID_FIELDS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    candidate_rank = row.get("_candidate_rank")
    if isinstance(candidate_rank, dict):
        rank = candidate_rank.get("rank")
        if rank is not None:
            return f"candidate_rank_{rank}"
    return f"row_{row_index}"


def _candidate_quality_score(row: dict) -> float | None:
    meta = row.get("_candidate_rank")
    if isinstance(meta, dict):
        return _safe_float(meta.get("candidate_quality_score"))
    return None


def _record_score(row: dict) -> float | None:
    for key in ("record_score", "record_score_context_boosted", "score_relato_confianca", "score_relato"):
        score = _safe_float(row.get(key))
        if score is not None:
            return score
    meta = row.get("_candidate_rank")
    if isinstance(meta, dict):
        return _safe_float(meta.get("record_score"))
    return None


def build_adjudication_row(
    row: dict,
    *,
    row_index: int,
    gliner2_entities: list[dict],
    allowed_labels: set[str],
    soft_match_min_chars: int,
    baseline_seed_score_threshold: float,
    gliner2_location_min_chars: int,
) -> dict:
    text = get_text(row)
    location_metadata_terms = _extract_location_metadata_terms(row)
    baseline_entities = normalize_baseline_entities(get_spans(row), allowed_labels)
    gliner2_entities = normalize_gliner2_entities(gliner2_entities, allowed_labels)
    matched = match_entities(
        baseline_entities,
        gliner2_entities,
        soft_match_min_chars=soft_match_min_chars,
    )
    agreed_entities = matched["agreed_entities"]
    baseline_only_entities = matched["baseline_only_entities"]
    gliner2_only_entities = matched["gliner2_only_entities"]
    conflicts = matched["conflicts"]
    review_seed_entities = build_review_seed_entities(
        agreed_entities=agreed_entities,
        baseline_only_entities=baseline_only_entities,
        gliner2_only_entities=gliner2_only_entities,
        location_metadata_terms=location_metadata_terms,
        baseline_seed_score_threshold=baseline_seed_score_threshold,
        gliner2_location_min_chars=gliner2_location_min_chars,
    )

    union_count = len(agreed_entities) + len(baseline_only_entities) + len(gliner2_only_entities)
    agreement_ratio = (len(agreed_entities) / union_count) if union_count else 0.0
    baseline_coverage_proxy = (len(agreed_entities) / len(baseline_entities)) if baseline_entities else 0.0
    gliner2_noise_proxy = (len(gliner2_only_entities) / len(gliner2_entities)) if gliner2_entities else 0.0

    return {
        "source_id": _resolve_source_id(row, row_index=row_index),
        "text": text,
        "record_score": _record_score(row),
        "candidate_quality_score": _candidate_quality_score(row),
        "baseline_entities": baseline_entities,
        "gliner2_entities": gliner2_entities,
        "agreed_entities": agreed_entities,
        "baseline_only_entities": baseline_only_entities,
        "gliner2_only_entities": gliner2_only_entities,
        "conflicts": conflicts,
        "review_seed_entities": review_seed_entities,
        "metadata": {
            "has_location": any(entity.get("label") == "Location" for entity in review_seed_entities),
            "entity_count_baseline": len(baseline_entities),
            "entity_count_gliner2": len(gliner2_entities),
            "entity_count_agreed": len(agreed_entities),
            "entity_count_baseline_only": len(baseline_only_entities),
            "entity_count_gliner2_only": len(gliner2_only_entities),
            "entity_count_conflicts": len(conflicts),
            "agreement_ratio": agreement_ratio,
            "baseline_coverage_proxy": baseline_coverage_proxy,
            "gliner2_noise_proxy": gliner2_noise_proxy,
            "location_metadata_terms": sorted(location_metadata_terms),
        },
        "_source": row,
    }


def build_summary(rows: list[dict], counters: Counter, args: argparse.Namespace) -> dict:
    agreed = sum(row["metadata"]["entity_count_agreed"] for row in rows)
    baseline_only = sum(row["metadata"]["entity_count_baseline_only"] for row in rows)
    gliner2_only = sum(row["metadata"]["entity_count_gliner2_only"] for row in rows)
    conflicts = sum(row["metadata"]["entity_count_conflicts"] for row in rows)
    return {
        "input": str(resolve_repo_artifact_path(__file__, args.input)),
        "output_jsonl": str(resolve_repo_artifact_path(__file__, args.output_jsonl)),
        "records_total": counters["records_total"],
        "records_emitted": len(rows),
        "gliner2_model": args.gliner2_model,
        "labels": _parse_csv(args.labels) or list(DEFAULT_LABELS),
        "entity_types": _parse_csv(args.entity_types) or list(DEFAULT_ENTITY_TYPES),
        "summary": {
            "agreed_entities_total": agreed,
            "baseline_only_entities_total": baseline_only,
            "gliner2_only_entities_total": gliner2_only,
            "conflicts_total": conflicts,
            "avg_agreement_ratio": (sum(row["metadata"]["agreement_ratio"] for row in rows) / len(rows)) if rows else 0.0,
            "avg_seed_entities": (sum(len(row["review_seed_entities"]) for row in rows) / len(rows)) if rows else 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSONL inputs for LLM-assisted adjudication of NER pseudolabel candidates.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL with baseline candidate rows and entities.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL enriched with GLiNER2 predictions and matching metadata.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--gliner2-model", default="fastino/gliner2-base-v1", help="HF repo id or local path for GLiNER2 base.")
    parser.add_argument("--entity-types", default="person,location,organization", help="Comma-separated GLiNER2 entity types.")
    parser.add_argument("--labels", default="Person,Location,Organization", help="Comma-separated allowed output labels.")
    parser.add_argument("--adapter-dir", default="", help="Optional GLiNER2 adapter directory.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional maximum number of input records to process.")
    parser.add_argument("--soft-match-min-chars", type=int, default=4, help="Minimum normalized entity length for soft matches.")
    parser.add_argument("--baseline-seed-score-threshold", type=float, default=0.80, help="Minimum baseline score for baseline-only entities to seed review.")
    parser.add_argument("--gliner2-location-min-chars", type=int, default=5, help="Minimum normalized length for gliner2-only location entities to be promoted via metadata match.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    rows = read_json_or_jsonl(resolve_repo_artifact_path(__file__, args.input))
    if args.max_records > 0:
        rows = rows[: args.max_records]

    allowed_labels = set(_parse_csv(args.labels) or DEFAULT_LABELS)
    entity_types = _parse_csv(args.entity_types) or list(DEFAULT_ENTITY_TYPES)

    from gliner2_inference import predict_entities_for_text as predict_gliner2_entities
    from gliner2_loader import load_gliner2_model

    model = load_gliner2_model(
        args.gliner2_model,
        adapter_dir=args.adapter_dir,
        logger=LOGGER,
        context="llm adjudication input",
    )

    enriched_rows = []
    counters = Counter()
    for row_index, row in enumerate(rows, start=1):
        text = get_text(row)
        if not text:
            counters["skipped_no_text"] += 1
            continue
        gliner2_entities = predict_gliner2_entities(model, text, entity_types)
        enriched_rows.append(
            build_adjudication_row(
                row,
                row_index=row_index,
                gliner2_entities=gliner2_entities,
                allowed_labels=allowed_labels,
                soft_match_min_chars=args.soft_match_min_chars,
                baseline_seed_score_threshold=args.baseline_seed_score_threshold,
                gliner2_location_min_chars=args.gliner2_location_min_chars,
            )
        )
        counters["records_total"] += 1

    write_jsonl(resolve_repo_artifact_path(__file__, args.output_jsonl), enriched_rows)
    LOGGER.info("Saved LLM adjudication input JSONL: %s", args.output_jsonl)

    if args.summary_json:
        summary_path = resolve_repo_artifact_path(__file__, args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(build_summary(enriched_rows, counters, args), indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Saved LLM adjudication summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()
