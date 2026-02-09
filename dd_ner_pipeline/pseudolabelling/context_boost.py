import json
import logging
import unicodedata
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from gliner_train.io_utils import load_jsonl, save_jsonl
from gliner_train.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def normalize_text(value):
    text = str(value or "").casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.split())


def get_primary_text(record, text_field_priority):
    for field in text_field_priority:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return "", ""


def extract_metadata_values(record, metadata_fields):
    values = []
    for field in metadata_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            values.append((field, value.strip()))
    return values


def _iter_entities(record):
    entities = record.get("entities")
    if isinstance(entities, list):
        return entities, "entities"
    ner = record.get("ner")
    if isinstance(ner, list):
        return ner, "ner"
    return [], "entities"


def _pick_entity_score(entity, base_score_field, fallback_score_fields):
    keys_to_try = [base_score_field] + [field for field in fallback_score_fields if field != base_score_field]
    for key in keys_to_try:
        if key not in entity:
            continue
        try:
            return float(entity[key]), key
        except (TypeError, ValueError):
            continue
    return None, ""


def _record_level_context_match(record, text_field_priority, metadata_fields):
    raw_text, text_field = get_primary_text(record, text_field_priority)
    normalized_text = normalize_text(raw_text)
    metadata_values = extract_metadata_values(record, metadata_fields)

    matched_metadata_fields = []
    match_count = 0
    for field, value in metadata_values:
        normalized_value = normalize_text(value)
        if normalized_value and normalized_value in normalized_text:
            match_count += 1
            matched_metadata_fields.append(field)
    return {
        "raw_text": raw_text,
        "text_field_used": text_field,
        "metadata_values": metadata_values,
        "matched_metadata_fields": matched_metadata_fields,
        "match_count": match_count,
        "has_match": match_count > 0,
    }


def _entity_matches_metadata(entity, metadata_values):
    entity_text = normalize_text(entity.get("text", ""))
    if not entity_text:
        return False
    for _field, value in metadata_values:
        normalized_value = normalize_text(value)
        if not normalized_value:
            continue
        if entity_text in normalized_value or normalized_value in entity_text:
            return True
    return False


def _should_boost_entity(entity, policy):
    if policy["boost_scope"] == "all-entities":
        return True
    if policy["boost_scope"] == "location-only":
        return str(entity.get(policy["label_field"], "")) in policy["location_labels"]
    if policy["boost_scope"] == "matched-only":
        return policy["entity_matches_metadata"]
    return True


def apply_context_boost_to_record(record, config):
    enriched = deepcopy(record)
    entities, entity_key = _iter_entities(enriched)
    match_data = _record_level_context_match(
        record=enriched,
        text_field_priority=config.text_field_priority,
        metadata_fields=config.metadata_fields,
    )

    if config.match_policy == "any-metadata-in-text":
        record_context_match = match_data["has_match"]
    elif config.match_policy == "entity-metadata-overlap":
        record_context_match = match_data["has_match"]
    else:
        raise ValueError(f"Unsupported match policy: {config.match_policy}")

    boost_multiplier = 1.0
    if record_context_match:
        if config.per_match:
            boost_multiplier = config.boost_factor ** max(1, match_data["match_count"])
        else:
            boost_multiplier = config.boost_factor

    boosted_scores = []
    boosted_entities = 0
    for entity in entities:
        score_value, score_source = _pick_entity_score(
            entity=entity,
            base_score_field=config.base_score_field,
            fallback_score_fields=config.fallback_score_fields,
        )
        if score_value is None:
            continue

        entity_matches_metadata = _entity_matches_metadata(entity, match_data["metadata_values"])
        policy = {
            "boost_scope": config.boost_scope,
            "label_field": config.label_field,
            "location_labels": set(config.location_labels),
            "entity_matches_metadata": entity_matches_metadata,
        }

        should_boost = record_context_match and _should_boost_entity(entity, policy)
        new_score = score_value * (boost_multiplier if should_boost else 1.0)

        if config.clamp_scores:
            new_score = min(max(new_score, 0.0), 1.0)

        entity[config.output_score_field] = float(new_score)
        entity["_context_boost_score_source"] = score_source
        if config.write_legacy_fields:
            entity["score_confianca"] = float(new_score)
            entity["score_confianca_from"] = score_source
            entity["score_confianca_factor"] = boost_multiplier if should_boost else 1.0
        boosted_scores.append(float(new_score))
        if should_boost:
            boosted_entities += 1
        if config.write_trace_fields:
            entity["_context_boost_applied"] = bool(should_boost)
            entity["_context_boost_multiplier"] = float(boost_multiplier if should_boost else 1.0)
            entity["_context_entity_metadata_match"] = bool(entity_matches_metadata)

    record_score = float(sum(boosted_scores) / len(boosted_scores)) if boosted_scores else 0.0
    enriched[config.output_record_score_field] = record_score
    if config.write_legacy_fields:
        enriched["score_relato_confianca"] = record_score

    if config.write_trace_fields:
        enriched["_context_boost_trace"] = {
            "text_field_used": match_data["text_field_used"],
            "matched_metadata_fields": match_data["matched_metadata_fields"],
            "match_count": match_data["match_count"],
            "record_context_match": bool(record_context_match),
            "boost_multiplier": float(boost_multiplier),
            "boost_scope": config.boost_scope,
            "match_policy": config.match_policy,
            "boosted_entities": boosted_entities,
            "total_entities": len(entities),
        }

    enriched[entity_key] = entities
    record_stats = {
        "record_context_match": bool(record_context_match),
        "boosted_entities": boosted_entities,
        "total_entities": len(entities),
        "match_count": int(match_data["match_count"]),
        "output_record_score": record_score,
    }
    return enriched, record_stats


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _build_stats_payload(config, summary, started_at, finished_at, runtime_seconds):
    return {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": {
            "input_jsonl": config.input_jsonl,
            "output_jsonl": config.output_jsonl,
            "text_field_priority": config.text_field_priority,
            "metadata_fields": config.metadata_fields,
            "base_score_field": config.base_score_field,
            "fallback_score_fields": config.fallback_score_fields,
            "output_score_field": config.output_score_field,
            "output_record_score_field": config.output_record_score_field,
            "boost_factor": config.boost_factor,
            "per_match": config.per_match,
            "clamp_scores": config.clamp_scores,
            "boost_scope": config.boost_scope,
            "match_policy": config.match_policy,
            "location_labels": config.location_labels,
            "write_trace_fields": config.write_trace_fields,
            "write_legacy_fields": config.write_legacy_fields,
        },
        "summary": summary,
    }


def run_context_boost(config, script_path):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    input_jsonl = resolve_path(script_dir, config.input_jsonl)
    output_jsonl = resolve_path(script_dir, config.output_jsonl)
    stats_json = resolve_path(script_dir, config.stats_json)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if config.boost_factor <= 0.0:
        raise ValueError("--boost-factor must be > 0")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(str(input_jsonl))
    LOGGER.info("Loaded %s rows from %s", len(rows), input_jsonl)

    output_rows = []
    counters = Counter()
    score_sums = Counter()

    for row in rows:
        boosted, row_stats = apply_context_boost_to_record(row, config)
        output_rows.append(boosted)
        counters["records_total"] += 1
        counters["entities_total"] += row_stats["total_entities"]
        counters["entities_boosted"] += row_stats["boosted_entities"]
        counters["records_with_context_match"] += int(row_stats["record_context_match"])
        score_sums["record_score_sum"] += row_stats["output_record_score"]
        score_sums["match_count_sum"] += row_stats["match_count"]

    save_jsonl(str(output_jsonl), output_rows)

    records_total = max(counters["records_total"], 1)
    entities_total = max(counters["entities_total"], 1)
    summary = {
        "records_total": int(counters["records_total"]),
        "entities_total": int(counters["entities_total"]),
        "entities_boosted": int(counters["entities_boosted"]),
        "records_with_context_match": int(counters["records_with_context_match"]),
        "pct_records_with_context_match": counters["records_with_context_match"] / records_total,
        "pct_entities_boosted": counters["entities_boosted"] / entities_total,
        "avg_record_score": score_sums["record_score_sum"] / records_total,
        "avg_metadata_matches_per_record": score_sums["match_count_sum"] / records_total,
    }

    finished_at = datetime.now(timezone.utc).isoformat()
    runtime_seconds = perf_counter() - timer
    stats_payload = _build_stats_payload(
        config=config,
        summary=summary,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=runtime_seconds,
    )
    with open(stats_json, "w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Saved context-boosted JSONL to: %s", output_jsonl)
    LOGGER.info("Saved context-boost run stats to: %s", stats_json)
    LOGGER.info(
        "Context boost finished | rows=%s matches=%s entities_boosted=%s runtime=%s",
        counters["records_total"],
        counters["records_with_context_match"],
        counters["entities_boosted"],
        _format_duration(runtime_seconds),
    )
