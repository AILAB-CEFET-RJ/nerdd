import json
import logging
import unicodedata
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from base_model_training.io_utils import load_jsonl, save_jsonl
from base_model_training.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_entities(record, entity_key):
    entities = record.get(entity_key)
    if isinstance(entities, list):
        return entities
    if entity_key != "ner" and isinstance(record.get("ner"), list):
        return record.get("ner")
    if entity_key != "entities" and isinstance(record.get("entities"), list):
        return record.get("entities")
    return []


def _strip_accents(text):
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_entity_text(text):
    text = " ".join(str(text).strip().lower().split())
    text = _strip_accents(text)
    return text.strip(" \t\r\n.,;:!?()[]{}\"'")


def _dedupe_entities(entities, score_field, dedupe_mode):
    if dedupe_mode == "off":
        return entities, 0

    if dedupe_mode != "label_text":
        raise ValueError(f"Unsupported dedupe_mode: {dedupe_mode}")

    deduped = []
    seen = {}
    removed = 0
    for entity in entities:
        label = str(entity.get("label", "")).strip()
        normalized_text = _normalize_entity_text(entity.get("text", ""))
        if not label or not normalized_text:
            deduped.append(entity)
            continue

        key = (label, normalized_text)
        current_score = _safe_float(entity.get(score_field))
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(entity)
            continue

        prev_idx = seen[key]
        prev_score = _safe_float(deduped[prev_idx].get(score_field))
        if current_score is not None and (prev_score is None or current_score > prev_score):
            deduped[prev_idx] = entity
        removed += 1

    return deduped, removed


def _aggregate(values, aggregation):
    if aggregation == "mean":
        return float(mean(values))
    if aggregation == "max":
        return float(max(values))
    if aggregation == "median":
        return float(median(values))
    if aggregation == "p75":
        ordered = sorted(values)
        idx = int(round((len(ordered) - 1) * 0.75))
        return float(ordered[idx])
    if aggregation == "mean_times_min":
        return float(mean(values) * min(values))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def compute_record_score(record, *, score_field, entity_key, aggregation, empty_entities_policy, dedupe_mode="off"):
    entities = _get_entities(record, entity_key)
    entities, deduped_entities = _dedupe_entities(entities, score_field=score_field, dedupe_mode=dedupe_mode)
    valid_scores = []
    invalid_scores = 0
    for entity in entities:
        raw = entity.get(score_field)
        parsed = _safe_float(raw)
        if parsed is None:
            invalid_scores += 1
            continue
        valid_scores.append(parsed)

    if valid_scores:
        return _aggregate(valid_scores, aggregation), len(valid_scores), invalid_scores, False, deduped_entities

    if empty_entities_policy == "zero":
        return 0.0, 0, invalid_scores, True, deduped_entities
    if empty_entities_policy == "null":
        return None, 0, invalid_scores, True, deduped_entities
    raise ValueError(
        f"No valid entity scores found for record and empty-entities-policy=error "
        f"(score_field={score_field}, entity_key={entity_key})"
    )


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def run_compute_record_score(
    *,
    input_jsonl,
    output_jsonl,
    stats_json,
    score_field="score",
    output_field="record_score",
    legacy_field_alias="score_relato",
    entity_key="entities",
    aggregation="median",
    empty_entities_policy="zero",
    dedupe_mode="off",
    trace_key="_record_score_meta",
    write_trace=True,
    script_path,
):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    input_path = resolve_path(script_dir, input_jsonl)
    output_path = resolve_path(script_dir, output_jsonl)
    stats_path = resolve_path(script_dir, stats_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(str(input_path))
    output_rows = []
    counters = Counter()
    record_scores = []

    for row in rows:
        updated = deepcopy(row)
        score, n_valid, n_invalid, was_empty, deduped_entities = compute_record_score(
            updated,
            score_field=score_field,
            entity_key=entity_key,
            aggregation=aggregation,
            empty_entities_policy=empty_entities_policy,
            dedupe_mode=dedupe_mode,
        )

        updated[output_field] = score
        if legacy_field_alias:
            updated[legacy_field_alias] = score
        if write_trace:
            updated[trace_key] = {
                "score_field": score_field,
                "entity_key": entity_key,
                "aggregation": aggregation,
                "dedupe_mode": dedupe_mode,
                "valid_entity_scores": n_valid,
                "invalid_entity_scores": n_invalid,
                "deduped_entities": deduped_entities,
                "empty_entities_fallback": bool(was_empty),
            }

        counters["rows_total"] += 1
        counters["rows_empty_entity_scores"] += int(was_empty)
        counters["valid_entity_scores_total"] += n_valid
        counters["invalid_entity_scores_total"] += n_invalid
        counters["deduped_entities_total"] += deduped_entities
        if score is not None:
            record_scores.append(float(score))
        output_rows.append(updated)

    save_jsonl(str(output_path), output_rows)

    finished_at = datetime.now(timezone.utc).isoformat()
    runtime_seconds = perf_counter() - timer
    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": {
            "input_jsonl": input_jsonl,
            "output_jsonl": output_jsonl,
            "score_field": score_field,
            "output_field": output_field,
            "legacy_field_alias": legacy_field_alias or None,
            "entity_key": entity_key,
            "aggregation": aggregation,
            "dedupe_mode": dedupe_mode,
            "empty_entities_policy": empty_entities_policy,
            "trace_key": trace_key if write_trace else None,
        },
        "summary": {
            "rows_total": int(counters["rows_total"]),
            "rows_empty_entity_scores": int(counters["rows_empty_entity_scores"]),
            "valid_entity_scores_total": int(counters["valid_entity_scores_total"]),
            "invalid_entity_scores_total": int(counters["invalid_entity_scores_total"]),
            "deduped_entities_total": int(counters["deduped_entities_total"]),
            "record_score_mean": (sum(record_scores) / len(record_scores)) if record_scores else None,
            "record_score_min": min(record_scores) if record_scores else None,
            "record_score_max": max(record_scores) if record_scores else None,
        },
    }
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Saved record-score JSONL to: %s", output_path)
    LOGGER.info("Saved record-score stats to: %s", stats_path)
