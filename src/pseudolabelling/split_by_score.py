import json
import logging
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from base_model_training.io_utils import load_jsonl, save_jsonl
from base_model_training.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_record_score(record, score_field, fallback_score_field, missing_policy):
    source = None
    score_value = None
    missing_reason = None

    if score_field in record:
        score_value = _safe_float(record.get(score_field))
        source = score_field
        if score_value is None:
            missing_reason = "invalid_primary_score"
    elif fallback_score_field and fallback_score_field in record:
        score_value = _safe_float(record.get(fallback_score_field))
        source = fallback_score_field
        if score_value is None:
            missing_reason = "invalid_fallback_score"
    else:
        missing_reason = "missing_score_fields"

    if score_value is not None:
        return score_value, source, missing_reason

    if missing_policy == "zero":
        return 0.0, "implicit_zero", missing_reason or "missing_score"
    if missing_policy == "discard":
        return None, source, missing_reason or "missing_score"
    raise ValueError(
        f"Score is missing/invalid for record and missing-policy=error "
        f"(score_field={score_field}, fallback={fallback_score_field}, reason={missing_reason})"
    )


def compare(value, threshold, operator):
    if operator == "ge":
        return value >= threshold
    if operator == "gt":
        return value > threshold
    if operator == "le":
        return value <= threshold
    if operator == "lt":
        return value < threshold
    raise ValueError(f"Unsupported operator: {operator}")


def _get_entities(record, entity_key):
    entities = record.get(entity_key)
    if isinstance(entities, list):
        return entities
    if entity_key != "ner" and isinstance(record.get("ner"), list):
        return record.get("ner")
    if entity_key != "entities" and isinstance(record.get("entities"), list):
        return record.get("entities")
    return []


def resolve_entity_gate_score(
    record,
    *,
    entity_key,
    score_field,
    label_field,
    labels,
    aggregation,
):
    entities = _get_entities(record, entity_key)
    label_set = set(labels)
    matched_scores = []
    matched_entities = []
    for entity in entities:
        if str(entity.get(label_field, "")) not in label_set:
            continue
        score = _safe_float(entity.get(score_field))
        if score is None:
            continue
        matched_scores.append(score)
        matched_entities.append(
            {
                "text": entity.get("text", ""),
                "label": entity.get(label_field, ""),
                "score": score,
            }
        )

    if not matched_scores:
        return None, matched_entities

    if aggregation == "mean":
        value = mean(matched_scores)
    elif aggregation == "max":
        value = max(matched_scores)
    elif aggregation == "min":
        value = min(matched_scores)
    else:
        raise ValueError(f"Unsupported entity gate aggregation: {aggregation}")
    return float(value), matched_entities


def split_records(
    rows,
    score_field,
    threshold,
    operator,
    fallback_score_field,
    missing_policy,
    trace_key,
    entity_gate=None,
):
    kept = []
    discarded = []
    counters = Counter()
    kept_scores = []
    discarded_scores = []

    for row in rows:
        gate_value = None
        gate_decision = None
        gate_entities = []
        gate_missing_reason = None
        score_value, score_source, missing_reason = resolve_record_score(
            record=row,
            score_field=score_field,
            fallback_score_field=fallback_score_field,
            missing_policy=missing_policy,
        )
        if score_value is None:
            decision = "discarded"
            target = discarded
            counters["missing_scores"] += 1
        else:
            is_kept = compare(score_value, threshold, operator)
            if entity_gate and is_kept:
                gate_value, gate_entities = resolve_entity_gate_score(
                    row,
                    entity_key=entity_gate["entity_key"],
                    score_field=entity_gate["score_field"],
                    label_field=entity_gate["label_field"],
                    labels=entity_gate["labels"],
                    aggregation=entity_gate["aggregation"],
                )
                if gate_value is None:
                    gate_decision = False
                    gate_missing_reason = "no_matching_entities"
                else:
                    gate_decision = compare(gate_value, entity_gate["threshold"], entity_gate["operator"])
                is_kept = is_kept and bool(gate_decision)
                if not is_kept and gate_decision is False:
                    counters["entity_gate_rejections"] += 1
            decision = "kept" if is_kept else "discarded"
            target = kept if is_kept else discarded
            if is_kept:
                kept_scores.append(score_value)
            else:
                discarded_scores.append(score_value)

        updated = deepcopy(row)
        updated[trace_key] = {
            "score_field": score_field,
            "fallback_score_field": fallback_score_field if fallback_score_field else None,
            "score_source": score_source,
            "score_used": score_value,
            "threshold": threshold,
            "operator": operator,
            "decision": decision,
            "missing_reason": missing_reason,
        }
        if entity_gate:
            updated[trace_key]["entity_gate"] = {
                "enabled": True,
                "entity_key": entity_gate["entity_key"],
                "score_field": entity_gate["score_field"],
                "label_field": entity_gate["label_field"],
                "labels": entity_gate["labels"],
                "aggregation": entity_gate["aggregation"],
                "threshold": entity_gate["threshold"],
                "operator": entity_gate["operator"],
                "gate_score_used": gate_value,
                "gate_decision": gate_decision,
                "gate_missing_reason": gate_missing_reason,
                "matched_entities": gate_entities,
            }
        target.append(updated)

    summary = {
        "records_total": len(rows),
        "kept_count": len(kept),
        "discarded_count": len(discarded),
        "missing_scores": int(counters["missing_scores"]),
        "entity_gate_rejections": int(counters["entity_gate_rejections"]),
        "kept_score_mean": mean(kept_scores) if kept_scores else None,
        "kept_score_min": min(kept_scores) if kept_scores else None,
        "kept_score_max": max(kept_scores) if kept_scores else None,
        "discarded_score_mean": mean(discarded_scores) if discarded_scores else None,
        "discarded_score_min": min(discarded_scores) if discarded_scores else None,
        "discarded_score_max": max(discarded_scores) if discarded_scores else None,
    }
    return kept, discarded, summary


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def run_split(
    *,
    input_jsonl,
    out_dir,
    score_field,
    threshold,
    operator="ge",
    fallback_score_field="",
    missing_policy="discard",
    trace_key="_split",
    entity_gate=None,
    legacy_filenames=False,
    script_path,
):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    input_path = resolve_path(script_dir, input_jsonl)
    output_dir = resolve_path(script_dir, out_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(str(input_path))
    kept, discarded, split_summary = split_records(
        rows=rows,
        score_field=score_field,
        threshold=threshold,
        operator=operator,
        fallback_score_field=fallback_score_field,
        missing_policy=missing_policy,
        trace_key=trace_key,
        entity_gate=entity_gate,
    )

    kept_path = output_dir / "kept.jsonl"
    discarded_path = output_dir / "discarded.jsonl"
    summary_path = output_dir / "summary.json"
    save_jsonl(str(kept_path), kept)
    save_jsonl(str(discarded_path), discarded)

    finished_at = datetime.now(timezone.utc).isoformat()
    runtime_seconds = perf_counter() - timer

    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": {
            "input_jsonl": input_jsonl,
            "score_field": score_field,
            "fallback_score_field": fallback_score_field or None,
            "threshold": threshold,
            "operator": operator,
            "missing_policy": missing_policy,
            "trace_key": trace_key,
            "entity_gate": entity_gate,
        },
        "summary": split_summary,
        "outputs": {
            "kept_jsonl": str(kept_path.resolve()),
            "discarded_jsonl": str(discarded_path.resolve()),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if legacy_filenames:
        save_jsonl(str(output_dir / "mantidos.jsonl"), kept)
        save_jsonl(str(output_dir / "descartados.jsonl"), discarded)
        with open(output_dir / "resumo.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Split completed | kept=%s discarded=%s", len(kept), len(discarded))
    LOGGER.info("kept file: %s", kept_path)
    LOGGER.info("discarded file: %s", discarded_path)
    LOGGER.info("summary file: %s", summary_path)
