import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from gliner import GLiNER
from tqdm import tqdm

from gliner_train.io_utils import load_jsonl, save_jsonl
from gliner_train.paths import resolve_path

LOGGER = logging.getLogger(__name__)
VALID_ENTITY_TEXT_PATTERN = re.compile(r"^[\wÀ-ÿ\s\-\.']+$")


def split_text(text, tokenizer, max_tokens):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    parts = []
    for index in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[index : index + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            parts.append(chunk_text)
    return parts


def find_with_cursor(text, part, cursor):
    position = text.find(part, cursor)
    if position == -1:
        position = text.find(part)
    return position


def clean_entities(entities, chunk_text):
    cleaned = []
    for entity in entities:
        entity_text = entity.get("text", "")
        if entity_text.strip() in {"[", "]", "CLS", "SEP"}:
            continue
        if entity.get("start", -1) < 0 or entity.get("end", 0) > len(chunk_text):
            continue
        if not VALID_ENTITY_TEXT_PATTERN.match(entity_text):
            continue
        cleaned.append(entity)
    return cleaned


def predict_batch_entities(model, batch_texts, labels, threshold):
    if hasattr(model, "inference"):
        try:
            predictions = model.inference(batch_texts, labels=labels, threshold=threshold)
            if isinstance(predictions, list):
                return predictions
        except TypeError:
            predictions = model.inference(batch_texts, labels, threshold)
            if isinstance(predictions, list):
                return predictions
    return model.batch_predict_entities(batch_texts, labels, threshold=threshold)


def build_inference_text(sample, text_fields, join_separator):
    values = [str(sample.get(field, "")).strip() for field in text_fields]
    values = [value for value in values if value]
    return join_separator.join(values).strip()


def deduplicate_entities(entities):
    seen = set()
    deduped = []
    for entity in entities:
        key = (
            entity.get("start"),
            entity.get("end"),
            entity.get("label"),
            entity.get("text", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def predict_entities_for_text(model, text, labels, batch_size, max_tokens, score_threshold):
    tokenizer = model.data_processor.transformer_tokenizer
    chunks = split_text(text, tokenizer=tokenizer, max_tokens=max_tokens)

    chunk_predictions = []
    for index in range(0, len(chunks), batch_size):
        batch = chunks[index : index + batch_size]
        chunk_predictions.extend(predict_batch_entities(model, batch, labels, score_threshold))

    merged = []
    cursor = 0
    for chunk_text, entities in zip(chunks, chunk_predictions):
        chunk_offset = find_with_cursor(text, chunk_text, cursor)
        if chunk_offset == -1:
            continue
        cursor = chunk_offset + len(chunk_text)
        for entity in clean_entities(entities, chunk_text):
            score = entity.get("score", 1.0)
            if score < score_threshold:
                continue
            fixed = dict(entity)
            fixed["start"] = fixed["start"] + chunk_offset
            fixed["end"] = fixed["end"] + chunk_offset
            merged.append(fixed)

    return deduplicate_entities(merged)


def _build_stats_payload(config, total_samples, failed_samples, total_entities, label_counts, started_at, finished_at, runtime_seconds):
    return {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": {
            "model_path": config.model_path,
            "input_jsonl": config.input_jsonl,
            "output_jsonl": config.output_jsonl,
            "labels": config.labels,
            "text_fields": config.text_fields,
            "batch_size": config.batch_size,
            "max_tokens": config.max_tokens,
            "score_threshold": config.score_threshold,
        },
        "summary": {
            "total_samples": total_samples,
            "failed_samples": failed_samples,
            "processed_samples": total_samples - failed_samples,
            "total_entities": total_entities,
            "entities_by_label": dict(label_counts),
        },
    }


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def run_corpus_prediction(config, script_path):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    input_jsonl = resolve_path(script_dir, config.input_jsonl)
    output_jsonl = resolve_path(script_dir, config.output_jsonl)
    stats_json = resolve_path(script_dir, config.stats_json)

    model_path_candidate = resolve_path(script_dir, config.model_path)
    model_path = str(model_path_candidate) if model_path_candidate.exists() else config.model_path

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    if config.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if config.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from: %s", model_path)
    model = GLiNER.from_pretrained(model_path, load_tokenizer=True)

    rows = load_jsonl(str(input_jsonl))
    LOGGER.info("Loaded %s input samples from %s", len(rows), input_jsonl)

    output_rows = []
    label_counts = Counter()
    total_entities = 0
    failed_samples = 0

    for index, row in enumerate(tqdm(rows, desc="Predicting", unit="sample"), start=1):
        inference_text = build_inference_text(
            sample=row,
            text_fields=config.text_fields,
            join_separator=config.join_separator,
        )

        if not inference_text:
            output_entry = dict(row)
            output_entry["entities"] = []
            if config.keep_inference_text:
                output_entry["inference_text"] = inference_text
            output_rows.append(output_entry)
            continue

        try:
            entities = predict_entities_for_text(
                model=model,
                text=inference_text,
                labels=config.labels,
                batch_size=config.batch_size,
                max_tokens=config.max_tokens,
                score_threshold=config.score_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            failed_samples += 1
            LOGGER.exception("Failed to process sample %s: %s", index, exc)
            entities = []

        for entity in entities:
            label_counts[entity.get("label", "UNKNOWN")] += 1
        total_entities += len(entities)

        output_entry = dict(row)
        output_entry["entities"] = entities
        if config.keep_inference_text:
            output_entry["inference_text"] = inference_text
        output_rows.append(output_entry)

    save_jsonl(str(output_jsonl), output_rows)

    finished_at = datetime.now(timezone.utc).isoformat()
    runtime_seconds = perf_counter() - timer
    stats_payload = _build_stats_payload(
        config=config,
        total_samples=len(rows),
        failed_samples=failed_samples,
        total_entities=total_entities,
        label_counts=label_counts,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=runtime_seconds,
    )

    with open(stats_json, "w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Saved predicted entities JSONL to: %s", output_jsonl)
    LOGGER.info("Saved run statistics JSON to: %s", stats_json)
    LOGGER.info(
        "Prediction finished | samples=%s processed=%s failed=%s entities=%s runtime=%s",
        len(rows),
        len(rows) - failed_samples,
        failed_samples,
        total_entities,
        _format_duration(runtime_seconds),
    )
