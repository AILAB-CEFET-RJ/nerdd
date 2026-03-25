import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from base_model_training.io_utils import save_jsonl
from base_model_training.paths import resolve_path
from gliner_loader import load_gliner_model
from text_chunking import split_text_fast
LOGGER = logging.getLogger(__name__)


def _load_gliner_model(model_path, model_max_length, map_location):
    return load_gliner_model(
        model_path,
        model_max_length=model_max_length,
        map_location=map_location,
        logger=LOGGER,
        context="evaluation",
    )


def clean_entities(entities, chunk_text):
    cleaned = []
    for entity in entities:
        entity_text = entity.get("text", "")
        if entity_text.strip() in {"[", "]", "CLS", "SEP"}:
            continue
        if entity.get("start", -1) < 0 or entity.get("end", 0) > len(chunk_text):
            continue
        cleaned.append(entity)
    return cleaned


def find_with_cursor(text, part, cursor):
    position = text.find(part, cursor)
    if position == -1:
        position = text.find(part)
    return position


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


def predict_entities_for_texts(model, texts, labels, batch_size, max_tokens, threshold):
    tokenizer = model.data_processor.transformer_tokenizer
    chunk_records = []
    merged_by_row = [[] for _ in texts]

    for row_index, text in enumerate(texts):
        if not text:
            continue
        chunks = split_text_fast(text, model=model, tokenizer=tokenizer, max_tokens=max_tokens)
        cursor = 0
        for chunk_text in chunks:
            chunk_offset = find_with_cursor(text, chunk_text, cursor)
            if chunk_offset == -1:
                continue
            cursor = chunk_offset + len(chunk_text)
            chunk_records.append(
                {
                    "row_index": row_index,
                    "chunk_text": chunk_text,
                    "chunk_offset": chunk_offset,
                }
            )

    for idx in range(0, len(chunk_records), batch_size):
        batch_records = chunk_records[idx : idx + batch_size]
        batch_texts = [record["chunk_text"] for record in batch_records]
        batch_predictions = predict_batch_entities(model, batch_texts, labels, threshold)
        for record, entities in zip(batch_records, batch_predictions):
            row_entities = merged_by_row[record["row_index"]]
            for entity in clean_entities(entities, record["chunk_text"]):
                fixed = dict(entity)
                fixed["start"] = fixed["start"] + record["chunk_offset"]
                fixed["end"] = fixed["end"] + record["chunk_offset"]
                row_entities.append(fixed)

    return [deduplicate_entities(entities) for entities in merged_by_row]


def load_gt_jsonl_strict(path):
    source = Path(path)
    text = source.read_text(encoding="utf-8")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                rows.append(_validate_gt_row(row, line_no))
        return rows

    if isinstance(payload, list):
        return [_validate_gt_row(row, idx) for idx, row in enumerate(payload, start=1)]

    if isinstance(payload, dict):
        return [_validate_gt_row(payload, 1)]

    raise ValueError("Ground-truth file must be JSONL, a JSON object, or a JSON array of rows")


def _validate_gt_row(row, line_no):
    if not isinstance(row, dict):
        raise ValueError(f"Line {line_no}: row must be a JSON object")
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Line {line_no}: missing non-empty 'text'")
    spans = row.get("spans")
    if not isinstance(spans, list):
        raise ValueError(f"Line {line_no}: missing list field 'spans'")
    normalized_spans = []
    for idx, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError(f"Line {line_no}: span[{idx}] must be object")
        try:
            start = int(span["start"])
            end = int(span["end"])
            label = str(span["label"])
        except Exception as exc:
            raise ValueError(f"Line {line_no}: span[{idx}] missing start/end/label") from exc
        if end <= start:
            raise ValueError(f"Line {line_no}: span[{idx}] has invalid range start={start} end={end}")
        normalized_spans.append({"start": start, "end": end, "label": label})
    normalized = dict(row)
    normalized["text"] = text
    normalized["spans"] = normalized_spans
    return normalized


def _to_span_set(spans):
    span_set = set()
    for span in spans:
        span_set.add((int(span["start"]), int(span["end"]), str(span["label"])))
    return span_set


def _prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_span_metrics(gold_spans_by_row, pred_spans_by_row, labels):
    gold_sets = [_to_span_set(spans) for spans in gold_spans_by_row]
    pred_sets = [_to_span_set(spans) for spans in pred_spans_by_row]

    per_label = {}
    micro_tp = micro_fp = micro_fn = 0
    per_label_errors = {}

    for label in labels:
        tp = fp = fn = support = 0
        for gold_set, pred_set in zip(gold_sets, pred_sets):
            gold_label = {(s, e, l) for (s, e, l) in gold_set if l == label}
            pred_label = {(s, e, l) for (s, e, l) in pred_set if l == label}
            tp += len(gold_label & pred_label)
            fp += len(pred_label - gold_label)
            fn += len(gold_label - pred_label)
            support += len(gold_label)
        metrics = _prf1(tp, fp, fn)
        per_label[label] = {**metrics, "support": support}
        per_label_errors[label] = {"fp": fp, "fn": fn}
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro = _prf1(micro_tp, micro_fp, micro_fn)
    macro_f1 = sum(per_label[label]["f1"] for label in per_label) / len(per_label) if per_label else 0.0
    return {
        "per_label": per_label,
        "per_label_errors": per_label_errors,
        "micro": micro,
        "macro_f1": macro_f1,
        "labels": labels,
        "overall_support": sum(per_label[label]["support"] for label in per_label),
    }


def format_classification_report(metrics):
    lines = ["label                precision   recall   f1-score   support"]
    for label in metrics["labels"]:
        row = metrics["per_label"][label]
        lines.append(
            f"{label:<20}  {row['precision']:>9.3f}  {row['recall']:>7.3f}  "
            f"{row['f1']:>9.3f}  {row['support']:>8d}"
        )
    lines.append("")
    micro = metrics["micro"]
    lines.append(
        f"micro avg           {micro['precision']:>9.3f}  {micro['recall']:>7.3f}  "
        f"{micro['f1']:>9.3f}  {metrics['overall_support']:>8d}"
    )
    lines.append(f"macro f1: {metrics['macro_f1']:.3f}")
    return "\n".join(lines)


def predict_entities_for_text(model, text, labels, batch_size, max_tokens, threshold):
    return predict_entities_for_texts(
        model=model,
        texts=[text],
        labels=labels,
        batch_size=batch_size,
        max_tokens=max_tokens,
        threshold=threshold,
    )[0]


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def run_evaluate_refit(config, script_path):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    gt_jsonl = resolve_path(script_dir, config["gt_jsonl"])
    out_dir = resolve_path(script_dir, config["out_dir"])
    model_path_candidate = resolve_path(script_dir, config["model_path"])
    model_path = str(model_path_candidate) if model_path_candidate.exists() else config["model_path"]

    if not gt_jsonl.exists():
        raise FileNotFoundError(f"Ground-truth JSONL not found: {gt_jsonl}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_gt_jsonl_strict(str(gt_jsonl))
    LOGGER.info("Loaded %s validated GT samples from %s", len(rows), gt_jsonl)

    model_max_length = int(config.get("model_max_length", 0) or 0)
    map_location = str(config.get("map_location", "") or "")
    model = _load_gliner_model(model_path, model_max_length, map_location)

    prediction_rows = []
    pred_spans = []
    gold_spans = []
    failed_predictions = 0
    label_counts = Counter()
    total_rows = len(rows)
    progress_every_batches = max(1, 10)

    for batch_index, start in enumerate(range(0, len(rows), config["batch_size"]), start=1):
        batch_rows = rows[start : start + config["batch_size"]]
        batch_texts = [row["text"] for row in batch_rows]
        if batch_index == 1 or batch_index % progress_every_batches == 0:
            LOGGER.info(
                "Starting evaluation batch %s | rows %s-%s of %s",
                batch_index,
                start + 1,
                start + len(batch_rows),
                total_rows,
            )
        try:
            batch_entities = predict_entities_for_texts(
                model=model,
                texts=batch_texts,
                labels=config["labels"],
                batch_size=config["batch_size"],
                max_tokens=config["max_tokens"],
                threshold=config["prediction_threshold"],
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Batch prediction failed for rows %s-%s: %s", start + 1, start + len(batch_rows), exc)
            batch_entities = []
            for row_offset, text in enumerate(batch_texts, start=1):
                try:
                    batch_entities.append(
                        predict_entities_for_text(
                            model=model,
                            text=text,
                            labels=config["labels"],
                            batch_size=config["batch_size"],
                            max_tokens=config["max_tokens"],
                            threshold=config["prediction_threshold"],
                        )
                    )
                except Exception as row_exc:  # noqa: BLE001
                    failed_predictions += 1
                    LOGGER.exception("Prediction failed for row %s: %s", start + row_offset, row_exc)
                    batch_entities.append([])

        for row, entities in zip(batch_rows, batch_entities):
            gold = row["spans"]
            for entity in entities:
                label_counts[str(entity.get("label", "UNKNOWN"))] += 1
            prediction_rows.append({"text": row["text"], "spans": gold, "entities": entities})
            pred_spans.append(
                [{"start": int(e["start"]), "end": int(e["end"]), "label": str(e["label"])} for e in entities]
            )
            gold_spans.append(gold)

        if batch_index % progress_every_batches == 0 or (start + len(batch_rows)) >= total_rows:
            processed = start + len(batch_rows)
            elapsed = perf_counter() - timer
            rows_per_second = processed / elapsed if elapsed > 0 else 0.0
            LOGGER.info(
                "Evaluation progress: %s/%s rows (%.1f%%) | elapsed=%s | rows/s=%.3f",
                processed,
                total_rows,
                (processed / total_rows * 100.0) if total_rows else 100.0,
                _format_duration(elapsed),
                rows_per_second,
            )

    metrics = compute_span_metrics(gold_spans, pred_spans, config["labels"])
    report_text = format_classification_report(metrics)

    predictions_path = out_dir / "predictions.jsonl"
    report_path = out_dir / "classification_report.txt"
    metrics_path = out_dir / "metrics.json"
    run_stats_path = out_dir / "run_stats.json"

    save_jsonl(str(predictions_path), prediction_rows)
    report_path.write_text(report_text, encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    runtime_seconds = perf_counter() - timer
    finished_at = datetime.now(timezone.utc).isoformat()
    run_stats = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": config,
        "summary": {
            "rows_total": len(rows),
            "failed_predictions": failed_predictions,
            "predicted_entities_total": int(sum(label_counts.values())),
            "predicted_entities_by_label": dict(label_counts),
            "micro_f1": metrics["micro"]["f1"],
            "macro_f1": metrics["macro_f1"],
        },
    }
    run_stats_path.write_text(json.dumps(run_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("Saved predictions: %s", predictions_path)
    LOGGER.info("Saved report: %s", report_path)
    LOGGER.info("Saved metrics: %s", metrics_path)
    LOGGER.info("Saved run stats: %s", run_stats_path)
