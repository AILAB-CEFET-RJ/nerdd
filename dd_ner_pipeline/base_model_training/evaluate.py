import json
import logging
import re
from collections import Counter
from pathlib import Path

from gliner import GLiNER
from gliner_train.io_utils import load_jsonl, save_jsonl
from gliner_train.paths import resolve_path
from sklearn.metrics import accuracy_score, classification_report, f1_score

LOGGER = logging.getLogger(__name__)


def split_text(text, tokenizer, max_tokens):
    """Split text into tokenizer-sized chunks."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    parts = []
    for index in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[index : index + max_tokens]
        parts.append(tokenizer.decode(chunk_ids, skip_special_tokens=True).strip())
    return [part for part in parts if part]


def clean_entities(entities, original_text):
    """Filter malformed or invalid entities."""
    cleaned = []
    for entity in entities:
        text = entity.get("text", "")
        if text.strip() in {"[", "]", "CLS", "SEP"}:
            continue
        if entity.get("start", -1) < 0 or entity.get("end", 0) > len(original_text):
            continue
        if not re.match(r"^[\wÀ-ÿ\s\-\.']+$", text):
            continue
        cleaned.append(entity)
    return cleaned


def find_with_cursor(text, part, cursor):
    """Find chunk position using a forward cursor to avoid repeated-match drift."""
    position = text.find(part, cursor)
    if position == -1:
        position = text.find(part)
    return position


def predict_batch_entities(model, batch_texts, labels, threshold):
    """Predict entities for a batch, preferring the modern GLiNER inference API."""
    if hasattr(model, "inference"):
        try:
            predictions = model.inference(batch_texts, labels=labels, threshold=threshold)
            if isinstance(predictions, list):
                return predictions
        except TypeError:
            # Some GLiNER versions expose inference with slightly different kwargs.
            predictions = model.inference(batch_texts, labels, threshold)
            if isinstance(predictions, list):
                return predictions

    # Backward compatibility with older GLiNER versions.
    return model.batch_predict_entities(batch_texts, labels, threshold=threshold)


def predict_entities_jsonl(model, input_path, output_path, labels, batch_size, chunk_size, threshold):
    """Run prediction and persist entities in JSONL."""
    rows = load_jsonl(input_path)
    texts = [row["text"] for row in rows]
    tokenizer = model.data_processor.transformer_tokenizer

    predictions = []
    for text in texts:
        chunks = split_text(text, tokenizer=tokenizer, max_tokens=chunk_size)
        chunk_preds = []
        for index in range(0, len(chunks), batch_size):
            batch = chunks[index : index + batch_size]
            chunk_preds.extend(predict_batch_entities(model, batch, labels, threshold))

        merged = []
        cursor = 0
        for chunk_text, entities in zip(chunks, chunk_preds):
            cleaned = clean_entities(entities, chunk_text)
            offset = find_with_cursor(text, chunk_text, cursor)
            if offset == -1:
                continue
            cursor = offset + len(chunk_text)
            for entity in cleaned:
                fixed = dict(entity)
                fixed["start"] = fixed["start"] + offset
                fixed["end"] = fixed["end"] + offset
                merged.append(fixed)

        predictions.append({"text": text, "entities": merged})

    save_jsonl(output_path, predictions)
    LOGGER.info("Saved predictions to %s", output_path)


def extract_entities(entry, key, score_threshold=None):
    """Extract entity tuples as (start, end, label), optionally filtering by score."""
    values = entry.get(key, [])
    entities = set()

    for entity in values:
        if score_threshold is not None and "score" in entity:
            if isinstance(score_threshold, dict):
                min_score = score_threshold.get(entity["label"], 0.5)
                if entity["score"] < min_score:
                    continue
            else:
                if entity["score"] < score_threshold:
                    continue
        entities.add((entity["start"], entity["end"], entity["label"]))

    return entities


def evaluate(gt_data, pred_data, score_threshold=None):
    """Evaluate prediction entities against ground-truth spans."""
    if len(gt_data) != len(pred_data):
        raise ValueError(
            f"Mismatched dataset lengths: ground truth={len(gt_data)} prediction={len(pred_data)}"
        )

    y_true = []
    y_pred = []
    per_class_counts = Counter()

    for gt_entry, pred_entry in zip(gt_data, pred_data):
        gt_entities = extract_entities(gt_entry, "spans")
        pred_entities = extract_entities(pred_entry, "entities", score_threshold=score_threshold)

        for entity in pred_entities & gt_entities:
            y_true.append(entity[2])
            y_pred.append(entity[2])
            per_class_counts[(entity[2], "TP")] += 1

        for entity in pred_entities - gt_entities:
            y_true.append("None")
            y_pred.append(entity[2])
            per_class_counts[(entity[2], "FP")] += 1

        for entity in gt_entities - pred_entities:
            y_true.append(entity[2])
            y_pred.append("None")
            per_class_counts[(entity[2], "FN")] += 1

    return y_true, y_pred, per_class_counts


def calibrate_thresholds(gt_data, pred_data, labels, thresholds):
    """Find best threshold per label using F1."""
    results = {}

    for label in labels:
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in thresholds:
            _, _, per_class_counts = evaluate(gt_data, pred_data, score_threshold=threshold)
            tp = per_class_counts.get((label, "TP"), 0)
            fp = per_class_counts.get((label, "FP"), 0)
            fn = per_class_counts.get((label, "FN"), 0)

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            label_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

            if label_f1 > best_f1:
                best_f1 = label_f1
                best_threshold = threshold

        results[label] = {"best_threshold": best_threshold, "best_f1": best_f1}
        LOGGER.info("Best threshold for %s: %.2f (F1=%.4f)", label, best_threshold, best_f1)

    return results


def build_class_summary(per_class_counts, labels):
    """Build per-class precision/recall/F1 summary lines."""
    lines = ["Class summary:"]
    for label in labels:
        tp = per_class_counts.get((label, "TP"), 0)
        fp = per_class_counts.get((label, "FP"), 0)
        fn = per_class_counts.get((label, "FN"), 0)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        label_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        lines.append(
            f"{label:12} | TP: {tp:<4} FP: {fp:<4} FN: {fn:<4} | "
            f"Precision: {precision:.2f} Recall: {recall:.2f} F1: {label_f1:.2f}"
        )
    return lines


def save_evaluation_report(path, y_true, y_pred, labels, per_class_counts):
    """Save classification report + accuracy + macro F1 + class summary."""
    valid_labels = sorted(set(labels) & (set(y_true) | set(y_pred)))
    report_text = classification_report(y_true, y_pred, labels=valid_labels, digits=4, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=valid_labels, average="macro", zero_division=0)
    summary = build_class_summary(per_class_counts, valid_labels)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Per-class TP/FP/FN summary:\n")
        handle.write("\n".join(summary) + "\n\n")
        handle.write("Classification report:\n")
        handle.write(report_text + "\n")
        handle.write(f"\nOverall accuracy: {accuracy:.4f}\n")
        handle.write(f"F1-macro: {macro_f1:.4f}\n\n")

    LOGGER.info("Saved evaluation report to %s", path)


def run_evaluation(config, script_path):
    """End-to-end evaluation pipeline with prediction + threshold calibration."""
    script_dir = Path(script_path).resolve().parent
    gt_jsonl = resolve_path(script_dir, config.gt_jsonl)
    pred_jsonl = resolve_path(script_dir, config.pred_jsonl)
    calibrated_thresholds_json = resolve_path(script_dir, config.calibrated_thresholds_json)
    report_path = resolve_path(script_dir, config.report_path)

    model_path_candidate = resolve_path(script_dir, config.model_path)
    model_path = str(model_path_candidate) if model_path_candidate.exists() else config.model_path

    pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    calibrated_thresholds_json.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = GLiNER.from_pretrained(model_path, load_tokenizer=True)

    predict_entities_jsonl(
        model=model,
        input_path=str(gt_jsonl),
        output_path=str(pred_jsonl),
        labels=config.labels,
        batch_size=config.batch_size,
        chunk_size=config.chunk_size,
        threshold=config.prediction_threshold,
    )

    gt_data = load_jsonl(str(gt_jsonl))
    pred_data = load_jsonl(str(pred_jsonl))

    best_thresholds = calibrate_thresholds(gt_data, pred_data, config.labels, config.threshold_grid)
    with open(calibrated_thresholds_json, "w", encoding="utf-8") as handle:
        json.dump(best_thresholds, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Saved calibrated thresholds to %s", calibrated_thresholds_json)

    threshold_map = {label: values["best_threshold"] for label, values in best_thresholds.items()}
    y_true, y_pred, per_class_counts = evaluate(gt_data, pred_data, score_threshold=threshold_map)

    save_evaluation_report(
        path=str(report_path),
        y_true=y_true,
        y_pred=y_pred,
        labels=config.labels,
        per_class_counts=per_class_counts,
    )
