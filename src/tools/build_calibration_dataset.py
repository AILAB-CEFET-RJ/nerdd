#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from gliner import GLiNER


VALID_ENTITY_TEXT_PATTERN = re.compile(r"^[\wÀ-ÿ\s\-\.']+$")


def read_json_or_jsonl(path):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    text = source.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        return rows

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Unsupported JSON format.")


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


def predict_entities_for_text(model, text, labels, batch_size, max_tokens, threshold):
    tokenizer = model.data_processor.transformer_tokenizer
    chunks = split_text(text, tokenizer=tokenizer, max_tokens=max_tokens)

    chunk_predictions = []
    for index in range(0, len(chunks), batch_size):
        batch = chunks[index : index + batch_size]
        chunk_predictions.extend(predict_batch_entities(model, batch, labels, threshold))

    merged = []
    cursor = 0
    for chunk_text, entities in zip(chunks, chunk_predictions):
        chunk_offset = find_with_cursor(text, chunk_text, cursor)
        if chunk_offset == -1:
            continue
        cursor = chunk_offset + len(chunk_text)
        for entity in clean_entities(entities, chunk_text):
            score = float(entity.get("score", 1.0))
            if score < threshold:
                continue
            fixed = dict(entity)
            fixed["start"] = fixed["start"] + chunk_offset
            fixed["end"] = fixed["end"] + chunk_offset
            merged.append(fixed)

    return deduplicate_entities(merged)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a calibration CSV from a labeled dataset and a GLiNER model."
    )
    parser.add_argument("--model-path", required=True, help="Path or HF id for the trained model.")
    parser.add_argument("--input", required=True, help="Labeled JSON/JSONL dataset with text+spans.")
    parser.add_argument("--output-csv", required=True, help="Output calibration CSV path.")
    parser.add_argument("--output-predictions-jsonl", default="", help="Optional predictions dump for auditing.")
    parser.add_argument("--labels", default="Person,Location,Organization", help="Comma-separated labels to predict.")
    parser.add_argument("--batch-size", type=int, default=4, help="Prediction batch size.")
    parser.add_argument("--max-tokens", type=int, default=384, help="Tokenizer chunk size.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Prediction threshold.")
    return parser.parse_args()


def main():
    args = parse_args()
    labels = [piece.strip() for piece in args.labels.split(",") if piece.strip()]
    if not labels:
        raise ValueError("At least one label must be provided.")

    rows = read_json_or_jsonl(args.input)
    model = GLiNER.from_pretrained(args.model_path, load_tokenizer=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    prediction_dump = []
    label_counts = Counter()
    validation_counts = Counter()

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "RowId",
                "Label",
                "Entidade_Referencia",
                "Entidade_Predita",
                "Start",
                "End",
                "Score",
                "Validacao",
            ],
        )
        writer.writeheader()

        for row_id, row in enumerate(rows):
            text = str(row.get("text", "")).strip()
            if not text:
                continue

            gold_spans = row.get("spans", []) or []
            gold_map = {
                (int(span["start"]), int(span["end"]), str(span["label"])): text[int(span["start"]) : int(span["end"])]
                for span in gold_spans
            }
            gold_set = set(gold_map.keys())

            predictions = predict_entities_for_text(
                model=model,
                text=text,
                labels=labels,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                threshold=args.threshold,
            )

            prediction_dump.append({"text": text, "spans": gold_spans, "entities": predictions})

            for entity in predictions:
                key = (int(entity["start"]), int(entity["end"]), str(entity["label"]))
                is_valid = int(key in gold_set)
                label = str(entity["label"])
                writer.writerow(
                    {
                        "RowId": row_id,
                        "Label": label,
                        "Entidade_Referencia": gold_map.get(key, ""),
                        "Entidade_Predita": entity.get("text", ""),
                        "Start": int(entity["start"]),
                        "End": int(entity["end"]),
                        "Score": float(entity.get("score", 1.0)),
                        "Validacao": is_valid,
                    }
                )
                label_counts[label] += 1
                validation_counts[is_valid] += 1

    if args.output_predictions_jsonl:
        output_predictions = Path(args.output_predictions_jsonl)
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        with output_predictions.open("w", encoding="utf-8") as handle:
            for row in prediction_dump:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved calibration CSV to: {output_csv}")
    print(f"Predicted entities: {sum(label_counts.values())}")
    print(f"Validacao=1: {validation_counts.get(1, 0)}")
    print(f"Validacao=0: {validation_counts.get(0, 0)}")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
