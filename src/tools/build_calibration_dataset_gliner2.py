#!/usr/bin/env python3

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.paths import resolve_repo_artifact_path
from gliner2_inference import predict_entities_for_text
from gliner2_loader import load_gliner2_model
from tools.build_calibration_dataset import read_json_or_jsonl


def _parse_csv(raw_value):
    if not raw_value:
        return []
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def _normalize_entity_types(labels):
    mapping = {
        "person": "person",
        "location": "location",
        "organization": "organization",
    }
    entity_types = []
    for label in labels:
        mapped = mapping.get(label.strip().lower())
        if mapped:
            entity_types.append(mapped)
    return entity_types


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a calibration CSV from a labeled dataset and a GLiNER2 model."
    )
    parser.add_argument("--model-path", required=True, help="Path or model id for GLiNER2 base or standalone model.")
    parser.add_argument("--adapter-dir", default="", help="Optional GLiNER2 LoRA adapter directory.")
    parser.add_argument("--input", required=True, help="Labeled JSON/JSONL dataset with text+spans.")
    parser.add_argument("--output-csv", required=True, help="Output calibration CSV path.")
    parser.add_argument("--output-predictions-jsonl", default="", help="Optional predictions dump for auditing.")
    parser.add_argument("--labels", default="Person,Location,Organization", help="Comma-separated labels to predict.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N rows.")
    return parser.parse_args()


def main():
    args = parse_args()
    labels = _parse_csv(args.labels)
    entity_types = _normalize_entity_types(labels)
    if not entity_types:
        raise ValueError("At least one supported label must be provided.")

    input_path = resolve_repo_artifact_path(__file__, args.input)
    rows = read_json_or_jsonl(str(input_path))
    model = load_gliner2_model(
        str(resolve_repo_artifact_path(__file__, args.model_path)),
        adapter_dir=str(resolve_repo_artifact_path(__file__, args.adapter_dir)) if args.adapter_dir else "",
    )

    output_csv = resolve_repo_artifact_path(__file__, args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    prediction_dump = []
    label_counts = Counter()
    validation_counts = Counter()
    started = time.perf_counter()
    total_rows = len(rows)

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

        for row_id, row in enumerate(rows, start=1):
            text = str(row.get("text", "")).strip()
            if not text:
                continue

            gold_spans = row.get("spans", []) or []
            gold_map = {
                (int(span["start"]), int(span["end"]), str(span["label"])): text[int(span["start"]) : int(span["end"])]
                for span in gold_spans
            }
            gold_set = set(gold_map.keys())

            predictions = predict_entities_for_text(model=model, text=text, entity_types=entity_types)
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

            if row_id % args.progress_every == 0 or row_id == total_rows:
                elapsed = time.perf_counter() - started
                rows_per_second = row_id / elapsed if elapsed > 0 else 0.0
                print(
                    json.dumps(
                        {
                            "progress": f"{row_id}/{total_rows}",
                            "elapsed_seconds": elapsed,
                            "rows_per_second": rows_per_second,
                            "predicted_entities": int(sum(label_counts.values())),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if args.output_predictions_jsonl:
        output_predictions = resolve_repo_artifact_path(__file__, args.output_predictions_jsonl)
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
