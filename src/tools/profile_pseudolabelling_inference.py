#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl
from base_model_training.paths import resolve_path
from text_chunking import effective_chunk_budget, split_text_fast


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile pseudolabelling inference on a small number of reports."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument(
        "--text-fields",
        default="assunto,relato,bairroLocal,logradouroLocal,cidadeLocal,pontodeReferenciaLocal",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--report-json", default="")
    return parser.parse_args()


def _parse_csv(value):
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def build_inference_text(sample, text_fields, join_separator):
    values = [str(sample.get(field, "")).strip() for field in text_fields]
    values = [value for value in values if value]
    return join_separator.join(values).strip()


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


def main():
    args = parse_args()
    from gliner import GLiNER

    labels = _parse_csv(args.labels)
    text_fields = _parse_csv(args.text_fields)

    script_dir = Path(__file__).resolve().parent
    input_jsonl = resolve_path(script_dir, args.input_jsonl)
    model_path_candidate = resolve_path(script_dir, args.model_path)
    model_path = str(model_path_candidate) if model_path_candidate.exists() else args.model_path

    rows = load_jsonl(str(input_jsonl))[: args.limit]
    model = GLiNER.from_pretrained(model_path, load_tokenizer=True)
    tokenizer = model.data_processor.transformer_tokenizer
    budget = effective_chunk_budget(model, tokenizer, args.max_tokens)

    total_chunking_seconds = 0.0
    total_inference_seconds = 0.0
    total_chunks = 0
    max_chunks = 0
    max_text_len = 0
    max_chunk_text_len = 0
    profiled_rows = []

    started = time.perf_counter()

    for row_idx, row in enumerate(rows, start=1):
        inference_text = build_inference_text(
            sample=row,
            text_fields=text_fields,
            join_separator=". ",
        )
        max_text_len = max(max_text_len, len(inference_text))

        t0 = time.perf_counter()
        chunks = split_text_fast(
            inference_text,
            model=model,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
        )
        t1 = time.perf_counter()

        chunk_texts = list(chunks)
        chunk_count = len(chunk_texts)
        total_chunks += chunk_count
        max_chunks = max(max_chunks, chunk_count)
        if chunk_texts:
            max_chunk_text_len = max(max_chunk_text_len, max(len(chunk) for chunk in chunk_texts))

        batch_seconds = 0.0
        for index in range(0, len(chunk_texts), args.batch_size):
            batch = chunk_texts[index : index + args.batch_size]
            t_batch_0 = time.perf_counter()
            predict_batch_entities(model, batch, labels, args.score_threshold)
            t_batch_1 = time.perf_counter()
            batch_seconds += t_batch_1 - t_batch_0

        chunking_seconds = t1 - t0
        total_chunking_seconds += chunking_seconds
        total_inference_seconds += batch_seconds

        profiled_rows.append(
            {
                "row_index": row_idx,
                "text_len_chars": len(inference_text),
                "chunk_count": chunk_count,
                "chunking_seconds": chunking_seconds,
                "inference_seconds": batch_seconds,
            }
        )

    runtime_seconds = time.perf_counter() - started
    payload = {
        "config": {
            "model_path": model_path,
            "input_jsonl": str(input_jsonl),
            "limit": args.limit,
            "batch_size": args.batch_size,
            "max_tokens_requested": args.max_tokens,
            "effective_chunk_budget": budget,
            "labels": labels,
            "text_fields": text_fields,
        },
        "summary": {
            "rows_profiled": len(profiled_rows),
            "runtime_seconds": runtime_seconds,
            "rows_per_second": (len(profiled_rows) / runtime_seconds) if runtime_seconds else 0.0,
            "avg_chunking_seconds_per_row": (total_chunking_seconds / len(profiled_rows)) if profiled_rows else 0.0,
            "avg_inference_seconds_per_row": (total_inference_seconds / len(profiled_rows)) if profiled_rows else 0.0,
            "avg_chunks_per_row": (total_chunks / len(profiled_rows)) if profiled_rows else 0.0,
            "max_chunks_per_row": max_chunks,
            "max_text_len_chars": max_text_len,
            "max_chunk_text_len_chars": max_chunk_text_len,
        },
        "rows": profiled_rows,
    }

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    if args.report_json:
        print(f"Report JSON: {args.report_json}")


if __name__ == "__main__":
    main()
