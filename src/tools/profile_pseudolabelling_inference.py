#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl
from base_model_training.paths import resolve_path
from text_chunking import effective_chunk_budget, model_position_limit, split_text_fast


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile pseudolabelling inference on a small number of reports."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-max-length", type=int, default=0)
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
    parser.add_argument("--progress-every", type=int, default=10)
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


def find_with_cursor(text, part, cursor):
    position = text.find(part, cursor)
    if position == -1:
        position = text.find(part)
    return position


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
    model_kwargs = {"load_tokenizer": True}
    if args.model_max_length > 0:
        model_kwargs["max_length"] = args.model_max_length
    model = GLiNER.from_pretrained(model_path, **model_kwargs)
    tokenizer = model.data_processor.transformer_tokenizer
    budget = effective_chunk_budget(model, tokenizer, args.max_tokens)
    position_limit = model_position_limit(model)

    total_chunking_seconds = 0.0
    total_inference_seconds = 0.0
    total_chunks = 0
    max_chunks = 0
    max_text_len = 0
    max_chunk_text_len = 0
    profiled_rows = []

    started = time.perf_counter()

    for start_idx in range(0, len(rows), args.batch_size):
        batch_rows = rows[start_idx : start_idx + args.batch_size]
        batch_texts = [
            build_inference_text(
                sample=row,
                text_fields=text_fields,
                join_separator=". ",
            )
            for row in batch_rows
        ]

        t0 = time.perf_counter()
        chunk_records = []
        batch_row_chunk_counts = []
        for row_offset, inference_text in enumerate(batch_texts):
            max_text_len = max(max_text_len, len(inference_text))
            chunk_texts = split_text_fast(
                inference_text,
                model=model,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
            )
            batch_row_chunk_counts.append(len(chunk_texts))
            total_chunks += len(chunk_texts)
            max_chunks = max(max_chunks, len(chunk_texts))
            if chunk_texts:
                max_chunk_text_len = max(max_chunk_text_len, max(len(chunk) for chunk in chunk_texts))
            cursor = 0
            for chunk_text in chunk_texts:
                chunk_offset = find_with_cursor(inference_text, chunk_text, cursor)
                if chunk_offset == -1:
                    continue
                cursor = chunk_offset + len(chunk_text)
                chunk_records.append(
                    {
                        "row_index": start_idx + row_offset + 1,
                        "chunk_text": chunk_text,
                        "chunk_offset": chunk_offset,
                    }
                )
        t1 = time.perf_counter()

        batch_seconds = 0.0
        for index in range(0, len(chunk_records), args.batch_size):
            batch = [record["chunk_text"] for record in chunk_records[index : index + args.batch_size]]
            t_batch_0 = time.perf_counter()
            predict_batch_entities(model, batch, labels, args.score_threshold)
            t_batch_1 = time.perf_counter()
            batch_seconds += t_batch_1 - t_batch_0

        rows_in_batch = len(batch_rows)
        total_chunking_seconds += t1 - t0
        total_inference_seconds += batch_seconds

        for row_offset, inference_text in enumerate(batch_texts, start=1):
            profiled_rows.append(
                {
                    "row_index": start_idx + row_offset,
                    "text_len_chars": len(inference_text),
                    "chunk_count": batch_row_chunk_counts[row_offset - 1],
                    "chunking_seconds": (t1 - t0) / rows_in_batch if rows_in_batch else 0.0,
                    "inference_seconds": batch_seconds / rows_in_batch if rows_in_batch else 0.0,
                }
            )

        rows_done = len(profiled_rows)
        if args.progress_every > 0 and (rows_done % args.progress_every == 0 or rows_done == len(rows)):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "progress": f"{rows_done}/{len(rows)}",
                        "elapsed_seconds": elapsed,
                        "rows_per_second": (rows_done / elapsed) if elapsed else 0.0,
                        "avg_chunking_seconds_per_row": total_chunking_seconds / rows_done,
                        "avg_inference_seconds_per_row": total_inference_seconds / rows_done,
                        "avg_chunks_per_row": total_chunks / rows_done,
                        "max_chunks_per_row": max_chunks,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    runtime_seconds = time.perf_counter() - started
    payload = {
        "config": {
            "model_path": model_path,
            "input_jsonl": str(input_jsonl),
            "limit": args.limit,
            "batch_size": args.batch_size,
            "model_max_length": args.model_max_length,
            "model_position_limit": position_limit,
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
