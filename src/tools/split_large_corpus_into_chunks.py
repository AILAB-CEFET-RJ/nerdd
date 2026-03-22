#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def _parse_jsonl(text):
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


def read_json_or_jsonl(path):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    text = source.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _parse_jsonl(text)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Unsupported JSON format.")


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_rows(rows, chunk_size):
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    for start in range(0, len(rows), chunk_size):
        end = min(start + chunk_size, len(rows))
        yield start, end, rows[start:end]


def build_chunk_name(prefix, index, total_digits):
    return f"{prefix}_{index:0{total_digits}d}.jsonl"


def summarize_chunks(rows_total, chunk_specs, input_path, output_dir, chunk_size):
    return {
        "input_path": str(Path(input_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "rows_total": rows_total,
        "chunk_size": chunk_size,
        "chunks_total": len(chunk_specs),
        "chunks": chunk_specs,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a large JSON/JSONL corpus into fixed-size JSONL chunks."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL corpus.")
    parser.add_argument("--output-dir", required=True, help="Directory where chunk JSONL files will be written.")
    parser.add_argument("--chunk-size", type=int, required=True, help="Rows per chunk.")
    parser.add_argument("--chunk-prefix", default="chunk", help="Prefix for chunk file names.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    total_digits = max(2, len(str(total_chunks)))
    chunk_specs = []

    for chunk_index, (start, end, chunk_rows_value) in enumerate(chunk_rows(rows, args.chunk_size), start=1):
        chunk_name = build_chunk_name(args.chunk_prefix, chunk_index, total_digits)
        chunk_path = output_dir / chunk_name
        write_jsonl(chunk_path, chunk_rows_value)
        chunk_specs.append(
            {
                "chunk_index": chunk_index,
                "output_path": str(chunk_path.resolve()),
                "row_count": len(chunk_rows_value),
                "start_index": start,
                "end_index_exclusive": end,
            }
        )

    summary = summarize_chunks(
        rows_total=len(rows),
        chunk_specs=chunk_specs,
        input_path=args.input,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
    )

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Input rows: {len(rows)}")
    print(f"Chunks written: {len(chunk_specs)}")
    print(f"Output dir: {output_dir}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
