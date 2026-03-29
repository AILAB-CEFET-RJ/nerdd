#!/usr/bin/env python3
"""Sample a reproducible subset from the large corpus and save summary metadata."""

import argparse
import json
from pathlib import Path
from random import Random


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


def sample_rows(rows, sample_size, seed, preserve_input_order=True):
    if sample_size < 1:
        raise ValueError("--sample-size must be >= 1")
    if sample_size > len(rows):
        raise ValueError(
            f"--sample-size ({sample_size}) is larger than the dataset size ({len(rows)})."
        )

    rng = Random(seed)
    indices = rng.sample(range(len(rows)), sample_size)
    if preserve_input_order:
        indices = sorted(indices)
    return [rows[idx] for idx in indices], indices


def summarize_sample(rows, indices, input_path, output_path, seed, preserve_input_order):
    return {
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(Path(output_path).resolve()),
        "rows_total": len(rows),
        "rows_sampled": len(indices),
        "seed": seed,
        "preserve_input_order": preserve_input_order,
        "sampled_index_min": min(indices) if indices else None,
        "sampled_index_max": max(indices) if indices else None,
        "sampled_indices_preview": indices[:20],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a reproducible JSONL sample from a large JSON/JSONL corpus."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL corpus.")
    parser.add_argument("--output-jsonl", required=True, help="Output sampled JSONL path.")
    parser.add_argument("--sample-size", type=int, required=True, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Write sampled rows in sampled order instead of preserving input order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    preserve_input_order = not args.shuffle_output
    sampled_rows, sampled_indices = sample_rows(
        rows,
        sample_size=args.sample_size,
        seed=args.seed,
        preserve_input_order=preserve_input_order,
    )
    write_jsonl(args.output_jsonl, sampled_rows)

    summary = summarize_sample(
        rows=rows,
        indices=sampled_indices,
        input_path=args.input,
        output_path=args.output_jsonl,
        seed=args.seed,
        preserve_input_order=preserve_input_order,
    )

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Input rows: {len(rows)}")
    print(f"Sampled rows: {len(sampled_rows)}")
    print(f"Output JSONL: {args.output_jsonl}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
