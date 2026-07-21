#!/usr/bin/env python3
"""Build a reproducible NER annotation sample for inter-annotator agreement.

The output is a JSON list compatible with src/tools/build_ner_annotation_editor_global.py.
By default, spans are cleared so each annotator starts from the same unannotated reports.

Example:
python src/tools/build_annotation_agreement_sample.py \
  --input data/large_sanitized/large_sanitized.jsonl \
  --output data/agreement_sample_100.json \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
from typing import Any


DEFAULT_INPUT = "data/large_sanitized/large_sanitized.jsonl"
DEFAULT_TEXT_FIELDS = ("text", "relato", "texto", "description", "descricao")


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid JSONL at line {line_no}: expected object.")
        rows.append(parsed)
    return rows


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    payload = source.read_text(encoding="utf-8")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return _parse_jsonl(payload)

    if isinstance(parsed, list):
        if not all(isinstance(row, dict) for row in parsed):
            raise ValueError("JSON list input must contain only objects.")
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError("Unsupported input format: expected JSON object/list or JSONL.")


def get_text(row: dict[str, Any], text_field: str) -> str:
    if text_field != "auto":
        value = row.get(text_field)
        return value.strip() if isinstance(value, str) else ""

    for field in DEFAULT_TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_existing_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    for field in ("spans", "entities", "ner"):
        spans = row.get(field)
        if isinstance(spans, list):
            return [span for span in spans if isinstance(span, dict)]
    return []


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def source_ref(row: dict[str, Any], source_index: int) -> str:
    for field in ("sample_id", "id", "source_id"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)

    sanitization = row.get("_sanitization")
    if isinstance(sanitization, dict):
        value = sanitization.get("row_index_1based")
        if value not in (None, ""):
            return f"row-{value}"

    return f"row-{source_index + 1}"


def filter_candidates(
    rows: list[dict[str, Any]],
    *,
    text_field: str,
    min_chars: int,
    max_chars: int,
) -> list[tuple[int, dict[str, Any], str]]:
    candidates: list[tuple[int, dict[str, Any], str]] = []
    for idx, row in enumerate(rows):
        text = get_text(row, text_field)
        if not text:
            continue
        if len(text) < min_chars:
            continue
        if max_chars > 0 and len(text) > max_chars:
            continue
        candidates.append((idx, row, text))
    return candidates


def sample_candidates(
    candidates: list[tuple[int, dict[str, Any], str]],
    *,
    sample_size: int,
    seed: int,
    preserve_input_order: bool,
) -> list[tuple[int, dict[str, Any], str]]:
    if sample_size < 1:
        raise ValueError("--sample-size must be >= 1")
    if sample_size > len(candidates):
        raise ValueError(
            f"--sample-size ({sample_size}) is larger than the candidate pool ({len(candidates)})."
        )

    rng = Random(seed)
    sampled = rng.sample(candidates, sample_size)
    if preserve_input_order:
        sampled = sorted(sampled, key=lambda item: item[0])
    return sampled


def build_editor_records(
    sampled: list[tuple[int, dict[str, Any], str]],
    *,
    input_path: str,
    seed: int,
    keep_existing_spans: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_position, (source_index, row, text) in enumerate(sampled, start=1):
        ref = source_ref(row, source_index)
        records.append(
            {
                "id": f"agreement-{sample_position:04d}-{text_hash(text)}",
                "text": text,
                "spans": get_existing_spans(row) if keep_existing_spans else [],
                "_agreement_sample": {
                    "sample_position": sample_position,
                    "source_ref": ref,
                    "source_index_0based": source_index,
                    "source_index_1based": source_index + 1,
                    "text_sha1_12": text_hash(text),
                    "seed": seed,
                    "input_path": str(Path(input_path)),
                },
            }
        )
    return records


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible unannotated NER sample compatible with "
            "build_ner_annotation_editor_global.py."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON or JSONL corpus.")
    parser.add_argument("--output", required=True, help="Output JSON sample path.")
    parser.add_argument("--sample-size", type=int, required=True, help="Number of reports to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--text-field",
        default="auto",
        help="Text field to sample from, or 'auto' for common corpus fields.",
    )
    parser.add_argument("--min-chars", type=int, default=1, help="Minimum text length.")
    parser.add_argument("--max-chars", type=int, default=0, help="Maximum text length (0 = no limit).")
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Write records in sampled order instead of preserving input order.",
    )
    parser.add_argument(
        "--keep-existing-spans",
        action="store_true",
        help="Copy existing spans/entities/ner from input instead of starting with empty spans.",
    )
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    candidates = filter_candidates(
        rows,
        text_field=args.text_field,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    sampled = sample_candidates(
        candidates,
        sample_size=args.sample_size,
        seed=args.seed,
        preserve_input_order=not args.shuffle_output,
    )
    records = build_editor_records(
        sampled,
        input_path=args.input,
        seed=args.seed,
        keep_existing_spans=args.keep_existing_spans,
    )
    write_json(args.output, records)

    summary = {
        "input_path": str(Path(args.input).resolve()),
        "output_path": str(Path(args.output).resolve()),
        "rows_total": len(rows),
        "candidate_rows": len(candidates),
        "rows_sampled": len(records),
        "seed": args.seed,
        "text_field": args.text_field,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "preserve_input_order": not args.shuffle_output,
        "keep_existing_spans": args.keep_existing_spans,
        "sampled_source_indices_0based": [source_index for source_index, _, _ in sampled],
    }
    if args.summary_json:
        write_json(args.summary_json, summary)

    print(f"Input rows: {len(rows)}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Sampled rows: {len(records)}")
    print(f"Output JSON: {args.output}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
