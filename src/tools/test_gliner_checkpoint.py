#!/usr/bin/env python3
"""Ad-hoc inference utility for quickly probing a GLiNER checkpoint.

This script is meant for fast qualitative checks on a handful of examples
without generating the full HTML/JSON review artifacts.

Supported inputs:
- `--text "..."` repeated multiple times
- `--file some.txt` with one sample per non-empty line
- `--file some.jsonl` or `--file some.json`
  The loader looks for text in common fields such as `text`, `relato`,
  `texto`, `description`, and `descricao`.

Typical use cases:
- inspect how a trained checkpoint behaves on a few problematic tips
- compare outputs from two checkpoints on the same examples
- re-run the model on ranked pseudolabel candidates stored as JSONL
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.paths import resolve_repo_artifact_path

LOGGER = logging.getLogger(__name__)
DEFAULT_LABELS = ["Person", "Location", "Organization"]


def _parse_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def _extract_text_from_record(record) -> str:
    if not isinstance(record, dict):
        return ""
    for key in ("text", "relato", "texto", "description", "descricao"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_rows_from_json_payload(payload: str, source_name: str) -> list[str]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        texts = []
        for line_number, line in enumerate(payload.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {source_name} at line {line_number}: {exc}") from exc
            text = _extract_text_from_record(record)
            if text:
                texts.append(text)
        return texts

    if isinstance(parsed, list):
        return [text for text in (_extract_text_from_record(item) for item in parsed) if text]
    if isinstance(parsed, dict):
        text = _extract_text_from_record(parsed)
        return [text] if text else []
    raise ValueError(f"Unsupported JSON structure in {source_name}. Expected object, list, or JSONL.")


def _load_texts(args) -> list[str]:
    texts = [value.strip() for value in (args.text or []) if value and value.strip()]
    if args.file:
        file_path = resolve_repo_artifact_path(__file__, args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        payload = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() in {".json", ".jsonl"}:
            file_texts = _load_rows_from_json_payload(payload, str(file_path))
        else:
            file_texts = [line.strip() for line in payload.splitlines() if line.strip()]
        texts.extend(file_texts)
    if not texts:
        raise ValueError("Provide at least one --text or a --file with readable text records.")
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a GLiNER checkpoint on a few texts.")
    parser.add_argument("--model-path", required=True, help="HF repo id or local GLiNER checkpoint path.")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels to predict. Default: Person,Location,Organization",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Input text to evaluate. Repeat --text for multiple samples.",
    )
    parser.add_argument(
        "--file",
        help="Optional UTF-8 text/JSON/JSONL file. Text files use one sample per non-empty line.",
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--model-max-length", type=int, default=0)
    parser.add_argument("--map-location", default="")
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    from base_model_training.evaluate import predict_entities_for_text
    from gliner_loader import load_gliner_model

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    labels = _parse_csv(args.labels)
    if not labels:
        labels = list(DEFAULT_LABELS)
    texts = _load_texts(args)

    model = load_gliner_model(
        str(resolve_repo_artifact_path(__file__, args.model_path)),
        model_max_length=args.model_max_length,
        map_location=args.map_location,
        logger=LOGGER,
        context="probe",
    )

    for index, text in enumerate(texts, start=1):
        entities = predict_entities_for_text(
            model,
            text,
            labels,
            threshold=args.prediction_threshold,
            chunk_size=args.max_tokens,
            batch_size=args.batch_size,
        )
        print("=" * 80)
        print(f"Sample {index}")
        print("=" * 80)
        print(text)
        print("Model output:")
        if not entities:
            print("  (no entities)")
            continue
        for entity in entities:
            score_suffix = f", score={entity.get('score'):.4f}" if args.show_scores and entity.get("score") is not None else ""
            print(f"  - text={entity.get('text')!r}, label={entity.get('label')!r}{score_suffix}")


if __name__ == "__main__":
    main()
