#!/usr/bin/env python3
"""Ad-hoc inference utility for quickly probing GLiNER and GLiNER2 checkpoints.

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
- probe either a classic GLiNER checkpoint or a GLiNER2 base/adapter setup
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
DEFAULT_GLINER2_ENTITY_TYPES = ["person", "location", "organization"]


def _parse_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def _resolve_model_path(raw_value: str) -> str:
    candidate = Path(str(raw_value))
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    if "/" in str(raw_value) or "\\" in str(raw_value):
        return str(raw_value)
    return str(resolve_repo_artifact_path(__file__, raw_value))


def _extract_text_from_record(record) -> str:
    if not isinstance(record, dict):
        return ""
    for key in ("text", "relato", "texto", "description", "descricao"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_record_score(record):
    if not isinstance(record, dict):
        return None
    for key in ("record_score", "record_score_context_boosted", "score_relato_confianca", "score_relato"):
        value = record.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_entity_score(entity):
    if not isinstance(entity, dict):
        return None
    for key in ("score", "confidence", "probability"):
        value = entity.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _load_rows_from_json_payload(payload: str, source_name: str) -> list[dict]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        rows = []
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
                rows.append({"text": text, "record_score": _extract_record_score(record)})
        return rows

    if isinstance(parsed, list):
        rows = []
        for item in parsed:
            text = _extract_text_from_record(item)
            if text:
                rows.append({"text": text, "record_score": _extract_record_score(item)})
        return rows
    if isinstance(parsed, dict):
        text = _extract_text_from_record(parsed)
        return [{"text": text, "record_score": _extract_record_score(parsed)}] if text else []
    raise ValueError(f"Unsupported JSON structure in {source_name}. Expected object, list, or JSONL.")


def _load_rows(args) -> list[dict]:
    rows = [{"text": value.strip(), "record_score": None} for value in (args.text or []) if value and value.strip()]
    if args.file:
        file_path = resolve_repo_artifact_path(__file__, args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        payload = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() in {".json", ".jsonl"}:
            file_rows = _load_rows_from_json_payload(payload, str(file_path))
        else:
            file_rows = [{"text": line.strip(), "record_score": None} for line in payload.splitlines() if line.strip()]
        rows.extend(file_rows)
    if not rows:
        raise ValueError("Provide at least one --text or a --file with readable text records.")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a GLiNER checkpoint on a few texts.")
    parser.add_argument(
        "--backend",
        default="gliner",
        choices=["gliner", "gliner2"],
        help="Model family to use. 'gliner' loads classic GLiNER checkpoints; 'gliner2' loads GLiNER2.",
    )
    parser.add_argument("--model-path", required=True, help="HF repo id or local GLiNER checkpoint path.")
    parser.add_argument("--adapter-dir", default="", help="Optional GLiNER2 adapter directory when --backend gliner2 is used.")
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


def _normalize_gliner2_labels(labels: list[str]) -> list[str]:
    mapping = {
        "person": "person",
        "location": "location",
        "organization": "organization",
    }
    normalized = []
    for label in labels:
        mapped = mapping.get(str(label).strip().lower())
        if mapped:
            normalized.append(mapped)
    return normalized or list(DEFAULT_GLINER2_ENTITY_TYPES)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    labels = _parse_csv(args.labels)
    if not labels:
        labels = list(DEFAULT_LABELS)
    rows = _load_rows(args)

    if args.backend == "gliner":
        from base_model_training.evaluate import predict_entities_for_text
        from gliner_loader import load_gliner_model

        model = load_gliner_model(
            _resolve_model_path(args.model_path),
            model_max_length=args.model_max_length,
            map_location=args.map_location,
            logger=LOGGER,
            context="probe",
        )

        def run_inference(text: str):
            return predict_entities_for_text(
                model,
                text,
                labels,
                threshold=args.prediction_threshold,
                chunk_size=args.max_tokens,
                batch_size=args.batch_size,
            )

    else:
        from gliner2_inference import predict_entities_for_text as predict_gliner2_entities
        from gliner2_loader import load_gliner2_model

        entity_types = _normalize_gliner2_labels(labels)
        model = load_gliner2_model(
            _resolve_model_path(args.model_path),
            adapter_dir=_resolve_model_path(args.adapter_dir) if args.adapter_dir else "",
            logger=LOGGER,
            context="probe",
        )

        def run_inference(text: str):
            return predict_gliner2_entities(model, text, entity_types)

    for index, row in enumerate(rows, start=1):
        text = row["text"]
        record_score = row.get("record_score")
        entities = run_inference(text)
        print("=" * 80)
        print(f"Sample {index}")
        print("=" * 80)
        if record_score is not None:
            print(f"Record score: {record_score:.6f}")
        print(text)
        print("Model output:")
        if not entities:
            print("  (no entities)")
            continue
        for entity in entities:
            entity_score = _extract_entity_score(entity)
            score_suffix = f", ner_score={entity_score:.4f}" if args.show_scores and entity_score is not None else ""
            print(f"  - text={entity.get('text')!r}, label={entity.get('label')!r}{score_suffix}")


if __name__ == "__main__":
    main()
