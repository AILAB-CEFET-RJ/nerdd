#!/usr/bin/env python3
"""Ad-hoc inference utility for quickly probing a GLiNER checkpoint on a few texts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
DEFAULT_LABELS = ["Person", "Location", "Organization"]


def _parse_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def _load_texts(args) -> list[str]:
    texts = [value.strip() for value in (args.text or []) if value and value.strip()]
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        file_texts = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts.extend(file_texts)
    if not texts:
        raise ValueError("Provide at least one --text or a --file with non-empty lines.")
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
        help="Optional UTF-8 text file with one sample per non-empty line.",
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
        args.model_path,
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
