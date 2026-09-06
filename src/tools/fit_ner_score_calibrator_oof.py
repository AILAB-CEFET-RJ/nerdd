#!/usr/bin/env python3
"""Fit a reusable NER score calibrator from out-of-fold predictions."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl
from base_model_training.paths import resolve_path
from calibration.ner_score_calibrator import (
    build_examples_from_oof_rows,
    fit_ner_score_calibrator,
    metric_summary,
)

LOGGER = logging.getLogger(__name__)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a per-label NER score calibrator from OOF predictions.")
    parser.add_argument("--oof-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--pred-field", default="pred_spans")
    parser.add_argument("--gold-field", default="gold_spans")
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    started = datetime.now(timezone.utc)
    timer = perf_counter()
    script_dir = Path(__file__).resolve().parent
    labels = parse_csv(args.labels)
    output_dir = resolve_path(script_dir, args.output_dir)
    oof_path = resolve_path(script_dir, args.oof_predictions)

    rows = load_jsonl(str(oof_path))
    examples, gold_support = build_examples_from_oof_rows(
        rows,
        labels=set(labels),
        score_field=args.score_field,
        pred_field=args.pred_field,
        gold_field=args.gold_field,
    )
    if not examples:
        raise ValueError("No calibration examples were built from OOF predictions.")

    calibrator, calibrated_examples = fit_ner_score_calibrator(
        examples,
        labels=labels,
        method=args.method,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
    )
    metrics, raw_reliability, calibrated_reliability = metric_summary(
        calibrated_examples,
        labels=labels,
        bins=args.bins,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator_path = output_dir / "calibrator.json"
    examples_csv = output_dir / "calibration_examples.csv"
    raw_bins_csv = output_dir / "reliability_raw_by_label.csv"
    calibrated_bins_csv = output_dir / "reliability_calibrated_by_label.csv"
    summary_path = output_dir / "calibration_summary.json"

    calibrator["metadata"] = {
        "fitted_at_utc": started.isoformat(),
        "oof_predictions": str(oof_path),
        "score_field": args.score_field,
        "pred_field": args.pred_field,
        "gold_field": args.gold_field,
        "gold_support": gold_support,
    }
    write_json(calibrator_path, calibrator)
    write_csv(examples_csv, calibrated_examples)
    write_csv(raw_bins_csv, raw_reliability)
    write_csv(calibrated_bins_csv, calibrated_reliability)
    write_json(
        summary_path,
        {
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": perf_counter() - timer,
            "config": vars(args),
            "oof_predictions": str(oof_path),
            "oof_rows": len(rows),
            "calibration_examples": len(calibrated_examples),
            "gold_support": gold_support,
            "label_models": calibrator["label_models"],
            "metrics": metrics,
            "artifacts": {
                "calibrator": str(calibrator_path),
                "calibration_examples": str(examples_csv),
                "reliability_raw_by_label": str(raw_bins_csv),
                "reliability_calibrated_by_label": str(calibrated_bins_csv),
                "summary": str(summary_path),
            },
        },
    )
    LOGGER.info("Saved calibrator: %s", calibrator_path)
    LOGGER.info("Saved calibration summary: %s", summary_path)
    LOGGER.info("Calibration examples: %s", len(calibrated_examples))


if __name__ == "__main__":
    main()
