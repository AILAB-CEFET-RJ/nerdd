#!/usr/bin/env python3
"""Run a GLiNER model on labeled data and generate side-by-side review artifacts."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.evaluate_refit_pipeline import (
    _load_gliner_model,
    compute_span_metrics,
    format_classification_report,
    load_gt_jsonl_strict,
    predict_entities_for_texts,
)
from tools.review_prediction_utils import enrich_rows, write_review_artifacts

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a model on a labeled test set and generate a side-by-side HTML review."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gt-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model-max-length", type=int, default=384)
    parser.add_argument("--map-location", default="")
    parser.add_argument("--title", default="Model Prediction Review")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap after sorting worst-first.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_gt_jsonl_strict(args.gt_jsonl)
    model = _load_gliner_model(args.model_path, args.model_max_length, args.map_location)
    texts = [row["text"] for row in rows]
    gold_spans = [row["spans"] for row in rows]
    pred_spans = predict_entities_for_texts(
        model=model,
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        threshold=args.prediction_threshold,
    )

    metrics = compute_span_metrics(gold_spans, pred_spans, labels)
    enriched_rows = enrich_rows(rows, pred_spans)
    if args.max_records > 0:
        enriched_rows = enriched_rows[: args.max_records]

    outputs = write_review_artifacts(out_dir, enriched_rows, metrics, args.title)
    LOGGER.info("Saved comparison JSONL: %s", outputs["comparison_jsonl"])
    LOGGER.info("Saved metrics JSON: %s", outputs["metrics_json"])
    LOGGER.info("Saved summary JSON: %s", outputs["summary_json"])
    LOGGER.info("Saved HTML review: %s", outputs["html_path"])
    LOGGER.info("Classification report:\n%s", format_classification_report(metrics))


if __name__ == "__main__":
    main()
