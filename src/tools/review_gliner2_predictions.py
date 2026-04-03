#!/usr/bin/env python3
"""Run a GLiNER2 model on labeled data and generate side-by-side review artifacts."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.paths import resolve_repo_artifact_path
from gliner2_inference import predict_entities_for_text
from gliner2_loader import load_gliner2_model
from pseudolabelling.evaluate_refit_pipeline import (
    compute_span_metrics,
    format_classification_report,
    load_gt_jsonl_strict,
)
from tools.review_prediction_utils import enrich_rows, write_review_artifacts

LOGGER = logging.getLogger(__name__)


def _parse_csv(raw_value):
    if not raw_value:
        return []
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def _normalize_entity_types(labels):
    mapping = {
        "person": "person",
        "location": "location",
        "organization": "organization",
    }
    entity_types = []
    for label in labels:
        mapped = mapping.get(label.strip().lower())
        if mapped:
            entity_types.append(mapped)
    return entity_types


def _resolve_model_path(raw_value):
    candidate = Path(str(raw_value))
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    if "/" in str(raw_value) or "\\" in str(raw_value):
        return str(raw_value)
    return str(resolve_repo_artifact_path(__file__, raw_value))


def predict_entities_for_texts(model, texts, entity_types):
    return [predict_entities_for_text(model, text, entity_types) for text in texts]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a GLiNER2 model on a labeled test set and generate a side-by-side HTML review."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", default="", help="Optional GLiNER2 LoRA adapter directory.")
    parser.add_argument("--gt-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument("--title", default="GLiNER2 Prediction Review")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap after sorting worst-first.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    labels = _parse_csv(args.labels)
    entity_types = _normalize_entity_types(labels)
    if not entity_types:
        raise ValueError("At least one supported label must be provided.")

    out_dir = resolve_repo_artifact_path(__file__, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = resolve_repo_artifact_path(__file__, args.gt_jsonl)
    rows = load_gt_jsonl_strict(str(gt_path))
    model = load_gliner2_model(
        _resolve_model_path(args.model_path),
        adapter_dir=_resolve_model_path(args.adapter_dir) if args.adapter_dir else "",
        logger=LOGGER,
        context="review",
    )
    texts = [row["text"] for row in rows]
    gold_spans = [row["spans"] for row in rows]
    pred_spans = predict_entities_for_texts(model, texts, entity_types)

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
