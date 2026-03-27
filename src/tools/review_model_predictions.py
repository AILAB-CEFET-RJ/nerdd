#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from collections import Counter
from html import escape
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import save_jsonl
from pseudolabelling.evaluate_refit_pipeline import (
    _load_gliner_model,
    compute_span_metrics,
    format_classification_report,
    load_gt_jsonl_strict,
    predict_entities_for_texts,
)
from tools.render_ner_html import build_label_colors, sanitize_spans, render_text_with_spans

LOGGER = logging.getLogger(__name__)


def _to_span_set(spans):
    return {(int(span["start"]), int(span["end"]), str(span["label"])) for span in spans}


def enrich_rows(rows, pred_spans):
    enriched = []
    for index, (row, predicted) in enumerate(zip(rows, pred_spans), start=1):
        gold = row["spans"]
        gold_set = _to_span_set(gold)
        pred_set = _to_span_set(predicted)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        enriched.append(
            {
                "text": row["text"],
                "spans": gold,
                "entities": predicted,
                "_review": {
                    "row_index_1based": index,
                    "gold_count": len(gold),
                    "pred_count": len(predicted),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "error_count": fp + fn,
                },
            }
        )
    enriched.sort(
        key=lambda row: (
            -row["_review"]["error_count"],
            -row["_review"]["fp"],
            -row["_review"]["fn"],
            -row["_review"]["pred_count"],
        )
    )
    return enriched


def _summary(rows, metrics):
    error_counts = [row["_review"]["error_count"] for row in rows]
    return {
        "records": len(rows),
        "avg_errors_per_record": (sum(error_counts) / len(error_counts)) if error_counts else 0.0,
        "max_errors_in_record": max(error_counts) if error_counts else 0,
        "micro_f1": metrics["micro"]["f1"],
        "macro_f1": metrics["macro_f1"],
    }


def build_html(rows, title, metrics):
    label_colors = build_label_colors(rows + [{"entities": row["spans"]} for row in rows])
    report_rows = []
    for row in rows:
        text = row["text"]
        gold = sanitize_spans(text, row["spans"])
        pred = sanitize_spans(text, row["entities"])
        gold_html = render_text_with_spans(text, gold, label_colors)
        pred_html = render_text_with_spans(text, pred, label_colors)
        review = row["_review"]
        report_rows.append(
            "<section class='report'>"
            f"<h3>Record #{review['row_index_1based']}</h3>"
            f"<div class='meta'>gold={review['gold_count']} pred={review['pred_count']} tp={review['tp']} fp={review['fp']} fn={review['fn']}</div>"
            "<div class='panel-grid'>"
            "<div class='panel'>"
            "<h4>Gold</h4>"
            f"<div class='text'>{gold_html}</div>"
            "</div>"
            "<div class='panel'>"
            "<h4>Prediction</h4>"
            f"<div class='text'>{pred_html}</div>"
            "</div>"
            "</div>"
            "</section>"
        )

    report_text = escape(format_classification_report(metrics))
    template = Template(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$title</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #1f2937; }
    h1 { margin-bottom: 8px; }
    .muted { color: #6b7280; margin-bottom: 18px; }
    .summary, .legend { border-collapse: collapse; margin-bottom: 18px; }
    .summary td, .legend td, .legend th { border: 1px solid #d1d5db; padding: 6px 10px; }
    .legend th { background: #f3f4f6; text-align: left; }
    .report { border: 1px solid #d1d5db; border-radius: 8px; margin: 0 0 16px 0; padding: 12px; }
    .report h3 { margin: 0 0 8px 0; font-size: 14px; }
    .report h4 { margin: 0 0 8px 0; font-size: 13px; }
    .meta { font-size: 12px; color: #6b7280; margin-bottom: 10px; }
    .panel-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .panel { border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; background: #fafafa; }
    .text { line-height: 1.7; white-space: pre-wrap; }
    .entity { color: #fff; border-radius: 4px; padding: 0 4px; margin: 0 1px; display: inline-block; }
    .entity .tag { font-size: 10px; margin-left: 6px; opacity: 0.9; }
    pre { background: #111827; color: #f9fafb; padding: 12px; border-radius: 8px; overflow: auto; }
  </style>
</head>
<body>
  <h1>$title</h1>
  <div class="muted">Gold vs prediction review, sorted by worst records first.</div>
  <table class="summary">
    <tr><td><b>Records</b></td><td>$records</td></tr>
    <tr><td><b>Micro F1</b></td><td>$micro_f1</td></tr>
    <tr><td><b>Macro F1</b></td><td>$macro_f1</td></tr>
  </table>
  <pre>$report_text</pre>
  $rows_html
</body>
</html>"""
    )
    return template.substitute(
        title=escape(title),
        records=len(rows),
        micro_f1=f"{metrics['micro']['f1']:.4f}",
        macro_f1=f"{metrics['macro_f1']:.4f}",
        report_text=report_text,
        rows_html="".join(report_rows),
    )


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

    comparison_jsonl = out_dir / "comparison.jsonl"
    metrics_json = out_dir / "metrics.json"
    summary_json = out_dir / "summary.json"
    html_path = out_dir / "review.html"

    save_jsonl(str(comparison_jsonl), enriched_rows)
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json.write_text(json.dumps(_summary(enriched_rows, metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(build_html(enriched_rows, args.title, metrics), encoding="utf-8")

    LOGGER.info("Saved comparison JSONL: %s", comparison_jsonl)
    LOGGER.info("Saved metrics JSON: %s", metrics_json)
    LOGGER.info("Saved summary JSON: %s", summary_json)
    LOGGER.info("Saved HTML review: %s", html_path)


if __name__ == "__main__":
    main()
