#!/usr/bin/env python3
"""Generate focused NER error-analysis artifacts for one label."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from html import escape
from pathlib import Path
from string import Template
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl, save_jsonl
from pseudolabelling.evaluate_refit_pipeline import compute_span_metrics, load_gt_jsonl_strict
from tools.render_ner_html import build_label_colors, render_text_with_spans, sanitize_spans

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare gold spans and prediction spans, then write focused error-analysis "
            "artifacts for a target label."
        )
    )
    parser.add_argument("--gold-json", default="", help="Gold dataset JSON/JSONL. Optional if predictions contain gold_spans.")
    parser.add_argument("--pred-jsonl", required=True, help="Predictions JSONL/JSON. Supports entities or pred_spans fields.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="Organization")
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument("--title", default="")
    parser.add_argument("--max-html-records", type=int, default=300)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _parse_labels(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["label"])


def _normalize_span(span: dict[str, Any], text: str) -> dict[str, Any]:
    normalized = dict(span)
    normalized["start"] = int(normalized["start"])
    normalized["end"] = int(normalized["end"])
    normalized["label"] = str(normalized["label"])
    normalized["mention"] = text[normalized["start"] : normalized["end"]]
    return normalized


def _normalize_spans(spans: list[dict[str, Any]] | None, text: str) -> list[dict[str, Any]]:
    return [_normalize_span(span, text) for span in (spans or [])]


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["start"]) < int(right["end"]) and int(right["start"]) < int(left["end"])


def _context(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return f"{prefix}{text[left:right]}{suffix}"


def _span_public(span: dict[str, Any]) -> dict[str, Any]:
    keys = ["start", "end", "label", "mention", "score", "ner_score", "confidence", "probability"]
    return {key: span[key] for key in keys if key in span}


def _classify_missed_gold(gold_span: dict[str, Any], pred_spans: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    overlapping = [span for span in pred_spans if _overlaps(gold_span, span)]
    if any(span["label"] == gold_span["label"] for span in overlapping):
        return "boundary_mismatch", overlapping
    if overlapping:
        return "label_confusion", overlapping
    return "missed", []


def _classify_extra_prediction(pred_span: dict[str, Any], gold_spans: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    overlapping = [span for span in gold_spans if _overlaps(pred_span, span)]
    if any(span["label"] == pred_span["label"] for span in overlapping):
        return "boundary_mismatch", overlapping
    if overlapping:
        return "label_confusion", overlapping
    return "spurious", []


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(str(path))
    normalized = []
    for idx, row in enumerate(rows, start=1):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Prediction row {idx} has no string text field.")
        pred_spans = row.get("pred_spans", row.get("entities", []))
        gold_spans = row.get("gold_spans", row.get("spans"))
        normalized.append(
            {
                "text": text,
                "pred_spans": pred_spans or [],
                "gold_spans": gold_spans,
                "sample_id": row.get("sample_id"),
                "fold": row.get("fold"),
            }
        )
    return normalized


def _pair_rows(gold_path: Path | None, prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if gold_path is None:
        paired = []
        missing = [idx for idx, row in enumerate(prediction_rows, start=1) if row.get("gold_spans") is None]
        if missing:
            raise ValueError(
                "--gold-json is required because prediction rows do not contain gold_spans "
                f"(first missing row: {missing[0]})."
            )
        for idx, row in enumerate(prediction_rows, start=1):
            paired.append(
                {
                    "row_index_1based": idx,
                    "sample_id": row.get("sample_id"),
                    "fold": row.get("fold"),
                    "text": row["text"],
                    "gold_spans": row["gold_spans"] or [],
                    "pred_spans": row["pred_spans"] or [],
                }
            )
        return paired

    gold_rows = load_gt_jsonl_strict(str(gold_path))
    if len(gold_rows) != len(prediction_rows):
        raise ValueError(
            f"Gold and prediction row counts differ: {len(gold_rows)} gold vs {len(prediction_rows)} predictions."
        )

    paired = []
    for idx, (gold_row, pred_row) in enumerate(zip(gold_rows, prediction_rows), start=1):
        if gold_row["text"] != pred_row["text"]:
            raise ValueError(f"Text mismatch at row {idx}; refusing to align predictions by index.")
        paired.append(
            {
                "row_index_1based": idx,
                "sample_id": gold_row.get("sample_id", pred_row.get("sample_id")),
                "fold": pred_row.get("fold"),
                "text": gold_row["text"],
                "gold_spans": gold_row.get("spans", []) or [],
                "pred_spans": pred_row.get("pred_spans", []) or [],
            }
        )
    return paired


def build_label_error_report(rows: list[dict[str, Any]], target_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    error_items: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for row_index_0based, row in enumerate(rows):
        text = row["text"]
        gold_spans = _normalize_spans(row["gold_spans"], text)
        pred_spans = _normalize_spans(row["pred_spans"], text)
        gold_set = {_span_key(span) for span in gold_spans}
        pred_set = {_span_key(span) for span in pred_spans}
        row_errors = []

        for gold_span in gold_spans:
            if gold_span["label"] != target_label or _span_key(gold_span) in pred_set:
                continue
            error_type, overlapping = _classify_missed_gold(gold_span, pred_spans)
            item = _build_error_item(
                row=row,
                row_index_0based=row_index_0based,
                direction="FN",
                error_type=error_type,
                target_span=gold_span,
                overlapping_spans=overlapping,
                text=text,
            )
            error_items.append(item)
            row_errors.append(item)

        for pred_span in pred_spans:
            if pred_span["label"] != target_label or _span_key(pred_span) in gold_set:
                continue
            error_type, overlapping = _classify_extra_prediction(pred_span, gold_spans)
            item = _build_error_item(
                row=row,
                row_index_0based=row_index_0based,
                direction="FP",
                error_type=error_type,
                target_span=pred_span,
                overlapping_spans=overlapping,
                text=text,
            )
            error_items.append(item)
            row_errors.append(item)

        if row_errors:
            review_rows.append(
                {
                    "row_index_0based": row_index_0based,
                    "row_index_1based": row["row_index_1based"],
                    "sample_id": row.get("sample_id"),
                    "fold": row.get("fold"),
                    "text": text,
                    "gold_spans": gold_spans,
                    "pred_spans": pred_spans,
                    "errors": row_errors,
                    "error_count": len(row_errors),
                    "fn_count": sum(1 for item in row_errors if item["direction"] == "FN"),
                    "fp_count": sum(1 for item in row_errors if item["direction"] == "FP"),
                }
            )

    error_items.sort(key=_error_sort_key)
    review_rows.sort(key=lambda row: (-row["error_count"], -row["fn_count"], -row["fp_count"], row["row_index_1based"]))
    return error_items, review_rows


def _build_error_item(
    *,
    row: dict[str, Any],
    row_index_0based: int,
    direction: str,
    error_type: str,
    target_span: dict[str, Any],
    overlapping_spans: list[dict[str, Any]],
    text: str,
) -> dict[str, Any]:
    return {
        "row_index_0based": row_index_0based,
        "row_index_1based": row["row_index_1based"],
        "sample_id": row.get("sample_id"),
        "fold": row.get("fold"),
        "direction": direction,
        "error_type": error_type,
        "label": target_span["label"],
        "mention": target_span["mention"],
        "start": target_span["start"],
        "end": target_span["end"],
        "score": target_span.get("score", target_span.get("ner_score")),
        "overlapping_spans": [_span_public(span) for span in overlapping_spans],
        "context": _context(text, target_span["start"], target_span["end"]),
        "text_preview": text[:180],
    }


def _error_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    direction_rank = {"FN": 0, "FP": 1}
    type_rank = {"label_confusion": 0, "boundary_mismatch": 1, "missed": 2, "spurious": 3}
    score = item.get("score")
    score_sort = -float(score) if isinstance(score, (int, float)) else 0.0
    return (
        direction_rank.get(item["direction"], 9),
        type_rank.get(item["error_type"], 9),
        item["mention"].casefold(),
        score_sort,
        item["row_index_1based"],
    )


def _summarize(
    *,
    target_label: str,
    labels: list[str],
    paired_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_spans = [row["gold_spans"] for row in paired_rows]
    pred_spans = [row["pred_spans"] for row in paired_rows]
    metrics = compute_span_metrics(gold_spans, pred_spans, labels)
    label_metrics = metrics["per_label"].get(target_label, {})
    type_counts = Counter(item["error_type"] for item in errors)
    direction_counts = Counter(item["direction"] for item in errors)
    mention_counts = Counter((item["direction"], item["error_type"], item["mention"]) for item in errors)
    return {
        "target_label": target_label,
        "records": len(paired_rows),
        "records_with_target_label_errors": len(review_rows),
        "error_items": len(errors),
        "direction_counts": dict(direction_counts),
        "error_type_counts": dict(type_counts),
        "target_label_metrics": label_metrics,
        "overall_metrics": metrics,
        "top_error_mentions": [
            {"direction": direction, "error_type": error_type, "mention": mention, "count": count}
            for (direction, error_type, mention), count in mention_counts.most_common(50)
        ],
    }


def _safe_filename_label(label: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "label"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_index_0based",
        "row_index_1based",
        "sample_id",
        "fold",
        "direction",
        "error_type",
        "label",
        "mention",
        "start",
        "end",
        "score",
        "overlapping_spans",
        "context",
        "text_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["overlapping_spans"] = json.dumps(row.get("overlapping_spans", []), ensure_ascii=False)
            writer.writerow(serialized)


def _build_html(review_rows: list[dict[str, Any]], summary: dict[str, Any], title: str, max_records: int) -> str:
    limited_rows = review_rows[:max_records] if max_records > 0 else review_rows
    label_colors = build_label_colors(
        [{"spans": row["gold_spans"], "entities": row["pred_spans"]} for row in limited_rows]
    )
    sections = []
    for row in limited_rows:
        gold_html = render_text_with_spans(row["text"], sanitize_spans(row["text"], row["gold_spans"]), label_colors)
        pred_html = render_text_with_spans(row["text"], sanitize_spans(row["text"], row["pred_spans"]), label_colors)
        error_lines = []
        for item in row["errors"]:
            overlap = ", ".join(
                f"{span.get('label')}[{span.get('start')}:{span.get('end')}]={span.get('mention')!r}"
                for span in item.get("overlapping_spans", [])
            )
            overlap_text = f" | overlap: {escape(overlap)}" if overlap else ""
            score_text = f" | score={item['score']:.4f}" if isinstance(item.get("score"), (int, float)) else ""
            error_lines.append(
                "<li>"
                f"<b>{escape(item['direction'])}</b> "
                f"{escape(item['error_type'])}: "
                f"<code>{escape(item['mention'])}</code> "
                f"[{item['start']}:{item['end']}]"
                f"{score_text}{overlap_text}"
                "</li>"
            )
        sections.append(
            "<section class='record'>"
            f"<h3>Relato #{row['row_index_1based']}</h3>"
            f"<div class='meta'>erros={row['error_count']} fn={row['fn_count']} fp={row['fp_count']}</div>"
            f"<ul class='errors'>{''.join(error_lines)}</ul>"
            "<div class='grid'>"
            f"<div><h4>Gold</h4><div class='text'>{gold_html}</div></div>"
            f"<div><h4>Prediction</h4><div class='text'>{pred_html}</div></div>"
            "</div>"
            "</section>"
        )

    metrics = summary["target_label_metrics"]
    template = Template(
        """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$title</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }
    h1 { margin: 0 0 8px; }
    h3 { margin: 0 0 6px; font-size: 15px; }
    h4 { margin: 0 0 8px; font-size: 13px; }
    code { background: #eef2f7; border-radius: 4px; padding: 1px 4px; }
    .muted, .meta { color: #667085; font-size: 13px; }
    .summary { border-collapse: collapse; margin: 16px 0 20px; }
    .summary td { border: 1px solid #d0d5dd; padding: 6px 10px; }
    .record { border: 1px solid #d0d5dd; border-radius: 8px; padding: 12px; margin-bottom: 14px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px; }
    .grid > div { border: 1px solid #eaecf0; border-radius: 6px; padding: 10px; background: #fcfcfd; }
    .text { line-height: 1.7; white-space: pre-wrap; }
    .errors { margin: 8px 0; padding-left: 20px; }
    .entity { color: #fff; border-radius: 4px; padding: 0 4px; margin: 0 1px; display: inline-block; }
    .entity .tag { font-size: 10px; margin-left: 6px; opacity: 0.9; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>$title</h1>
  <div class="muted">Relatos com erro na label alvo, ordenados por número de erros.</div>
  <table class="summary">
    <tr><td><b>Label</b></td><td>$label</td></tr>
    <tr><td><b>Relatos</b></td><td>$records</td></tr>
    <tr><td><b>Relatos com erro</b></td><td>$error_records</td></tr>
    <tr><td><b>Erros</b></td><td>$errors</td></tr>
    <tr><td><b>Precision</b></td><td>$precision</td></tr>
    <tr><td><b>Recall</b></td><td>$recall</td></tr>
    <tr><td><b>F1</b></td><td>$f1</td></tr>
  </table>
  $sections
</body>
</html>"""
    )
    return template.substitute(
        title=escape(title),
        label=escape(summary["target_label"]),
        records=summary["records"],
        error_records=summary["records_with_target_label_errors"],
        errors=summary["error_items"],
        precision=f"{metrics.get('precision', 0.0):.4f}",
        recall=f"{metrics.get('recall', 0.0):.4f}",
        f1=f"{metrics.get('f1', 0.0):.4f}",
        sections="".join(sections),
    )


def write_outputs(out_dir: Path, target_label: str, errors: list[dict[str, Any]], review_rows: list[dict[str, Any]], summary: dict[str, Any], title: str, max_html_records: int) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_filename_label(target_label)
    errors_jsonl = out_dir / f"{prefix}_errors.jsonl"
    review_jsonl = out_dir / f"{prefix}_review_rows.jsonl"
    summary_json = out_dir / f"{prefix}_error_summary.json"
    fp_csv = out_dir / f"{prefix}_false_positives.csv"
    fn_csv = out_dir / f"{prefix}_false_negatives.csv"
    all_csv = out_dir / f"{prefix}_errors.csv"
    html_path = out_dir / f"{prefix}_error_review.html"

    save_jsonl(str(errors_jsonl), errors)
    save_jsonl(str(review_jsonl), review_rows)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(all_csv, errors)
    _write_csv(fp_csv, [item for item in errors if item["direction"] == "FP"])
    _write_csv(fn_csv, [item for item in errors if item["direction"] == "FN"])
    html_path.write_text(_build_html(review_rows, summary, title, max_html_records), encoding="utf-8")

    return {
        "errors_jsonl": errors_jsonl,
        "review_jsonl": review_jsonl,
        "summary_json": summary_json,
        "errors_csv": all_csv,
        "false_positives_csv": fp_csv,
        "false_negatives_csv": fn_csv,
        "html": html_path,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    labels = _parse_labels(args.labels)
    if args.label not in labels:
        labels.append(args.label)

    pred_path = Path(args.pred_jsonl)
    gold_path = Path(args.gold_json) if args.gold_json else None
    prediction_rows = _load_prediction_rows(pred_path)
    paired_rows = _pair_rows(gold_path, prediction_rows)
    errors, review_rows = build_label_error_report(paired_rows, args.label)
    summary = _summarize(
        target_label=args.label,
        labels=labels,
        paired_rows=paired_rows,
        errors=errors,
        review_rows=review_rows,
    )
    title = args.title or f"{args.label} NER Error Analysis"
    outputs = write_outputs(Path(args.output_dir), args.label, errors, review_rows, summary, title, args.max_html_records)
    for name, path in outputs.items():
        LOGGER.info("Saved %s: %s", name, path)
    LOGGER.info(
        "%s errors: %s | records with errors: %s | F1=%.4f",
        args.label,
        len(errors),
        len(review_rows),
        summary["target_label_metrics"].get("f1", 0.0),
    )


if __name__ == "__main__":
    main()
