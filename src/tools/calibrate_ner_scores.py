#!/usr/bin/env python3
"""Evaluate how NER prediction scores map to exact-match precision."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl
from pseudolabelling.evaluate_refit_pipeline import load_gt_jsonl_strict

LOGGER = logging.getLogger(__name__)
SCORE_FIELDS = ("score", "ner_score", "confidence", "probability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read NER predictions with scores, compare them against gold spans, "
            "and report empirical precision by score bin and threshold."
        )
    )
    parser.add_argument("--gold-json", default="", help="Gold dataset JSON/JSONL. Optional if predictions contain gold_spans.")
    parser.add_argument("--pred-jsonl", required=True, help="Predictions JSONL/JSON. Supports entities or pred_spans fields.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument("--bins", type=int, default=10, help="Number of equal-width reliability bins over [0, 1].")
    parser.add_argument(
        "--thresholds",
        default="0.5,0.6,0.7,0.8,0.85,0.9,0.95",
        help="Comma-separated thresholds for precision-at-threshold reporting.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_labels(value: str) -> list[str]:
    labels = _parse_csv_values(value)
    if not labels:
        raise ValueError("At least one label is required.")
    return labels


def _parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item) for item in _parse_csv_values(value)]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return sorted(set(thresholds))


def _span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["label"])


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["start"]) < int(right["end"]) and int(right["start"]) < int(left["end"])


def _score(span: dict[str, Any]) -> float | None:
    for field in SCORE_FIELDS:
        if field in span and span[field] is not None:
            try:
                value = float(span[field])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def _normalize_span(span: dict[str, Any], text: str) -> dict[str, Any]:
    normalized = dict(span)
    normalized["start"] = int(normalized["start"])
    normalized["end"] = int(normalized["end"])
    normalized["label"] = str(normalized["label"])
    normalized["mention"] = text[normalized["start"] : normalized["end"]]
    score = _score(normalized)
    if score is not None:
        normalized["score"] = score
    return normalized


def _normalize_spans(spans: list[dict[str, Any]] | None, text: str) -> list[dict[str, Any]]:
    return [_normalize_span(span, text) for span in (spans or [])]


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(str(path))
    normalized = []
    for idx, row in enumerate(rows, start=1):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Prediction row {idx} has no string text field.")
        normalized.append(
            {
                "text": text,
                "pred_spans": row.get("pred_spans", row.get("entities", [])) or [],
                "gold_spans": row.get("gold_spans", row.get("spans")),
                "sample_id": row.get("sample_id"),
                "fold": row.get("fold"),
            }
        )
    return normalized


def _pair_rows(gold_path: Path | None, prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if gold_path is None:
        missing = [idx for idx, row in enumerate(prediction_rows, start=1) if row.get("gold_spans") is None]
        if missing:
            raise ValueError(
                "--gold-json is required because prediction rows do not contain gold_spans "
                f"(first missing row: {missing[0]})."
            )
        return [
            {
                "row_index_1based": idx,
                "sample_id": row.get("sample_id"),
                "fold": row.get("fold"),
                "text": row["text"],
                "gold_spans": row["gold_spans"] or [],
                "pred_spans": row["pred_spans"] or [],
            }
            for idx, row in enumerate(prediction_rows, start=1)
        ]

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


def _classify_prediction(pred_span: dict[str, Any], gold_spans: list[dict[str, Any]], gold_set: set[tuple[int, int, str]]) -> tuple[str, list[dict[str, Any]]]:
    if _span_key(pred_span) in gold_set:
        return "exact", []
    overlapping = [span for span in gold_spans if _overlaps(pred_span, span)]
    if any(span["label"] == pred_span["label"] for span in overlapping):
        return "boundary_mismatch", overlapping
    if overlapping:
        return "label_confusion", overlapping
    return "spurious", []


def build_prediction_calibration_rows(rows: list[dict[str, Any]], allowed_labels: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    calibration_rows: list[dict[str, Any]] = []
    gold_support = Counter()

    for row_index_0based, row in enumerate(rows):
        text = row["text"]
        gold_spans = _normalize_spans(row["gold_spans"], text)
        pred_spans = _normalize_spans(row["pred_spans"], text)
        if allowed_labels is not None:
            gold_spans = [span for span in gold_spans if span["label"] in allowed_labels]
            pred_spans = [span for span in pred_spans if span["label"] in allowed_labels]
        gold_set = {_span_key(span) for span in gold_spans}
        for span in gold_spans:
            gold_support[span["label"]] += 1

        for pred_span in pred_spans:
            score = pred_span.get("score")
            if score is None:
                continue
            outcome, overlapping = _classify_prediction(pred_span, gold_spans, gold_set)
            calibration_rows.append(
                {
                    "row_index_0based": row_index_0based,
                    "row_index_1based": row["row_index_1based"],
                    "sample_id": row.get("sample_id"),
                    "fold": row.get("fold"),
                    "label": pred_span["label"],
                    "mention": pred_span["mention"],
                    "start": pred_span["start"],
                    "end": pred_span["end"],
                    "score": float(score),
                    "target": 1 if outcome == "exact" else 0,
                    "outcome": outcome,
                    "overlap_labels": "|".join(sorted({span["label"] for span in overlapping})),
                    "overlap_mentions": " | ".join(span["mention"] for span in overlapping),
                    "text_preview": text[:180],
                }
            )
    return calibration_rows, dict(gold_support)


def _precision(correct: int, predicted: int) -> float | None:
    return (correct / predicted) if predicted else None


def _recall(correct: int, support: int) -> float | None:
    return (correct / support) if support else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def reliability_rows(calibration_rows: list[dict[str, Any]], labels: list[str], bins: int) -> list[dict[str, Any]]:
    if bins < 1:
        raise ValueError("bins must be >= 1.")
    scoped = [("ALL", calibration_rows)]
    scoped.extend((label, [row for row in calibration_rows if row["label"] == label]) for label in labels)
    output = []
    for label, rows in scoped:
        for index in range(bins):
            low = index / bins
            high = (index + 1) / bins
            if index == bins - 1:
                bucket = [row for row in rows if low <= row["score"] <= high]
            else:
                bucket = [row for row in rows if low <= row["score"] < high]
            count = len(bucket)
            correct = sum(int(row["target"]) for row in bucket)
            score_mean = (sum(row["score"] for row in bucket) / count) if count else None
            precision = _precision(correct, count)
            gap = abs(score_mean - precision) if score_mean is not None and precision is not None else None
            output.append(
                {
                    "label": label,
                    "bin_index": index,
                    "score_low": low,
                    "score_high": high,
                    "count": count,
                    "correct": correct,
                    "precision": precision,
                    "score_mean": score_mean,
                    "calibration_gap": gap,
                }
            )
    return output


def threshold_rows(
    calibration_rows: list[dict[str, Any]],
    labels: list[str],
    thresholds: list[float],
    gold_support: dict[str, int],
) -> list[dict[str, Any]]:
    scoped = [("ALL", calibration_rows, sum(gold_support.values()))]
    scoped.extend((label, [row for row in calibration_rows if row["label"] == label], gold_support.get(label, 0)) for label in labels)
    output = []
    for label, rows, support in scoped:
        for threshold in thresholds:
            selected = [row for row in rows if row["score"] >= threshold]
            predicted = len(selected)
            correct = sum(int(row["target"]) for row in selected)
            precision = _precision(correct, predicted)
            recall = _recall(correct, support)
            output.append(
                {
                    "label": label,
                    "threshold": threshold,
                    "predicted": predicted,
                    "correct": correct,
                    "false_positive": predicted - correct,
                    "gold_support": support,
                    "precision": precision,
                    "recall": recall,
                    "f1": _f1(precision, recall),
                }
            )
    return output


def outcome_summary_rows(calibration_rows: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    scoped = [("ALL", calibration_rows)]
    scoped.extend((label, [row for row in calibration_rows if row["label"] == label]) for label in labels)
    output = []
    for label, rows in scoped:
        counts = Counter(row["outcome"] for row in rows)
        total = len(rows)
        for outcome in ["exact", "boundary_mismatch", "label_confusion", "spurious"]:
            count = counts.get(outcome, 0)
            output.append(
                {
                    "label": label,
                    "outcome": outcome,
                    "count": count,
                    "share": (count / total) if total else None,
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_float(value) if isinstance(value, float) else value for key, value in row.items()})


def write_outputs(
    out_dir: Path,
    *,
    calibration_rows: list[dict[str, Any]],
    reliability: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_csv = out_dir / "prediction_score_rows.csv"
    reliability_csv = out_dir / "precision_by_label_and_score_bin.csv"
    threshold_csv = out_dir / "precision_at_threshold_by_label.csv"
    outcomes_csv = out_dir / "outcome_counts_by_label.csv"
    summary_json = out_dir / "calibration_summary.json"

    _write_csv(
        prediction_csv,
        calibration_rows,
        [
            "row_index_0based",
            "row_index_1based",
            "sample_id",
            "fold",
            "label",
            "mention",
            "start",
            "end",
            "score",
            "target",
            "outcome",
            "overlap_labels",
            "overlap_mentions",
            "text_preview",
        ],
    )
    _write_csv(
        reliability_csv,
        reliability,
        ["label", "bin_index", "score_low", "score_high", "count", "correct", "precision", "score_mean", "calibration_gap"],
    )
    _write_csv(
        threshold_csv,
        thresholds,
        ["label", "threshold", "predicted", "correct", "false_positive", "gold_support", "precision", "recall", "f1"],
    )
    _write_csv(outcomes_csv, outcomes, ["label", "outcome", "count", "share"])
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "prediction_score_rows": prediction_csv,
        "precision_by_label_and_score_bin": reliability_csv,
        "precision_at_threshold_by_label": threshold_csv,
        "outcome_counts_by_label": outcomes_csv,
        "summary": summary_json,
    }


def _best_thresholds(thresholds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in thresholds:
        label = row["label"]
        current = best.get(label)
        if current is None:
            best[label] = row
            continue
        row_f1 = row["f1"] if row["f1"] is not None else -1.0
        current_f1 = current["f1"] if current["f1"] is not None else -1.0
        if (row_f1, row["precision"] or -1.0, row["threshold"]) > (
            current_f1,
            current["precision"] or -1.0,
            current["threshold"],
        ):
            best[label] = row
    return best


def build_summary(
    *,
    rows: list[dict[str, Any]],
    labels: list[str],
    bins: int,
    thresholds: list[float],
    gold_support: dict[str, int],
    reliability: list[dict[str, Any]],
    threshold_report: list[dict[str, Any]],
) -> dict[str, Any]:
    outcome_counts = Counter(row["outcome"] for row in rows)
    label_counts = Counter(row["label"] for row in rows)
    best = _best_thresholds(threshold_report)
    high_conf_errors = [
        row
        for row in rows
        if row["target"] == 0 and row["score"] >= 0.9
    ]
    return {
        "prediction_rows": len(rows),
        "labels": labels,
        "bins": bins,
        "thresholds": thresholds,
        "gold_support": gold_support,
        "prediction_counts_by_label": dict(label_counts),
        "outcome_counts": dict(outcome_counts),
        "exact_precision_overall": _precision(outcome_counts.get("exact", 0), len(rows)),
        "best_threshold_by_label": best,
        "high_confidence_errors_ge_0_9": len(high_conf_errors),
        "high_confidence_errors_by_label": dict(Counter(row["label"] for row in high_conf_errors)),
        "top_high_confidence_error_mentions": [
            {"label": label, "outcome": outcome, "mention": mention, "count": count}
            for (label, outcome, mention), count in Counter(
                (row["label"], row["outcome"], row["mention"]) for row in high_conf_errors
            ).most_common(50)
        ],
        "artifacts": {
            "precision_by_label_and_score_bin": "precision_by_label_and_score_bin.csv",
            "precision_at_threshold_by_label": "precision_at_threshold_by_label.csv",
            "prediction_score_rows": "prediction_score_rows.csv",
            "outcome_counts_by_label": "outcome_counts_by_label.csv",
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    labels = _parse_labels(args.labels)
    threshold_values = _parse_thresholds(args.thresholds)
    pred_rows = _load_prediction_rows(Path(args.pred_jsonl))
    paired_rows = _pair_rows(Path(args.gold_json) if args.gold_json else None, pred_rows)
    calibration_rows, gold_support = build_prediction_calibration_rows(paired_rows, set(labels))
    reliability = reliability_rows(calibration_rows, labels, args.bins)
    threshold_report = threshold_rows(calibration_rows, labels, threshold_values, gold_support)
    outcomes = outcome_summary_rows(calibration_rows, labels)
    summary = build_summary(
        rows=calibration_rows,
        labels=labels,
        bins=args.bins,
        thresholds=threshold_values,
        gold_support=gold_support,
        reliability=reliability,
        threshold_report=threshold_report,
    )
    outputs = write_outputs(
        Path(args.output_dir),
        calibration_rows=calibration_rows,
        reliability=reliability,
        thresholds=threshold_report,
        outcomes=outcomes,
        summary=summary,
    )
    for name, path in outputs.items():
        LOGGER.info("Saved %s: %s", name, path)
    LOGGER.info("Prediction rows with usable scores: %s", len(calibration_rows))


if __name__ == "__main__":
    main()
