#!/usr/bin/env python3
"""Audit high-score tail behavior for raw and calibrated scores."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_csv_list(raw_value):
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def _load_calibration_helpers():
    from calibration.serialization import apply_calibrator_to_score, load_calibrator

    return apply_calibrator_to_score, load_calibrator


def load_rows(csv_path, *, score_col, label_col, class_col, allowed_labels, calibrator):
    apply_calibrator_to_score = None
    if calibrator is not None:
        apply_calibrator_to_score, _ = _load_calibration_helpers()

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for required in (score_col, label_col, class_col):
            if required not in fieldnames:
                raise ValueError(f"Calibration CSV missing column '{required}'.")

        for row in reader:
            raw_score = _safe_float(row.get(score_col))
            target = _safe_float(row.get(label_col))
            label = str(row.get(class_col, "")).strip()
            if raw_score is None or target is None:
                continue
            target_int = int(target)
            if target_int not in (0, 1):
                continue
            if allowed_labels and label not in allowed_labels:
                continue

            raw_score = min(max(raw_score, 1e-6), 1.0 - 1e-6)
            calibrated_score = (
                apply_calibrator_to_score(raw_score, label, calibrator) if calibrator is not None else None
            )
            rows.append(
                {
                    "label": label,
                    "target": target_int,
                    "score_raw": float(raw_score),
                    "score_calibrated": float(calibrated_score) if calibrated_score is not None else None,
                }
            )

    if not rows:
        raise ValueError("No valid rows found.")
    return rows


def _summarize_tail(rows, *, score_key, threshold):
    selected = [row for row in rows if row.get(score_key) is not None and row[score_key] >= threshold]
    if not selected:
        return {
            "threshold": threshold,
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "score_mean": None,
            "accuracy": None,
            "overconfidence_gap": None,
        }

    count = len(selected)
    positive_count = sum(1 for row in selected if row["target"] == 1)
    negative_count = count - positive_count
    score_mean = float(sum(row[score_key] for row in selected) / count)
    accuracy = float(positive_count / count)
    return {
        "threshold": threshold,
        "count": count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "score_mean": score_mean,
        "accuracy": accuracy,
        "overconfidence_gap": float(score_mean - accuracy),
    }


def summarize_rows(rows, *, thresholds):
    labels = sorted(Counter(row["label"] for row in rows).keys())
    has_calibrated = any(row["score_calibrated"] is not None for row in rows)

    payload = {
        "rows_total": len(rows),
        "rows_by_label": dict(sorted(Counter(row["label"] for row in rows).items())),
        "rows_by_target": dict(sorted(Counter(int(row["target"]) for row in rows).items())),
        "thresholds": thresholds,
        "overall": {
            "raw": [_summarize_tail(rows, score_key="score_raw", threshold=threshold) for threshold in thresholds],
        },
        "per_label": {},
    }

    if has_calibrated:
        payload["overall"]["calibrated"] = [
            _summarize_tail(rows, score_key="score_calibrated", threshold=threshold) for threshold in thresholds
        ]

    for label in labels:
        label_rows = [row for row in rows if row["label"] == label]
        payload["per_label"][label] = {
            "raw": [_summarize_tail(label_rows, score_key="score_raw", threshold=threshold) for threshold in thresholds],
        }
        if has_calibrated:
            payload["per_label"][label]["calibrated"] = [
                _summarize_tail(label_rows, score_key="score_calibrated", threshold=threshold) for threshold in thresholds
            ]

    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit high-score tail behavior for raw and calibrated scores from a calibration/evaluation CSV."
    )
    parser.add_argument("--evaluation-csv", required=True, help="Predictions CSV with Score, Validacao, and Label columns.")
    parser.add_argument("--output-json", required=True, help="Output summary JSON.")
    parser.add_argument("--calibrator-path", default="", help="Optional calibrator.json to apply.")
    parser.add_argument("--score-col", default="Score")
    parser.add_argument("--label-col", default="Validacao")
    parser.add_argument("--class-col", default="Label")
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument(
        "--thresholds",
        default="0.8,0.9,0.95",
        help="Comma-separated score thresholds for the high-score tail audit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    allowed_labels = set(_parse_csv_list(args.labels))
    thresholds = [float(piece) for piece in _parse_csv_list(args.thresholds)]

    calibrator = None
    if args.calibrator_path:
        _, load_calibrator = _load_calibration_helpers()
        calibrator = load_calibrator(args.calibrator_path)

    rows = load_rows(
        args.evaluation_csv,
        score_col=args.score_col,
        label_col=args.label_col,
        class_col=args.class_col,
        allowed_labels=allowed_labels,
        calibrator=calibrator,
    )
    summary = summarize_rows(rows, thresholds=thresholds)
    payload = {
        "evaluation_csv": str(Path(args.evaluation_csv).resolve()),
        "calibrator_path": str(Path(args.calibrator_path).resolve()) if args.calibrator_path else None,
        "summary": summary,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved high-score tail audit JSON: {output_path}")
    for bucket in summary["overall"]["raw"]:
        threshold = bucket["threshold"]
        raw_gap = bucket["overconfidence_gap"]
        calibrated_gap = None
        if "calibrated" in summary["overall"]:
            calibrated_gap = next(
                (item["overconfidence_gap"] for item in summary["overall"]["calibrated"] if item["threshold"] == threshold),
                None,
            )
        print(f"threshold={threshold}: raw_gap={raw_gap} calibrated_gap={calibrated_gap}")


if __name__ == "__main__":
    main()
