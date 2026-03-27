#!/usr/bin/env python3

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


def _quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _summarize_scores(values, *, high_score_threshold):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
            "share_ge_threshold": None,
            "threshold": high_score_threshold,
        }
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "mean": float(sum(ordered) / count),
        "min": float(ordered[0]),
        "p25": _quantile(ordered, 0.25),
        "p50": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "max": float(ordered[-1]),
        "share_ge_threshold": float(sum(1 for item in ordered if item >= high_score_threshold) / count),
        "threshold": high_score_threshold,
    }


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
        raise ValueError("No valid calibration rows found.")
    return rows


def summarize_rows(rows, *, high_score_threshold):
    overall = defaultdict(list)
    per_label = defaultdict(lambda: defaultdict(list))

    for row in rows:
        bucket_name = "positives" if row["target"] == 1 else "negatives"
        overall[f"raw_{bucket_name}"].append(row["score_raw"])
        if row["score_calibrated"] is not None:
            overall[f"calibrated_{bucket_name}"].append(row["score_calibrated"])
        per_label[row["label"]][f"raw_{bucket_name}"].append(row["score_raw"])
        if row["score_calibrated"] is not None:
            per_label[row["label"]][f"calibrated_{bucket_name}"].append(row["score_calibrated"])

    payload = {
        "rows_total": len(rows),
        "rows_by_label": dict(sorted(Counter(row["label"] for row in rows).items())),
        "rows_by_target": dict(sorted(Counter(row["target"] for row in rows).items())),
        "overall": {
            "raw": {
                "positives": _summarize_scores(overall["raw_positives"], high_score_threshold=high_score_threshold),
                "negatives": _summarize_scores(overall["raw_negatives"], high_score_threshold=high_score_threshold),
            }
        },
        "per_label": {},
    }

    has_calibrated = any(row["score_calibrated"] is not None for row in rows)
    if has_calibrated:
        payload["overall"]["calibrated"] = {
            "positives": _summarize_scores(overall["calibrated_positives"], high_score_threshold=high_score_threshold),
            "negatives": _summarize_scores(overall["calibrated_negatives"], high_score_threshold=high_score_threshold),
        }

    for label in sorted(per_label):
        payload["per_label"][label] = {
            "raw": {
                "positives": _summarize_scores(per_label[label]["raw_positives"], high_score_threshold=high_score_threshold),
                "negatives": _summarize_scores(per_label[label]["raw_negatives"], high_score_threshold=high_score_threshold),
            }
        }
        if has_calibrated:
            payload["per_label"][label]["calibrated"] = {
                "positives": _summarize_scores(
                    per_label[label]["calibrated_positives"], high_score_threshold=high_score_threshold
                ),
                "negatives": _summarize_scores(
                    per_label[label]["calibrated_negatives"], high_score_threshold=high_score_threshold
                ),
            }

    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit calibration CSV by label, separating positives and negatives."
    )
    parser.add_argument("--calibration-csv", required=True, help="Calibration predictions CSV.")
    parser.add_argument("--output-json", required=True, help="Output summary JSON.")
    parser.add_argument("--calibrator-path", default="", help="Optional calibrator.json to apply on top of raw scores.")
    parser.add_argument("--score-col", default="Score")
    parser.add_argument("--label-col", default="Validacao")
    parser.add_argument("--class-col", default="Label")
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument(
        "--high-score-threshold",
        type=float,
        default=0.8,
        help="Threshold used to measure high-confidence share among positives/negatives.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    allowed_labels = set(_parse_csv_list(args.labels))
    calibrator = None
    if args.calibrator_path:
        _, load_calibrator = _load_calibration_helpers()
        calibrator = load_calibrator(args.calibrator_path)
    rows = load_rows(
        args.calibration_csv,
        score_col=args.score_col,
        label_col=args.label_col,
        class_col=args.class_col,
        allowed_labels=allowed_labels,
        calibrator=calibrator,
    )
    summary = summarize_rows(rows, high_score_threshold=args.high_score_threshold)
    payload = {
        "calibration_csv": str(Path(args.calibration_csv).resolve()),
        "calibrator_path": str(Path(args.calibrator_path).resolve()) if args.calibrator_path else None,
        "high_score_threshold": args.high_score_threshold,
        "summary": summary,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved calibration audit JSON: {output_path}")
    for label, label_summary in summary["per_label"].items():
        raw_neg = label_summary["raw"]["negatives"]["share_ge_threshold"]
        cal_neg = None
        if "calibrated" in label_summary:
            cal_neg = label_summary["calibrated"]["negatives"]["share_ge_threshold"]
        print(f"{label}: raw_neg_ge={raw_neg} calibrated_neg_ge={cal_neg}")


if __name__ == "__main__":
    main()
