import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from calibration.evaluate_calibration import _brier, _nll, ece_mce_from_bins, reliability_bins
    from calibration.serialization import apply_calibrator_to_score, load_calibrator
except ImportError:  # pragma: no cover
    from evaluate_calibration import _brier, _nll, ece_mce_from_bins, reliability_bins
    from serialization import apply_calibrator_to_score, load_calibrator


def _parse_csv_list(raw_value):
    return [piece.strip() for piece in raw_value.split(",") if piece.strip()]


def load_rows(csv_path, score_col, label_col, class_col, allowed_labels):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if score_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing score column '{score_col}'.")
        if label_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing label column '{label_col}'.")
        if class_col and class_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing class column '{class_col}'.")

        for row in reader:
            try:
                score = float(row[score_col])
                target = int(float(row[label_col]))
            except (TypeError, ValueError):
                continue
            if target not in (0, 1):
                continue
            label = str(row[class_col]).strip() if class_col else ""
            if allowed_labels and label and label not in allowed_labels:
                continue
            rows.append(
                {
                    "score": float(min(max(score, 1e-6), 1.0 - 1e-6)),
                    "target": target,
                    "label": label,
                }
            )
    if not rows:
        raise ValueError("No valid rows were loaded from the evaluation CSV.")
    return rows


def summarize(scores, targets, bins):
    bin_df = reliability_bins(scores, targets, bins=bins)
    ece, mce = ece_mce_from_bins(bin_df)
    return {
        "rows": int(len(scores)),
        "positive_rate": float(np.mean(targets)),
        "brier": float(_brier(targets, scores)),
        "nll": float(_nll(targets, scores)),
        "ece": float(ece),
        "mce": float(mce),
    }


def _bin_rows(scores, targets, bins):
    return reliability_bins(scores, targets, bins=bins).to_dict(orient="records")


def evaluate(rows, calibrator, bins):
    raw_scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target"] for row in rows], dtype=np.float64)
    calibrated_scores = np.asarray(
        [apply_calibrator_to_score(row["score"], row["label"], calibrator) for row in rows],
        dtype=np.float64,
    )

    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    per_label = {}
    for label, label_rows in sorted(by_label.items()):
        label_raw = np.asarray([row["score"] for row in label_rows], dtype=np.float64)
        label_targets = np.asarray([row["target"] for row in label_rows], dtype=np.float64)
        label_cal = np.asarray(
            [apply_calibrator_to_score(row["score"], row["label"], calibrator) for row in label_rows],
            dtype=np.float64,
        )
        per_label[label] = {
            "raw": summarize(label_raw, label_targets, bins=bins),
            "calibrated": summarize(label_cal, label_targets, bins=bins),
        }

    return {
        "overall": {
            "raw": {
                **summarize(raw_scores, targets, bins=bins),
                "bins": _bin_rows(raw_scores, targets, bins=bins),
            },
            "calibrated": {
                **summarize(calibrated_scores, targets, bins=bins),
                "bins": _bin_rows(calibrated_scores, targets, bins=bins),
            },
        },
        "per_label": per_label,
        "row_counts_by_label": dict(sorted(Counter(row["label"] for row in rows).items())),
        "row_counts_by_target": dict(sorted(Counter(int(row["target"]) for row in rows).items())),
    }


def _write_bin_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bin", "bin_lower", "bin_upper", "count", "conf_mean", "acc", "gap"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_reliability_plot(path, raw_rows, calibrated_rows):
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6), dpi=140)
    diagonal = np.linspace(0.0, 1.0, 201)
    plt.plot(diagonal, diagonal, linestyle="--", linewidth=1.0, label="Perfect calibration")

    for rows, label in ((raw_rows, "RAW"), (calibrated_rows, "Calibrated")):
        xs = [row["conf_mean"] for row in rows if row["count"] > 0 and row["conf_mean"] is not None]
        ys = [row["acc"] for row in rows if row["count"] > 0 and row["acc"] is not None]
        if xs and ys:
            plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Mean confidence per bin")
    plt.ylabel("Empirical accuracy per bin")
    plt.title("Reliability Curve")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved calibrator artifact on a held-out calibration CSV.")
    parser.add_argument("--evaluation-csv", required=True)
    parser.add_argument("--calibrator-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--score-col", default="Score")
    parser.add_argument("--label-col", default="Validacao")
    parser.add_argument("--class-col", default="Label")
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--reliability-plot", default="")
    parser.add_argument("--raw-bins-csv", default="")
    parser.add_argument("--calibrated-bins-csv", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    allowed_labels = set(_parse_csv_list(args.labels))
    rows = load_rows(
        csv_path=args.evaluation_csv,
        score_col=args.score_col,
        label_col=args.label_col,
        class_col=args.class_col,
        allowed_labels=allowed_labels,
    )
    calibrator = load_calibrator(args.calibrator_path)
    payload = {
        "evaluation_csv": str(Path(args.evaluation_csv).resolve()),
        "calibrator_path": str(Path(args.calibrator_path).resolve()),
        "bins": args.bins,
        "results": evaluate(rows, calibrator, bins=args.bins),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reliability_plot = Path(args.reliability_plot) if args.reliability_plot else output_path.with_name("reliability_curve.png")
    raw_bins_csv = Path(args.raw_bins_csv) if args.raw_bins_csv else output_path.with_name("reliability_raw.csv")
    calibrated_bins_csv = (
        Path(args.calibrated_bins_csv) if args.calibrated_bins_csv else output_path.with_name("reliability_calibrated.csv")
    )
    _write_bin_csv(raw_bins_csv, payload["results"]["overall"]["raw"]["bins"])
    _write_bin_csv(calibrated_bins_csv, payload["results"]["overall"]["calibrated"]["bins"])
    _save_reliability_plot(
        reliability_plot,
        payload["results"]["overall"]["raw"]["bins"],
        payload["results"]["overall"]["calibrated"]["bins"],
    )
    payload["artifacts"] = {
        "reliability_plot": str(reliability_plot.resolve()),
        "raw_bins_csv": str(raw_bins_csv.resolve()),
        "calibrated_bins_csv": str(calibrated_bins_csv.resolve()),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    overall = payload["results"]["overall"]
    print(f"Saved calibrator evaluation to: {output_path}")
    print(f"Saved reliability plot to: {reliability_plot}")
    print(f"Overall Brier raw: {overall['raw']['brier']:.6f}")
    print(f"Overall Brier calibrated: {overall['calibrated']['brier']:.6f}")
    print(f"Overall ECE raw: {overall['raw']['ece']:.6f}")
    print(f"Overall ECE calibrated: {overall['calibrated']['ece']:.6f}")


if __name__ == "__main__":
    main()
