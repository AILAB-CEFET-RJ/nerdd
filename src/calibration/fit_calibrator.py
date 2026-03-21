import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import brier_score_loss

try:
    from calibration.methods.isotonic import fit_isotonic
    from calibration.methods.temperature import apply_temperature, fit_temperature
    from calibration.serialization import save_calibrator
except ImportError:  # pragma: no cover
    from methods.isotonic import fit_isotonic
    from methods.temperature import apply_temperature, fit_temperature
    from serialization import save_calibrator

LOGGER = logging.getLogger(__name__)


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_csv_list(raw_value):
    return [piece.strip() for piece in raw_value.split(",") if piece.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit and persist a score calibrator from a labeled calibration CSV.")
    parser.add_argument("--method", choices=["temperature", "temperature-per-class", "isotonic"], default="temperature")
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--output-calibrator", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--score-col", default="Score")
    parser.add_argument("--label-col", default="Validacao")
    parser.add_argument("--class-col", default="Label")
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument("--temperature-min", type=float, default=0.5)
    parser.add_argument("--temperature-max", type=float, default=5.0)
    parser.add_argument("--temperature-grid-size", type=int, default=181)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_calibration_csv(csv_path, score_col, label_col, class_col, allowed_labels):
    scores = []
    labels = []
    classes = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if score_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing score column '{score_col}'.")
        if label_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing label column '{label_col}'.")
        use_class = bool(class_col)
        if use_class and class_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing class column '{class_col}'.")

        for row in reader:
            try:
                score = float(row[score_col])
                label = int(float(row[label_col]))
            except (TypeError, ValueError):
                continue
            score = min(max(score, 1e-6), 1 - 1e-6)
            if label not in (0, 1):
                continue

            class_name = ""
            if use_class:
                class_name = str(row[class_col]).strip()
                if allowed_labels and class_name not in allowed_labels:
                    continue

            scores.append(score)
            labels.append(label)
            classes.append(class_name)

    if not scores:
        raise ValueError("Calibration CSV yielded zero valid rows.")

    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64), classes


def fit_global_temperature(scores, labels, args):
    temperature, fit_loss = fit_temperature(
        scores=scores,
        labels=labels,
        t_min=args.temperature_min,
        t_max=args.temperature_max,
        grid_size=args.temperature_grid_size,
    )
    calibrated = apply_temperature(scores, temperature)
    payload = {
        "method": "temperature",
        "parameters": {"temperature": float(temperature), "fit_loss": float(fit_loss)},
    }
    return calibrated, payload


def fit_per_class_temperature(scores, labels, classes, args):
    if not any(classes):
        raise ValueError("--class-col is required for temperature-per-class.")

    global_temperature = None
    if len(set(labels.tolist())) >= 2:
        global_temperature, _ = fit_temperature(
            scores=scores,
            labels=labels,
            t_min=args.temperature_min,
            t_max=args.temperature_max,
            grid_size=args.temperature_grid_size,
        )

    calibrated = np.copy(scores)
    class_to_indices = defaultdict(list)
    for idx, class_name in enumerate(classes):
        class_to_indices[class_name].append(idx)

    per_class = {}
    for class_name, indices in class_to_indices.items():
        idx_array = np.asarray(indices, dtype=np.int64)
        if len(set(labels[idx_array].tolist())) < 2:
            per_class[class_name] = {
                "temperature": float(global_temperature) if global_temperature is not None else None,
                "fit_loss": None,
                "support": int(len(idx_array)),
                "source": "global-fallback" if global_temperature is not None else "identity",
            }
            if global_temperature is not None:
                calibrated[idx_array] = apply_temperature(scores[idx_array], global_temperature)
            continue

        temperature, fit_loss = fit_temperature(
            scores=scores[idx_array],
            labels=labels[idx_array],
            t_min=args.temperature_min,
            t_max=args.temperature_max,
            grid_size=args.temperature_grid_size,
        )
        calibrated[idx_array] = apply_temperature(scores[idx_array], temperature)
        per_class[class_name] = {
            "temperature": float(temperature),
            "fit_loss": float(fit_loss),
            "support": int(len(idx_array)),
            "source": "per-class",
        }

    payload = {
        "method": "temperature-per-class",
        "parameters": {
            "per_class_temperature": per_class,
            "global_fallback_temperature": float(global_temperature) if global_temperature is not None else None,
        },
    }
    return calibrated, payload


def fit_global_isotonic(scores, labels):
    model = fit_isotonic(scores, labels)
    calibrated = model.predict(scores)
    payload = {
        "method": "isotonic",
        "parameters": {
            "isotonic_x_thresholds": [float(x) for x in model.X_thresholds_],
            "isotonic_y_thresholds": [float(y) for y in model.y_thresholds_],
        },
    }
    return calibrated, payload


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()

    labels_filter = set(_parse_csv_list(args.labels))
    scores, labels, classes = load_calibration_csv(
        csv_path=args.calibration_csv,
        score_col=args.score_col,
        label_col=args.label_col,
        class_col=args.class_col,
        allowed_labels=labels_filter,
    )

    if args.method == "temperature":
        calibrated, calibrator = fit_global_temperature(scores, labels, args)
    elif args.method == "temperature-per-class":
        calibrated, calibrator = fit_per_class_temperature(scores, labels, classes, args)
    else:
        calibrated, calibrator = fit_global_isotonic(scores, labels)

    brier_before = float(brier_score_loss(labels, scores))
    brier_after = float(brier_score_loss(labels, calibrated))
    finished_at = datetime.now(timezone.utc).isoformat()
    runtime_seconds = perf_counter() - timer

    calibrator.update(
        {
            "version": 1,
            "fitted_at_utc": finished_at,
            "score_field_in": args.score_col,
            "score_field_out": "score_calibrated",
            "class_column": args.class_col,
            "fitted_from": {
                "calibration_csv": str(Path(args.calibration_csv).resolve()),
            },
        }
    )

    stats_payload = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": {
            "method": args.method,
            "calibration_csv": args.calibration_csv,
            "score_col": args.score_col,
            "label_col": args.label_col,
            "class_col": args.class_col,
            "labels": sorted(labels_filter),
            "temperature_min": args.temperature_min,
            "temperature_max": args.temperature_max,
            "temperature_grid_size": args.temperature_grid_size,
        },
        "summary": {
            "rows": int(len(scores)),
            "positive_rate": float(np.mean(labels)),
            "brier_before": brier_before,
            "brier_after": brier_after,
            "rows_by_class": dict(sorted(Counter(classes).items())),
            "rows_by_validation": dict(sorted(Counter(int(x) for x in labels.tolist()).items())),
        },
        "calibrator": calibrator,
    }

    save_calibrator(args.output_calibrator, calibrator)
    stats_path = Path(args.stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("Fitted calibrator method: %s", args.method)
    LOGGER.info("Calibration rows: %s", len(scores))
    LOGGER.info("Brier before: %.6f | after: %.6f", brier_before, brier_after)
    LOGGER.info("Saved calibrator to: %s", args.output_calibrator)
    LOGGER.info("Saved calibration fit stats to: %s", args.stats_json)


if __name__ == "__main__":
    main()
