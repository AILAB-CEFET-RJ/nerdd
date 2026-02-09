import argparse

from config import CalibrationConfig


def _parse_csv_list(raw_value):
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one value must be provided.")
    return values


def parse_args():
    defaults = CalibrationConfig()
    parser = argparse.ArgumentParser(description="Calibrate entity confidence scores in a JSONL corpus")
    parser.add_argument("--method", choices=["temperature", "temperature-per-class", "isotonic"], default=defaults.method)
    parser.add_argument("--input-jsonl", default=defaults.input_jsonl)
    parser.add_argument("--output-jsonl", default=defaults.output_jsonl)
    parser.add_argument("--stats-json", default=defaults.stats_json)
    parser.add_argument("--score-field", default=defaults.score_field)
    parser.add_argument("--output-score-field", default=defaults.output_score_field)
    parser.add_argument("--preserve-original-score-field", default=defaults.preserve_original_score_field)
    parser.add_argument("--label-field", default=defaults.label_field)
    parser.add_argument("--labels", default=",".join(defaults.labels))
    parser.add_argument(
        "--label-source",
        choices=["score-threshold", "quantile-bands", "calibration-csv"],
        default=defaults.label_source,
    )
    parser.add_argument("--calibration-csv", default=defaults.calibration_csv)
    parser.add_argument("--csv-score-col", default=defaults.csv_score_col)
    parser.add_argument("--csv-label-col", default=defaults.csv_label_col)
    parser.add_argument("--csv-class-col", default=defaults.csv_class_col)
    parser.add_argument("--positive-threshold", type=float, default=defaults.positive_threshold)
    parser.add_argument("--lower-quantile", type=float, default=defaults.lower_quantile)
    parser.add_argument("--upper-quantile", type=float, default=defaults.upper_quantile)
    parser.add_argument("--temperature-min", type=float, default=defaults.temperature_min)
    parser.add_argument("--temperature-max", type=float, default=defaults.temperature_max)
    parser.add_argument("--temperature-grid-size", type=int, default=defaults.temperature_grid_size)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    return CalibrationConfig(
        method=args.method,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        stats_json=args.stats_json,
        score_field=args.score_field,
        output_score_field=args.output_score_field,
        preserve_original_score_field=args.preserve_original_score_field,
        label_field=args.label_field,
        labels=_parse_csv_list(args.labels),
        label_source=args.label_source,
        calibration_csv=args.calibration_csv,
        csv_score_col=args.csv_score_col,
        csv_label_col=args.csv_label_col,
        csv_class_col=args.csv_class_col,
        positive_threshold=args.positive_threshold,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        temperature_min=args.temperature_min,
        temperature_max=args.temperature_max,
        temperature_grid_size=args.temperature_grid_size,
    )
