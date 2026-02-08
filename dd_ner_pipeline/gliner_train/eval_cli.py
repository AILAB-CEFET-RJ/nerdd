import argparse

from gliner_train.eval_config import EvaluationConfig


def parse_labels(raw_value):
    """Parse comma-separated labels."""
    labels = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not labels:
        raise ValueError("At least one label is required.")
    return labels


def parse_threshold_grid(raw_value):
    """Parse threshold list; supports explicit values or start:stop:step."""
    raw_value = raw_value.strip()
    if ":" in raw_value:
        start_text, stop_text, step_text = raw_value.split(":")
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        values = []
        current = start
        while current <= stop + 1e-12:
            values.append(round(current, 4))
            current += step
        return values
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def parse_args():
    """Build CLI args for evaluation pipeline."""
    defaults = EvaluationConfig()
    parser = argparse.ArgumentParser(description="GLiNER evaluation pipeline")
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--gt-jsonl", default=defaults.gt_jsonl)
    parser.add_argument("--pred-jsonl", default=defaults.pred_jsonl)
    parser.add_argument("--labels", default=",".join(defaults.labels))
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--prediction-threshold", type=float, default=defaults.prediction_threshold)
    parser.add_argument(
        "--threshold-grid",
        default=f"{defaults.threshold_grid[0]}:{defaults.threshold_grid[-1]}:{defaults.threshold_grid[1] - defaults.threshold_grid[0]}",
    )
    parser.add_argument("--calibrated-thresholds-json", default=defaults.calibrated_thresholds_json)
    parser.add_argument("--report-path", default=defaults.report_path)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    """Build EvaluationConfig from argparse namespace."""
    return EvaluationConfig(
        model_path=args.model_path,
        gt_jsonl=args.gt_jsonl,
        pred_jsonl=args.pred_jsonl,
        labels=parse_labels(args.labels),
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        prediction_threshold=args.prediction_threshold,
        threshold_grid=parse_threshold_grid(args.threshold_grid),
        calibrated_thresholds_json=args.calibrated_thresholds_json,
        report_path=args.report_path,
    )
