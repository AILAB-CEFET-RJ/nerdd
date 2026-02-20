import argparse


def _parse_csv_list(raw_value):
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one value must be provided.")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate refit GLiNER model on labeled JSONL (span-exact metrics)")
    parser.add_argument("--model-path", required=True, help="Trained model path/name")
    parser.add_argument("--gt-jsonl", required=True, help="Ground-truth JSONL with text and spans")
    parser.add_argument("--out-dir", required=True, help="Output directory for evaluation artifacts")
    parser.add_argument("--labels", default="Person,Location,Organization", help="Comma-separated label list")
    parser.add_argument("--prediction-threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--match-mode", choices=["exact"], default="exact")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    return {
        "model_path": args.model_path,
        "gt_jsonl": args.gt_jsonl,
        "out_dir": args.out_dir,
        "labels": _parse_csv_list(args.labels),
        "prediction_threshold": args.prediction_threshold,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "match_mode": args.match_mode,
    }
