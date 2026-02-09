import argparse

from pseudolabelling.config import CorpusPredictConfig


def _parse_csv_list(raw_value):
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one value must be provided.")
    return values


def parse_args():
    defaults = CorpusPredictConfig()
    parser = argparse.ArgumentParser(description="Generate entity predictions over a large JSONL corpus")
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--input-jsonl", default=defaults.input_jsonl)
    parser.add_argument("--output-jsonl", default=defaults.output_jsonl)
    parser.add_argument("--stats-json", default=defaults.stats_json)
    parser.add_argument("--labels", default=",".join(defaults.labels))
    parser.add_argument("--text-fields", default=",".join(defaults.text_fields))
    parser.add_argument("--join-separator", default=defaults.join_separator)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--score-threshold", type=float, default=defaults.score_threshold)
    parser.add_argument("--keep-inference-text", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    return CorpusPredictConfig(
        model_path=args.model_path,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        stats_json=args.stats_json,
        labels=_parse_csv_list(args.labels),
        text_fields=_parse_csv_list(args.text_fields),
        join_separator=args.join_separator,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        score_threshold=args.score_threshold,
        keep_inference_text=args.keep_inference_text,
    )
