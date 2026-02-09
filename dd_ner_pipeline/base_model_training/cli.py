import argparse

from gliner_train.train_config import TrainConfig


def parse_thresholds(raw_value):
    """Parse comma-separated thresholds into a float list."""
    values = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if piece:
            values.append(float(piece))
    if not values:
        raise ValueError("At least one threshold must be provided.")
    return values


def parse_float_list(raw_value):
    """Parse comma-separated floats into a list."""
    values = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if piece:
            values.append(float(piece))
    if not values:
        raise ValueError("At least one float value is required.")
    return values


def parse_args():
    """Build CLI for experiment configuration."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="GLiNER nested KFold training")
    parser.add_argument("--train-path", default=defaults.train_path)
    parser.add_argument("--model-base", default=defaults.model_base)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--n-splits", type=int, default=defaults.n_splits)
    parser.add_argument("--n-inner-splits", type=int, default=defaults.n_inner_splits)
    parser.add_argument("--num-trials", type=int, default=defaults.num_trials)
    parser.add_argument("--search-mode", choices=["grid", "random"], default=defaults.search_mode)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--overlap", type=int, default=defaults.overlap)
    parser.add_argument("--thresholds", default=",".join(str(value) for value in defaults.thresholds))
    parser.add_argument("--lr-values", default=",".join(str(value) for value in defaults.lr_values))
    parser.add_argument(
        "--weight-decay-values",
        default=",".join(str(value) for value in defaults.weight_decay_values),
    )
    parser.add_argument("--refit-val-size", type=float, default=defaults.refit_val_size)
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--early-stopping-threshold", type=float, default=defaults.early_stopping_threshold)
    parser.add_argument("--results-file", default=defaults.results_file)
    parser.add_argument("--results-json-file", default=defaults.results_json_file)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    """Build TrainConfig from argparse namespace."""
    return TrainConfig(
        seed=args.seed,
        train_path=args.train_path,
        model_base=args.model_base,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        n_splits=args.n_splits,
        n_inner_splits=args.n_inner_splits,
        num_trials=args.num_trials,
        search_mode=args.search_mode,
        max_length=args.max_length,
        overlap=args.overlap,
        thresholds=parse_thresholds(args.thresholds),
        lr_values=parse_float_list(args.lr_values),
        weight_decay_values=parse_float_list(args.weight_decay_values),
        refit_val_size=args.refit_val_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        results_file=args.results_file,
        results_json_file=args.results_json_file,
        output_dir=args.output_dir,
    )
