import argparse
from dataclasses import dataclass, field


@dataclass
class RefitConfig:
    input_path: str = "./pseudolabel_split"
    output_model_dir: str = "./refit_model"
    stats_json: str = "./refit_stats.json"
    train_manifest_jsonl: str = "./train_manifest.jsonl"
    val_manifest_jsonl: str = "./val_manifest.jsonl"
    base_model: str = ""
    epochs: int = 10
    patience: int = 3
    batch_size: int = 8
    lr: float = 3e-5
    weight_decay: float = 0.01
    val_jsonl: str = ""
    val_ratio: float = 0.1
    seed: int = 42
    allowed_labels: list[str] = field(default_factory=lambda: ["Person", "Location", "Organization"])
    num_workers: int = 2


def _parse_csv_list(raw_value):
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one value must be provided.")
    return values


def parse_args():
    defaults = RefitConfig()
    parser = argparse.ArgumentParser(description="Refit GLiNER model on kept pseudolabel records")
    parser.add_argument("--input-path", default=defaults.input_path, help="JSONL file or split directory with kept.jsonl")
    parser.add_argument("--output-model-dir", default=defaults.output_model_dir)
    parser.add_argument("--stats-json", default=defaults.stats_json)
    parser.add_argument("--train-manifest-jsonl", default=defaults.train_manifest_jsonl)
    parser.add_argument("--val-manifest-jsonl", default=defaults.val_manifest_jsonl)
    parser.add_argument("--base-model", default=defaults.base_model)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--val-jsonl", default=defaults.val_jsonl, help="Optional external validation JSONL")
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--allowed-labels", default=",".join(defaults.allowed_labels))
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    return RefitConfig(
        input_path=args.input_path,
        output_model_dir=args.output_model_dir,
        stats_json=args.stats_json,
        train_manifest_jsonl=args.train_manifest_jsonl,
        val_manifest_jsonl=args.val_manifest_jsonl,
        base_model=args.base_model,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_jsonl=args.val_jsonl,
        val_ratio=args.val_ratio,
        seed=args.seed,
        allowed_labels=_parse_csv_list(args.allowed_labels),
        num_workers=args.num_workers,
    )
