#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

from gliner2 import GLiNER2
from gliner2.training.data import TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


LABEL_MAP = {
    "Person": "person",
    "Location": "location",
    "Organization": "organization",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GLiNER2 LoRA adapter on the NERDD dataset."
    )
    parser.add_argument(
        "--data",
        default="nerdd/dd_corpus_small_train.jsonl",
        help="Path to the training dataset in GLiNER2 JSONL format.",
    )
    parser.add_argument(
        "--model",
        default="fastino/gliner2-base-v1",
        help="Base GLiNER2 model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/nerdd_lora",
        help="Directory for checkpoints and final adapter artifacts.",
    )
    parser.add_argument(
        "--experiment-name",
        default="nerdd_lora",
        help="Experiment name used in logs and saved config.",
    )
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--task-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--keep-empty-examples",
        action="store_true",
        help="Keep examples with no annotated entities.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision. Recommended when training on GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    temp_jsonl_path: Path | None = None
    load_path = data_path

    try:
        if data_path.suffix == ".json":
            records = json.loads(data_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON array in {data_path}")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", encoding="utf-8", delete=False
            ) as temp_file:
                for record in records:
                    if "input" in record and "output" in record:
                        converted = record
                    else:
                        text = record["text"]
                        entities: dict[str, list[str]] = {}

                        for span in record.get("spans", []):
                            start = span["start"]
                            end = span["end"]
                            label = LABEL_MAP.get(span["label"], span["label"].lower())
                            mention = text[start:end].strip()
                            if mention:
                                entities.setdefault(label, [])
                                if mention not in entities[label]:
                                    entities[label].append(mention)

                        converted = {
                            "input": text,
                            "output": {"entities": entities},
                        }

                    temp_file.write(json.dumps(converted, ensure_ascii=False) + "\n")
                temp_jsonl_path = Path(temp_file.name)

            load_path = temp_jsonl_path

        dataset = TrainingDataset.load(load_path, shuffle=True, seed=args.seed)

        if not args.keep_empty_examples:
            dataset = dataset.filter(lambda ex: bool(ex.entities))

        train_data, val_data, _ = dataset.split(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
            shuffle=True,
            seed=args.seed,
        )

        if len(train_data.examples) == 0:
            raise ValueError("Training split is empty.")
        if len(val_data.examples) == 0:
            raise ValueError("Validation split is empty.")

        model = GLiNER2.from_pretrained(args.model)

        config = TrainingConfig(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            task_lr=args.task_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            eval_strategy="epoch",
            eval_steps=args.eval_steps,
            save_best=True,
            early_stopping=True,
            early_stopping_patience=2,
            validate_data=True,
            fp16=args.fp16,
            use_lora=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=["encoder"],
            save_adapter_only=True,
            seed=args.seed,
        )

        trainer = GLiNER2Trainer(model=model, config=config)
        results = trainer.train(train_data=train_data, eval_data=val_data)

        print("Training completed.")
        print(f"Train examples: {len(train_data.examples)}")
        print(f"Validation examples: {len(val_data.examples)}")
        print(f"Best metric: {results.get('best_metric')}")
        print(f"Output dir: {args.output_dir}")
    finally:
        if temp_jsonl_path and temp_jsonl_path.exists():
            temp_jsonl_path.unlink()


if __name__ == "__main__":
    main()
