"""Quick single-split GLiNER2 training for fast ablations."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from random import Random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from base_model_training.io_utils import save_jsonl
from base_model_training.paths import resolve_path, resolve_repo_artifact_path
from gliner2_inference import predict_entities_for_text
from gliner2_loader import load_gliner2_model
from pseudolabelling.evaluate_refit_pipeline import (
    compute_span_metrics,
    format_classification_report,
    load_gt_jsonl_strict,
)
from tools.build_calibration_dataset import read_json_or_jsonl

LOGGER = logging.getLogger(__name__)

LABEL_MAP = {
    "Person": "person",
    "Location": "location",
    "Organization": "organization",
}


@dataclass
class QuickTrainConfig:
    train_path: str = "../data/dd_corpus_small_train_reshuffled_dedup.json"
    test_path: str = "../data/dd_corpus_small_test_final_reshuffled_dedup.json"
    pseudolabel_path: str = ""
    train_mode: str = "supervised_only"
    model_base: str = "fastino/gliner2-base-v1"
    output_dir: str = "./artifacts/gliner2_training/quick_run"
    experiment_name: str = "gliner2_quick_lora"
    seed: int = 42
    keep_empty_examples: bool = False
    deduplicate_by_text: bool = True
    pseudolabel_sample_ratio: float = 1.0
    max_pseudolabel_records: int = 0
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    num_epochs: int = 8
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 16
    encoder_lr: float = 1.0e-5
    task_lr: float = 5.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    eval_steps: int = 500
    early_stopping_patience: int = 2
    fp16: bool = True
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    log_level: str = "INFO"


def parse_args():
    defaults = QuickTrainConfig()
    parser = argparse.ArgumentParser(description="Quick single-split GLiNER2 training")
    parser.add_argument("--train-path", default=defaults.train_path)
    parser.add_argument("--test-path", default=defaults.test_path)
    parser.add_argument("--pseudolabel-path", default=defaults.pseudolabel_path)
    parser.add_argument(
        "--train-mode",
        choices=["supervised_only", "supervised_plus_pseudolabels", "pseudolabel_only"],
        default=defaults.train_mode,
    )
    parser.add_argument("--model-base", default=defaults.model_base)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--experiment-name", default=defaults.experiment_name)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--keep-empty-examples", action="store_true", default=defaults.keep_empty_examples)
    parser.add_argument("--disable-deduplicate-by-text", action="store_true")
    parser.add_argument("--pseudolabel-sample-ratio", type=float, default=defaults.pseudolabel_sample_ratio)
    parser.add_argument("--max-pseudolabel-records", type=int, default=defaults.max_pseudolabel_records)
    parser.add_argument("--train-ratio", type=float, default=defaults.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    parser.add_argument("--grad-accum", type=int, default=defaults.grad_accum)
    parser.add_argument("--encoder-lr", type=float, default=defaults.encoder_lr)
    parser.add_argument("--task-lr", type=float, default=defaults.task_lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--logging-steps", type=int, default=defaults.logging_steps)
    parser.add_argument("--eval-steps", type=int, default=defaults.eval_steps)
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=defaults.fp16)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--use-lora", action="store_true", default=defaults.use_lora)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora-alpha", type=float, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--log-level", default=defaults.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    return QuickTrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        pseudolabel_path=args.pseudolabel_path,
        train_mode=args.train_mode,
        model_base=args.model_base,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        keep_empty_examples=args.keep_empty_examples,
        deduplicate_by_text=(not args.disable_deduplicate_by_text),
        pseudolabel_sample_ratio=args.pseudolabel_sample_ratio,
        max_pseudolabel_records=args.max_pseudolabel_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping_patience,
        fp16=args.fp16,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        log_level=args.log_level,
    )


def _normalize_label(label):
    return LABEL_MAP.get(str(label), str(label).lower())


def _normalize_spans_row(row):
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    spans = []
    for span in row.get("spans", []) or []:
        start = int(span["start"])
        end = int(span["end"])
        if end <= start or end > len(text):
            continue
        spans.append({"start": start, "end": end, "label": str(span["label"])})
    return {"text": text, "spans": spans}


def _convert_entity_rows_to_spans(rows):
    converted = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        entities = row.get("entities")
        if not isinstance(entities, list):
            continue
        spans = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            start = int(entity["start"])
            end = int(entity["end"])
            label = str(entity["label"])
            if end <= start or end > len(text):
                continue
            if text[start:end] != str(entity.get("text", "")):
                continue
            spans.append({"start": start, "end": end, "label": label})
        converted.append({"text": text, "spans": spans})
    return converted


def _sample_rows(rows, *, seed, sample_ratio, max_records):
    sampled = list(rows)
    if sample_ratio <= 0:
        return []
    if sample_ratio < 1.0:
        rng = Random(seed)
        keep = max(1, int(round(len(sampled) * sample_ratio)))
        if keep < len(sampled):
            sampled = rng.sample(sampled, keep)
    if max_records and len(sampled) > max_records:
        rng = Random(seed)
        sampled = rng.sample(sampled, max_records)
    return sampled


def _merge_training_rows(supervised_rows, pseudolabel_rows, *, train_mode, deduplicate_by_text):
    if train_mode == "supervised_only":
        merged = list(supervised_rows)
    elif train_mode == "pseudolabel_only":
        merged = list(pseudolabel_rows)
    else:
        merged = list(supervised_rows) + list(pseudolabel_rows)

    if not deduplicate_by_text:
        return merged

    deduped = []
    seen = {}
    if train_mode == "supervised_plus_pseudolabels":
        for row in supervised_rows:
            key = row["text"]
            if key in seen:
                continue
            seen[key] = "supervised"
            deduped.append(row)
        for row in pseudolabel_rows:
            key = row["text"]
            if key in seen:
                continue
            seen[key] = "pseudolabel"
            deduped.append(row)
        return deduped

    for row in merged:
        key = row["text"]
        if key in seen:
            continue
        seen[key] = True
        deduped.append(row)
    return deduped


def _convert_rows_to_gliner2_jsonl(rows, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            entities = {}
            for span in row.get("spans", []) or []:
                start = int(span["start"])
                end = int(span["end"])
                mention = text[start:end].strip()
                if not mention:
                    continue
                label = _normalize_label(span["label"])
                entities.setdefault(label, [])
                if mention not in entities[label]:
                    entities[label].append(mention)
            handle.write(json.dumps({"input": text, "output": {"entities": entities}}, ensure_ascii=False) + "\n")


def _write_report(report_dir: Path, metrics, predictions):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "classification_report.txt").write_text(format_classification_report(metrics), encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_jsonl(str(report_dir / "predictions.jsonl"), predictions)


def _predict_rows(model, rows, entity_types):
    pred_rows = []
    for row in rows:
        entities = predict_entities_for_text(model, row["text"], entity_types)
        pred_rows.append({"text": row["text"], "entities": entities})
    return pred_rows


def run_quick_experiment(config: QuickTrainConfig, script_path: str):
    from gliner2 import GLiNER2
    from gliner2.training.data import TrainingDataset
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    started_at = datetime.now(timezone.utc)
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    output_dir = resolve_repo_artifact_path(__file__, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_path(script_dir, config.train_path)
    test_path = resolve_path(script_dir, config.test_path)
    pseudolabel_path = resolve_path(script_dir, config.pseudolabel_path) if config.pseudolabel_path else None
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    if config.train_mode != "supervised_only":
        if pseudolabel_path is None or not pseudolabel_path.exists():
            raise FileNotFoundError("Pseudolabel dataset is required for train_mode != supervised_only.")

    supervised_rows_all = [_normalize_spans_row(row) for row in read_json_or_jsonl(str(train_path))]
    supervised_rows_all = [row for row in supervised_rows_all if row is not None]
    test_rows = load_gt_jsonl_strict(str(test_path))
    pseudolabel_rows_all = []
    if pseudolabel_path is not None and pseudolabel_path.exists():
        pseudolabel_rows_all = _convert_entity_rows_to_spans(read_json_or_jsonl(str(pseudolabel_path)))
        pseudolabel_rows_all = _sample_rows(
            pseudolabel_rows_all,
            seed=config.seed,
            sample_ratio=config.pseudolabel_sample_ratio,
            max_records=config.max_pseudolabel_records,
        )

    with tempfile.TemporaryDirectory(prefix="gliner2_quick_") as temp_dir:
        model = GLiNER2.from_pretrained(config.model_base)
        raw_rows = _merge_training_rows(
            supervised_rows_all,
            pseudolabel_rows_all,
            train_mode=config.train_mode,
            deduplicate_by_text=config.deduplicate_by_text,
        )
        if not config.keep_empty_examples:
            raw_rows = [row for row in raw_rows if row.get("spans")]
        if not raw_rows:
            raise ValueError("Training dataset is empty after preprocessing.")

        entity_labels = sorted({str(span["label"]) for row in raw_rows for span in (row.get("spans") or [])})
        entity_types = [_normalize_label(label) for label in entity_labels]

        temp_train = Path(temp_dir) / "train.jsonl"
        _convert_rows_to_gliner2_jsonl(raw_rows, temp_train)

        dataset = TrainingDataset.load(temp_train, shuffle=True, seed=config.seed)
        invalid_indices = []
        invalid_errors = []
        if hasattr(dataset, "validate"):
            report = dataset.validate(raise_on_error=False)
            invalid_indices = sorted(set(report.get("invalid_indices", []) or []))
            invalid_errors = report.get("errors", []) or []
            if invalid_indices:
                LOGGER.warning(
                    "GLiNER2 validation marked %s/%s training examples as invalid. First errors: %s",
                    len(invalid_indices),
                    len(dataset.examples),
                    invalid_errors[:3],
                )
                invalid_set = set(invalid_indices)
                invalid_examples_path = output_dir / "invalid_train_examples.jsonl"
                with temp_train.open("r", encoding="utf-8") as handle:
                    invalid_lines = [line.rstrip("\n") for idx, line in enumerate(handle) if idx in invalid_set]
                invalid_examples_path.write_text("\n".join(invalid_lines) + ("\n" if invalid_lines else ""), encoding="utf-8")
                dataset = TrainingDataset([example for idx, example in enumerate(dataset.examples) if idx not in invalid_set])
                if not dataset.examples:
                    raise ValueError("All training examples were rejected by GLiNER2 validation.")

        train_data, val_data, _ = dataset.split(
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=1.0 - config.train_ratio - config.val_ratio,
            shuffle=True,
            seed=config.seed,
        )
        if len(train_data.examples) == 0:
            raise ValueError("Training split is empty.")
        if len(val_data.examples) == 0:
            raise ValueError("Validation split is empty.")

        train_config = TrainingConfig(
            output_dir=str(output_dir),
            experiment_name=config.experiment_name,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=config.grad_accum,
            encoder_lr=config.encoder_lr,
            task_lr=config.task_lr,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            eval_strategy="epoch",
            eval_steps=config.eval_steps,
            save_best=True,
            early_stopping=True,
            early_stopping_patience=config.early_stopping_patience,
            validate_data=True,
            fp16=config.fp16,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=["encoder"],
            save_adapter_only=bool(config.use_lora),
            seed=config.seed,
        )

        trainer = GLiNER2Trainer(model=model, config=train_config)
        results = trainer.train(train_data=train_data, eval_data=val_data)

    if config.use_lora:
        eval_model = load_gliner2_model(config.model_base, adapter_dir=str(output_dir / "final"), logger=LOGGER, context="quick-eval")
        model_artifact = output_dir / "final"
    else:
        eval_model = load_gliner2_model(str(output_dir / "best"), logger=LOGGER, context="quick-eval")
        model_artifact = output_dir / "best"

    test_predictions = _predict_rows(eval_model, test_rows, entity_types)
    test_metrics = compute_span_metrics(
        gold_spans_by_row=[row["spans"] for row in test_rows],
        pred_spans_by_row=[row["entities"] for row in test_predictions],
        labels=entity_labels,
    )
    _write_report(output_dir / "eval_test", test_metrics, test_predictions)

    finished_at = datetime.now(timezone.utc)
    runtime_seconds = perf_counter() - timer
    training_losses = results.get("training_losses", []) or []
    validation_losses = results.get("validation_losses", []) or []
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "runtime_seconds": runtime_seconds,
        "config": asdict(config),
        "dataset": {
            "raw_supervised_rows": len(supervised_rows_all),
            "raw_pseudolabel_rows": len(pseudolabel_rows_all),
            "train_mode": config.train_mode,
            "raw_train_rows": len(raw_rows),
            "filtered_train_rows": len(raw_rows),
            "invalid_train_examples": len(invalid_indices),
            "train_rows": len(train_data.examples),
            "val_rows": len(val_data.examples),
            "entity_labels": entity_labels,
        },
        "training": {
            "best_validation_metric": results.get("best_metric"),
            "mean_train_loss": mean(training_losses) if training_losses else None,
            "mean_val_loss": mean(validation_losses) if validation_losses else None,
        },
        "test_metrics": test_metrics,
        "artifacts": {
            "model_dir": str(model_artifact),
            "eval_dir": str(output_dir / "eval_test"),
        },
    }
    (output_dir / "quick_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("GLiNER2 quick training completed. Summary: %s", output_dir / "quick_summary.json")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = build_config(args)
    run_quick_experiment(config, script_path=__file__)


if __name__ == "__main__":
    main()
