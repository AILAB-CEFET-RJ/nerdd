"""Quick single-split GLiNER training for fast ablations."""

from __future__ import annotations

import argparse
import json
import logging
from random import Random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from base_model_training.cli import parse_thresholds
from base_model_training.cv import (
    _build_refit_split,
    _build_seen_entity_keys,
    _compute_seen_unseen_breakdown,
    _evaluate_thresholds,
    _load_model,
    _prepare_char_offsets,
    _run_single_training,
    clear_cuda_cache,
    materialize_model_base,
    set_seed,
)
from base_model_training.data import load_dataset, split_long_sentences
from base_model_training.evaluate import predict_entities_for_text
from base_model_training.io_utils import save_jsonl
from base_model_training.paths import resolve_path
from pseudolabelling.evaluate_refit_pipeline import (
    compute_span_metrics,
    format_classification_report,
    load_gt_jsonl_strict,
)
from tools.inspect_dense_tips import read_json_or_jsonl

LOGGER = logging.getLogger(__name__)


@dataclass
class QuickTrainConfig:
    train_path: str = "../data/dd_corpus_small_train.json"
    test_path: str = "../data/dd_corpus_small_test.json"
    pseudolabel_path: str = ""
    train_mode: str = "supervised_only"
    model_base: str = "urchade/gliner_multi-v2.1"
    output_dir: str = "./artifacts/base_model_training/quick_run"
    seed: int = 42
    keep_empty_samples: bool = False
    keep_empty_chunks: bool = False
    deduplicate_by_text: bool = True
    pseudolabel_sample_ratio: float = 1.0
    max_pseudolabel_records: int = 0
    tokenization_strategy: str = "whitespace"
    batch_size: int = 16
    num_epochs: int = 20
    max_length: int = 384
    overlap: int = 100
    backbone_lr: float = 1.0e-5
    ner_lr: float = 3.38e-5
    weight_decay: float = 0.086619
    train_sampling: str = "weighted"
    refit_val_size: float = 0.2
    early_stopping_patience: int = 7
    early_stopping_threshold: float = 0.5
    thresholds: list[float] | None = None
    log_level: str = "INFO"


def parse_args():
    defaults = QuickTrainConfig(thresholds=[0.6])
    parser = argparse.ArgumentParser(description="Quick single-split GLiNER training")
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
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--keep-empty-samples", action="store_true", default=defaults.keep_empty_samples)
    parser.add_argument("--keep-empty-chunks", action="store_true", default=defaults.keep_empty_chunks)
    parser.add_argument("--disable-deduplicate-by-text", action="store_true")
    parser.add_argument("--pseudolabel-sample-ratio", type=float, default=defaults.pseudolabel_sample_ratio)
    parser.add_argument("--max-pseudolabel-records", type=int, default=defaults.max_pseudolabel_records)
    parser.add_argument(
        "--tokenization-strategy",
        choices=["whitespace", "regex"],
        default=defaults.tokenization_strategy,
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--overlap", type=int, default=defaults.overlap)
    parser.add_argument("--backbone-lr", type=float, default=defaults.backbone_lr)
    parser.add_argument("--ner-lr", type=float, default=defaults.ner_lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--train-sampling", choices=["random", "weighted"], default=defaults.train_sampling)
    parser.add_argument("--refit-val-size", type=float, default=defaults.refit_val_size)
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--early-stopping-threshold", type=float, default=defaults.early_stopping_threshold)
    parser.add_argument("--thresholds", default=",".join(str(value) for value in defaults.thresholds))
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
        seed=args.seed,
        keep_empty_samples=args.keep_empty_samples,
        keep_empty_chunks=args.keep_empty_chunks,
        deduplicate_by_text=(not args.disable_deduplicate_by_text),
        pseudolabel_sample_ratio=args.pseudolabel_sample_ratio,
        max_pseudolabel_records=args.max_pseudolabel_records,
        tokenization_strategy=args.tokenization_strategy,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        overlap=args.overlap,
        backbone_lr=args.backbone_lr,
        ner_lr=args.ner_lr,
        weight_decay=args.weight_decay,
        train_sampling=args.train_sampling,
        refit_val_size=args.refit_val_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        thresholds=parse_thresholds(args.thresholds),
        log_level=args.log_level,
    )


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
    sample_id = row.get("sample_id")
    normalized = {"text": text, "spans": spans}
    if sample_id is not None:
        normalized["sample_id"] = sample_id
    return normalized


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


def _deduplicate_pseudolabel_rows_against_supervised(supervised_rows, pseudolabel_rows):
    supervised_texts = {row["text"] for row in supervised_rows}
    return [row for row in pseudolabel_rows if row["text"] not in supervised_texts]


def _load_gt_rows(path: Path):
    return load_gt_jsonl_strict(str(path))


def _predict_rows(model, rows, labels, threshold):
    pred_rows = []
    for row in rows:
        entities = predict_entities_for_text(model, row["text"], labels, threshold)
        pred_rows.append({"text": row["text"], "entities": [entity for entity in entities if entity["label"] in labels]})
    return pred_rows


def _write_report(report_dir: Path, metrics, predictions, threshold):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "classification_report.txt").write_text(format_classification_report(metrics), encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "selected_threshold.json").write_text(
        json.dumps({"threshold": threshold}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_jsonl(str(report_dir / "predictions.jsonl"), predictions)


def run_quick_experiment(config: QuickTrainConfig, script_path: str):
    set_seed(config.seed)
    started_at = datetime.now(timezone.utc)
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    output_dir = resolve_path(script_dir, config.output_dir)
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

    model_base_candidate = resolve_path(script_dir, config.model_base)
    model_base = str(model_base_candidate) if model_base_candidate.exists() else config.model_base
    model_base = materialize_model_base(model_base)

    base_model = _load_model(model_base=model_base, local_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    supervised_rows_all = [_normalize_spans_row(row) for row in load_gt_jsonl_strict(str(train_path))]
    supervised_rows_all = [row for row in supervised_rows_all if row is not None]
    pseudolabel_rows_all = []
    if pseudolabel_path is not None and pseudolabel_path.exists():
        pseudolabel_rows_all = _convert_entity_rows_to_spans(read_json_or_jsonl(str(pseudolabel_path)))
        pseudolabel_rows_all = _sample_rows(
            pseudolabel_rows_all,
            seed=config.seed,
            sample_ratio=config.pseudolabel_sample_ratio,
            max_records=config.max_pseudolabel_records,
        )
        if config.deduplicate_by_text:
            pseudolabel_rows_all = _deduplicate_pseudolabel_rows_against_supervised(
                supervised_rows_all,
                pseudolabel_rows_all,
            )

    raw_data = load_dataset(str(train_path), tokenization_strategy=config.tokenization_strategy)
    pseudolabel_raw_data = []
    if config.train_mode != "supervised_only":
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp_handle:
            tmp_path = Path(tmp_handle.name)
        try:
            save_jsonl(str(tmp_path), pseudolabel_rows_all)
            pseudolabel_raw_data = load_dataset(str(tmp_path), tokenization_strategy=config.tokenization_strategy)
        finally:
            tmp_path.unlink(missing_ok=True)

    if config.keep_empty_samples:
        filtered_data = list(raw_data)
    else:
        filtered_data = [sample for sample in raw_data if sample["ner"]]

    dataset = split_long_sentences(
        filtered_data,
        max_length=config.max_length,
        overlap=config.overlap,
        tokenizer=getattr(base_model.data_processor, "transformer_tokenizer", None),
        keep_empty_chunks=config.keep_empty_chunks,
    )
    if not dataset:
        raise ValueError("Dataset is empty after preprocessing.")

    entity_labels = sorted({label for sample in dataset for _, _, label in sample["ner"]})
    if not entity_labels:
        raise ValueError("No entity labels found after preprocessing.")

    train_groups = [sample.get("sample_id") for sample in dataset]
    import numpy as np

    train_subset, val_subset = _build_refit_split(
        trainval_data=dataset,
        trainval_groups=np.asarray(train_groups),
        refit_val_size=config.refit_val_size,
        seed=config.seed,
    )

    pseudolabel_filtered = list(pseudolabel_raw_data) if config.keep_empty_samples else [sample for sample in pseudolabel_raw_data if sample["ner"]]
    pseudolabel_dataset = split_long_sentences(
        pseudolabel_filtered,
        max_length=config.max_length,
        overlap=config.overlap,
        tokenizer=getattr(base_model.data_processor, "transformer_tokenizer", None),
        keep_empty_chunks=config.keep_empty_chunks,
    ) if pseudolabel_filtered else []

    if config.train_mode == "pseudolabel_only":
        train_subset = list(pseudolabel_dataset)
    elif config.train_mode == "supervised_plus_pseudolabels":
        train_subset = list(train_subset) + list(pseudolabel_dataset)

    model, history = _run_single_training(
        base_model=base_model,
        train_data=train_subset,
        val_data=val_subset,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        backbone_lr=config.backbone_lr,
        ner_lr=config.ner_lr,
        weight_decay=config.weight_decay,
        train_sampling=config.train_sampling,
        patience=config.early_stopping_patience,
        thresholds=config.thresholds,
        early_stopping_threshold=config.early_stopping_threshold,
        entity_labels=entity_labels,
        device=device,
        stage_label="quick-train",
    )
    if model is None:
        raise ValueError("Training failed because the train or validation loader was empty.")

    val_processed = _prepare_char_offsets(val_subset)
    threshold_scores = _evaluate_thresholds(model, val_processed, config.thresholds, entity_labels)
    best_threshold = max(threshold_scores, key=threshold_scores.get)

    test_rows = _load_gt_rows(test_path)
    test_predictions = _predict_rows(model, test_rows, entity_labels, threshold=best_threshold)
    test_metrics = compute_span_metrics(
        gold_spans_by_row=[row["spans"] for row in test_rows],
        pred_spans_by_row=[row["entities"] for row in test_predictions],
        labels=entity_labels,
    )

    seen_unseen = _compute_seen_unseen_breakdown(
        model=model,
        dataset=[(row["text"], row["spans"]) for row in test_rows],
        threshold=best_threshold,
        entity_labels=entity_labels,
        seen_entity_keys=_build_seen_entity_keys(train_subset),
    )

    model_dir = output_dir / "best_quick_gliner_model"
    model.save_pretrained(model_dir)
    _write_report(output_dir / "eval_test", test_metrics, test_predictions, best_threshold)

    finished_at = datetime.now(timezone.utc)
    runtime_seconds = perf_counter() - timer
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "runtime_seconds": runtime_seconds,
        "config": asdict(config),
        "dataset": {
            "raw_supervised_rows": len(supervised_rows_all),
            "raw_pseudolabel_rows": len(pseudolabel_rows_all),
            "train_mode": config.train_mode,
            "raw_train_rows": len(raw_data),
            "filtered_train_rows": len(filtered_data),
            "processed_rows": len(dataset),
            "pseudolabel_processed_rows": len(pseudolabel_dataset),
            "train_rows": len(train_subset),
            "val_rows": len(val_subset),
            "entity_labels": entity_labels,
        },
        "training": {
            "best_validation_metric": history["best_metric"],
            "mean_train_loss": mean(history["training_losses"]) if history["training_losses"] else None,
            "mean_val_loss": mean(history["validation_losses"]) if history["validation_losses"] else None,
        },
        "threshold_selection": {
            "scores": threshold_scores,
            "best_threshold": best_threshold,
        },
        "test_metrics": test_metrics,
        "seen_unseen": seen_unseen,
        "artifacts": {
            "model_dir": str(model_dir),
            "eval_dir": str(output_dir / "eval_test"),
        },
    }
    (output_dir / "quick_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    clear_cuda_cache()
    LOGGER.info("Quick training completed. Summary: %s", output_dir / "quick_summary.json")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = build_config(args)
    run_quick_experiment(config, script_path=__file__)


if __name__ == "__main__":
    main()
