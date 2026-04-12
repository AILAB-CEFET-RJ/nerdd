#!/usr/bin/env python3
"""Mine out-of-fold baseline errors on the training corpus."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger(__name__)


@dataclass
class OofConfig:
    train_path: str = "../data/dd_corpus_small_train.json"
    model_base: str = "urchade/gliner_multi-v2.1"
    output_dir: str = "../artifacts/error_mining/train_oof_baseline"
    seed: int = 42
    n_splits: int = 5
    keep_empty_samples: bool = False
    keep_empty_chunks: bool = False
    tokenization_strategy: str = "whitespace"
    batch_size: int = 4
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
    threshold: float = 0.6
    log_level: str = "INFO"


def parse_args() -> argparse.Namespace:
    cfg = OofConfig()
    parser = argparse.ArgumentParser(description="Mine out-of-fold baseline errors on the train corpus.")
    parser.add_argument("--train-path", default=cfg.train_path)
    parser.add_argument("--model-base", default=cfg.model_base)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--n-splits", type=int, default=cfg.n_splits)
    parser.add_argument("--keep-empty-samples", action="store_true", default=cfg.keep_empty_samples)
    parser.add_argument("--keep-empty-chunks", action="store_true", default=cfg.keep_empty_chunks)
    parser.add_argument("--tokenization-strategy", choices=["whitespace", "regex"], default=cfg.tokenization_strategy)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--num-epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--overlap", type=int, default=cfg.overlap)
    parser.add_argument("--backbone-lr", type=float, default=cfg.backbone_lr)
    parser.add_argument("--ner-lr", type=float, default=cfg.ner_lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--train-sampling", choices=["random", "weighted"], default=cfg.train_sampling)
    parser.add_argument("--refit-val-size", type=float, default=cfg.refit_val_size)
    parser.add_argument("--early-stopping-patience", type=int, default=cfg.early_stopping_patience)
    parser.add_argument("--early-stopping-threshold", type=float, default=cfg.early_stopping_threshold)
    parser.add_argument("--threshold", type=float, default=cfg.threshold)
    parser.add_argument("--log-level", default=cfg.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _span_set(spans: list[dict]) -> set[tuple[int, int, str]]:
    return {(int(span["start"]), int(span["end"]), str(span["label"])) for span in spans}


def _span_text(text: str, span: dict) -> str:
    return text[int(span["start"]) : int(span["end"])]


def _extract_error_tags(text: str, gold_spans: list[dict], pred_spans: list[dict]) -> list[str]:
    tags = []
    gold_set = _span_set(gold_spans)
    pred_set = _span_set(pred_spans)
    fn = gold_set - pred_set
    fp = pred_set - gold_set
    if fn:
        tags.append("has_fn")
    if fp:
        tags.append("has_fp")

    gold_by_label = {}
    for span in gold_spans:
        gold_by_label.setdefault(span["label"], []).append(span)
    pred_by_label = {}
    for span in pred_spans:
        pred_by_label.setdefault(span["label"], []).append(span)

    for label, gold_list in gold_by_label.items():
        pred_list = pred_by_label.get(label, [])
        for gold in gold_list:
            for pred in pred_list:
                if gold["start"] == pred["start"] and gold["end"] == pred["end"]:
                    continue
                if _span_text(text, gold).strip() == _span_text(text, pred).strip():
                    tags.append("label_confusion")
                gold_text = _span_text(text, gold)
                pred_text = _span_text(text, pred)
                if gold["label"] == pred["label"] and gold_text in pred_text and gold_text != pred_text:
                    tags.append("boundary_expansion")
                if gold["label"] == pred["label"] and pred_text in gold_text and gold_text != pred_text:
                    tags.append("boundary_truncation")

    return sorted(set(tags))


def _build_error_row(row_index: int, sample: dict, gold_spans: list[dict], pred_spans: list[dict]) -> dict | None:
    gold_set = _span_set(gold_spans)
    pred_set = _span_set(pred_spans)
    if gold_set == pred_set:
        return None
    text = sample["text"]
    return {
        "row_index_1based": row_index,
        "sample_id": sample.get("sample_id"),
        "text": text,
        "gold_spans": gold_spans,
        "pred_spans": pred_spans,
        "fn_spans": [span for span in gold_spans if (span["start"], span["end"], span["label"]) in (gold_set - pred_set)],
        "fp_spans": [span for span in pred_spans if (span["start"], span["end"], span["label"]) in (pred_set - gold_set)],
        "error_tags": _extract_error_tags(text, gold_spans, pred_spans),
    }


def main() -> None:
    import torch

    from base_model_training.cv import (
        _build_group_splitter,
        _extract_groups,
        _build_refit_split,
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
    from pseudolabelling.evaluate_refit_pipeline import compute_span_metrics

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    set_seed(args.seed)
    started = perf_counter()

    script_dir = Path(__file__).resolve().parent
    train_path = resolve_path(script_dir, args.train_path)
    output_dir = resolve_path(script_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_base_candidate = resolve_path(script_dir, args.model_base)
    model_base = str(model_base_candidate) if model_base_candidate.exists() else args.model_base
    model_base = materialize_model_base(model_base)
    base_model = _load_model(model_base=model_base, local_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data = load_dataset(str(train_path), tokenization_strategy=args.tokenization_strategy)
    filtered_data = list(raw_data) if args.keep_empty_samples else [sample for sample in raw_data if sample["ner"]]
    dataset = split_long_sentences(
        filtered_data,
        max_length=args.max_length,
        overlap=args.overlap,
        tokenizer=getattr(base_model.data_processor, "transformer_tokenizer", None),
        keep_empty_chunks=args.keep_empty_chunks,
    )
    if not dataset:
        raise ValueError("Dataset is empty after preprocessing.")

    entity_labels = sorted({label for sample in dataset for _, _, label in sample["ner"]})
    groups = _extract_groups(dataset)
    splitter = _build_group_splitter(dataset, groups, args.n_splits, "OOF train error mining", args.seed)

    all_gold = []
    all_pred = []
    prediction_rows = []
    error_rows = []
    fold_summaries = []

    for fold_idx, (trainval_idx, holdout_idx) in enumerate(splitter.split(dataset, groups=groups), start=1):
        trainval_data = [dataset[index] for index in trainval_idx]
        holdout_data = [dataset[index] for index in holdout_idx]
        trainval_groups = _extract_groups(trainval_data)
        train_subset, val_subset = _build_refit_split(
            trainval_data=trainval_data,
            trainval_groups=trainval_groups,
            refit_val_size=args.refit_val_size,
            seed=args.seed + fold_idx,
        )

        model, history = _run_single_training(
            base_model=base_model,
            train_data=train_subset,
            val_data=val_subset,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            backbone_lr=args.backbone_lr,
            ner_lr=args.ner_lr,
            weight_decay=args.weight_decay,
            train_sampling=args.train_sampling,
            patience=args.early_stopping_patience,
            thresholds=[args.threshold],
            early_stopping_threshold=args.early_stopping_threshold,
            entity_labels=entity_labels,
            device=device,
            stage_label=f"oof-fold-{fold_idx}",
        )
        if model is None:
            raise RuntimeError(f"Training failed for fold {fold_idx}.")

        holdout_gold = _prepare_char_offsets(holdout_data)
        holdout_pred = []
        for sample in holdout_gold:
            preds = predict_entities_for_text(model, sample["text"], entity_labels, args.threshold)
            holdout_pred.append([entity for entity in preds if entity["label"] in entity_labels])

        all_gold.extend([row["spans"] for row in holdout_gold])
        all_pred.extend(holdout_pred)

        for row_index, (sample, pred_spans) in enumerate(zip(holdout_gold, holdout_pred), start=1):
            prediction_rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "text": sample["text"],
                    "gold_spans": sample["spans"],
                    "pred_spans": pred_spans,
                    "fold": fold_idx,
                }
            )
            error_row = _build_error_row(row_index, sample, sample["spans"], pred_spans)
            if error_row is not None:
                error_row["fold"] = fold_idx
                error_rows.append(error_row)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_rows": len(train_subset),
                "val_rows": len(val_subset),
                "holdout_rows": len(holdout_data),
                "best_validation_metric": history["best_metric"],
            }
        )
        clear_cuda_cache()

    metrics = compute_span_metrics(all_gold, all_pred, entity_labels)
    tag_counts = Counter(tag for row in error_rows for tag in row.get("error_tags", []))

    save_jsonl(str(output_dir / "oof_predictions.jsonl"), prediction_rows)
    save_jsonl(str(output_dir / "oof_error_cases.jsonl"), error_rows)
    summary = {
        "train_path": str(train_path),
        "model_base": model_base,
        "runtime_seconds": perf_counter() - started,
        "n_splits": splitter.n_splits,
        "dataset_rows": len(dataset),
        "entity_labels": entity_labels,
        "metrics": metrics,
        "error_rows": len(error_rows),
        "error_tag_counts": dict(tag_counts),
        "folds": fold_summaries,
        "artifacts": {
            "oof_predictions_jsonl": str((output_dir / "oof_predictions.jsonl").resolve()),
            "oof_error_cases_jsonl": str((output_dir / "oof_error_cases.jsonl").resolve()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved OOF summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
