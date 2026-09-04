#!/usr/bin/env python3
"""Generate out-of-fold NER predictions on the training corpus.

The output is meant for error analysis, score calibration, and boost-factor
simulation. Each prediction row preserves the original train row identity and,
optionally, metadata fields recovered from a separate source file by exact
normalized text match.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl

LOGGER = logging.getLogger(__name__)


@dataclass
class OofConfig:
    train_path: str = "data/dd_corpus_small_train.json"
    model_base: str = "urchade/gliner_multi-v2.1"
    output_dir: str = "artifacts/error_analysis/train_oof_baseline"
    seed: int = 42
    n_splits: int = 5
    keep_empty_samples: bool = False
    keep_empty_chunks: bool = False
    tokenization_strategy: str = "regex"
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
    validation_threshold: float = 0.6
    prediction_threshold: float = 0.0
    eval_threshold: float = 0.6
    metadata_source: str = ""
    metadata_text_fields: str = "relato,text"
    metadata_fields: str = "logradouroLocal,bairroLocal,cidadeLocal,pontodeReferenciaLocal"
    log_level: str = "INFO"


def parse_args() -> argparse.Namespace:
    cfg = OofConfig()
    parser = argparse.ArgumentParser(
        description="Generate out-of-fold NER predictions on the train corpus."
    )
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
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=cfg.validation_threshold,
        help="Threshold used inside validation/early-stopping metric during fold training.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Alias for --validation-threshold.",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=cfg.prediction_threshold,
        help="Threshold used to emit OOF pred_spans. Use 0.0 for boost-factor simulation.",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=cfg.eval_threshold,
        help="Threshold used to derive pred_spans_eval and summary/error metrics.",
    )
    parser.add_argument(
        "--metadata-source",
        default=cfg.metadata_source,
        help="Optional JSON/JSONL source used to recover metadata fields by normalized text match.",
    )
    parser.add_argument(
        "--metadata-text-fields",
        default=cfg.metadata_text_fields,
        help="Comma-separated text fields to try in --metadata-source.",
    )
    parser.add_argument(
        "--metadata-fields",
        default=cfg.metadata_fields,
        help="Comma-separated metadata fields to copy from --metadata-source.",
    )
    parser.add_argument("--log-level", default=cfg.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.threshold is not None:
        args.validation_threshold = args.threshold
    return args


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _span_set(spans: list[dict[str, Any]]) -> set[tuple[int, int, str]]:
    return {(int(span["start"]), int(span["end"]), str(span["label"])) for span in spans}


def _span_text(text: str, span: dict[str, Any]) -> str:
    return text[int(span["start"]) : int(span["end"])]


def _score_value(span: dict[str, Any]) -> float | None:
    try:
        return float(span["score"])
    except (KeyError, TypeError, ValueError):
        return None


def _filter_spans_by_score(spans: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    output = []
    for span in spans:
        score = _score_value(span)
        if score is None or score >= threshold:
            output.append(span)
    return output


def _extract_error_tags(text: str, gold_spans: list[dict[str, Any]], pred_spans: list[dict[str, Any]]) -> list[str]:
    tags = []
    gold_set = _span_set(gold_spans)
    pred_set = _span_set(pred_spans)
    fn = gold_set - pred_set
    fp = pred_set - gold_set
    if fn:
        tags.append("has_fn")
    if fp:
        tags.append("has_fp")

    gold_by_label: dict[str, list[dict[str, Any]]] = {}
    for span in gold_spans:
        gold_by_label.setdefault(str(span["label"]), []).append(span)
    pred_by_label: dict[str, list[dict[str, Any]]] = {}
    for span in pred_spans:
        pred_by_label.setdefault(str(span["label"]), []).append(span)

    for label, gold_list in gold_by_label.items():
        pred_list = pred_by_label.get(label, [])
        for gold in gold_list:
            for pred in pred_list:
                if gold["start"] == pred["start"] and gold["end"] == pred["end"]:
                    continue
                gold_text = _span_text(text, gold)
                pred_text = _span_text(text, pred)
                if gold_text.strip() == pred_text.strip() and gold["label"] != pred["label"]:
                    tags.append("label_confusion")
                if gold["label"] == pred["label"] and gold_text in pred_text and gold_text != pred_text:
                    tags.append("boundary_expansion")
                if gold["label"] == pred["label"] and pred_text in gold_text and gold_text != pred_text:
                    tags.append("boundary_truncation")

    return sorted(set(tags))


def _build_error_row(
    *,
    source_row: dict[str, Any],
    text: str,
    gold_spans: list[dict[str, Any]],
    pred_spans: list[dict[str, Any]],
) -> dict[str, Any] | None:
    gold_set = _span_set(gold_spans)
    pred_set = _span_set(pred_spans)
    if gold_set == pred_set:
        return None
    row = {
        "row_index_0based": source_row["row_index_0based"],
        "row_index_1based": source_row["row_index_1based"],
        "sample_id": source_row.get("sample_id"),
        "text": text,
        "gold_spans": gold_spans,
        "pred_spans": pred_spans,
        "fn_spans": [span for span in gold_spans if (span["start"], span["end"], span["label"]) in (gold_set - pred_set)],
        "fp_spans": [span for span in pred_spans if (span["start"], span["end"], span["label"]) in (pred_set - gold_set)],
        "error_tags": _extract_error_tags(text, gold_spans, pred_spans),
        "source_fields": source_row.get("source_fields", {}),
        "metadata_match": source_row.get("metadata_match", {}),
    }
    row.update(source_row.get("source_fields", {}))
    return row


def _normalize_for_match(value: str) -> str:
    from pseudolabelling.context_boost import normalize_text

    return normalize_text(value)


def _pick_source_text(row: dict[str, Any], text_fields: list[str]) -> str:
    for field in text_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_metadata_lookup(
    metadata_source: Path | None,
    *,
    text_fields: list[str],
    metadata_fields: list[str],
) -> tuple[dict[str, dict[str, Any]], Counter]:
    stats = Counter()
    if metadata_source is None:
        return {}, stats
    rows = load_jsonl(str(metadata_source))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        text = _pick_source_text(row, text_fields)
        if not text:
            stats["source_rows_without_text"] += 1
            continue
        key = _normalize_for_match(text)
        if not key:
            stats["source_rows_without_text"] += 1
            continue
        grouped[key].append(
            {
                "source_row_index_0based": row_index,
                "source_row_index_1based": row_index + 1,
                "metadata_fields": {
                    field: row.get(field)
                    for field in metadata_fields
                    if isinstance(row.get(field), str) and row.get(field).strip()
                },
            }
        )
    lookup = {}
    for key, matches in grouped.items():
        if len(matches) == 1:
            lookup[key] = matches[0]
            stats["unique_text_keys"] += 1
        else:
            stats["ambiguous_text_keys"] += 1
    stats["source_rows"] = len(rows)
    return lookup, stats


def _metadata_for_text(text: str, lookup: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    key = _normalize_for_match(text)
    match = lookup.get(key)
    if not match:
        return {}, {"status": "unmatched"}
    return dict(match["metadata_fields"]), {
        "status": "matched_unique",
        "source_row_index_0based": match["source_row_index_0based"],
        "source_row_index_1based": match["source_row_index_1based"],
    }


def _build_source_row(
    row: dict[str, Any],
    row_index: int,
    metadata_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    text = row["text"]
    sample_id = str(row.get("sample_id") or f"sample_{row_index}")
    metadata_fields, metadata_match = _metadata_for_text(text, metadata_lookup)
    source_fields = {
        key: value
        for key, value in row.items()
        if key not in {"text", "spans"} and not key.startswith("_")
    }
    source_fields.update(metadata_fields)
    return {
        "row_index_0based": row_index,
        "row_index_1based": row_index + 1,
        "sample_id": sample_id,
        "text": text,
        "spans": row.get("spans", []) or [],
        "source_fields": source_fields,
        "metadata_match": metadata_match,
    }


def _load_oof_dataset(
    train_path: Path,
    *,
    tokenization_strategy: str,
    metadata_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Counter]:
    from base_model_training.data import process_sample

    raw_rows = load_jsonl(str(train_path))
    dataset = []
    source_by_sample_id = {}
    stats = Counter()
    for row_index, row in enumerate(raw_rows):
        if not isinstance(row.get("text"), str):
            raise ValueError(f"Train row {row_index + 1} has no string text field.")
        source_row = _build_source_row(row, row_index, metadata_lookup)
        sample = {
            "text": source_row["text"],
            "spans": source_row["spans"],
            "sample_id": source_row["sample_id"],
        }
        processed = process_sample(sample, tokenization_strategy=tokenization_strategy)
        dataset.append(processed)
        source_by_sample_id[source_row["sample_id"]] = source_row
        if source_row["metadata_match"].get("status") == "matched_unique":
            stats["metadata_matched_rows"] += 1
        else:
            stats["metadata_unmatched_rows"] += 1
    stats["raw_rows"] = len(raw_rows)
    return dataset, source_by_sample_id, stats


def _materialize_holdout_rows(
    dataset: list[dict[str, Any]],
    source_by_sample_id: dict[str, dict[str, Any]],
    chunk_counters: Counter,
) -> list[dict[str, Any]]:
    from base_model_training.cv import _prepare_char_offsets

    rows = []
    for sample, (text, spans) in zip(dataset, _prepare_char_offsets(dataset)):
        sample_id = str(sample.get("sample_id"))
        source_row = source_by_sample_id[sample_id]
        chunk_counters[sample_id] += 1
        rows.append(
            {
                "sample_id": sample_id,
                "text": text,
                "spans": spans,
                "source_row": source_row,
                "chunk_index_for_source_row": chunk_counters[sample_id],
            }
        )
    return rows


def _prediction_row(
    *,
    holdout_row: dict[str, Any],
    pred_spans: list[dict[str, Any]],
    pred_spans_eval: list[dict[str, Any]],
    fold_idx: int,
    prediction_threshold: float,
    eval_threshold: float,
) -> dict[str, Any]:
    source_row = holdout_row["source_row"]
    row = {
        "row_index_0based": source_row["row_index_0based"],
        "row_index_1based": source_row["row_index_1based"],
        "sample_id": holdout_row.get("sample_id"),
        "fold": fold_idx,
        "chunk_index_for_source_row": holdout_row.get("chunk_index_for_source_row"),
        "text": holdout_row["text"],
        "gold_spans": holdout_row["spans"],
        "pred_spans": pred_spans,
        "pred_spans_eval": pred_spans_eval,
        "prediction_threshold": prediction_threshold,
        "eval_threshold": eval_threshold,
        "source_fields": source_row.get("source_fields", {}),
        "metadata_match": source_row.get("metadata_match", {}),
    }
    row.update(source_row.get("source_fields", {}))
    return row


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import torch

    from base_model_training.cv import (
        _build_group_splitter,
        _build_refit_split,
        _extract_groups,
        _load_model,
        _run_single_training,
        clear_cuda_cache,
        materialize_model_base,
        set_seed,
    )
    from base_model_training.data import split_long_sentences
    from base_model_training.evaluate import predict_entities_for_text
    from base_model_training.io_utils import save_jsonl
    from base_model_training.paths import resolve_path
    from pseudolabelling.evaluate_refit_pipeline import compute_span_metrics

    set_seed(args.seed)
    started = perf_counter()

    script_dir = Path(__file__).resolve().parent
    train_path = resolve_path(script_dir, args.train_path)
    output_dir = resolve_path(script_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_fields = _parse_csv(args.metadata_fields)
    metadata_text_fields = _parse_csv(args.metadata_text_fields)
    metadata_source = resolve_path(script_dir, args.metadata_source) if args.metadata_source else None
    if metadata_source is not None and not metadata_source.exists():
        raise FileNotFoundError(f"Metadata source not found: {metadata_source}")
    metadata_lookup, metadata_lookup_stats = _build_metadata_lookup(
        metadata_source,
        text_fields=metadata_text_fields,
        metadata_fields=metadata_fields,
    )

    model_base_candidate = resolve_path(script_dir, args.model_base)
    model_base = str(model_base_candidate) if model_base_candidate.exists() else args.model_base
    model_base = materialize_model_base(model_base)
    base_model = _load_model(model_base=model_base, local_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data, source_by_sample_id, source_stats = _load_oof_dataset(
        train_path,
        tokenization_strategy=args.tokenization_strategy,
        metadata_lookup=metadata_lookup,
    )
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
    all_pred_eval = []
    prediction_rows = []
    error_rows = []
    fold_summaries = []
    chunk_counters: Counter = Counter()

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
            thresholds=[args.validation_threshold],
            early_stopping_threshold=args.early_stopping_threshold,
            entity_labels=entity_labels,
            device=device,
            stage_label=f"oof-fold-{fold_idx}",
        )
        if model is None:
            raise RuntimeError(f"Training failed for fold {fold_idx}.")

        holdout_gold = _materialize_holdout_rows(holdout_data, source_by_sample_id, chunk_counters)
        holdout_pred = []
        holdout_pred_eval = []
        for sample in holdout_gold:
            preds = predict_entities_for_text(
                model,
                sample["text"],
                entity_labels,
                args.prediction_threshold,
            )
            pred_spans = [entity for entity in preds if entity["label"] in entity_labels]
            pred_spans_eval = _filter_spans_by_score(pred_spans, args.eval_threshold)
            holdout_pred.append(pred_spans)
            holdout_pred_eval.append(pred_spans_eval)

        all_gold.extend([row["spans"] for row in holdout_gold])
        all_pred_eval.extend(holdout_pred_eval)

        for sample, pred_spans, pred_spans_eval in zip(holdout_gold, holdout_pred, holdout_pred_eval):
            prediction_rows.append(
                _prediction_row(
                    holdout_row=sample,
                    pred_spans=pred_spans,
                    pred_spans_eval=pred_spans_eval,
                    fold_idx=fold_idx,
                    prediction_threshold=args.prediction_threshold,
                    eval_threshold=args.eval_threshold,
                )
            )
            error_row = _build_error_row(
                source_row=sample["source_row"],
                text=sample["text"],
                gold_spans=sample["spans"],
                pred_spans=pred_spans_eval,
            )
            if error_row is not None:
                error_row["fold"] = fold_idx
                error_row["chunk_index_for_source_row"] = sample.get("chunk_index_for_source_row")
                error_row["prediction_threshold"] = args.prediction_threshold
                error_row["eval_threshold"] = args.eval_threshold
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

    metrics = compute_span_metrics(all_gold, all_pred_eval, entity_labels)
    tag_counts = Counter(tag for row in error_rows for tag in row.get("error_tags", []))

    save_jsonl(str(output_dir / "oof_predictions.jsonl"), prediction_rows)
    save_jsonl(str(output_dir / "oof_error_cases.jsonl"), error_rows)
    summary = {
        "train_path": str(train_path),
        "model_base": model_base,
        "runtime_seconds": perf_counter() - started,
        "n_splits": splitter.n_splits,
        "raw_rows": source_stats["raw_rows"],
        "filtered_rows": len(filtered_data),
        "dataset_rows": len(dataset),
        "entity_labels": entity_labels,
        "thresholds": {
            "validation_threshold": args.validation_threshold,
            "prediction_threshold": args.prediction_threshold,
            "eval_threshold": args.eval_threshold,
        },
        "metadata_source": str(metadata_source) if metadata_source else "",
        "metadata_fields": metadata_fields,
        "metadata_text_fields": metadata_text_fields,
        "metadata_lookup_stats": dict(metadata_lookup_stats),
        "source_stats": dict(source_stats),
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
