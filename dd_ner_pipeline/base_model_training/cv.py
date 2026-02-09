import gc
import json
import logging
import os
import random
import copy
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from gliner import GLiNER
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.optim import AdamW

from gliner_train.collator_factory import build_data_collator
from gliner_train.data import (
    create_dataloader,
    load_dataset,
    split_long_sentences,
    token_spans_to_char_offsets,
)
from gliner_train.engine import train_with_early_stopping
from gliner_train.metrics import compute_f1_by_threshold, f1_score_from_span_lists
from gliner_train.paths import resolve_path
from gliner_train.plots import save_loss_plot
from gliner_train.search import generate_trial_params

LOGGER = logging.getLogger(__name__)
TOKENIZER_REGEX_WARNING_PATTERN = re.compile(r"incorrect regex pattern", re.IGNORECASE)
TOKENIZER_REGEX_WARNING_SEEN = False


def set_seed(seed):
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda_cache():
    """Release GPU cache memory between trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_param(value, fmt):
    """Safely format optional numeric values."""
    if value is None:
        return "N/A"
    return format(value, fmt)


def format_duration(seconds):
    """Format duration in HH:MM:SS."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def materialize_model_base(model_base):
    """Resolve remote HF repo id into a local snapshot once, to avoid network during training."""
    if "/" not in model_base and "\\" not in model_base:
        return model_base

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return model_base

    LOGGER.info("Resolving model base '%s' to local cache...", model_base)
    local_dir = snapshot_download(repo_id=model_base)
    LOGGER.info("Model base cached at: %s", local_dir)
    return local_dir


def _prime_and_freeze_hf_cache(model_base):
    """Load model once to warm dependent caches, then force offline mode for stability."""
    LOGGER.info("Priming GLiNER dependencies in local cache...")
    model = _load_model(model_base=model_base, local_only=False)
    del model
    clear_cuda_cache()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from huggingface_hub import constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass
    try:
        from transformers.utils import import_utils as tf_import_utils

        tf_import_utils._is_offline_mode = True
    except Exception:
        pass
    LOGGER.info("HF Hub offline mode enabled for the rest of this run.")


def _load_model(model_base, local_only):
    global TOKENIZER_REGEX_WARNING_SEEN

    def _load_with_warning_capture(callable_fn):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            loaded_model = callable_fn()

        for warning_item in caught_warnings:
            warning_message = str(warning_item.message)
            if TOKENIZER_REGEX_WARNING_PATTERN.search(warning_message):
                TOKENIZER_REGEX_WARNING_SEEN = True
                continue
            warnings.warn_explicit(
                warning_item.message,
                warning_item.category,
                warning_item.filename,
                warning_item.lineno,
            )
        return loaded_model

    # Try modern kwargs first; gracefully fall back for older GLiNER/transformers versions.
    try:
        return _load_with_warning_capture(
            lambda: GLiNER.from_pretrained(
                model_base,
                local_files_only=local_only,
                fix_mistral_regex=True,
            )
        )
    except TypeError:
        try:
            return _load_with_warning_capture(
                lambda: GLiNER.from_pretrained(model_base, local_files_only=local_only)
            )
        except TypeError:
            return _load_with_warning_capture(lambda: GLiNER.from_pretrained(model_base))


def _subset_by_indices(dataset, indices):
    return [dataset[index] for index in indices]


def _extract_groups(dataset):
    groups = []
    for index, sample in enumerate(dataset):
        group = sample.get("sample_id")
        if group is None:
            group = f"sample_{index}"
        groups.append(group)
    return np.asarray(groups)


def _effective_group_kfold(groups, requested_splits, stage_name):
    unique_groups = len(set(groups.tolist()))
    effective_splits = min(requested_splits, unique_groups)
    if effective_splits < requested_splits:
        LOGGER.warning(
            "%s requested %s splits, but only %s unique groups are available. "
            "Using %s splits.",
            stage_name,
            requested_splits,
            unique_groups,
            effective_splits,
        )
    if effective_splits < 2:
        raise ValueError(
            f"{stage_name} needs at least 2 unique groups; found {unique_groups}."
        )
    return GroupKFold(n_splits=effective_splits)


def _create_optimizer_scheduler(model, lr, weight_decay, num_epochs, train_loader_len):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = num_epochs * train_loader_len
    if total_steps < 3:
        return optimizer, None
    try:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="linear",
        )
        return optimizer, scheduler
    except ZeroDivisionError:
        LOGGER.warning(
            "OneCycleLR is unstable for total_steps=%s in this run; falling back to no scheduler.",
            total_steps,
        )
        return optimizer, None


def _prepare_char_offsets(dataset):
    return [token_spans_to_char_offsets(sample["tokenized_text"], sample["ner"]) for sample in dataset]


def _evaluate_thresholds(model, val_processed, thresholds, entity_labels):
    return {
        threshold: compute_f1_by_threshold(model, val_processed, threshold, entity_labels)
        for threshold in thresholds
    }


def _normalize_entity_surface(value):
    return re.sub(r"\s+", " ", value).strip().lower()


def _entity_key(label, surface):
    return (label, _normalize_entity_surface(surface))


def _build_seen_entity_keys(train_data):
    seen = set()
    for sample in train_data:
        tokens = sample["tokenized_text"]
        for start_idx, end_idx, label in sample["ner"]:
            if 0 <= start_idx <= end_idx < len(tokens):
                surface = " ".join(tokens[start_idx : end_idx + 1])
                seen.add(_entity_key(label, surface))
    return seen


def _split_spans_by_seen(text, spans, seen_entity_keys):
    seen_spans = []
    unseen_spans = []
    for span in spans:
        if "start" not in span or "end" not in span or "label" not in span:
            continue
        start = max(0, int(span["start"]))
        end = min(len(text), int(span["end"]))
        if end <= start:
            continue
        key = _entity_key(span["label"], text[start:end])
        if key in seen_entity_keys:
            seen_spans.append(span)
        else:
            unseen_spans.append(span)
    return seen_spans, unseen_spans


def _safe_f1(pred_spans_list, gold_spans_list):
    total_gold = sum(len(spans) for spans in gold_spans_list)
    total_pred = sum(len(spans) for spans in pred_spans_list)
    if total_gold == 0 and total_pred == 0:
        return None
    return f1_score_from_span_lists(pred_spans_list, gold_spans_list, average="macro")


def _compute_seen_unseen_breakdown(model, dataset, threshold, entity_labels, seen_entity_keys):
    seen_preds = []
    seen_gold = []
    unseen_preds = []
    unseen_gold = []

    for text, gold_spans in dataset:
        preds = model.predict_entities(text, labels=entity_labels, threshold=threshold)
        filtered_preds = [pred for pred in preds if pred["label"] in entity_labels]

        gold_seen, gold_unseen = _split_spans_by_seen(text, gold_spans, seen_entity_keys)
        pred_seen, pred_unseen = _split_spans_by_seen(text, filtered_preds, seen_entity_keys)

        seen_gold.append(gold_seen)
        seen_preds.append(pred_seen)
        unseen_gold.append(gold_unseen)
        unseen_preds.append(pred_unseen)

    seen_gold_support = sum(len(spans) for spans in seen_gold)
    unseen_gold_support = sum(len(spans) for spans in unseen_gold)

    return {
        "seen_f1": _safe_f1(seen_preds, seen_gold),
        "unseen_f1": _safe_f1(unseen_preds, unseen_gold),
        "seen_gold_support": seen_gold_support,
        "unseen_gold_support": unseen_gold_support,
    }


def _run_single_training(
    base_model,
    train_data,
    val_data,
    batch_size,
    num_epochs,
    lr,
    weight_decay,
    patience,
    early_stopping_threshold,
    entity_labels,
    device,
):
    model = copy.deepcopy(base_model)
    model.to(device)
    data_collator = build_data_collator(model)

    train_loader = create_dataloader(train_data, batch_size, data_collator, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size, data_collator, shuffle=False)

    if len(train_loader) == 0 or len(val_loader) == 0:
        return None, None

    optimizer, scheduler = _create_optimizer_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        train_loader_len=len(train_loader),
    )

    val_processed = _prepare_char_offsets(val_data)
    history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        metric_fn=lambda current_model: compute_f1_by_threshold(
            current_model,
            val_processed,
            threshold=early_stopping_threshold,
            entity_labels=entity_labels,
        ),
    )
    return model, history


def _build_refit_split(trainval_data, trainval_groups, refit_val_size, seed):
    unique_groups = len(set(trainval_groups.tolist()))
    if unique_groups >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=refit_val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(trainval_data, groups=trainval_groups))
        return _subset_by_indices(trainval_data, train_idx), _subset_by_indices(trainval_data, val_idx)

    val_size = max(1, int(round(len(trainval_data) * refit_val_size)))
    val_idx = np.arange(val_size)
    train_idx = np.arange(val_size, len(trainval_data))
    if len(train_idx) == 0:
        train_idx = val_idx
    return _subset_by_indices(trainval_data, train_idx), _subset_by_indices(trainval_data, val_idx)


def _run_inner_search(
    trainval_data,
    trainval_groups,
    config,
    base_model,
    entity_labels,
    device,
    fold,
    output_dir,
):
    splitter = _effective_group_kfold(
        groups=trainval_groups,
        requested_splits=config.n_inner_splits,
        stage_name=f"Inner CV (outer fold {fold})",
    )

    rng = np.random.default_rng(config.seed + fold)
    trial_candidates = generate_trial_params(config, rng)
    if config.search_mode == "random":
        LOGGER.info("Random search selected %s trial candidates.", len(trial_candidates))

    best_score = -1.0
    best_lr = None
    best_wd = None
    best_threshold = None
    trial_reports = []

    for trial_index, (lr, weight_decay) in enumerate(trial_candidates, start=1):
        LOGGER.info(
            "[Trial %s/%s] LR=%.7f | WD=%.6f",
            trial_index,
            len(trial_candidates),
            lr,
            weight_decay,
        )

        threshold_scores = {threshold: [] for threshold in config.thresholds}
        inner_fold_reports = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            splitter.split(trainval_data, groups=trainval_groups),
            start=1,
        ):
            inner_train_data = _subset_by_indices(trainval_data, inner_train_idx)
            inner_val_data = _subset_by_indices(trainval_data, inner_val_idx)

            model, history = _run_single_training(
                base_model=base_model,
                train_data=inner_train_data,
                val_data=inner_val_data,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                lr=lr,
                weight_decay=weight_decay,
                patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
                entity_labels=entity_labels,
                device=device,
            )
            if model is None:
                LOGGER.warning(
                    "Inner fold %s has an empty DataLoader; skipping this inner split.",
                    inner_fold,
                )
                clear_cuda_cache()
                continue

            val_processed = _prepare_char_offsets(inner_val_data)
            threshold_metrics = _evaluate_thresholds(
                model=model,
                val_processed=val_processed,
                thresholds=config.thresholds,
                entity_labels=entity_labels,
            )
            for threshold, score in threshold_metrics.items():
                threshold_scores[threshold].append(score)

            inner_fold_reports.append(
                {
                    "inner_fold": inner_fold,
                    "early_stopping_metric": history["best_metric"],
                    "threshold_scores": threshold_metrics,
                }
            )
            save_loss_plot(
                training_losses=history["training_losses"],
                validation_losses=history["validation_losses"],
                output_dir=output_dir,
                fold=fold,
                trial=f"t{trial_index}_inner{inner_fold}",
                lr=lr,
                weight_decay=weight_decay,
            )

            del history
            del model
            clear_cuda_cache()

        threshold_means = {
            threshold: float(np.mean(scores)) if scores else float("-inf")
            for threshold, scores in threshold_scores.items()
        }
        trial_best_threshold = max(threshold_means, key=threshold_means.get)
        trial_score = threshold_means[trial_best_threshold]

        trial_report = {
            "trial": trial_index,
            "lr": lr,
            "weight_decay": weight_decay,
            "mean_threshold_scores": threshold_means,
            "best_threshold": trial_best_threshold,
            "best_score": trial_score,
            "inner_folds": inner_fold_reports,
        }
        trial_reports.append(trial_report)

        if trial_score > best_score:
            best_score = trial_score
            best_lr = lr
            best_wd = weight_decay
            best_threshold = trial_best_threshold

    return best_lr, best_wd, best_threshold, best_score, trial_reports


def run_experiment(config, script_path):
    """Run nested cross-validation fine-tuning."""
    global TOKENIZER_REGEX_WARNING_SEEN
    TOKENIZER_REGEX_WARNING_SEEN = False
    set_seed(config.seed)
    started_at_utc = datetime.now(timezone.utc)
    started_at_iso = started_at_utc.isoformat()
    experiment_timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    output_dir = resolve_path(script_dir, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_path(script_dir, config.train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset not found: {train_path}")

    model_base_path = resolve_path(script_dir, config.model_base)
    if model_base_path.exists():
        model_base = str(model_base_path)
    else:
        model_base = config.model_base
        if "/" not in model_base and "\\" not in model_base:
            raise FileNotFoundError(
                f"Base model path not found locally: {model_base_path}. "
                "Pass --model-base with a local folder or a Hugging Face repo id "
                "(example: birdred/glinerdd)."
            )
    model_base = materialize_model_base(model_base)
    _prime_and_freeze_hf_cache(model_base)
    base_model = _load_model(model_base=model_base, local_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data = load_dataset(str(train_path))
    filtered_data = [sample for sample in raw_data if sample["ner"]]
    dataset = split_long_sentences(
        filtered_data,
        max_length=config.max_length,
        overlap=config.overlap,
    )

    if not dataset:
        raise ValueError("Dataset is empty after preprocessing.")

    entity_labels = sorted({label for sample in dataset for _, _, label in sample["ner"]})
    if not entity_labels:
        raise ValueError("No entity labels found after preprocessing.")

    groups = _extract_groups(dataset)
    outer_splitter = _effective_group_kfold(
        groups=groups,
        requested_splits=config.n_splits,
        stage_name="Outer CV",
    )

    has_best_overall_model = False
    best_overall_score = -1.0
    best_model_dir = output_dir / "best_overall_gliner_model"

    results_path = output_dir / config.results_file
    results_json_path = output_dir / config.results_json_file
    with open(results_path, "w", encoding="utf-8") as handle:
        handle.write("Nested Cross-Validation Results\n\n")
        handle.write(f"Run started (UTC): {started_at_iso}\n")
        handle.write("-" * 40 + "\n")

    outer_reports = []
    for fold, (trainval_idx, test_idx) in enumerate(
        outer_splitter.split(dataset, groups=groups),
        start=1,
    ):
        LOGGER.info("===== Outer Fold %s/%s =====", fold, outer_splitter.n_splits)

        trainval_data = _subset_by_indices(dataset, trainval_idx)
        test_data = _subset_by_indices(dataset, test_idx)
        trainval_groups = groups[trainval_idx]

        (
            best_lr,
            best_wd,
            best_threshold,
            best_inner_score,
            trial_reports,
        ) = _run_inner_search(
            trainval_data=trainval_data,
            trainval_groups=trainval_groups,
            config=config,
            base_model=base_model,
            entity_labels=entity_labels,
            device=device,
            fold=fold,
            output_dir=output_dir,
        )

        if best_lr is None or best_wd is None or best_threshold is None:
            LOGGER.warning(
                "No valid trial result was produced for outer fold %s.",
                fold,
            )
            outer_reports.append(
                {
                    "outer_fold": fold,
                    "best_params": None,
                    "selected_threshold": None,
                    "inner_best_score": None,
                    "test_f1": None,
                    "seen_entity_test_f1": None,
                    "unseen_entity_test_f1": None,
                    "seen_entity_gold_support": None,
                    "unseen_entity_gold_support": None,
                    "trials": trial_reports,
                }
            )
            continue

        LOGGER.info(
            "Best params for outer fold %s: LR=%.7f | WD=%.6f | THRESH=%s | Inner Mean F1=%.4f",
            fold,
            best_lr,
            best_wd,
            best_threshold,
            best_inner_score,
        )

        refit_train_data, refit_val_data = _build_refit_split(
            trainval_data=trainval_data,
            trainval_groups=trainval_groups,
            refit_val_size=config.refit_val_size,
            seed=config.seed + fold,
        )
        model, refit_history = _run_single_training(
            base_model=base_model,
            train_data=refit_train_data,
            val_data=refit_val_data,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            lr=best_lr,
            weight_decay=best_wd,
            patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
            entity_labels=entity_labels,
            device=device,
        )
        if model is None:
            LOGGER.warning("Refit stage produced an empty DataLoader for outer fold %s.", fold)
            outer_reports.append(
                {
                    "outer_fold": fold,
                    "best_params": {"lr": best_lr, "weight_decay": best_wd},
                    "selected_threshold": best_threshold,
                    "inner_best_score": best_inner_score,
                    "test_f1": None,
                    "seen_entity_test_f1": None,
                    "unseen_entity_test_f1": None,
                    "seen_entity_gold_support": None,
                    "unseen_entity_gold_support": None,
                    "trials": trial_reports,
                }
            )
            continue
        save_loss_plot(
            training_losses=refit_history["training_losses"],
            validation_losses=refit_history["validation_losses"],
            output_dir=output_dir,
            fold=fold,
            trial="refit",
            lr=best_lr,
            weight_decay=best_wd,
        )

        test_processed = _prepare_char_offsets(test_data)
        final_f1_test = compute_f1_by_threshold(
            model,
            test_processed,
            best_threshold,
            entity_labels,
        )
        seen_entity_keys = _build_seen_entity_keys(refit_train_data)
        seen_unseen_breakdown = _compute_seen_unseen_breakdown(
            model=model,
            dataset=test_processed,
            threshold=best_threshold,
            entity_labels=entity_labels,
            seen_entity_keys=seen_entity_keys,
        )
        LOGGER.info("Final test F1 for outer fold %s: %.4f", fold, final_f1_test)
        LOGGER.info(
            "Outer fold %s breakdown | Seen-entity F1=%s (support=%s) | Unseen-entity F1=%s (support=%s)",
            fold,
            format_param(seen_unseen_breakdown["seen_f1"], ".4f"),
            seen_unseen_breakdown["seen_gold_support"],
            format_param(seen_unseen_breakdown["unseen_f1"], ".4f"),
            seen_unseen_breakdown["unseen_gold_support"],
        )

        with open(results_path, "a", encoding="utf-8") as handle:
            handle.write(f"Outer Fold {fold}\n")
            handle.write(
                f"Best params (LR, WD): {format_param(best_lr, '.7f')}, {format_param(best_wd, '.6f')}\n"
            )
            handle.write(f"Selected threshold: {best_threshold}\n")
            handle.write(f"Inner best score: {best_inner_score:.4f}\n")
            handle.write(f"Test F1: {final_f1_test:.4f}\n")
            handle.write(
                "Seen-entity Test F1: "
                f"{format_param(seen_unseen_breakdown['seen_f1'], '.4f')} "
                f"(gold support: {seen_unseen_breakdown['seen_gold_support']})\n"
            )
            handle.write(
                "Unseen-entity Test F1: "
                f"{format_param(seen_unseen_breakdown['unseen_f1'], '.4f')} "
                f"(gold support: {seen_unseen_breakdown['unseen_gold_support']})\n"
            )
            handle.write("Tried hyperparameter combinations:\n")
            for trial in trial_reports:
                threshold_scores_str = ", ".join(
                    f"{threshold}:{format_param(score, '.4f')}"
                    for threshold, score in trial["mean_threshold_scores"].items()
                )
                handle.write(
                    f"  - Trial {trial['trial']}: "
                    f"LR={format_param(trial['lr'], '.7f')}, "
                    f"WD={format_param(trial['weight_decay'], '.6f')}, "
                    f"best_threshold={trial['best_threshold']}, "
                    f"best_score={format_param(trial['best_score'], '.4f')}, "
                    f"mean_threshold_scores=[{threshold_scores_str}]\n"
                )
            handle.write("-" * 40 + "\n")

        if final_f1_test > best_overall_score:
            best_overall_score = final_f1_test
            has_best_overall_model = True
            model.save_pretrained(best_model_dir)
            LOGGER.info("New best overall model saved (F1=%.4f): %s", best_overall_score, best_model_dir)

        outer_reports.append(
            {
                "outer_fold": fold,
                "best_params": {"lr": best_lr, "weight_decay": best_wd},
                "selected_threshold": best_threshold,
                "inner_best_score": best_inner_score,
                "test_f1": final_f1_test,
                "seen_entity_test_f1": seen_unseen_breakdown["seen_f1"],
                "unseen_entity_test_f1": seen_unseen_breakdown["unseen_f1"],
                "seen_entity_gold_support": seen_unseen_breakdown["seen_gold_support"],
                "unseen_entity_gold_support": seen_unseen_breakdown["unseen_gold_support"],
                "trials": trial_reports,
            }
        )
        del model
        clear_cuda_cache()

    valid_test_scores = [report["test_f1"] for report in outer_reports if report["test_f1"] is not None]
    valid_seen_scores = [
        report["seen_entity_test_f1"]
        for report in outer_reports
        if report.get("seen_entity_test_f1") is not None
    ]
    valid_unseen_scores = [
        report["unseen_entity_test_f1"]
        for report in outer_reports
        if report.get("unseen_entity_test_f1") is not None
    ]
    mean_test_f1 = float(np.mean(valid_test_scores)) if valid_test_scores else None
    std_test_f1 = float(np.std(valid_test_scores)) if valid_test_scores else None
    mean_seen_test_f1 = float(np.mean(valid_seen_scores)) if valid_seen_scores else None
    mean_unseen_test_f1 = float(np.mean(valid_unseen_scores)) if valid_unseen_scores else None
    duration_seconds = perf_counter() - experiment_timer
    finished_at_utc = datetime.now(timezone.utc)
    finished_at_iso = finished_at_utc.isoformat()
    duration_hms = format_duration(duration_seconds)

    with open(results_path, "a", encoding="utf-8") as handle:
        handle.write("\nRun summary\n")
        handle.write(f"Run finished (UTC): {finished_at_iso}\n")
        handle.write(f"Total runtime (HH:MM:SS): {duration_hms}\n")
        handle.write(f"Total runtime (seconds): {duration_seconds:.2f}\n")

    report_payload = {
        "config": {
            "train_path": str(train_path),
            "model_base": str(model_base),
            "n_splits": outer_splitter.n_splits,
            "n_inner_splits": config.n_inner_splits,
            "search_mode": config.search_mode,
            "num_trials": config.num_trials,
            "thresholds": config.thresholds,
            "lr_values": config.lr_values,
            "weight_decay_values": config.weight_decay_values,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
        },
        "outer_folds": outer_reports,
        "summary": {
            "best_overall_test_f1": best_overall_score if has_best_overall_model else None,
            "mean_test_f1": mean_test_f1,
            "std_test_f1": std_test_f1,
            "mean_seen_entity_test_f1": mean_seen_test_f1,
            "mean_unseen_entity_test_f1": mean_unseen_test_f1,
            "best_model_dir": str(best_model_dir) if has_best_overall_model else None,
            "results_file": str(results_path),
            "started_at_utc": started_at_iso,
            "finished_at_utc": finished_at_iso,
            "runtime_seconds": duration_seconds,
            "runtime_hms": duration_hms,
            "tokenizer_regex_warning_seen": TOKENIZER_REGEX_WARNING_SEEN,
        },
    }
    with open(results_json_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    if has_best_overall_model:
        LOGGER.info("Training finished. Best overall test F1: %.4f", best_overall_score)
        LOGGER.info("Best model directory: %s", best_model_dir)
    else:
        LOGGER.warning("Training finished but no best model was saved.")
    LOGGER.info("Detailed fold metrics written to: %s", results_path)
    LOGGER.info("Structured nested CV report written to: %s", results_json_path)
    LOGGER.info(
        "Total training runtime: %s (%0.2f seconds)",
        duration_hms,
        duration_seconds,
    )
    if TOKENIZER_REGEX_WARNING_SEEN:
        LOGGER.warning(
            "Tokenizer regex warning was detected during model loading. "
            "This was suppressed in stdout and recorded in nested_cv_results.json."
        )
