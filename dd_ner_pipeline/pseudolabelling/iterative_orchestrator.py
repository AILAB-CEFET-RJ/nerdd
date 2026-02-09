import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from gliner_train.paths import resolve_path
from pseudolabelling.compute_record_score import run_compute_record_score
from pseudolabelling.config import ContextBoostConfig, CorpusPredictConfig
from pseudolabelling.context_boost import run_context_boost
from pseudolabelling.evaluate_refit_pipeline import run_evaluate_refit
from pseudolabelling.prepare_next_iteration_pipeline import run_prepare_next_iteration
from pseudolabelling.refit_cli import RefitConfig
from pseudolabelling.refit_pipeline import run_refit
from pseudolabelling.split_by_score import run_split

LOGGER = logging.getLogger(__name__)


def _csv_list(raw_value):
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one comma-separated value must be provided.")
    return values


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _run_corpus_prediction(config, script_path):
    from pseudolabelling.pipeline import run_corpus_prediction

    return run_corpus_prediction(config, script_path=script_path)


@dataclass
class IterativeCycleConfig:
    run_dir: str = "./iterative_cycle_run"
    model_path: str = "best_overall_gliner_model"
    input_jsonl: str = "dd_corpus_large.json"
    labels: list[str] = field(default_factory=lambda: ["Person", "Location", "Organization"])
    text_fields: list[str] = field(
        default_factory=lambda: [
            "assunto",
            "relato",
            "bairroLocal",
            "logradouroLocal",
            "cidadeLocal",
            "pontodeReferenciaLocal",
        ]
    )
    prediction_batch_size: int = 4
    prediction_max_tokens: int = 384
    prediction_threshold: float = 0.0

    use_calibration: bool = False
    calibration_method: str = "temperature"
    calibration_label_source: str = "score-threshold"
    calibration_csv: str = "../data/comparacao_calibracao.csv"
    calibration_positive_threshold: float = 0.9
    calibration_score_field: str = "score"
    calibration_output_score_field: str = "score_calibrated"

    context_boost_enabled: bool = True
    context_boost_factor: float = 1.2
    context_boost_scope: str = "all-entities"
    context_match_policy: str = "any-metadata-in-text"
    context_base_score_field: str = "score"
    context_output_score_field: str = "score_context_boosted"
    context_output_record_score_field: str = "record_score_context_boosted"

    record_score_field: str = "score_context_boosted"
    record_score_output_field: str = "record_score"
    record_score_aggregation: str = "mean"
    record_score_empty_policy: str = "zero"

    split_threshold: float = 0.80
    split_operator: str = "ge"
    split_missing_policy: str = "discard"
    split_fallback_score_field: str = "score_relato_confianca"

    refit_output_model_dir: str = ""
    refit_base_model: str = ""
    refit_epochs: int = 10
    refit_patience: int = 3
    refit_batch_size: int = 8
    refit_lr: float = 3e-5
    refit_weight_decay: float = 0.01
    refit_val_ratio: float = 0.1
    refit_seed: int = 42
    refit_num_workers: int = 2

    evaluate_refit: bool = False
    eval_gt_jsonl: str = ""
    eval_prediction_threshold: float = 0.05
    eval_batch_size: int = 4
    eval_max_tokens: int = 384

    prepare_next_iteration: bool = False
    prepare_keep_fields: list[str] = field(
        default_factory=lambda: [
            "assunto",
            "relato",
            "logradouroLocal",
            "bairroLocal",
            "cidadeLocal",
            "pontodeReferenciaLocal",
        ]
    )
    prepare_required_fields: list[str] = field(default_factory=lambda: ["relato"])
    prepare_deduplicate_by: list[str] = field(default_factory=lambda: ["relato", "bairroLocal"])


def run_iterative_cycle(config: IterativeCycleConfig, script_path: str):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    run_dir = resolve_path(script_dir, config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_jsonl = run_dir / "01_predictions.jsonl"
    predictions_stats = run_dir / "01_predictions_stats.json"
    current_entities_jsonl = predictions_jsonl

    LOGGER.info("Step 1/7: Predict entities on large corpus")
    prediction_cfg = CorpusPredictConfig(
        model_path=config.model_path,
        input_jsonl=config.input_jsonl,
        output_jsonl=str(predictions_jsonl),
        stats_json=str(predictions_stats),
        labels=config.labels,
        text_fields=config.text_fields,
        batch_size=config.prediction_batch_size,
        max_tokens=config.prediction_max_tokens,
        score_threshold=config.prediction_threshold,
        keep_inference_text=False,
    )
    _run_corpus_prediction(prediction_cfg, script_path=script_path)

    calibration_stats = None
    if config.use_calibration:
        LOGGER.info("Step 2/7: Calibrate entity scores (%s)", config.calibration_method)
        calibrated_jsonl = run_dir / "02_calibrated.jsonl"
        calibration_stats = run_dir / "02_calibration_stats.json"
        try:
            from calibration.config import CalibrationConfig
            from calibration.pipeline import run_calibration
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Calibration module import failed. Verify dd_ner_pipeline/calibration package."
            ) from exc

        calibration_cfg = CalibrationConfig(
            method=config.calibration_method,
            input_jsonl=str(current_entities_jsonl),
            output_jsonl=str(calibrated_jsonl),
            stats_json=str(calibration_stats),
            score_field=config.calibration_score_field,
            output_score_field=config.calibration_output_score_field,
            labels=config.labels,
            label_source=config.calibration_label_source,
            calibration_csv=config.calibration_csv,
            positive_threshold=config.calibration_positive_threshold,
        )
        run_calibration(calibration_cfg, script_path=script_path)
        current_entities_jsonl = calibrated_jsonl

    context_input = current_entities_jsonl
    context_stats = None
    if config.context_boost_enabled:
        LOGGER.info("Step 3/7: Apply metadata-aware context boost")
        context_jsonl = run_dir / "03_context_boosted.jsonl"
        context_stats = run_dir / "03_context_boost_stats.json"
        context_cfg = ContextBoostConfig(
            input_jsonl=str(context_input),
            output_jsonl=str(context_jsonl),
            stats_json=str(context_stats),
            base_score_field=config.context_base_score_field,
            output_score_field=config.context_output_score_field,
            output_record_score_field=config.context_output_record_score_field,
            boost_factor=config.context_boost_factor,
            boost_scope=config.context_boost_scope,
            match_policy=config.context_match_policy,
        )
        run_context_boost(context_cfg, script_path=script_path)
        context_input = context_jsonl

    LOGGER.info("Step 4/7: Compute record-level score")
    scored_jsonl = run_dir / "04_scored.jsonl"
    scored_stats = run_dir / "04_score_stats.json"
    run_compute_record_score(
        input_jsonl=str(context_input),
        output_jsonl=str(scored_jsonl),
        stats_json=str(scored_stats),
        score_field=config.record_score_field,
        output_field=config.record_score_output_field,
        legacy_field_alias="score_relato",
        aggregation=config.record_score_aggregation,
        empty_entities_policy=config.record_score_empty_policy,
        script_path=script_path,
    )

    LOGGER.info("Step 5/7: Split kept/discarded pseudolabels")
    split_dir = run_dir / "05_split"
    run_split(
        input_jsonl=str(scored_jsonl),
        out_dir=str(split_dir),
        score_field=config.record_score_output_field,
        threshold=config.split_threshold,
        operator=config.split_operator,
        fallback_score_field=config.split_fallback_score_field,
        missing_policy=config.split_missing_policy,
        legacy_filenames=True,
        script_path=script_path,
    )

    LOGGER.info("Step 6/7: Refit model on kept records")
    refit_model_dir = (
        run_dir / "06_refit_model" if not config.refit_output_model_dir else resolve_path(script_dir, config.refit_output_model_dir)
    )
    refit_stats = run_dir / "06_refit_stats.json"
    train_manifest = run_dir / "06_train_manifest.jsonl"
    val_manifest = run_dir / "06_val_manifest.jsonl"
    refit_cfg = RefitConfig(
        input_path=str(split_dir),
        output_model_dir=str(refit_model_dir),
        stats_json=str(refit_stats),
        train_manifest_jsonl=str(train_manifest),
        val_manifest_jsonl=str(val_manifest),
        base_model=config.refit_base_model,
        epochs=config.refit_epochs,
        patience=config.refit_patience,
        batch_size=config.refit_batch_size,
        lr=config.refit_lr,
        weight_decay=config.refit_weight_decay,
        val_ratio=config.refit_val_ratio,
        seed=config.refit_seed,
        allowed_labels=config.labels,
        num_workers=config.refit_num_workers,
    )
    run_refit(refit_cfg, script_path=script_path)

    eval_dir = None
    if config.evaluate_refit:
        if not config.eval_gt_jsonl:
            raise ValueError("--eval-gt-jsonl is required when --evaluate-refit is enabled.")
        LOGGER.info("Step 7/7: Evaluate refit model")
        eval_dir = run_dir / "07_eval_refit"
        eval_cfg = {
            "model_path": str(refit_model_dir),
            "gt_jsonl": config.eval_gt_jsonl,
            "out_dir": str(eval_dir),
            "labels": config.labels,
            "prediction_threshold": config.eval_prediction_threshold,
            "batch_size": config.eval_batch_size,
            "max_tokens": config.eval_max_tokens,
            "match_mode": "exact",
        }
        run_evaluate_refit(eval_cfg, script_path=script_path)

    next_iter_stats = None
    if config.prepare_next_iteration:
        LOGGER.info("Extra step: Prepare discarded records for next iteration")
        next_iter_dir = run_dir / "08_next_iteration_input"
        next_iter_stats = next_iter_dir / "prepare_next_iteration_stats.json"
        prepare_cfg = {
            "input_jsonl": str(split_dir / "discarded.jsonl"),
            "input_glob": "",
            "output_jsonl": str(next_iter_dir / "discarded_next_iter.jsonl"),
            "out_dir": str(next_iter_dir),
            "output_suffix": "_next_iter",
            "keep_fields": config.prepare_keep_fields,
            "required_fields": config.prepare_required_fields,
            "fill_missing_with": "",
            "coerce_non_string": "stringify",
            "drop_empty_relato": True,
            "deduplicate_by": config.prepare_deduplicate_by,
            "allow_json": False,
            "stats_json": str(next_iter_stats),
        }
        run_prepare_next_iteration(prepare_cfg, script_path=script_path)

    finished_at = datetime.now(timezone.utc).isoformat()
    runtime_seconds = perf_counter() - timer
    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": {
            "run_dir": str(run_dir.resolve()),
            "model_path": config.model_path,
            "input_jsonl": config.input_jsonl,
            "labels": config.labels,
            "use_calibration": config.use_calibration,
            "evaluate_refit": config.evaluate_refit,
            "prepare_next_iteration": config.prepare_next_iteration,
        },
        "artifacts": {
            "predictions_jsonl": str(predictions_jsonl.resolve()),
            "predictions_stats": str(predictions_stats.resolve()),
            "calibration_stats": str(calibration_stats.resolve()) if calibration_stats else None,
            "context_boost_stats": str(context_stats.resolve()) if context_stats else None,
            "scored_jsonl": str(scored_jsonl.resolve()),
            "split_dir": str(split_dir.resolve()),
            "refit_model_dir": str(refit_model_dir.resolve()),
            "refit_stats": str(refit_stats.resolve()),
            "eval_dir": str(eval_dir.resolve()) if eval_dir else None,
            "next_iteration_stats": str(next_iter_stats.resolve()) if next_iter_stats else None,
        },
    }
    summary_path = run_dir / "cycle_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Iterative cycle completed. Summary: %s", summary_path)


def parse_args():
    import argparse

    defaults = IterativeCycleConfig()
    parser = argparse.ArgumentParser(description="Run the full pseudolabelling iterative cycle in one command")
    parser.add_argument("--run-dir", default=defaults.run_dir)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--input-jsonl", default=defaults.input_jsonl)
    parser.add_argument("--labels", default=",".join(defaults.labels))
    parser.add_argument("--text-fields", default=",".join(defaults.text_fields))
    parser.add_argument("--prediction-batch-size", type=int, default=defaults.prediction_batch_size)
    parser.add_argument("--prediction-max-tokens", type=int, default=defaults.prediction_max_tokens)
    parser.add_argument("--prediction-threshold", type=float, default=defaults.prediction_threshold)

    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument(
        "--calibration-method",
        choices=["temperature", "temperature-per-class", "isotonic"],
        default=defaults.calibration_method,
    )
    parser.add_argument(
        "--calibration-label-source",
        choices=["score-threshold", "quantile-bands", "calibration-csv"],
        default=defaults.calibration_label_source,
    )
    parser.add_argument("--calibration-csv", default=defaults.calibration_csv)
    parser.add_argument("--calibration-positive-threshold", type=float, default=defaults.calibration_positive_threshold)

    parser.add_argument("--disable-context-boost", action="store_true")
    parser.add_argument("--context-boost-factor", type=float, default=defaults.context_boost_factor)
    parser.add_argument(
        "--context-boost-scope",
        choices=["all-entities", "location-only", "matched-only"],
        default=defaults.context_boost_scope,
    )
    parser.add_argument(
        "--context-match-policy",
        choices=["any-metadata-in-text", "entity-metadata-overlap"],
        default=defaults.context_match_policy,
    )
    parser.add_argument("--context-base-score-field", default=defaults.context_base_score_field)
    parser.add_argument("--context-output-score-field", default=defaults.context_output_score_field)
    parser.add_argument("--record-score-field", default=defaults.record_score_field)
    parser.add_argument(
        "--record-score-aggregation",
        choices=["mean", "max", "median", "p75"],
        default=defaults.record_score_aggregation,
    )
    parser.add_argument(
        "--record-score-empty-policy",
        choices=["zero", "null", "error"],
        default=defaults.record_score_empty_policy,
    )
    parser.add_argument("--split-threshold", type=float, default=defaults.split_threshold)
    parser.add_argument("--split-operator", choices=["ge", "gt", "le", "lt"], default=defaults.split_operator)
    parser.add_argument("--split-missing-policy", choices=["discard", "zero", "error"], default=defaults.split_missing_policy)
    parser.add_argument("--split-fallback-score-field", default=defaults.split_fallback_score_field)

    parser.add_argument("--refit-output-model-dir", default=defaults.refit_output_model_dir)
    parser.add_argument("--refit-base-model", default=defaults.refit_base_model)
    parser.add_argument("--refit-epochs", type=int, default=defaults.refit_epochs)
    parser.add_argument("--refit-patience", type=int, default=defaults.refit_patience)
    parser.add_argument("--refit-batch-size", type=int, default=defaults.refit_batch_size)
    parser.add_argument("--refit-lr", type=float, default=defaults.refit_lr)
    parser.add_argument("--refit-weight-decay", type=float, default=defaults.refit_weight_decay)
    parser.add_argument("--refit-val-ratio", type=float, default=defaults.refit_val_ratio)
    parser.add_argument("--refit-seed", type=int, default=defaults.refit_seed)
    parser.add_argument("--refit-num-workers", type=int, default=defaults.refit_num_workers)

    parser.add_argument("--evaluate-refit", action="store_true")
    parser.add_argument("--eval-gt-jsonl", default=defaults.eval_gt_jsonl)
    parser.add_argument("--eval-prediction-threshold", type=float, default=defaults.eval_prediction_threshold)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    parser.add_argument("--eval-max-tokens", type=int, default=defaults.eval_max_tokens)

    parser.add_argument("--prepare-next-iteration", action="store_true")
    parser.add_argument("--prepare-keep-fields", default=",".join(defaults.prepare_keep_fields))
    parser.add_argument("--prepare-required-fields", default=",".join(defaults.prepare_required_fields))
    parser.add_argument("--prepare-deduplicate-by", default=",".join(defaults.prepare_deduplicate_by))

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    return IterativeCycleConfig(
        run_dir=args.run_dir,
        model_path=args.model_path,
        input_jsonl=args.input_jsonl,
        labels=_csv_list(args.labels),
        text_fields=_csv_list(args.text_fields),
        prediction_batch_size=args.prediction_batch_size,
        prediction_max_tokens=args.prediction_max_tokens,
        prediction_threshold=args.prediction_threshold,
        use_calibration=args.use_calibration,
        calibration_method=args.calibration_method,
        calibration_label_source=args.calibration_label_source,
        calibration_csv=args.calibration_csv,
        calibration_positive_threshold=args.calibration_positive_threshold,
        context_boost_enabled=(not args.disable_context_boost),
        context_boost_factor=args.context_boost_factor,
        context_boost_scope=args.context_boost_scope,
        context_match_policy=args.context_match_policy,
        context_base_score_field=args.context_base_score_field,
        context_output_score_field=args.context_output_score_field,
        record_score_field=args.record_score_field,
        record_score_aggregation=args.record_score_aggregation,
        record_score_empty_policy=args.record_score_empty_policy,
        split_threshold=args.split_threshold,
        split_operator=args.split_operator,
        split_missing_policy=args.split_missing_policy,
        split_fallback_score_field=args.split_fallback_score_field,
        refit_output_model_dir=args.refit_output_model_dir,
        refit_base_model=args.refit_base_model,
        refit_epochs=args.refit_epochs,
        refit_patience=args.refit_patience,
        refit_batch_size=args.refit_batch_size,
        refit_lr=args.refit_lr,
        refit_weight_decay=args.refit_weight_decay,
        refit_val_ratio=args.refit_val_ratio,
        refit_seed=args.refit_seed,
        refit_num_workers=args.refit_num_workers,
        evaluate_refit=args.evaluate_refit,
        eval_gt_jsonl=args.eval_gt_jsonl,
        eval_prediction_threshold=args.eval_prediction_threshold,
        eval_batch_size=args.eval_batch_size,
        eval_max_tokens=args.eval_max_tokens,
        prepare_next_iteration=args.prepare_next_iteration,
        prepare_keep_fields=_csv_list(args.prepare_keep_fields),
        prepare_required_fields=_csv_list(args.prepare_required_fields),
        prepare_deduplicate_by=_csv_list(args.prepare_deduplicate_by),
    )
