# Pipeline Overview

## Subpipeline Boundaries

1. **Training + Evaluation Subpipeline**
- Scope: model development, selection, and validation.
- Entrypoints: `python3 -m base_model_training.train_nested_kfold`, `base_model_training/evaluate_gliner.py`.
- Main package: `base_model_training/`.
- Primary output: trained model directory (`best_overall_gliner_model/`) and training/evaluation reports.

2. **Pseudolabelling Subpipeline**
- Scope: large-corpus entity prediction, score-based pseudolabel selection, refit, and paired base-vs-refit evaluation.
- Entrypoints: `pseudolabelling/run_iterative_cycle.py` (orchestrator), `pseudolabelling/generate_corpus_predictions.py`, `pseudolabelling/apply_context_boost.py`, `pseudolabelling/compute_record_scores.py`, `pseudolabelling/split_pseudolabels.py`, `pseudolabelling/refit_model.py`, `pseudolabelling/evaluate_refit.py`, `pseudolabelling/prepare_next_iteration.py`.
- Main package: `pseudolabelling/`.
- Primary outputs: predicted-entities JSONL, context-boosted JSONL, record-scored JSONL, split kept/discarded JSONL, refit model directory, base/refit evaluation artifacts, comparison JSON, and next-iteration input JSONL.

3. **Calibration Subpipeline**
- Scope: fit and apply reusable probability calibrators for entity confidence scores.
- Entrypoints: `calibration/fit_calibrator.py`, `calibration/apply_calibrator.py` (`calibration/run_calibration.py` remains as a legacy fit+apply path).
- Main package: `calibration/`.
- Primary outputs: calibrator artifact JSON, calibrated JSONL, and calibration stats JSON.

## Training Flow (`python3 -m base_model_training.train_nested_kfold`)
1. Parse CLI args into `TrainConfig`.
2. Resolve dataset/model paths.
3. Materialize/cache base model and initialize offline-friendly behavior.
4. Load JSONL dataset and preprocess:
   - tokenization,
   - token-span conversion,
   - long-sample chunking.
5. Run nested cross-validation:
   - outer folds: generalization estimate,
   - inner folds: hyperparameter selection.
6. Refit best params per outer fold and score on outer test split.
7. Save best overall model and reports.

## Evaluation Flow (`base_model_training/evaluate_gliner.py`)
1. Load trained model.
2. Predict entities on ground-truth JSONL.
3. Calibrate thresholds by label.
4. Generate classification report and per-class TP/FP/FN summary.

## Pseudolabelling Flow (`pseudolabelling/generate_corpus_predictions.py`)
1. Load trained model from nested-CV output.
2. Optionally load a persisted calibrator artifact.
3. Read large input corpus (JSONL).
4. Build inference text from configured source fields.
5. Chunk long texts by tokenizer length and run entity prediction.
6. Merge chunk-level offsets into full-text offsets.
7. Optionally write calibrated score fields alongside raw scores.
8. Save predicted entities JSONL and execution stats JSON.

## Context Boost Flow (`pseudolabelling/apply_context_boost.py`)
1. Read predicted-entities JSONL.
2. Select primary text field from configured priority (`relato` then `text` by default).
3. Detect metadata context matches (`logradouroLocal`, `bairroLocal`, `cidadeLocal`, `pontodeReferenciaLocal`).
4. Apply configurable confidence boost policy:
   - scope: `all-entities`, `location-only`, or `matched-only`,
   - policy: `any-metadata-in-text` or `entity-metadata-overlap`.
5. Write per-entity `score_context_boosted` and per-record `record_score_context_boosted`.
6. Save context-boost run stats JSON.

## Record Score Flow (`pseudolabelling/compute_record_scores.py`)
1. Read context-boosted (or calibrated) JSONL.
2. Extract entity-level scores (`score`, `score_context_boosted`, etc.).
3. Aggregate to record-level score (`mean`, `max`, `median`, `p75`).
4. Write record-level score field and run stats JSON.

## Split Flow (`pseudolabelling/split_pseudolabels.py`)
1. Read record-scored JSONL.
2. Resolve record score from primary/fallback score fields.
3. Apply threshold operator (`ge`, `gt`, `le`, `lt`).
4. Route each record to `kept` or `discarded`, with per-record split trace.
5. Write split outputs and summary JSON.

## Refit Flow (`pseudolabelling/refit_model.py`)
1. Read kept pseudolabel JSONL from split output.
2. Optionally read the original supervised training set (`JSON` array or `JSONL`).
3. Normalize and validate both sources into a shared training format.
4. Merge supervised rows plus pseudolabel rows, preferring supervised rows on duplicate text when deduplication is enabled.
5. Build train/validation split (or use external validation JSONL).
6. Load base model and run iterative refit training.
7. Save refit model plus run manifests/stats, including source breakdown.

## Refit Evaluation Flow (`pseudolabelling/evaluate_refit.py`)
1. Read labeled ground-truth JSONL (`text` + `spans`).
2. Validate schema and run model inference.
3. Compare predicted spans to gold spans using exact match.
4. Compute per-label metrics plus micro/macro F1.
5. Save predictions, report, metrics, and run stats.

## Base-vs-Refit Comparison In The Orchestrator
When `--evaluate-refit` is enabled in `pseudolabelling/run_iterative_cycle.py`, the orchestrator now:

1. evaluates the original base model on the configured holdout
2. evaluates the refit model on the same holdout
3. writes:
   - `07_eval_base/`
   - `08_eval_refit/`
   - `09_base_vs_refit_comparison.json`

The comparison JSON reports base, refit, and delta for:

- micro F1
- macro F1
- per-label precision/recall/F1/support

## Next Iteration Prep Flow (`pseudolabelling/prepare_next_iteration.py`)
1. Read discarded JSONL files (single file or glob batch mode).
2. Project only configured fields required by next inference iteration.
3. Enforce required-field checks and optional deduplication.
4. Write cleaned next-iteration JSONL outputs.
5. Save per-file mapping and drop-reason stats.

## Calibration Flow
1. Run the base model on a labeled calibration subset.
2. Build a calibration CSV with raw `Score`, binary correctness target, and ideally entity `Label`.
3. Fit a selected method (`temperature`, `temperature-per-class`, `isotonic`) using `calibration/fit_calibrator.py`.
4. Persist a reusable calibrator artifact JSON.
5. Apply that artifact either during large-corpus prediction or later with `calibration/apply_calibrator.py`.

## Main Artifacts
- Artifact root (recommended): `src/artifacts/`
- Training outputs: `src/artifacts/base_model_training/`
- Pseudolabelling outputs: `src/artifacts/pseudolabelling/`
- Calibration outputs: `src/artifacts/calibration/`

Examples:
- `artifacts/base_model_training/smoke/run_nested_tiny/best_overall_gliner_model/`
- `artifacts/base_model_training/experiments/run_batch16/nested_cv_results.json`
- `artifacts/base_model_training/experiments/run_batch16/eval/report.txt`
- `artifacts/pseudolabelling/iter01/01_predictions.jsonl`
- `artifacts/pseudolabelling/iter01/04_split/kept.jsonl`
- `artifacts/pseudolabelling/iter01/05_refit_model/`
- `artifacts/pseudolabelling/iter01/07_eval_base/metrics.json`
- `artifacts/pseudolabelling/iter01/08_eval_refit/metrics.json`
- `artifacts/pseudolabelling/iter01/09_base_vs_refit_comparison.json`
- `artifacts/calibration/base_model/calibrator.json`
- `artifacts/calibration/iter01/01_calibrated.jsonl`
- `artifacts/calibration/base_model/fit_stats.json`

## Metrics Added for Audit
- `seen_entity_test_f1`: fold test F1 on entity mentions seen during fold training.
- `unseen_entity_test_f1`: fold test F1 on unseen mentions.
- Runtime metadata in `summary`.
