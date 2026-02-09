# Pipeline Overview

## Subpipeline Boundaries

1. **Training + Evaluation Subpipeline**
- Scope: model development, selection, and validation.
- Entrypoints: `train_nested_kfold.py`, `gliner_train/evaluate_gliner.py`.
- Main package: `gliner_train/`.
- Primary output: trained model directory (`best_overall_gliner_model/`) and training/evaluation reports.

2. **Pseudolabelling Subpipeline**
- Scope: large-corpus entity prediction plus metadata-aware context boost.
- Entrypoints: `pseudolabelling/run_iterative_cycle.py` (orchestrator), `pseudolabelling/generate_corpus_predictions.py`, `pseudolabelling/apply_context_boost.py`, `pseudolabelling/compute_record_scores.py`, `pseudolabelling/split_pseudolabels.py`, `pseudolabelling/refit_model.py`, `pseudolabelling/evaluate_refit.py`, `pseudolabelling/prepare_next_iteration.py`.
- Main package: `pseudolabelling/`.
- Primary outputs: predicted-entities JSONL, context-boosted JSONL, record-scored JSONL, split kept/discarded JSONL, refit model directory, evaluation artifacts, and next-iteration input JSONL.

3. **Calibration Subpipeline**
- Scope: post-process and calibrate confidence scores from pseudolabel outputs.
- Entrypoint: `calibration/run_calibration.py`.
- Main package: `calibration/`.
- Primary output: calibrated JSONL + calibration stats JSON.

## Training Flow (`train_nested_kfold.py`)
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

## Evaluation Flow (`gliner_train/evaluate_gliner.py`)
1. Load trained model.
2. Predict entities on ground-truth JSONL.
3. Calibrate thresholds by label.
4. Generate classification report and per-class TP/FP/FN summary.

## Pseudolabelling Flow (`pseudolabelling/generate_corpus_predictions.py`)
1. Load trained model from nested-CV output.
2. Read large input corpus (JSONL).
3. Build inference text from configured source fields.
4. Chunk long texts by tokenizer length and run entity prediction.
5. Merge chunk-level offsets into full-text offsets.
6. Save predicted entities JSONL and execution stats JSON.

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
2. Normalize and validate training records (text + entity spans + labels).
3. Build train/validation split (or use external validation JSONL).
4. Load base model and run iterative refit training.
5. Save refit model plus run manifests/stats.

## Refit Evaluation Flow (`pseudolabelling/evaluate_refit.py`)
1. Read labeled ground-truth JSONL (`text` + `spans`).
2. Validate schema and run model inference.
3. Compare predicted spans to gold spans using exact match.
4. Compute per-label metrics plus micro/macro F1.
5. Save predictions, report, metrics, and run stats.

## Next Iteration Prep Flow (`pseudolabelling/prepare_next_iteration.py`)
1. Read discarded JSONL files (single file or glob batch mode).
2. Project only configured fields required by next inference iteration.
3. Enforce required-field checks and optional deduplication.
4. Write cleaned next-iteration JSONL outputs.
5. Save per-file mapping and drop-reason stats.

## Calibration Flow (`calibration/run_calibration.py`)
1. Load pseudolabel JSONL with entity scores.
2. Build pseudo-targets for calibration fit (score-threshold or quantile-bands).
3. Fit selected method (`temperature`, `temperature-per-class`, `isotonic`).
4. Apply calibrated scores to all entities and persist output JSONL.
5. Save calibration diagnostics and method parameters to stats JSON.

## Main Artifacts
- Artifact root (recommended): `dd_ner_pipeline/artifacts/`
- Training outputs: `dd_ner_pipeline/artifacts/gliner_train/`
- Pseudolabelling outputs: `dd_ner_pipeline/artifacts/pseudolabelling/`
- Calibration outputs: `dd_ner_pipeline/artifacts/calibration/`

Examples:
- `artifacts/gliner_train/smoke/run_nested_tiny/best_overall_gliner_model/`
- `artifacts/gliner_train/experiments/run_batch16/nested_cv_results.json`
- `artifacts/gliner_train/experiments/run_batch16/eval/report.txt`
- `artifacts/pseudolabelling/iter01/01_predictions.jsonl`
- `artifacts/pseudolabelling/iter01/04_split/kept.jsonl`
- `artifacts/pseudolabelling/iter01/05_refit_model/`
- `artifacts/calibration/iter01/01_calibrated.jsonl`
- `artifacts/calibration/iter01/01_calibration_stats.json`

## Metrics Added for Audit
- `seen_entity_test_f1`: fold test F1 on entity mentions seen during fold training.
- `unseen_entity_test_f1`: fold test F1 on unseen mentions.
- Runtime metadata in `summary`.
