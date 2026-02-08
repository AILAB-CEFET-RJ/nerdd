# Pipeline Overview

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

## Evaluation Flow (`evaluate_gliner.py`)
1. Load trained model.
2. Predict entities on ground-truth JSONL.
3. Calibrate thresholds by label.
4. Generate classification report and per-class TP/FP/FN summary.

## Main Artifacts
- `best_overall_gliner_model/`: best model snapshot.
- `nested_cv_results.txt`: human-readable CV report.
- `nested_cv_results.json`: structured CV report (for scripts/analysis).
- `loss_fold*_trial*.png`: training/validation loss curves.

## Metrics Added for Audit
- `seen_entity_test_f1`: fold test F1 on entity mentions seen during fold training.
- `unseen_entity_test_f1`: fold test F1 on unseen mentions.
- Runtime metadata in `summary`.
