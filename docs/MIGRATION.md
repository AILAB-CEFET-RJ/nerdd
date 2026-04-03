# Migration Guide (Gustavo)

## Goal
This document summarizes the refactor from legacy scripts to the current modular training/evaluation pipeline.

## Old to New Entrypoints
- Training:
  - Old: `GliNER - Treino - Nested KFold BayesSearch v4.py`
  - New: `python3 -m base_model_training.train_nested_kfold`
- Evaluation:
  - Old: `GliNER - Resultado - v3.py`
  - New: `base_model_training/evaluate_gliner.py`
- Large corpus prediction (inference-only):
  - Old: `GliNER - PseudoLabel v3.py`
  - New: `pseudolabelling/generate_corpus_predictions.py` or `pseudolabelling/run_iterative_cycle.py`
- Score calibration:
  - Old:
    - `GliNER -  Temperature Scaling.py`
    - `GliNER -  Temperature Scaling - Classe.py`
    - `GliNER - Isolation Regression.py`
  - New:
    - `calibration/fit_calibrator.py`
    - `calibration/apply_calibrator.py`
- Sanity JSONL -> BIO CSV conversion:
  - Old: `GliNER - Conversao Arquivo Sanidade.py`
  - New: `tools/convert_sanity_jsonl_to_bio_csv.py`

## Key Behavioral Changes
- Nested CV implementation was hardened:
  - outer and inner splits use group-aware splitting by `sample_id`.
  - avoids leakage between chunks from the same original report.
- Hyperparameter search became explicit and configurable from CLI:
  - `--search-mode`, `--num-trials`, `--backbone-lr-values`, `--ner-lr-values`, `--weight-decay-values`.
- Full run reporting improved:
  - human report: `nested_cv_results.txt`
  - structured report: `nested_cv_results.json`
  - runtime metadata (start/end/runtime seconds + HH:MM:SS)
  - fold-level seen vs unseen entity metrics.
- Training now saves loss curves for audit:
  - one plot per inner trial/fold
  - one plot for each refit stage.
- Model loading/caching was adjusted for more stable offline/repeated runs.
- Calibration is now treated as a reusable artifact attached to the base model:
  - fit on labeled holdout predictions
  - persist a calibrator JSON
  - reuse it during large-corpus prediction

## Dataset Convention
- Dataset is JSONL (one JSON object per line), even when extension is `.json`.
- Current default training dataset: `../data/dd_corpus_small_train.json`.
- Current default calibration dataset: `../data/dd_corpus_small_calibration.json`.
- Current default final evaluation dataset: `../data/dd_corpus_small_test.json`.

## Migration Checklist
1. Use only `python3 -m base_model_training.train_nested_kfold` and `base_model_training/evaluate_gliner.py` for training/evaluation.
2. Use `nested_cv_results.json` as source of truth for programmatic analysis.
3. Fit calibration from a labeled holdout subset, not from unlabeled pseudolabelling outputs.
4. Keep smoke-test command for local sanity checks before server runs.
5. For publications/reports, describe whether evaluation is nested-CV-only or includes a separate holdout.
