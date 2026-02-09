# Migration Guide (Gustavo)

## Goal
This document summarizes the refactor from legacy scripts to the current modular training/evaluation pipeline.

## Old to New Entrypoints
- Training:
  - Old: `GliNER - Treino - Nested KFold BayesSearch v4.py`
  - New: `train_nested_kfold.py`
- Evaluation:
  - Old: `GliNER - Resultado - v3.py`
  - New: `gliner_train/evaluate_gliner.py`
- Large corpus prediction (inference-only):
  - Old: `GliNER - PseudoLabel v3.py`
  - New: `pseudolabelling/generate_corpus_predictions.py` or `pseudolabelling/run_iterative_cycle.py`
- Score calibration:
  - Old:
    - `GliNER -  Temperature Scaling.py`
    - `GliNER -  Temperature Scaling - Classe.py`
    - `GliNER - Isolation Regression.py`
  - New: `calibration/run_calibration.py`
- Sanity JSONL -> BIO CSV conversion:
  - Old: `GliNER - Conversao Arquivo Sanidade.py`
  - New: `tools/convert_sanity_jsonl_to_bio_csv.py`

## Key Behavioral Changes
- Nested CV implementation was hardened:
  - outer and inner splits use group-aware splitting by `sample_id`.
  - avoids leakage between chunks from the same original report.
- Hyperparameter search became explicit and configurable from CLI:
  - `--search-mode`, `--num-trials`, `--lr-values`, `--weight-decay-values`.
- Full run reporting improved:
  - human report: `nested_cv_results.txt`
  - structured report: `nested_cv_results.json`
  - runtime metadata (start/end/runtime seconds + HH:MM:SS)
  - fold-level seen vs unseen entity metrics.
- Training now saves loss curves for audit:
  - one plot per inner trial/fold
  - one plot for each refit stage.
- Model loading/caching was adjusted for more stable offline/repeated runs.

## Dataset Convention
- Dataset is JSONL (one JSON object per line), even when extension is `.json`.
- Current default training dataset: `../data/dd_corpus_small_train.json`.

## Migration Checklist
1. Use only `train_nested_kfold.py` and `gliner_train/evaluate_gliner.py` for training/evaluation.
2. Use `nested_cv_results.json` as source of truth for programmatic analysis.
3. Keep smoke-test command for local sanity checks before server runs.
4. For publications/reports, describe whether evaluation is nested-CV-only or includes a separate holdout.
