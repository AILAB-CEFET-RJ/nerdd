# Installation Guide

This repo is centered on the `src/` pipeline (training, pseudolabelling, and calibration).

## Prerequisites

- Git
- Python 3.11+
- Pip
- Optional: virtualenv or venv

## Clone

```bash
git clone https://github.com/MLRG-CEFET-RJ/nerdd.git
cd nerdd
```

## Project Setup

Requires Python 3.11+.

Dependencies live in `src/requirements.txt`.

```bash
cd src
pip install -r requirements.txt
python3 -m base_model_training.train_nested_kfold --train-path ../data/dd_corpus_small_train.json
python3 base_model_training/evaluate_gliner.py --model-path ./artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model --gt-jsonl ../data/dd_corpus_small_test_filtered.json
```

Notes:
- `pywin32` is Windows-only. It is already marked with a platform guard in the requirements file.
- For local smoke tests (CPU, low RAM), prefer offline cached execution:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m base_model_training.train_nested_kfold \
  --model-base urchade/gliner_small-v2.1 \
  --train-path ../data/dd_corpus_smoke_20.json \
  --batch-size 1 \
  --max-length 32 \
  --overlap 2 \
  --num-epochs 1 \
  --n-splits 2 \
  --n-inner-splits 2 \
  --search-mode grid \
  --lr-values 3e-5 \
  --weight-decay-values 0.01 \
  --thresholds 0.6 \
  --refit-val-size 0.5 \
  --output-dir ./artifacts/base_model_training/smoke/run_nested_tiny \
  --log-level INFO
```

- `nested_cv_results.json` now includes:
  - `seen_entity_test_f1`: F1 on test mentions whose `(label, surface)` appeared in fold training data.
  - `unseen_entity_test_f1`: F1 on test mentions whose `(label, surface)` did not appear in fold training data.

## Troubleshooting

- If `pip install` fails, confirm you are in `src/` and using `src/requirements.txt`.
- If you are on Windows, consider using PowerShell or Git Bash for the commands above.
