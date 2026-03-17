# Runbook

Assumes the environment from `docs/INSTALL.md` is already set up.

## 1) First Run

Use this smoke test first to validate the training stack end-to-end on a tiny dataset.

If you are running on an NVIDIA Blackwell GPU such as an RTX 5090, confirm your environment uses PyTorch `cu128` or newer before starting.

## 2) Local Smoke Test (CPU / low RAM)
```bash
cd src
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

Expected outputs:
- `./artifacts/base_model_training/smoke/run_nested_tiny/nested_cv_results.txt`
- `./artifacts/base_model_training/smoke/run_nested_tiny/nested_cv_results.json`
- `./artifacts/base_model_training/smoke/run_nested_tiny/best_overall_gliner_model/`

## 3) Choosing The GLiNER Base Model

`base_model_training.train_nested_kfold` accepts the GLiNER base model through `--model-base`.

Recommended options:

- `urchade/gliner_small-v2.1`: best for smoke tests, CPU runs, and low-VRAM debugging
- `urchade/gliner_medium-v2.1`: recommended default for real training when you want a stronger model with moderate cost
- `urchade/gliner_large-v2.1`: recommended when the training machine has enough VRAM and you want to prioritize model quality over runtime
- `urchade/gliner_multi-v2.1`: worth testing when the corpus is multilingual or primarily non-English

Operational guidance:

- use `small` to validate the pipeline end-to-end before launching a long run
- use `medium` as the first serious baseline for nested CV
- use `large` only after a short smoke run confirms your batch size and `max-length` fit in GPU memory
- on an RTX 5090, `large` is a reasonable candidate, but a short run with a reduced batch size is still the safest first step

Example substitutions:

```bash
--model-base urchade/gliner_small-v2.1
--model-base urchade/gliner_medium-v2.1
--model-base urchade/gliner_large-v2.1
--model-base urchade/gliner_multi-v2.1
```

## 4) Server Training (example with batch-size 16)
```bash
cd src
python3 -m base_model_training.train_nested_kfold \
  --model-base urchade/gliner_medium-v2.1 \
  --train-path ../data/dd_corpus_small_train.json \
  --batch-size 16 \
  --num-epochs 20 \
  --n-splits 3 \
  --n-inner-splits 3 \
  --search-mode random \
  --num-trials 20 \
  --lr-values 1e-6,2e-6,5e-6,1e-5,2e-5,3e-5,5e-5,8e-5,1e-4,2e-4,3e-4 \
  --weight-decay-values 0.0,0.01,0.05 \
  --thresholds 0.5,0.6 \
  --output-dir ./artifacts/base_model_training/experiments/run_batch16 \
  --log-level INFO
```

## 5) Evaluation
```bash
cd src
python3 base_model_training/evaluate_gliner.py \
  --model-path ./artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --gt-jsonl ../data/dd_corpus_small_test_filtered.json \
  --pred-jsonl ./artifacts/base_model_training/experiments/run_batch16/eval/pred.jsonl \
  --report-path ./artifacts/base_model_training/experiments/run_batch16/eval/report.txt \
  --calibrated-thresholds-json ./artifacts/base_model_training/experiments/run_batch16/eval/thresholds.json \
  --labels Person,Location,Organization \
  --chunk-size 128 \
  --batch-size 1 \
  --prediction-threshold 0.6 \
  --log-level INFO
```

## 6) Large Corpus Prediction (inference-only)
```bash
cd src
python3 pseudolabelling/generate_corpus_predictions.py \
  --model-path ./artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --input-jsonl ../data/dd_corpus_large.json \
  --output-jsonl ./artifacts/pseudolabelling/iter01/01_predictions.jsonl \
  --stats-json ./artifacts/pseudolabelling/iter01/01_predictions_stats.json \
  --labels Person,Location,Organization \
  --text-fields assunto,relato,bairroLocal,logradouroLocal,cidadeLocal,pontodeReferenciaLocal \
  --max-tokens 384 \
  --batch-size 4 \
  --score-threshold 0.0 \
  --log-level INFO
```

## 7) Score Calibration (post-pseudolabelling)
```bash
cd src
python3 calibration/run_calibration.py \
  --method temperature-per-class \
  --input-jsonl ./artifacts/pseudolabelling/iter01/01_predictions.jsonl \
  --output-jsonl ./artifacts/calibration/iter01/01_calibrated.jsonl \
  --stats-json ./artifacts/calibration/iter01/01_calibration_stats.json \
  --score-field score \
  --output-score-field score_calibrated \
  --preserve-original-score-field score_original \
  --labels Person,Location,Organization \
  --label-source calibration-csv \
  --calibration-csv ../data/comparacao_calibracao.csv \
  --csv-score-col Score \
  --csv-label-col Validacao \
  --log-level INFO
```

## Troubleshooting
- `Killed`: reduce `batch-size`, `max-length`, number of folds/trials.
- HF timeout/network instability: run after cache warmup with offline env vars.
- JSON parse issue `Extra data`: file is JSONL, not a single JSON array.
