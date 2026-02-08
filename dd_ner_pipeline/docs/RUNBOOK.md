# Runbook

## 1) Local Smoke Test (CPU / low RAM)
```bash
cd dd_ner_pipeline
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 train_nested_kfold.py \
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
  --output-dir ./smoke_run_nested_tiny \
  --log-level INFO
```

## 2) Server Training (example with batch-size 16)
```bash
cd dd_ner_pipeline
python3 train_nested_kfold.py \
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
  --output-dir ./run_batch16 \
  --log-level INFO
```

## 3) Evaluation
```bash
cd dd_ner_pipeline
python3 evaluate_gliner.py \
  --model-path ./run_batch16/best_overall_gliner_model \
  --gt-jsonl ../data/dd_corpus_small_test_filtered.json \
  --pred-jsonl ./eval/pred.jsonl \
  --report-path ./eval/report.txt \
  --calibrated-thresholds-json ./eval/thresholds.json \
  --labels Person,Location,Organization \
  --chunk-size 128 \
  --batch-size 1 \
  --prediction-threshold 0.6 \
  --log-level INFO
```

## Troubleshooting
- `Killed`: reduce `batch-size`, `max-length`, number of folds/trials.
- HF timeout/network instability: run after cache warmup with offline env vars.
- JSON parse issue `Extra data`: file is JSONL, not a single JSON array.
