# base_model_training Subpipeline

Training and evaluation stack for nested cross-validation fine-tuning.

## Entrypoints

- `python3 -m base_model_training.train_nested_kfold`
- `evaluate_gliner.py`

## Setup

```bash
cd src
pip install -r requirements.txt
```

## Local Smoke Test

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
  --backbone-lr-values 1e-5 \
  --ner-lr-values 3e-5 \
  --weight-decay-values 0.01 \
  --train-sampling weighted \
  --thresholds 0.6 \
  --refit-val-size 0.5 \
  --output-dir ./artifacts/base_model_training/smoke/run_nested_tiny \
  --log-level INFO
```

## Production Example (batch-size 16)

```bash
python3 -m base_model_training.train_nested_kfold \
  --train-path ../data/dd_corpus_small_train.json \
  --batch-size 16 \
  --num-epochs 20 \
  --n-splits 3 \
  --n-inner-splits 3 \
  --search-mode random \
  --num-trials 20 \
  --backbone-lr-values 1e-6,2e-6,5e-6,1e-5 \
  --ner-lr-values 1e-5,2e-5,3e-5,5e-5,8e-5,1e-4,2e-4,3e-4 \
  --weight-decay-values 0.0,0.01,0.05 \
  --train-sampling weighted \
  --thresholds 0.5,0.6 \
  --output-dir ./artifacts/base_model_training/experiments/run_batch16 \
  --log-level INFO
```

## Evaluation Example

```bash
python3 base_model_training/evaluate_gliner.py \
  --model-path ./artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --gt-jsonl ../data/dd_corpus_small_test_final.json \
  --pred-jsonl ./artifacts/base_model_training/experiments/run_batch16/eval/pred.jsonl \
  --report-path ./artifacts/base_model_training/experiments/run_batch16/eval/report.txt \
  --calibrated-thresholds-json ./artifacts/base_model_training/experiments/run_batch16/eval/thresholds.json \
  --labels Person,Location,Organization \
  --chunk-size 128 \
  --batch-size 1 \
  --prediction-threshold 0.6 \
  --log-level INFO
```

The base-model evaluation path now uses the shared inference/evaluation GLiNER loader in `src/gliner_loader.py`.
The nested-CV training path in `src/base_model_training/cv.py` still keeps its own specialized model-loading fallback logic.

## Outputs

- `nested_cv_results.txt`
- `nested_cv_results.json`
- `best_overall_gliner_model/`
- `loss_fold*_trial*.png`
- Recommended base output root: `artifacts/base_model_training/`
