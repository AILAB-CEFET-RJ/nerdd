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

## 3) Server Training (example with batch-size 16)
```bash
cd src
python3 -m base_model_training.train_nested_kfold \
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

## 4) Evaluation
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

## 5) Large Corpus Prediction (inference-only)
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

## 6) Split Holdout For Calibration
```bash
cd src
python3 tools/split_dataset_for_calibration.py \
  --input ../data/dd_corpus_small_test.json \
  --calibration-output ../data/dd_corpus_small_test_calibration.json \
  --final-test-output ../data/dd_corpus_small_test_final.json \
  --summary-json ../data/dd_corpus_small_test_split_summary.json \
  --calibration-ratio 0.2 \
  --seed 42 \
  --mode random
```

## 7) Build Calibration CSV From Labeled Holdout
```bash
cd src
python3 tools/build_calibration_dataset.py \
  --model-path ./artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --input ../data/dd_corpus_small_test_calibration.json \
  --output-csv ../data/dd_corpus_small_test_calibration_predictions.csv \
  --output-predictions-jsonl ./artifacts/calibration/base_model/calibration_predictions.jsonl \
  --labels Person,Location,Organization \
  --batch-size 4 \
  --max-tokens 384 \
  --threshold 0.0
```

## 8) Fit Calibrator Artifact
```bash
cd src
python3 calibration/fit_calibrator.py \
  --method temperature-per-class \
  --calibration-csv ../data/dd_corpus_small_test_calibration_predictions.csv \
  --output-calibrator ./artifacts/calibration/base_model/calibrator.json \
  --stats-json ./artifacts/calibration/base_model/fit_stats.json \
  --score-col Score \
  --label-col Validacao \
  --class-col Label \
  --labels Person,Location,Organization \
  --log-level INFO
```

## 9) Large Corpus Prediction With Calibrator
```bash
cd src
python3 pseudolabelling/generate_corpus_predictions.py \
  --model-path ./artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --calibrator-path ./artifacts/calibration/base_model/calibrator.json \
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

## 10) Sample A Reproducible Fraction Of The Large Corpus

Before running expensive pseudolabelling experiments on the full large corpus, create a fixed sample to study the score distribution and choose a threshold budget more safely.

```bash
cd .
python3 src/tools/sample_large_corpus.py \
  --input data/dd_corpus_large.json \
  --output-jsonl data/dd_corpus_large_sample_10k.jsonl \
  --sample-size 10000 \
  --seed 42 \
  --summary-json data/dd_corpus_large_sample_10k_summary.json
```

Recommended use:

- run prediction, context boost, and record scoring on `data/dd_corpus_large_sample_10k.jsonl`
- inspect the resulting score distribution
- choose the kept/discarded threshold based on the observed volume of candidate reports

Observed probe results on the `10k` sample:

- `threshold=0.20` -> `281` kept
- `threshold=0.30` -> `52` kept
- `threshold=0.40` -> `14` kept

Recommended first full-corpus operating point:

- start with `threshold=0.30`
- treat `0.20` as a more aggressive follow-up
- treat `0.40` as a more conservative follow-up

## 11) Controlled Refit Comparison For Dissertation Experiments

Use the same final holdout `../data/dd_corpus_small_test_filtered.json` for both runs below.

First, measure the gain from additional supervised refit only:

```bash
cd src
python3 -m pseudolabelling.run_iterative_cycle \
  --run-dir ./artifacts/pseudolabelling/iter_cycle_supervised_only_t050 \
  --model-path ./artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --prediction-calibrator-path ./artifacts/calibration/base_model/calibrator.json \
  --input-jsonl ../data/dd_corpus_small_test_calibration.jsonl \
  --labels Person,Location,Organization \
  --text-fields text \
  --prediction-batch-size 16 \
  --prediction-max-tokens 512 \
  --prediction-threshold 0.0 \
  --record-score-field score_context_boosted \
  --split-threshold 0.50 \
  --refit-mode supervised_only \
  --refit-base-model ./artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --refit-supervised-train-path ../data/dd_corpus_small_train.json \
  --refit-epochs 3 \
  --refit-patience 2 \
  --refit-batch-size 16 \
  --refit-max-length 512 \
  --refit-overlap 128 \
  --refit-lr 1e-5 \
  --refit-weight-decay 0.01 \
  --evaluate-refit \
  --eval-gt-jsonl ../data/dd_corpus_small_test_filtered.json \
  --eval-prediction-threshold 0.05 \
  --eval-batch-size 16 \
  --eval-max-tokens 512 \
  --prepare-next-iteration \
  --log-level INFO
```

Then, measure the full semisupervised recipe:

```bash
cd src
python3 -m pseudolabelling.run_iterative_cycle \
  --run-dir ./artifacts/pseudolabelling/iter_cycle_supervised_plus_pseudolabels_t050 \
  --model-path ./artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --prediction-calibrator-path ./artifacts/calibration/base_model/calibrator.json \
  --input-jsonl ../data/dd_corpus_small_test_calibration.jsonl \
  --labels Person,Location,Organization \
  --text-fields text \
  --prediction-batch-size 16 \
  --prediction-max-tokens 512 \
  --prediction-threshold 0.0 \
  --record-score-field score_context_boosted \
  --split-threshold 0.50 \
  --refit-mode supervised_plus_pseudolabels \
  --refit-base-model ./artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --refit-supervised-train-path ../data/dd_corpus_small_train.json \
  --refit-epochs 3 \
  --refit-patience 2 \
  --refit-batch-size 16 \
  --refit-max-length 512 \
  --refit-overlap 128 \
  --refit-lr 1e-5 \
  --refit-weight-decay 0.01 \
  --evaluate-refit \
  --eval-gt-jsonl ../data/dd_corpus_small_test_filtered.json \
  --eval-prediction-threshold 0.05 \
  --eval-batch-size 16 \
  --eval-max-tokens 512 \
  --prepare-next-iteration \
  --log-level INFO
```

Interpretation:

- compare `09_base_vs_refit_comparison.json` from the two runs
- the marginal effect of pseudolabelling is:
  - `(supervised_plus_pseudolabels) - (supervised_only)`
- do not attribute all improvement over the base model to pseudolabelling

Observed controlled result for `t020`:

- base:
  - `micro_f1=0.4440`
  - `macro_f1=0.4362`
- supervised-only refit:
  - `micro_f1=0.4952`
  - `macro_f1=0.4772`
- supervised-plus-pseudolabels refit:
  - `micro_f1=0.5260`
  - `macro_f1=0.5097`

Interpretation of the `t020` result:

- extra supervised refit gain over base:
  - `micro_f1 +0.0512`
  - `macro_f1 +0.0411`
- additional pseudolabelling gain over supervised-only:
  - `micro_f1 +0.0308`
  - `macro_f1 +0.0324`

## Troubleshooting
- `Killed`: reduce `batch-size`, `max-length`, number of folds/trials.
- HF timeout/network instability: run after cache warmup with offline env vars.
- JSON parse issue `Extra data`: file is JSONL, not a single JSON array.
