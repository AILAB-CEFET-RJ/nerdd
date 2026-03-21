# calibration Subpipeline

Fits and applies probability calibrators for entity confidence scores produced by the base model.

## Entrypoints

- `fit_calibrator.py`: fit and persist a reusable calibrator artifact from a labeled calibration CSV.
- `apply_calibrator.py`: apply a persisted calibrator artifact to a JSONL corpus with entity scores.
- `run_calibration.py`: legacy fit+apply command kept for backward compatibility.
- `evaluate_calibration.py`: evaluation-only helper for reliability curves and summary metrics.

## Methods

- `temperature`: global temperature scaling.
- `temperature-per-class`: one temperature per entity label.
- `isotonic`: global isotonic regression calibration.

## Recommended Flow

1. Run the trained base model on a labeled holdout subset.
2. Build a calibration CSV with `Score`, binary `Validacao`, and ideally entity `Label`.
3. Fit a calibrator artifact once.
4. Apply that artifact during large-corpus prediction or as a separate post-processing step.

## Fit Example

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

## Apply Example

```bash
cd src
python3 calibration/apply_calibrator.py \
  --input-jsonl ./artifacts/pseudolabelling/iter01/01_predictions.jsonl \
  --output-jsonl ./artifacts/calibration/iter01/01_calibrated.jsonl \
  --calibrator-path ./artifacts/calibration/base_model/calibrator.json \
  --score-field score \
  --output-score-field score_calibrated \
  --preserve-original-score-field score_original \
  --log-level INFO
```

## Evaluation Example

```bash
cd src
python3 calibration/evaluate_calibration.py \
  --calibration-csv ../data/dd_corpus_small_test_calibration_predictions.csv \
  --out-dir ./artifacts/calibration/iter01/eval \
  --score-col Score \
  --label-col Validacao \
  --bins 10 \
  --cv 5
```

## Artifact Convention

Write calibration outputs under:
- `src/artifacts/calibration/`

## Outputs

- calibrator artifact JSON
- calibrated JSONL (same records, with calibrated score field in entities)
- fit/apply stats JSON with method parameters and diagnostics
- evaluation artifacts (from `evaluate_calibration.py`):
  - `metrics_summary.csv`
  - `reliability_raw.csv`
  - `reliability_ts.csv`
  - `reliability_iso.csv` (unless `--no-isotonic`)
  - `reliability_curves.png`
