# Changelog

## Unreleased
- Refactored training/evaluation into modular pipeline (`gliner_train/`).
- Replaced legacy entrypoints with:
  - `train_nested_kfold.py`
  - `gliner_train/evaluate_gliner.py`
- Hardened nested CV with group-aware splitting by `sample_id`.
- Added structured reporting (`nested_cv_results.json`).
- Standardized human-readable report name (`nested_cv_results.txt`).
- Added runtime tracking (start/end/duration in logs + reports).
- Added seen vs unseen entity metrics in CV output.
- Added automatic loss-curve saving for audit.
- Improved model loading stability for repeated/offline runs.
- Added documentation set under `dd_ner_pipeline/docs/`.
- Refactored legacy corpus pseudo-label script into a modular inference pipeline:
  - `pseudolabelling/generate_corpus_predictions.py`
  - `pseudolabelling/pipeline.py`
  - `pseudolabelling/cli.py`
  - `pseudolabelling/config.py`
- Removed redundant legacy pseudolabelling scripts from `dd_ner_pipeline/`.
- Moved sanity conversion utility to `tools/convert_sanity_jsonl_to_bio_csv.py`.
- Replaced legacy calibration scripts with a modular calibration subpipeline:
  - `calibration/run_calibration.py`
  - `calibration/pipeline.py`
  - `calibration/cli.py`
  - `calibration/config.py`
  - `calibration/methods/temperature.py`
  - `calibration/methods/isotonic.py`
