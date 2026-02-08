# Changelog

## Unreleased
- Refactored training/evaluation into modular pipeline (`gliner_train/`).
- Replaced legacy entrypoints with:
  - `train_nested_kfold.py`
  - `evaluate_gliner.py`
- Hardened nested CV with group-aware splitting by `sample_id`.
- Added structured reporting (`nested_cv_results.json`).
- Standardized human-readable report name (`nested_cv_results.txt`).
- Added runtime tracking (start/end/duration in logs + reports).
- Added seen vs unseen entity metrics in CV output.
- Added automatic loss-curve saving for audit.
- Improved model loading stability for repeated/offline runs.
- Added documentation set under `dd_ner_pipeline/docs/`.
