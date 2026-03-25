# src

This folder now contains multiple subpipelines with separate documentation.

## Subpipelines

1. `base_model_training/`
- Purpose: nested-CV fine-tuning and evaluation for NER.
- Docs: `base_model_training/README.md`
- Entrypoints: `python3 -m base_model_training.train_nested_kfold`, `base_model_training/evaluate_gliner.py`

2. `pseudolabelling/`
- Purpose: large-corpus entity prediction plus metadata-aware context boost, optionally consuming a fitted calibrator artifact.
- Docs: `pseudolabelling/README.md`
- Entrypoints: `pseudolabelling/run_iterative_cycle.py`, `pseudolabelling/generate_corpus_predictions.py`, `pseudolabelling/apply_context_boost.py`, `pseudolabelling/compute_record_scores.py`, `pseudolabelling/split_pseudolabels.py`, `pseudolabelling/refit_model.py`, `pseudolabelling/evaluate_refit.py`, `pseudolabelling/prepare_next_iteration.py`

3. `calibration/`
- Purpose: fit and apply reusable probability calibrators for base-model entity scores.
- Docs: `calibration/README.md`
- Entrypoints: `calibration/fit_calibrator.py`, `calibration/apply_calibrator.py`

4. `tools/`
- Purpose: auxiliary scripts not part of core train/pseudolabelling/calibration execution.
- Utility: `tools/convert_sanity_jsonl_to_bio_csv.py`
- Utility: `tools/replace_label_in_jsonl.py`
- Utility: `tools/list_distinct_labels.py`
- Utility: `tools/count_dataset_entities.py`
- Utility: `tools/split_dataset_for_calibration.py`
- Utility: `tools/build_calibration_dataset.py`
- Utility: `tools/render_ner_html.py`
- Utility: `tools/build_annotation_editor.py`
- Utility: `tools/clean_generic_spans.py`

## Artifact Convention

- Runtime outputs should be written under `src/artifacts/`.
- Recommended training paths:
  - smoke: `artifacts/base_model_training/smoke/`
  - experiments: `artifacts/base_model_training/experiments/`

## Shared Utilities

- `gliner_loader.py`
  - shared GLiNER loader for inference/evaluation paths
  - centralizes `load_tokenizer=True` and optional `max_length`
  - does not replace the specialized training loader behavior in `src/base_model_training/cv.py`

## Shared Docs

- `../docs/MIGRATION.md`
- `../docs/PIPELINE_OVERVIEW.md`
- `../docs/RUNBOOK.md`
- `../docs/ARCHITECTURE.md`
- `../docs/ARCHITECTURAL_DECISIONS.md`
- `CHANGELOG.md`
