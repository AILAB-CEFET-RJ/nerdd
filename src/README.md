# src

This folder now contains multiple subpipelines with separate documentation.

## Subpipelines

1. `base_model_training/`
- Purpose: nested-CV fine-tuning and evaluation for NER.
- Docs: `base_model_training/README.md`
- Entrypoints: `python3 -m base_model_training.train_nested_kfold`, `base_model_training/evaluate_gliner.py`

2. `pseudolabelling/`
- Purpose: large-corpus entity prediction plus metadata-aware context boost.
- Docs: `pseudolabelling/README.md`
- Entrypoints: `pseudolabelling/run_iterative_cycle.py`, `pseudolabelling/generate_corpus_predictions.py`, `pseudolabelling/apply_context_boost.py`, `pseudolabelling/compute_record_scores.py`, `pseudolabelling/split_pseudolabels.py`, `pseudolabelling/refit_model.py`, `pseudolabelling/evaluate_refit.py`, `pseudolabelling/prepare_next_iteration.py`

3. `calibration/`
- Purpose: confidence score calibration over pseudolabelled outputs.
- Docs: `calibration/README.md`
- Entrypoint: `calibration/run_calibration.py`

4. `tools/`
- Purpose: auxiliary scripts not part of core train/pseudolabelling/calibration execution.
- Utility: `tools/convert_sanity_jsonl_to_bio_csv.py`
- Utility: `tools/replace_label_in_jsonl.py`
- Utility: `tools/list_distinct_labels.py`
- Utility: `tools/render_ner_html.py`
- Utility: `tools/build_annotation_editor.py`
- Utility: `tools/clean_generic_spans.py`

## Artifact Convention

- Runtime outputs should be written under `src/artifacts/`.
- Recommended training paths:
  - smoke: `artifacts/base_model_training/smoke/`
  - experiments: `artifacts/base_model_training/experiments/`

## Shared Docs

- `../docs/MIGRATION.md`
- `../docs/PIPELINE_OVERVIEW.md`
- `../docs/RUNBOOK.md`
- `../docs/ARCHITECTURE.md`
- `CHANGELOG.md`
