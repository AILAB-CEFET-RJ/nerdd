# pseudolabelling Subpipeline

Inference-only pipeline to label a large unlabeled corpus using a trained model.

## Entrypoint

- `run_iterative_cycle.py` (new unified orchestrator)
- `generate_corpus_predictions.py`
- `apply_context_boost.py`
- `compute_record_scores.py`
- `split_pseudolabels.py`
- `refit_model.py`
- `evaluate_refit.py`
- `prepare_next_iteration.py`

## Typical Flow

1. Train model with nested CV in the training subpipeline.
2. Use best model snapshot for large-corpus prediction.
3. Optionally apply metadata-aware confidence boost before threshold filtering.
4. Compute record-level score from entity-level confidence.
5. Split records into kept/discarded sets using a record-level score threshold.
6. Refit the model on kept pseudolabel records.
7. Evaluate refit model on labeled sanity/test JSONL.
8. Prepare discarded rows for the next pseudolabelling iteration.

## Command Example

```bash
cd dd_ner_pipeline
python3 pseudolabelling/generate_corpus_predictions.py \
  --model-path ./artifacts/gliner_train/smoke/run_nested_tiny/best_overall_gliner_model \
  --input-jsonl dd_corpus_large.json \
  --output-jsonl ./artifacts/pseudolabelling/iter01/01_predictions.jsonl \
  --stats-json ./artifacts/pseudolabelling/iter01/01_predictions_stats.json \
  --labels Person,Location,Organization \
  --text-fields assunto,relato,bairroLocal,logradouroLocal,cidadeLocal,pontodeReferenciaLocal \
  --max-tokens 384 \
  --batch-size 4 \
  --score-threshold 0.0 \
  --log-level INFO
```

## Unified Orchestrator (Recommended)

Run the full iterative cycle (prediction -> optional calibration -> context boost -> scoring -> split -> refit -> optional evaluation -> optional next-iteration prep):

```bash
cd dd_ner_pipeline
python3 pseudolabelling/run_iterative_cycle.py \
  --run-dir ./artifacts/pseudolabelling/iter_cycle_01 \
  --model-path ./artifacts/gliner_train/experiments/run_batch16/best_overall_gliner_model \
  --input-jsonl dd_corpus_large.json \
  --labels Person,Location,Organization \
  --text-fields assunto,relato,bairroLocal,logradouroLocal,cidadeLocal,pontodeReferenciaLocal \
  --prediction-batch-size 4 \
  --prediction-max-tokens 384 \
  --prediction-threshold 0.0 \
  --use-calibration \
  --calibration-method isotonic \
  --calibration-label-source calibration-csv \
  --calibration-csv ../data/comparacao_calibracao.csv \
  --context-base-score-field score_calibrated \
  --record-score-field score_context_boosted \
  --split-threshold 0.80 \
  --refit-base-model ./best_overall_gliner_model \
  --refit-epochs 10 \
  --refit-batch-size 8 \
  --evaluate-refit \
  --eval-gt-jsonl ../data/dd_corpus_small_test_filtered.json \
  --prepare-next-iteration \
  --log-level INFO
```

## Context Boost Example

```bash
cd dd_ner_pipeline
python3 pseudolabelling/apply_context_boost.py \
  --input-jsonl ./artifacts/pseudolabelling/iter01/01_predictions.jsonl \
  --output-jsonl ./artifacts/pseudolabelling/iter01/02_context_boosted.jsonl \
  --stats-json ./artifacts/pseudolabelling/iter01/02_context_boost_stats.json \
  --base-score-field score \
  --fallback-score-fields score_calibrated,score_ts,score_iso \
  --output-score-field score_context_boosted \
  --output-record-score-field record_score_context_boosted \
  --boost-factor 1.2 \
  --boost-scope all-entities \
  --match-policy any-metadata-in-text \
  --log-level INFO
```

## Record Score Example

```bash
cd dd_ner_pipeline
python3 pseudolabelling/compute_record_scores.py \
  --input-jsonl ./artifacts/pseudolabelling/iter01/02_context_boosted.jsonl \
  --output-jsonl ./artifacts/pseudolabelling/iter01/03_scored.jsonl \
  --stats-json ./artifacts/pseudolabelling/iter01/03_score_stats.json \
  --score-field score_context_boosted \
  --output-field record_score \
  --legacy-field-alias score_relato \
  --aggregation mean \
  --empty-entities-policy zero \
  --log-level INFO
```

## Split Example

```bash
cd dd_ner_pipeline
python3 pseudolabelling/split_pseudolabels.py \
  --input-jsonl ./artifacts/pseudolabelling/iter01/03_scored.jsonl \
  --out-dir ./artifacts/pseudolabelling/iter01/04_split \
  --score-field record_score \
  --threshold 0.80 \
  --operator ge \
  --fallback-score-field score_relato_confianca \
  --missing-policy discard \
  --legacy-filenames \
  --log-level INFO
```

## Refit Example

```bash
cd dd_ner_pipeline
python3 pseudolabelling/refit_model.py \
  --input-path ./artifacts/pseudolabelling/iter01/04_split \
  --output-model-dir ./artifacts/pseudolabelling/iter01/05_refit_model \
  --base-model ./artifacts/gliner_train/experiments/run_batch16/best_overall_gliner_model \
  --epochs 10 \
  --patience 3 \
  --batch-size 8 \
  --lr 3e-5 \
  --weight-decay 0.01 \
  --val-ratio 0.1 \
  --allowed-labels Person,Location,Organization \
  --log-level INFO
```

## Evaluate Refit Example

```bash
cd dd_ner_pipeline
python3 pseudolabelling/evaluate_refit.py \
  --model-path ./artifacts/pseudolabelling/iter01/05_refit_model \
  --gt-jsonl ../data/dd_corpus_small_test_filtered.json \
  --out-dir ./artifacts/pseudolabelling/iter01/06_eval_refit \
  --labels Person,Location,Organization \
  --prediction-threshold 0.05 \
  --batch-size 4 \
  --max-tokens 384 \
  --log-level INFO
```

## Prepare Next Iteration Example

```bash
cd dd_ner_pipeline
python3 pseudolabelling/prepare_next_iteration.py \
  --input-glob ./artifacts/pseudolabelling/iter01/04_split/*_descartados.jsonl \
  --out-dir ./artifacts/pseudolabelling/iter02_input \
  --output-suffix _iter02_input \
  --keep-fields assunto,relato,logradouroLocal,bairroLocal,cidadeLocal,pontodeReferenciaLocal \
  --required-fields relato \
  --coerce-non-string stringify \
  --deduplicate-by relato,bairroLocal \
  --stats-json ./artifacts/pseudolabelling/iter02_input/prepare_next_iteration_stats.json \
  --log-level INFO
```

## Artifact Convention

Write pseudolabelling outputs under:
- `dd_ner_pipeline/artifacts/pseudolabelling/`

## `text-fields` Profiles

This argument controls which JSONL fields are concatenated into the inference text before entity prediction.

1. `strict` profile (recommended for cleaner NER behavior)

```bash
--text-fields relato
```

Use when you want predictions based only on the report narrative.

2. `contextual` profile (legacy-compatible / metadata-aware)

```bash
--text-fields assunto,relato,bairroLocal,logradouroLocal,cidadeLocal,pontodeReferenciaLocal
```

Use when downstream pseudolabelling stages also depend on metadata context from those fields.

## Main Options

- `--model-path`: trained model path (local folder).
- `--input-jsonl`: large input corpus (JSONL).
- `--output-jsonl`: output with predicted `entities`.
- `--stats-json`: run statistics and metadata.
- `--labels`: entity labels to predict.
- `--text-fields`: source fields to compose inference text.
- `--max-tokens`: tokenizer chunk size.
- `--batch-size`: prediction batch size.
- `--score-threshold`: minimum entity score.
- `--keep-inference-text`: include generated inference text in output rows.

## Context Boost Rules

- `--match-policy any-metadata-in-text`:
  enables boosting if any metadata location field appears in the selected text field.
- `--match-policy entity-metadata-overlap`:
  requires metadata/text context and then checks per-entity overlap for matched scope.
- `--boost-scope all-entities`:
  boosts all entities in a matched record.
- `--boost-scope location-only`:
  boosts only entities with labels listed in `--location-labels`.
- `--boost-scope matched-only`:
  boosts only entities whose text overlaps metadata values.

Output fields:
- per-entity: `score_context_boosted`
- per-record: `record_score_context_boosted`
- legacy aliases (default on): `score_confianca`, `score_relato_confianca`

## Record Score Outputs

- per-record: `record_score` (default, configurable)
- optional legacy alias: `score_relato`
- run stats JSON includes invalid-score counts and score distribution

## Split Outputs

- `kept.jsonl`
- `discarded.jsonl`
- `summary.json`
- Optional legacy names (`--legacy-filenames`):
  - `mantidos.jsonl`
  - `descartados.jsonl`
  - `resumo.json`

## Refit Outputs

- model directory (`--output-model-dir`)
- `refit_stats.json`
- `train_manifest.jsonl`
- `val_manifest.jsonl`

## Evaluation Outputs

- `predictions.jsonl`
- `classification_report.txt`
- `metrics.json`
- `run_stats.json`

## Next Iteration Prep Outputs

- `<stem><output-suffix>.jsonl` for each matched discarded file
- `prepare_next_iteration_stats.json` with per-file mappings and drop reasons
