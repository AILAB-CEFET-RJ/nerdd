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
2. Optionally fit a reusable score calibrator on a labeled holdout subset.
3. Use best model snapshot for large-corpus prediction, optionally applying the calibrator artifact.
4. Optionally apply metadata-aware confidence boost before threshold filtering.
5. Compute record-level score from entity-level confidence.
6. Split records into kept/discarded sets using a record-level score threshold.
7. Refit the model on the original supervised training set plus kept pseudolabel records.
8. Evaluate both base and refit models on the same labeled holdout and write an explicit comparison artifact.
9. Prepare discarded rows for the next pseudolabelling iteration.

Two semisupervised regimes are now possible:

- `single shot`
  - one unlabeled input
  - one kept/discarded split
  - one refit
- `iterative`
  - multiple fixed unlabeled chunks
  - one split per chunk
  - an accumulated pseudolabel artifact that grows across iterations
  - one refit per accumulated state

## Command Example

```bash
cd src
python3 pseudolabelling/generate_corpus_predictions.py \
  --model-path ./artifacts/base_model_training/smoke/run_nested_tiny/best_overall_gliner_model \
  --model-max-length 384 \
  --calibrator-path ./artifacts/calibration/base_model/calibrator.json \
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

Run the full iterative cycle (prediction -> optional legacy calibration step -> context boost -> scoring -> split -> refit -> paired base/refit evaluation -> optional next-iteration prep):

```bash
cd src
python3 pseudolabelling/run_iterative_cycle.py \
  --run-dir ./artifacts/pseudolabelling/iter_cycle_01 \
  --model-path ./artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --prediction-calibrator-path ./artifacts/calibration/base_model/calibrator.json \
  --input-jsonl dd_corpus_large.json \
  --labels Person,Location,Organization \
  --text-fields assunto,relato,bairroLocal,logradouroLocal,cidadeLocal,pontodeReferenciaLocal \
  --prediction-batch-size 4 \
  --prediction-max-tokens 384 \
  --prediction-model-max-length 384 \
  --prediction-threshold 0.0 \
  --record-score-field score_context_boosted \
  --split-threshold 0.80 \
  --refit-base-model ./best_overall_gliner_model \
  --refit-supervised-train-path ../data/dd_corpus_small_train.json \
  --refit-epochs 10 \
  --refit-batch-size 8 \
  --evaluate-refit \
  --eval-gt-jsonl ../data/dd_corpus_small_test_final.json \
  --eval-model-max-length 384 \
  --prepare-next-iteration \
  --log-level INFO
```

Use `--model-max-length`, `--prediction-model-max-length`, or `--eval-model-max-length` when you need to pass GLiNER's own `max_length` into `GLiNER.from_pretrained(...)`. This is separate from `--max-tokens` / `--prediction-max-tokens` / `--eval-max-tokens`, which only control the pipeline's external chunking.

Inference-oriented entrypoints now share the same loader in `src/gliner_loader.py`, keeping GLiNER load-time behavior aligned across:

- corpus prediction
- refit evaluation
- inference profiling
- calibration-dataset generation

## Context Boost Example

```bash
cd src
python3 pseudolabelling/apply_context_boost.py \
  --input-jsonl ./artifacts/pseudolabelling/iter01/01_predictions.jsonl \
  --output-jsonl ./artifacts/pseudolabelling/iter01/02_context_boosted.jsonl \
  --stats-json ./artifacts/pseudolabelling/iter01/02_context_boost_stats.json \
  --base-score-field score \
  --fallback-score-fields score_calibrated,score_ts,score_iso \
  --output-score-field score_context_boosted \
  --output-record-score-field record_score_context_boosted \
  --boost-factor 1.2 \
  --boost-scope location-matched-only \
  --match-policy any-metadata-in-text \
  --log-level INFO
```

## Record Score Example

```bash
cd src
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
cd src
python3 pseudolabelling/split_pseudolabels.py \
  --input-jsonl ./artifacts/pseudolabelling/iter01/03_scored.jsonl \
  --out-dir ./artifacts/pseudolabelling/iter01/04_split \
  --score-field record_score \
  --threshold 0.80 \
  --operator ge \
  --entity-gate-score-field score_context_boosted \
  --entity-gate-labels Location \
  --entity-gate-aggregation max \
  --entity-gate-threshold 0.50 \
  --fallback-score-field score_relato_confianca \
  --missing-policy discard \
  --legacy-filenames \
  --log-level INFO
```

## Refit Example

```bash
cd src
python3 pseudolabelling/refit_model.py \
  --input-path ./artifacts/pseudolabelling/iter01/04_split \
  --pseudolabel-path ./artifacts/pseudolabelling/iter01/04_split/kept.jsonl \
  --output-model-dir ./artifacts/pseudolabelling/iter01/05_refit_model \
  --base-model ./artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --supervised-train-path ../data/dd_corpus_small_train.json \
  --epochs 10 \
  --patience 3 \
  --batch-size 8 \
  --lr 3e-5 \
  --weight-decay 0.01 \
  --val-ratio 0.1 \
  --allowed-labels Person,Location,Organization \
  --log-level INFO
```

Refit behavior:

- supervised examples are loaded first from `--supervised-train-path`
- pseudolabel records are loaded from `--pseudolabel-path` when provided, otherwise from `--input-path`
- kept pseudolabel records are appended afterward
- when no external validation file is provided and supervised data exists, the supervised dataset is split first and validation remains supervised-only
- if `--disable-deduplicate-by-text` is not used, duplicate texts are dropped with preference for the supervised row
- `refit_stats.json` records how many rows came from each source
- iterative experiments can therefore refit on an accumulated pseudolabel JSONL artifact instead of only the current run's `kept.jsonl`

Recommended iterative artifact pattern:

- chunk-local split outputs:
  - `iter_01/05_split/kept.jsonl`
  - `iter_02/05_split/kept.jsonl`
  - `iter_03/05_split/kept.jsonl`
- accumulated pseudolabel artifacts:
  - `accumulated/kept_acc_01.jsonl`
  - `accumulated/kept_acc_02.jsonl`
  - `accumulated/kept_acc_03.jsonl`

When `--pseudolabel-path` or `--refit-pseudolabel-path` is provided, refit uses the accumulated artifact and not just the current run's local split output.

## Evaluate Refit Example

```bash
cd src
python3 pseudolabelling/evaluate_refit.py \
  --model-path ./artifacts/pseudolabelling/iter01/05_refit_model \
  --model-max-length 384 \
  --gt-jsonl ../data/dd_corpus_small_test_final.json \
  --out-dir ./artifacts/pseudolabelling/iter01/06_eval_refit \
  --labels Person,Location,Organization \
  --prediction-threshold 0.05 \
  --batch-size 4 \
  --max-tokens 384 \
  --log-level INFO
```

## Prepare Next Iteration Example

```bash
cd src
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
- `src/artifacts/pseudolabelling/`

When `--evaluate-refit` is enabled in the orchestrator, expect:

- `07_eval_base/`
- `08_eval_refit/`
- `09_base_vs_refit_comparison.json`

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
- `--boost-scope location-only`:
  boosts only entities with labels listed in `--location-labels`.
- `--boost-scope matched-only`:
  boosts only entities whose text overlaps metadata values.
- `--boost-scope location-matched-only`:
  boosts only `Location` entities whose text overlaps metadata values. This is the recommended default for the current metadata set.
- `--boost-scope all-entities`:
  boosts all entities in a matched record. This is available for ablations, but it is not the recommended default because it can inflate unrelated entity scores.

Output fields:
- per-entity: `score_context_boosted`
- per-record: `record_score_context_boosted`
- legacy aliases (default on): `score_confianca`, `score_relato_confianca`
- detailed audit artifact: `03_context_boost_details.jsonl` in orchestrated runs, including which entities were boosted and the recorded reason

## Record Score Outputs

- per-record: `record_score` (default, configurable)
- optional legacy alias: `score_relato`
- run stats JSON includes invalid-score counts and score distribution

## Split Outputs

- `kept.jsonl`
- `discarded.jsonl`
- `summary.json`
- per-record decision trace under `_split`, including optional entity-gate audit details
- Optional legacy names (`--legacy-filenames`):
  - `mantidos.jsonl`
  - `descartados.jsonl`
  - `resumo.json`

Current recommended selection rule:

- primary gate: `record_score >= threshold`
- secondary entity gate: at least one `Location` entity with `score_context_boosted >= 0.50`

This avoids keeping a report only because non-locational entities scored well.

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
