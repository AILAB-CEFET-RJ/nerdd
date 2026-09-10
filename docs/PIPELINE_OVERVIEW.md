# Pipeline Overview

This document summarizes the current NER workflow for annotation, supervised
training, score auditing, pseudolabel selection, and controlled refit
experiments.

## Operating Assumptions

- Run commands from the repository root whenever possible.
- Keep canonical input datasets in `data/`.
- Keep generated artifacts in `artifacts/`.
- Run expensive training and large-corpus inference on `workstation02`.
- After the Hugging Face model cache is warmed, run training commands with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to avoid network metadata checks
  and keep repeated runs tied to the cached snapshot.
- Use the annotation guide in `docs/LABELLING_GUIDE.md` as the source of truth
  for `Person`, `Location`, and `Organization` spans.

Current canonical labeled inputs:

- `data/dd_corpus_small_train.json`
- `data/dd_corpus_small_test.json`
- `data/dd_corpus_small_calibration.json`

Current large unlabeled input for the frozen-baseline pilot:

- `data/large/large_sanitized_no_labeled_overlap.jsonl`

Older artifacts and commands may still mention
`artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl` or
`data/large_sanitized/large_sanitized.jsonl`. Treat those as legacy paths unless
an experiment explicitly depends on them.

## Stage Naming Convention

Use `number + semantic_name` for pipeline stages and derived artifact names.

Examples:

- `01_predictions`
- `02_scored_no_boost`
- `04_context_boosted`
- `05d_scored_context_boost_location_p75`
- `07_location_p75_top1000`

This preserves execution order in directory listings and keeps handoffs
unambiguous.

## Subpipeline Boundaries

### Annotation And Corpus QA

Scope:

- edit `spans` in train/test/calibration corpora
- enforce the annotation guide
- find likely missed repeated spans
- audit model errors by label
- calibrate model scores from OOF predictions

Important entrypoints:

- `src/tools/serve_ner_annotation_editor.py`
- `src/tools/build_ner_annotation_editor_global.py`
- `src/tools/find_missing_repeated_spans.py`
- `src/tools/mine_train_oof_errors.py`
- `src/tools/audit_ner_errors_by_label.py`
- `src/tools/calibrate_ner_scores.py`
- `src/tools/fit_ner_score_calibrator_oof.py`
- `src/tools/apply_ner_score_calibrator.py`
- `src/tools/extract_app_dd_metadata_matches.py`

The server editor can save directly back to the dataset and creates timestamped
backups beside the edited JSON file.

The legacy spreadsheet `data/app_dd.xlsx` can be used to recover original
location metadata for many labeled small-corpus rows. Use
`src/tools/extract_app_dd_metadata_matches.py` to produce an auditable JSONL/CSV
mapping before using those metadata in context-boost experiments.

### Supervised Training And Evaluation

Scope:

- train or evaluate GLiNER on the manually annotated corpus
- run single experiments or JSON-configured sweeps
- preserve model dumps and summaries

Current primary entrypoints:

- `src/base_model_training/train_quick.py`
- `src/tools/run_experiment_config.py`
- config files in `configs/experiments/`

`train_quick.py` writes:

- `best_model/`
- `eval_test/`
- `quick_summary.json`
- `effective_config.json`

The previous nested-CV pipeline still exists, but the current working baseline
uses the JSON-configured quick-training path.

Current frozen baseline:

- model: `artifacts/base_model_training/quick_supervised_only_regex/best_model`
- summary: `artifacts/base_model_training/quick_supervised_only_regex/quick_summary.json`
- mode: `supervised_only`
- tokenizer strategy: `regex`
- seed: `42`
- selected threshold: `0.6`

Observed holdout metrics for this baseline:

- micro F1: `0.8711`
- macro F1: `0.8271`
- `Location` F1: `0.8924`
- `Organization` F1: `0.7138`
- `Person` F1: `0.8751`

### Pseudolabeling

Scope:

- run the frozen baseline over the large unlabeled corpus
- compute record-level scores from entity-level scores
- apply metadata/context boosts
- select high-value candidate reports for manual review or refit
- compare no-boost and boosted candidate sets before retraining

Important entrypoints:

- `src/pseudolabelling/generate_corpus_predictions.py`
- `src/pseudolabelling/apply_context_boost.py`
- `src/pseudolabelling/compute_record_scores.py`
- `src/pseudolabelling/split_pseudolabels.py`
- `src/tools/rank_pseudolabel_candidates.py`
- `src/tools/optimize_context_boost_factor.py`

Current frozen-baseline artifact root:

- `artifacts/pseudolabelling/frozen_baseline_regex_seed42/`

Observed prediction run:

- input rows: `182008`
- processed rows: `182008`
- failed rows: `0`
- predicted entities: `2087284`
- output: `01_predictions.jsonl`

### Controlled Refit

Scope:

- test whether selected pseudolabels improve the final model
- isolate pseudolabel effects from extra supervised fine-tuning

Required comparison conditions:

- `base`
- `supervised_only`
- `supervised_plus_pseudolabels`

Only the pseudolabel input should vary between `supervised_only` and
`supervised_plus_pseudolabels`; seed, checkpoint, train corpus, validation
scheme, hyperparameters, and final holdout must remain fixed.

## Current Location-First Pilot

The active pseudolabeling direction is a `Location`-first pilot.

Rationale:

- metadata context in the large corpus mostly supports geographic entities
- `Location` has the largest support and the strongest baseline behavior
- `Organization` remains noisy and should be monitored, not optimized first

Training still consumes full reports in the current refit path. Therefore,
accepted reports may carry `Person` and `Organization` pseudolabels too. For
this pilot, success is primarily measured by `Location` F1 on the fixed test
set; large regressions in other labels remain a warning signal.

## Frozen Prediction Flow

Run on `workstation02` from the repository root:

```bash
PYTHONPATH=src python3 -m pseudolabelling.generate_corpus_predictions \
  --model-path artifacts/base_model_training/quick_supervised_only_regex/best_model \
  --model-max-length 384 \
  --input-jsonl data/large/large_sanitized_no_labeled_overlap.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/01_predictions.jsonl \
  --stats-json artifacts/pseudolabelling/frozen_baseline_regex_seed42/01_predictions_stats.json \
  --labels Person,Location,Organization \
  --text-fields relato \
  --max-tokens 384 \
  --batch-size 16 \
  --score-threshold 0.0 \
  --keep-inference-text \
  --log-level INFO
```

This artifact is the shared baseline for no-boost, semantic/context boost, and
future generative-boost comparisons.

## Context Boost Flow

`pseudolabelling.apply_context_boost` raises entity scores when model-predicted
locations are supported by location metadata that appears literally in the
report text.

The boost factor should be selected from OOF predictions on the labeled training
corpus with `src/tools/optimize_context_boost_factor.py`, not tuned on the final
test set.

If the OOF artifact lacks location metadata, recover it inside the optimizer
with `--metadata-sources`. Do not use
`data/large/large_sanitized_no_labeled_overlap.jsonl` for this recovery step:
that file intentionally removes overlap with labeled corpora, so train rows will
not match and boost simulation becomes a no-op.

Current factor-optimization command:

```bash
PYTHONPATH=src python3 src/tools/optimize_context_boost_factor.py \
  --oof-predictions artifacts/error_analysis/train_oof_regex_for_boost_factor/oof_predictions.jsonl \
  --output-dir artifacts/boost_factor_optimization/context_location_oof_with_metadata_sources \
  --metadata-sources data/large_sanitized/large_sanitized.jsonl,data/large_sanitized/large_dropped.jsonl,data/large_sanitized/large_flagged.jsonl \
  --metadata-source-text-fields relato,text \
  --boost-factors 1.00,1.05,1.10,1.15,1.20,1.30,1.50 \
  --target-label Location \
  --score-thresholds 0.6,0.7,0.8,0.9,0.95 \
  --record-score-aggregation p75 \
  --record-thresholds 0.8,0.9,0.95 \
  --precision-floor 0.90 \
  --boost-scope location-matched-only \
  --match-policy any-metadata-in-text \
  --log-level INFO
```

Current pilot configuration:

```bash
PYTHONPATH=src python3 -m pseudolabelling.apply_context_boost \
  --input-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/01_predictions.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_context_boosted.jsonl \
  --stats-json artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_context_boost_stats.json \
  --boosted-entities-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_context_boost_boosted_entities.jsonl \
  --base-score-field score \
  --output-score-field score_context_boosted \
  --output-record-score-field record_score_context_boosted \
  --boost-factor 1.2 \
  --boost-scope location-matched-only \
  --match-policy any-metadata-in-text \
  --log-level INFO
```

Observed run:

- rows: `182008`
- metadata matches: `102495`
- boosted entities: `30891`

## Record Score Flow

`pseudolabelling.compute_record_scores` aggregates entity-level scores into a
record-level score.

For broad full-entity acceptance, use `median` over all labels:

```bash
PYTHONPATH=src python3 -m pseudolabelling.compute_record_scores \
  --input-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_context_boosted.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/05_scored_context_boost.jsonl \
  --stats-json artifacts/pseudolabelling/frozen_baseline_regex_seed42/05_scored_context_boost_stats.json \
  --score-field score_context_boosted \
  --output-field record_score \
  --legacy-field-alias score_relato \
  --aggregation median \
  --dedupe-mode label_text \
  --empty-entities-policy zero \
  --log-level INFO
```

For the active `Location` pilot, filter scoring to `Location`:

```bash
PYTHONPATH=src python3 -m pseudolabelling.compute_record_scores \
  --input-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_context_boosted.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/05d_scored_context_boost_location_p75.jsonl \
  --stats-json artifacts/pseudolabelling/frozen_baseline_regex_seed42/05d_scored_context_boost_location_p75_stats.json \
  --score-field score_context_boosted \
  --output-field record_score_location \
  --legacy-field-alias score_relato_location \
  --include-labels Location \
  --aggregation p75 \
  --dedupe-mode label_text \
  --empty-entities-policy zero \
  --log-level INFO
```

Observed lesson:

- median over all entities was too conservative for a `Location` pilot
- `max` over `Location` was too permissive and saturated candidate counts
- `p75` over `Location` is the current ranking score to inspect first

## Candidate Selection Flow

`src/tools/rank_pseudolabel_candidates.py` is now a simple top-k selector over a
record-level score. It no longer applies the old density and label-ratio
heuristics.

Current top-1000 command:

```bash
PYTHONPATH=src python3 src/tools/rank_pseudolabel_candidates.py \
  --input artifacts/pseudolabelling/frozen_baseline_regex_seed42/05d_scored_context_boost_location_p75.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/07_location_p75_top1000.jsonl \
  --output-csv artifacts/pseudolabelling/frozen_baseline_regex_seed42/07_location_p75_top1000.csv \
  --output-html artifacts/pseudolabelling/frozen_baseline_regex_seed42/07_location_p75_top1000.html \
  --summary-json artifacts/pseudolabelling/frozen_baseline_regex_seed42/07_location_p75_top1000_summary.json \
  --score-fields record_score_location \
  --required-labels Location \
  --min-score 0.80 \
  --top-n 1000 \
  --title "Location p75 top 1000 pseudolabel candidates"
```

The output row metadata `_pseudolabel_selection` records source row, rank, score,
score field, text length, entity count, and label counts.

When the top-k set is too concentrated in repeated places or near-duplicate
reports, use `src/tools/select_diverse_pseudolabels.py` on the same candidate
pool. It keeps the fixed top-k budget but applies normalized text deduplication
and caps per `Location` term and per `Location`-set signature. This is the
current follow-up after the `top500` audit showed strong concentration in a few
locations.

```bash
PYTHONPATH=src python3 src/tools/select_diverse_pseudolabels.py \
  --input artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500_diverse.jsonl \
  --summary-json artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500_diverse_summary.json \
  --audit-csv artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500_diverse_audit.csv \
  --output-html artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500_diverse.html \
  --top-n 500 \
  --score-fields record_score_location,_pseudolabel.record_score_location \
  --target-labels Location \
  --max-per-entity 10 \
  --max-per-signature 2 \
  --signature-max-terms 8 \
  --title "Diverse Location pseudolabels t097 top500"
```

## Split Flow

`pseudolabelling.split_pseudolabels` is still useful for threshold-based counts
and ablations, but the active pilot should not rely only on a hard threshold
because `Location` scores can yield very large kept sets.

Observed counts:

- no boost, all-label median, threshold `0.80`: `273` kept
- context boost, all-label median, threshold `0.80`: `274` kept
- context boost, `Location` max, threshold `0.80`: `37698` kept
- context boost, `Location` p75, threshold `0.80`: `14255` kept

For the next refit experiment, prefer fixed top-k volumes such as `1000`, `3000`,
and `5000`.

## Calibration Flow

Use `src/tools/calibrate_ner_scores.py` on OOF prediction artifacts to understand
how raw GLiNER scores behave by label and threshold.

Use `src/tools/fit_ner_score_calibrator_oof.py` when a reusable score calibrator
is needed. It fits per-label calibrators from out-of-fold predictions only, so
the final test set remains untouched. The saved `calibrator.json` can then be
applied to frozen large-corpus predictions with `src/tools/apply_ner_score_calibrator.py`
or passed to the prediction pipeline through the existing calibrator hook.

Current OOF calibrator command:

```bash
PYTHONPATH=src python3 src/tools/fit_ner_score_calibrator_oof.py \
  --oof-predictions artifacts/error_analysis/train_oof_regex_for_boost_factor/oof_predictions.jsonl \
  --output-dir artifacts/calibration/ner_score_calibrator_oof_regex \
  --labels Person,Location,Organization \
  --method isotonic \
  --score-field score \
  --pred-field pred_spans \
  --gold-field gold_spans \
  --min-positive 20 \
  --min-negative 20 \
  --bins 10 \
  --log-level INFO
```

Interpretation:

- GLiNER scores are useful ranking signals
- do not treat them as calibrated probabilities without evidence
- inspect precision by score bin separately for `Organization`, `Location`, and
  `Person`

## Methodological Rules

- Do not use the final test set to select thresholds or pseudolabel volumes.
- Use one frozen prediction file when comparing boost strategies.
- Compare boost strategies before retraining.
- Include a `supervised_only` control before claiming a pseudolabel gain.
- Keep corpus versions, model path, config file, thresholds, and top-k volume
  explicit in every run.
- Treat `Organization` separately in audits; it has different error behavior
  from `Location`.

## Main Artifact Roots

- `artifacts/base_model_training/`
- `artifacts/error_analysis/`
- `artifacts/calibration/`
- `artifacts/pseudolabelling/`
- `artifacts/annotation_review/`
