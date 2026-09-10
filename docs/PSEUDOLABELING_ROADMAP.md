# NER Pseudolabeling Roadmap

This document tracks the planned work, decisions, and progress for using unlabeled corpus entries in a pseudolabeling workflow.

## Goal

Compare data-driven boosting strategies for selecting unlabeled reports that are likely to improve NER training coverage.

The active pilot is scoped to `Location` first. `Organization` remains the weakest class and should keep receiving focused audits, but the first pseudolabeling validation experiment should not depend on improving `Organization`.

## Current Status

- The annotated train/test corpora have gone through several rounds of manual correction.
- The annotation guide has been expanded with stricter rules for locations, organizations, brands, police units, criminal groups, and overlapping spans.
- The annotation editor supports direct dataset saving, backup creation, global corrections, and string-based filtering.
- JSON-based experiment configuration has been added for quick GLiNER experiments.
- The training pipeline now stores the resulting model under `best_model`.
- OOF error auditing is available for label-focused review.
- OOF score calibration is available for analyzing how reliable model scores are by label and threshold.
- OOF score calibrator fitting is available for producing reusable per-label calibrator JSON artifacts.
- `Organization` remains the most problematic class and should receive focused review.
- A frozen fine-tuned baseline has been generated over the unlabeled corpus.
- Context boost has been applied to the frozen predictions.
- Record-level scoring now supports label filtering through `--include-labels`.
- Candidate ranking now uses a simple top-k selector over a configured record-level score.

## Main Hypothesis

Boosting strategies can identify useful unlabeled reports that would not pass the acceptance threshold using raw model scores alone.

The comparison must isolate the effect of each boosting strategy from the effect of later retraining cycles.

## Frozen Baseline

The first comparison should start from one fixed baseline:

1. Train a first model using only the manually annotated corpus.
2. Generate predictions for the unlabeled corpus.
3. Freeze those predictions as the shared baseline.
4. Apply each boosting strategy to the same frozen predictions.

Frozen baseline artifacts:

- Fine-tuned model directory: `artifacts/base_model_training/quick_supervised_only_regex/best_model`
- Experiment summary: `artifacts/base_model_training/quick_supervised_only_regex/quick_summary.json`
- Train/test corpus versions used: `data/dd_corpus_small_train.json` and `data/dd_corpus_small_test.json`
- Unlabeled input: `data/large/large_sanitized_no_labeled_overlap.jsonl`
- Predictions over the unlabeled corpus: `artifacts/pseudolabelling/frozen_baseline_regex_seed42/01_predictions.jsonl`
- Prediction stats: `artifacts/pseudolabelling/frozen_baseline_regex_seed42/01_predictions_stats.json`
- Context-boosted predictions: `artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_context_boosted.jsonl`

## Strategies to Compare

### 1. No Boost

Accept pseudolabeled reports using only the original model scores.

This is the control condition.

### 2. Semantic Match Boost

Apply the existing semantic matching strategy to the frozen baseline predictions.

This represents the previous boosting approach.

### 3. Generative AI Boost

Apply a generative AI-based strategy to the same frozen baseline predictions.

This strategy should be compared against semantic matching before any retraining happens.

## Initial Acceptance Criterion

The initial candidate rule was:

```text
mean report score > 0.8
```

This criterion has been revised for the `Location` pilot. The current preferred selection rule is fixed top-k by `record_score_location`, using `p75` over `Location` entity scores after context boost.

Rationale:

- all-label median at threshold `0.80` selected only `274` boosted reports;
- `Location` max at threshold `0.80` selected `37698` reports and was too permissive;
- `Location` p75 gives a more useful ranking signal, but still needs a fixed top-k budget.

## Comparison Metrics

For each strategy, measure:

- Number of accepted reports.
- Number of accepted spans.
- Accepted spans by label.
- Accepted reports containing `Organization`.
- Score distribution before and after boosting.
- Number of reports crossing the acceptance threshold only after boosting.
- Overlap between strategies.
- Reports accepted only by semantic match.
- Reports accepted only by generative AI.
- Estimated error rate from manual audit samples.

## Methodological Notes

- Do not use the test set to choose boosting thresholds.
- Do not compare strategies after different retraining histories.
- Use the same frozen prediction baseline for all boost comparisons.
- Keep corpus versions, model versions, config files, and thresholds explicit.
- Treat GLiNER scores as ranking/confidence signals, not calibrated probabilities, unless calibration evidence supports that interpretation.
- Analyze thresholds by label when possible, since `Organization` behaves differently from `Location` and `Person`.
- For future training commands on `workstation02`, prefer
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` after confirming that the GLiNER
  and backbone model snapshots are already cached. This avoids Hugging Face
  metadata checks during repeated runs and keeps experiments tied to the cached
  model revision.

## Near-Term Plan

1. Fit a reusable OOF score calibrator from `artifacts/error_analysis/train_oof_regex_for_boost_factor/oof_predictions.jsonl`.
2. Apply the calibrator to the frozen unlabeled predictions, producing `score_calibrated`.
3. Use `artifacts/metadata/app_dd_labeled_metadata_matches.jsonl` as the metadata source for train OOF context-boost optimization.
4. Use `src/tools/optimize_context_boost_factor.py` to choose a data-driven context boost factor from calibrated OOF predictions enriched with `app_dd.xlsx` metadata.
5. Re-apply context boost to the calibrated frozen unlabeled predictions with the selected factor.
6. Generate the top-k `Location` candidate files from the updated boosted/scored predictions.
7. Manually inspect the top-1000 HTML review artifact.
8. Freeze one or more candidate volumes, starting with `1000`.
9. Build refit-compatible pseudolabel inputs from the selected candidates.
10. Run a controlled `supervised_only` vs `supervised_plus_pseudolabels` refit comparison.
11. Use `Location` F1 as the primary success metric and monitor micro/macro F1 plus `Person`/`Organization` regressions.
12. Only after the `Location` pilot is stable, compare semantic/context boost against a generative AI boost over the same frozen predictions.

## TODO

- Apply the same `app_dd.xlsx` matching strategy to the large unlabeled corpus to measure overlap and identify additional non-labeled records with recoverable metadata. This should be used for corpus accounting and possible metadata enrichment, while preserving the no-labeled-overlap constraint for pseudolabel selection.
- Evaluate a metadata- and rule-aware underboost strategy for reducing confidence in likely false-positive entities. This should be optimized on OOF predictions before use, measuring removed false positives, lost true positives, class-level F1 changes, and especially effects on `Location` and `Organization`. Treat underboost first as an audit/selection mechanism, not as an automatic training-data transformation.

## Location-First Pilot

The first pseudolabeling pilot will focus on validating whether the approach can improve `Location`.

This is a deliberately scoped experiment:

- The contextual metadata available in the large corpus mostly supports geographic entities.
- The current context boost implementation is primarily designed for `Location`.
- `Location` has the largest support and the strongest baseline performance.
- `Organization` and `Person` will still be monitored, but they are not the primary success criterion for this pilot.

The training unit remains the full report. If a boosted report is accepted, the refit stage may consume all pseudolabeled entities in that report, not only `Location` entities. This is acceptable for the pilot because the purpose is to test whether contextual pseudolabeling can improve `Location` under the current pipeline design.

Primary metric:

- `Location` F1 on the fixed labeled test set.

Secondary metrics:

- micro F1;
- macro F1;
- `Person` F1;
- `Organization` F1;
- number of accepted pseudolabeled reports;
- entity-label distribution in the accepted pseudolabel set.

Interpretation rule:

- A gain in `Location` is the main positive signal.
- Small regressions in `Person` or `Organization` do not automatically invalidate the pilot.
- Large regressions in micro F1, macro F1, or non-Location labels should be treated as evidence that the accepted full-report pseudolabels are too noisy.
- If the full-report approach improves `Location` but harms other labels, a follow-up variant should test a `Location`-only pseudolabel artifact.

## Open Questions

- Which unlabeled corpus file is the canonical source for pseudolabeling?
- Which first fine-tuned model should be treated as the frozen baseline?
- Should acceptance use one global threshold or label-specific thresholds?
- Should report-level score be the mean, minimum, median, or a weighted aggregate of span scores?
- How should reports with many high-confidence `Location` spans but weak `Organization` spans be handled?
- How large should the manual audit sample be for each strategy?

## Progress Log

### 2026-09-07

- Synchronized the current labeled train/test/calibration corpora and the large no-overlap corpus to `workstation02`; SHA-256 checksums matched across machines.
- Re-ran the supervised-only regex baseline and the conservative `Location` pseudolabel condition on the same current train/test corpora.
- Built the conservative pseudolabel condition from `score_calibrated` with:
  - record selection: top 1000 records from the `t097` pool;
  - entity selection: `Location` spans only;
  - pseudolabel volume: `1000` reports and `2103` `Location` entities.
- Completed a three-seed comparison for seeds `42`, `43`, and `44`.
- Aggregate result for `quick_supervised_plus_location_pseudolabels_calibrated_t097_top1000_repeats3_current` versus `quick_supervised_only_regex_repeats3_current`:
  - micro F1: `+0.001526 +/- 0.001843`;
  - macro F1: `+0.003229 +/- 0.005157`;
  - `Location` F1: `+0.000983 +/- 0.000963`;
  - `Organization` F1: `+0.005216 +/- 0.010930`;
  - `Person` F1: `+0.003488 +/- 0.005075`.
- Interpretation: the `top1000` pseudolabel condition produced a small positive signal, with `Location` improving in all three seeds, but the effect is still too small and variable to treat as a robust pipeline improvement.
- Completed a repeated-seed top-k curve over the calibrated `t097` `Location` pseudolabel pool:
  - `top500`: `500` reports and `1033` `Location` entities;
  - `top1000`: `1000` reports and `2103` `Location` entities;
  - `top2000`: `2000` reports and `4475` `Location` entities;
  - `top3000`: `3000` reports and `7062` `Location` entities.
- Top-k curve aggregate deltas versus the supervised-only repeated baseline:
  - `top500`: micro F1 `+0.002261`, macro F1 `+0.003203`, `Location` F1 `+0.002437`;
  - `top1000`: micro F1 `+0.001526`, macro F1 `+0.003229`, `Location` F1 `+0.000983`;
  - `top2000`: micro F1 `+0.000901`, macro F1 `+0.002469`, `Location` F1 `+0.000316`;
  - `top3000`: micro F1 `-0.001331`, macro F1 `-0.002470`, `Location` F1 `-0.000741`.
- Interpretation: the positive effect is volume-sensitive. `top500` is the strongest current condition, improving `Location` in all three seeds; larger volumes dilute the gain and `top3000` becomes harmful.
- Current decision: treat `top500` as the main pseudolabel condition and `top1000` as a secondary condition for comparison.
- Next step: qualitatively audit the `top500` pseudolabel set to characterize the extra `Location` evidence it adds and identify systematic noise before expanding the pseudolabeling strategy.
- Qualitative audit of `top500` found strong concentration in repeated `Location` strings and near-duplicate reports. A diverse selector was added as the next controlled variant: keep the same `top500` budget, but apply normalized text deduplication plus caps per `Location` term and per `Location`-set signature.
- The first strict diversity condition (`max_per_signature=2`) retained a small `Location` gain but underperformed the pure `top500` condition and reduced `Person` F1. Its caps likely removed useful recurring high-confidence contexts.
- Next experiment: run the softer `top500_diverse_sig5` condition with the same three seeds, preserving exact-text deduplication and the `max_per_entity=10` cap while relaxing `max_per_signature` from `2` to `5`.

## Top-K Curve Plan

The next experiment should vary only the number of accepted pseudolabeled records while keeping the base model, train/test corpora, tokenization, hyperparameters, threshold, and seeds fixed.

Candidate volumes:

- `top500`;
- `top1000`;
- `top2000`;
- `top3000`;
- optionally `top5000` if disk and training time permit.

Controls:

- supervised-only regex baseline with the same seeds;
- same calibrated `Location` candidate pool;
- same entity filter, keeping only `Location` spans from the selected pseudolabel records;
- same repeated seeds, initially `42`, `43`, and `44`;
- same fixed test set.

Primary analysis:

- mean and standard deviation of `Location` F1 by top-k size;
- per-seed deltas against the supervised-only baseline;
- micro and macro F1 deltas;
- `Organization` and `Person` regressions.

Decision rule:

- Prefer the smallest top-k that improves mean `Location` F1 without consistent degradation in micro/macro F1 or non-Location labels.
- If the curve is flat or unstable, treat pseudolabeling as not yet validated and move to error analysis or underboost/negative-filtering experiments.

Current outcome:

- `top500` is the selected operating point for the next pilot stage.
- `top1000` remains a useful secondary point.
- `top2000` and `top3000` should not be used as default conditions without additional filtering or diversity controls.

### 2026-09-04

- Refactored `src/tools/mine_train_oof_errors.py` so OOF outputs preserve row identity, low-threshold predictions, eval-threshold predictions, and optional metadata enrichment.
- Added `src/tools/optimize_context_boost_factor.py` to simulate context boost factors over OOF predictions without retraining.
- Found that using `data/large/large_sanitized_no_labeled_overlap.jsonl` as the metadata source for train OOF makes context-boost optimization a no-op, because the file intentionally excludes labeled-overlap records.
- Extended `src/tools/optimize_context_boost_factor.py` so it can enrich OOF rows from explicit metadata source files by unique normalized text match, avoiding the need to rerun OOF just to recover metadata.
- Added `src/tools/extract_app_dd_metadata_matches.py` to recover original `app_dd.xlsx` metadata for the labeled small corpora. The first run recovered unique geographic metadata for `3792/5230` labeled rows, including `3087/4226` train rows.
- Added reusable OOF NER score calibration scripts: `src/tools/fit_ner_score_calibrator_oof.py` and `src/tools/apply_ner_score_calibrator.py`.

### 2026-09-01

- Updated the pipeline documentation to match the current JSON-configured quick-training and pseudolabeling workflow.
- Standardized the active large-corpus input as `data/large/large_sanitized_no_labeled_overlap.jsonl`.
- Documented the frozen baseline at `artifacts/base_model_training/quick_supervised_only_regex/best_model`.
- Documented the first frozen prediction run:
  - input rows: `182008`
  - processed rows: `182008`
  - failed rows: `0`
  - predicted entities: `2087284`
- Documented no-boost vs context-boost split behavior:
  - no boost, all-label median, threshold `0.80`: `273` kept
  - context boost, all-label median, threshold `0.80`: `274` kept
  - context boost, `Location` max, threshold `0.80`: `37698` kept
  - context boost, `Location` p75, threshold `0.80`: `14255` kept
- Decided that the next selection step should use fixed top-k volumes over `Location` p75 scores instead of a raw threshold-only rule.
- Simplified `src/tools/rank_pseudolabel_candidates.py` into a top-k selector over record-level scores.

### 2026-08-31

- Created this roadmap.
- Current next step: locate or generate the frozen baseline predictions over the unlabeled corpus.
- Selected a `Location`-first pilot as the first pseudolabeling validation experiment.
- The pilot will use full-report pseudolabels and evaluate success primarily through `Location` F1.
