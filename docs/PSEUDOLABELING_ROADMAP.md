# NER Pseudolabeling Roadmap

This document tracks the planned work, decisions, and progress for using unlabeled corpus entries in a pseudolabeling workflow.

## Goal

Compare data-driven boosting strategies for selecting unlabeled reports that are likely to improve NER training coverage, especially for the weakest class, `Organization`.

## Current Status

- The annotated train/test corpora have gone through several rounds of manual correction.
- The annotation guide has been expanded with stricter rules for locations, organizations, brands, police units, criminal groups, and overlapping spans.
- The annotation editor supports direct dataset saving, backup creation, global corrections, and string-based filtering.
- JSON-based experiment configuration has been added for quick GLiNER experiments.
- The training pipeline now stores the resulting model under `best_model`.
- OOF error auditing is available for label-focused review.
- OOF score calibration is available for analyzing how reliable model scores are by label and threshold.
- `Organization` remains the most problematic class and should receive focused review.

## Main Hypothesis

Boosting strategies can identify useful unlabeled reports that would not pass the acceptance threshold using raw model scores alone.

The comparison must isolate the effect of each boosting strategy from the effect of later retraining cycles.

## Frozen Baseline

The first comparison should start from one fixed baseline:

1. Train a first model using only the manually annotated corpus.
2. Generate predictions for the unlabeled corpus.
3. Freeze those predictions as the shared baseline.
4. Apply each boosting strategy to the same frozen predictions.

Expected baseline artifacts:

- Fine-tuned model directory.
- Experiment config used to train the model.
- Train/test corpus versions used.
- Predictions over the unlabeled corpus.
- Span-level labels, offsets, text, and scores.
- Report-level aggregate scores.
- Acceptance threshold or thresholds.

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

The initial candidate rule is:

```text
mean report score > 0.8
```

This criterion may be revised after score calibration, especially if different labels require different thresholds.

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

## Near-Term Plan

1. Identify or generate predictions from the first fine-tuned model over the unlabeled corpus.
2. Define the exact prediction file schema for downstream boosting.
3. Compute report-level aggregate scores.
4. Produce the no-boost accepted set.
5. Apply semantic match boosting.
6. Apply generative AI boosting.
7. Compare accepted sets.
8. Sample accepted reports for manual audit.
9. Decide which pseudolabels are eligible for retraining.
10. Run retraining only after the first-stage comparison is complete.

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

### 2026-08-31

- Created this roadmap.
- Current next step: locate or generate the frozen baseline predictions over the unlabeled corpus.
- Selected a `Location`-first pilot as the first pseudolabeling validation experiment.
- The pilot will use full-report pseudolabels and evaluate success primarily through `Location` F1.
