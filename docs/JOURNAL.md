# Journal

Operational notes about non-trivial project changes, debugging findings, and behavior corrections that are easy to lose across refactors.

## 2026-03-16

### Encoder-Aware Chunking In Training

The training pipeline previously split long samples using whitespace token counts only. That could still produce examples longer than the transformer encoder budget used by GLiNER, which surfaced as warnings such as:

```text
Sentence of length 451 has been truncated to 384
```

The fix was implemented in `src/base_model_training/data.py`:

- chunk sizing now uses the GLiNER encoder tokenizer when available
- the chunk budget reserves space for special tokens
- overlap is still applied in word units
- a warning is emitted only for pathological cases where a single word expands beyond the encoder budget by itself

Implication:

- `max_length` in training now better matches the real encoder limit
- fewer examples should be silently truncated inside the GLiNER collator
- runtime estimates and training quality should now be more trustworthy on long texts

### Remote GPU Compatibility Note

On an NVIDIA RTX 5090, the environment required a PyTorch build with `cu128` so CUDA kernels would support `sm_120`. Older builds failed with:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### Early Stopping Patience Logging

The training log previously reported `Patience` before updating the counter for the current epoch. That made lines such as `Patience=3/7` misleading because the value shown was effectively ahead of the true post-epoch state.

The fix was implemented in `src/base_model_training/engine.py` so the log now reflects the actual patience counter after the validation metric has been evaluated.

### Training Stage Labels In Logs

The nested-CV training flow can run several training stages back-to-back:

- inner fold training during hyperparameter search
- refit training after selecting the best parameters for an outer fold

Without explicit stage labels, epoch logs looked like a continuous stream and it was unclear what had changed between one batch of epochs and the next.

The fix was implemented in `src/base_model_training/cv.py` and `src/base_model_training/engine.py`:

- each training stage now emits an explicit `Starting training stage: ...` log line
- epoch logs now carry a stage prefix such as `outer 1/3 | trial 2/4 | inner 3/3`
- refit epochs are logged as their own stage, for example `outer 1/3 | refit`

### Early Stopping Metric Alignment

The training loop previously used a fixed `early_stopping_threshold` to compute validation F1 for early stopping, while model selection later chose the best threshold from `config.thresholds`.

That meant early stopping could optimize for one threshold while the final fold selection optimized for another.

The fix was implemented in `src/base_model_training/cv.py`:

- early stopping now uses the best validation F1 across the configured `thresholds`
- `early_stopping_threshold` remains only as a compatibility fallback when no threshold list is available

Implication:

- checkpoint stopping is now aligned with the same threshold search space used for fold selection
- the monitored validation metric is more faithful to the final objective of the run

### Group-Aware Multilabel Fold Balancing

The original nested-CV implementation used `GroupKFold` on `sample_id`, which prevented leakage across chunks from the same report but did not attempt to balance label distributions between folds.

That meant outer and inner folds could drift substantially in the distribution of entity labels, especially for rarer classes.

The fix was implemented in `src/base_model_training/group_stratified.py` and integrated into `src/base_model_training/cv.py`:

- splitting still happens at the `sample_id` level
- each group is represented by a multilabel profile and label span counts
- groups are assigned to folds with a greedy cost function that balances group count, example count, label presence, and label span counts
- fold summaries are now logged for audit

Implication:

- folds should remain leakage-safe while becoming more comparable in label composition
- nested-CV estimates should be less sensitive to accidental label imbalance across folds

### Input Dataset `sample_id` Validation

The training loader now emits explicit logging about `sample_id` quality in the input dataset:

- missing `sample_id` values are auto-generated and counted
- duplicated `sample_id` values in the raw input are warned about

This matters because nested CV treats `sample_id` as the grouping key for leakage prevention and fold balancing.

### Calibration Redesign

The original calibration flow had two conceptual problems:

- it was wired as a pseudolabelling substep instead of an artifact attached to the base model
- the iterative pipeline could run calibration and still continue using the raw `score` field downstream unless manually overridden

The redesign introduces an explicit fit/apply split:

- `src/calibration/fit_calibrator.py` fits a reusable calibrator artifact from a labeled calibration CSV
- `src/calibration/apply_calibrator.py` applies that artifact to a JSONL corpus with entity scores
- `src/pseudolabelling/pipeline.py` can now load a calibrator artifact during prediction and write `score_calibrated` alongside the raw score

Implication:

- calibration is now aligned with the original goal of correcting base-model confidence estimates
- the calibrator becomes a reusable artifact that can be versioned together with the model
- large-corpus prediction no longer needs to refit calibration ad hoc on each run

## 2026-03-21

### Pseudolabelling Refit Modes And Experimental Separation

The pseudolabelling refit flow now exposes an explicit `refit_mode` to prevent a methodological ambiguity that surfaced during validation:

- `supervised_only`
- `supervised_plus_pseudolabels`
- `pseudolabel_only`

This was necessary because a supervised-only refit from the base checkpoint already improved holdout performance, which means any later gain from pseudolabelling must be measured against that supervised-only baseline, not only against the base model.

Implication:

- the marginal contribution of pseudolabelling should be interpreted as:
  - `(supervised_plus_pseudolabels) - (supervised_only)`
- not merely as:
  - `(supervised_plus_pseudolabels) - (base)`

### Refit Backend Now Chunks Long Examples Before GLiNER Collation

The pseudolabelling refit backend previously sent long examples straight into the GLiNER processor, which produced warnings like:

```text
Sentence of length 777 has been truncated to 384
```

The fix was implemented in `src/pseudolabelling/refit_backend.py` by reusing the same preprocessing path as base-model training:

- `process_sample`
- `split_long_sentences`

with configurable `max_length` and `overlap`.

Implication:

- refit now handles long reports more consistently with the main training pipeline
- large silent truncation during refit should be reduced substantially

### Report-Level Selection Versus Entity-Level Filtering

The pseudolabelling split logic currently selects at the report level:

- compute a `record_score`
- keep or discard the full report

Once a report is kept, all predicted entities from that report enter refit. There is no second-stage filtering that drops low-confidence entities inside a kept report.

Implication:

- the current pseudolabelling experiments are report-selection experiments
- they are not yet span-filtering experiments

### First Controlled Pseudolabelling Result (`t020`)

The first controlled pair of runs that cleanly separated supervised refit from pseudolabel augmentation used `split-threshold=0.20`.

Observed counts:

- kept reports: `12`
- supervised-plus-pseudolabels training mix:
  - `3031` supervised examples
  - `11` pseudolabel examples in train
  - `1` pseudolabel example in validation

Observed holdout metrics on `data/dd_corpus_small_test_filtered.json`:

- base:
  - `micro_f1=0.4440`
  - `macro_f1=0.4362`
- supervised-only refit:
  - `micro_f1=0.4952`
  - `macro_f1=0.4772`
- supervised-plus-pseudolabels refit:
  - `micro_f1=0.5260`
  - `macro_f1=0.5097`

Implication:

- additional supervised refit explains part of the gain
- pseudolabel augmentation explains an additional positive margin beyond that supervised-only baseline
