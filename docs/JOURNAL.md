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

## 2026-03-17

### Stratified Group Splitter No Longer Emits Empty Folds

The first version of `StratifiedGroupKFoldNER` used a purely greedy assignment strategy. In some skewed datasets, that heuristic could place all groups into only a subset of folds and leave another fold empty.

The failure mode appeared in nested-CV logs such as:

```text
Outer CV fold 3 summary | groups=0 | examples=0 | spans=[Location=0, Organization=0, Person=0]
```

That behavior is invalid for `n_splits=3` because the run is no longer performing a real 3-fold split. It also propagated to inner CV, where empty folds were later skipped.

The fix was implemented in `src/base_model_training/group_stratified.py`:

- fold construction now starts with mandatory seeding of one group per fold
- the remaining groups are assigned greedily using the balancing objective
- a local-search refinement step now tries beneficial group moves and swaps across folds
- the global cost now penalizes empty folds, missing labels, group imbalance, example imbalance, and span imbalance
- the splitter now raises if an empty fold somehow survives refinement

Validation:

- `src/tests/test_group_stratified.py` now includes a regression test asserting that all emitted folds are non-empty

Implication:

- outer and inner CV should now always emit the requested number of non-empty folds
- fold summaries should better reflect the dataset distribution without silently dropping a fold

### Stratified Group Splitter Now Enforces Fold Capacity And Global Label Balance

After the empty-fold fix, a second failure mode remained visible in nested-CV logs: fold sizes became balanced by group count, but some folds still received almost none of the rarer entity labels.

That produced summaries shaped like:

```text
Outer CV fold 1 summary | groups=1222 | examples=1393 | spans=[Location=5515, Organization=2043, Person=3416]
Outer CV fold 3 summary | groups=1221 | examples=1227 | spans=[Location=3220, Organization=42, Person=26]
```

This was better than an empty fold, but still not a faithful stratified split for NER.

The splitter in `src/base_model_training/group_stratified.py` was tightened again:

- each fold now has an explicit group-capacity limit derived from `n_groups / n_splits`
- greedy placement can only assign to folds that still have remaining capacity
- local-search moves also respect those fold-capacity constraints
- candidate assignment quality is now evaluated using the global partition cost after a temporary assignment, instead of only a local fold-level estimate

Validation:

- `src/tests/test_group_stratified.py` now also includes a regression case asserting that rare-label groups are distributed across folds when the dataset makes that possible

Implication:

- folds should now stay balanced both in size and in rare-label coverage more reliably than before
- training diagnostics should be less likely to be distorted by a fold that is large enough but nearly devoid of one entity type

### Split Learning Rates For Backbone And NER Layers

The nested-CV training pipeline previously used a single global learning rate for every trainable parameter in GLiNER.

That meant the transformer backbone and the task-specific NER layers were optimized with the same step size, which is a poor fit for fine-tuning in domain-specific NER runs.

The fix was implemented in `src/base_model_training/cv.py`, `src/base_model_training/cli.py`, `src/base_model_training/search.py`, and `src/base_model_training/train_config.py`:

- `--lr-values` was removed from the training CLI
- the training entrypoint now requires `--backbone-lr-values` and `--ner-lr-values`
- hyperparameter search now evaluates `(backbone_lr, ner_lr, weight_decay)` combinations
- the optimizer now uses two parameter groups:
  - `model.token_rep_layer.*` for the transformer backbone
  - every other trainable parameter for the NER-specific layers
- `OneCycleLR` now tracks the two learning rates separately

The surrounding reporting was updated as well:

- trial logs and best-parameter logs now print backbone LR and NER LR explicitly
- text and JSON reports now store `backbone_lr` and `ner_lr` instead of a single `lr`
- loss plots now include both learning rates in the title

Implication:

- the training search space now matches the common fine-tuning pattern of using a smaller LR for the pretrained backbone and a larger LR for task-specific layers
- nested-CV runs should be able to tune stability and adaptation speed independently

### Weighted Training Sampling For Rare Entity Labels

Even after the fold splitter was corrected, the training `DataLoader` still used plain shuffled batches.

That meant individual batches could remain heavily skewed toward the most common entity label, especially after long reports were chunked into smaller training examples.

The training pipeline now supports two sampling modes in `base_model_training.train_nested_kfold`:

- `random`: the previous behavior, using only shuffled batches
- `weighted`: a label-aware sampler that increases the draw probability of examples containing rarer entity labels

The weighted mode was implemented in `src/base_model_training/data.py` and wired into `src/base_model_training/cv.py`:

- per-example sampling weights are derived from the rarity of the labels present in each example
- `WeightedRandomSampler` is used only for the training loader
- validation keeps deterministic sequential loading
- the training CLI now exposes `--train-sampling`, defaulting to `weighted`

Implication:

- training epochs should expose the model to rare-label examples more consistently
- this reduces the chance that a long run is dominated by batches containing mostly the majority entity class
