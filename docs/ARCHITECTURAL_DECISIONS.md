# Architectural Decisions

This document records architecture and experiment-design decisions that materially affect how the pipeline should be described in reports, papers, and the dissertation text.

Each entry captures:

- context
- decision
- rationale
- implications for experimentation
- implications for dissertation wording

## 2026-03-21 - Calibration Becomes a Base-Model Artifact

### Context

The original motivation for calibration was that the base GLiNER model was producing overconfident entity scores.

The previous pipeline design placed calibration inside the pseudolabelling flow as an optional post-processing step over prediction JSONL files. That design blurred two separate concerns:

- fitting a probability calibrator from labeled data
- applying a fitted calibrator during large-corpus inference

### Decision

Calibration is now treated as a reusable artifact attached to the base model.

The pipeline is split into:

1. fit the calibrator on a labeled calibration subset
2. persist the calibrator as JSON
3. apply the calibrator later during prediction on unlabeled data

Canonical scripts:

- `src/tools/build_calibration_dataset.py`
- `src/calibration/fit_calibrator.py`
- `src/calibration/apply_calibrator.py`

### Rationale

This matches the actual statistical purpose of calibration:

- the model produces raw confidence scores
- labeled holdout data is used to estimate how those scores map to empirical correctness
- the learned mapping is then reused during inference

### Implications For Experimentation

- calibration should be fit on labeled holdout predictions, not on unlabeled pseudolabelling outputs
- the calibrator should be evaluated on a separate held-out final test subset
- pseudolabelling should consume the saved calibrator, not refit calibration ad hoc on each run

### Implications For Dissertation Wording

The dissertation should not describe calibration as merely a post-processing trick inside pseudolabelling.

It should describe calibration as:

- a correction layer over the base model's probability estimates
- fit using labeled holdout data
- validated on a separate final test subset
- then applied during large-corpus prediction

## 2026-03-21 - Global Temperature Scaling Accepted As Initial Calibration Baseline

### Context

A held-out calibration subset is maintained as `data/dd_corpus_small_calibration.json`, and final evaluation is run on `data/dd_corpus_small_test.json`.

### Decision

Global temperature scaling is the current baseline calibration method.

### Rationale

It is simpler and more stable than per-class or isotonic alternatives, especially under class imbalance.

Held-out final-test evaluation showed consistent improvement in:

- Brier score
- NLL
- ECE
- MCE

including per-label improvements for:

- `Location`
- `Organization`
- `Person`

### Implications For Experimentation

- the project now has an empirically validated default calibrator
- more complex methods such as `temperature-per-class` or `isotonic` can be studied as comparisons, not as the operational default

### Implications For Dissertation Wording

The dissertation can state that:

- calibration was motivated by score overconfidence in the base model
- a global temperature-scaling baseline was adopted first
- improvements were verified on held-out data, not only on the fitting subset

## 2026-03-21 - Refit Must Combine Supervised Data Plus Kept Pseudolabels

### Context

The main hypothesis under discussion is:

> incorporating new examples obtained through pseudolabelling into fine-tuning produces NER models that are better than the originally trained base model

The earlier refit implementation in `src/pseudolabelling/` trained only on kept pseudolabel records. That design did not match the hypothesis above.

### Decision

Refit now combines:

- the original supervised training set
- the selected kept pseudolabel records

The merged training set is the canonical input to `src/pseudolabelling/refit_model.py`.

When duplicate texts are present, the supervised example is preferred by default.

### Rationale

Training only on pseudolabelled records changes the question being tested and weakens the connection between implementation and claim.

The merged design keeps the highest-quality human supervision as an anchor while still testing whether pseudolabel augmentation helps.

It also reduces the risk of:

- reinforcing model-generated errors
- drifting toward a pseudolabel-only regime that no longer reflects the main experimental question

### Implications For Experimentation

Refit experiments should now be interpreted as supervised-plus-pseudolabel augmentation experiments, not pseudolabel-only self-training.

Each run should record:

- how many rows came from supervised data
- how many rows came from pseudolabels
- whether duplicate texts were discarded in favor of supervised rows

### Implications For Dissertation Wording

The dissertation can now describe the refit stage as fine-tuning on:

- the original labeled training set
- plus a selected set of pseudolabelled examples

It should still avoid claiming cumulative multi-iteration augmentation unless that part is implemented explicitly.

## 2026-03-21 - Base Versus Refit Comparison Must Be Explicit

### Context

The project hypothesis depends on comparing the original base model with the pseudolabelling-refit model on the same holdout set.

### Decision

Direct base-versus-refit evaluation on the same holdout is a required comparison for hypothesis testing, and the iterative orchestrator now writes it automatically when `--evaluate-refit` is enabled.

### Rationale

Without a controlled paired comparison, the main dissertation claim cannot be established cleanly.

### Implications For Experimentation

Every refit experiment intended to support the dissertation hypothesis should report:

- base model metrics on the holdout
- refit model metrics on the same holdout
- the delta between them

Canonical artifacts now include:

- `07_eval_base/metrics.json`
- `08_eval_refit/metrics.json`
- `09_base_vs_refit_comparison.json`

### Implications For Dissertation Wording

The dissertation should frame improvements only in terms of direct holdout comparison between the original base model and the pseudolabelling-refit model.

## 2026-03-21 - Pseudolabelling Effect Must Be Measured Against A Supervised-Only Refit Baseline

### Context

Once refit was corrected to support supervised data plus kept pseudolabels, an immediate experimental risk remained:

- improvements from an additional supervised refit could be mistaken for improvements from pseudolabel augmentation

This is not hypothetical. A controlled run with `refit_mode=supervised_only` already improved over the base model on the legacy filtered holdout used in the early experiments.

### Decision

Pseudolabelling experiments must use an explicit three-condition interpretation:

- base model
- supervised-only refit
- supervised-plus-pseudolabels refit

The primary hypothesis test is:

- `(supervised_plus_pseudolabels) - (supervised_only)`

not merely:

- `(supervised_plus_pseudolabels) - (base)`

The pipeline now exposes `refit_mode` explicitly with:

- `supervised_only`
- `supervised_plus_pseudolabels`
- `pseudolabel_only`

### Rationale

Without the supervised-only control, the effect of pseudolabelling is confounded with the effect of simply continuing supervised fine-tuning from the base checkpoint.

### Implications For Experimentation

Every pseudolabelling experiment intended to support the main dissertation hypothesis should report:

- base vs supervised-only refit
- base vs supervised-plus-pseudolabels refit
- supervised-only refit vs supervised-plus-pseudolabels refit

The third comparison is the clean estimate of the marginal contribution of pseudolabelled examples.

### Implications For Dissertation Wording

The dissertation should not attribute all post-refit gains to pseudolabelling.

It should distinguish:

- gains from additional supervised refit
- gains from adding pseudolabelled examples on top of that supervised refit baseline

### First Controlled Result

On the shared legacy filtered holdout used in those early runs, a controlled pair of `t020` experiments produced:

- base: `micro_f1=0.4440`, `macro_f1=0.4362`
- supervised-only refit: `micro_f1=0.4952`, `macro_f1=0.4772`
- supervised-plus-pseudolabels refit: `micro_f1=0.5260`, `macro_f1=0.5097`

Interpretation:

- supervised-only refit gain over base:
  - `micro_f1 +0.0512`
  - `macro_f1 +0.0411`
- additional gain from pseudolabelling over the supervised-only control:
  - `micro_f1 +0.0308`
  - `macro_f1 +0.0324`

This is the first controlled result in the project where the marginal contribution of pseudolabelled examples was separated from the gain of an additional supervised refit.

### Practical Follow-Up For Scaling

After the controlled `t020` result, threshold choice for the large unlabeled corpus was treated as a separate engineering decision rather than a direct carryover from the small holdout experiments.

A `10k` sample from `data/dd_corpus_large.json` was used to probe score distribution under the real metadata-rich inference regime.

Observed kept counts on the `10k` sample:

- `threshold=0.20` -> `281` kept
- `threshold=0.30` -> `52` kept
- `threshold=0.40` -> `14` kept

Decision:

- use `0.30` as the first full-corpus threshold for overnight runs

Rationale:

- `0.20` likely yields a very large pseudolabel set on the full corpus
- `0.40` is likely too conservative for a first real semisupervised run
- `0.30` is a better first operating point for balancing pseudolabel volume and expected noise

### Validation Set Must Remain Supervised In Semisupervised Refit

During the first full-corpus `supervised_plus_pseudolabels` run at `threshold=0.30`, the refit stage allowed pseudolabelled examples to enter the internal validation split. The supervised train/validation partition also shifted relative to the `supervised_only` control.

That made the comparison methodologically unsound for two reasons:

- early stopping and checkpoint choice were influenced by noisy pseudolabel validation examples
- the supervised control split was no longer held constant across regimes

Decision:

- when no external `val_jsonl` is supplied and supervised data is available, the supervised dataset must be split first
- that supervised split becomes the fixed validation set
- pseudolabelled examples may be appended only to the training side

Implications:

- `supervised_only` and `supervised_plus_pseudolabels` now share the same supervised train/validation partition for a given seed
- the marginal effect of pseudolabelling is no longer confounded by validation contamination
- earlier full-corpus results produced before this correction should be treated as diagnostic, not as final evidence

## Iterative Pseudolabelling Requires Explicit Chunk Inputs And Explicit Accumulated Pseudolabel Inputs

### Context

The original single-shot pseudolabelling flow assumes that refit consumes the `kept.jsonl` produced by the current run.

That assumption is too restrictive for iterative semisupervised experiments, where refit should consume:

- the fixed supervised dataset
- plus an explicitly accumulated pseudolabel dataset assembled across previous chunks

### Decision

- keep the current split-directory flow as the default
- add an explicit pseudolabel override path for refit
- support chunk-based experimentation through a dedicated large-corpus chunking utility

Implementation consequences:

- `src/tools/split_large_corpus_into_chunks.py` now creates fixed JSONL chunks from `data/dd_corpus_large.json`
- `src/pseudolabelling/refit_model.py` and the orchestrator now accept an explicit accumulated pseudolabel path
- when that explicit path is supplied, refit no longer depends on the current run's `05_split/kept.jsonl`

### Rationale

- iterative experiments need stable chunk boundaries
- accumulated pseudolabel sets must be first-class artifacts, not implied by directory layout
- this keeps the future `single shot` vs `iterative` comparison operationally clean and auditable

### Implications For Experimentation

The project now has two semisupervised regimes that should be described separately:

- `single shot`
  - one unlabeled input
  - one split step
  - one refit step
- `iterative`
  - multiple fixed unlabeled chunks
  - one split step per chunk
  - one accumulated pseudolabel artifact per iteration
  - one refit step per accumulated state

The iterative regime should use:

- fixed chunk boundaries
- a stable threshold per experiment family
- explicit accumulated files such as `kept_acc_01.jsonl`, `kept_acc_02.jsonl`, `kept_acc_03.jsonl`

This ensures that a later comparison between `single shot` and `iterative` is primarily a comparison of incorporation strategy rather than a comparison of undocumented data handling.

### Implications For Dissertation Wording

The dissertation should not describe the iterative regime as merely "rerunning the same pipeline several times."

It should describe it as:

- a chunk-based semisupervised procedure
- where pseudolabelled reports are accumulated across iterations
- and where refit can be driven by an explicit accumulated pseudolabel artifact rather than only the current split output

## 2026-03-21 - Pseudolabelling Selection Happens At Record Level, Not Entity Level

### Context

The training grain of this project is the report (`relato`), not an isolated entity mention.

That means pseudolabelling selection cannot literally choose "entities as examples" for refit. The natural training unit remains the full report.

However, there are still two distinct selection layers that can exist in a report-level pipeline:

- keep or discard the report
- keep or discard individual predicted entities inside a kept report

### Decision

The current pipeline performs selection only at the report level.

It does this by:

- computing an aggregated `record_score` from entity scores
- splitting reports into `kept` and `discarded`
- sending every predicted entity from each kept report into refit

It does **not** currently filter low-confidence entities inside a kept report.

### Rationale

This is the simplest semisupervised design compatible with report-level NER training.

It keeps the training grain aligned with the original supervised dataset, where each example is a full report containing zero or more spans.

### Implications For Experimentation

Current refit experiments should be interpreted as:

- report-level pseudolabel selection
- with entity sets inherited wholesale from each kept report

This means some noisy low-confidence entities can still enter refit if they belong to a report whose aggregate score passes the threshold.

### Shared GLiNER Loader For Inference/Evaluation

GLiNER loading policy outside nested-CV training was centralized in `src/gliner_loader.py`.

This shared loader now serves:

- corpus prediction
- refit evaluation
- calibration-dataset generation
- inference profiling
- base-model evaluation

The goal is to keep `load_tokenizer=True` and optional `max_length` handling consistent across inference-oriented code paths.

The nested-CV training loader in `src/base_model_training/cv.py` remains separate for now because it still carries specialized fallback behavior (`fix_mistral_regex`, `local_files_only`, and compatibility fallbacks).

### Implications For Dissertation Wording

The dissertation should describe the current method as selecting pseudolabelled **reports**, not isolated entity spans.

If a later version adds span-level filtering inside kept reports, that should be described as a refinement on top of the current report-level selection strategy, not as something already present.
