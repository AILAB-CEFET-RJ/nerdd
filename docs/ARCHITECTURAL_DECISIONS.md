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

A held-out calibration subset was created from `data/dd_corpus_small_test.json`, and a global temperature-scaling calibrator was fit and then evaluated on a separate held-out final test subset.

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

## 2026-03-21 - Current Refit Stage Does Not Yet Match The Main Dissertation Hypothesis

### Context

The main hypothesis under discussion is:

> incorporating new examples obtained through pseudolabelling into fine-tuning produces NER models that are better than the originally trained base model

However, the current refit implementation in `src/pseudolabelling/` does not merge pseudolabelled examples with the original supervised training set.

### Decision

This mismatch must be treated as an explicit architectural limitation of the current codebase.

At present, the implemented experiment is closer to:

- start from the base model weights
- refit using selected pseudolabelled records only

It is not yet the stronger design:

- original labeled training set plus selected pseudolabelled records

### Rationale

Training only on pseudolabelled records changes the question being tested.

It risks:

- discarding the highest-quality supervised signal
- reinforcing model-generated errors
- weakening the direct connection between implementation and the dissertation claim

### Implications For Experimentation

Results from the current refit flow should be interpreted as self-training or pseudolabel-only refit, not as supervised-plus-pseudolabel augmentation.

If the dissertation claims dataset augmentation, the code should be updated so refit consumes:

- original labeled data
- kept pseudolabelled records

and ideally accumulates pseudolabelled data across iterations.

### Implications For Dissertation Wording

Until the code is changed, the dissertation should not overclaim that the implemented pipeline already performs cumulative augmentation of the original training set.

It should either:

- describe the current method honestly as pseudolabel-only refit from a pretrained base model

or

- update the implementation before finalizing the dissertation text

## 2026-03-21 - Base Versus Refit Comparison Must Be Explicit

### Context

The current pseudolabelling flow can evaluate a refit model, but it does not automatically produce a paired comparison between:

- the original base model
- the refit model

on the same holdout set.

### Decision

The project should treat direct base-versus-refit evaluation on the same holdout as a required comparison for hypothesis testing.

### Rationale

Without a controlled paired comparison, the main dissertation claim cannot be established cleanly.

### Implications For Experimentation

Every refit experiment intended to support the dissertation hypothesis should report:

- base model metrics on the holdout
- refit model metrics on the same holdout
- the delta between them

### Implications For Dissertation Wording

The dissertation should frame improvements only in terms of direct holdout comparison between the original base model and the pseudolabelling-refit model.
