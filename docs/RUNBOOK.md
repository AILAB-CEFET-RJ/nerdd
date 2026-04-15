# Runbook

Assumes the environment from `docs/INSTALL.md` is already set up.

## Artifact Convention

Operational rules for this repository:

1. Prefer running commands from the repository root whenever practical.
2. `artifacts/` is the only official artifact root.
3. When running from inside `src/`, always point artifact paths to `../artifacts/...`.
4. Prefer `python3 -m ...` for project modules such as `pseudolabelling` and `base_model_training`.

## Data Convention

Operational rules for dataset and artifact ownership:

1. `data/` stores only canonical input datasets.
2. `artifacts/` stores every derived output.
3. Never write prediction CSVs, calibration CSVs, reports, summaries, HTML viewers, JSONL predictions, or model outputs into `data/`.
4. If a file can be regenerated from code plus canonical inputs, it belongs in `artifacts/`, not `data/`.

Canonical datasets currently expected in `data/`:

- `data/dd_corpus_large.json`
- `data/dd_corpus_small_train.json`
- `data/dd_corpus_small_calibration.json`
- `data/dd_corpus_small_test.json`

Examples of files that must go under `artifacts/`:

- calibration prediction CSVs
- calibration prediction JSONL files
- calibrator JSON artifacts
- training outputs and checkpoints
- pseudolabelling predictions, rankings, adjudication outputs, and summaries

## Stage Naming Convention

Use `number + semantic_name` for pipeline stages and derived artifact names.

Examples:

- `01_predictions`
- `02_context_boosted`
- `03_record_scored`
- `04_ranked_candidates`
- `04b_gliner2_predictions`
- `05_llm_input`
- `06_llm_adjudicated`

Rationale:

- preserves execution order in directory listings
- keeps stage purpose readable without consulting code
- reduces ambiguity during handoffs and context resets

Operational note:

- use `src/tools/rank_pseudolabel_candidates.py` as the default implementation of `04_ranked_candidates`
- use `src/tools/inspect_dense_tips.py` only for dense-outlier auditing, not as the primary ranking step before `05_llm_input`
- use `src/tools/generate_gliner2_predictions.py` as the default implementation of `04b_gliner2_predictions`
- `05_llm_input` must consume rows already enriched with `gliner2_entities`; inline GLiNER2 inference is not allowed in this stage
- for current DD corpus conventions, do not use `--max-location-ratio` as an operational filter; `Location`-heavy relatos are common and often valid

## 1) First Run

Use this smoke test first to validate the training stack end-to-end on a tiny dataset.

If you are running on an NVIDIA Blackwell GPU such as an RTX 5090, confirm your environment uses PyTorch `cu128` or newer before starting.

## 2) Local Smoke Test (CPU / low RAM)
```bash
cd src
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m base_model_training.train_nested_kfold \
  --model-base urchade/gliner_small-v2.1 \
  --train-path ../data/dd_corpus_smoke_20.json \
  --batch-size 1 \
  --max-length 32 \
  --overlap 2 \
  --num-epochs 1 \
  --n-splits 2 \
  --n-inner-splits 2 \
  --search-mode grid \
  --lr-values 3e-5 \
  --weight-decay-values 0.01 \
  --thresholds 0.6 \
  --refit-val-size 0.5 \
  --output-dir ../artifacts/base_model_training/smoke/run_nested_tiny \
  --log-level INFO
```

Expected outputs:
- `artifacts/base_model_training/smoke/run_nested_tiny/nested_cv_results.txt`
- `artifacts/base_model_training/smoke/run_nested_tiny/nested_cv_results.json`
- `artifacts/base_model_training/smoke/run_nested_tiny/best_overall_gliner_model/`

## 3) Server Training (example with batch-size 16)
```bash
cd src
python3 -m base_model_training.train_nested_kfold \
  --train-path ../data/dd_corpus_small_train.json \
  --batch-size 16 \
  --num-epochs 20 \
  --n-splits 3 \
  --n-inner-splits 3 \
  --search-mode random \
  --num-trials 20 \
  --lr-values 1e-6,2e-6,5e-6,1e-5,2e-5,3e-5,5e-5,8e-5,1e-4,2e-4,3e-4 \
  --weight-decay-values 0.0,0.01,0.05 \
  --thresholds 0.5,0.6 \
  --output-dir ../artifacts/base_model_training/experiments/run_batch16 \
  --log-level INFO
```

## 4) Evaluation
```bash
cd src
python3 base_model_training/evaluate_gliner.py \
  --model-path ../artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --gt-jsonl ../data/dd_corpus_small_test.json \
  --pred-jsonl ../artifacts/base_model_training/experiments/run_batch16/eval/pred.jsonl \
  --report-path ../artifacts/base_model_training/experiments/run_batch16/eval/report.txt \
  --calibrated-thresholds-json ../artifacts/base_model_training/experiments/run_batch16/eval/thresholds.json \
  --labels Person,Location,Organization \
  --chunk-size 128 \
  --batch-size 1 \
  --prediction-threshold 0.6 \
  --log-level INFO
```

## 5) Large Corpus Prediction (inference-only)
```bash
cd src
python3 -m pseudolabelling.generate_corpus_predictions \
  --model-path ../artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --model-max-length 384 \
  --input-jsonl ../artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl \
  --output-jsonl ../artifacts/pseudolabelling/iter01/01_predictions.jsonl \
  --stats-json ../artifacts/pseudolabelling/iter01/01_predictions_stats.json \
  --labels Person,Location,Organization \
  --text-fields relato \
  --max-tokens 384 \
  --batch-size 4 \
  --score-threshold 0.0 \
  --log-level INFO
```

## 5b) Compute Record Scores For Ranking

Use `pseudolabelling.compute_record_scores` to convert entity-level confidence into a
single `record_score` per relato before ranking or thresholding candidates.

Recommended aggregation for current pseudolabelling runs:

- `median`

Rationale:

- keeps the candidate pool broad enough for downstream adjudication-oriented filtering
- is less brittle than `mean_times_min` when a small number of entities are good and one span is still correctable
- remains more robust than plain arithmetic mean against single noisy entity scores

Example:

```bash
cd src
python3 -m pseudolabelling.compute_record_scores \
  --input-jsonl ../artifacts/pseudolabelling/iter01/02_context_boosted.jsonl \
  --output-jsonl ../artifacts/pseudolabelling/iter01/03_record_scored.jsonl \
  --stats-json ../artifacts/pseudolabelling/iter01/03_score_stats.json \
  --score-field score_context_boosted \
  --output-field record_score \
  --legacy-field-alias score_relato \
  --aggregation median \
  --empty-entities-policy zero \
  --log-level INFO
```

## 6) Split Holdout For Calibration

This step is only needed when rebuilding calibration and final-test splits from a pre-split dataset. In the current cleaned `data/` layout, the canonical files already exist as:

- `../data/dd_corpus_small_calibration.json`
- `../data/dd_corpus_small_test.json`

```bash
cd src
python3 tools/split_dataset_for_calibration.py \
  --input ../data/dd_corpus_small_test.json \
  --calibration-output ../data/dd_corpus_small_calibration.json \
  --final-test-output ../data/dd_corpus_small_test.json \
  --summary-json ../data/dd_corpus_small_split_summary.json \
  --calibration-ratio 0.2 \
  --seed 42 \
  --mode random
```

## 7) Build Calibration CSV From Labeled Holdout
```bash
cd src
python3 tools/build_calibration_dataset.py \
  --model-path ../artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --input ../data/dd_corpus_small_calibration.json \
  --output-csv ../data/dd_corpus_small_calibration_predictions.csv \
  --output-predictions-jsonl ../artifacts/calibration/base_model/calibration_predictions.jsonl \
  --labels Person,Location,Organization \
  --batch-size 4 \
  --max-tokens 384 \
  --threshold 0.0
```

## 8) Fit Calibrator Artifact
```bash
cd src
python3 calibration/fit_calibrator.py \
  --method temperature-per-class \
  --calibration-csv ../data/dd_corpus_small_calibration_predictions.csv \
  --output-calibrator ../artifacts/calibration/base_model/calibrator.json \
  --stats-json ../artifacts/calibration/base_model/fit_stats.json \
  --score-col Score \
  --label-col Validacao \
  --class-col Label \
  --labels Person,Location,Organization \
  --log-level INFO
```

## 9) Large Corpus Prediction With Calibrator
```bash
cd src
python3 -m pseudolabelling.generate_corpus_predictions \
  --model-path ../artifacts/base_model_training/experiments/run_batch16/best_overall_gliner_model \
  --model-max-length 384 \
  --calibrator-path ../artifacts/calibration/base_model/calibrator.json \
  --input-jsonl ../artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl \
  --output-jsonl ../artifacts/pseudolabelling/iter01/01_predictions.jsonl \
  --stats-json ../artifacts/pseudolabelling/iter01/01_predictions_stats.json \
  --labels Person,Location,Organization \
  --text-fields relato \
  --max-tokens 384 \
  --batch-size 4 \
  --score-threshold 0.0 \
  --log-level INFO
```

## 10) Sanitize The Large Corpus Before Pseudolabelling

Promote a deduplicated and conservatively filtered corpus before running expensive large-corpus inference.

```bash
cd .
python3 src/tools/sanitize_dd_corpus.py \
  --input data/dd_corpus_large.json \
  --output-sanitized-jsonl artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl \
  --output-dropped-jsonl artifacts/corpus_sanitization/dd_corpus_large_dropped_safe.jsonl \
  --output-flagged-jsonl artifacts/corpus_sanitization/dd_corpus_large_flagged_review.jsonl \
  --summary-json artifacts/corpus_sanitization/dd_corpus_large_sanitization_summary.json
```

Operational convention:

- `data/dd_corpus_large.json` remains the raw corpus.
- `artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl` is the official pseudolabelling input.
- `artifacts/corpus_sanitization/dd_corpus_large_flagged_review.jsonl` is held out for later inspection.

Sanitization intent:

- this stage should remove not only obvious formatting junk and duplicates, but also clearly off-domain low-context texts that do not resemble the supervised DD train/test corpora
- examples of texts that should now be discarded earlier include short nominal lists and very short snippets with no narrative or locative anchor

## 11) Sample A Reproducible Fraction Of The Large Corpus

Before running expensive pseudolabelling experiments on the full large corpus, create a fixed sample to study the score distribution and choose a threshold budget more safely.

```bash
cd .
python3 src/tools/sample_large_corpus.py \
  --input artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl \
  --output-jsonl artifacts/corpus_sanitization/dd_corpus_large_sanitized_sample_10k.jsonl \
  --sample-size 10000 \
  --seed 42 \
  --summary-json artifacts/corpus_sanitization/dd_corpus_large_sanitized_sample_10k_summary.json
```

Recommended use:

- run prediction, context boost, and record scoring on `artifacts/corpus_sanitization/dd_corpus_large_sanitized_sample_10k.jsonl`
- inspect the resulting score distribution
- choose the kept/discarded threshold based on the observed volume of candidate reports

Observed probe results on the `10k` sample:

- `threshold=0.20` -> `281` kept
- `threshold=0.30` -> `52` kept
- `threshold=0.40` -> `14` kept

Recommended first full-corpus operating point:

## 12b) Codex-vs-GPT Adjudication Benchmark

Use `src/tools/manage_codex_adjudication_benchmark.py` to initialize and consolidate the benchmark, and use `scripts/codex_benchmark.sh` as the operational wrapper during chunk-by-chunk work.

Initialize once:

```bash
python3 src/tools/manage_codex_adjudication_benchmark.py init \
  --input artifacts/pseudolabelling/baseline_quick_2026-04-03/05_llm_input_t06_top1000.jsonl \
  --benchmark-dir artifacts/benchmarks/codex_adjudication_t06_top1000 \
  --benchmark-name codex_vs_gpt5_t06_top1000 \
  --chunk-size 10
```

Operational loop:

```bash
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 open-next
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 ingest-latest
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 status
```

Automatic loop via API, without copy/paste:

```bash
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 auto-next
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 auto-complete-next
```

Single-command iteration after saving each response:

```bash
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 complete-next
```

Behavior of the wrapper:

- `open-next` reserves the next chunk, prints the response path you should write to, prints the exact `cat > ... <<'EOF'` prefix to use, and then prints the chunk contents.
- `open-next` also reminds you that `accept` and `accept_with_edits` may only keep entities already present in `review_seed_entities`.
- `ingest-latest` ingests the most recently exported chunk without requiring you to retype the chunk id.
- `complete-next` ingests the latest exported chunk if its response file already exists, then immediately opens the next pending chunk.
- `status` prints progress and reminds you to continue with `open-next` if there are pending chunks.
- `auto-next` reserves the next chunk, calls `src/tools/run_llm_adjudication.py` on that chunk, writes the response JSONL into `manual_responses/`, and ingests it immediately.
- `auto-complete-next` ingests the latest exported chunk if needed, then auto-runs the next pending chunk through the API and ingests it.

Operational note for train-oriented benchmarks:

- initialize those benchmarks with `--annotation-mode train_annotation`
- in that mode, the prompt is conservative but not restricted to `review_seed_entities`
- the benchmark wrapper propagates the benchmark `annotation_mode` into the automatic API call

Operational constraint:

- in this benchmark, `accept` and `accept_with_edits` are literal seed-set decisions, not open extraction
- `entities_final` may only contain spans already present in `review_seed_entities`
- do not add `baseline_only_entities`, `gliner2_only_entities`, normalized place names, or corrected spellings unless they are already in `review_seed_entities`

Manual alternatives:

```bash
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 next
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 show chunk_001
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 show-latest
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 response-path chunk_001
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 ingest chunk_001
```

Final consolidation:

```bash
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 build-output
```

- start with `threshold=0.30`
- treat `0.20` as a more aggressive follow-up
- treat `0.40` as a more conservative follow-up

If inference throughput degrades unexpectedly after prediction-pipeline changes, profile the first `N` reports before launching another full overnight run:

```bash
cd src
python3 tools/profile_pseudolabelling_inference.py \
  --model-path ../artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --model-max-length 384 \
  --input-jsonl ../artifacts/corpus_sanitization/dd_corpus_large_sanitized_sample_10k.jsonl \
  --labels Person,Location,Organization \
  --text-fields relato \
  --batch-size 16 \
  --max-tokens 512 \
  --score-threshold 0.0 \
  --limit 100 \
  --report-json ../artifacts/pseudolabelling/profile_inference_100.json
```

This probe reports:

- effective chunk budget
- average chunking seconds per row
- average inference seconds per row
- average and maximum chunk count per row
- overall rows per second

Inference knobs:

- `--model-max-length`
  - passed into `GLiNER.from_pretrained(...)`
  - influences GLiNER's internal processor/tokenizer behavior
- `--max-tokens`
  - used by the pseudolabelling pipeline to chunk long texts before calling GLiNER

## 12) Split The Large Corpus Into Fixed Chunks For Iterative Experiments

When comparing single-shot pseudolabelling against iterative pseudolabelling, split the large corpus into fixed JSONL chunks first. This keeps chunk boundaries reproducible and prevents ad hoc slicing during long-running experiments.

```bash
cd .
python3 src/tools/split_large_corpus_into_chunks.py \
  --input artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl \
  --output-dir artifacts/corpus_sanitization/dd_corpus_large_sanitized_chunks_50k \
  --chunk-size 50000 \
  --chunk-prefix dd_corpus_large_sanitized_chunk \
  --summary-json artifacts/corpus_sanitization/dd_corpus_large_sanitized_chunks_50k_summary.json
```

Recommended use:

- keep chunk size fixed across all iterative runs
- treat the generated chunk files as immutable experiment inputs
- accumulate `kept.jsonl` outputs across iterations into explicit JSONL artifacts

The refit stage now supports an explicit accumulated pseudolabel input via:

- `--refit-pseudolabel-path`

Use this when an iterative run should refit on:

- supervised training data
- plus a manually accumulated pseudolabel file

Chunk summary expectations:

- each chunk is a standalone JSONL file
- chunk ordering is fixed by source order
- the summary JSON records:
  - `rows_total`
  - `chunk_size`
  - `chunks_total`
  - `start_index` / `end_index_exclusive` per chunk

Recommended artifact convention for iterative experiments:

- raw chunks:
  - `artifacts/corpus_sanitization/dd_corpus_large_sanitized_chunks_50k/dd_corpus_large_sanitized_chunk_01.jsonl`
  - `artifacts/corpus_sanitization/dd_corpus_large_sanitized_chunks_50k/dd_corpus_large_sanitized_chunk_02.jsonl`
  - `artifacts/corpus_sanitization/dd_corpus_large_sanitized_chunks_50k/dd_corpus_large_sanitized_chunk_03.jsonl`
- per-iteration runs:
  - `artifacts/pseudolabelling/iterative_chunks_t030/iter_01/`
  - `artifacts/pseudolabelling/iterative_chunks_t030/iter_02/`
  - `artifacts/pseudolabelling/iterative_chunks_t030/iter_03/`
- accumulated pseudolabel artifacts:
  - `artifacts/pseudolabelling/iterative_chunks_t030/accumulated/kept_acc_01.jsonl`
  - `artifacts/pseudolabelling/iterative_chunks_t030/accumulated/kept_acc_02.jsonl`
  - `artifacts/pseudolabelling/iterative_chunks_t030/accumulated/kept_acc_03.jsonl`

## 12) Iterative Pseudolabelling Protocol

The iterative regime is not the same as the current single-shot flow.

Single-shot semisupervised run:

- infer on one large unlabeled input
- split kept/discarded once
- refit once using a single pseudolabel set

Iterative semisupervised run:

- infer on chunk 1
- split and save `kept_1`
- accumulate `kept_acc_01`
- refit on `supervised + kept_acc_01`
- repeat on chunk 2, chunk 3, and so on

Recommended first operational design:

- chunk size: `50000`
- threshold: `0.30`
- fixed number of iterations: `3`
- supervised dataset stays fixed
- accumulated pseudolabel file grows after each chunk

Recommended refit semantics:

- always refit from the same base model checkpoint
- use the accumulated pseudolabel JSONL as the variable input

This keeps the comparison cleaner because iteration results differ mainly by training data volume, not by continuing optimization from a previously adapted checkpoint.

Example iterative refit invocation:

```bash
cd src
python3 -m pseudolabelling.run_iterative_cycle \
  --run-dir ../artifacts/pseudolabelling/iterative_chunks_t030/iter_02 \
  --model-path ../artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --prediction-calibrator-path ../artifacts/calibration/base_model/calibrator.json \
  --input-jsonl ../artifacts/corpus_sanitization/dd_corpus_large_sanitized_chunks_50k/dd_corpus_large_sanitized_chunk_02.jsonl \
  --labels Person,Location,Organization \
  --text-fields relato \
  --prediction-batch-size 16 \
  --prediction-max-tokens 512 \
  --prediction-threshold 0.0 \
  --record-score-field score_context_boosted \
  --split-threshold 0.30 \
  --refit-mode supervised_plus_pseudolabels \
  --refit-base-model ../artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --refit-pseudolabel-path ../artifacts/pseudolabelling/iterative_chunks_t030/accumulated/kept_acc_02.jsonl \
  --refit-supervised-train-path ../data/dd_corpus_small_train.json \
  --refit-epochs 3 \
  --refit-patience 2 \
  --refit-batch-size 16 \
  --refit-max-length 512 \
  --refit-overlap 128 \
  --refit-lr 1e-5 \
  --refit-weight-decay 0.01 \
  --evaluate-refit \
  --eval-gt-jsonl ../data/dd_corpus_small_test.json \
  --eval-prediction-threshold 0.05 \
  --eval-batch-size 16 \
  --eval-max-tokens 512 \
  --log-level INFO
```

Important behavior:

- inference and split still run on the current chunk
- refit ignores the current run's `05_split/kept.jsonl` when `--refit-pseudolabel-path` is supplied
- refit uses the explicit accumulated pseudolabel artifact instead

Operational note:

- accumulation of `kept_acc_0N.jsonl` is still an external step
- the current code now supports that artifact cleanly, but does not yet build it automatically

## 12a) Converting `06_llm_adjudicated` Into Refit Pseudolabels

`06_llm_adjudicated` is not yet the final refit input. The refit stage expects a JSONL with:

- `text`
- `entities`

Use the converter below to keep only approved adjudications and project `adjudication.entities_final` into `entities`.

```bash
cd .
python3 src/tools/build_refit_pseudolabel_dataset.py \
  --input artifacts/pseudolabelling/baseline_quick_2026-04-03/06_llm_adjudicated_t06_top1000.jsonl \
  --output-jsonl artifacts/pseudolabelling/baseline_quick_2026-04-03/07_refit_pseudolabels_t06_top1000.jsonl \
  --summary-json artifacts/pseudolabelling/baseline_quick_2026-04-03/07_refit_pseudolabels_t06_top1000_summary.json
```

To reuse a larger already-adjudicated ranking prefix and emit only the first `n`
examples in refit format, pass `--top-n`:

```bash
cd src
python3 tools/build_refit_pseudolabel_dataset.py \
  --input ../artifacts/benchmarks/codex_train_annotation_top20_median_v1/codex_output.jsonl \
  --output-jsonl ../artifacts/benchmarks/codex_train_annotation_top10_median_v1/refit_pseudolabels.jsonl \
  --summary-json ../artifacts/benchmarks/codex_train_annotation_top10_median_v1/refit_pseudolabels_summary.json \
  --top-n 10
```

This avoids re-adjudicating `top10` when `top20` has already been adjudicated.

Recommended defaults:

- keep `accept` and `accept_with_edits`
- do not materialize a merged `small_train + pseudolabels` dataset
- pass the emitted file directly through `--pseudolabel-path`
- keep pseudolabels on the training side only; validation must remain supervised-only

Example refit invocation:

```bash
cd src
python3 -m pseudolabelling.refit_model \
  --pseudolabel-path ../artifacts/pseudolabelling/baseline_quick_2026-04-03/07_refit_pseudolabels_t06_top1000.jsonl \
  --supervised-train-path ../data/dd_corpus_small_train.json \
  --refit-mode supervised_plus_pseudolabels \
  --base-model ../artifacts/base_model_training/experiments/baseline_quick_2026-04-03/best_quick_gliner_model \
  --output-model-dir ../artifacts/pseudolabelling/baseline_quick_2026-04-03/08_refit_model_t06_top1000
```

Quick ablation commands using the adjudicated pseudolabels without materializing a merged dataset:

`base_model_training.train_quick`

```bash
cd src
python3 -m base_model_training.train_quick \
  --train-path ../data/dd_corpus_small_train.json \
  --test-path ../data/dd_corpus_small_test.json \
  --pseudolabel-path ../artifacts/benchmarks/codex_adjudication_disagreement_top100_v2/refit_pseudolabels.jsonl \
  --train-mode supervised_plus_pseudolabels \
  --output-dir ../artifacts/base_model_training/quick_supervised_plus_codex \
  --log-level INFO
```

`gliner2_training.train_quick`

```bash
cd src
python3 -m gliner2_training.train_quick \
  --train-path ../data/dd_corpus_small_train.json \
  --test-path ../data/dd_corpus_small_test.json \
  --pseudolabel-path ../artifacts/benchmarks/codex_adjudication_disagreement_top100_v2/refit_pseudolabels.jsonl \
  --train-mode supervised_plus_pseudolabels \
  --output-dir ../artifacts/gliner2_training/quick_supervised_plus_codex \
  --log-level INFO
```

For both quick-training entrypoints:

- the supervised split is created first
- the internal validation set remains supervised-only
- pseudolabel rows are appended only to the training side
- this avoids validation leakage from adjudicated pseudolabels

Experimental note from the disagreement benchmarks:

- `codex_adjudication_disagreement_top100_v2` and `codex_adjudication_disagreement_top300_v1` were useful as directed evaluation/audit sets
- as training pseudolabels, the same artifacts were not stable enough to promote by default
- a single positive run is not enough; use multiple seeds before attributing gains to adjudicated pseudolabels
- in the `top300_v1` study, a single `gliner2_training` run improved, but the multi-seed check turned negative on average
- the conservative subset (`review_confidence=high` and `decision=accept`) still degraded across the tested seeds
- do not assume that disagreement adjudication artifacts are suitable training pseudolabels just because they are useful review benchmarks

When designing a new adjudication workflow specifically for training:

- keep it separate from the literal disagreement benchmark
- use a separate selector and benchmark directory
- do not reuse the old assumption that seed-bounded disagreement adjudication is automatically suitable for training

## 12b) Codex-vs-GPT Adjudication Benchmark

When comparing adjudication outputs from `gpt-5` and Codex over the same `05_llm_input` cases, use a chunked benchmark workflow instead of manual ad hoc copying.

Initialize a benchmark:

```bash
cd .
python3 src/tools/manage_codex_adjudication_benchmark.py init \
  --input artifacts/pseudolabelling/baseline_quick_2026-04-03/05_llm_input_t06_top1000.jsonl \
  --benchmark-dir artifacts/benchmarks/codex_adjudication_t06_top1000 \
  --benchmark-name codex_vs_gpt5_t06_top1000 \
  --chunk-size 10
```

Inspect progress:

```bash
python3 src/tools/manage_codex_adjudication_benchmark.py status \
  --state-json artifacts/benchmarks/codex_adjudication_t06_top1000/state.json
```

Export the next pending chunk:

```bash
python3 src/tools/manage_codex_adjudication_benchmark.py next \
  --state-json artifacts/benchmarks/codex_adjudication_t06_top1000/state.json
```

After Codex adjudicates the emitted chunk, ingest the structured responses:

```bash
python3 src/tools/manage_codex_adjudication_benchmark.py ingest \
  --state-json artifacts/benchmarks/codex_adjudication_t06_top1000/state.json \
  --chunk-id chunk_001 \
  --response-jsonl artifacts/benchmarks/codex_adjudication_t06_top1000/manual_responses/chunk_001.jsonl
```

Build the final consolidated Codex output:

```bash
python3 src/tools/manage_codex_adjudication_benchmark.py build-output \
  --state-json artifacts/benchmarks/codex_adjudication_t06_top1000/state.json
```

Operational notes:

- chunks are frozen at initialization time
- the tool validates adjudication structure and offsets on ingest
- the final output JSONL is directly comparable to the output of `run_llm_adjudication.py`

Operational semantics for the literal benchmark mode:

- `accept` and `accept_with_edits` are seed-set decisions, not open-ended extraction
- in this mode, adjudication must stay within `review_seed_entities`
- if you want a semantically broader adjudication workflow, treat it as a different benchmark mode and keep the artifacts separate

## 12c) Train-Oriented Adjudication Candidate Selection

If the goal is to generate better training pseudolabels rather than evaluate disagreement, do not start from the disagreement selector.

Use:

```bash
python3 src/tools/score_adjudication_candidates.py \
  --input artifacts/pseudolabelling/baseline_quick_2026-04-03/05_llm_input_t06_top1000.jsonl \
  --output-jsonl artifacts/pseudolabelling/baseline_quick_2026-04-03/05b_llm_input_t06_top1000_priority.jsonl \
  --summary-json artifacts/pseudolabelling/baseline_quick_2026-04-03/05b_llm_input_t06_top1000_priority_summary.json

python3 src/tools/select_train_annotation_cases.py \
  --input artifacts/pseudolabelling/baseline_quick_2026-04-03/05b_llm_input_t06_top1000_priority.jsonl \
  --output-jsonl artifacts/pseudolabelling/baseline_quick_2026-04-03/05_llm_input_train_adjudication_top100.jsonl \
  --summary-json artifacts/pseudolabelling/baseline_quick_2026-04-03/05_llm_input_train_adjudication_top100_summary.json \
  --top-n 100 \
  --max-text-length 900 \
  --min-seed-entities 1 \
  --max-seed-entities 4 \
  --max-union-entities 8 \
  --max-gliner2-noise-proxy 0.6 \
  --min-agreement-ratio 0.15 \
  --max-agreement-ratio 0.8 \
  --require-agreed-or-baseline-seed \
  --penalize-generic-seeds \
  --ranking-field adjudication_priority_score \
  --log-level INFO
```

This selector is biased toward:

- more stable cases
- lower-noise cases
- seeds with stronger origin evidence
- examples more likely to yield complete, conservative train annotations

The priority scorer is biased toward:

- domain-aligned DD-like narratives
- moderate agreement and moderate baseline confidence instead of trivial high-confidence cases
- presence of plausible `Location` seeds
- cases that are still easy enough for low-cost adjudication

Operational note:

- this still only selects candidates
- the LLM protocol for `train_annotation` must remain separate from the literal benchmark protocol

## 12d) Auditing Refit Regressions

Use `src/tools/audit_refit_regressions.py` after any `supervised_only` vs `supervised_plus_pseudolabels` comparison when you need to understand *why* the metrics moved.

The tool compares:

- one gold evaluation file
- one baseline `predictions.jsonl`
- one candidate/refit `predictions.jsonl`

and emits:

- `summary.json`
- `regressions.jsonl`
- `wins.jsonl`
- `ties.jsonl`
- `top_regressions.md`

Example:

```bash
python3 src/tools/audit_refit_regressions.py \
  --gold data/dd_corpus_small_test.json \
  --baseline-pred artifacts/gliner2_training/quick_supervised_only/eval_test/predictions.jsonl \
  --candidate-pred artifacts/gliner2_training/quick_supervised_plus_codex_top300_v1/eval_test/predictions.jsonl \
  --output-dir artifacts/audits/gliner2_top300_v1_vs_supervised_only \
  --title "GLiNER2 top300_v1 vs supervised_only"
```

The summary currently tracks:

- `wins`, `losses`, `ties`
- `micro_f1_delta`, `macro_f1_delta`
- `loss_reason_counts`
- `loss_reason_counts_by_label`
- `wrong_label_confusions`

Interpretation guidance:

- if a run gains on aggregate but also increases `spurious_entity`, the improvement may be recall-heavy and unstable
- high `boundary_or_partial` suggests span drift or partial-supervision effects
- high `wrong_label_confusions` suggests label drift induced by the refit data
- use this audit before deciding to scale any new pseudolabel source

## 12e) Prompt Probing Before Launching A New Train-Annotation Benchmark

Before spending more benchmark budget on a new `train_annotation` workflow, test the prompt on a small, diagnostic probe.

Use:

```bash
python3 src/tools/build_train_annotation_prompt_probe.py \
  --regressions-jsonl artifacts/audits/gliner2_train_adjudication_partial50_seed17_vs_supervised_only/regressions.jsonl \
  --wins-jsonl artifacts/audits/gliner2_train_adjudication_partial50_seed13_vs_supervised_only/wins.jsonl \
  --source-input artifacts/pseudolabelling/baseline_quick_2026-04-03/05_llm_input_train_adjudication_top100.jsonl \
  --output-jsonl artifacts/prompt_probes/train_annotation_prompt_probe_10.jsonl \
  --summary-json artifacts/prompt_probes/train_annotation_prompt_probe_10_summary.json \
  --top-location-person 3 \
  --top-location-org 2 \
  --top-boundary 2 \
  --top-spurious 2 \
  --top-wins 1 \
  --log-level INFO
```

The output JSONL is designed to be pasted into ChatGPT/Codex for prompt testing. It keeps only:

- `source_id`
- `text`
- `review_seed_entities`
- `_probe_meta`

Recommended use:

- test the prompt on `5-10` cases first
- cover failure modes such as:
  - `Location -> Person`
  - `Location -> Organization`
  - `boundary_or_partial`
  - `spurious_entity`
- refine the prompt before opening a new full benchmark

## 12f) Location Prefix Expansion As Corpus Normalization

If the corpus mixes `Rua X` and `X` as `Location` boundaries, normalize that first before attributing instability to pseudolabels.

Use:

```bash
python3 src/tools/expand_location_spans_with_markers.py \
  --input data/dd_corpus_small_train.json \
  --output artifacts/corpus_normalization/dd_corpus_small_train_location_prefix_expanded.json \
  --summary-json artifacts/corpus_normalization/dd_corpus_small_train_location_prefix_expanded_summary.json
```

```bash
python3 src/tools/expand_location_spans_with_markers.py \
  --input data/dd_corpus_small_test.json \
  --output artifacts/corpus_normalization/dd_corpus_small_test_location_prefix_expanded.json \
  --summary-json artifacts/corpus_normalization/dd_corpus_small_test_location_prefix_expanded_summary.json
```

Operational reading:

- this step expands `Location` spans to include explicit locative markers such as `rua`, `tr`, `trav`, `trv`, `av`, `avenida`, and similar forms
- it is meant to reduce boundary inconsistency in logradouro mentions
- a small number of noisy cases may remain, but the suspicious cases appear to be a minority relative to the total number of expanded spans

Observed result in `gliner2_training` baseline:

- single-run baseline with the normalized corpora improved from:
  - `micro_f1 0.6320 -> 0.6735`
  - `macro_f1 0.6137 -> 0.6444`
- multi-seed baseline comparison also stayed positive on average:
  - `mean delta micro = +0.0204`
  - `mean delta macro = +0.0112`
  - `mean delta Location F1 = +0.0368`
  - `mean delta Organization F1 = +0.0040`
  - `mean delta Person F1 = -0.0072`

Interpretation:

- location-boundary inconsistency in the gold corpus was a material source of noise
- this corpus normalization produced a stronger and more stable gain than the recent train-adjudication pseudolabel experiments
- treat `quick_supervised_only_locprefix_expanded` as the candidate reference baseline for subsequent comparisons
- normalize the corpus first, then re-evaluate pseudolabel experiments on top of the corrected baseline

## 13) Controlled Refit Comparison For Dissertation Experiments

Use the same final holdout `../data/dd_corpus_small_test.json` for both runs below.

First, measure the gain from additional supervised refit only:

```bash
cd src
python3 -m pseudolabelling.run_iterative_cycle \
  --run-dir ../artifacts/pseudolabelling/iter_cycle_supervised_only_t050 \
  --model-path ../artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --prediction-calibrator-path ../artifacts/calibration/base_model/calibrator.json \
  --input-jsonl ../data/dd_corpus_small_calibration.json \
  --labels Person,Location,Organization \
  --text-fields text \
  --prediction-batch-size 16 \
  --prediction-max-tokens 512 \
  --prediction-threshold 0.0 \
  --record-score-field score_context_boosted \
  --split-threshold 0.50 \
  --refit-mode supervised_only \
  --refit-base-model ../artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --refit-supervised-train-path ../data/dd_corpus_small_train.json \
  --refit-epochs 3 \
  --refit-patience 2 \
  --refit-batch-size 16 \
  --refit-max-length 512 \
  --refit-overlap 128 \
  --refit-lr 1e-5 \
  --refit-weight-decay 0.01 \
  --evaluate-refit \
  --eval-gt-jsonl ../data/dd_corpus_small_test.json \
  --eval-prediction-threshold 0.05 \
  --eval-batch-size 16 \
  --eval-max-tokens 512 \
  --prepare-next-iteration \
  --log-level INFO
```

Then, measure the full semisupervised recipe:

```bash
cd src
python3 -m pseudolabelling.run_iterative_cycle \
  --run-dir ../artifacts/pseudolabelling/iter_cycle_supervised_plus_pseudolabels_t050 \
  --model-path ../artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --prediction-calibrator-path ../artifacts/calibration/base_model/calibrator.json \
  --input-jsonl ../data/dd_corpus_small_calibration.json \
  --labels Person,Location,Organization \
  --text-fields text \
  --prediction-batch-size 16 \
  --prediction-max-tokens 512 \
  --prediction-threshold 0.0 \
  --record-score-field score_context_boosted \
  --split-threshold 0.50 \
  --refit-mode supervised_plus_pseudolabels \
  --refit-base-model ../artifacts/base_model_training/experiments/baseline_real_bs16_ml512/best_overall_gliner_model \
  --refit-supervised-train-path ../data/dd_corpus_small_train.json \
  --refit-epochs 3 \
  --refit-patience 2 \
  --refit-batch-size 16 \
  --refit-max-length 512 \
  --refit-overlap 128 \
  --refit-lr 1e-5 \
  --refit-weight-decay 0.01 \
  --evaluate-refit \
  --eval-gt-jsonl ../data/dd_corpus_small_test.json \
  --eval-prediction-threshold 0.05 \
  --eval-batch-size 16 \
  --eval-max-tokens 512 \
  --prepare-next-iteration \
  --log-level INFO
```

Interpretation:

- compare `09_base_vs_refit_comparison.json` from the two runs
- the marginal effect of pseudolabelling is:
  - `(supervised_plus_pseudolabels) - (supervised_only)`
- do not attribute all improvement over the base model to pseudolabelling

Observed controlled result for `t020`:

- base:
  - `micro_f1=0.4440`
  - `macro_f1=0.4362`
- supervised-only refit:
  - `micro_f1=0.4952`
  - `macro_f1=0.4772`
- supervised-plus-pseudolabels refit:
  - `micro_f1=0.5260`
  - `macro_f1=0.5097`

Interpretation of the `t020` result:

- extra supervised refit gain over base:
  - `micro_f1 +0.0512`
  - `macro_f1 +0.0411`
- additional pseudolabelling gain over supervised-only:
  - `micro_f1 +0.0308`
  - `macro_f1 +0.0324`

## 13a) Operational Protocol For Experimental Pseudolabel Comparisons

Use this protocol whenever a run is intended to support the experimental claim
that pseudolabel augmentation improves NER performance.

### Required Experimental Conditions

Every controlled comparison must include:

1. `base`
2. `supervised_only`
3. `supervised_plus_pseudolabels`

Interpret them as:

- `base -> supervised_only`
  - effect of additional supervised fine-tuning from the baseline checkpoint
- `base -> supervised_plus_pseudolabels`
  - total effect of the semisupervised recipe
- `supervised_only -> supervised_plus_pseudolabels`
  - marginal effect of pseudolabel augmentation

Do not attribute all post-refit gain over `base` to pseudolabelling.

### Fixed Protocol Requirements

Keep the following fixed across `supervised_only` and `supervised_plus_pseudolabels`:

- same starting checkpoint
- same supervised training corpus
- same supervised validation split or same explicit `val_jsonl`
- same seed
- same learning rate
- same batch size
- same max length
- same overlap
- same patience
- same epoch budget
- same holdout for final evaluation

Only the pseudolabel input should vary.

### Refit Semantics

The controlled refit comparison is not a train-from-scratch comparison.

It is a paired comparison where both conditions:

- start from the same baseline checkpoint
- split supervised data first
- keep validation supervised-only
- append pseudolabels only to the training side

### Pseudolabel Volume Schedule

When the goal is to characterize the gain curve rather than just detect any
positive signal, run more than one pseudolabel volume. Recommended schedule:

- `+100`
- `+250`
- `+500`
- `+1000`

For the full schedule, keep the adjudicator fixed:

- same adjudication model
- same prompt
- same schema
- same acceptance policy

### Required Reporting Per Run

For each controlled run, preserve and report:

- supervised train size
- supervised validation size
- pseudolabel rows emitted
- pseudolabel rows actually merged into train
- pseudolabel rows dropped by deduplication
- pseudolabel label distribution
- final holdout metrics

Prefer keeping these artifacts:

- refit stats JSON
- train manifest JSONL
- validation manifest JSONL
- evaluation metrics JSON
- evaluation summary JSON

### Provisional vs Final Evidence

If a run was executed before corpus cleanup, before protocol stabilization, or
before the supervised-only control was included, mark it explicitly as:

- exploratory
- provisional
- not final experimental evidence

Those runs are still useful for deciding whether to scale adjudication or refit,
but they should not be treated as the final answer to the research question.

## Troubleshooting
- `Killed`: reduce `batch-size`, `max-length`, number of folds/trials.
- HF timeout/network instability: run after cache warmup with offline env vars.
- JSON parse issue `Extra data`: file is JSONL, not a single JSON array.
