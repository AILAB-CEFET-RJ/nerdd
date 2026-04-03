# NERDD

NER pipeline for the Disque Denúncia context, organized into training, calibration, and pseudolabelling subpipelines.

## Current Structure

- `src/`: main source code.
- `src/base_model_training/`: base training and evaluation with nested CV.
- `src/pseudolabelling/`: pseudolabel generation, score-based split, and refit.
- `src/calibration/`: fit/apply reusable probability calibrators for the base model scores.
- `src/tools/`: auxiliary utilities.
- `docs/`: operational and architectural documentation.
- `data/`: training, test, and calibration datasets.
- `artifacts/corpus_sanitization/`: derived large-corpus artifacts promoted for pseudolabelling input.

## Prerequisites

- Git
- Python 3.11+
- pip

## Quick Setup

```bash
git clone https://github.com/MLRG-CEFET-RJ/nerdd.git
cd nerdd
cd src
pip install -r requirements.txt
```

## Next Steps

- Detailed installation: `docs/INSTALL.md`
- Runbook: `docs/RUNBOOK.md`
- Pipeline overview: `docs/PIPELINE_OVERVIEW.md`
- Architecture: `docs/ARCHITECTURE.md`
- Architectural decisions: `docs/ARCHITECTURAL_DECISIONS.md`
- Migration: `docs/MIGRATION.md`

## Canonical Flow

1. Train the base model in `src/base_model_training/`.
2. Build a labeled calibration subset and fit a reusable calibrator artifact in `src/calibration/`.
3. Sanitize the raw large corpus with `src/tools/sanitize_dd_corpus.py`.
4. Run large-corpus prediction in `src/pseudolabelling/`, optionally applying the calibrator during inference, using `artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl`.

Operational note:

- `data/dd_corpus_large.json` is the raw corpus.
- `data/` should contain only canonical input datasets.
- `artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl` is the official pseudolabelling input.
- derived outputs such as calibration CSVs, model checkpoints, prediction JSONL files, summaries, and HTML inspections belong under `artifacts/`

## Contributing

Open an issue or PR with fixes and improvements.
