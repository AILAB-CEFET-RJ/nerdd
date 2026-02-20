# NERDD

NER pipeline for the Disque Denúncia context, organized into training, pseudolabelling, and calibration subpipelines.

## Current Structure

- `src/`: main source code.
- `src/base_model_training/`: base training and evaluation with nested CV.
- `src/pseudolabelling/`: pseudolabel generation, score-based split, and refit.
- `src/calibration/`: score calibration.
- `src/tools/`: auxiliary utilities.
- `docs/`: operational and architectural documentation.
- `data/`: training, test, and calibration datasets.

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
- Migration: `docs/MIGRATION.md`

## Contributing

Open an issue or PR with fixes and improvements.
