# Architecture

## Entrypoints
- `train_nested_kfold.py`: training CLI entrypoint.
- `evaluate_gliner.py`: evaluation CLI entrypoint.

## Training Package (`gliner_train/`)
- `cli.py`: training CLI parsing.
- `train_config.py`: training defaults.
- `cv.py`: nested CV orchestration, reporting, model selection.
- `engine.py`: epoch loops + early stopping.
- `data.py`: dataset loading, preprocessing, chunking, dataloaders.
- `collator_factory.py`: GLiNER collator compatibility resolver.
- `metrics.py`: F1 computations.
- `search.py`: hyperparameter candidate generation.
- `plots.py`: loss curve plotting.
- `paths.py`: path resolution utilities.
- `io_utils.py`: JSON/JSONL helpers.

## Evaluation Package (`gliner_train/`)
- `eval_cli.py`: evaluation CLI parsing.
- `eval_config.py`: evaluation defaults.
- `evaluate.py`: prediction, calibration, report generation.

## Extension Points
- Tune search space: `train_config.py` + CLI flags.
- Add metrics: `metrics.py` and reporting in `cv.py`.
- Change split strategy: `cv.py` (`_effective_group_kfold`, `_extract_groups`).
