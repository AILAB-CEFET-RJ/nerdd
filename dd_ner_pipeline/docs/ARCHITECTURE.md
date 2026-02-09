# Architecture

## Entrypoints
- `train_nested_kfold.py`: training CLI entrypoint.
- `gliner_train/evaluate_gliner.py`: evaluation CLI entrypoint.
- `pseudolabelling/generate_corpus_predictions.py`: inference-only large-corpus entity prediction.
- `calibration/run_calibration.py`: score calibration over pseudolabelled outputs.

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

## Pseudolabelling Package (`pseudolabelling/`)
- `pipeline.py`: large-corpus prediction pipeline.
- `cli.py`: CLI for large-corpus prediction.
- `config.py`: defaults for large-corpus prediction.

## Calibration Package (`calibration/`)
- `pipeline.py`: calibration orchestration and report generation.
- `cli.py`: calibration CLI parsing.
- `config.py`: calibration defaults.
- `methods/temperature.py`: global/per-class temperature scaling helpers.
- `methods/isotonic.py`: isotonic regression helpers.

## Extension Points
- Tune search space: `train_config.py` + CLI flags.
- Add metrics: `metrics.py` and reporting in `cv.py`.
- Change split strategy: `cv.py` (`_effective_group_kfold`, `_extract_groups`).
