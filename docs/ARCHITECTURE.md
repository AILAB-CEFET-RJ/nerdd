# Architecture

## Entrypoints
- `python3 -m base_model_training.train_nested_kfold`: training CLI entrypoint.
- `base_model_training/evaluate_gliner.py`: evaluation CLI entrypoint.
- `pseudolabelling/generate_corpus_predictions.py`: inference-only large-corpus entity prediction.
- `calibration/fit_calibrator.py`: fit a reusable calibration artifact from labeled holdout predictions.
- `calibration/apply_calibrator.py`: apply a saved calibration artifact to entity-score JSONL.

## Training Package (`base_model_training/`)
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

## Evaluation Package (`base_model_training/`)
- `eval_cli.py`: evaluation CLI parsing.
- `eval_config.py`: evaluation defaults.
- `evaluate.py`: prediction, calibration, report generation.

## Pseudolabelling Package (`pseudolabelling/`)
- `pipeline.py`: large-corpus prediction pipeline.
- `cli.py`: CLI for large-corpus prediction.
- `config.py`: defaults for large-corpus prediction.

## Calibration Package (`calibration/`)
- `fit_calibrator.py`: fit and persist calibrator artifacts.
- `apply_calibrator.py`: apply calibrator artifacts to prediction JSONL.
- `serialization.py`: calibrator JSON serialization and score application.
- `pipeline.py`: legacy fit+apply orchestration path.
- `cli.py`: legacy calibration CLI parsing.
- `config.py`: legacy calibration defaults.
- `methods/temperature.py`: global/per-class temperature scaling helpers.
- `methods/isotonic.py`: isotonic regression helpers.

## Extension Points
- Tune search space: `train_config.py` + CLI flags.
- Add metrics: `metrics.py` and reporting in `cv.py`.
- Change split strategy: `cv.py` (`_effective_group_kfold`, `_extract_groups`).
