from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    seed: int = 42
    train_path: str = "../data/dd_corpus_small_train.json"
    model_base: str = "birdred/glinerdd"
    batch_size: int = 4
    num_epochs: int = 20
    n_splits: int = 3
    n_inner_splits: int = 3
    num_trials: int = 5
    search_mode: str = "grid"
    max_length: int = 384
    overlap: int = 100
    thresholds: list[float] = field(default_factory=lambda: [0.6])
    lr_values: list[float] = field(default_factory=lambda: [3.38e-5])
    weight_decay_values: list[float] = field(default_factory=lambda: [0.086619])
    refit_val_size: float = 0.2
    early_stopping_patience: int = 7
    early_stopping_threshold: float = 0.5
    results_file: str = "nested_cv_results.txt"
    results_json_file: str = "nested_cv_results.json"
    output_dir: str = "."
