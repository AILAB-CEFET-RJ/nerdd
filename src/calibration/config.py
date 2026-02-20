from dataclasses import dataclass, field


@dataclass
class CalibrationConfig:
    method: str = "temperature"
    input_jsonl: str = "dd_corpus_large_pseudolabel_score0.json"
    output_jsonl: str = "dd_corpus_large_calibrated.jsonl"
    stats_json: str = "dd_corpus_large_calibration_stats.json"
    score_field: str = "score"
    output_score_field: str = "score_calibrated"
    preserve_original_score_field: str = "score_original"
    label_field: str = "label"
    labels: list[str] = field(default_factory=lambda: ["Person", "Location", "Organization"])
    label_source: str = "score-threshold"
    calibration_csv: str = "../data/comparacao_calibracao.csv"
    csv_score_col: str = "Score"
    csv_label_col: str = "Validacao"
    csv_class_col: str = ""
    positive_threshold: float = 0.9
    lower_quantile: float = 30.0
    upper_quantile: float = 70.0
    temperature_min: float = 0.5
    temperature_max: float = 5.0
    temperature_grid_size: int = 181
