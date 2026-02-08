from dataclasses import dataclass, field


@dataclass
class EvaluationConfig:
    model_path: str = "best_overall_gliner_model"
    gt_jsonl: str = "gliner_teste_sanidade.json"
    pred_jsonl: str = "gliner_teste_sanidade_resultado.json"
    labels: list[str] = field(default_factory=lambda: ["Person", "Location", "Organization"])
    batch_size: int = 4
    chunk_size: int = 384
    prediction_threshold: float = 0.0
    threshold_grid: list[float] = field(default_factory=lambda: [round(x * 0.05, 2) for x in range(0, 21)])
    calibrated_thresholds_json: str = "calibrated_thresholds.json"
    report_path: str = "classification_report_calibrated.txt"
