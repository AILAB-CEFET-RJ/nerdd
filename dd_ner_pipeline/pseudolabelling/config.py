from dataclasses import dataclass, field


@dataclass
class CorpusPredictConfig:
    model_path: str = "best_overall_gliner_model"
    input_jsonl: str = "dd_corpus_large.json"
    output_jsonl: str = "dd_corpus_large_predicted_entities.jsonl"
    stats_json: str = "corpus_prediction_stats.json"
    labels: list[str] = field(default_factory=lambda: ["Person", "Location", "Organization"])
    text_fields: list[str] = field(
        default_factory=lambda: [
            "assunto",
            "relato",
            "bairroLocal",
            "logradouroLocal",
            "cidadeLocal",
            "pontodeReferenciaLocal",
        ]
    )
    join_separator: str = ". "
    batch_size: int = 4
    max_tokens: int = 384
    score_threshold: float = 0.0
    keep_inference_text: bool = False


@dataclass
class ContextBoostConfig:
    input_jsonl: str = "dd_corpus_large_predicted_entities.jsonl"
    output_jsonl: str = "dd_corpus_large_context_boosted.jsonl"
    stats_json: str = "context_boost_stats.json"
    text_field_priority: list[str] = field(default_factory=lambda: ["relato", "text"])
    metadata_fields: list[str] = field(
        default_factory=lambda: [
            "logradouroLocal",
            "bairroLocal",
            "cidadeLocal",
            "pontodeReferenciaLocal",
        ]
    )
    label_field: str = "label"
    base_score_field: str = "score"
    fallback_score_fields: list[str] = field(default_factory=lambda: ["score_calibrated", "score_ts", "score_iso"])
    output_score_field: str = "score_context_boosted"
    output_record_score_field: str = "record_score_context_boosted"
    boost_factor: float = 1.2
    per_match: bool = False
    clamp_scores: bool = True
    boost_scope: str = "all-entities"  # all-entities | location-only | matched-only
    match_policy: str = "any-metadata-in-text"  # any-metadata-in-text | entity-metadata-overlap
    location_labels: list[str] = field(default_factory=lambda: ["Location"])
    write_trace_fields: bool = False
    write_legacy_fields: bool = True
