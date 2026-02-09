import argparse

from pseudolabelling.config import ContextBoostConfig


def _parse_csv_list(raw_value):
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one value must be provided.")
    return values


def parse_args():
    defaults = ContextBoostConfig()
    parser = argparse.ArgumentParser(description="Apply metadata-aware confidence boost to pseudolabel entities")
    parser.add_argument("--input-jsonl", default=defaults.input_jsonl)
    parser.add_argument("--output-jsonl", default=defaults.output_jsonl)
    parser.add_argument("--stats-json", default=defaults.stats_json)
    parser.add_argument("--text-field-priority", default=",".join(defaults.text_field_priority))
    parser.add_argument("--metadata-fields", default=",".join(defaults.metadata_fields))
    parser.add_argument("--label-field", default=defaults.label_field)
    parser.add_argument("--base-score-field", default=defaults.base_score_field)
    parser.add_argument("--fallback-score-fields", default=",".join(defaults.fallback_score_fields))
    parser.add_argument("--output-score-field", default=defaults.output_score_field)
    parser.add_argument("--output-record-score-field", default=defaults.output_record_score_field)
    parser.add_argument("--boost-factor", type=float, default=defaults.boost_factor)
    parser.add_argument("--per-match", action="store_true")
    parser.add_argument(
        "--boost-scope",
        choices=["all-entities", "location-only", "matched-only"],
        default=defaults.boost_scope,
    )
    parser.add_argument(
        "--match-policy",
        choices=["any-metadata-in-text", "entity-metadata-overlap"],
        default=defaults.match_policy,
    )
    parser.add_argument("--location-labels", default=",".join(defaults.location_labels))
    parser.add_argument("--no-clamp", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Write per-record trace fields")
    parser.add_argument("--no-legacy-fields", action="store_true", help="Disable old score_confianca fields")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    return ContextBoostConfig(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        stats_json=args.stats_json,
        text_field_priority=_parse_csv_list(args.text_field_priority),
        metadata_fields=_parse_csv_list(args.metadata_fields),
        label_field=args.label_field,
        base_score_field=args.base_score_field,
        fallback_score_fields=_parse_csv_list(args.fallback_score_fields),
        output_score_field=args.output_score_field,
        output_record_score_field=args.output_record_score_field,
        boost_factor=args.boost_factor,
        per_match=args.per_match,
        clamp_scores=(not args.no_clamp),
        boost_scope=args.boost_scope,
        match_policy=args.match_policy,
        location_labels=_parse_csv_list(args.location_labels),
        write_trace_fields=args.trace,
        write_legacy_fields=(not args.no_legacy_fields),
    )
