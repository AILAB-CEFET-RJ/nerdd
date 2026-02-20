import logging

from pseudolabelling.compute_record_score import run_compute_record_score
from pseudolabelling.compute_record_score_cli import parse_args


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_compute_record_score(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        stats_json=args.stats_json,
        score_field=args.score_field,
        output_field=args.output_field,
        legacy_field_alias=args.legacy_field_alias,
        entity_key=args.entity_key,
        aggregation=args.aggregation,
        empty_entities_policy=args.empty_entities_policy,
        trace_key=args.trace_key,
        write_trace=(not args.no_trace),
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
