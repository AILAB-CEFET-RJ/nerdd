import logging

from pseudolabelling.split_by_score import run_split
from pseudolabelling.split_by_score_cli import parse_args


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_split(
        input_jsonl=args.input_jsonl,
        out_dir=args.out_dir,
        score_field=args.score_field,
        threshold=args.threshold,
        operator=args.operator,
        fallback_score_field=args.fallback_score_field,
        missing_policy=args.missing_policy,
        trace_key=args.trace_key,
        legacy_filenames=args.legacy_filenames,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
