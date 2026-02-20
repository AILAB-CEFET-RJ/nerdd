import argparse


DEFAULT_KEEP_FIELDS = [
    "assunto",
    "relato",
    "logradouroLocal",
    "bairroLocal",
    "cidadeLocal",
    "pontodeReferenciaLocal",
]


def _parse_csv_list(raw_value):
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one value must be provided.")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare discarded pseudolabel records for the next iteration (projection + cleanup)"
    )
    parser.add_argument("--input-jsonl", default="", help="Single input JSONL path")
    parser.add_argument("--input-glob", default="", help="Glob pattern for batch mode (e.g. ./iter01/*_descartados.jsonl)")
    parser.add_argument("--output-jsonl", default="", help="Single output JSONL path (required in single mode)")
    parser.add_argument("--out-dir", default=".", help="Output directory (used in batch mode)")
    parser.add_argument("--output-suffix", default="_next_iter", help="Suffix appended to output stems in batch mode")
    parser.add_argument("--keep-fields", default=",".join(DEFAULT_KEEP_FIELDS))
    parser.add_argument("--required-fields", default="relato", help="Fields required to be non-empty after projection")
    parser.add_argument("--fill-missing-with", default="", help="Default value for missing fields")
    parser.add_argument(
        "--coerce-non-string",
        choices=["stringify", "empty", "error"],
        default="stringify",
        help="How to handle non-string values in kept fields",
    )
    parser.add_argument("--drop-empty-relato", action="store_true", help="Drop rows with empty 'relato' after projection")
    parser.add_argument("--deduplicate-by", default="", help="Comma-separated fields used as deduplication key")
    parser.add_argument("--allow-json", action="store_true", help="Allow parsing JSON object/list fallback when input is not JSONL")
    parser.add_argument("--stats-json", default="./prepare_next_iteration_stats.json")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args):
    input_jsonl = args.input_jsonl.strip()
    input_glob = args.input_glob.strip()
    if bool(input_jsonl) == bool(input_glob):
        raise ValueError("Use exactly one of --input-jsonl or --input-glob.")
    if input_jsonl and not args.output_jsonl.strip():
        raise ValueError("--output-jsonl is required when using --input-jsonl.")

    return {
        "input_jsonl": input_jsonl,
        "input_glob": input_glob,
        "output_jsonl": args.output_jsonl.strip(),
        "out_dir": args.out_dir,
        "output_suffix": args.output_suffix,
        "keep_fields": _parse_csv_list(args.keep_fields),
        "required_fields": _parse_csv_list(args.required_fields),
        "fill_missing_with": args.fill_missing_with,
        "coerce_non_string": args.coerce_non_string,
        "drop_empty_relato": args.drop_empty_relato,
        "deduplicate_by": _parse_csv_list(args.deduplicate_by) if args.deduplicate_by.strip() else [],
        "allow_json": args.allow_json,
        "stats_json": args.stats_json,
    }
