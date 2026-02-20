import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Compute record-level score from entity-level scores")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL with entity predictions")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with record-level score field")
    parser.add_argument("--stats-json", required=True, help="Output JSON with run statistics")
    parser.add_argument("--score-field", default="score", help="Entity score field")
    parser.add_argument("--output-field", default="record_score", help="Record-level score output field")
    parser.add_argument("--legacy-field-alias", default="score_relato", help="Optional legacy alias field")
    parser.add_argument("--entity-key", default="entities", help="Primary entity list key (fallback to 'ner')")
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max", "median", "p75"],
        default="mean",
        help="Aggregation method for record-level score",
    )
    parser.add_argument(
        "--empty-entities-policy",
        choices=["zero", "null", "error"],
        default="zero",
        help="How to handle records without valid entity scores",
    )
    parser.add_argument("--trace-key", default="_record_score_meta", help="Optional metadata key to store per-record trace")
    parser.add_argument("--no-trace", action="store_true", help="Disable per-record trace metadata")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()
