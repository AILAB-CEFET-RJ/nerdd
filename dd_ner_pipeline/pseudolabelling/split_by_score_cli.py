import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Split pseudolabel records by a record-level score threshold")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL with record-level score fields")
    parser.add_argument("--out-dir", required=True, help="Output directory for split artifacts")
    parser.add_argument("--score-field", required=True, help="Primary record-level score field")
    parser.add_argument("--threshold", type=float, required=True, help="Decision threshold")
    parser.add_argument(
        "--operator",
        choices=["ge", "gt", "le", "lt"],
        default="ge",
        help="Comparison operator against threshold",
    )
    parser.add_argument("--fallback-score-field", default="", help="Fallback score field if primary is missing")
    parser.add_argument(
        "--missing-policy",
        choices=["discard", "zero", "error"],
        default="discard",
        help="How to handle records with missing score fields",
    )
    parser.add_argument("--trace-key", default="_split", help="Per-record decision trace key")
    parser.add_argument(
        "--legacy-filenames",
        action="store_true",
        help="Also write mantidos/descartados/resumo legacy filenames for compatibility",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()
