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
    parser.add_argument("--entity-gate-score-field", default="", help="Optional entity-level score field for a second gate")
    parser.add_argument("--entity-gate-entity-key", default="entities", help="Entity list key used by the optional entity gate")
    parser.add_argument("--entity-gate-label-field", default="label", help="Entity label field used by the optional entity gate")
    parser.add_argument("--entity-gate-labels", default="Location", help="Comma-separated labels eligible for the optional entity gate")
    parser.add_argument(
        "--entity-gate-aggregation",
        choices=["mean", "max", "min"],
        default="max",
        help="Aggregation used by the optional entity gate",
    )
    parser.add_argument("--entity-gate-threshold", type=float, default=0.5, help="Threshold used by the optional entity gate")
    parser.add_argument(
        "--entity-gate-operator",
        choices=["ge", "gt", "le", "lt"],
        default="ge",
        help="Comparison operator used by the optional entity gate",
    )
    parser.add_argument(
        "--legacy-filenames",
        action="store_true",
        help="Also write mantidos/descartados/resumo legacy filenames for compatibility",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()
