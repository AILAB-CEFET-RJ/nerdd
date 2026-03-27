#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl


def save_json(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def _normalize_row(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def summarize_labels(rows, label_field):
    counts = Counter()
    entities = 0
    for row in rows:
        spans = row.get("spans")
        if not isinstance(spans, list):
            continue
        entities += len(spans)
        for span in spans:
            label = span.get(label_field)
            if label is not None:
                counts[str(label)] += 1
    return {
        "rows": len(rows),
        "entities": entities,
        "labels": dict(sorted(counts.items())),
    }


def remove_exact_duplicates_across_inputs(train_rows, test_rows):
    train_keys = {_normalize_row(row) for row in train_rows}
    deduplicated_test = []
    dropped_from_test = 0
    for row in test_rows:
        if _normalize_row(row) in train_keys:
            dropped_from_test += 1
            continue
        deduplicated_test.append(row)
    return train_rows, deduplicated_test, dropped_from_test


def reshuffle_train_test(train_rows, test_rows, seed, train_output_size=None):
    combined = []
    for row in train_rows:
        combined.append({"row": row, "source": "train"})
    for row in test_rows:
        combined.append({"row": row, "source": "test"})

    order = list(range(len(combined)))
    Random(seed).shuffle(order)
    shuffled = [combined[idx]["row"] for idx in order]

    train_size = len(train_rows) if train_output_size is None else train_output_size
    new_train = shuffled[:train_size]
    new_test = shuffled[train_size:]
    return new_train, new_test, order


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild train/test splits by merging, shuffling, and re-splitting with original sizes."
    )
    parser.add_argument("--train-input", required=True, help="Original train JSON/JSONL.")
    parser.add_argument("--test-input", required=True, help="Original test JSON/JSONL.")
    parser.add_argument("--train-output", required=True, help="Output path for reshuffled train split.")
    parser.add_argument("--test-output", required=True, help="Output path for reshuffled test split.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--label-field", default="label", help="Span label field name.")
    parser.add_argument(
        "--remove-duplicates-across-inputs",
        action="store_true",
        help="Drop exact duplicate rows that appear in both inputs, keeping the train-side copy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_rows = load_jsonl(args.train_input)
    test_rows = load_jsonl(args.test_input)

    train_row_keys = {_normalize_row(row) for row in train_rows}
    test_row_keys = {_normalize_row(row) for row in test_rows}
    duplicates_across_inputs = len(train_row_keys & test_row_keys)

    effective_train_rows = train_rows
    effective_test_rows = test_rows
    dropped_duplicate_rows_from_test = 0
    if args.remove_duplicates_across_inputs:
        effective_train_rows, effective_test_rows, dropped_duplicate_rows_from_test = remove_exact_duplicates_across_inputs(
            train_rows, test_rows
        )

    new_train, new_test, order = reshuffle_train_test(
        effective_train_rows,
        effective_test_rows,
        seed=args.seed,
        train_output_size=len(effective_train_rows),
    )
    save_json(args.train_output, new_train)
    save_json(args.test_output, new_test)

    summary = {
        "train_input": str(Path(args.train_input).resolve()),
        "test_input": str(Path(args.test_input).resolve()),
        "train_output": str(Path(args.train_output).resolve()),
        "test_output": str(Path(args.test_output).resolve()),
        "seed": args.seed,
        "remove_duplicates_across_inputs": args.remove_duplicates_across_inputs,
        "sizes": {
            "train_input_rows": len(train_rows),
            "test_input_rows": len(test_rows),
            "train_effective_rows": len(effective_train_rows),
            "test_effective_rows": len(effective_test_rows),
            "train_output_rows": len(new_train),
            "test_output_rows": len(new_test),
        },
        "duplicates_across_inputs_exact_row_match": duplicates_across_inputs,
        "duplicates_removed_from_test_rows": dropped_duplicate_rows_from_test,
        "shuffle_order_preview": order[:20],
        "train_summary": summarize_labels(new_train, args.label_field),
        "test_summary": summarize_labels(new_test, args.label_field),
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Train rows in: {len(train_rows)} | out: {len(new_train)}")
    print(f"Test rows in: {len(test_rows)} | out: {len(new_test)}")
    print(f"Exact duplicate rows across inputs: {duplicates_across_inputs}")
    if args.remove_duplicates_across_inputs:
        print(f"Duplicate rows removed from test before reshuffle: {dropped_duplicate_rows_from_test}")
    print(f"Train output: {args.train_output}")
    print(f"Test output: {args.test_output}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
