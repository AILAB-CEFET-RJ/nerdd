#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random


def read_json_array(path):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array dataset.")
    return data


def write_json_array(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def extract_spans(row):
    spans = row.get("spans")
    if not isinstance(spans, list):
        return []
    return spans


def build_profiles(rows, label_field):
    label_names = sorted(
        {
            str(span.get(label_field))
            for row in rows
            for span in extract_spans(row)
            if span.get(label_field) is not None
        }
    )
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}

    profiles = []
    global_presence = [0.0] * len(label_names)
    global_spans = [0.0] * len(label_names)

    for index, row in enumerate(rows):
        presence = [0.0] * len(label_names)
        span_counts = [0.0] * len(label_names)
        spans = extract_spans(row)
        for span in spans:
            label = str(span.get(label_field))
            if label not in label_to_idx:
                continue
            label_idx = label_to_idx[label]
            presence[label_idx] = 1.0
            span_counts[label_idx] += 1.0

        for label_idx, value in enumerate(presence):
            global_presence[label_idx] += value
        for label_idx, value in enumerate(span_counts):
            global_spans[label_idx] += value

        profiles.append(
            {
                "index": index,
                "presence": presence,
                "span_counts": span_counts,
                "num_spans": int(sum(span_counts)),
            }
        )

    return profiles, label_names, global_presence, global_spans


def build_split(rows, ratio, seed, label_field):
    if not 0.0 < ratio < 1.0:
        raise ValueError("--calibration-ratio must be between 0 and 1.")

    profiles, label_names, global_presence, global_spans = build_profiles(rows, label_field=label_field)
    if not profiles:
        raise ValueError("Dataset is empty.")

    calibration_target = max(1, min(len(rows) - 1, round(len(rows) * ratio)))
    final_test_target = len(rows) - calibration_target

    rng = Random(seed)
    order = list(range(len(profiles)))
    rng.shuffle(order)

    rarity_scores = []
    for profile in profiles:
        rarity = 0.0
        for label_idx, present in enumerate(profile["presence"]):
            if present:
                rarity += 1.0 / max(1.0, global_presence[label_idx])
        rarity_scores.append(rarity)

    order.sort(
        key=lambda idx: (
            rarity_scores[idx],
            profiles[idx]["num_spans"],
        ),
        reverse=True,
    )

    target_presence = [value * ratio for value in global_presence]
    target_spans = [value * ratio for value in global_spans]

    calibration = {
        "indices": [],
        "presence": [0.0] * len(label_names),
        "span_counts": [0.0] * len(label_names),
        "rows": 0,
    }

    def cost_for_state(rows_count, presence_values, span_values):
        row_cost = ((rows_count - calibration_target) / max(1.0, calibration_target)) ** 2

        presence_cost = 0.0
        span_cost = 0.0
        valid_presence = [idx for idx, value in enumerate(target_presence) if value > 0]
        if valid_presence:
            terms = []
            for idx in valid_presence:
                terms.append(((presence_values[idx] - target_presence[idx]) / target_presence[idx]) ** 2)
            presence_cost = sum(terms) / len(terms)

        valid_spans = [idx for idx, value in enumerate(target_spans) if value > 0]
        if valid_spans:
            terms = []
            for idx in valid_spans:
                terms.append(((span_values[idx] - target_spans[idx]) / target_spans[idx]) ** 2)
            span_cost = sum(terms) / len(terms)

        return row_cost + (3.0 * presence_cost) + (2.0 * span_cost)

    final_indices = []
    total_profiles = len(order)

    for position, profile_idx in enumerate(order):
        profile = profiles[profile_idx]
        remaining = total_profiles - position
        slots_needed = calibration_target - calibration["rows"]

        if slots_needed <= 0:
            choose_calibration = False
        elif remaining == slots_needed:
            choose_calibration = True
        else:
            current_cost = cost_for_state(
                calibration["rows"],
                calibration["presence"],
                calibration["span_counts"],
            )
            projected_presence = [
                calibration["presence"][idx] + profile["presence"][idx]
                for idx in range(len(label_names))
            ]
            projected_spans = [
                calibration["span_counts"][idx] + profile["span_counts"][idx]
                for idx in range(len(label_names))
            ]
            add_cost = cost_for_state(
                calibration["rows"] + 1,
                projected_presence,
                projected_spans,
            )
            choose_calibration = add_cost <= current_cost

        if choose_calibration:
            calibration["indices"].append(profile["index"])
            calibration["rows"] += 1
            for idx, value in enumerate(profile["presence"]):
                calibration["presence"][idx] += value
            for idx, value in enumerate(profile["span_counts"]):
                calibration["span_counts"][idx] += value
        else:
            final_indices.append(profile["index"])

    calibration_rows = [rows[idx] for idx in sorted(calibration["indices"])]
    final_test_rows = [rows[idx] for idx in sorted(final_indices)]
    summary = {
        "label_names": label_names,
        "calibration": summarize_rows(calibration_rows, label_field),
        "final_test": summarize_rows(final_test_rows, label_field),
    }
    return calibration_rows, final_test_rows, summary


def build_random_split(rows, ratio, seed, label_field):
    if not 0.0 < ratio < 1.0:
        raise ValueError("--calibration-ratio must be between 0 and 1.")
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows to create calibration and final-test splits.")

    indices = list(range(len(rows)))
    rng = Random(seed)
    rng.shuffle(indices)
    calibration_target = max(1, min(len(rows) - 1, round(len(rows) * ratio)))

    calibration_indices = sorted(indices[:calibration_target])
    final_test_indices = sorted(indices[calibration_target:])

    calibration_rows = [rows[idx] for idx in calibration_indices]
    final_test_rows = [rows[idx] for idx in final_test_indices]
    summary = {
        "label_names": sorted(
            {
                str(span.get(label_field))
                for row in rows
                for span in extract_spans(row)
                if span.get(label_field) is not None
            }
        ),
        "calibration": summarize_rows(calibration_rows, label_field),
        "final_test": summarize_rows(final_test_rows, label_field),
    }
    return calibration_rows, final_test_rows, summary


def summarize_rows(rows, label_field):
    label_counts = Counter()
    total_entities = 0
    for row in rows:
        spans = extract_spans(row)
        total_entities += len(spans)
        for span in spans:
            label_counts[str(span.get(label_field, "UNKNOWN"))] += 1
    return {
        "rows": len(rows),
        "entities": total_entities,
        "labels": dict(sorted(label_counts.items())),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a labeled JSON dataset into calibration and final-test subsets."
    )
    parser.add_argument("--input", required=True, help="Input JSON array dataset.")
    parser.add_argument("--calibration-output", required=True, help="Output JSON path for calibration subset.")
    parser.add_argument("--final-test-output", required=True, help="Output JSON path for final test subset.")
    parser.add_argument("--summary-json", default="", help="Optional JSON summary output.")
    parser.add_argument("--calibration-ratio", type=float, default=0.2, help="Fraction of rows assigned to calibration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic tie-breaking.")
    parser.add_argument("--label-field", default="label", help="Label field name inside each span.")
    parser.add_argument(
        "--mode",
        choices=["random", "heuristic"],
        default="random",
        help="Split strategy. Use random by default for calibration holdout creation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_array(args.input)
    if args.mode == "random":
        calibration_rows, final_test_rows, summary = build_random_split(
            rows,
            ratio=args.calibration_ratio,
            seed=args.seed,
            label_field=args.label_field,
        )
    else:
        calibration_rows, final_test_rows, summary = build_split(
            rows,
            ratio=args.calibration_ratio,
            seed=args.seed,
            label_field=args.label_field,
        )

    write_json_array(args.calibration_output, calibration_rows)
    write_json_array(args.final_test_output, final_test_rows)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary["mode"] = args.mode
        summary["seed"] = args.seed
        summary["calibration_ratio"] = args.calibration_ratio
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Calibration split ({args.mode})")
    print(f"  output: {args.calibration_output}")
    print(f"  relatos: {summary['calibration']['rows']}")
    print(f"  entidades: {summary['calibration']['entities']}")
    for label, count in summary["calibration"]["labels"].items():
        print(f"  {label}: {count}")
    print()

    print("Final test split")
    print(f"  output: {args.final_test_output}")
    print(f"  relatos: {summary['final_test']['rows']}")
    print(f"  entidades: {summary['final_test']['entities']}")
    for label, count in summary["final_test"]["labels"].items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
