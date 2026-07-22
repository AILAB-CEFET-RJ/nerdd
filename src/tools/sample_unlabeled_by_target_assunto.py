#!/usr/bin/env python3
"""Sample an unlabeled pool toward a target assunto distribution.

The target distribution is usually derived from enriched labeled train/test files.
When the unlabeled pool lacks enough rows for rare target assuntos, allocation is
best-effort: scarce assuntos are exhausted and the remaining budget is
redistributed among available assuntos according to the remaining target mass.

Example:
python3 src/tools/sample_unlabeled_by_target_assunto.py \
  --input-jsonl data/large_sanitized/large_sanitized_no_labeled_overlap.jsonl \
  --target-input data/annotations_corrected_2001_with_assunto.json \
  --target-input data/dd_corpus_small_test_with_assunto.json \
  --sample-size 10000 \
  --output-jsonl data/large_sanitized/pseudolabel_pool_target_assunto_10k.jsonl \
  --summary-json data/large_sanitized/pseudolabel_pool_target_assunto_10k_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inventory_labeled_source_assuntos import read_json_or_jsonl  # noqa: E402


DEFAULT_INPUT = "data/large_sanitized/large_sanitized_no_labeled_overlap.jsonl"


def iter_jsonl(path: str | Path):
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {source} at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL in {source} at line {line_no}: expected object.")
            yield line_no, row


def assunto_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        return ""
    text = str(value).strip()
    return text


def load_target_counts(target_inputs: list[str], *, assunto_field: str) -> Counter:
    counts: Counter = Counter()
    for target_input in target_inputs:
        rows = read_json_or_jsonl(target_input)
        for row in rows:
            assunto = assunto_value(row, assunto_field)
            if assunto:
                counts[assunto] += 1
    if not counts:
        raise ValueError("No target rows with assunto were found.")
    return counts


def load_pool_by_assunto(input_jsonl: str, *, assunto_field: str) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    by_assunto: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for line_no, row in iter_jsonl(input_jsonl):
        assunto = assunto_value(row, assunto_field)
        if assunto:
            by_assunto[assunto].append((line_no, row))
    return dict(by_assunto)


def _largest_remainder_allocation(weights: dict[str, float], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("Allocation total must be non-negative.")
    if total == 0 or not weights:
        return {key: 0 for key in weights}

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Allocation weights must sum to a positive value.")

    raw = {key: total * value / weight_sum for key, value in weights.items()}
    allocation = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(allocation.values())
    order = sorted(raw, key=lambda key: (raw[key] - allocation[key], raw[key], key), reverse=True)
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def allocate_best_effort(
    *,
    target_counts: Counter,
    pool_counts: dict[str, int],
    sample_size: int,
) -> dict[str, int]:
    if sample_size < 1:
        raise ValueError("--sample-size must be >= 1")
    pool_total = sum(pool_counts.values())
    if sample_size > pool_total:
        raise ValueError(f"--sample-size ({sample_size}) is larger than the available pool ({pool_total}).")

    target_keys = set(target_counts)
    pool_keys = {key for key, count in pool_counts.items() if count > 0}
    allocatable_keys = sorted(target_keys & pool_keys)
    if not allocatable_keys:
        raise ValueError("No overlap between target assuntos and pool assuntos.")

    allocations = {key: 0 for key in allocatable_keys}
    remaining = sample_size
    active = set(allocatable_keys)

    while remaining > 0 and active:
        weights = {key: float(target_counts[key]) for key in active}
        proposal = _largest_remainder_allocation(weights, remaining)
        saturated = []
        for key in sorted(active):
            available_after_current = pool_counts[key] - allocations[key]
            proposed = proposal.get(key, 0)
            if proposed >= available_after_current:
                allocations[key] += available_after_current
                remaining -= available_after_current
                saturated.append(key)
        if not saturated:
            for key, count in proposal.items():
                allocations[key] += count
                remaining -= count
            break
        active.difference_update(saturated)

    if remaining > 0:
        spillover_keys = sorted(pool_keys - set(allocations))
        spillover_weights = {key: float(pool_counts[key]) for key in spillover_keys}
        spillover = _largest_remainder_allocation(spillover_weights, remaining)
        for key, count in spillover.items():
            allocations[key] = count
        remaining -= sum(spillover.values())

    if remaining != 0:
        raise ValueError(f"Could not allocate full sample budget; remaining={remaining}")
    return {key: count for key, count in sorted(allocations.items()) if count > 0}


def sample_by_allocation(
    pool_by_assunto: dict[str, list[tuple[int, dict[str, Any]]]],
    allocations: dict[str, int],
    *,
    seed: int,
    preserve_input_order: bool,
) -> list[tuple[int, dict[str, Any]]]:
    rng = Random(seed)
    sampled: list[tuple[int, dict[str, Any]]] = []
    for assunto in sorted(allocations):
        count = allocations[assunto]
        rows = pool_by_assunto.get(assunto, [])
        if count > len(rows):
            raise ValueError(f"Allocation for {assunto!r} exceeds pool size: {count} > {len(rows)}")
        sampled.extend(rng.sample(rows, count))
    if preserve_input_order:
        sampled.sort(key=lambda item: item[0])
    return sampled


def total_variation_distance(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    total_a = sum(a.values())
    total_b = sum(b.values())
    if total_a <= 0 or total_b <= 0:
        return 0.0
    return 0.5 * sum(abs((a.get(key, 0) / total_a) - (b.get(key, 0) / total_b)) for key in keys)


def distribution_table(
    *,
    target_counts: Counter,
    pool_counts: dict[str, int],
    allocations: dict[str, int],
) -> list[dict[str, Any]]:
    keys = sorted(set(target_counts) | set(pool_counts) | set(allocations))
    target_total = sum(target_counts.values())
    pool_total = sum(pool_counts.values())
    sample_total = sum(allocations.values())
    rows = []
    for key in keys:
        target_n = int(target_counts.get(key, 0))
        pool_n = int(pool_counts.get(key, 0))
        sample_n = int(allocations.get(key, 0))
        rows.append(
            {
                "assunto": key,
                "target_n": target_n,
                "target_share": target_n / target_total if target_total else 0.0,
                "pool_n": pool_n,
                "pool_share": pool_n / pool_total if pool_total else 0.0,
                "sample_n": sample_n,
                "sample_share": sample_n / sample_total if sample_total else 0.0,
                "sample_minus_target_share": (sample_n / sample_total if sample_total else 0.0)
                - (target_n / target_total if target_total else 0.0),
                "pool_exhausted": sample_n == pool_n and pool_n > 0,
            }
        )
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample an unlabeled JSONL pool toward a target assunto distribution."
    )
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT, help="Input unlabeled JSONL pool.")
    parser.add_argument(
        "--target-input",
        action="append",
        required=True,
        help="Enriched labeled JSON/JSONL file used to estimate target assunto distribution. Repeatable.",
    )
    parser.add_argument("--sample-size", type=int, required=True, help="Number of pool rows to sample.")
    parser.add_argument("--output-jsonl", required=True, help="Output sampled JSONL.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--assunto-field", default="assunto", help="Assunto field name.")
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Write sampled rows in sampled order instead of preserving input order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_counts = load_target_counts(args.target_input, assunto_field=args.assunto_field)
    pool_by_assunto = load_pool_by_assunto(args.input_jsonl, assunto_field=args.assunto_field)
    pool_counts = {key: len(rows) for key, rows in pool_by_assunto.items()}
    allocations = allocate_best_effort(
        target_counts=target_counts,
        pool_counts=pool_counts,
        sample_size=args.sample_size,
    )
    sampled = sample_by_allocation(
        pool_by_assunto,
        allocations,
        seed=args.seed,
        preserve_input_order=not args.shuffle_output,
    )
    sampled_rows = [row for _, row in sampled]
    write_jsonl(args.output_jsonl, sampled_rows)

    table = distribution_table(target_counts=target_counts, pool_counts=pool_counts, allocations=allocations)
    sample_counts = Counter({key: value for key, value in allocations.items()})
    summary = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "target_inputs": [str(Path(path).resolve()) for path in args.target_input],
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
        "sample_size_requested": args.sample_size,
        "sample_size_written": len(sampled_rows),
        "seed": args.seed,
        "preserve_input_order": not args.shuffle_output,
        "target_rows_with_assunto": sum(target_counts.values()),
        "pool_rows_with_assunto": sum(pool_counts.values()),
        "target_counts": dict(sorted(target_counts.items())),
        "pool_counts": dict(sorted(pool_counts.items())),
        "sample_allocations": dict(sorted(allocations.items())),
        "target_vs_pool_total_variation": total_variation_distance(dict(target_counts), pool_counts),
        "target_vs_sample_total_variation": total_variation_distance(dict(target_counts), dict(sample_counts)),
        "pool_vs_sample_total_variation": total_variation_distance(pool_counts, dict(sample_counts)),
        "distribution_table": table,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)

    print(f"Target rows with assunto: {summary['target_rows_with_assunto']}")
    print(f"Pool rows with assunto: {summary['pool_rows_with_assunto']}")
    print(f"Sample rows written: {len(sampled_rows)}")
    print(f"Target vs sample TVD: {summary['target_vs_sample_total_variation']:.6f}")
    print(f"Output JSONL: {args.output_jsonl}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
