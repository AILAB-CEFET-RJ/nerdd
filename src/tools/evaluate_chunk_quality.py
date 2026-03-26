#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl


RUN_NAME_RE = re.compile(
    r"multi_with_negatives_chunk(?P<chunk>\d+)_50k_t(?P<threshold>\d+)_cuda_(?P<version>v\d+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate chunk quality from pseudolabelling cycle artifacts."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Specific run directory to analyze. Can be repeated.",
    )
    parser.add_argument(
        "--run-glob",
        default="",
        help="Optional glob for run directories, e.g. './artifacts/pseudolabelling/multi_with_negatives_chunk*_50k_t037_cuda_v*'.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--strong-score-threshold",
        type=float,
        default=0.8,
        help="Entity score threshold considered strong.",
    )
    parser.add_argument(
        "--weak-score-threshold",
        type=float,
        default=0.5,
        help="Entity score threshold considered weak.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_audit_dir(run_dir: Path):
    match = RUN_NAME_RE.fullmatch(run_dir.name)
    if not match:
        return None
    chunk = match.group("chunk")
    version = match.group("version")
    return run_dir.parent / f"context_boost_audit_{version}_chunk{chunk}"


def get_text_key(row):
    for key in ("relato", "text", "texto"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_entity_score(entity):
    for key in ("score_context_boosted", "score_calibrated", "score"):
        score = safe_float(entity.get(key))
        if score is not None:
            return score
    return None


def summarize_kept_rows(rows, strong_score_threshold, weak_score_threshold):
    text_counter = Counter()
    label_counter = Counter()
    entity_count_values = []
    strong_entity_count_values = []
    weak_entity_count_values = []
    max_entity_score_values = []
    record_score_values = []
    entity_score_values = []

    for row in rows:
        text = get_text_key(row)
        if text:
            text_counter[text] += 1

        entities = row.get("entities") or []
        entity_count_values.append(len(entities))

        strong_count = 0
        weak_count = 0
        max_entity_score = None

        for entity in entities:
            label_counter[str(entity.get("label", ""))] += 1
            score = get_entity_score(entity)
            if score is None:
                continue
            entity_score_values.append(score)
            if score >= strong_score_threshold:
                strong_count += 1
            if score < weak_score_threshold:
                weak_count += 1
            if max_entity_score is None or score > max_entity_score:
                max_entity_score = score

        strong_entity_count_values.append(strong_count)
        weak_entity_count_values.append(weak_count)
        if max_entity_score is not None:
            max_entity_score_values.append(max_entity_score)

        split_trace = row.get("_split") or {}
        record_score = safe_float(split_trace.get("score_value"))
        if record_score is not None:
            record_score_values.append(record_score)

    unique_texts = len(text_counter)
    duplicate_rows = sum(count - 1 for count in text_counter.values() if count > 1)

    return {
        "kept_rows": len(rows),
        "unique_texts": unique_texts,
        "duplicate_text_rows": duplicate_rows,
        "duplicate_text_rate": (duplicate_rows / len(rows)) if rows else 0.0,
        "avg_entities_per_kept": mean(entity_count_values),
        "avg_strong_entities_per_kept": mean(strong_entity_count_values),
        "avg_weak_entities_per_kept": mean(weak_entity_count_values),
        "avg_max_entity_score": mean(max_entity_score_values),
        "avg_entity_score": mean(entity_score_values),
        "avg_record_score_in_kept": mean(record_score_values),
        "label_counts": dict(label_counter),
    }


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def build_flags(row):
    flags = []
    kept_count = row.get("kept_count")
    duplicate_rate = row.get("duplicate_text_rate")
    weak_entities = row.get("avg_weak_entities_per_kept")
    boosted_records = row.get("boosted_records")
    micro_delta = row.get("micro_delta")

    if kept_count is not None and kept_count >= 100:
        flags.append("high_kept_count")
    if duplicate_rate is not None and duplicate_rate >= 0.1:
        flags.append("duplicate_texts")
    if weak_entities is not None and weak_entities >= 1.0:
        flags.append("many_weak_entities")
    if boosted_records == 0:
        flags.append("no_context_boost")
    if micro_delta is not None and micro_delta <= -0.01:
        flags.append("strong_negative_delta")
    if micro_delta is not None and micro_delta >= 0.01:
        flags.append("strong_positive_delta")

    return ",".join(flags)


def summarize_run(run_dir: Path, strong_score_threshold, weak_score_threshold):
    comparison_path = run_dir / "09_base_vs_refit_comparison.json"
    split_path = run_dir / "05_split" / "summary.json"
    kept_path = run_dir / "05_split" / "kept.jsonl"
    audit_dir = infer_audit_dir(run_dir)
    audit_path = audit_dir / "summary.json" if audit_dir else None

    row = {
        "run_dir": str(run_dir),
        "chunk": "",
        "threshold": "",
        "version": "",
        "status": "missing",
        "kept_count": None,
        "kept_rate": None,
        "kept_score_mean": None,
        "entity_gate_rejections": None,
        "micro_delta": None,
        "macro_delta": None,
        "location_f1_delta": None,
        "organization_f1_delta": None,
        "person_f1_delta": None,
        "boosted_records": None,
        "boosted_entities_total": None,
        "records_with_context_match": None,
        "avg_entities_per_kept": None,
        "avg_strong_entities_per_kept": None,
        "avg_weak_entities_per_kept": None,
        "avg_max_entity_score": None,
        "avg_entity_score": None,
        "avg_record_score_in_kept": None,
        "unique_texts": None,
        "duplicate_text_rows": None,
        "duplicate_text_rate": None,
        "top_label": "",
        "top_label_count": None,
        "flags": "",
    }

    match = RUN_NAME_RE.fullmatch(run_dir.name)
    if match:
        row["chunk"] = int(match.group("chunk"))
        row["threshold"] = int(match.group("threshold")) / 100.0
        row["version"] = match.group("version")

    if comparison_path.exists() and split_path.exists():
        comparison = load_json(comparison_path)
        split_summary = load_json(split_path)
        summary = split_summary.get("summary", {})
        row["status"] = "completed"
        row["kept_count"] = summary.get("kept_count")
        total = summary.get("records_total") or 0
        row["kept_rate"] = (row["kept_count"] / total) if total and row["kept_count"] is not None else None
        row["kept_score_mean"] = summary.get("kept_score_mean")
        row["entity_gate_rejections"] = summary.get("entity_gate_rejections")
        row["micro_delta"] = comparison.get("micro_f1", {}).get("delta")
        row["macro_delta"] = comparison.get("macro_f1", {}).get("delta")
        row["location_f1_delta"] = comparison.get("per_label", {}).get("Location", {}).get("f1", {}).get("delta")
        row["organization_f1_delta"] = comparison.get("per_label", {}).get("Organization", {}).get("f1", {}).get("delta")
        row["person_f1_delta"] = comparison.get("per_label", {}).get("Person", {}).get("f1", {}).get("delta")
    elif run_dir.exists():
        row["status"] = "partial"

    if audit_path and audit_path.exists():
        audit = load_json(audit_path)
        row["boosted_records"] = audit.get("records_with_boosted_entities")
        row["boosted_entities_total"] = audit.get("entity_boosts_total")
        row["records_with_context_match"] = audit.get("records_with_context_match")

    if kept_path.exists():
        kept_rows = load_jsonl(str(kept_path))
        kept_summary = summarize_kept_rows(
            kept_rows,
            strong_score_threshold=strong_score_threshold,
            weak_score_threshold=weak_score_threshold,
        )
        row.update({key: kept_summary.get(key) for key in kept_summary if key != "label_counts"})
        label_counts = kept_summary.get("label_counts", {})
        if label_counts:
            top_label, top_count = max(label_counts.items(), key=lambda item: item[1])
            row["top_label"] = top_label
            row["top_label_count"] = top_count

    row["flags"] = build_flags(row)
    return row


def collect_run_dirs(run_dirs, run_glob):
    paths = []
    for item in run_dirs:
        paths.append(Path(item))
    if run_glob:
        paths.extend(sorted(Path().glob(run_glob)))
    deduped = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def write_csv(path: Path, rows):
    fieldnames = [
        "run_dir",
        "chunk",
        "threshold",
        "version",
        "status",
        "kept_count",
        "kept_rate",
        "kept_score_mean",
        "entity_gate_rejections",
        "micro_delta",
        "macro_delta",
        "location_f1_delta",
        "organization_f1_delta",
        "person_f1_delta",
        "boosted_records",
        "boosted_entities_total",
        "records_with_context_match",
        "avg_entities_per_kept",
        "avg_strong_entities_per_kept",
        "avg_weak_entities_per_kept",
        "avg_max_entity_score",
        "avg_entity_score",
        "avg_record_score_in_kept",
        "unique_texts",
        "duplicate_text_rows",
        "duplicate_text_rate",
        "top_label",
        "top_label_count",
        "flags",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    run_dirs = collect_run_dirs(args.run_dir, args.run_glob)
    if not run_dirs:
        raise SystemExit("No run directories provided. Use --run-dir and/or --run-glob.")

    rows = [
        summarize_run(
            run_dir=run_dir,
            strong_score_threshold=args.strong_score_threshold,
            weak_score_threshold=args.weak_score_threshold,
        )
        for run_dir in run_dirs
    ]
    rows.sort(key=lambda item: (item.get("chunk") or 0, str(item.get("run_dir"))))

    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
        print(f"Saved CSV: {args.output_csv}")

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON: {args.output_json}")

    for row in rows:
        chunk = row.get("chunk") or "?"
        print(
            f"chunk={chunk} kept={row.get('kept_count')} micro_delta={row.get('micro_delta')} "
            f"boosted_records={row.get('boosted_records')} dup_rate={row.get('duplicate_text_rate')} "
            f"flags={row.get('flags')}"
        )


if __name__ == "__main__":
    main()
