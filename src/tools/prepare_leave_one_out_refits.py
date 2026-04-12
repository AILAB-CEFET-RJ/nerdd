#!/usr/bin/env python3
"""Prepare and summarize leave-one-out refit experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl
from base_model_training.paths import resolve_path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(summary: dict, dotted_key: str) -> float | None:
    current = summary
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return float(current) if current is not None else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and summarize leave-one-out refit experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Generate leave-one-out pseudolabel JSONLs.")
    build.add_argument("--input-jsonl", required=True, help="Base pseudolabel JSONL.")
    build.add_argument("--output-dir", required=True, help="Output directory for leave-one-out JSONLs.")
    build.add_argument("--summary-json", default="", help="Optional summary JSON.")

    summarize = subparsers.add_parser("summarize", help="Summarize leave-one-out quick_summary artifacts.")
    summarize.add_argument("--baseline-summary", required=True, help="Baseline quick_summary.json path.")
    summarize.add_argument("--full-summary", required=True, help="Full-set quick_summary.json path.")
    summarize.add_argument("--experiment-dir", required=True, help="Directory containing leave-one-out run subdirectories.")
    summarize.add_argument("--summary-json", default="", help="Optional summary JSON.")
    return parser


def cmd_build(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = resolve_path(script_dir, args.input_jsonl)
    output_dir = resolve_path(script_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(load_jsonl(str(input_path)))
    source_ids = [str(row.get("source_id", "")).strip() for row in rows]
    source_ids = [source_id for source_id in source_ids if source_id]

    outputs = []
    for drop_id in source_ids:
        out_path = output_dir / f"without__{drop_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if str(row.get("source_id", "")).strip() == drop_id:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs.append({"dropped_source_id": drop_id, "output_jsonl": str(out_path.resolve())})

    summary = {
        "input_jsonl": str(input_path.resolve()),
        "rows_total": len(rows),
        "leave_one_out_count": len(outputs),
        "experiments": outputs,
    }
    if args.summary_json:
        summary_path = resolve_path(script_dir, args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_summarize(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    baseline_summary = _read_json(resolve_path(script_dir, args.baseline_summary))
    full_summary = _read_json(resolve_path(script_dir, args.full_summary))
    experiment_dir = resolve_path(script_dir, args.experiment_dir)

    baseline_micro = _metric(baseline_summary, "test_metrics.micro.f1")
    baseline_macro = _metric(baseline_summary, "test_metrics.macro_f1")
    full_micro = _metric(full_summary, "test_metrics.micro.f1")
    full_macro = _metric(full_summary, "test_metrics.macro_f1")

    experiments = []
    for summary_path in sorted(experiment_dir.glob("*/quick_summary.json")):
        summary = _read_json(summary_path)
        run_name = summary_path.parent.name
        dropped_source_id = run_name.replace("without__", "", 1) if run_name.startswith("without__") else run_name
        micro = _metric(summary, "test_metrics.micro.f1")
        macro = _metric(summary, "test_metrics.macro_f1")
        experiments.append(
            {
                "run_name": run_name,
                "dropped_source_id": dropped_source_id,
                "micro_f1": micro,
                "macro_f1": macro,
                "delta_vs_baseline_micro": micro - baseline_micro if micro is not None and baseline_micro is not None else None,
                "delta_vs_full_micro": micro - full_micro if micro is not None and full_micro is not None else None,
                "delta_vs_baseline_macro": macro - baseline_macro if macro is not None and baseline_macro is not None else None,
                "delta_vs_full_macro": macro - full_macro if macro is not None and full_macro is not None else None,
                "summary_json": str(summary_path.resolve()),
            }
        )

    experiments.sort(
        key=lambda row: (
            row["delta_vs_full_micro"] is None,
            -(row["delta_vs_full_micro"] or float("-inf")),
            -(row["delta_vs_full_macro"] or float("-inf")),
        )
    )

    summary = {
        "baseline_summary": args.baseline_summary,
        "full_summary": args.full_summary,
        "baseline_micro_f1": baseline_micro,
        "baseline_macro_f1": baseline_macro,
        "full_micro_f1": full_micro,
        "full_macro_f1": full_macro,
        "experiments": experiments,
    }
    if args.summary_json:
        summary_path = resolve_path(script_dir, args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
        return
    if args.command == "summarize":
        cmd_summarize(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
