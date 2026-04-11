#!/usr/bin/env python3
"""Run remaining chunk probe jobs and summarize context-boost audit status."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run remaining 50k chunk pseudolabelling probes for a fixed threshold."
    )
    parser.add_argument(
        "--chunks-dir",
        default="../data/dd_corpus_large_chunks_50k",
        help="Directory containing 50k chunk JSONL files.",
    )
    parser.add_argument(
        "--chunk-pattern",
        default="dd_corpus_large_chunk_*.jsonl",
        help="Glob pattern for chunk files inside --chunks-dir.",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=3,
        help="First chunk index to run.",
    )
    parser.add_argument(
        "--end-chunk",
        type=int,
        default=8,
        help="Last chunk index to run.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.37,
        help="Split threshold for pseudolabel selection.",
    )
    parser.add_argument(
        "--version",
        default="v13",
        help="Run suffix version label.",
    )
    parser.add_argument(
        "--run-root",
        default="./artifacts/pseudolabelling",
        help="Root output directory for run artifacts.",
    )
    parser.add_argument(
        "--audit-prefix",
        default="context_boost_audit",
        help="Prefix for audit directories under --run-root.",
    )
    parser.add_argument(
        "--summary-csv",
        default="./artifacts/pseudolabelling/chunk_probe_status_t037.csv",
        help="CSV file to write status summary.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_threshold_token(value: float):
    return f"t{int(round(value * 100)):03d}"


def build_run_name(chunk_idx: int, threshold_token: str, version: str):
    return f"multi_with_negatives_chunk{chunk_idx:02d}_50k_{threshold_token}_cuda_{version}"


def build_audit_name(version: str):
    return f"context_boost_audit_{version}"


def build_cycle_command(args, chunk_path: Path, run_dir: Path):
    return [
        args.python_bin,
        "-m",
        "pseudolabelling.run_iterative_cycle",
        "--run-dir",
        str(run_dir),
        "--model-path",
        "./artifacts/base_model_training/experiments/multi_lr_wd_grid_bs16_with_negatives/best_overall_gliner_model",
        "--prediction-calibrator-path",
        "./artifacts/calibration/multi_with_negatives/calibrator.json",
        "--input-jsonl",
        str(chunk_path),
        "--labels",
        "Person,Location,Organization",
        "--text-fields",
        "assunto,relato,bairroLocal,logradouroLocal,cidadeLocal,pontodeReferenciaLocal",
        "--prediction-batch-size",
        "16",
        "--prediction-max-tokens",
        "512",
        "--prediction-model-max-length",
        "384",
        "--prediction-map-location",
        "cuda",
        "--prediction-threshold",
        "0.0",
        "--record-score-field",
        "score_context_boosted",
        "--split-threshold",
        str(args.threshold),
        "--refit-mode",
        "supervised_plus_pseudolabels",
        "--refit-base-model",
        "./artifacts/base_model_training/experiments/multi_lr_wd_grid_bs16_with_negatives/best_overall_gliner_model",
        "--refit-supervised-train-path",
        "../data/dd_corpus_small_train.jsonl",
        "--refit-epochs",
        "1",
        "--refit-patience",
        "1",
        "--refit-batch-size",
        "16",
        "--refit-max-length",
        "512",
        "--refit-overlap",
        "128",
        "--refit-lr",
        "1e-5",
        "--refit-weight-decay",
        "0.01",
        "--evaluate-refit",
        "--eval-gt-jsonl",
        "../data/dd_corpus_small_test.json",
        "--eval-batch-size",
        "8",
        "--eval-max-tokens",
        "512",
        "--eval-model-max-length",
        "384",
        "--eval-map-location",
        "cuda",
        "--log-level",
        "INFO",
    ]


def build_audit_command(args, run_dir: Path, audit_dir: Path):
    return [
        args.python_bin,
        "tools/summarize_context_boost_audit.py",
        "--details-jsonl",
        str(run_dir / "03_context_boost_details.jsonl"),
        "--summary-json",
        str(audit_dir / "summary.json"),
        "--rows-csv",
        str(audit_dir / "boosted_entities.csv"),
        "--top-n",
        "30",
    ]


def summarize_run(chunk_idx: int, run_dir: Path, audit_dir: Path):
    row = {
        "chunk": chunk_idx,
        "run_dir": str(run_dir),
        "status": "missing",
        "kept_count": "",
        "micro_delta": "",
        "macro_delta": "",
        "boosted_records": "",
        "boosted_entities_total": "",
    }

    comparison_path = run_dir / "09_base_vs_refit_comparison.json"
    split_path = run_dir / "05_split" / "summary.json"
    audit_path = audit_dir / "summary.json"

    if comparison_path.exists() and split_path.exists():
        comparison = load_json(comparison_path)
        split_summary = load_json(split_path)
        row["status"] = "completed"
        row["kept_count"] = split_summary.get("summary", {}).get("kept_count", "")
        row["micro_delta"] = comparison.get("micro_f1", {}).get("delta", "")
        row["macro_delta"] = comparison.get("macro_f1", {}).get("delta", "")
    elif run_dir.exists():
        row["status"] = "partial"

    if audit_path.exists():
        audit = load_json(audit_path)
        row["boosted_records"] = audit.get("records_with_boosted_entities", "")
        row["boosted_entities_total"] = audit.get("entity_boosts_total", "")

    return row


def write_summary_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chunk",
        "run_dir",
        "status",
        "kept_count",
        "micro_delta",
        "macro_delta",
        "boosted_records",
        "boosted_entities_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    chunks_dir = Path(args.chunks_dir)
    run_root = Path(args.run_root)
    threshold_token = format_threshold_token(args.threshold)
    audit_name = build_audit_name(args.version)
    rows = []

    chunk_files = {path.name: path for path in sorted(chunks_dir.glob(args.chunk_pattern))}

    for chunk_idx in range(args.start_chunk, args.end_chunk + 1):
        chunk_name = f"dd_corpus_large_chunk_{chunk_idx:02d}.jsonl"
        chunk_path = chunk_files.get(chunk_name)
        run_name = build_run_name(chunk_idx, threshold_token, args.version)
        run_dir = run_root / run_name
        audit_dir = run_root / f"{audit_name}_chunk{chunk_idx:02d}"

        if chunk_path is None:
            rows.append(
                {
                    "chunk": chunk_idx,
                    "run_dir": str(run_dir),
                    "status": "missing_chunk",
                    "kept_count": "",
                    "micro_delta": "",
                    "macro_delta": "",
                    "boosted_records": "",
                    "boosted_entities_total": "",
                }
            )
            continue

        comparison_path = run_dir / "09_base_vs_refit_comparison.json"
        if comparison_path.exists():
            print(f"[skip] chunk {chunk_idx:02d}: comparison already exists at {comparison_path}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run] chunk {chunk_idx:02d}: {chunk_path}")
            subprocess.run(build_cycle_command(args, chunk_path, run_dir), check=True)

        details_path = run_dir / "03_context_boost_details.jsonl"
        audit_summary_path = audit_dir / "summary.json"
        if details_path.exists() and not audit_summary_path.exists():
            audit_dir.mkdir(parents=True, exist_ok=True)
            print(f"[audit] chunk {chunk_idx:02d}: {audit_dir}")
            subprocess.run(build_audit_command(args, run_dir, audit_dir), check=True)

        rows.append(summarize_run(chunk_idx, run_dir, audit_dir))
        write_summary_csv(Path(args.summary_csv), rows)

    write_summary_csv(Path(args.summary_csv), rows)
    print(f"Summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
