#!/usr/bin/env python3
"""Manage a chunked Codex-vs-GPT adjudication benchmark workflow.

This utility supports a semiautomated benchmark where:

1. a fixed adjudication benchmark JSONL is initialized from the same input used
   by `run_llm_adjudication.py`
2. the next pending chunk is exported for manual/Codex adjudication
3. structured responses are validated and ingested incrementally
4. a final consolidated output JSONL is produced for comparison against GPT-5

It is intentionally stateful so progress can be resumed without redoing chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl
from tools.run_llm_adjudication import validate_adjudication

LOGGER = logging.getLogger(__name__)


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk_id(index_1based: int) -> str:
    return f"chunk_{index_1based:03d}"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"State file must contain a JSON object: {path}")
    return payload


def _rows_by_source_id(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            raise ValueError("Each benchmark row must contain a non-empty source_id.")
        if source_id in indexed:
            raise ValueError(f"Duplicate source_id in benchmark input: {source_id}")
        indexed[source_id] = row
    return indexed


def _init_state(args):
    rows = read_json_or_jsonl(args.input)
    rows_by_id = _rows_by_source_id(rows)
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    benchmark_dir = Path(args.benchmark_dir)
    chunks_dir = benchmark_dir / "chunks"
    responses_dir = benchmark_dir / "responses"
    state_path = benchmark_dir / "state.json"
    benchmark_input_path = benchmark_dir / "benchmark_input.jsonl"

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(benchmark_input_path, rows)

    chunks = []
    all_source_ids = list(rows_by_id.keys())
    for start in range(0, len(all_source_ids), chunk_size):
        source_ids = all_source_ids[start : start + chunk_size]
        chunk_index = len(chunks) + 1
        chunk_name = _chunk_id(chunk_index)
        chunk_path = chunks_dir / f"{chunk_name}.jsonl"
        chunk_rows = [rows_by_id[source_id] for source_id in source_ids]
        write_jsonl(chunk_path, chunk_rows)
        chunks.append(
            {
                "chunk_id": chunk_name,
                "chunk_index_1based": chunk_index,
                "path": str(chunk_path.resolve()),
                "source_ids": source_ids,
                "status": "pending",
                "response_path": "",
            }
        )

    state = {
        "benchmark_name": args.benchmark_name,
        "input": str(Path(args.input).resolve()),
        "benchmark_input_jsonl": str(benchmark_input_path.resolve()),
        "chunk_size": chunk_size,
        "records_total": len(rows),
        "chunks_total": len(chunks),
        "chunks_completed": 0,
        "responses_ingested": 0,
        "output_jsonl": str((benchmark_dir / "codex_output.jsonl").resolve()),
        "summary_json": str((benchmark_dir / "summary.json").resolve()),
        "chunks": chunks,
    }
    _write_json(state_path, state)
    LOGGER.info("Initialized Codex adjudication benchmark: %s", benchmark_dir)
    LOGGER.info("State file: %s", state_path)


def _print_status(args):
    state = _load_state(Path(args.state_json))
    counts = Counter(chunk["status"] for chunk in state.get("chunks", []))
    payload = {
        "benchmark_name": state.get("benchmark_name", ""),
        "records_total": state.get("records_total", 0),
        "chunks_total": state.get("chunks_total", 0),
        "chunks_by_status": dict(counts),
        "chunks_completed": sum(1 for chunk in state.get("chunks", []) if chunk.get("status") == "completed"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _export_next(args):
    state_path = Path(args.state_json)
    state = _load_state(state_path)
    pending = [chunk for chunk in state.get("chunks", []) if chunk.get("status") == "pending"]
    if not pending:
        raise RuntimeError("No pending chunks remain.")

    chunk = pending[0]
    chunk["status"] = "exported"
    _write_json(state_path, state)
    payload = {
        "chunk_id": chunk["chunk_id"],
        "path": chunk["path"],
        "source_ids": chunk["source_ids"],
        "count": len(chunk["source_ids"]),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_responses(path: Path) -> list[dict]:
    rows = read_json_or_jsonl(str(path))
    if not isinstance(rows, list):
        raise ValueError("Responses must decode to a list of JSON objects.")
    return rows


def _ingest_chunk(args):
    state_path = Path(args.state_json)
    state = _load_state(state_path)
    benchmark_rows = read_json_or_jsonl(state["benchmark_input_jsonl"])
    source_rows = _rows_by_source_id(benchmark_rows)
    response_path = Path(args.response_jsonl)
    responses = _load_responses(response_path)

    chunk_id = str(args.chunk_id).strip()
    chunk = next((item for item in state.get("chunks", []) if item.get("chunk_id") == chunk_id), None)
    if chunk is None:
        raise ValueError(f"Unknown chunk_id: {chunk_id}")
    if chunk.get("status") == "completed" and not args.force:
        raise RuntimeError(f"Chunk already completed: {chunk_id}. Use --force to overwrite.")

    expected_ids = list(chunk["source_ids"])
    responses_by_id = {}
    for row in responses:
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            raise ValueError("Each response row must contain source_id.")
        if source_id in responses_by_id:
            raise ValueError(f"Duplicate source_id in response file: {source_id}")
        responses_by_id[source_id] = row

    missing = [source_id for source_id in expected_ids if source_id not in responses_by_id]
    extra = [source_id for source_id in responses_by_id if source_id not in expected_ids]
    if missing:
        raise ValueError(f"Response file missing source_ids for {chunk_id}: {missing}")
    if extra:
        raise ValueError(f"Response file has unexpected source_ids for {chunk_id}: {extra}")

    validated_rows = []
    decision_counts = Counter()
    for source_id in expected_ids:
        source_row = source_rows[source_id]
        response_row = responses_by_id[source_id]
        adjudication = response_row.get("adjudication")
        if not isinstance(adjudication, dict):
            raise ValueError(f"Response row for {source_id} must contain an adjudication object.")
        validated = validate_adjudication(adjudication, source_row)
        decision_counts[validated["decision"]] += 1
        validated_rows.append(
            {
                "source_id": source_id,
                "model": response_row.get("model", "codex"),
                "temperature": response_row.get("temperature", 0.0),
                "adjudication": validated,
                "_source": source_row,
            }
        )

    response_store = Path(state_path.parent) / "responses" / f"{chunk_id}.jsonl"
    write_jsonl(str(response_store), validated_rows)

    chunk["status"] = "completed"
    chunk["response_path"] = str(response_store.resolve())
    chunk["decision_counts"] = dict(decision_counts)
    state["chunks_completed"] = sum(1 for item in state.get("chunks", []) if item.get("status") == "completed")
    state["responses_ingested"] = sum(len(item.get("source_ids", [])) for item in state.get("chunks", []) if item.get("status") == "completed")
    _write_json(state_path, state)
    LOGGER.info("Ingested validated Codex responses for %s", chunk_id)


def _build_output(args):
    state = _load_state(Path(args.state_json))
    output_jsonl = Path(args.output_jsonl or state["output_jsonl"])
    summary_json = Path(args.summary_json or state["summary_json"])

    completed_chunks = [chunk for chunk in state.get("chunks", []) if chunk.get("status") == "completed"]
    if not completed_chunks:
        raise RuntimeError("No completed chunks available to consolidate.")

    all_rows = []
    decision_counts = Counter()
    for chunk in sorted(completed_chunks, key=lambda item: item["chunk_index_1based"]):
        response_path = Path(chunk["response_path"])
        rows = read_json_or_jsonl(str(response_path))
        all_rows.extend(rows)
        for row in rows:
            adjudication = row.get("adjudication", {})
            decision_counts[str(adjudication.get("decision", ""))] += 1

    write_jsonl(str(output_jsonl), all_rows)
    summary = {
        "benchmark_name": state.get("benchmark_name", ""),
        "state_json": str(Path(args.state_json).resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "records_total": len(all_rows),
        "chunks_completed": len(completed_chunks),
        "decision_counts": dict(decision_counts),
    }
    _write_json(summary_json, summary)
    LOGGER.info("Built consolidated Codex benchmark output: %s", output_jsonl)


def parse_args():
    parser = argparse.ArgumentParser(description="Manage a chunked Codex adjudication benchmark workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize benchmark chunks and state.")
    init_parser.add_argument("--input", required=True, help="Benchmark input JSONL.")
    init_parser.add_argument("--benchmark-dir", required=True, help="Directory to store state, chunks, and responses.")
    init_parser.add_argument("--benchmark-name", default="codex_adjudication_benchmark")
    init_parser.add_argument("--chunk-size", type=int, default=10)

    status_parser = subparsers.add_parser("status", help="Print benchmark progress summary.")
    status_parser.add_argument("--state-json", required=True, help="Benchmark state JSON path.")

    next_parser = subparsers.add_parser("next", help="Mark and print the next pending chunk.")
    next_parser.add_argument("--state-json", required=True, help="Benchmark state JSON path.")

    ingest_parser = subparsers.add_parser("ingest", help="Validate and ingest responses for a completed chunk.")
    ingest_parser.add_argument("--state-json", required=True, help="Benchmark state JSON path.")
    ingest_parser.add_argument("--chunk-id", required=True, help="Chunk identifier, e.g. chunk_001.")
    ingest_parser.add_argument("--response-jsonl", required=True, help="Response JSONL with Codex adjudications.")
    ingest_parser.add_argument("--force", action="store_true", help="Overwrite a chunk already marked completed.")

    build_parser = subparsers.add_parser("build-output", help="Build consolidated benchmark output JSONL.")
    build_parser.add_argument("--state-json", required=True, help="Benchmark state JSON path.")
    build_parser.add_argument("--output-jsonl", default="", help="Optional override for consolidated output JSONL.")
    build_parser.add_argument("--summary-json", default="", help="Optional override for consolidated summary JSON.")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.command == "init":
        _init_state(args)
    elif args.command == "status":
        _print_status(args)
    elif args.command == "next":
        _export_next(args)
    elif args.command == "ingest":
        _ingest_chunk(args)
    elif args.command == "build-output":
        _build_output(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
