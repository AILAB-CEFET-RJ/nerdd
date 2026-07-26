#!/usr/bin/env python3
"""Serve the NER annotation editor with direct save support for a local dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from .build_ner_annotation_editor_global import (
        DEFAULT_LABELS,
        build_html,
        build_label_colors,
        load_corpus,
        load_initial_banlist,
        normalize_records,
    )
except ImportError:
    from build_ner_annotation_editor_global import (
        DEFAULT_LABELS,
        build_html,
        build_label_colors,
        load_corpus,
        load_initial_banlist,
        normalize_records,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the NER annotation editor and save corrections directly to the dataset."
    )
    parser.add_argument("--input", required=True, help="Dataset JSON file to edit and save.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the local editor server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local editor server.",
    )
    parser.add_argument(
        "--title",
        default="NER Annotation Editor",
        help="Page title.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label list.",
    )
    parser.add_argument(
        "--banlist-input",
        default="",
        help="Optional JSON file with initial remove rules: {label: [normalized_terms...]}",
    )
    parser.add_argument(
        "--banlist-output",
        default="annotation_banlist.json",
        help="Default filename used when exporting remove rules from the HTML editor.",
    )
    parser.add_argument(
        "--rules-output",
        default="annotation_global_rules.json",
        help="Default filename used when exporting all global cleanup rules from the HTML editor.",
    )
    return parser.parse_args()


def build_editor_html(input_path: Path, args: argparse.Namespace) -> tuple[str, list[bool], int]:
    labels = [x.strip() for x in args.labels.split(",") if x.strip()] or list(DEFAULT_LABELS)
    original_rows = load_corpus(input_path)
    original_has_editor_index = [
        isinstance(row, dict) and "_editor_row_index" in row for row in original_rows
    ]
    records = normalize_records(original_rows)
    html = build_html(
        args.title,
        records,
        labels,
        build_label_colors(labels),
        load_initial_banlist(args.banlist_input),
        args.banlist_output,
        args.rules_output,
        save_enabled=True,
        source_display_name=str(input_path),
    )
    return html, original_has_editor_index, len(original_rows)


def validate_and_clean_records(
    payload: Any,
    *,
    expected_records: int,
    original_has_editor_index: list[bool],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Request body must contain a records list.")
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} records, received {len(records)}.")

    cleaned_records = []
    for row_idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Record {row_idx} must be a JSON object.")
        cleaned = dict(row)
        if not original_has_editor_index[row_idx]:
            cleaned.pop("_editor_row_index", None)

        text = cleaned.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Record {row_idx} must contain a text string.")

        spans = cleaned.get("spans", [])
        if not isinstance(spans, list):
            raise ValueError(f"Record {row_idx} spans must be a list.")
        for span_idx, span in enumerate(spans):
            if not isinstance(span, dict):
                raise ValueError(f"Record {row_idx} span {span_idx} must be an object.")
            start = span.get("start")
            end = span.get("end")
            label = span.get("label")
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"Record {row_idx} span {span_idx} must have integer offsets.")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Record {row_idx} span {span_idx} must have a label.")
            if start < 0 or end <= start or end > len(text):
                raise ValueError(f"Record {row_idx} span {span_idx} has invalid offsets.")
        cleaned_records.append(cleaned)
    return cleaned_records


def save_dataset(path: Path, records: list[dict[str, Any]]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak-{timestamp}")
    tmp_path = path.with_name(f".{path.name}.tmp-{timestamp}")

    shutil.copy2(path, backup_path)
    tmp_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    return backup_path


def make_handler(
    *,
    html: str,
    input_path: Path,
    expected_records: int,
    original_has_editor_index: list[bool],
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NERAnnotationEditor/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404, "Not found")
                return
            encoded = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:
            if self.path != "/save":
                self.send_error(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                records = validate_and_clean_records(
                    payload,
                    expected_records=expected_records,
                    original_has_editor_index=original_has_editor_index,
                )
                backup_path = save_dataset(input_path, records)
            except Exception as exc:
                self.send_json(400, {"ok": False, "error": str(exc)})
                return
            self.send_json(
                200,
                {
                    "ok": True,
                    "records_saved": len(records),
                    "dataset_path": str(input_path),
                    "backup_path": str(backup_path),
                },
            )

    return Handler


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    html, original_has_editor_index, expected_records = build_editor_html(input_path, args)
    handler = make_handler(
        html=html,
        input_path=input_path,
        expected_records=expected_records,
        original_has_editor_index=original_has_editor_index,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[ok] Serving annotation editor for {input_path}")
    print(f"[ok] Open: {url}")
    print("[info] Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ok] Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
