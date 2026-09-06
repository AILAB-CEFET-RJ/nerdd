#!/usr/bin/env python3
"""Recover DD small-corpus metadata from the malformed app_dd.xlsx file.

The spreadsheet appears to come from a CSV that was split on unescaped commas.
This script matches current labeled NER rows back to that spreadsheet by
normalized text and extracts the metadata cells immediately after the matched
report text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl, save_jsonl
from base_model_training.paths import resolve_path

XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

EXPECTED_COLUMNS = [
    "id",
    "hash",
    "numEnvolvidos",
    "numPics",
    "assunto",
    "relato",
    "cepLocal",
    "cidadeLocal",
    "tipoLocal",
    "logradouroLocal",
    "numeroLocal",
    "complementoLocal",
    "bairroLocal",
    "regiaoLocal",
    "pontodeReferenciaLocal",
    "enderecoLocal",
    "data",
    "capturada",
    "lida",
]

METADATA_FIELDS = [
    "cepLocal",
    "cidadeLocal",
    "tipoLocal",
    "logradouroLocal",
    "numeroLocal",
    "complementoLocal",
    "bairroLocal",
    "regiaoLocal",
    "pontodeReferenciaLocal",
    "enderecoLocal",
]

GEO_METADATA_FIELDS = {
    "cidadeLocal",
    "logradouroLocal",
    "bairroLocal",
    "pontodeReferenciaLocal",
}


def repair_mojibake(value: Any) -> str:
    text = "" if value is None else str(value)
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", repair_mojibake(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def xlsx_column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref}")
    number = 0
    for char in match.group(1):
        number = number * 26 + ord(char) - ord("A") + 1
    return number - 1


def xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//m:t", XLSX_NS))

    value_node = cell.find("m:v", XLSX_NS)
    if value_node is None:
        return ""
    raw = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(raw)] if raw else ""
    return raw


def read_xlsx_rows(path: str | Path) -> list[list[str]]:
    """Read the first worksheet from an XLSX file without external dependencies."""
    rows: list[list[str]] = []
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", XLSX_NS):
                shared_strings.append("".join(node.text or "" for node in item.findall(".//m:t", XLSX_NS)))

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall(".//m:sheetData/m:row", XLSX_NS):
            values: dict[int, str] = {}
            for cell in row.findall("m:c", XLSX_NS):
                values[xlsx_column_index(cell.attrib["r"])] = xlsx_cell_value(cell, shared_strings)
            width = max(values.keys(), default=-1) + 1
            rows.append([values.get(index, "") for index in range(width)])
    return rows


def app_row_candidate_text(row: list[str]) -> str:
    return " ".join(repair_mojibake(value) for value in row[5:] if str(value).strip())


def extract_metadata_after_labeled_text(app_row: list[str], labeled_text: str) -> tuple[dict[str, str], str]:
    """Extract metadata cells after the segment that reconstructs labeled_text."""
    target = normalize_text(labeled_text)
    if not target:
        return {}, "empty_labeled_text"

    accumulated = ""
    text_end_index: int | None = None
    for index in range(5, len(app_row)):
        accumulated = (accumulated + " " + repair_mojibake(app_row[index])).strip()
        if target in normalize_text(accumulated):
            text_end_index = index
            break

    if text_end_index is None:
        return {}, "text_not_found_inside_app_row"

    metadata_start = text_end_index + 1
    metadata_end = metadata_start + len(METADATA_FIELDS)
    if metadata_end > len(app_row):
        return {}, "not_enough_cells_after_text"

    metadata = {}
    for field, value in zip(METADATA_FIELDS, app_row[metadata_start:metadata_end]):
        repaired = repair_mojibake(value).strip()
        if repaired:
            metadata[field] = repaired
    return metadata, "ok"


def metadata_signature(metadata: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(metadata.items()))


def load_app_dd_rows(path: str | Path) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    rows = read_xlsx_rows(path)
    if not rows:
        raise ValueError(f"Empty XLSX file: {path}")
    header = [repair_mojibake(value).strip() for value in rows[0]]
    data_rows = rows[1:]
    width_counts = Counter(len(row) for row in data_rows)
    profile = {
        "xlsx_path": str(path),
        "header": header,
        "header_matches_expected": header[: len(EXPECTED_COLUMNS)] == EXPECTED_COLUMNS,
        "data_rows": len(data_rows),
        "width_counts": {str(width): count for width, count in sorted(width_counts.items())},
        "well_formed_19_column_rows": width_counts.get(19, 0),
        "malformed_gt_19_column_rows": sum(count for width, count in width_counts.items() if width > 19),
    }
    return header, data_rows, profile


def build_app_indexes(app_rows: list[list[str]]) -> tuple[dict[str, list[int]], list[str]]:
    exact_relatos: dict[str, list[int]] = defaultdict(list)
    full_texts: list[str] = []
    for row_index, row in enumerate(app_rows):
        relato_key = normalize_text(row[5] if len(row) > 5 else "")
        if relato_key:
            exact_relatos[relato_key].append(row_index)
        full_texts.append(normalize_text(app_row_candidate_text(row)))
    return exact_relatos, full_texts


def find_app_matches(
    *,
    labeled_text: str,
    exact_relatos: dict[str, list[int]],
    full_texts: list[str],
) -> tuple[str, list[int]]:
    key = normalize_text(labeled_text)
    if not key:
        return "empty_labeled_text", []
    exact_matches = exact_relatos.get(key, [])
    if exact_matches:
        return "exact_relato", exact_matches
    contains_matches = [index for index, full_text in enumerate(full_texts) if key in full_text]
    if contains_matches:
        return "contains_reconstructed_row", contains_matches
    return "unmatched", []


def labeled_input_spec(value: str) -> tuple[str, Path]:
    if ":" in value:
        split, path = value.split(":", 1)
        split = split.strip()
        if not split:
            raise ValueError(f"Invalid labeled input spec without split name: {value}")
        return split, Path(path)
    path = Path(value)
    split = path.stem
    if split.startswith("dd_corpus_small_"):
        split = split.removeprefix("dd_corpus_small_")
    return split, path


def match_labeled_rows(
    *,
    app_rows: list[list[str]],
    labeled_inputs: list[tuple[str, Path]],
    labeled_text_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    exact_relatos, full_texts = build_app_indexes(app_rows)
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    stats = Counter()
    split_stats: dict[str, Counter] = defaultdict(Counter)

    for split, labeled_path in labeled_inputs:
        rows = load_jsonl(str(labeled_path))
        stats["labeled_rows"] += len(rows)
        split_stats[split]["rows"] += len(rows)

        for row_index, row in enumerate(rows):
            text = str(row.get(labeled_text_field, ""))
            match_strategy, matches = find_app_matches(
                labeled_text=text,
                exact_relatos=exact_relatos,
                full_texts=full_texts,
            )
            if not matches:
                status = "unmatched"
            elif len(matches) == 1:
                status = "matched_unique"
            else:
                status = "matched_ambiguous"

            metadata: dict[str, str] = {}
            extraction_status = ""
            selected_excel_row_1based: int | None = None
            if status == "matched_unique":
                selected_app_index = matches[0]
                selected_excel_row_1based = selected_app_index + 2
                metadata, extraction_status = extract_metadata_after_labeled_text(app_rows[selected_app_index], text)
                if extraction_status != "ok":
                    status = "matched_unique_metadata_extract_failed"
                elif not metadata:
                    status = "matched_unique_without_metadata"
                elif not any(metadata.get(field) for field in GEO_METADATA_FIELDS):
                    status = "matched_unique_without_geo_metadata"
            else:
                extraction_status = "not_attempted"

            stats[status] += 1
            split_stats[split][status] += 1
            if status.startswith("matched_unique"):
                stats["matched_unique_total"] += 1
                split_stats[split]["matched_unique_total"] += 1
            if any(metadata.get(field) for field in GEO_METADATA_FIELDS):
                stats["matched_unique_with_geo_metadata"] += 1
                split_stats[split]["matched_unique_with_geo_metadata"] += 1

            payload = {
                "split": split,
                "labeled_input": str(labeled_path),
                "row_index_0based": row_index,
                "row_index_1based": row_index + 1,
                "text": text,
                "spans": row.get("spans", []),
                "assunto": row.get("assunto", ""),
                "app_dd_match": {
                    "status": status,
                    "match_strategy": match_strategy,
                    "match_count": len(matches),
                    "app_excel_row_1based": selected_excel_row_1based,
                    "app_excel_rows_preview_1based": [index + 2 for index in matches[:10]],
                    "metadata_extraction_status": extraction_status,
                    "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                },
            }
            if metadata:
                payload["source_fields"] = metadata
                payload.update(metadata)

            output_rows.append(payload)
            audit_rows.append(
                {
                    "split": split,
                    "row_index_1based": row_index + 1,
                    "status": status,
                    "match_strategy": match_strategy,
                    "match_count": len(matches),
                    "app_excel_row_1based": selected_excel_row_1based or "",
                    "metadata_extraction_status": extraction_status,
                    "has_geo_metadata": bool(any(metadata.get(field) for field in GEO_METADATA_FIELDS)),
                    "cidadeLocal": metadata.get("cidadeLocal", ""),
                    "logradouroLocal": metadata.get("logradouroLocal", ""),
                    "bairroLocal": metadata.get("bairroLocal", ""),
                    "pontodeReferenciaLocal": metadata.get("pontodeReferenciaLocal", ""),
                    "text_preview": text[:180],
                }
            )

    summary = {
        "status_counts": dict(stats),
        "split_status_counts": {split: dict(counter) for split, counter in split_stats.items()},
    }
    return output_rows, audit_rows, summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match labeled DD NER corpora to app_dd.xlsx and recover original location metadata."
    )
    parser.add_argument("--app-xlsx", default="data/app_dd.xlsx")
    parser.add_argument(
        "--labeled-input",
        action="append",
        default=[],
        help="Labeled corpus path, optionally prefixed as split:path. Repeatable.",
    )
    parser.add_argument("--labeled-text-field", default="text")
    parser.add_argument(
        "--output-jsonl",
        default="artifacts/metadata/app_dd_labeled_metadata_matches.jsonl",
    )
    parser.add_argument(
        "--audit-csv",
        default="artifacts/metadata/app_dd_labeled_metadata_matches_audit.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="artifacts/metadata/app_dd_labeled_metadata_matches_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    app_path = resolve_path(script_dir, args.app_xlsx)
    if not app_path.exists():
        raise FileNotFoundError(f"app_dd XLSX not found: {app_path}")

    raw_labeled_inputs = args.labeled_input or [
        "train:data/dd_corpus_small_train.json",
        "test:data/dd_corpus_small_test.json",
        "calibration:data/dd_corpus_small_calibration.json",
    ]
    labeled_inputs = [
        (split, resolve_path(script_dir, path))
        for split, path in (labeled_input_spec(value) for value in raw_labeled_inputs)
    ]
    for _split, path in labeled_inputs:
        if not path.exists():
            raise FileNotFoundError(f"Labeled input not found: {path}")

    _header, app_rows, app_profile = load_app_dd_rows(app_path)
    output_rows, audit_rows, match_summary = match_labeled_rows(
        app_rows=app_rows,
        labeled_inputs=labeled_inputs,
        labeled_text_field=args.labeled_text_field,
    )

    output_path = resolve_path(script_dir, args.output_jsonl)
    audit_path = resolve_path(script_dir, args.audit_csv)
    summary_path = resolve_path(script_dir, args.summary_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(str(output_path), output_rows)
    write_csv(audit_path, audit_rows)
    write_json(
        summary_path,
        {
            "app_dd": app_profile,
            "labeled_inputs": [{"split": split, "path": str(path)} for split, path in labeled_inputs],
            **match_summary,
            "outputs": {
                "jsonl": str(output_path),
                "audit_csv": str(audit_path),
                "summary_json": str(summary_path),
            },
        },
    )

    status = match_summary["status_counts"]
    print(
        "Matched labeled rows with unique geographic metadata: "
        f"{status.get('matched_unique_with_geo_metadata', 0)}/{status.get('labeled_rows', 0)}"
    )
    print(f"Wrote: {output_path}")
    print(f"Wrote: {audit_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
