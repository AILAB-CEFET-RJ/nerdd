#!/usr/bin/env python3
"""Extract a Rio de Janeiro logradouro lexicon from a shapefile DBF into flat files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import write_jsonl

DEFAULT_ENCODING_CANDIDATES = ("utf-8", "cp1252", "latin1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a flat Rio logradouro lexicon from the Logradouros DBF."
    )
    parser.add_argument("--input-dbf", required=True, help="Input DBF path from the shapefile bundle.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument("--output-csv", default="", help="Optional output CSV path.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON path.")
    parser.add_argument(
        "--encoding",
        default="",
        help="Optional DBF text encoding override. If omitted, tries UTF-8, then cp1252, then latin1.",
    )
    parser.add_argument(
        "--deduplicate-by-completo-bairro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate rows by canonical logradouro name plus bairro.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    text = " ".join(text.split())
    return text


def _score_decoded_text(value: str) -> int:
    score = 0
    if "\ufffd" in value:
        score -= 20
    if "Ã" in value or "�" in value:
        score -= 10
    if any(ch in value for ch in "ãõçáéíóúâêôàÃÕÇÁÉÍÓÚÂÊÔÀ"):
        score += 5
    return score


def _read_dbf_fields(handle):
    header = handle.read(32)
    if len(header) != 32:
        raise ValueError("Invalid DBF header.")
    num_records = struct.unpack("<I", header[4:8])[0]
    header_len = struct.unpack("<H", header[8:10])[0]
    record_len = struct.unpack("<H", header[10:12])[0]
    num_fields = (header_len - 33) // 32
    fields = []
    for _ in range(num_fields):
        descriptor = handle.read(32)
        name = descriptor[:11].split(b"\x00", 1)[0].decode("latin1")
        field_type = chr(descriptor[11])
        field_len = descriptor[16]
        decimal_count = descriptor[17]
        fields.append((name, field_type, field_len, decimal_count))
    terminator = handle.read(1)
    if terminator != b"\r":
        raise ValueError("Invalid DBF field terminator.")
    return num_records, record_len, fields


def _decode_bytes(raw: bytes, encoding_candidates: list[str]) -> str:
    stripped = raw.rstrip(b" \x00")
    if not stripped:
        return ""
    best_value = None
    best_score = None
    for encoding in encoding_candidates:
        try:
            value = stripped.decode(encoding)
        except UnicodeDecodeError:
            continue
        score = _score_decoded_text(value)
        if best_score is None or score > best_score:
            best_score = score
            best_value = value
    if best_value is not None:
        return best_value.strip()
    return stripped.decode("latin1", errors="ignore").strip()


def read_dbf_rows(path: Path, *, encoding_override: str = "") -> list[dict]:
    encoding_candidates = [encoding_override] if encoding_override else list(DEFAULT_ENCODING_CANDIDATES)
    rows = []
    with path.open("rb") as handle:
        num_records, record_len, fields = _read_dbf_fields(handle)
        for _ in range(num_records):
            record = handle.read(record_len)
            if not record:
                break
            if record[0:1] == b"*":
                continue
            offset = 1
            row = {}
            for name, field_type, field_len, _decimal_count in fields:
                raw = record[offset : offset + field_len]
                offset += field_len
                if field_type == "N":
                    row[name] = raw.decode("latin1", errors="ignore").strip()
                else:
                    row[name] = _decode_bytes(raw, encoding_candidates)
            rows.append(row)
    return rows


def _clean_numeric(value: str):
    stripped = str(value or "").strip()
    if not stripped or set(stripped) == {"*"}:
        return None
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    return stripped


def build_lexicon_rows(rows: list[dict], *, deduplicate_by_completo_bairro: bool) -> tuple[list[dict], dict]:
    output = []
    seen = set()
    bairro_counts = Counter()
    tipo_counts = Counter()

    for row in rows:
        completo = str(row.get("COMPLETO", "")).strip()
        bairro = str(row.get("Nome", "")).strip()
        tipo_abrev = str(row.get("TIPO_LOGRA", "")).strip()
        tipo_extenso = str(row.get("TIPO_LOG_1", "")).strip()
        nome_parcial = str(row.get("NOME_PARCI", "")).strip()

        if not completo:
            continue

        normalized = {
            "canonical_name": completo,
            "canonical_name_norm": normalize_text(completo),
            "tipo_abrev": tipo_abrev,
            "tipo_extenso": tipo_extenso,
            "nome_parcial": nome_parcial,
            "nome_parcial_norm": normalize_text(nome_parcial),
            "bairro": bairro,
            "bairro_norm": normalize_text(bairro),
            "cod_bairro": _clean_numeric(row.get("Cod_Bairro", "")),
            "cod_trecho": _clean_numeric(row.get("COD_TRECHO", "")),
            "situacao_trecho": str(row.get("SIT_TRECHO", "")).strip(),
            "hierarquia": str(row.get("HIERARQUIA", "")).strip(),
            "np_ini_par": _clean_numeric(row.get("NP_INI_PAR", "")),
            "np_fin_par": _clean_numeric(row.get("NP_FIN_PAR", "")),
            "np_ini_imp": _clean_numeric(row.get("NP_INI_IMP", "")),
            "np_fin_imp": _clean_numeric(row.get("NP_FIN_IMP", "")),
        }

        dedup_key = (
            normalized["canonical_name_norm"],
            normalized["bairro_norm"],
        ) if deduplicate_by_completo_bairro else (
            normalized["canonical_name_norm"],
            normalized["bairro_norm"],
            normalized["cod_trecho"],
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        output.append(normalized)

        if normalized["bairro"]:
            bairro_counts[normalized["bairro"]] += 1
        if normalized["tipo_extenso"]:
            tipo_counts[normalized["tipo_extenso"]] += 1

    summary = {
        "rows_output": len(output),
        "distinct_bairros": len(bairro_counts),
        "top_bairros": dict(bairro_counts.most_common(20)),
        "top_tipos": dict(tipo_counts.most_common(20)),
    }
    return output, summary


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "canonical_name",
        "canonical_name_norm",
        "tipo_abrev",
        "tipo_extenso",
        "nome_parcial",
        "nome_parcial_norm",
        "bairro",
        "bairro_norm",
        "cod_bairro",
        "cod_trecho",
        "situacao_trecho",
        "hierarquia",
        "np_ini_par",
        "np_fin_par",
        "np_ini_imp",
        "np_fin_imp",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    input_path = Path(args.input_dbf)
    rows = read_dbf_rows(input_path, encoding_override=args.encoding)
    lexicon_rows, summary = build_lexicon_rows(
        rows,
        deduplicate_by_completo_bairro=args.deduplicate_by_completo_bairro,
    )

    write_jsonl(args.output_jsonl, lexicon_rows)
    if args.output_csv:
        write_csv(Path(args.output_csv), lexicon_rows)

    summary_payload = {
        "input_dbf": str(input_path.resolve()),
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
        "output_csv": str(Path(args.output_csv).resolve()) if args.output_csv else "",
        "rows_input": len(rows),
        "deduplicate_by_completo_bairro": bool(args.deduplicate_by_completo_bairro),
        **summary,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Input rows: {len(rows)}")
    print(f"Lexicon rows: {len(lexicon_rows)}")
    print(f"Output JSONL: {args.output_jsonl}")
    if args.output_csv:
        print(f"Output CSV: {args.output_csv}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
