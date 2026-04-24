#!/usr/bin/env python3
"""Extract a flat logradouro lexicon from IBGE Base de Faces JSON ZIP files."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import write_jsonl

DEFAULT_TARGET_MUNICIPIOS = {
    "3300209": "Araruama",
    "3300407": "Barra Mansa",
    "3300456": "Belford Roxo",
    "3300704": "Cabo Frio",
    "3301702": "Duque de Caxias",
    "3301850": "Guapimirim",
    "3301900": "Itaboraí",
    "3302007": "Itaguaí",
    "3302270": "Japeri",
    "3302502": "Magé",
    "3302601": "Mangaratiba",
    "3302700": "Maricá",
    "3302858": "Mesquita",
    "3303203": "Nilópolis",
    "3303302": "Niterói",
    "3303500": "Nova Iguaçu",
    "3303609": "Paracambi",
    "3304144": "Queimados",
    "3304557": "Rio de Janeiro",
    "3304904": "São Gonçalo",
    "3305109": "São João de Meriti",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract logradouro names from an IBGE Base de Faces JSON ZIP."
    )
    parser.add_argument("--input-zip", required=True, help="IBGE *_faces_de_logradouros_2022_json.zip path.")
    parser.add_argument("--output-jsonl", required=True, help="Output lexicon JSONL path.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output path.")
    parser.add_argument(
        "--municipio-codes",
        default=",".join(DEFAULT_TARGET_MUNICIPIOS),
        help="Comma-separated IBGE municipality codes to keep. Use 'all' for every file in the ZIP.",
    )
    parser.add_argument(
        "--min-normalized-length",
        type=int,
        default=5,
        help="Minimum normalized canonical name length.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    return " ".join(text.split())


def parse_municipio_codes(raw_value: str) -> set[str] | None:
    if str(raw_value).strip().lower() == "all":
        return None
    return {piece.strip() for piece in str(raw_value).split(",") if piece.strip()}


def municipio_code_from_name(path_name: str) -> str:
    match = re.search(r"/(\d{7})_faces_de_logradouros_", path_name)
    return match.group(1) if match else ""


def canonical_name(properties: dict) -> str:
    parts = [
        str(properties.get("NM_TIP_LOG") or "").strip(),
        str(properties.get("NM_TIT_LOG") or "").strip(),
        str(properties.get("NM_LOG") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def numeric_or_none(value):
    raw = str(value or "").strip()
    if not raw:
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def extract_rows(
    zip_path: Path,
    *,
    municipio_codes: set[str] | None,
    min_normalized_length: int,
) -> tuple[list[dict], dict]:
    rows = []
    seen = set()
    counters = Counter()
    tipo_counts = Counter()
    municipio_counts = Counter()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".json"):
                continue
            municipio_code = municipio_code_from_name(member)
            if municipio_codes is not None and municipio_code not in municipio_codes:
                continue

            municipio_name = DEFAULT_TARGET_MUNICIPIOS.get(municipio_code, "")
            data = json.load(archive.open(member))
            features = data.get("features", []) if isinstance(data, dict) else []
            counters["features_seen"] += len(features)

            for feature in features:
                properties = feature.get("properties", {}) or {}
                name = canonical_name(properties)
                name_norm = normalize_text(name)
                if len(name_norm) < min_normalized_length:
                    counters["dropped_short_or_missing_name"] += 1
                    continue

                key = (municipio_code, name_norm)
                if key in seen:
                    counters["deduplicated_faces"] += 1
                    continue
                seen.add(key)

                row = {
                    "municipio_code": municipio_code,
                    "municipio": municipio_name,
                    "municipio_norm": normalize_text(municipio_name),
                    "tipo": str(properties.get("NM_TIP_LOG") or "").strip(),
                    "titulo": str(properties.get("NM_TIT_LOG") or "").strip(),
                    "nome": str(properties.get("NM_LOG") or "").strip(),
                    "canonical_name": name,
                    "canonical_name_norm": name_norm,
                    "tot_res": numeric_or_none(properties.get("TOT_RES")),
                    "tot_geral": numeric_or_none(properties.get("TOT_GERAL")),
                    "source": "IBGE Base de Faces de Logradouros 2022",
                }
                rows.append(row)
                municipio_counts[municipio_name or municipio_code] += 1
                if row["tipo"]:
                    tipo_counts[row["tipo"]] += 1

    summary = {
        "rows_output": len(rows),
        "counters": dict(counters),
        "municipio_counts": dict(municipio_counts.most_common()),
        "tipo_counts": dict(tipo_counts.most_common(30)),
    }
    return rows, summary


def main():
    args = parse_args()
    input_zip = Path(args.input_zip)
    municipio_codes = parse_municipio_codes(args.municipio_codes)
    rows, summary = extract_rows(
        input_zip,
        municipio_codes=municipio_codes,
        min_normalized_length=args.min_normalized_length,
    )

    write_jsonl(args.output_jsonl, rows)

    payload = {
        "input_zip": str(input_zip.resolve()),
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
        "municipio_codes": "all" if municipio_codes is None else sorted(municipio_codes),
        "min_normalized_length": args.min_normalized_length,
        **summary,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Rows output: {payload['rows_output']}")
    print(f"Output JSONL: {args.output_jsonl}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
