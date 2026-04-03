#!/usr/bin/env python3
"""Sanitize the large DD corpus before pseudolabelling.

This utility applies conservative hygiene rules over the raw corpus and splits
records into:
- kept (safe to process)
- dropped_safe (high-confidence discard)
- flagged_review (heuristic suspicion, kept separate for audit)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.paths import resolve_repo_artifact_path
from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl

PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
DOCUMENT_KEYWORDS = (
    "processo",
    "sentença",
    "sentenca",
    "acusado",
    "autos",
    "ajuizou",
    "ministério público",
    "ministerio publico",
    "vistos",
)
TOPONYM_HINTS = (
    "rio",
    "janeiro",
    "nilópolis",
    "nilopolis",
    "mesquita",
    "duque",
    "caxias",
    "niterói",
    "niteroi",
    "campos",
    "bonito",
    "senador",
    "meriti",
    "morro",
    "favela",
    "comunidade",
    "bairro",
    "rua",
    "avenida",
)


def _parse_jsonl_lines(text: str) -> list[dict]:
    rows = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_no} is not a JSON object")
        rows.append(payload)
    return rows


def _collapse_spaces(text: str) -> str:
    return " ".join(str(text).split())


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_relato(text: str) -> str:
    normalized = _collapse_spaces(str(text).strip().lower())
    normalized = _strip_accents(normalized)
    return normalized


def relato_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _tokenize_normalized(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", normalize_relato(text)) if token]


def _looks_like_mojibake(text: str) -> bool:
    if "�" in text:
        return True
    bad_sequences = ("Ã", "â", "ð", "ï¸", "�lvaro", "ATé", "POLÁT")
    return any(piece in text for piece in bad_sequences)


def _looks_like_document(text: str) -> bool:
    normalized = normalize_relato(text)
    return any(keyword in normalized for keyword in DOCUMENT_KEYWORDS)


def _looks_like_short_toponym(text: str) -> bool:
    tokens = _tokenize_normalized(text)
    if not tokens or len(tokens) > 4:
        return False
    if len(normalize_relato(text)) < 5:
        return False
    return any(token in TOPONYM_HINTS for token in tokens)


def _looks_like_uppercase_short(text: str) -> bool:
    stripped = str(text).strip()
    if not stripped or len(stripped) > 40:
        return False
    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio >= 0.9


def _relato_value(row: dict) -> str:
    value = row.get("relato")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_row(
    row: dict,
    *,
    row_index: int,
    seen_exact: set[str],
    seen_normalized: set[str],
    max_relato_chars: int,
    min_flag_short_chars: int,
) -> tuple[str, list[str], dict]:
    relato = _relato_value(row)
    relato_stripped = relato.strip()
    normalized_relato = normalize_relato(relato)
    record_meta = {
        "row_index_1based": row_index,
        "relato_length": len(relato),
        "normalized_relato_hash": relato_hash(normalized_relato) if normalized_relato else "",
    }

    dropped_reasons = []
    if not relato:
        dropped_reasons.append("missing_relato")
    elif not relato_stripped:
        dropped_reasons.append("empty_relato")
    elif PUNCT_ONLY_RE.match(relato_stripped):
        dropped_reasons.append("punctuation_only_relato")
    elif relato in seen_exact:
        dropped_reasons.append("duplicate_exact_relato")
    elif normalized_relato and normalized_relato in seen_normalized:
        dropped_reasons.append("duplicate_normalized_relato")
    elif len(relato) > max_relato_chars:
        dropped_reasons.append("relato_too_long")

    if dropped_reasons:
        return "dropped_safe", dropped_reasons, record_meta

    exact_key = relato
    normalized_key = normalized_relato
    seen_exact.add(exact_key)
    if normalized_key:
        seen_normalized.add(normalized_key)

    flagged_reasons = []
    if 0 < len(relato) < min_flag_short_chars:
        flagged_reasons.append("relato_too_short_for_review")
    if _looks_like_short_toponym(relato):
        flagged_reasons.append("looks_like_short_toponym")
    if _looks_like_document(relato):
        flagged_reasons.append("looks_like_document")
    if _looks_like_uppercase_short(relato):
        flagged_reasons.append("uppercase_short_relato")
    if _looks_like_mojibake(relato):
        flagged_reasons.append("mojibake_suspected")

    if flagged_reasons:
        return "flagged_review", flagged_reasons, record_meta
    return "kept", [], record_meta


def build_summary(
    *,
    input_path: str,
    kept_rows: list[dict],
    dropped_rows: list[dict],
    flagged_rows: list[dict],
    counters: Counter,
    max_relato_chars: int,
    min_flag_short_chars: int,
) -> dict:
    all_rows = kept_rows + dropped_rows + flagged_rows
    lengths = [
        item["_sanitization"]["relato_length"]
        for item in all_rows
        if isinstance(item.get("_sanitization"), dict)
    ]

    def _bucket_count(threshold: int) -> int:
        return sum(1 for length in lengths if length >= threshold)

    return {
        "input": input_path,
        "config": {
            "max_relato_chars": max_relato_chars,
            "min_flag_short_chars": min_flag_short_chars,
        },
        "rows_total": len(all_rows),
        "rows_kept": len(kept_rows),
        "rows_dropped_safe": len(dropped_rows),
        "rows_flagged_review": len(flagged_rows),
        "reasons": dict(counters),
        "length_distribution": {
            "avg_relato_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "max_relato_length": max(lengths) if lengths else 0,
            "relato_len_ge_20": _bucket_count(20),
            "relato_len_ge_100": _bucket_count(100),
            "relato_len_ge_500": _bucket_count(500),
            "relato_len_ge_1000": _bucket_count(1000),
            "relato_len_ge_5000": _bucket_count(5000),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize the DD large corpus before pseudolabelling.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL corpus.")
    parser.add_argument("--output-sanitized-jsonl", required=True, help="Output JSONL for kept records.")
    parser.add_argument("--output-dropped-jsonl", required=True, help="Output JSONL for safely dropped records.")
    parser.add_argument("--output-flagged-jsonl", required=True, help="Output JSONL for heuristic review flags.")
    parser.add_argument("--summary-json", required=True, help="Output JSON summary.")
    parser.add_argument("--max-relato-chars", type=int, default=5000, help="Hard drop records whose relato exceeds this length.")
    parser.add_argument("--min-flag-short-chars", type=int, default=20, help="Flag records shorter than this many chars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_json_or_jsonl(resolve_repo_artifact_path(__file__, args.input))

    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    counters = Counter()
    kept_rows = []
    dropped_rows = []
    flagged_rows = []

    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            counters["invalid_row_type"] += 1
            continue

        status, reasons, meta = classify_row(
            row,
            row_index=row_index,
            seen_exact=seen_exact,
            seen_normalized=seen_normalized,
            max_relato_chars=args.max_relato_chars,
            min_flag_short_chars=args.min_flag_short_chars,
        )
        enriched = dict(row)
        enriched["_sanitization"] = {
            "status": status,
            "reasons": reasons,
            **meta,
        }

        if status == "kept":
            kept_rows.append(enriched)
            counters["kept"] += 1
        elif status == "dropped_safe":
            dropped_rows.append(enriched)
            counters["dropped_safe"] += 1
        else:
            flagged_rows.append(enriched)
            counters["flagged_review"] += 1

        for reason in reasons:
            counters[reason] += 1

    sanitized_path = resolve_repo_artifact_path(__file__, args.output_sanitized_jsonl)
    dropped_path = resolve_repo_artifact_path(__file__, args.output_dropped_jsonl)
    flagged_path = resolve_repo_artifact_path(__file__, args.output_flagged_jsonl)
    summary_path = resolve_repo_artifact_path(__file__, args.summary_json)

    write_jsonl(sanitized_path, kept_rows)
    write_jsonl(dropped_path, dropped_rows)
    write_jsonl(flagged_path, flagged_rows)

    summary = build_summary(
        input_path=str(resolve_repo_artifact_path(__file__, args.input)),
        kept_rows=kept_rows,
        dropped_rows=dropped_rows,
        flagged_rows=flagged_rows,
        counters=counters,
        max_relato_chars=args.max_relato_chars,
        min_flag_short_chars=args.min_flag_short_chars,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved sanitized JSONL: {sanitized_path}")
    print(f"Saved dropped JSONL: {dropped_path}")
    print(f"Saved flagged JSONL: {flagged_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Rows total: {summary['rows_total']}")
    print(f"Rows kept: {summary['rows_kept']}")
    print(f"Rows dropped_safe: {summary['rows_dropped_safe']}")
    print(f"Rows flagged_review: {summary['rows_flagged_review']}")


if __name__ == "__main__":
    main()
