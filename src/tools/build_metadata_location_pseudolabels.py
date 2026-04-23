#!/usr/bin/env python3
"""Build conservative Location-only pseudolabel candidates from metadata literal matches."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl

DEFAULT_METADATA_FIELDS = ("logradouroLocal", "bairroLocal")
DEFAULT_SUBJECT_BONUS = {
    "Homicídios": 1.0,
    "Roubos em Geral": 0.5,
    "Roubo carga/ veículo": 0.5,
    "Armas": 0.2,
    "Tráfico de Drogas - Armas": 0.0,
}
GENERIC_VALUES = {
    "bairro",
    "centro",
    "cidade",
    "endereco",
    "endereço",
    "estrada",
    "local",
    "logradouro",
    "praca",
    "praça",
    "rua",
    "travessa",
    "avenida",
}
ROAD_MARKERS = {
    "alameda",
    "av",
    "avenida",
    "beco",
    "estrada",
    "ladeira",
    "largo",
    "praca",
    "praça",
    "rodovia",
    "rua",
    "trav",
    "travessa",
    "trv",
}
CONNECTOR_TOKENS = {"d", "da", "das", "de", "do", "dos"}
LOCATIVE_NAME_PREFIXES = {
    "bairro",
    "chacara",
    "chácara",
    "cidade",
    "comunidade",
    "conjunto",
    "favela",
    "jardim",
    "loteamento",
    "morro",
    "parque",
    "portal",
    "praia",
    "residencial",
    "sitio",
    "sítio",
    "vila",
}
ORGISH_SINGLETONS = {
    "bar",
    "casa",
    "cemiterio",
    "cemitério",
    "colegio",
    "colégio",
    "condominio",
    "condomínio",
    "creche",
    "empresa",
    "escola",
    "fabrica",
    "fábrica",
    "frigorifico",
    "frigorífico",
    "hospital",
    "igreja",
    "mercado",
    "shopping",
    "supermercado",
}
STATE_NAMES = {
    "acre",
    "alagoas",
    "amapa",
    "amapá",
    "amazonas",
    "bahia",
    "ceara",
    "ceará",
    "distrito federal",
    "espirito santo",
    "espírito santo",
    "goias",
    "goiás",
    "maranhao",
    "maranhão",
    "mato grosso",
    "mato grosso do sul",
    "minas gerais",
    "para",
    "pará",
    "paraiba",
    "paraíba",
    "parana",
    "paraná",
    "pernambuco",
    "piaui",
    "piauí",
    "rio de janeiro",
    "rio grande do norte",
    "rio grande do sul",
    "rondonia",
    "rondônia",
    "roraima",
    "santa catarina",
    "sao paulo",
    "são paulo",
    "sergipe",
    "tocantins",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a conservative Location-only pseudolabel candidate pool from metadata fields."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL corpus.")
    parser.add_argument("--train", required=True, help="Supervised training corpus used to estimate novelty.")
    parser.add_argument("--output-pool-jsonl", required=True, help="Full candidate pool JSONL.")
    parser.add_argument("--output-review-jsonl", required=True, help="Top-N review JSONL.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output.")
    parser.add_argument(
        "--output-pseudolabel-jsonl",
        default="",
        help="Optional final pseudolabel JSONL with only text/entities/source_id from the top-N selection.",
    )
    parser.add_argument("--top-n", type=int, default=100, help="Number of top candidates to export for review.")
    parser.add_argument(
        "--metadata-fields",
        default=",".join(DEFAULT_METADATA_FIELDS),
        help="Comma-separated metadata fields to mine as Location-only seeds.",
    )
    parser.add_argument(
        "--max-entities-per-row",
        type=int,
        default=2,
        help="Drop rows with more than this number of matched location entities.",
    )
    parser.add_argument(
        "--min-normalized-length",
        type=int,
        default=5,
        help="Minimum normalized metadata length to consider.",
    )
    parser.add_argument(
        "--min-single-token-length",
        type=int,
        default=6,
        help="Minimum length for a single-token location value.",
    )
    parser.add_argument(
        "--require-logradouro-marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require logradouroLocal values to contain a road-like marker such as rua/av/estrada.",
    )
    parser.add_argument(
        "--drop-bairro-equal-city",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop bairroLocal values that normalize to the same string as cidadeLocal.",
    )
    parser.add_argument(
        "--field-bonus-json",
        default='{"logradouroLocal": 2.0, "bairroLocal": 1.5}',
        help="JSON object mapping metadata field name to ranking bonus.",
    )
    parser.add_argument(
        "--subject-bonus-json",
        default=json.dumps(DEFAULT_SUBJECT_BONUS, ensure_ascii=False),
        help="JSON object mapping assunto to ranking bonus.",
    )
    return parser.parse_args()


def _parse_csv(raw_value: str) -> list[str]:
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def _load_bonus_map(raw_value: str, flag_name: str) -> dict[str, float]:
    try:
        data = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {flag_name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object.")
    normalized = {}
    for key, value in data.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {flag_name}[{key!r}]") from exc
    return normalized


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    return " ".join(text.split())


def has_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _find_literal_case_insensitive(text: str, value: str):
    pattern = re.compile(re.escape(str(value).strip()), re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    return match.start(), match.end(), text[match.start() : match.end()]


def _is_viable_location_value(value: str, *, min_normalized_length: int, min_single_token_length: int) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return False
    if len(normalized) < min_normalized_length:
        return False
    if normalized in GENERIC_VALUES:
        return False
    tokens = normalized.split()
    if len(tokens) == 1 and len(tokens[0]) < min_single_token_length:
        return False
    return True


def _is_viable_logradouro(value: str, *, require_logradouro_marker: bool) -> bool:
    normalized = normalize_text(value)
    tokens = normalized.split()
    if require_logradouro_marker and not any(token in ROAD_MARKERS for token in tokens):
        return False
    return True


def _is_viable_bairro(value: str, *, city_value: str, drop_bairro_equal_city: bool) -> bool:
    normalized = normalize_text(value)
    city_normalized = normalize_text(city_value)
    tokens = normalized.split()

    if normalized in STATE_NAMES:
        return False
    if drop_bairro_equal_city and city_normalized and normalized == city_normalized:
        return False
    if len(tokens) == 1 and tokens[0] in ORGISH_SINGLETONS:
        return False
    if tokens and tokens[0] in CONNECTOR_TOKENS and not any(token in LOCATIVE_NAME_PREFIXES for token in tokens):
        return False
    return True


def _rows_have_overlap(spans: list[dict]) -> bool:
    spans = sorted(spans, key=lambda item: (item["start"], item["end"]))
    for left, right in zip(spans, spans[1:]):
        if left["end"] > right["start"]:
            return True
    return False


def _extract_location_counts(train_rows: list[dict]) -> Counter:
    counts = Counter()
    for row in train_rows:
        text = row.get("text") or row.get("relato") or ""
        for span in row.get("spans", []) or []:
            if str(span.get("label", "")) != "Location":
                continue
            start = span.get("start")
            end = span.get("end")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                span_text = text[start:end]
            else:
                span_text = span.get("text", "")
            normalized = normalize_text(span_text)
            if normalized:
                counts[normalized] += 1
    return counts


def _score_row(
    *,
    matched_entities: list[dict],
    assunto: str,
    subject_bonus: dict[str, float],
) -> float:
    novelty_bonus = 0.0
    saturated_penalty = 0.0
    field_bonus = 0.0
    for entity in matched_entities:
        field_bonus += float(entity["_field_bonus"])
        train_count = int(entity["_train_count"])
        if train_count == 0:
            novelty_bonus += 2.0
        elif train_count <= 2:
            novelty_bonus += 1.0
        elif train_count <= 5:
            novelty_bonus += 0.5
        elif train_count >= 20:
            saturated_penalty += 0.5
    multi_match_bonus = 0.75 if len(matched_entities) >= 2 else 0.0
    return field_bonus + novelty_bonus + multi_match_bonus + float(subject_bonus.get(assunto, 0.1)) - saturated_penalty


def build_candidates(
    rows: list[dict],
    *,
    train_location_counts: Counter,
    metadata_fields: list[str],
    field_bonus: dict[str, float],
    subject_bonus: dict[str, float],
    max_entities_per_row: int,
    min_normalized_length: int,
    min_single_token_length: int,
    require_logradouro_marker: bool,
    drop_bairro_equal_city: bool,
) -> tuple[list[dict], dict]:
    stats = Counter()
    candidates = []

    for index, row in enumerate(rows, start=1):
        text = row.get("relato") or row.get("text") or ""
        if not has_text(text):
            continue
        stats["rows_seen"] += 1
        assunto = str(row.get("assunto", "") or "")

        matched = []
        for field in metadata_fields:
            value = row.get(field, "")
            if not has_text(value):
                continue
            if not _is_viable_location_value(
                value,
                min_normalized_length=min_normalized_length,
                min_single_token_length=min_single_token_length,
            ):
                continue
            if field == "logradouroLocal" and not _is_viable_logradouro(
                value,
                require_logradouro_marker=require_logradouro_marker,
            ):
                stats["dropped_invalid_logradouro_value"] += 1
                continue
            if field == "bairroLocal" and not _is_viable_bairro(
                value,
                city_value=row.get("cidadeLocal", ""),
                drop_bairro_equal_city=drop_bairro_equal_city,
            ):
                stats["dropped_invalid_bairro_value"] += 1
                continue
            found = _find_literal_case_insensitive(text, value)
            if not found:
                continue
            start, end, surface = found
            normalized = normalize_text(surface)
            matched.append(
                {
                    "start": start,
                    "end": end,
                    "text": surface,
                    "label": "Location",
                    "_source_field": field,
                    "_norm": normalized,
                    "_train_count": int(train_location_counts.get(normalized, 0)),
                    "_field_bonus": float(field_bonus.get(field, 1.0)),
                }
            )

        deduped = {}
        for entity in matched:
            key = (entity["start"], entity["end"], entity["label"])
            deduped.setdefault(key, entity)
        matched = list(deduped.values())

        if not matched:
            continue
        if len(matched) > max_entities_per_row:
            stats["dropped_too_many_entities"] += 1
            continue
        if _rows_have_overlap(matched):
            stats["dropped_overlapping"] += 1
            continue

        source_id = row.get("sample_id") or row.get("id") or row.get("source_id") or f"row_{index}"
        score = _score_row(matched_entities=matched, assunto=assunto, subject_bonus=subject_bonus)
        clean_entities = [
            {
                "start": entity["start"],
                "end": entity["end"],
                "text": entity["text"],
                "label": "Location",
                "seed_origin": f"metadata_literal_{entity['_source_field']}",
            }
            for entity in sorted(matched, key=lambda item: (item["start"], item["end"]))
        ]
        candidates.append(
            {
                "source_id": str(source_id),
                "text": text,
                "entities": clean_entities,
                "assunto": assunto,
                "logradouroLocal": row.get("logradouroLocal", ""),
                "bairroLocal": row.get("bairroLocal", ""),
                "cidadeLocal": row.get("cidadeLocal", ""),
                "pontodeReferenciaLocal": row.get("pontodeReferenciaLocal", ""),
                "_pilot_meta": {
                    "score": score,
                    "matched_fields": [entity["_source_field"] for entity in matched],
                    "train_location_counts": {
                        entity["_norm"]: entity["_train_count"] for entity in matched
                    },
                },
            }
        )
        stats["candidates_total"] += 1

    candidates.sort(
        key=lambda row: (
            -float(row.get("_pilot_meta", {}).get("score", 0.0)),
            -len(row.get("entities", [])),
            str(row.get("source_id", "")),
        )
    )
    return candidates, dict(stats)


def _build_summary(
    *,
    input_path: str,
    train_path: str,
    metadata_fields: list[str],
    top_n: int,
    candidates: list[dict],
    stats: dict,
) -> dict:
    review_rows = candidates[:top_n]
    return {
        "input_path": str(Path(input_path).resolve()),
        "train_path": str(Path(train_path).resolve()),
        "metadata_fields": metadata_fields,
        "candidates_total": len(candidates),
        "review_total": len(review_rows),
        **stats,
        "review_assunto_counts": dict(Counter(row.get("assunto", "") for row in review_rows)),
        "review_matched_field_counts": dict(
            Counter(
                field
                for row in review_rows
                for field in row.get("_pilot_meta", {}).get("matched_fields", [])
            )
        ),
    }


def main():
    args = parse_args()
    metadata_fields = _parse_csv(args.metadata_fields)
    if not metadata_fields:
        raise ValueError("--metadata-fields must contain at least one field.")
    field_bonus = _load_bonus_map(args.field_bonus_json, "--field-bonus-json")
    subject_bonus = _load_bonus_map(args.subject_bonus_json, "--subject-bonus-json")

    rows = read_json_or_jsonl(args.input)
    train_rows = read_json_or_jsonl(args.train)
    train_location_counts = _extract_location_counts(train_rows)

    candidates, stats = build_candidates(
        rows,
        train_location_counts=train_location_counts,
        metadata_fields=metadata_fields,
        field_bonus=field_bonus,
        subject_bonus=subject_bonus,
        max_entities_per_row=args.max_entities_per_row,
        min_normalized_length=args.min_normalized_length,
        min_single_token_length=args.min_single_token_length,
        require_logradouro_marker=args.require_logradouro_marker,
        drop_bairro_equal_city=args.drop_bairro_equal_city,
    )

    review_rows = candidates[: args.top_n]
    write_jsonl(args.output_pool_jsonl, candidates)
    write_jsonl(args.output_review_jsonl, review_rows)

    if args.output_pseudolabel_jsonl:
        pseudolabel_rows = [
            {
                "source_id": row["source_id"],
                "text": row["text"],
                "entities": row["entities"],
            }
            for row in review_rows
        ]
        write_jsonl(args.output_pseudolabel_jsonl, pseudolabel_rows)

    summary = _build_summary(
        input_path=args.input,
        train_path=args.train,
        metadata_fields=metadata_fields,
        top_n=args.top_n,
        candidates=candidates,
        stats=stats,
    )
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Input rows: {len(rows)}")
    print(f"Candidate pool: {len(candidates)}")
    print(f"Review rows: {len(review_rows)}")
    print(f"Pool JSONL: {args.output_pool_jsonl}")
    print(f"Review JSONL: {args.output_review_jsonl}")
    print(f"Summary JSON: {args.summary_json}")
    if args.output_pseudolabel_jsonl:
        print(f"Pseudolabel JSONL: {args.output_pseudolabel_jsonl}")


if __name__ == "__main__":
    main()
