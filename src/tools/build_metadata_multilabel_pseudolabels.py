#!/usr/bin/env python3
"""Build conservative multilabel pseudolabels from a metadata-anchored Location candidate pool."""

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

NAME_TOKEN = r"[A-ZÀ-Ú][A-Za-zÀ-ÿ]+"
NAME_SEP = r"[ \t]+"
FULL_NAME = rf"{NAME_TOKEN}(?:{NAME_SEP}{NAME_TOKEN}){{1,4}}"
ROLE_NAME = rf"{NAME_TOKEN}(?:{NAME_SEP}{NAME_TOKEN}){{0,2}}"
PERSON_ROLE_WORDS = (
    "Sargento",
    "Cabo",
    "Soldado",
    "Tenente",
    "Capitao",
    "Capitão",
    "Major",
    "Coronel",
    "Delegado",
    "Investigador",
    "Agente",
    "Inspetor",
    "Detetive",
)
PERSON_ALIAS_MARKERS = (
    "alcunha",
    "apelidado",
    "apelido",
    "chamado",
    "conhecido",
    "vulgo",
)
WEAK_PERSON_PATTERNS = {
    "weak_person_known_as": re.compile(
        rf"(?i:\b(?:conhecido|conhecida|chamado|chamada|apelidado|apelidada)\s+(?:como|de)\s+)(?P<name>{NAME_TOKEN})"
    ),
    "weak_person_vulgo_alias": re.compile(
        rf"(?i:\bvulgo\s+)(?P<name>{NAME_TOKEN})"
    ),
}
ORG_LITERAL_PATTERNS = (
    r"\b\d{1,3}[ªa°º]?\s*DP\b(?:\s+de\s+[A-ZÀ-Ú][A-Za-zÀ-ÿ]+(?:\s+[A-ZÀ-Ú][A-Za-zÀ-ÿ]+){0,3})?",
    r"\b\d{1,3}[ªa°º]?\s*BPM\b",
    r"\b(?:BOPE|DRACO|GAECO|CEDAE|Light)\b",
    r"\b(?:Comando Vermelho|Terceiro Comando|Liga da Justica|Liga da Justiça)\b",
    r"\b(?:TCP|ADA|CV|PCC)\b",
    r"\b[Mm]il[ií]c(?:ia|ía)\s+(?:de|da|do)\s+(?-i:[A-ZÀ-Ú][A-Za-zÀ-ÿ]+)(?:\s+(?-i:[A-ZÀ-Ú][A-Za-zÀ-ÿ]+)){0,3}",
    r"\b\d{1,3}\s*(?:Batalh[aã]o|Batalhao)\b",
    r"\bDH\b",
    r"\b(?:batalhao|batalhão)\s+de\s+choque\b",
)
PERSON_FORBIDDEN_SUBSTRINGS = (
    "avenida",
    "bar",
    "bairro",
    "casa",
    "chamado",
    "de nome",
    "estrada",
    "justica",
    "justiça",
    "local",
    "operacao",
    "operação",
    "policia",
    "polícia",
    "por",
    "praca",
    "praça",
    "rua",
    "que",
    "traficante",
)
PERSON_FORBIDDEN_TOKENS = {
    "area",
    "familia",
    "filho",
    "mando",
    "mandaram",
    "morte",
    "moradores",
    "pm",
    "raiva",
    "realizando",
}
PERSON_FORBIDDEN_START_TOKENS = {
    "a",
    "as",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "outro",
    "outra",
    "um",
    "uma",
}
PERSON_FOLLOWING_VERBS = {
    "ajuda",
    "ajudar",
    "ajudando",
    "esta",
    "está",
    "fazer",
    "fechar",
    "fugiu",
    "mandou",
    "mantem",
    "mantém",
    "manter",
    "matou",
    "realiza",
    "realizar",
    "realizando",
}
PERSON_ALL_CAPS_FORBIDDEN_TOKENS = PERSON_FORBIDDEN_TOKENS | {
    "bahia",
    "bope",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "na",
    "no",
}
LOCATION_FORBIDDEN_SUBSTRINGS = (
    " bar ",
    " com ",
)
LOCATION_MARKERS = (
    "alameda",
    "av",
    "avenida",
    "estrada",
    "praca",
    "praça",
    "rodovia",
    "rua",
    "travessa",
    "trav",
    "trv",
)

PERSON_PATTERNS = {
    "person_vulgo_fullname": re.compile(
        rf"(?P<fullname>{FULL_NAME})\s*,?\s+(?i:vulgo)\s+(?P<alias>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]+)",
    ),
    "person_role_name": re.compile(
        rf"\b(?i:(?:{'|'.join(PERSON_ROLE_WORDS)}))\s+(?P<name>{ROLE_NAME})",
    ),
}
ORG_PATTERNS = {
    f"organization_pattern_{idx}": re.compile(pattern, re.IGNORECASE)
    for idx, pattern in enumerate(ORG_LITERAL_PATTERNS, start=1)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build conservative Person/Location/Organization pseudolabels from a metadata candidate pool."
    )
    parser.add_argument("--input", required=True, help="Input candidate pool JSONL or JSON.")
    parser.add_argument("--output-pool-jsonl", required=True, help="Full accepted multilabel pool JSONL.")
    parser.add_argument("--output-review-jsonl", required=True, help="Top-N review JSONL with audit metadata.")
    parser.add_argument("--output-pseudolabel-jsonl", required=True, help="Final top-N pseudolabel JSONL.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output.")
    parser.add_argument("--top-n", type=int, default=20, help="How many accepted rows to export for review/refit.")
    parser.add_argument(
        "--require-two-location-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exactly two existing Location seeds in the source row.",
    )
    parser.add_argument(
        "--min-person-seeds",
        type=int,
        default=1,
        help="Minimum number of conservative Person seeds required.",
    )
    parser.add_argument(
        "--min-organization-seeds",
        type=int,
        default=1,
        help="Minimum number of conservative Organization seeds required.",
    )
    parser.add_argument(
        "--drop-incomplete-person-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with strong weak-Person signal not covered by accepted Person seeds.",
    )
    parser.add_argument(
        "--max-uncovered-person-signals",
        type=int,
        default=2,
        help="Maximum weak Person signals not covered by accepted Person seeds before dropping a row.",
    )
    parser.add_argument(
        "--max-rows-per-cluster",
        type=int,
        default=2,
        help="Maximum accepted rows to keep per normalized near-duplicate cluster. Use 0 to disable.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    return " ".join(text.split())


def get_text(row: dict) -> str:
    for key in ("text", "relato", "texto", "description", "descricao"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_entities(row: dict) -> list[dict]:
    entities = row.get("entities")
    return entities if isinstance(entities, list) else []


def metadata_cluster_key(row: dict) -> str:
    parts = [part for part in (
        normalize_text(row.get("assunto", "")),
        normalize_text(row.get("cidadeLocal", "")),
        normalize_text(row.get("bairroLocal", "")),
        normalize_text(row.get("logradouroLocal", "")),
    ) if part]
    if len(parts) < 3:
        text_signature = " ".join(normalize_text(get_text(row)).split()[:80])
        parts.append(text_signature)
    return "|".join(parts)


def validate_entity_offset(text: str, entity: dict) -> tuple[bool, str]:
    try:
        start = int(entity["start"])
        end = int(entity["end"])
    except (KeyError, TypeError, ValueError):
        return False, "missing_or_non_integer_offset"
    if start < 0 or end <= start or end > len(text):
        return False, "out_of_bounds_offset"

    expected_text = entity.get("text")
    if isinstance(expected_text, str) and expected_text:
        actual_text = text[start:end]
        if expected_text != actual_text:
            return False, "entity_text_mismatch"
    return True, "ok"


def validated_location_entities(text: str, entities: list[dict], counters: Counter) -> list[dict] | None:
    locations = []
    for entity in entities:
        if entity.get("label") != "Location":
            continue
        valid, reason = validate_entity_offset(text, entity)
        if not valid:
            counters[f"dropped_location_{reason}"] += 1
            return None
        locations.append(entity)
    return locations


def _next_token(text: str, end: int) -> str:
    match = re.match(r"\s*([A-Za-zÀ-ÿ]+)", text[end:])
    return normalize_text(match.group(1)) if match else ""


def validate_person_seed(
    full_text: str,
    start: int,
    end: int,
    *,
    allow_single_token: bool,
) -> tuple[bool, str]:
    if start < 0 or end <= start or end > len(full_text):
        return False, "invalid_offset"
    raw = full_text[start:end]
    if raw != raw.strip():
        return False, "edge_whitespace"
    if "\n" in raw or "\r" in raw:
        return False, "linebreak"
    if raw[:1].islower():
        return False, "starts_lowercase"

    normalized = normalize_text(raw)
    if not normalized:
        return False, "empty"
    tokens = normalized.split()
    raw_tokens = raw.split()
    if not tokens or not raw_tokens:
        return False, "empty"
    if tokens[0] in PERSON_FORBIDDEN_START_TOKENS:
        return False, "starts_with_function_word"
    if any(piece in normalized for piece in PERSON_FORBIDDEN_SUBSTRINGS):
        return False, "forbidden_substring"
    if any(token in PERSON_FORBIDDEN_TOKENS for token in tokens):
        return False, "forbidden_token"
    if len(tokens) > 4:
        return False, "too_long"
    if not allow_single_token and len(tokens) < 2:
        return False, "too_short"
    if allow_single_token and len(tokens) == 1 and len(tokens[0]) < 3:
        return False, "too_short"
    if any(not token[:1].isupper() for token in raw_tokens):
        return False, "non_title_token"
    if any(token.isupper() and normalize_text(token) in PERSON_ALL_CAPS_FORBIDDEN_TOKENS for token in raw_tokens):
        return False, "all_caps_context_token"
    if _next_token(full_text, end) in PERSON_FOLLOWING_VERBS:
        return False, "followed_by_verb"
    return True, "ok"


def is_valid_person_candidate(text: str, *, allow_alias_single_token: bool) -> bool:
    raw = str(text or "").strip()
    if not raw or "\n" in raw or "\r" in raw:
        return False
    if raw[:1].islower():
        return False

    normalized = normalize_text(raw)
    if not normalized:
        return False
    tokens = normalized.split()
    if not tokens:
        return False
    if tokens[0] in PERSON_FORBIDDEN_START_TOKENS:
        return False
    if any(piece in normalized for piece in PERSON_FORBIDDEN_SUBSTRINGS):
        return False
    if allow_alias_single_token:
        return len(tokens) == 1 and len(tokens[0]) >= 3
    return len(tokens) >= 2


def is_valid_location_candidate(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or "\n" in raw or "\r" in raw:
        return False
    normalized = f" {normalize_text(raw)} "
    if any(piece in normalized for piece in LOCATION_FORBIDDEN_SUBSTRINGS):
        return False
    marker_count = sum(normalized.count(f" {marker} ") for marker in LOCATION_MARKERS)
    if marker_count > 1:
        return False
    return True


def deduplicate_matches(matches: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for match in sorted(matches, key=lambda item: (item["start"], item["end"], item["label"], item["text"])):
        key = (match["start"], match["end"], match["label"], normalize_text(match["text"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def span_is_covered(candidate: dict, accepted: list[dict]) -> bool:
    for entity in accepted:
        if candidate["start"] < entity["end"] and entity["start"] < candidate["end"]:
            return True
    return False


def find_weak_person_signals(text: str) -> list[dict]:
    matches = []
    for rule_name, pattern in WEAK_PERSON_PATTERNS.items():
        for match in pattern.finditer(text):
            name = match.group("name")
            valid, reason = validate_person_seed(
                text,
                match.start("name"),
                match.end("name"),
                allow_single_token=True,
            )
            if not valid:
                continue
            matches.append(
                {
                    "start": match.start("name"),
                    "end": match.end("name"),
                    "text": name,
                    "label": "Person",
                    "seed_origin": rule_name,
                }
            )
    return deduplicate_matches(matches)


def has_dense_alias_signal(text: str) -> bool:
    normalized = normalize_text(text)
    return sum(normalized.count(marker) for marker in PERSON_ALIAS_MARKERS) >= 3


def uncovered_person_signal_count(text: str, accepted_person_seeds: list[dict]) -> int:
    weak_signals = find_weak_person_signals(text)
    uncovered = [signal for signal in weak_signals if not span_is_covered(signal, accepted_person_seeds)]
    if has_dense_alias_signal(text) and uncovered:
        return max(len(uncovered), 3)
    return len(uncovered)


def add_person_match(
    matches: list[dict],
    rejection_counts: Counter,
    *,
    text: str,
    start: int,
    end: int,
    label_text: str,
    seed_origin: str,
    allow_single_token: bool,
) -> None:
    valid, reason = validate_person_seed(
        text,
        start,
        end,
        allow_single_token=allow_single_token,
    )
    if not valid:
        rejection_counts[f"person_rejected_{reason}"] += 1
        return
    matches.append(
        {
            "start": start,
            "end": end,
            "text": label_text,
            "label": "Person",
            "seed_origin": seed_origin,
        }
    )


def find_person_seeds(text: str, rejection_counts: Counter | None = None) -> list[dict]:
    matches = []
    if rejection_counts is None:
        rejection_counts = Counter()
    for rule_name, pattern in PERSON_PATTERNS.items():
        for match in pattern.finditer(text):
            groups = match.groupdict()
            if "fullname" in groups:
                fullname = groups["fullname"]
                add_person_match(
                    matches,
                    rejection_counts,
                    text=text,
                    start=match.start("fullname"),
                    end=match.end("fullname"),
                    label_text=fullname,
                    seed_origin=rule_name,
                    allow_single_token=False,
                )
            if "alias" in groups:
                alias = groups["alias"]
                add_person_match(
                    matches,
                    rejection_counts,
                    text=text,
                    start=match.start("alias"),
                    end=match.end("alias"),
                    label_text=alias,
                    seed_origin=f"{rule_name}_alias",
                    allow_single_token=True,
                )
            if "name" in groups:
                person_name = groups["name"]
                add_person_match(
                    matches,
                    rejection_counts,
                    text=text,
                    start=match.start("name"),
                    end=match.end("name"),
                    label_text=person_name,
                    seed_origin=rule_name,
                    allow_single_token=True,
                )
    return deduplicate_matches(matches)


def find_org_seeds(text: str) -> list[dict]:
    matches = []
    for rule_name, pattern in ORG_PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                    "label": "Organization",
                    "seed_origin": rule_name,
                }
            )
    return deduplicate_matches(matches)


def overlaps(existing: list[dict], candidate: dict) -> bool:
    for entity in existing:
        if candidate["start"] < entity["end"] and entity["start"] < candidate["end"]:
            return True
    return False


def merge_entities(text: str, base_entities: list[dict], person_seeds: list[dict], org_seeds: list[dict]) -> list[dict]:
    merged = []
    for entity in sorted(base_entities, key=lambda item: (item["start"], item["end"], item.get("label", ""))):
        entity_text = text[int(entity["start"]) : int(entity["end"])]
        if str(entity.get("label")) == "Location" and not is_valid_location_candidate(entity_text):
            continue
        normalized = {
            "start": int(entity["start"]),
            "end": int(entity["end"]),
            "text": entity_text,
            "label": str(entity["label"]),
            "seed_origin": entity.get("seed_origin", ""),
        }
        merged.append(normalized)

    for seed in sorted(person_seeds + org_seeds, key=lambda item: (item["start"], item["end"], item["label"])):
        if overlaps(merged, seed):
            continue
        merged.append(seed)

    merged.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    return merged


def score_row(row: dict, person_count: int, org_count: int) -> float:
    score = float(row.get("_pilot_meta", {}).get("score", 0.0))
    assunto = str(row.get("assunto", "")).strip()
    cidade = normalize_text(row.get("cidadeLocal", ""))
    lex_matches = row.get("_pilot_meta", {}).get("lexicon_matches", []) or []

    if assunto == "Homicídios":
        score += 0.25
    elif assunto == "Roubos em Geral":
        score += 0.10
    if cidade == "rio de janeiro":
        score += 0.15
    if lex_matches:
        score += 0.10

    score += min(person_count, 3) * 0.20
    score += min(org_count, 3) * 0.15
    return score


def build_multilabel_pool(
    rows: list[dict],
    *,
    require_two_location_seeds: bool,
    min_person_seeds: int,
    min_organization_seeds: int,
    drop_incomplete_person_signal: bool,
    max_uncovered_person_signals: int,
    max_rows_per_cluster: int,
) -> tuple[list[dict], dict]:
    kept = []
    counters = Counter()
    person_rule_counts = Counter()
    org_rule_counts = Counter()
    person_rejection_counts = Counter()

    for row in rows:
        text = get_text(row)
        if not text:
            counters["dropped_missing_text"] += 1
            continue

        base_entities = get_entities(row)
        location_entities = validated_location_entities(text, base_entities, counters)
        if location_entities is None:
            continue
        if require_two_location_seeds and len(location_entities) != 2:
            counters["dropped_location_seed_count"] += 1
            continue
        if not location_entities:
            counters["dropped_missing_location"] += 1
            continue

        person_seeds = find_person_seeds(text, person_rejection_counts)
        org_seeds = find_org_seeds(text)
        uncovered_person_signals = uncovered_person_signal_count(text, person_seeds)
        if drop_incomplete_person_signal:
            if uncovered_person_signals > max_uncovered_person_signals:
                counters["dropped_incomplete_person_signal"] += 1
                continue
        if len(person_seeds) < min_person_seeds:
            counters["dropped_missing_person"] += 1
            continue
        if len(org_seeds) < min_organization_seeds:
            counters["dropped_missing_organization"] += 1
            continue

        merged_entities = merge_entities(text, location_entities, person_seeds, org_seeds)
        if not any(entity["label"] == "Location" for entity in merged_entities):
            counters["dropped_location_filtered"] += 1
            continue
        if not any(entity["label"] == "Person" for entity in merged_entities):
            counters["dropped_person_overlap"] += 1
            continue
        if not any(entity["label"] == "Organization" for entity in merged_entities):
            counters["dropped_organization_overlap"] += 1
            continue

        for seed in person_seeds:
            person_rule_counts[seed["seed_origin"]] += 1
        for seed in org_seeds:
            org_rule_counts[seed["seed_origin"]] += 1

        enriched = dict(row)
        enriched["entities"] = merged_entities
        enriched["_multilabel_meta"] = {
            "cluster_key": metadata_cluster_key(row),
            "person_seed_count": len(person_seeds),
            "organization_seed_count": len(org_seeds),
            "uncovered_person_signal_count": uncovered_person_signals,
            "person_seed_rules": sorted({seed["seed_origin"] for seed in person_seeds}),
            "organization_seed_rules": sorted({seed["seed_origin"] for seed in org_seeds}),
            "selection_score": score_row(row, len(person_seeds), len(org_seeds)),
        }
        kept.append(enriched)

    kept.sort(
        key=lambda row: (
            -float(row["_multilabel_meta"]["selection_score"]),
            -int(row["_multilabel_meta"]["person_seed_count"]),
            -int(row["_multilabel_meta"]["organization_seed_count"]),
            row.get("source_id", ""),
        )
    )

    if max_rows_per_cluster > 0:
        clustered = []
        cluster_counts = Counter()
        for row in kept:
            cluster_key = row["_multilabel_meta"]["cluster_key"]
            if cluster_counts[cluster_key] >= max_rows_per_cluster:
                counters["dropped_cluster_limit"] += 1
                continue
            cluster_counts[cluster_key] += 1
            clustered.append(row)
        kept = clustered

    summary = {
        "rows_seen": len(rows),
        "rows_kept": len(kept),
        "dropped_counts": dict(counters),
        "person_rejection_counts": dict(person_rejection_counts),
        "person_rule_counts": dict(person_rule_counts),
        "organization_rule_counts": dict(org_rule_counts),
        "assunto_counts": dict(Counter(str(row.get("assunto", "")).strip() for row in kept)),
        "cidade_counts": dict(Counter(str(row.get("cidadeLocal", "")).strip() for row in kept).most_common(20)),
        "cluster_counts": dict(Counter(row["_multilabel_meta"]["cluster_key"] for row in kept).most_common(20)),
        "label_counts_kept": dict(Counter(entity["label"] for row in kept for entity in row["entities"])),
    }
    return kept, summary


def project_for_refit(row: dict) -> dict:
    return {
        "source_id": row.get("source_id", ""),
        "text": get_text(row),
        "entities": [
            {
                "start": entity["start"],
                "end": entity["end"],
                "text": entity["text"],
                "label": entity["label"],
                "seed_origin": entity.get("seed_origin", ""),
            }
            for entity in row.get("entities", [])
        ],
    }


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    kept, summary = build_multilabel_pool(
        rows,
        require_two_location_seeds=args.require_two_location_seeds,
        min_person_seeds=args.min_person_seeds,
        min_organization_seeds=args.min_organization_seeds,
        drop_incomplete_person_signal=args.drop_incomplete_person_signal,
        max_uncovered_person_signals=args.max_uncovered_person_signals,
        max_rows_per_cluster=args.max_rows_per_cluster,
    )

    review_rows = kept[: args.top_n]
    pseudolabel_rows = [project_for_refit(row) for row in review_rows]

    write_jsonl(args.output_pool_jsonl, kept)
    write_jsonl(args.output_review_jsonl, review_rows)
    write_jsonl(args.output_pseudolabel_jsonl, pseudolabel_rows)

    payload = {
        "input": str(Path(args.input).resolve()),
        "output_pool_jsonl": str(Path(args.output_pool_jsonl).resolve()),
        "output_review_jsonl": str(Path(args.output_review_jsonl).resolve()),
        "output_pseudolabel_jsonl": str(Path(args.output_pseudolabel_jsonl).resolve()),
        "top_n": args.top_n,
        "require_two_location_seeds": args.require_two_location_seeds,
        "min_person_seeds": args.min_person_seeds,
        "min_organization_seeds": args.min_organization_seeds,
        "drop_incomplete_person_signal": args.drop_incomplete_person_signal,
        "max_uncovered_person_signals": args.max_uncovered_person_signals,
        "max_rows_per_cluster": args.max_rows_per_cluster,
        **summary,
        "review_rows": len(review_rows),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Rows seen: {payload['rows_seen']}")
    print(f"Rows kept: {payload['rows_kept']}")
    print(f"Review rows: {payload['review_rows']}")
    print(f"Pool JSONL: {args.output_pool_jsonl}")
    print(f"Review JSONL: {args.output_review_jsonl}")
    print(f"Pseudolabel JSONL: {args.output_pseudolabel_jsonl}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
