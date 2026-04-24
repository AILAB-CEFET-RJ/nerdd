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
FULL_NAME = rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,4}}"
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
PERSON_ACTION_WORDS = (
    "matou",
    "assassinou",
    "executou",
    "esfaqueou",
    "atirou",
    "tramou",
    "planejou",
)
ORG_LITERAL_PATTERNS = (
    r"\b\d{1,3}[ªa°º]?\s*DP\b(?:\s+de\s+[A-ZÀ-Ú][A-Za-zÀ-ÿ]+(?:\s+[A-ZÀ-Ú][A-Za-zÀ-ÿ]+){0,3})?",
    r"\b\d{1,3}[ªa°º]?\s*BPM\b",
    r"\b(?:BOPE|DRACO|GAECO|CEDAE|Light)\b",
    r"\b(?:Comando Vermelho|Terceiro Comando|Liga da Justica|Liga da Justiça)\b",
    r"\b(?:TCP|ADA|CV|PCC)\b",
    r"\b[Mm]ilic(?:ia|ía)\s+(?:de|da|do)\s+[A-ZÀ-Ú][A-Za-zÀ-ÿ]+(?:\s+[A-ZÀ-Ú][A-Za-zÀ-ÿ]+){0,4}",
    r"\b(?:batalhao|batalhão)\s+de\s+choque\b",
)

PERSON_PATTERNS = {
    "person_vulgo_fullname": re.compile(
        rf"(?P<fullname>{FULL_NAME})\s*,?\s*vulgo\s+(?P<alias>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]+)",
        re.IGNORECASE,
    ),
    "person_role_name": re.compile(
        rf"\b(?:{'|'.join(PERSON_ROLE_WORDS)})\s+(?P<name>{FULL_NAME})",
        re.IGNORECASE,
    ),
    "person_named_perpetrator": re.compile(
        rf"(?P<name>{FULL_NAME})\s+(?:{'|'.join(PERSON_ACTION_WORDS)})\b",
        re.IGNORECASE,
    ),
    "person_named_victim": re.compile(
        rf"(?:vitima\s+(?:e|é|foi)\s*:?\s*|morte\s+d[eo]\s+|vida\s+de\s+)(?P<name>{FULL_NAME})",
        re.IGNORECASE,
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


def find_person_seeds(text: str) -> list[dict]:
    matches = []
    for rule_name, pattern in PERSON_PATTERNS.items():
        for match in pattern.finditer(text):
            groups = match.groupdict()
            if "fullname" in groups:
                matches.append(
                    {
                        "start": match.start("fullname"),
                        "end": match.end("fullname"),
                        "text": groups["fullname"],
                        "label": "Person",
                        "seed_origin": rule_name,
                    }
                )
            if "alias" in groups:
                matches.append(
                    {
                        "start": match.start("alias"),
                        "end": match.end("alias"),
                        "text": groups["alias"],
                        "label": "Person",
                        "seed_origin": f"{rule_name}_alias",
                    }
                )
            if "name" in groups:
                matches.append(
                    {
                        "start": match.start("name"),
                        "end": match.end("name"),
                        "text": groups["name"],
                        "label": "Person",
                        "seed_origin": rule_name,
                    }
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
        normalized = {
            "start": int(entity["start"]),
            "end": int(entity["end"]),
            "text": text[int(entity["start"]) : int(entity["end"])],
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
) -> tuple[list[dict], dict]:
    kept = []
    counters = Counter()
    person_rule_counts = Counter()
    org_rule_counts = Counter()

    for row in rows:
        text = get_text(row)
        if not text:
            counters["dropped_missing_text"] += 1
            continue

        base_entities = get_entities(row)
        location_entities = [entity for entity in base_entities if entity.get("label") == "Location"]
        if require_two_location_seeds and len(location_entities) != 2:
            counters["dropped_location_seed_count"] += 1
            continue
        if not location_entities:
            counters["dropped_missing_location"] += 1
            continue

        person_seeds = find_person_seeds(text)
        org_seeds = find_org_seeds(text)
        if len(person_seeds) < min_person_seeds:
            counters["dropped_missing_person"] += 1
            continue
        if len(org_seeds) < min_organization_seeds:
            counters["dropped_missing_organization"] += 1
            continue

        merged_entities = merge_entities(text, location_entities, person_seeds, org_seeds)
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
            "person_seed_count": len(person_seeds),
            "organization_seed_count": len(org_seeds),
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

    summary = {
        "rows_seen": len(rows),
        "rows_kept": len(kept),
        "dropped_counts": dict(counters),
        "person_rule_counts": dict(person_rule_counts),
        "organization_rule_counts": dict(org_rule_counts),
        "assunto_counts": dict(Counter(str(row.get("assunto", "")).strip() for row in kept)),
        "cidade_counts": dict(Counter(str(row.get("cidadeLocal", "")).strip() for row in kept).most_common(20)),
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
