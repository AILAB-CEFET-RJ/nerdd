#!/usr/bin/env python3
"""Profile conservative Person/Organization signal inside a metadata-based candidate pool."""

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
NAME_TOKEN = r"[A-ZÀ-Ú][A-Za-zÀ-ÿ]+"
FULL_NAME = rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,4}}"

PERSON_PATTERNS = {
    "vulgo_fullname_alias": re.compile(
        rf"(?P<fullname>{FULL_NAME})\s*,?\s*vulgo\s+(?P<alias>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]+)",
        re.IGNORECASE,
    ),
    "role_name": re.compile(
        rf"\b(?:{'|'.join(PERSON_ROLE_WORDS)})\s+(?P<name>{FULL_NAME})",
        re.IGNORECASE,
    ),
    "named_perpetrator": re.compile(
        rf"(?P<name>{FULL_NAME})\s+(?:{'|'.join(PERSON_ACTION_WORDS)})\b",
        re.IGNORECASE,
    ),
    "named_victim": re.compile(
        rf"(?:vitima\s+(?:e|é|foi)\s*:?\s*|morte\s+d[eo]\s+|vida\s+de\s+)(?P<name>{FULL_NAME})",
        re.IGNORECASE,
    ),
}
ORG_PATTERNS = {
    f"org_pattern_{idx}": re.compile(pattern, re.IGNORECASE)
    for idx, pattern in enumerate(ORG_LITERAL_PATTERNS, start=1)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure conservative Person/Organization signal in a metadata candidate pool."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL path.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output path.")
    parser.add_argument(
        "--output-samples-jsonl",
        default="",
        help="Optional JSONL with sampled rows from each signal bucket.",
    )
    parser.add_argument(
        "--output-bucket-csv",
        default="",
        help="Optional CSV-like TSV output with one row per source_id and bucket.",
    )
    parser.add_argument(
        "--max-samples-per-bucket",
        type=int,
        default=20,
        help="Maximum sampled rows to keep per bucket when --output-samples-jsonl is used.",
    )
    parser.add_argument(
        "--location-label",
        default="Location",
        help="Location label name inside existing entities/spans.",
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


def get_spans(row: dict) -> list[dict]:
    for key in ("entities", "spans", "ner"):
        spans = row.get(key)
        if isinstance(spans, list):
            return spans
    return []


def find_person_signal(text: str) -> list[dict]:
    matches = []
    for rule_name, pattern in PERSON_PATTERNS.items():
        for match in pattern.finditer(text):
            groups = match.groupdict()
            if "fullname" in groups:
                matches.append(
                    {
                        "rule": rule_name,
                        "text": groups["fullname"],
                        "start": match.start("fullname"),
                        "end": match.end("fullname"),
                    }
                )
            if "alias" in groups:
                matches.append(
                    {
                        "rule": f"{rule_name}_alias",
                        "text": groups["alias"],
                        "start": match.start("alias"),
                        "end": match.end("alias"),
                    }
                )
            if "name" in groups:
                matches.append(
                    {
                        "rule": rule_name,
                        "text": groups["name"],
                        "start": match.start("name"),
                        "end": match.end("name"),
                    }
                )
    return deduplicate_matches(matches)


def find_org_signal(text: str) -> list[dict]:
    matches = []
    for rule_name, pattern in ORG_PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append(
                {
                    "rule": rule_name,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return deduplicate_matches(matches)


def deduplicate_matches(matches: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for match in sorted(matches, key=lambda item: (item["start"], item["end"], item["text"])):
        key = (match["start"], match["end"], normalize_text(match["text"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def bucket_name(*, has_location: bool, has_person: bool, has_org: bool) -> str:
    if has_location and has_person and has_org:
        return "location_person_org"
    if has_location and has_person:
        return "location_person"
    if has_location and has_org:
        return "location_org"
    if has_location:
        return "location_only"
    if has_person and has_org:
        return "person_org_only"
    if has_person:
        return "person_only"
    if has_org:
        return "org_only"
    return "no_signal"


def summarize_rows(rows: list[dict], *, location_label: str, max_samples_per_bucket: int):
    bucket_counts = Counter()
    person_rule_counts = Counter()
    org_rule_counts = Counter()
    samples_by_bucket: dict[str, list[dict]] = {}
    city_counts = Counter()
    tsv_rows = []

    for row in rows:
        text = get_text(row)
        spans = get_spans(row)
        has_location = any(str(span.get("label", "")) == location_label for span in spans)
        person_matches = find_person_signal(text)
        org_matches = find_org_signal(text)
        has_person = bool(person_matches)
        has_org = bool(org_matches)
        bucket = bucket_name(has_location=has_location, has_person=has_person, has_org=has_org)

        bucket_counts[bucket] += 1
        city_counts[str(row.get("cidadeLocal", "")).strip()] += 1
        for match in person_matches:
            person_rule_counts[match["rule"]] += 1
        for match in org_matches:
            org_rule_counts[match["rule"]] += 1

        enriched = {
            "source_id": row.get("source_id", ""),
            "assunto": row.get("assunto", ""),
            "cidadeLocal": row.get("cidadeLocal", ""),
            "bucket": bucket,
            "person_signal_count": len(person_matches),
            "org_signal_count": len(org_matches),
            "person_matches": person_matches,
            "org_matches": org_matches,
            "entities": spans,
            "text": text,
        }
        samples = samples_by_bucket.setdefault(bucket, [])
        if len(samples) < max_samples_per_bucket:
            samples.append(enriched)
        tsv_rows.append(
            {
                "bucket": bucket,
                "source_id": enriched["source_id"],
                "assunto": enriched["assunto"],
                "cidadeLocal": enriched["cidadeLocal"],
                "person_signal_count": enriched["person_signal_count"],
                "org_signal_count": enriched["org_signal_count"],
            }
        )

    rows_total = len(rows)
    summary = {
        "rows_seen": rows_total,
        "bucket_counts": dict(bucket_counts),
        "bucket_rates": {
            name: (count / rows_total if rows_total else 0.0)
            for name, count in sorted(bucket_counts.items())
        },
        "rows_with_location": bucket_counts["location_only"]
        + bucket_counts["location_person"]
        + bucket_counts["location_org"]
        + bucket_counts["location_person_org"],
        "rows_with_person_signal": bucket_counts["person_only"]
        + bucket_counts["person_org_only"]
        + bucket_counts["location_person"]
        + bucket_counts["location_person_org"],
        "rows_with_org_signal": bucket_counts["org_only"]
        + bucket_counts["person_org_only"]
        + bucket_counts["location_org"]
        + bucket_counts["location_person_org"],
        "person_rule_counts": dict(person_rule_counts),
        "org_rule_counts": dict(org_rule_counts),
        "top_cidades": dict(city_counts.most_common(20)),
    }
    return summary, samples_by_bucket, tsv_rows


def write_bucket_tsv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("bucket\tsource_id\tassunto\tcidadeLocal\tperson_signal_count\torg_signal_count\n")
        for row in rows:
            handle.write(
                f"{row['bucket']}\t{row['source_id']}\t{row['assunto']}\t{row['cidadeLocal']}\t"
                f"{row['person_signal_count']}\t{row['org_signal_count']}\n"
            )


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    summary, samples_by_bucket, tsv_rows = summarize_rows(
        rows,
        location_label=args.location_label,
        max_samples_per_bucket=args.max_samples_per_bucket,
    )
    summary["input"] = str(Path(args.input).resolve())

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.output_samples_jsonl:
        output_rows = []
        for bucket in sorted(samples_by_bucket):
            output_rows.extend(samples_by_bucket[bucket])
        write_jsonl(args.output_samples_jsonl, output_rows)

    if args.output_bucket_csv:
        write_bucket_tsv(Path(args.output_bucket_csv), tsv_rows)

    print(f"Rows seen: {summary['rows_seen']}")
    for bucket, count in sorted(summary["bucket_counts"].items()):
        print(f"{bucket}: {count}")
    print(f"Summary JSON: {args.summary_json}")
    if args.output_samples_jsonl:
        print(f"Samples JSONL: {args.output_samples_jsonl}")
    if args.output_bucket_csv:
        print(f"Bucket TSV: {args.output_bucket_csv}")


if __name__ == "__main__":
    main()
