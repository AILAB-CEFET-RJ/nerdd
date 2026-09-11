#!/usr/bin/env python3
"""Create a current-guide revision of the legacy calibration corpus.

The legacy calibration set predates ``docs/LABELLING_GUIDE.md``.  This tool
keeps the source file untouched, applies a curated set of current-guide
corrections, and records an auditable before/after diff.  It is deliberately
specific to this historical corpus rather than a general automatic annotator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LABELS = {"Person", "Location", "Organization"}
EXPECTED_SOURCE_SHA256 = "d74820e5fe698a0d34478a68db4a9cf71f2d94fb36bd7d09e45d79e4d4984fff"


# These terms are generic under the current annotation guide, platforms used
# only as a medium, or old partial spans that cannot stand by themselves.
GLOBAL_DROPS = {
    ("Organization", "policia"),
    ("Organization", "polícia"),
    ("Organization", "polícia militar , civil e federal"),
    ("Organization", "prefeitura"),
    ("Organization", "milicia"),
    ("Organization", "milícia"),
    ("Organization", "batalhão"),
    ("Organization", "bpm"),
    ("Organization", "upp"),
    ("Organization", "facebook"),
    ("Organization", "youtube"),
    ("Organization", "uber"),
    ("Organization", "grindr"),
    ("Organization", "gmail"),
    ("Organization", "sinesp cidadão"),
    ("Organization", "caps"),
    ("Organization", "conselho tutelar"),
    ("Organization", "associação de moradores"),
    ("Organization", "quadrilhas"),
    ("Location", "bailes funk"),
    ("Person", "mendigo"),
    ("Person", "tio"),
}


# Each operation is intentionally tied to a source row.  Rows are 1-based so
# they can be located directly in the annotation editor and in review notes.
#
# add:       ("add", mention, label, occurrence_in_text)
# remove:    ("remove", label, old_mention, occurrence_among_spans)
# relabel:   ("relabel", old_label, old_mention, new_label, occurrence_among_spans)
# replace:   ("replace", old_label, old_mention, new_mention, new_label,
#             occurrence_among_spans, occurrence_in_text)
# split:     ("split", old_label, old_mention, [(mention, label), ...],
#             occurrence_among_spans)
# relabel_all: ("relabel_all", old_label, old_mention, new_label)
# remove_all:  ("remove_all", label, old_mention)
ROW_OPERATIONS: dict[int, list[tuple[Any, ...]]] = {
    2: [("relabel", "Organization", "colégio Zenóbio", "Location", 0)],
    4: [
        ("replace", "Organization", "Rádio Tupi", "Antiga Rádio Tupi", "Location", 0, 0),
    ],
    6: [
        ("replace", "Location", "massambaba", "reserva biológica de massambaba", "Location", 0, 0),
        ("remove", "Location", "lagoa", 0),
        ("replace", "Location", "Araruama", "lagoa de Araruama", "Location", 0, 0),
    ],
    8: [
        ("replace", "Person", "Catiri", "comunidade do Catiri", "Location", 0, 0),
        ("relabel", "Organization", "Stilo show", "Location", 0),
    ],
    14: [
        ("remove", "Organization", "Prisma", 0),
        ("add", "39 BPM", "Organization", 0),
        ("replace", "Person", "traficante mexicano", "mexicano", "Person", 0, 0),
        ("split", "Location", "Alfredo e Antônio", [("Alfredo", "Location"), ("Antônio", "Location")], 0),
        ("add", "Ere", "Person", 0),
        ("remove", "Organization", "sistema prisional", 0),
    ],
    21: [("relabel", "Organization", "Faet", "Location", 0)],
    24: [
        ("replace", "Location", "rua . 11 . Próximo casa 2 Ltd 32", "rua . 11", "Location", 0, 0),
    ],
    25: [
        ("remove", "Organization", "ministério público", 0),
        ("remove", "Location", "queimados", 0),
        ("add", "ministério público de queimados", "Organization", 0),
        ("remove", "Organization", "PM", 0),
        ("add", "24 BPM", "Organization", 0),
        ("relabel", "Organization", "hospital da posse", "Location", 0),
    ],
    33: [
        ("remove", "Organization", "39° / 64° DP", 0),
        ("add", "39°", "Organization", 0),
        ("add", "64° DP", "Organization", 0),
        ("remove", "Organization", "21° / 41° BPM", 0),
        ("add", "21°", "Organization", 0),
        ("add", "41° BPM", "Organization", 0),
        ("remove", "Location", "Pavuna", 0),
        ("add", "Estação de trem da Pavuna", "Location", 0),
        ("remove", "Organization", "Sesc", 0),
        ("remove", "Location", "SJM", 0),
        ("add", "Sesc SJM", "Location", 0),
    ],
    36: [("remove", "Location", "rua aldrovando", 0)],
    48: [
        ("remove", "Organization", "policia ambiental", 0),
        ("remove", "Organization", "policia militar", 0),
        ("remove", "Location", "estado do rio de janeiro", 0),
        ("remove", "Organization", "UPAM", 0),
        (
            "add",
            "8 ( oitava ) compania de policia ambiental da policia militar do estado do rio de janeiro UPAM",
            "Organization",
            0,
        ),
    ],
    62: [
        ("relabel", "Organization", "colégio Marcílio dias", "Location", 0),
        ("relabel", "Organization", "igreja batista", "Location", 0),
    ],
    66: [
        ("replace", "Organization", "polícia militar", "20 polícia militar", "Organization", 0, 0),
    ],
    68: [
        ("replace", "Organization", "associação", "associação da comunidade do guarda", "Location", 0, 0),
        ("remove", "Location", "comunidade do guarda", 0),
        ("remove", "Location", "guarda", 0),
        ("relabel", "Organization", "associação do guarda", "Location", 0),
    ],
    74: [("relabel", "Organization", "Garota do Papai", "Location", 0)],
    75: [
        ("remove", "Person", "NEM", 0),
        ("remove", "Person", "·JOHN", 0),
        ("remove", "Location", "rua FLEXALAQUI", 0),
        ("relabel_all", "Organization", "PADARIA VERÃO VERMELHO", "Location"),
        ("relabel_all", "Organization", "Guanabara", "Location"),
    ],
    84: [("relabel", "Organization", "bar Rainha dos Caldos", "Location", 0)],
    83: [
        ("remove", "Location", "estrada da pedra", 0),
        ("replace", "Organization", "Melícia", "Melícia da estrada da pedra", "Organization", 0, 0),
    ],
    85: [
        ("relabel_all", "Organization", "comlurb", "Location"),
        ("remove", "Location", "CAJU", 2),
        ("add", "UPP CAJU", "Organization", 0),
    ],
    86: [("relabel", "Organization", "fábrica Apólo", "Location", 0)],
    88: [("replace", "Organization", "Dp", "5 Dp", "Organization", 0, 0)],
    89: [("relabel", "Organization", "American Pet", "Location", 0)],
    92: [("replace", "Location", "ruas Mato Grosso", "Mato Grosso", "Location", 0, 0)],
    95: [
        ("split", "Location", "Ruas 39 e 40", [("39", "Location"), ("40", "Location")], 0),
        (
            "split",
            "Location",
            "Ruas 38 37 36 35 34 33 32",
            [(str(number), "Location") for number in (38, 37, 36, 35, 34, 33, 32)],
            0,
        ),
        ("split", "Location", "Rua 39 e 40", [("39", "Location"), ("40", "Location")], 0),
        ("add", "7o BPM", "Organization", 0),
        ("add", "7 Bpm", "Organization", 0),
        ("add", "7 Bpm", "Organization", 1),
    ],
    101: [
        ("add", "9 batalhão", "Organization", 0),
        ("add", "9 batalhão", "Organization", 1),
    ],
    108: [],
    109: [("relabel", "Person", "Vítor dumas bosque", "Location", 0)],
    117: [("relabel", "Organization", "mercado mundial", "Location", 0)],
    121: [("relabel", "Person", "Nonato Farias", "Location", 0)],
    126: [("replace", "Location", "Rua Novo", "Rua Novo Horizonte", "Location", 0, 0)],
    139: [("split", "Location", "ruas 6 , 5 e 4", [("6", "Location"), ("5", "Location"), ("4", "Location")], 0)],
    147: [
        ("remove", "Location", "Salgueiro", 0),
        ("replace", "Organization", "Brizolao", "Brizolao do Salgueiro", "Location", 0, 0),
    ],
    148: [("remove", "Organization", "clínica da família", 0)],
    151: [("relabel_all", "Organization", "OPERA CLUB", "Location")],
    160: [("remove", "Location", "brisolao", 0)],
    164: [
        ("replace", "Organization", "América Football Club", "sede do América Football Club", "Location", 0, 0),
    ],
    167: [
        ("replace", "Organization", "CV", "facção CV", "Organization", 0, 0),
        ("add", "7°", "Organization", 0),
    ],
    170: [
        ("replace", "Organization", "Tupi", "antigo prédio da Tupi", "Location", 0, 0),
    ],
    178: [
        ("add", "rua 2033", "Location", 0),
        ("replace", "Location", "rua inhumay", "inhumay", "Location", 0, 0),
        ("replace", "Location", "rua Major Augusto César", "Major Augusto César", "Location", 0, 0),
    ],
    183: [
        ("remove", "Location", "varjao", 0),
        ("remove", "Location", "rj", 0),
        ("add", "rj", "Location", 1),
        ("add", "10 pelotao", "Organization", 0),
        ("remove", "Location", "bar do campo", 0),
    ],
    184: [("add", "20° BPM", "Organization", 0)],
    185: [("remove", "Person", "X9", 0)],
    186: [("relabel", "Location", "VK", "Organization", 0)],
    192: [
        ("remove_all", "Location", "Central"),
        ("remove_all", "Location", "Magé"),
        ("remove_all", "Location", "Saracuruna"),
        ("relabel_all", "Organization", "Feirão das malhas", "Location"),
        ("relabel_all", "Organization", "Reduc", "Location"),
        ("replace", "Location", "Campos Elísios", "Delegacia de Campos Elísios", "Location", 0, 0),
        ("remove", "Organization", "DEDIC", 0),
        ("relabel", "Organization", "Porcão", "Location", 0),
        ("remove", "Location", "BR", 0),
        ("remove", "Location", "Caxias", 0),
        ("remove", "Person", "Roberto Santos", 0),
    ],
    195: [("relabel", "Organization", "posto BR", "Location", 0)],
    209: [("replace", "Location", "queimados", "centro queimados", "Location", 0, 0)],
    212: [("replace", "Organization", "TCP", "facção TCP", "Organization", 0, 0)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an audited current-guide revision of the legacy calibration corpus."
    )
    parser.add_argument("--input", required=True, help="Legacy JSON array to revise.")
    parser.add_argument("--output", required=True, help="New JSON array. The input is never overwritten.")
    parser.add_argument("--audit-json", required=True, help="JSON audit with every changed record.")
    parser.add_argument("--summary-json", required=True, help="JSON summary of the revision.")
    parser.add_argument(
        "--allow-source-sha256-mismatch",
        action="store_true",
        help="Allow applying this historical, row-specific revision to a different source file.",
    )
    return parser.parse_args()


def normalize_mention(value: str) -> str:
    return " ".join(str(value).casefold().split())


def span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["label"])


def span_text(text: str, span: dict[str, Any]) -> str:
    return text[int(span["start"]) : int(span["end"])]


def find_text_occurrences(text: str, mention: str) -> list[tuple[int, int]]:
    needle = mention.casefold()
    haystack = text.casefold()
    if not needle:
        raise ValueError("Cannot add an empty mention.")
    positions: list[tuple[int, int]] = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            return positions
        positions.append((index, index + len(mention)))
        start = index + 1


def matching_span_indexes(
    record: dict[str, Any],
    label: str,
    mention: str,
) -> list[int]:
    text = str(record["text"])
    target = normalize_mention(mention)
    return [
        index
        for index, span in enumerate(record["spans"])
        if span["label"] == label and normalize_mention(span_text(text, span)) == target
    ]


def get_matching_span_index(
    record: dict[str, Any],
    label: str,
    mention: str,
    occurrence: int,
) -> int:
    matches = matching_span_indexes(record, label, mention)
    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError(
            f"Could not find span {label}:{mention!r} occurrence {occurrence}; found {len(matches)}."
        )
    return matches[occurrence]


def add_span(
    record: dict[str, Any],
    mention: str,
    label: str,
    occurrence: int,
) -> None:
    if label not in LABELS:
        raise ValueError(f"Unsupported label: {label}")
    matches = find_text_occurrences(str(record["text"]), mention)
    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError(
            f"Could not find text {mention!r} occurrence {occurrence}; found {len(matches)}."
        )
    start, end = matches[occurrence]
    candidate = {"start": start, "end": end, "label": label}
    if span_key(candidate) not in {span_key(span) for span in record["spans"]}:
        record["spans"].append(candidate)


def remove_span(record: dict[str, Any], label: str, mention: str, occurrence: int) -> None:
    index = get_matching_span_index(record, label, mention, occurrence)
    record["spans"].pop(index)


def remove_all(record: dict[str, Any], label: str, mention: str) -> None:
    indexes = matching_span_indexes(record, label, mention)
    for index in reversed(indexes):
        record["spans"].pop(index)


def relabel_span(
    record: dict[str, Any],
    old_label: str,
    mention: str,
    new_label: str,
    occurrence: int,
) -> None:
    if new_label not in LABELS:
        raise ValueError(f"Unsupported label: {new_label}")
    index = get_matching_span_index(record, old_label, mention, occurrence)
    record["spans"][index]["label"] = new_label


def relabel_all(record: dict[str, Any], old_label: str, mention: str, new_label: str) -> None:
    for index in matching_span_indexes(record, old_label, mention):
        record["spans"][index]["label"] = new_label


def replace_span(
    record: dict[str, Any],
    old_label: str,
    old_mention: str,
    new_mention: str,
    new_label: str,
    span_occurrence: int,
    text_occurrence: int,
) -> None:
    remove_span(record, old_label, old_mention, span_occurrence)
    add_span(record, new_mention, new_label, text_occurrence)


def split_span(
    record: dict[str, Any],
    old_label: str,
    old_mention: str,
    replacements: Iterable[tuple[str, str]],
    occurrence: int,
) -> None:
    index = get_matching_span_index(record, old_label, old_mention, occurrence)
    old_span = record["spans"].pop(index)
    text = str(record["text"])
    old_start, old_end = int(old_span["start"]), int(old_span["end"])
    old_text = text[old_start:old_end]

    for mention, label in replacements:
        position = old_text.casefold().find(mention.casefold())
        if position < 0:
            raise ValueError(
                f"Cannot split {old_mention!r}: replacement {mention!r} is outside the old span."
            )
        record["spans"].append(
            {"start": old_start + position, "end": old_start + position + len(mention), "label": label}
        )


def apply_operation(record: dict[str, Any], operation: tuple[Any, ...]) -> None:
    kind = operation[0]
    if kind == "add":
        _, mention, label, occurrence = operation
        add_span(record, mention, label, occurrence)
    elif kind == "remove":
        _, label, mention, occurrence = operation
        remove_span(record, label, mention, occurrence)
    elif kind == "remove_all":
        _, label, mention = operation
        remove_all(record, label, mention)
    elif kind == "relabel":
        _, old_label, mention, new_label, occurrence = operation
        relabel_span(record, old_label, mention, new_label, occurrence)
    elif kind == "relabel_all":
        _, old_label, mention, new_label = operation
        relabel_all(record, old_label, mention, new_label)
    elif kind == "replace":
        _, old_label, old_mention, new_mention, new_label, span_occurrence, text_occurrence = operation
        replace_span(
            record,
            old_label,
            old_mention,
            new_mention,
            new_label,
            span_occurrence,
            text_occurrence,
        )
    elif kind == "split":
        _, old_label, old_mention, replacements, occurrence = operation
        split_span(record, old_label, old_mention, replacements, occurrence)
    else:
        raise ValueError(f"Unknown operation: {kind}")


def trim_span_boundaries(text: str, span: dict[str, Any]) -> dict[str, Any] | None:
    start, end = int(span["start"]), int(span["end"])
    while start < end and (text[start].isspace() or text[start] in ".,;:!?()[]{}\"·"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ".,;:!?()[]{}\""):
        end -= 1
    if end <= start:
        return None
    return {"start": start, "end": end, "label": str(span["label"])}


def resolve_overlaps(text: str, spans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the broader span when stale nested spans remain after revision."""
    unique = {span_key(span): span for span in spans}
    ordered = sorted(
        unique.values(),
        key=lambda span: (int(span["start"]), -(int(span["end"]) - int(span["start"])), str(span["label"])),
    )
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for span in ordered:
        start, end = int(span["start"]), int(span["end"])
        if any(start < int(current["end"]) and int(current["start"]) < end for current in kept):
            dropped.append(span)
            continue
        kept.append(span)
    kept.sort(key=lambda span: (int(span["start"]), int(span["end"]), str(span["label"])))
    return kept, dropped


def clean_spans(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(record["text"])
    trimmed: list[dict[str, Any]] = []
    for span in record["spans"]:
        if span["label"] not in LABELS:
            raise ValueError(f"Unexpected label {span['label']!r}")
        cleaned = trim_span_boundaries(text, span)
        if cleaned is None:
            continue
        if not (0 <= cleaned["start"] < cleaned["end"] <= len(text)):
            raise ValueError(f"Invalid offsets after cleanup: {cleaned}")
        trimmed.append(cleaned)
    resolved, _ = resolve_overlaps(text, trimmed)
    return resolved


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        span["label"]
        for row in rows
        for span in row["spans"]
    )
    return {label: int(counts[label]) for label in sorted(LABELS)}


def validate_rows(rows: list[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows, start=1):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Row {row_index} has no text string.")
        previous_end = -1
        for span in row.get("spans", []):
            start, end = int(span["start"]), int(span["end"])
            if span["label"] not in LABELS:
                raise ValueError(f"Row {row_index} has unknown label: {span['label']!r}")
            if not (0 <= start < end <= len(text)):
                raise ValueError(f"Row {row_index} has invalid offsets: {span}")
            if start < previous_end:
                raise ValueError(f"Row {row_index} retains overlapping spans.")
            previous_end = end


def synchronize_identical_text_rows(rows: list[dict[str, Any]]) -> None:
    """Give exact duplicate reports the union of their revised annotations."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["text"]), []).append(row)

    for duplicate_rows in grouped.values():
        if len(duplicate_rows) < 2:
            continue
        canonical = {
            span_key(span): {"start": span["start"], "end": span["end"], "label": span["label"]}
            for row in duplicate_rows
            for span in row["spans"]
        }
        canonical_spans = list(canonical.values())
        for row in duplicate_rows:
            row["spans"] = clean_spans({"text": row["text"], "spans": canonical_spans})


def apply_global_drops(record: dict[str, Any]) -> None:
    text = str(record["text"])
    record["spans"] = [
        span
        for span in record["spans"]
        if (span["label"], normalize_mention(span_text(text, span))) not in GLOBAL_DROPS
    ]


def build_revision(source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    revised_rows = deepcopy(source_rows)

    for index, revised in enumerate(revised_rows, start=1):
        if not isinstance(revised, dict):
            raise ValueError(f"Row {index} is not a JSON object.")
        if not isinstance(revised.get("text"), str) or not isinstance(revised.get("spans"), list):
            raise ValueError(f"Row {index} must have text and spans fields.")
        apply_global_drops(revised)
        for operation in ROW_OPERATIONS.get(index, []):
            apply_operation(revised, operation)
        revised["spans"] = clean_spans(revised)

    synchronize_identical_text_rows(revised_rows)

    audit_rows: list[dict[str, Any]] = []
    for index, (source, revised) in enumerate(zip(source_rows, revised_rows), start=1):
        source_spans = deepcopy(source["spans"])
        source_keys = {span_key(span) for span in source_spans}
        revised_keys = {span_key(span) for span in revised["spans"]}
        if source_keys != revised_keys:
            audit_rows.append(
                {
                    "row_index_1based": index,
                    "text": revised["text"],
                    "source_spans": source_spans,
                    "revised_spans": deepcopy(revised["spans"]),
                    "removed_spans": [
                        {"start": start, "end": end, "label": label, "text": revised["text"][start:end]}
                        for start, end, label in sorted(source_keys - revised_keys)
                    ],
                    "added_spans": [
                        {"start": start, "end": end, "label": label, "text": revised["text"][start:end]}
                        for start, end, label in sorted(revised_keys - source_keys)
                    ],
                }
            )

    validate_rows(revised_rows)
    return revised_rows, audit_rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit_json)
    summary_path = Path(args.summary_json)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --input to preserve the legacy corpus.")

    source_bytes = input_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256 and not args.allow_source_sha256_mismatch:
        raise ValueError(
            "The input SHA-256 does not match the legacy calibration corpus this revision targets. "
            "Use --allow-source-sha256-mismatch only after manually confirming row order and spans."
        )
    source_rows = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError("The legacy calibration corpus must be a JSON array.")

    revised_rows, audit_rows = build_revision(source_rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path.resolve()),
        "source_sha256": source_sha256,
        "output": str(output_path.resolve()),
        "rows_total": len(source_rows),
        "rows_changed": len(audit_rows),
        "source_label_counts": label_counts(source_rows),
        "revised_label_counts": label_counts(revised_rows),
        "source_span_count": sum(len(row.get("spans", [])) for row in source_rows),
        "revised_span_count": sum(len(row.get("spans", [])) for row in revised_rows),
        "guide": "docs/LABELLING_GUIDE.md",
        "audit": str(audit_path.resolve()),
        "revision_scope": (
            "Current-guide revision of known legacy inconsistencies, including generic mentions, "
            "platform mentions, span boundaries, police-unit spans, non-overlap cleanup, and "
            "Organization-versus-Location context corrections."
        ),
        "review_note": (
            "This is an AI-assisted corpus revision. Review the audit in the annotation editor "
            "before treating it as adjudicated gold data."
        ),
    }

    write_json(output_path, revised_rows)
    write_json(audit_path, audit_rows)
    write_json(summary_path, summary)
    print(f"Wrote revised corpus: {output_path}")
    print(f"Wrote audit: {audit_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Rows changed: "
        f"{summary['rows_changed']}/{summary['rows_total']} | "
        f"spans: {summary['source_span_count']} -> {summary['revised_span_count']}"
    )


if __name__ == "__main__":
    main()
