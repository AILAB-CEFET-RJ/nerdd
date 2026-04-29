#!/usr/bin/env python3
"""Project entities observed in a supervised NER corpus onto pseudolabel records.

The output keeps the same JSONL schema commonly used for pseudolabels:

    {"source_id": ..., "text": ..., "entities": [
        {"start": int, "end": int, "text": str, "label": str, "seed_origin": str}
    ]}

This is intended as a conservative distant-supervision / lexicon-projection utility.
It extracts entity surface forms from the training corpus, filters ambiguous forms,
then adds exact case/accent-insensitive matches to the pseudolabel file.

Example:
    python src/tools/project_train_entities_to_pseudolabels.py \
      --train-path data/dd_corpus_small_train.json \
      --pseudolabel-path artifacts/benchmarks/.../refit_pseudolabels_top20_manual_corrected.jsonl \
      --output-path artifacts/benchmarks/.../refit_pseudolabels_top20_manual_corrected_trainlex.jsonl \
      --report-path artifacts/benchmarks/.../trainlex_projection_report.csv \
      --lexicon-report-path artifacts/benchmarks/.../trainlex_lexicon.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

VALID_LABELS = {"Location", "Organization", "Person"}

# Conservative Portuguese/common-token stop list for single-token candidates.
COMMON_SINGLETONS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "das", "do", "dos", "em", "no", "na", "nos", "nas",
    "por", "para", "pra", "com", "sem", "sob", "sobre", "entre", "ate", "apos",
    "e", "ou", "que", "se", "ao", "aos", "pela", "pelo", "pelos", "pelas",
    "ele", "ela", "eles", "elas", "dele", "dela", "deles", "delas", "nele", "nela",
    "esse", "essa", "esses", "essas", "este", "esta", "estes", "estas", "isso", "isto",
    "tal", "mesmo", "mesma", "local", "casa", "bairro", "rua", "morro", "favela",
    "comunidade", "trafico", "drogas", "policia", "militar", "civil", "batalhao",
    "centro", "campo", "caixa", "ponto", "boca", "fumo", "praca", "avenida",
}

# Short orgs can be meaningful in this corpus. Keep this explicit.
DEFAULT_SHORT_ORG_WHITELIST = {
    "CV", "TCP", "ADA", "PM", "PC", "DP", "DH", "UPP", "BPM", "BOPE",
    "CORE", "DRACO", "GAECO", "PMERJ", "PCERJ", "TJRJ", "INSS",
}

# Single-token Person names/aliases are high risk. This list blocks some common first names
# unless the user explicitly disables it.
COMMON_PERSON_SINGLETONS = {
    "joao", "jose", "maria", "luiz", "luis", "carlos", "marcos", "paulo",
    "pedro", "diego", "igor", "leo", "leandro", "lucas", "thiago", "bruno",
    "rafael", "gabriel", "felipe", "vitor", "victor", "william", "eduardo",
    "cristiano", "aline", "ruan", "natan", "nelson", "marcelo", "jessica",
}

LOCATION_PREFIXES = (
    "rua", "r", "avenida", "av", "estrada", "travessa", "praca", "praça",
    "rodovia", "br", "morro", "comunidade", "favela", "complexo", "bairro",
    "vila", "jardim", "parque", "condominio", "condomínio", "beco",
)


@dataclass(frozen=True)
class LexiconEntry:
    label: str
    surface: str
    norm: str
    freq: int
    examples: tuple[str, ...]


@dataclass
class Candidate:
    start: int
    end: int
    text: str
    label: str
    seed_origin: str
    lexicon_surface: str
    lexicon_freq: int


def strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def normalize_for_match(text: str) -> str:
    # Preserve string length as much as possible for char-offset projection. Do not collapse spaces here.
    return strip_accents(text).lower()


def normalize_key(text: str) -> str:
    text = strip_accents(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_count(text: str) -> int:
    return len(re.findall(r"[\wÀ-ÿ]+", text, flags=re.UNICODE))


def is_single_token(text: str) -> bool:
    return token_count(text) == 1


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data
    rows = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_entities_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return entities from either supervised {spans} or pseudolabel {entities}."""
    text = str(row.get("text", ""))
    entities: list[dict[str, Any]] = []

    if isinstance(row.get("spans"), list):
        for span in row["spans"]:
            try:
                start = int(span["start"])
                end = int(span["end"])
                label = str(span["label"])
            except Exception:
                continue
            if label not in VALID_LABELS or start < 0 or end <= start or end > len(text):
                continue
            entities.append({"start": start, "end": end, "text": text[start:end], "label": label})

    if isinstance(row.get("entities"), list):
        for ent in row["entities"]:
            try:
                start = int(ent["start"])
                end = int(ent["end"])
                label = str(ent["label"])
            except Exception:
                continue
            if label not in VALID_LABELS or start < 0 or end <= start or end > len(text):
                continue
            ent_text = str(ent.get("text", text[start:end]))
            # Prefer offsets as source of truth but keep only valid aligned spans.
            if text[start:end] != ent_text:
                ent_text = text[start:end]
            entities.append({"start": start, "end": end, "text": ent_text, "label": label})

    return entities


def candidate_is_allowed(
    *,
    label: str,
    surface: str,
    freq: int,
    min_chars: int,
    min_freq: int,
    allow_person_singletons: bool,
    person_singleton_min_freq: int,
    person_singleton_min_chars: int,
    disable_common_person_singleton_blocklist: bool,
    short_org_whitelist: set[str],
) -> tuple[bool, str]:
    surface_clean = re.sub(r"\s+", " ", surface.strip())
    norm = normalize_key(surface_clean)
    tokens = token_count(surface_clean)

    if freq < min_freq:
        return False, "below_min_freq"
    if len(norm) < min_chars:
        return False, "below_min_chars"
    if not re.search(r"[a-z0-9]", norm):
        return False, "no_alnum"
    if norm in COMMON_SINGLETONS:
        return False, "common_singleton"

    if label == "Organization":
        if tokens == 1 and len(norm) <= 3 and surface_clean.upper() not in short_org_whitelist:
            return False, "short_org_not_whitelisted"
        return True, "ok"

    if label == "Location":
        # Keep single-token locations if they are not too short/common. They are frequent in the corpus.
        if tokens == 1 and len(norm) < 4:
            return False, "short_location_singleton"
        return True, "ok"

    if label == "Person":
        if tokens >= 2:
            return True, "ok"
        if not allow_person_singletons:
            return False, "person_singleton_blocked"
        if len(norm) < person_singleton_min_chars:
            return False, "person_singleton_below_min_chars"
        if freq < person_singleton_min_freq:
            return False, "person_singleton_below_min_freq"
        if (not disable_common_person_singleton_blocklist) and norm in COMMON_PERSON_SINGLETONS:
            return False, "common_person_singleton"
        return True, "ok"

    return False, "invalid_label"


def build_lexicon(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[LexiconEntry], list[dict[str, Any]]]:
    counts: Counter[tuple[str, str]] = Counter()
    surfaces: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    examples: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []

    enabled_labels = set(args.labels)
    for row_idx, row in enumerate(rows):
        text = str(row.get("text", ""))
        for ent in extract_entities_from_row(row):
            label = ent["label"]
            if label not in enabled_labels:
                continue
            surface = re.sub(r"\s+", " ", str(ent["text"]).strip())
            norm = normalize_key(surface)
            if not norm:
                continue
            key = (label, norm)
            counts[key] += 1
            surfaces[key][surface] += 1
            if len(examples[key]) < args.max_examples_per_lexicon_entry:
                examples[key].append(text[:200].replace("\n", " "))

    short_org_whitelist = {item.strip().upper() for item in args.short_org_whitelist.split(",") if item.strip()}
    entries: list[LexiconEntry] = []
    for (label, norm), freq in counts.items():
        surface = surfaces[(label, norm)].most_common(1)[0][0]
        allowed, reason = candidate_is_allowed(
            label=label,
            surface=surface,
            freq=freq,
            min_chars=args.min_chars,
            min_freq=args.min_freq,
            allow_person_singletons=args.allow_person_singletons,
            person_singleton_min_freq=args.person_singleton_min_freq,
            person_singleton_min_chars=args.person_singleton_min_chars,
            disable_common_person_singleton_blocklist=args.disable_common_person_singleton_blocklist,
            short_org_whitelist=short_org_whitelist,
        )
        if allowed:
            entries.append(
                LexiconEntry(
                    label=label,
                    surface=surface,
                    norm=norm,
                    freq=freq,
                    examples=tuple(examples[(label, norm)]),
                )
            )
        else:
            rejected.append({"label": label, "surface": surface, "norm": norm, "freq": freq, "reason": reason})

    # Longest first helps prevent shorter candidates from blocking longer additions.
    entries.sort(key=lambda e: (len(e.norm), e.freq), reverse=True)
    return entries, rejected


def make_pattern(norm_phrase: str) -> re.Pattern[str]:
    # Match normalized phrase in normalized text. Spaces in the phrase match any whitespace run.
    parts = [re.escape(part) for part in norm_phrase.split(" ") if part]
    pattern = r"\s+".join(parts)
    return re.compile(pattern)


def has_word_boundaries(norm_text: str, start: int, end: int) -> bool:
    if start > 0 and (norm_text[start - 1].isalnum() or norm_text[start - 1] == "_"):
        return False
    if end < len(norm_text) and (norm_text[end].isalnum() or norm_text[end] == "_"):
        return False
    return True


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def existing_entity_key(ent: dict[str, Any]) -> tuple[int, int, str]:
    return int(ent["start"]), int(ent["end"]), str(ent["label"])


def project_lexicon_to_row(row: dict[str, Any], lexicon: list[LexiconEntry], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    text = str(row.get("text", ""))
    norm_text = normalize_for_match(text)

    existing = list(row.get("entities", []) or [])
    cleaned_existing = []
    for ent in existing:
        try:
            start = int(ent["start"])
            end = int(ent["end"])
            label = str(ent["label"])
        except Exception:
            continue
        if label not in VALID_LABELS or start < 0 or end <= start or end > len(text):
            continue
        ent = dict(ent)
        ent["start"] = start
        ent["end"] = end
        ent["text"] = text[start:end]
        ent["label"] = label
        cleaned_existing.append(ent)

    occupied = [(int(e["start"]), int(e["end"]), str(e["label"]), "existing") for e in cleaned_existing]
    present_keys = {existing_entity_key(e) for e in cleaned_existing}
    additions: list[Candidate] = []
    conflicts: list[dict[str, Any]] = []

    for entry in lexicon:
        pattern = make_pattern(entry.norm)
        for match in pattern.finditer(norm_text):
            start, end = match.span()
            if start == end:
                continue
            if not has_word_boundaries(norm_text, start, end):
                continue
            matched_text = text[start:end]
            if not matched_text.strip():
                continue
            candidate_key = (start, end, entry.label)
            if candidate_key in present_keys:
                continue

            overlap_items = [item for item in occupied if spans_overlap(start, end, item[0], item[1])]
            if overlap_items:
                conflicts.append({
                    "source_id": row.get("source_id", ""),
                    "candidate_text": matched_text,
                    "candidate_label": entry.label,
                    "candidate_start": start,
                    "candidate_end": end,
                    "lexicon_surface": entry.surface,
                    "lexicon_freq": entry.freq,
                    "reason": "overlap_existing_or_added",
                    "overlaps": ";".join(f"{s}-{e}:{lab}:{kind}" for s, e, lab, kind in overlap_items),
                })
                continue

            cand = Candidate(
                start=start,
                end=end,
                text=matched_text,
                label=entry.label,
                seed_origin=args.seed_origin,
                lexicon_surface=entry.surface,
                lexicon_freq=entry.freq,
            )
            additions.append(cand)
            occupied.append((start, end, entry.label, "projected"))
            present_keys.add(candidate_key)

    output_entities = cleaned_existing + [
        {
            "start": cand.start,
            "end": cand.end,
            "text": cand.text,
            "label": cand.label,
            "seed_origin": cand.seed_origin,
        }
        for cand in additions
    ]
    output_entities.sort(key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"])))

    output_row = dict(row)
    output_row["text"] = text
    output_row["entities"] = output_entities

    report_rows = [
        {
            "source_id": row.get("source_id", ""),
            "start": cand.start,
            "end": cand.end,
            "text": cand.text,
            "label": cand.label,
            "seed_origin": cand.seed_origin,
            "lexicon_surface": cand.lexicon_surface,
            "lexicon_freq": cand.lexicon_freq,
        }
        for cand in additions
    ]
    return output_row, report_rows, conflicts


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project train-corpus NER entities onto a pseudolabel JSONL file."
    )
    parser.add_argument("--train-path", required=True, help="Supervised train JSON/JSONL with text + spans/entities.")
    parser.add_argument("--pseudolabel-path", required=True, help="Input pseudolabel JSONL with text + entities.")
    parser.add_argument("--output-path", required=True, help="Output JSONL path, same schema as pseudolabel input.")
    parser.add_argument("--report-path", default="", help="CSV report with projected entities.")
    parser.add_argument("--lexicon-report-path", default="", help="CSV report with retained lexicon entries.")
    parser.add_argument("--rejected-lexicon-report-path", default="", help="CSV report with rejected lexicon entries.")
    parser.add_argument("--conflicts-report-path", default="", help="CSV report with candidate additions rejected due to overlaps.")
    parser.add_argument("--labels", nargs="+", default=sorted(VALID_LABELS), choices=sorted(VALID_LABELS))
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum frequency in train corpus for a lexicon entry.")
    parser.add_argument("--min-chars", type=int, default=3, help="Minimum normalized character length.")
    parser.add_argument("--allow-person-singletons", action="store_true", help="Allow one-token Person entries subject to extra filters.")
    parser.add_argument("--person-singleton-min-freq", type=int, default=2)
    parser.add_argument("--person-singleton-min-chars", type=int, default=5)
    parser.add_argument("--disable-common-person-singleton-blocklist", action="store_true")
    parser.add_argument("--short-org-whitelist", default=",".join(sorted(DEFAULT_SHORT_ORG_WHITELIST)))
    parser.add_argument("--seed-origin", default="train_lexicon_projection")
    parser.add_argument("--max-examples-per-lexicon-entry", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Do not write JSONL output; only reports/stats.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_path = Path(args.train_path)
    pseudo_path = Path(args.pseudolabel_path)
    output_path = Path(args.output_path)

    train_rows = read_json_or_jsonl(train_path)
    pseudo_rows = read_json_or_jsonl(pseudo_path)

    lexicon, rejected = build_lexicon(train_rows, args)

    output_rows: list[dict[str, Any]] = []
    projection_report: list[dict[str, Any]] = []
    conflict_report: list[dict[str, Any]] = []

    for row in pseudo_rows:
        out, added, conflicts = project_lexicon_to_row(row, lexicon, args)
        output_rows.append(out)
        projection_report.extend(added)
        conflict_report.extend(conflicts)

    if not args.dry_run:
        write_jsonl(output_path, output_rows)

    if args.report_path:
        write_csv(
            Path(args.report_path),
            projection_report,
            ["source_id", "start", "end", "text", "label", "seed_origin", "lexicon_surface", "lexicon_freq"],
        )

    if args.lexicon_report_path:
        lex_rows = [
            {
                "label": entry.label,
                "surface": entry.surface,
                "norm": entry.norm,
                "freq": entry.freq,
                "examples": " || ".join(entry.examples),
            }
            for entry in lexicon
        ]
        write_csv(Path(args.lexicon_report_path), lex_rows, ["label", "surface", "norm", "freq", "examples"])

    if args.rejected_lexicon_report_path:
        write_csv(Path(args.rejected_lexicon_report_path), rejected, ["label", "surface", "norm", "freq", "reason"])

    if args.conflicts_report_path:
        write_csv(
            Path(args.conflicts_report_path),
            conflict_report,
            [
                "source_id", "candidate_text", "candidate_label", "candidate_start", "candidate_end",
                "lexicon_surface", "lexicon_freq", "reason", "overlaps",
            ],
        )

    added_by_label = Counter(row["label"] for row in projection_report)
    lex_by_label = Counter(entry.label for entry in lexicon)
    print("[train-lexicon-projection]")
    print(f"train_rows={len(train_rows)}")
    print(f"pseudolabel_rows={len(pseudo_rows)}")
    print(f"lexicon_entries={len(lexicon)} {dict(sorted(lex_by_label.items()))}")
    print(f"added_entities={len(projection_report)} {dict(sorted(added_by_label.items()))}")
    print(f"overlap_conflicts={len(conflict_report)}")
    if not args.dry_run:
        print(f"output={output_path}")
    if args.report_path:
        print(f"report={args.report_path}")
    if args.lexicon_report_path:
        print(f"lexicon_report={args.lexicon_report_path}")
    if args.rejected_lexicon_report_path:
        print(f"rejected_lexicon_report={args.rejected_lexicon_report_path}")
    if args.conflicts_report_path:
        print(f"conflicts_report={args.conflicts_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
