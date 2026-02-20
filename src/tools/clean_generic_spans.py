#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path


DEFAULT_BANLIST = {
    "Person": [
        "cidadao",
        "cidadaos",
        "morador",
        "moradores",
        "pessoa",
        "pessoas",
        "envolvido",
        "envolvidos",
    ],
    "Location": [
        "casa",
        "casas",
        "local",
    ],
}


def _parse_jsonl(text):
    rows = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc
    return rows


def read_json_or_jsonl(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _parse_jsonl(text), "jsonl"
    if isinstance(obj, list):
        return obj, "json"
    if isinstance(obj, dict):
        return [obj], "json"
    raise ValueError("Unsupported JSON format.")


def write_rows(rows, path, fmt):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(value):
    value = str(value or "").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.split())


def get_text(record):
    for key in ("text", "relato", "texto", "description", "descricao"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def iter_span_lists(record):
    for key in ("spans", "entities", "ner"):
        spans = record.get(key)
        if isinstance(spans, list):
            yield key, spans


def extract_span_text(record_text, span):
    start = span.get("start")
    end = span.get("end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(record_text):
        return record_text[start:end]
    mention = span.get("text")
    if isinstance(mention, str):
        return mention
    return ""


def load_banlist(config_path):
    if not config_path:
        source = DEFAULT_BANLIST
    else:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Config must be a JSON object: {label: [terms...]}")
        source = payload

    result = {}
    for label, terms in source.items():
        if not isinstance(label, str):
            continue
        if not isinstance(terms, list):
            continue
        label_norm = normalize_text(label)
        values = {normalize_text(t) for t in terms if isinstance(t, str) and normalize_text(t)}
        if values:
            result[label_norm] = values
    return result


def clean_rows(rows, banlist, show_examples=10):
    removed_records = []
    removed_by_label = Counter()
    removed_by_term = Counter()
    examples = []

    for rec_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        record_text = get_text(row)
        for span_key, spans in iter_span_lists(row):
            kept = []
            for span in spans:
                if not isinstance(span, dict):
                    kept.append(span)
                    continue
                label_raw = span.get("label")
                if not isinstance(label_raw, str):
                    kept.append(span)
                    continue
                label_norm = normalize_text(label_raw)
                banned_terms = banlist.get(label_norm)
                if not banned_terms:
                    kept.append(span)
                    continue

                mention = extract_span_text(record_text, span)
                mention_norm = normalize_text(mention)
                if not mention_norm or mention_norm not in banned_terms:
                    kept.append(span)
                    continue

                removed_by_label[label_raw] += 1
                removed_by_term[(label_raw, mention_norm)] += 1
                removed_records.append(
                    {
                        "record_index": rec_idx,
                        "span_key": span_key,
                        "label": label_raw,
                        "mention": mention,
                        "mention_normalized": mention_norm,
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "reason": "generic_term_banlist",
                    }
                )
                if len(examples) < show_examples:
                    examples.append(
                        f"record={rec_idx} key={span_key} label={label_raw} mention='{mention}'"
                    )

            row[span_key] = kept

    return removed_records, removed_by_label, removed_by_term, examples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove generic NER spans by exact text match (normalized)."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file.")
    parser.add_argument(
        "--output",
        default="",
        help="Output file path. Required unless --dry-run or --inplace.",
    )
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file.")
    parser.add_argument(
        "--config",
        default="",
        help="Optional JSON config with banlist: {label: [terms...]}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write cleaned dataset.")
    parser.add_argument(
        "--removed-report",
        default="",
        help="Optional JSONL path for removed spans report.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "json", "jsonl"),
        default="auto",
        help="Output format (default: match input).",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=10,
        help="Number of removed-span examples to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, input_fmt = read_json_or_jsonl(args.input)
    banlist = load_banlist(args.config)

    removed_records, removed_by_label, removed_by_term, examples = clean_rows(
        rows, banlist, show_examples=max(0, args.show_examples)
    )

    total_removed = len(removed_records)
    print(f"[info] total_removed={total_removed}")
    if removed_by_label:
        print("[info] removed_by_label:")
        for label, count in sorted(removed_by_label.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {label}: {count}")
    if removed_by_term:
        print("[info] top_removed_terms:")
        for (label, term), count in sorted(
            removed_by_term.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
        )[:20]:
            print(f"  - {label}::{term} -> {count}")
    if examples:
        print("[info] examples:")
        for row in examples:
            print(f"  - {row}")

    if args.removed_report:
        write_rows(removed_records, args.removed_report, "jsonl")
        print(f"[ok] removed report written: {args.removed_report}")

    if args.dry_run:
        print("[ok] dry-run completed (dataset not written).")
        return

    if args.inplace and args.output:
        raise ValueError("Use either --inplace or --output, not both.")

    output_path = args.input if args.inplace else args.output
    if not output_path:
        raise ValueError("Provide --output (or use --inplace / --dry-run).")

    output_fmt = input_fmt if args.format == "auto" else args.format
    write_rows(rows, output_path, output_fmt)
    print(f"[ok] cleaned dataset written: {output_path}")


if __name__ == "__main__":
    main()
