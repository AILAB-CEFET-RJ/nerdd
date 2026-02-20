#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


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
        return _parse_jsonl(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Unsupported JSON format.")


def write_jsonl(items, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def replace_labels(items, source_label, target_label, case_insensitive):
    replaced = 0
    for row in items:
        entities = row.get("entities") if row.get("entities") is not None else row.get("ner")
        if not isinstance(entities, list):
            continue
        for ent in entities:
            label = ent.get("label")
            if not isinstance(label, str):
                continue
            is_match = label.lower() == source_label.lower() if case_insensitive else label == source_label
            if is_match:
                ent["label"] = target_label
                replaced += 1
    return replaced


def parse_args():
    parser = argparse.ArgumentParser(description="Replace entity label values in JSON/JSONL files.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file")
    parser.add_argument("--output", default="", help="Output JSONL file")
    parser.add_argument("--source-label", default="Comunidade")
    parser.add_argument("--target-label", default="Location")
    parser.add_argument("--case-insensitive", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    replaced = replace_labels(rows, args.source_label, args.target_label, args.case_insensitive)
    output = args.input if args.inplace else (args.output or (str(Path(args.input)) + ".fixed.jsonl"))
    write_jsonl(rows, output)
    print(f"[ok] replaced={replaced} output={output}")


if __name__ == "__main__":
    main()

