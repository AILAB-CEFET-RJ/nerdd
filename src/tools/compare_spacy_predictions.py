#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import get_spans, get_text, read_json_or_jsonl, write_jsonl
from tools.render_ner_html import build_html, build_label_colors, render_text_with_spans, sanitize_spans


DEFAULT_LABEL_MAP = {
    "PERSON": "Person",
    "PER": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
}


def parse_label_map(raw_value):
    mapping = dict(DEFAULT_LABEL_MAP)
    if not raw_value:
        return mapping
    for item in [piece.strip() for piece in raw_value.split(",") if piece.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid label map entry: {item}. Expected SOURCE=TARGET.")
        source, target = item.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Invalid label map entry: {item}. Expected SOURCE=TARGET.")
        mapping[source] = target
    return mapping


def map_spacy_entities(doc, label_map):
    entities = []
    for ent in doc.ents:
        mapped = label_map.get(ent.label_)
        if not mapped:
            continue
        entities.append(
            {
                "start": int(ent.start_char),
                "end": int(ent.end_char),
                "label": mapped,
                "text": ent.text,
            }
        )
    return entities


def build_comparison_rows(rows, nlp, label_map):
    compared = []
    for idx, row in enumerate(rows, start=1):
        text = get_text(row)
        doc = nlp(text)
        baseline_entities = get_spans(row)
        spacy_entities = map_spacy_entities(doc, label_map)
        compared.append(
            {
                "text": text,
                "baseline_entities": baseline_entities,
                "spacy_entities": spacy_entities,
                "_comparison_meta": {
                    "row_index_1based": idx,
                    "baseline_entity_count": len(baseline_entities),
                    "spacy_entity_count": len(spacy_entities),
                },
                "_source": row,
            }
        )
    return compared


def build_summary(rows):
    baseline_counts = Counter()
    spacy_counts = Counter()
    for row in rows:
        for span in row["baseline_entities"]:
            label = span.get("label")
            if label:
                baseline_counts[str(label)] += 1
        for span in row["spacy_entities"]:
            label = span.get("label")
            if label:
                spacy_counts[str(label)] += 1
    return {
        "records": len(rows),
        "baseline_total_spans": int(sum(baseline_counts.values())),
        "spacy_total_spans": int(sum(spacy_counts.values())),
        "baseline_label_counts": dict(sorted(baseline_counts.items())),
        "spacy_label_counts": dict(sorted(spacy_counts.items())),
    }


def render_comparison_html(rows, output_path, title):
    baseline_label_rows = [{"entities": row["baseline_entities"]} for row in rows]
    spacy_label_rows = [{"entities": row["spacy_entities"]} for row in rows]
    label_colors = build_label_colors(baseline_label_rows + spacy_label_rows)

    baseline_counts = Counter()
    spacy_counts = Counter()
    rendered_rows = []
    for row in rows:
        text = row["text"]
        baseline_spans = sanitize_spans(text, row["baseline_entities"])
        spacy_spans = sanitize_spans(text, row["spacy_entities"])
        for span in baseline_spans:
            baseline_counts[span["label"]] += 1
        for span in spacy_spans:
            spacy_counts[span["label"]] += 1
        meta = row["_comparison_meta"]
        rendered_rows.append(
            "<section class='report'>"
            f"<h3>Record #{meta['row_index_1based']}</h3>"
            f"<div class='meta'>baseline_entities={meta['baseline_entity_count']} | spacy_entities={meta['spacy_entity_count']}</div>"
            "<div class='meta'><b>Baseline</b></div>"
            f"<div class='text'>{render_text_with_spans(text, baseline_spans, label_colors)}</div>"
            "<div class='meta' style='margin-top:8px;'><b>spaCy</b></div>"
            f"<div class='text'>{render_text_with_spans(text, spacy_spans, label_colors)}</div>"
            "</section>"
        )

    label_union = sorted(set(baseline_counts) | set(spacy_counts))
    summary_html = (
        "<table class='summary'>"
        f"<tr><td><b>Records</b></td><td>{len(rows)}</td></tr>"
        f"<tr><td><b>Baseline spans</b></td><td>{sum(baseline_counts.values())}</td></tr>"
        f"<tr><td><b>spaCy spans</b></td><td>{sum(spacy_counts.values())}</td></tr>"
        "</table>"
    )
    legend_rows = []
    for label in label_union:
        color = label_colors.get(label, "#444444")
        legend_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td><span class='swatch' style='background:{color}'></span></td>"
            f"<td>{baseline_counts.get(label, 0)}</td>"
            f"<td>{spacy_counts.get(label, 0)}</td>"
            "</tr>"
        )
    legend_html = (
        "<table class='legend'><tr><th>Label</th><th>Color</th><th>Baseline</th><th>spaCy</th></tr>{}</table>".format(
            "".join(legend_rows)
        )
        if legend_rows
        else ""
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_html(title, "".join(rendered_rows), legend_html, summary_html), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare existing predictions against spaCy predictions on the same texts.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL. Existing predictions should be in entities/spans/ner.")
    parser.add_argument("--model", default="pt_core_news_sm", help="spaCy model name to load.")
    parser.add_argument("--label-map", default="", help="Optional comma-separated spaCy-to-target label map, e.g. PERSON=Person,ORG=Organization,GPE=Location.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with baseline vs spaCy entities.")
    parser.add_argument("--output-html", required=True, help="Output HTML with side-by-side comparison.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--title", default="Baseline vs spaCy comparison", help="HTML title.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional limit of input records to compare.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("spaCy is not installed. Install it first, then rerun this tool.") from exc

    rows = read_json_or_jsonl(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    nlp = spacy.load(args.model)
    compared_rows = build_comparison_rows(rows, nlp, parse_label_map(args.label_map))
    write_jsonl(args.output_jsonl, compared_rows)
    render_comparison_html(compared_rows, args.output_html, args.title)
    summary = build_summary(compared_rows)

    print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Saved HTML: {args.output_html}")

    if args.summary_json:
        target = Path(args.summary_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved summary JSON: {args.summary_json}")

    print(f"Records compared: {summary['records']}")
    print(f"Baseline spans: {summary['baseline_total_spans']}")
    print(f"spaCy spans: {summary['spacy_total_spans']}")


if __name__ == "__main__":
    main()
