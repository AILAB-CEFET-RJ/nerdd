#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gliner_loader import load_gliner_model
from tools.build_calibration_dataset import predict_entities_for_text
from tools.inspect_dense_tips import get_spans, get_text, read_json_or_jsonl, write_jsonl
from tools.render_ner_html import build_html, build_label_colors, render_text_with_spans, sanitize_spans


def _parse_csv(raw_value):
    if not raw_value:
        return []
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def build_comparison_rows(rows, model, labels, batch_size, max_tokens, threshold):
    compared = []
    for idx, row in enumerate(rows, start=1):
        text = get_text(row)
        baseline_entities = get_spans(row)
        model_entities = predict_entities_for_text(
            model=model,
            text=text,
            labels=labels,
            batch_size=batch_size,
            max_tokens=max_tokens,
            threshold=threshold,
        )
        compared.append(
            {
                "text": text,
                "baseline_entities": baseline_entities,
                "model_entities": model_entities,
                "_comparison_meta": {
                    "row_index_1based": idx,
                    "baseline_entity_count": len(baseline_entities),
                    "model_entity_count": len(model_entities),
                },
                "_source": row,
            }
        )
    return compared


def build_summary(rows):
    baseline_counts = Counter()
    model_counts = Counter()
    for row in rows:
        for span in row["baseline_entities"]:
            label = span.get("label")
            if label:
                baseline_counts[str(label)] += 1
        for span in row["model_entities"]:
            label = span.get("label")
            if label:
                model_counts[str(label)] += 1
    return {
        "records": len(rows),
        "baseline_total_spans": int(sum(baseline_counts.values())),
        "model_total_spans": int(sum(model_counts.values())),
        "baseline_label_counts": dict(sorted(baseline_counts.items())),
        "model_label_counts": dict(sorted(model_counts.items())),
    }


def render_comparison_html(rows, output_path, title, right_model_name):
    baseline_label_rows = [{"entities": row["baseline_entities"]} for row in rows]
    model_label_rows = [{"entities": row["model_entities"]} for row in rows]
    label_colors = build_label_colors(baseline_label_rows + model_label_rows)

    baseline_counts = Counter()
    model_counts = Counter()
    rendered_rows = []
    for row in rows:
        text = row["text"]
        baseline_spans = sanitize_spans(text, row["baseline_entities"])
        model_spans = sanitize_spans(text, row["model_entities"])
        for span in baseline_spans:
            baseline_counts[span["label"]] += 1
        for span in model_spans:
            model_counts[span["label"]] += 1
        meta = row["_comparison_meta"]
        rendered_rows.append(
            "<section class='report'>"
            f"<h3>Record #{meta['row_index_1based']}</h3>"
            f"<div class='meta'>baseline_entities={meta['baseline_entity_count']} | comparison_entities={meta['model_entity_count']}</div>"
            "<div class='meta'><b>Baseline / existing predictions</b></div>"
            f"<div class='text'>{render_text_with_spans(text, baseline_spans, label_colors)}</div>"
            f"<div class='meta' style='margin-top:8px;'><b>{right_model_name}</b></div>"
            f"<div class='text'>{render_text_with_spans(text, model_spans, label_colors)}</div>"
            "</section>"
        )

    label_union = sorted(set(baseline_counts) | set(model_counts))
    summary_html = (
        "<table class='summary'>"
        f"<tr><td><b>Records</b></td><td>{len(rows)}</td></tr>"
        f"<tr><td><b>Baseline spans</b></td><td>{sum(baseline_counts.values())}</td></tr>"
        f"<tr><td><b>Comparison spans</b></td><td>{sum(model_counts.values())}</td></tr>"
        "</table>"
    )
    legend_rows = []
    for label in label_union:
        color = label_colors.get(label, "#444444")
        legend_rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td><span class='swatch' style='background:{color}'></span></td>"
            f"<td>{baseline_counts.get(label, 0)}</td>"
            f"<td>{model_counts.get(label, 0)}</td>"
            "</tr>"
        )
    legend_html = (
        "<table class='legend'><tr><th>Label</th><th>Color</th><th>Baseline</th><th>Comparison</th></tr>{}</table>".format(
            "".join(legend_rows)
        )
        if legend_rows
        else ""
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_html(title, "".join(rendered_rows), legend_html, summary_html), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare existing predictions against a GLiNER model on the same texts.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL. Existing predictions should be in entities/spans/ner.")
    parser.add_argument("--model-path", required=True, help="GLiNER model path or HF id to compare against.")
    parser.add_argument("--labels", default="Person,Location,Organization", help="Comma-separated labels for GLiNER inference.")
    parser.add_argument("--batch-size", type=int, default=8, help="Prediction batch size.")
    parser.add_argument("--max-tokens", type=int, default=384, help="Chunk size for GLiNER inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold for GLiNER inference.")
    parser.add_argument("--model-max-length", type=int, default=0, help="Optional GLiNER max_length.")
    parser.add_argument("--map-location", default="", help="Optional GLiNER device, e.g. cuda.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with baseline vs model entities.")
    parser.add_argument("--output-html", required=True, help="Output HTML with side-by-side comparison.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--title", default="Baseline vs GLiNER comparison", help="HTML title.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional limit of input records to compare.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    labels = _parse_csv(args.labels)
    if not labels:
        raise ValueError("At least one label must be provided.")

    model = load_gliner_model(
        args.model_path,
        model_max_length=args.model_max_length,
        map_location=args.map_location,
        context="comparison",
    )
    compared_rows = build_comparison_rows(
        rows,
        model=model,
        labels=labels,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        threshold=args.threshold,
    )
    write_jsonl(args.output_jsonl, compared_rows)
    render_comparison_html(compared_rows, args.output_html, args.title, right_model_name=args.model_path)
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
    print(f"Comparison spans: {summary['model_total_spans']}")


if __name__ == "__main__":
    main()
