#!/usr/bin/env python3
"""Render adjudication cases with baseline, GLiNER2, seeds, and optional final entities."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl
from tools.render_ner_html import (
    DEFAULT_SCORE_FIELDS,
    build_html,
    build_label_colors,
    get_text,
    render_entity_list,
    render_text_with_spans,
    sanitize_spans,
)


DEFAULT_LAYERS = (
    ("baseline_entities", "Baseline"),
    ("gliner2_entities", "GLiNER2"),
    ("review_seed_entities", "Review Seeds"),
    ("adjudication.entities_final", "Adjudicated Final"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render adjudication cases as a side-by-side HTML report with multiple entity layers."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL with adjudication-case rows.")
    parser.add_argument("--output", required=True, help="Output HTML file.")
    parser.add_argument("--title", default="Adjudication Case Review", help="HTML title.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional limit of rendered rows.")
    parser.add_argument(
        "--layers",
        default=",".join(field for field, _ in DEFAULT_LAYERS),
        help=(
            "Comma-separated layer field paths to render. "
            "Supported examples: baseline_entities, gliner2_entities, review_seed_entities, adjudication.entities_final."
        ),
    )
    parser.add_argument(
        "--score-fields",
        default=",".join(DEFAULT_SCORE_FIELDS),
        help="Comma-separated score fields to display per entity in fallback order.",
    )
    return parser.parse_args()


def _parse_csv(raw_value: str) -> list[str]:
    return [piece.strip() for piece in str(raw_value or "").split(",") if piece.strip()]


def _resolve_field_path(record: dict, field_path: str):
    current = record
    for part in str(field_path).split("."):
        if not isinstance(current, dict):
            current = None
            break
        current = current.get(part)
    if current is not None:
        return current

    source = record.get("_source")
    if not isinstance(source, dict):
        return None

    current = source
    for part in str(field_path).split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _record_text(record: dict) -> str:
    text = get_text(record)
    if text:
        return text
    source = record.get("_source")
    if isinstance(source, dict):
        return get_text(source)
    return ""


def _layer_title(field_path: str) -> str:
    for known_field, title in DEFAULT_LAYERS:
        if field_path == known_field:
            return title
    return field_path.replace(".", " / ")


def _layer_spans(record: dict, field_path: str) -> list[dict]:
    value = _resolve_field_path(record, field_path)
    return value if isinstance(value, list) else []


def _record_meta_bits(record: dict) -> list[str]:
    bits = []
    source_id = str(record.get("source_id", "") or ((record.get("_source") or {}).get("source_id", ""))).strip()
    if source_id:
        bits.append(f"source_id={escape(source_id)}")

    adjudication = record.get("adjudication")
    if isinstance(adjudication, dict):
        decision = str(adjudication.get("decision", "")).strip()
        review_confidence = str(adjudication.get("review_confidence", "")).strip()
        if decision:
            bits.append(f"decision={escape(decision)}")
        if review_confidence:
            bits.append(f"review_confidence={escape(review_confidence)}")

    priority = record.get("adjudication_priority_score")
    if isinstance(priority, (int, float)):
        bits.append(f"adjudication_priority_score={priority:.6f}")

    record_score = record.get("record_score")
    if isinstance(record_score, (int, float)):
        bits.append(f"record_score={record_score:.6f}")

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("agreement_ratio", "baseline_coverage_proxy", "gliner2_noise_proxy"):
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                bits.append(f"{key}={value:.6f}")

    return bits


def _build_summary(rows: list[dict], layers: list[str]) -> tuple[str, str]:
    label_rows = []
    counts_by_layer = {layer: Counter() for layer in layers}

    for row in rows:
        text = _record_text(row)
        for layer in layers:
            spans = sanitize_spans(text, _layer_spans(row, layer))
            label_rows.append({"entities": spans})
            for span in spans:
                counts_by_layer[layer][span["label"]] += 1

    label_colors = build_label_colors(label_rows)
    layer_totals = {layer: int(sum(counter.values())) for layer, counter in counts_by_layer.items()}
    summary_html = (
        "<table class='summary'>"
        f"<tr><td><b>Records</b></td><td>{len(rows)}</td></tr>"
        + "".join(
            f"<tr><td><b>{escape(_layer_title(layer))} spans</b></td><td>{layer_totals[layer]}</td></tr>"
            for layer in layers
        )
        + "</table>"
    )

    label_union = sorted({label for counter in counts_by_layer.values() for label in counter})
    if not label_union:
        return summary_html, ""

    header = "".join(f"<th>{escape(_layer_title(layer))}</th>" for layer in layers)
    rows_html = []
    for label in label_union:
        color = label_colors.get(label, "#444444")
        cols = "".join(f"<td>{counts_by_layer[layer].get(label, 0)}</td>" for layer in layers)
        rows_html.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td><span class='swatch' style='background:{color}'></span></td>"
            f"{cols}"
            "</tr>"
        )
    legend_html = f"<table class='legend'><tr><th>Label</th><th>Color</th>{header}</tr>{''.join(rows_html)}</table>"
    return summary_html, legend_html


def render_adjudication_review(
    rows: list[dict],
    *,
    output_path: str,
    title: str,
    layers: list[str],
    score_fields: list[str],
    max_records: int = 0,
) -> None:
    if max_records > 0:
        rows = rows[:max_records]

    label_rows = []
    for row in rows:
        text = _record_text(row)
        for layer in layers:
            label_rows.append({"entities": sanitize_spans(text, _layer_spans(row, layer))})
    label_colors = build_label_colors(label_rows)
    summary_html, legend_html = _build_summary(rows, layers)

    rendered_rows = []
    for idx, row in enumerate(rows, start=1):
        text = _record_text(row)
        meta_bits = _record_meta_bits(row)
        sections = []
        for layer in layers:
            spans = sanitize_spans(text, _layer_spans(row, layer))
            sections.append(
                "<div class='meta' style='margin-top:8px;'><b>{}</b> | entities={}</div>{}<div class='text'>{}</div>".format(
                    escape(_layer_title(layer)),
                    len(spans),
                    render_entity_list(text, spans, score_fields=score_fields),
                    render_text_with_spans(text, spans, label_colors, score_fields=score_fields),
                )
            )

        rendered_rows.append(
            "<section class='report'>"
            f"<h3>Record #{idx}</h3>"
            f"<div class='meta'>{' | '.join(meta_bits) if meta_bits else 'No extra metadata.'}</div>"
            + "".join(sections)
            + "</section>"
        )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_html(title, "".join(rendered_rows), legend_html, summary_html), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    render_adjudication_review(
        rows,
        output_path=args.output,
        title=args.title,
        layers=_parse_csv(args.layers),
        score_fields=_parse_csv(args.score_fields),
        max_records=args.max_records,
    )
    print(f"[ok] HTML saved to: {args.output}")


if __name__ == "__main__":
    main()
