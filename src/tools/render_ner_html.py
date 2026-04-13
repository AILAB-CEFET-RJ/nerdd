#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render NER spans as standalone HTML snippets and reports."""

import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path
from string import Template


PALETTE = [
    "#0B6E4F",
    "#1D4E89",
    "#8B1E3F",
    "#8C510A",
    "#5C2E91",
    "#006D77",
    "#264653",
    "#7A3E00",
]
DEFAULT_SCORE_FIELDS = [
    "ner_score",
    "score_context_boosted",
    "score_calibrated",
    "score",
    "confidence",
    "probability",
]
DEFAULT_RECORD_SCORE_FIELDS = [
    "adjudication_priority_score",
    "record_score",
    "score",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a labeled NER corpus (JSON/JSONL) into an HTML report."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input corpus file (JSON list/object or JSONL).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output HTML file.",
    )
    parser.add_argument(
        "--title",
        default="NER Annotation Viewer",
        help="HTML report title.",
    )
    parser.add_argument(
        "--max-reports",
        type=int,
        default=0,
        help="Maximum number of records to render (0 = all).",
    )
    parser.add_argument(
        "--span-field",
        default="auto",
        help=(
            "Which span list to render. Use 'auto' for spans/entities/ner fallback, "
            "or pass a field such as review_seed_entities, baseline_entities, "
            "gliner2_entities, or adjudication.entities_final."
        ),
    )
    parser.add_argument(
        "--score-fields",
        default=",".join(DEFAULT_SCORE_FIELDS),
        help=(
            "Comma-separated score fields to display per entity. "
            "Defaults to ner_score,score_context_boosted,score_calibrated,score,confidence,probability."
        ),
    )
    return parser.parse_args()


def _parse_csv(value):
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def load_corpus(path):
    payload = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return _parse_jsonl(payload)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError("Unsupported input format: expected JSON object/list or JSONL.")


def get_text(record):
    for key in ("text", "relato", "texto", "description", "descricao"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _resolve_field_path(record, field_path):
    current = record
    for part in str(field_path).split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def get_spans(record, span_field="auto"):
    if span_field and span_field != "auto":
        selected = _resolve_field_path(record, span_field)
        if isinstance(selected, list):
            return selected
        return []

    spans = record.get("spans")
    if isinstance(spans, list):
        return spans
    entities = record.get("entities")
    if isinstance(entities, list):
        return entities
    ner = record.get("ner")
    if isinstance(ner, list):
        return ner
    review_seed_entities = record.get("review_seed_entities")
    if isinstance(review_seed_entities, list):
        return review_seed_entities
    return []


def normalize_span(span):
    start = span.get("start")
    end = span.get("end")
    label = span.get("label")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if not isinstance(label, str) or not label.strip():
        return None
    if end <= start:
        return None
    normalized = {"start": start, "end": end, "label": label.strip()}
    for key, value in span.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


def build_label_colors(rows, span_field="auto"):
    labels = []
    seen = set()
    for row in rows:
        for raw_span in get_spans(row, span_field=span_field):
            span = normalize_span(raw_span)
            if not span:
                continue
            label = span["label"]
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(sorted(labels))}


def sanitize_spans(text, spans):
    valid = []
    for raw_span in spans:
        span = normalize_span(raw_span)
        if not span:
            continue
        if span["start"] < 0 or span["end"] > len(text):
            continue
        valid.append(span)
    return sorted(valid, key=lambda s: (s["start"], s["end"]))


def _pick_span_score(span, score_fields):
    for key in score_fields:
        score = _safe_float(span.get(key))
        if score is not None:
            return score, key
    return None, ""


def _pick_record_score(record, score_fields=None):
    for key in (score_fields or DEFAULT_RECORD_SCORE_FIELDS):
        score = _safe_float(record.get(key))
        if score is not None:
            return score, key
    return None, ""


def render_text_with_spans(text, spans, label_colors, score_fields=None):
    if not spans:
        return escape(text)
    score_fields = score_fields or []

    parts = []
    cursor = 0
    for span in spans:
        start = span["start"]
        end = span["end"]
        if start < cursor:
            continue
        if cursor < start:
            parts.append(escape(text[cursor:start]))

        label = span["label"]
        color = label_colors.get(label, "#444444")
        mention = escape(text[start:end])
        label_html = escape(label)
        score, score_key = _pick_span_score(span, score_fields)
        score_html = ""
        if score is not None:
            score_html = f"<span class='score' title='{escape(score_key)}'>{score:.3f}</span>"
        parts.append(
            f"<span class='entity' style='background:{color};' "
            f"title='{label_html}'>{mention}<span class='tag'>{label_html}{score_html}</span></span>"
        )
        cursor = end

    if cursor < len(text):
        parts.append(escape(text[cursor:]))
    return "".join(parts)


def render_entity_list(text, spans, score_fields=None):
    if not spans:
        return "<div class='entity-list muted'>No valid entities.</div>"
    score_fields = score_fields or []

    items = []
    for span in spans:
        mention = escape(text[span["start"]:span["end"]])
        label = escape(span["label"])
        score, score_key = _pick_span_score(span, score_fields)
        score_suffix = f" <span class='entity-score' title='{escape(score_key)}'>[{score:.3f}]</span>" if score is not None else ""
        items.append(f"<li><code>{mention}</code> <span class='entity-kind'>({label})</span>{score_suffix}</li>")
    return f"<div class='entity-list'><b>Entities</b><ul>{''.join(items)}</ul></div>"


def build_html(title, rows_html, legend_html, summary_html):
    tpl = Template(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$title</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #1f2937; }
    h1 { margin-bottom: 8px; }
    .muted { color: #6b7280; margin-bottom: 18px; }
    .summary, .legend { border-collapse: collapse; margin-bottom: 18px; }
    .summary td, .legend td, .legend th { border: 1px solid #d1d5db; padding: 6px 10px; }
    .legend th { background: #f3f4f6; text-align: left; }
    .report { border: 1px solid #d1d5db; border-radius: 8px; margin: 0 0 12px 0; padding: 10px 12px; }
    .report h3 { margin: 0 0 8px 0; font-size: 14px; color: #111827; }
    .report .meta { font-size: 12px; color: #6b7280; margin-bottom: 8px; }
    .report .text { line-height: 1.7; white-space: pre-wrap; }
    .entity-list { margin-top: 10px; font-size: 13px; }
    .entity-list ul { margin: 6px 0 0 18px; padding: 0; }
    .entity-list li { margin: 2px 0; }
    .entity-kind { color: #6b7280; }
    .entity-score { color: #6b7280; font-size: 12px; }
    .entity { color: #fff; border-radius: 4px; padding: 0 4px; margin: 0 1px; display: inline-block; }
    .entity .tag { font-size: 10px; margin-left: 6px; opacity: 0.9; }
    .entity .score { font-size: 9px; vertical-align: super; margin-left: 4px; opacity: 0.95; }
    .swatch { width: 28px; height: 14px; border-radius: 3px; display: inline-block; border: 1px solid #11111133; }
  </style>
</head>
<body>
  <h1>$title</h1>
  <div class="muted">NER annotation viewer (HTML export)</div>
  $summary_html
  $legend_html
  $rows_html
</body>
</html>"""
    )
    return tpl.substitute(
        title=escape(title),
        summary_html=summary_html,
        legend_html=legend_html,
        rows_html=rows_html,
    )


def render_html(rows, output_path, title, max_reports, span_field="auto", score_fields=None):
    if max_reports > 0:
        rows = rows[:max_reports]
    score_fields = score_fields or []

    label_colors = build_label_colors(rows, span_field=span_field)
    label_counts = Counter()
    for row in rows:
        for raw_span in get_spans(row, span_field=span_field):
            span = normalize_span(raw_span)
            if span:
                label_counts[span["label"]] += 1

    summary_html = (
        "<table class='summary'>"
        "<tr><td><b>Records</b></td><td>{}</td></tr>"
        "<tr><td><b>Total spans</b></td><td>{}</td></tr>"
        "<tr><td><b>Distinct labels</b></td><td>{}</td></tr>"
        "</table>"
    ).format(len(rows), sum(label_counts.values()), len(label_counts))

    legend_rows = []
    for label in sorted(label_colors):
        color = label_colors[label]
        legend_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td><span class='swatch' style='background:{color}'></span></td>"
            f"<td>{label_counts.get(label, 0)}</td>"
            "</tr>"
        )
    legend_html = (
        "<table class='legend'><tr><th>Label</th><th>Color</th><th>Count</th></tr>{}</table>".format(
            "".join(legend_rows)
        )
        if legend_rows
        else ""
    )

    rendered_rows = []
    for idx, row in enumerate(rows, start=1):
        text = get_text(row)
        spans = sanitize_spans(text, get_spans(row, span_field=span_field))
        rendered_text = render_text_with_spans(text, spans, label_colors, score_fields=score_fields)
        entity_list_html = render_entity_list(text, spans, score_fields=score_fields)
        source_id = escape(str(row.get("source_id", ""))).strip()
        decision = escape(str(row.get("decision", ""))).strip()
        record_score, record_score_key = _pick_record_score(row)
        header_bits = [f"Record #{idx}"]
        if source_id:
            header_bits.append(f"source_id={source_id}")
        if decision:
            header_bits.append(f"decision={decision}")
        meta_bits = [f"entities={len(spans)}"]
        if record_score is not None:
            meta_bits.append(f"{escape(record_score_key)}={record_score:.6f}")
        rendered_rows.append(
            "<section class='report'>"
            f"<h3>{' | '.join(header_bits)}</h3>"
            f"<div class='meta'>{' | '.join(meta_bits)}</div>"
            f"<div class='text'>{rendered_text}</div>"
            f"{entity_list_html}"
            "</section>"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(title, "".join(rendered_rows), legend_html, summary_html), encoding="utf-8")


def main():
    args = parse_args()
    rows = load_corpus(args.input)
    render_html(
        rows,
        args.output,
        args.title,
        args.max_reports,
        span_field=args.span_field,
        score_fields=_parse_csv(args.score_fields),
    )
    print(f"[ok] HTML saved to: {args.output}")


if __name__ == "__main__":
    main()
