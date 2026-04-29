#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render NER spans as standalone HTML snippets and reports.

Enhancements over the original viewer:
- Shows each entity's seed_origin in the tooltip and entity list.
- Optionally highlights entities whose seed_origin matches a chosen value,
  e.g. train_lexicon_projection.
- Adds origin-level summary counts.
"""

from __future__ import annotations

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
DEFAULT_ORIGIN_FIELDS = ["seed_origin", "origin", "source", "rule", "matched_by"]
DEFAULT_HIGHLIGHT_ORIGIN = "train_lexicon_projection"


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
    parser.add_argument(
        "--origin-fields",
        default=",".join(DEFAULT_ORIGIN_FIELDS),
        help=(
            "Comma-separated fields to treat as entity origin/rule metadata. "
            "Defaults to seed_origin,origin,source,rule,matched_by."
        ),
    )
    parser.add_argument(
        "--highlight-origin",
        default=DEFAULT_HIGHLIGHT_ORIGIN,
        help=(
            "Origin value to visually highlight, e.g. train_lexicon_projection. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--show-origin-in-text",
        action="store_true",
        help="Show a compact origin code inside the highlighted span tag. Tooltips/lists always include origin.",
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


def _pick_span_origin(span, origin_fields):
    for key in origin_fields:
        value = span.get(key)
        if value is not None and str(value).strip():
            return str(value).strip(), key
    return "", ""


def _origin_short(origin):
    if not origin:
        return ""
    if origin == "train_lexicon_projection":
        return "proj"
    parts = origin.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return origin[:12]


def render_text_with_spans(
    text,
    spans,
    label_colors,
    score_fields=None,
    origin_fields=None,
    highlight_origin=DEFAULT_HIGHLIGHT_ORIGIN,
    show_origin_in_text=False,
):
    if not spans:
        return escape(text)
    score_fields = score_fields or []
    origin_fields = origin_fields or []

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
        origin, origin_key = _pick_span_origin(span, origin_fields)
        score, score_key = _pick_span_score(span, score_fields)

        score_html = ""
        if score is not None:
            score_html = f"<span class='score' title='{escape(score_key)}'>{score:.3f}</span>"

        origin_badge = ""
        if show_origin_in_text and origin:
            origin_badge = f"<span class='origin-mini'>{escape(_origin_short(origin))}</span>"

        classes = ["entity"]
        if highlight_origin and origin == highlight_origin:
            classes.append("projected")
        elif origin:
            classes.append("has-origin")
        class_attr = " ".join(classes)

        title_bits = [f"label={label}", f"span={start}-{end}"]
        if origin:
            title_bits.append(f"{origin_key}={origin}")
        if score is not None:
            title_bits.append(f"{score_key}={score:.6f}")
        title_html = escape(" | ".join(title_bits))

        parts.append(
            f"<span class='{class_attr}' style='background:{color};' title='{title_html}'>"
            f"{mention}<span class='tag'>{label_html}{score_html}{origin_badge}</span></span>"
        )
        cursor = end

    if cursor < len(text):
        parts.append(escape(text[cursor:]))
    return "".join(parts)


def render_entity_list(text, spans, score_fields=None, origin_fields=None, highlight_origin=DEFAULT_HIGHLIGHT_ORIGIN):
    if not spans:
        return "<div class='entity-list muted'>No valid entities.</div>"
    score_fields = score_fields or []
    origin_fields = origin_fields or []

    items = []
    for span in spans:
        mention = escape(text[span["start"]:span["end"]])
        label = escape(span["label"])
        origin, origin_key = _pick_span_origin(span, origin_fields)
        origin_suffix = ""
        li_class = ""
        if origin:
            origin_label = escape(origin)
            origin_suffix = f" <span class='entity-origin' title='{escape(origin_key)}'>origin={origin_label}</span>"
            if highlight_origin and origin == highlight_origin:
                li_class = " class='projected-item'"
        score, score_key = _pick_span_score(span, score_fields)
        score_suffix = f" <span class='entity-score' title='{escape(score_key)}'>[{score:.3f}]</span>" if score is not None else ""
        items.append(
            f"<li{li_class}><code>{mention}</code> "
            f"<span class='entity-kind'>({label})</span>{score_suffix}{origin_suffix}</li>"
        )
    return f"<div class='entity-list'><b>Entities</b><ul>{''.join(items)}</ul></div>"


def build_html(title, rows_html, legend_html, summary_html, origin_summary_html):
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
    h2 { margin-top: 22px; font-size: 18px; }
    .muted { color: #6b7280; margin-bottom: 18px; }
    .summary, .legend, .origin-summary { border-collapse: collapse; margin-bottom: 18px; }
    .summary td, .legend td, .legend th, .origin-summary td, .origin-summary th { border: 1px solid #d1d5db; padding: 6px 10px; }
    .legend th, .origin-summary th { background: #f3f4f6; text-align: left; }
    .report { border: 1px solid #d1d5db; border-radius: 8px; margin: 0 0 12px 0; padding: 10px 12px; }
    .report h3 { margin: 0 0 8px 0; font-size: 14px; color: #111827; }
    .report .meta { font-size: 12px; color: #6b7280; margin-bottom: 8px; }
    .report .text { line-height: 1.7; white-space: pre-wrap; }
    .entity-list { margin-top: 10px; font-size: 13px; }
    .entity-list ul { margin: 6px 0 0 18px; padding: 0; }
    .entity-list li { margin: 2px 0; }
    .entity-kind { color: #6b7280; }
    .entity-score { color: #6b7280; font-size: 12px; }
    .entity-origin { color: #374151; background: #f3f4f6; border-radius: 4px; padding: 1px 4px; font-size: 11px; }
    .entity { color: #fff; border-radius: 4px; padding: 0 4px; margin: 0 1px; display: inline-block; }
    .entity .tag { font-size: 10px; margin-left: 6px; opacity: 0.95; }
    .entity .score { font-size: 9px; vertical-align: super; margin-left: 4px; opacity: 0.95; }
    .entity .origin-mini { background: rgba(255,255,255,.25); border: 1px solid rgba(255,255,255,.35); border-radius: 3px; padding: 0 3px; margin-left: 4px; }
    .entity.projected { outline: 3px solid #facc15; box-shadow: 0 0 0 1px rgba(0,0,0,.15) inset; }
    .projected-item { background: #fef9c3; }
    .swatch { width: 28px; height: 14px; border-radius: 3px; display: inline-block; border: 1px solid #11111133; }
  </style>
</head>
<body>
  <h1>$title</h1>
  <div class="muted">NER annotation viewer (HTML export). Entity tooltips and lists include origin metadata when available.</div>
  $summary_html
  $legend_html
  $origin_summary_html
  $rows_html
</body>
</html>"""
    )
    return tpl.substitute(
        title=escape(title),
        summary_html=summary_html,
        legend_html=legend_html,
        origin_summary_html=origin_summary_html,
        rows_html=rows_html,
    )


def render_html(
    rows,
    output_path,
    title,
    max_reports,
    span_field="auto",
    score_fields=None,
    origin_fields=None,
    highlight_origin=DEFAULT_HIGHLIGHT_ORIGIN,
    show_origin_in_text=False,
):
    if max_reports > 0:
        rows = rows[:max_reports]
    score_fields = score_fields or []
    origin_fields = origin_fields or []

    label_colors = build_label_colors(rows, span_field=span_field)
    label_counts = Counter()
    origin_counts = Counter()
    label_origin_counts = Counter()

    for row in rows:
        text = get_text(row)
        for raw_span in get_spans(row, span_field=span_field):
            span = normalize_span(raw_span)
            if not span:
                continue
            if span["start"] < 0 or span["end"] > len(text):
                continue
            label = span["label"]
            origin, _ = _pick_span_origin(span, origin_fields)
            origin = origin or "(missing)"
            label_counts[label] += 1
            origin_counts[origin] += 1
            label_origin_counts[(label, origin)] += 1

    summary_html = (
        "<table class='summary'>"
        "<tr><td><b>Records</b></td><td>{}</td></tr>"
        "<tr><td><b>Total spans</b></td><td>{}</td></tr>"
        "<tr><td><b>Distinct labels</b></td><td>{}</td></tr>"
        "<tr><td><b>Distinct origins</b></td><td>{}</td></tr>"
        "</table>"
    ).format(len(rows), sum(label_counts.values()), len(label_counts), len(origin_counts))

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

    origin_rows = []
    for origin, count in origin_counts.most_common():
        label_bits = []
        for label in sorted(label_counts):
            n = label_origin_counts.get((label, origin), 0)
            if n:
                label_bits.append(f"{escape(label)}={n}")
        highlight_note = " <b>highlighted</b>" if highlight_origin and origin == highlight_origin else ""
        origin_rows.append(
            "<tr>"
            f"<td>{escape(origin)}{highlight_note}</td>"
            f"<td>{count}</td>"
            f"<td>{', '.join(label_bits)}</td>"
            "</tr>"
        )
    origin_summary_html = (
        "<h2>Origin summary</h2>"
        "<table class='origin-summary'><tr><th>Origin</th><th>Count</th><th>By label</th></tr>{}</table>".format(
            "".join(origin_rows)
        )
        if origin_rows
        else ""
    )

    rendered_rows = []
    for idx, row in enumerate(rows, start=1):
        text = get_text(row)
        spans = sanitize_spans(text, get_spans(row, span_field=span_field))
        projected_count = 0
        if highlight_origin:
            for span in spans:
                origin, _ = _pick_span_origin(span, origin_fields)
                if origin == highlight_origin:
                    projected_count += 1

        rendered_text = render_text_with_spans(
            text,
            spans,
            label_colors,
            score_fields=score_fields,
            origin_fields=origin_fields,
            highlight_origin=highlight_origin,
            show_origin_in_text=show_origin_in_text,
        )
        entity_list_html = render_entity_list(
            text,
            spans,
            score_fields=score_fields,
            origin_fields=origin_fields,
            highlight_origin=highlight_origin,
        )
        source_id = escape(str(row.get("source_id", ""))).strip()
        decision = escape(str(row.get("decision", ""))).strip()
        record_score, record_score_key = _pick_record_score(row)
        header_bits = [f"Record #{idx}"]
        if source_id:
            header_bits.append(f"source_id={source_id}")
        if decision:
            header_bits.append(f"decision={decision}")
        meta_bits = [f"entities={len(spans)}"]
        if highlight_origin:
            meta_bits.append(f"{escape(highlight_origin)}={projected_count}")
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
    output.write_text(
        build_html(title, "".join(rendered_rows), legend_html, summary_html, origin_summary_html),
        encoding="utf-8",
    )


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
        origin_fields=_parse_csv(args.origin_fields),
        highlight_origin=args.highlight_origin.strip(),
        show_origin_in_text=args.show_origin_in_text,
    )
    print(f"[ok] HTML saved to: {args.output}")


if __name__ == "__main__":
    main()
