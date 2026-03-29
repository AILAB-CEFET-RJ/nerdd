#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.paths import resolve_repo_artifact_path
from tools.inspect_dense_tips import get_spans, get_text, read_json_or_jsonl, write_jsonl
from tools.render_ner_html import build_html, build_label_colors, render_text_with_spans, sanitize_spans


DEFAULT_ENTITY_TYPES = ["person", "location", "organization"]
DEFAULT_LABEL_MAP = {
    "person": "Person",
    "location": "Location",
    "organization": "Organization",
}


def _parse_csv(raw_value):
    if not raw_value:
        return []
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def normalize_entities(entities):
    if isinstance(entities, list):
        normalized = []
        for entity in entities:
            if isinstance(entity, dict):
                normalized.append(entity)
            elif isinstance(entity, str):
                normalized.append({"text": entity})
        return sort_entities(normalized)

    if isinstance(entities, dict):
        normalized = []
        for label, mentions in entities.items():
            if isinstance(mentions, list):
                for mention in mentions:
                    if isinstance(mention, dict):
                        normalized.append({"label": label, **mention})
                    else:
                        normalized.append({"text": str(mention), "label": label})
            elif isinstance(mentions, dict):
                normalized.append({"label": label, **mentions})
            else:
                normalized.append({"text": str(mentions), "label": label})
        return sort_entities(normalized)

    return []


def sort_entities(entities):
    return sorted(
        entities,
        key=lambda entity: (
            str(entity.get("label", "")),
            str(entity.get("text", "")),
            str(entity.get("start", "")),
            str(entity.get("end", "")),
        ),
    )


def map_gliner2_labels(entities):
    mapped = []
    for entity in normalize_entities(entities):
        label = str(entity.get("label", "")).strip().lower()
        mapped_label = DEFAULT_LABEL_MAP.get(label, entity.get("label", ""))
        mapped.append(
            {
                "text": entity.get("text", ""),
                "label": mapped_label,
                "start": entity.get("start"),
                "end": entity.get("end"),
                "confidence": entity.get("confidence"),
                "score": entity.get("score"),
            }
        )
    return sort_entities(mapped)


def extract_entities(model, text, entity_types):
    result = model.extract_entities(text, entity_types)
    if isinstance(result, dict):
        result = result.get("entities", result)
    return map_gliner2_labels(result)


def build_comparison_rows(rows, model, entity_types, adapter_dir=""):
    compared = []
    use_adapter = bool(adapter_dir)

    if use_adapter:
        model.load_adapter(str(adapter_dir))

    for idx, row in enumerate(rows, start=1):
        text = get_text(row)
        baseline_entities = get_spans(row)
        adapter_entities = extract_entities(model, text, entity_types) if use_adapter else []

        if use_adapter:
            model.unload_adapter()
        base_entities = extract_entities(model, text, entity_types)
        if use_adapter:
            model.load_adapter(str(adapter_dir))

        compared.append(
            {
                "text": text,
                "baseline_entities": baseline_entities,
                "gliner2_base_entities": base_entities,
                "gliner2_adapter_entities": adapter_entities,
                "_comparison_meta": {
                    "row_index_1based": idx,
                    "baseline_entity_count": len(baseline_entities),
                    "gliner2_base_entity_count": len(base_entities),
                    "gliner2_adapter_entity_count": len(adapter_entities),
                },
                "_source": row,
            }
        )
    return compared


def build_summary(rows):
    counts = {
        "baseline": Counter(),
        "gliner2_base": Counter(),
        "gliner2_adapter": Counter(),
    }
    for row in rows:
        for span in row["baseline_entities"]:
            label = span.get("label")
            if label:
                counts["baseline"][str(label)] += 1
        for span in row["gliner2_base_entities"]:
            label = span.get("label")
            if label:
                counts["gliner2_base"][str(label)] += 1
        for span in row["gliner2_adapter_entities"]:
            label = span.get("label")
            if label:
                counts["gliner2_adapter"][str(label)] += 1
    return {
        "records": len(rows),
        "baseline_total_spans": int(sum(counts["baseline"].values())),
        "gliner2_base_total_spans": int(sum(counts["gliner2_base"].values())),
        "gliner2_adapter_total_spans": int(sum(counts["gliner2_adapter"].values())),
        "baseline_label_counts": dict(sorted(counts["baseline"].items())),
        "gliner2_base_label_counts": dict(sorted(counts["gliner2_base"].items())),
        "gliner2_adapter_label_counts": dict(sorted(counts["gliner2_adapter"].items())),
    }


def render_comparison_html(rows, output_path, title, show_adapter):
    label_rows = [{"entities": row["baseline_entities"]} for row in rows]
    label_rows.extend({"entities": row["gliner2_base_entities"]} for row in rows)
    if show_adapter:
        label_rows.extend({"entities": row["gliner2_adapter_entities"]} for row in rows)
    label_colors = build_label_colors(label_rows)

    counts = {
        "baseline": Counter(),
        "gliner2_base": Counter(),
        "gliner2_adapter": Counter(),
    }
    rendered_rows = []
    for row in rows:
        text = row["text"]
        baseline_spans = sanitize_spans(text, row["baseline_entities"])
        gliner2_base_spans = sanitize_spans(text, row["gliner2_base_entities"])
        gliner2_adapter_spans = sanitize_spans(text, row["gliner2_adapter_entities"])
        for span in baseline_spans:
            counts["baseline"][span["label"]] += 1
        for span in gliner2_base_spans:
            counts["gliner2_base"][span["label"]] += 1
        for span in gliner2_adapter_spans:
            counts["gliner2_adapter"][span["label"]] += 1
        meta = row["_comparison_meta"]
        sections = [
            "<div class='meta'><b>Baseline / existing predictions</b></div>"
            f"<div class='text'>{render_text_with_spans(text, baseline_spans, label_colors)}</div>",
            "<div class='meta' style='margin-top:8px;'><b>GLiNER2 base</b></div>"
            f"<div class='text'>{render_text_with_spans(text, gliner2_base_spans, label_colors)}</div>",
        ]
        if show_adapter:
            sections.append(
                "<div class='meta' style='margin-top:8px;'><b>GLiNER2 + LoRA</b></div>"
                f"<div class='text'>{render_text_with_spans(text, gliner2_adapter_spans, label_colors)}</div>"
            )
        rendered_rows.append(
            "<section class='report'>"
            f"<h3>Record #{meta['row_index_1based']}</h3>"
            f"<div class='meta'>baseline_entities={meta['baseline_entity_count']} | gliner2_base_entities={meta['gliner2_base_entity_count']}"
            + (
                f" | gliner2_adapter_entities={meta['gliner2_adapter_entity_count']}"
                if show_adapter
                else ""
            )
            + "</div>"
            + "".join(sections)
            + "</section>"
        )

    label_union = sorted(set(counts["baseline"]) | set(counts["gliner2_base"]) | set(counts["gliner2_adapter"]))
    summary_html = (
        "<table class='summary'>"
        f"<tr><td><b>Records</b></td><td>{len(rows)}</td></tr>"
        f"<tr><td><b>Baseline spans</b></td><td>{sum(counts['baseline'].values())}</td></tr>"
        f"<tr><td><b>GLiNER2 base spans</b></td><td>{sum(counts['gliner2_base'].values())}</td></tr>"
        + (
            f"<tr><td><b>GLiNER2 + LoRA spans</b></td><td>{sum(counts['gliner2_adapter'].values())}</td></tr>"
            if show_adapter
            else ""
        )
        + "</table>"
    )
    legend_rows = []
    for label in label_union:
        color = label_colors.get(label, "#444444")
        legend_rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td><span class='swatch' style='background:{color}'></span></td>"
            f"<td>{counts['baseline'].get(label, 0)}</td>"
            f"<td>{counts['gliner2_base'].get(label, 0)}</td>"
            + (f"<td>{counts['gliner2_adapter'].get(label, 0)}</td>" if show_adapter else "")
            + "</tr>"
        )
    legend_html = (
        "<table class='legend'><tr><th>Label</th><th>Color</th><th>Baseline</th><th>GLiNER2 base</th>{}</tr>{}</table>".format(
            "<th>GLiNER2 + LoRA</th>" if show_adapter else "",
            "".join(legend_rows),
        )
        if legend_rows
        else ""
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_html(title, "".join(rendered_rows), legend_html, summary_html), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare existing predictions against GLiNER2 base and optional LoRA adapter.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL. Existing predictions should be in entities/spans/ner.")
    parser.add_argument("--model", default="fastino/gliner2-base-v1", help="GLiNER2 base model name or local path.")
    parser.add_argument("--adapter-dir", default="", help="Optional LoRA adapter directory.")
    parser.add_argument("--entity-types", default="person,location,organization", help="Comma-separated GLiNER2 entity types.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with baseline vs GLiNER2 entities.")
    parser.add_argument("--output-html", required=True, help="Output HTML with side-by-side comparison.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--title", default="Baseline vs GLiNER2 comparison", help="HTML title.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional limit of input records to compare.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from gliner2 import GLiNER2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("gliner2 is not installed in this environment.") from exc

    input_path = resolve_repo_artifact_path(__file__, args.input)
    rows = read_json_or_jsonl(str(input_path))
    if args.max_records > 0:
        rows = rows[: args.max_records]

    entity_types = _parse_csv(args.entity_types) or list(DEFAULT_ENTITY_TYPES)
    model = GLiNER2.from_pretrained(str(resolve_repo_artifact_path(__file__, args.model)))
    adapter_dir = str(resolve_repo_artifact_path(__file__, args.adapter_dir)) if args.adapter_dir else ""
    output_jsonl = resolve_repo_artifact_path(__file__, args.output_jsonl)
    output_html = resolve_repo_artifact_path(__file__, args.output_html)
    compared_rows = build_comparison_rows(rows, model=model, entity_types=entity_types, adapter_dir=adapter_dir)
    write_jsonl(str(output_jsonl), compared_rows)
    render_comparison_html(compared_rows, output_html, args.title, show_adapter=bool(adapter_dir))
    summary = build_summary(compared_rows)

    print(f"Saved JSONL: {output_jsonl}")
    print(f"Saved HTML: {output_html}")
    if args.summary_json:
        target = resolve_repo_artifact_path(__file__, args.summary_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved summary JSON: {target}")

    print(f"Records compared: {summary['records']}")
    print(f"Baseline spans: {summary['baseline_total_spans']}")
    print(f"GLiNER2 base spans: {summary['gliner2_base_total_spans']}")
    if args.adapter_dir:
        print(f"GLiNER2 + LoRA spans: {summary['gliner2_adapter_total_spans']}")


if __name__ == "__main__":
    main()
