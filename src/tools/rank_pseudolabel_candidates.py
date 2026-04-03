#!/usr/bin/env python3
"""Rank pseudolabel candidates with heuristic quality scores for manual review."""

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import get_spans, get_text, read_json_or_jsonl, write_jsonl
from tools.render_ner_html import build_html, build_label_colors, render_text_with_spans, sanitize_spans

DEFAULT_RECORD_SCORE_FIELDS = ("record_score", "record_score_context_boosted", "score_relato_confianca")
DEFAULT_ENTITY_SCORE_FIELDS = ("score_context_boosted", "score_calibrated", "score", "score_ts")
DEFAULT_GENERIC_ENTITY_TEXTS = (
    ".",
    ",",
    "urgente",
    "hurgente",
    "batalhao",
    "batalhão",
    "p2",
    "comandante geral",
)
DEFAULT_POLITICAL_COPYPASTA_TERMS = (
    "corrupcao",
    "corrupção",
    "pec da transicao",
    "pec da transição",
    "presidente eleito",
    "lula",
    "investigad",
    "esquema",
)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_csv(raw_value):
    if not raw_value:
        return []
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def _strip_accents(text):
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_entity_text(text):
    text = " ".join(str(text).strip().lower().split())
    text = _strip_accents(text)
    return text.strip(" \t\r\n.,;:!?()[]{}\"'")


DEFAULT_GENERIC_ENTITY_TEXTS_NORMALIZED = {_normalize_entity_text(value) for value in DEFAULT_GENERIC_ENTITY_TEXTS}
DEFAULT_POLITICAL_COPYPASTA_TERMS_NORMALIZED = {_normalize_entity_text(value) for value in DEFAULT_POLITICAL_COPYPASTA_TERMS}


def _is_list_like_person_dump(text, label_counts):
    if not isinstance(text, str):
        return False
    comma_count = text.count(",")
    semicolon_count = text.count(";")
    if comma_count + semicolon_count < 4:
        return False
    person_count = int(label_counts.get("Person", 0))
    entity_total = sum(int(v) for v in label_counts.values())
    if person_count < 4 or entity_total == 0:
        return False
    if person_count / entity_total < 0.6:
        return False
    lowered = _normalize_entity_text(text)
    narrative_markers = ("trafico", "roubo", "assalto", "moto", "arma", "tiro", "bairro", "rua")
    if any(marker in lowered for marker in narrative_markers):
        return False
    return True


def _is_political_copypasta(text, label_counts):
    if not isinstance(text, str):
        return False
    normalized = _normalize_entity_text(text)
    hits = sum(1 for term in DEFAULT_POLITICAL_COPYPASTA_TERMS_NORMALIZED if term and term in normalized)
    if hits < 2:
        return False
    comma_count = text.count(",")
    semicolon_count = text.count(";")
    entity_total = sum(int(v) for v in label_counts.values())
    return (comma_count + semicolon_count >= 4) or entity_total >= 8


def _pick_record_score(row, score_fields):
    for key in score_fields:
        score = _safe_float(row.get(key))
        if score is not None:
            return score, key
    return None, ""


def _pick_entity_score(entity, score_fields):
    for key in score_fields:
        score = _safe_float(entity.get(key))
        if score is not None:
            return score, key
    return None, ""


def _entity_length(entity):
    start = entity.get("start")
    end = entity.get("end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return end - start
    text = entity.get("text")
    if isinstance(text, str):
        return len(text)
    return 0


def _compute_candidate_quality(
    *,
    record_score,
    mean_entity_score,
    high_score_share,
    low_score_share,
    short_span_ratio,
    entity_density_per_1k_chars,
    location_max_score,
    organization_ratio,
):
    density_penalty = min(entity_density_per_1k_chars / 30.0, 1.0)
    organization_overload = min(max(organization_ratio - 0.5, 0.0) / 0.5, 1.0)
    quality = (
        0.45 * record_score
        + 0.30 * mean_entity_score
        + 0.15 * high_score_share
        + 0.10 * location_max_score
        - 0.20 * low_score_share
        - 0.20 * short_span_ratio
        - 0.15 * density_penalty
        - 0.10 * organization_overload
    )
    return quality


def build_candidate(
    row,
    *,
    row_index,
    record_score_fields,
    entity_score_fields,
    label_field,
    short_span_max_chars,
    high_entity_score_threshold,
    low_entity_score_threshold,
):
    text = get_text(row)
    entities = get_spans(row)
    text_length = len(text)
    record_score, record_score_field = _pick_record_score(row, record_score_fields)
    label_counts = Counter()
    entity_scores = []
    high_score_count = 0
    low_score_count = 0
    short_span_count = 0
    location_scores = []

    for entity in entities:
        label = str(entity.get(label_field, "UNKNOWN"))
        label_counts[label] += 1
        score, _ = _pick_entity_score(entity, entity_score_fields)
        if score is not None:
            entity_scores.append(score)
            if score >= high_entity_score_threshold:
                high_score_count += 1
            if score < low_entity_score_threshold:
                low_score_count += 1
            if label == "Location":
                location_scores.append(score)
        if _entity_length(entity) <= short_span_max_chars:
            short_span_count += 1

    entity_count = len(entities)
    mean_entity_score = (sum(entity_scores) / len(entity_scores)) if entity_scores else 0.0
    min_entity_score = min(entity_scores) if entity_scores else None
    max_entity_score = max(entity_scores) if entity_scores else None
    high_score_share = (high_score_count / entity_count) if entity_count else 0.0
    low_score_share = (low_score_count / entity_count) if entity_count else 0.0
    short_span_ratio = (short_span_count / entity_count) if entity_count else 0.0
    entity_density_per_1k_chars = (entity_count * 1000.0 / text_length) if text_length > 0 else 0.0
    organization_ratio = (label_counts.get("Organization", 0) / entity_count) if entity_count else 0.0
    location_ratio = (label_counts.get("Location", 0) / entity_count) if entity_count else 0.0
    location_max_score = max(location_scores) if location_scores else 0.0
    generic_entity_texts = sorted(
        {
            normalized
            for entity in entities
            for normalized in [_normalize_entity_text(entity.get("text", ""))]
            if normalized and normalized in DEFAULT_GENERIC_ENTITY_TEXTS_NORMALIZED
        }
    )
    list_like_person_dump = _is_list_like_person_dump(text, label_counts)
    political_copypasta = _is_political_copypasta(text, label_counts)

    effective_record_score = record_score if record_score is not None else mean_entity_score
    quality_score = _compute_candidate_quality(
        record_score=effective_record_score,
        mean_entity_score=mean_entity_score,
        high_score_share=high_score_share,
        low_score_share=low_score_share,
        short_span_ratio=short_span_ratio,
        entity_density_per_1k_chars=entity_density_per_1k_chars,
        location_max_score=location_max_score,
        organization_ratio=organization_ratio,
    )

    candidate = dict(row)
    candidate["entities"] = entities
    candidate["_candidate_rank"] = {
        "row_index_1based": row_index,
        "candidate_quality_score": quality_score,
        "record_score": record_score,
        "record_score_field": record_score_field,
        "entity_count": entity_count,
        "text_length": text_length,
        "entity_density_per_1k_chars": entity_density_per_1k_chars,
        "mean_entity_score": mean_entity_score,
        "min_entity_score": min_entity_score,
        "max_entity_score": max_entity_score,
        "high_score_share": high_score_share,
        "low_score_share": low_score_share,
        "short_span_ratio": short_span_ratio,
        "location_max_score": location_max_score,
        "organization_ratio": organization_ratio,
        "location_ratio": location_ratio,
        "generic_entity_texts": generic_entity_texts,
        "list_like_person_dump": list_like_person_dump,
        "political_copypasta": political_copypasta,
        "label_counts": dict(label_counts),
    }
    return candidate


def keep_candidate(
    candidate,
    *,
    min_record_score,
    min_entities,
    max_entities,
    min_text_length,
    max_low_score_share,
    max_location_ratio,
    max_short_span_ratio,
    drop_generic_entity_texts,
    drop_list_like_person_dumps,
    drop_political_copypasta,
    required_labels,
):
    meta = candidate["_candidate_rank"]
    label_counts = meta["label_counts"]
    record_score = meta["record_score"]

    if record_score is not None and record_score < min_record_score:
        return False, "record_score"
    if min_entities > 0 and meta["entity_count"] < min_entities:
        return False, "min_entities"
    if max_entities > 0 and meta["entity_count"] > max_entities:
        return False, "entity_count"
    if min_text_length > 0 and meta["text_length"] < min_text_length:
        return False, "text_length"
    if meta["low_score_share"] > max_low_score_share:
        return False, "low_score_share"
    if meta["location_ratio"] > max_location_ratio:
        return False, "location_ratio"
    if meta["short_span_ratio"] > max_short_span_ratio:
        return False, "short_span_ratio"
    if drop_generic_entity_texts and meta["generic_entity_texts"]:
        return False, "generic_entity_texts"
    if drop_list_like_person_dumps and meta["list_like_person_dump"]:
        return False, "list_like_person_dump"
    if drop_political_copypasta and meta["political_copypasta"]:
        return False, "political_copypasta"
    if required_labels and not any(label_counts.get(label, 0) > 0 for label in required_labels):
        return False, "required_labels"
    return True, ""


def rank_candidates(
    rows,
    *,
    record_score_fields,
    entity_score_fields,
    label_field,
    min_record_score,
    min_entities,
    max_entities,
    min_text_length,
    max_low_score_share,
    max_location_ratio,
    max_short_span_ratio,
    drop_generic_entity_texts,
    drop_list_like_person_dumps,
    drop_political_copypasta,
    short_span_max_chars,
    high_entity_score_threshold,
    low_entity_score_threshold,
    required_labels,
):
    counters = Counter()
    kept = []
    for idx, row in enumerate(rows, start=1):
        candidate = build_candidate(
            row,
            row_index=idx,
            record_score_fields=record_score_fields,
            entity_score_fields=entity_score_fields,
            label_field=label_field,
            short_span_max_chars=short_span_max_chars,
            high_entity_score_threshold=high_entity_score_threshold,
            low_entity_score_threshold=low_entity_score_threshold,
        )
        counters["rows_total"] += 1
        ok, reason = keep_candidate(
            candidate,
            min_record_score=min_record_score,
            min_entities=min_entities,
            max_entities=max_entities,
            min_text_length=min_text_length,
            max_low_score_share=max_low_score_share,
            max_location_ratio=max_location_ratio,
            max_short_span_ratio=max_short_span_ratio,
            drop_generic_entity_texts=drop_generic_entity_texts,
            drop_list_like_person_dumps=drop_list_like_person_dumps,
            drop_political_copypasta=drop_political_copypasta,
            required_labels=required_labels,
        )
        if not ok:
            counters[f"dropped_{reason}"] += 1
            continue
        kept.append(candidate)

    kept.sort(
        key=lambda row: (
            -row["_candidate_rank"]["candidate_quality_score"],
            -(row["_candidate_rank"]["record_score"] if row["_candidate_rank"]["record_score"] is not None else -1.0),
            -row["_candidate_rank"]["mean_entity_score"],
            row["_candidate_rank"]["entity_count"],
        )
    )
    for rank, row in enumerate(kept, start=1):
        row["_candidate_rank"]["rank"] = rank
    counters["rows_kept"] = len(kept)
    return kept, counters


def write_csv(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "row_index_1based",
        "candidate_quality_score",
        "record_score",
        "record_score_field",
        "entity_count",
        "text_length",
        "entity_density_per_1k_chars",
        "mean_entity_score",
        "min_entity_score",
        "max_entity_score",
        "high_score_share",
        "low_score_share",
        "short_span_ratio",
        "location_max_score",
        "organization_ratio",
        "location_count",
        "organization_count",
        "person_count",
        "text_preview",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            meta = row["_candidate_rank"]
            counts = meta["label_counts"]
            writer.writerow(
                {
                    "rank": meta["rank"],
                    "row_index_1based": meta["row_index_1based"],
                    "candidate_quality_score": f"{meta['candidate_quality_score']:.6f}",
                    "record_score": "" if meta["record_score"] is None else f"{meta['record_score']:.6f}",
                    "record_score_field": meta["record_score_field"],
                    "entity_count": meta["entity_count"],
                    "text_length": meta["text_length"],
                    "entity_density_per_1k_chars": f"{meta['entity_density_per_1k_chars']:.3f}",
                    "mean_entity_score": f"{meta['mean_entity_score']:.6f}",
                    "min_entity_score": "" if meta["min_entity_score"] is None else f"{meta['min_entity_score']:.6f}",
                    "max_entity_score": "" if meta["max_entity_score"] is None else f"{meta['max_entity_score']:.6f}",
                    "high_score_share": f"{meta['high_score_share']:.6f}",
                    "low_score_share": f"{meta['low_score_share']:.6f}",
                    "short_span_ratio": f"{meta['short_span_ratio']:.6f}",
                    "location_max_score": f"{meta['location_max_score']:.6f}",
                    "organization_ratio": f"{meta['organization_ratio']:.6f}",
                    "location_count": counts.get("Location", 0),
                    "organization_count": counts.get("Organization", 0),
                    "person_count": counts.get("Person", 0),
                    "text_preview": get_text(row).replace("\n", " ")[:160],
                }
            )


def render_ranked_html(rows, output_path, title):
    label_colors = build_label_colors(rows)
    label_counts = Counter()
    rendered_rows = []
    for row in rows:
        text = get_text(row)
        spans = sanitize_spans(text, get_spans(row))
        for span in spans:
            label_counts[span["label"]] += 1
        meta = row["_candidate_rank"]
        label_summary = ", ".join(f"{label}={count}" for label, count in sorted(meta["label_counts"].items())) or "none"
        rendered_rows.append(
            "<section class='report'>"
            f"<h3>Rank #{meta['rank']}</h3>"
            "<div class='meta'>"
            f"quality={meta['candidate_quality_score']:.4f} | "
            f"record_score={meta['record_score'] if meta['record_score'] is not None else 'NA'} | "
            f"entities={meta['entity_count']} | "
            f"density={meta['entity_density_per_1k_chars']:.1f}/1k chars | "
            f"short_span_ratio={meta['short_span_ratio']:.2f} | "
            f"labels={escape(label_summary)}"
            "</div>"
            f"<div class='text'>{render_text_with_spans(text, spans, label_colors)}</div>"
            "</section>"
        )

    summary_html = (
        "<table class='summary'>"
        f"<tr><td><b>Records</b></td><td>{len(rows)}</td></tr>"
        f"<tr><td><b>Total spans</b></td><td>{sum(label_counts.values())}</td></tr>"
        f"<tr><td><b>Distinct labels</b></td><td>{len(label_counts)}</td></tr>"
        "</table>"
    )
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
        "<table class='legend'><tr><th>Label</th><th>Color</th><th>Count</th></tr>{}</table>".format("".join(legend_rows))
        if legend_rows
        else ""
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_html(title, "".join(rendered_rows), legend_html, summary_html), encoding="utf-8")


def build_summary(rows, counters, args):
    quality_scores = [row["_candidate_rank"]["candidate_quality_score"] for row in rows]
    record_scores = [row["_candidate_rank"]["record_score"] for row in rows if row["_candidate_rank"]["record_score"] is not None]
    entity_counts = [row["_candidate_rank"]["entity_count"] for row in rows]
    return {
        "input": str(Path(args.input).resolve()),
        "rows_total": counters["rows_total"],
        "rows_kept": counters["rows_kept"],
        "top_n": args.top_n,
        "filters": {
            "min_record_score": args.min_record_score,
            "min_entities": args.min_entities,
            "max_entities": args.max_entities,
            "min_text_length": args.min_text_length,
            "max_low_score_share": args.max_low_score_share,
            "max_location_ratio": args.max_location_ratio,
            "max_short_span_ratio": args.max_short_span_ratio,
            "drop_generic_entity_texts": args.drop_generic_entity_texts,
            "drop_list_like_person_dumps": args.drop_list_like_person_dumps,
            "drop_political_copypasta": args.drop_political_copypasta,
            "required_labels": _parse_csv(args.required_labels),
        },
        "ranking": {
            "record_score_fields": _parse_csv(args.record_score_fields),
            "entity_score_fields": _parse_csv(args.entity_score_fields),
            "short_span_max_chars": args.short_span_max_chars,
            "high_entity_score_threshold": args.high_entity_score_threshold,
            "low_entity_score_threshold": args.low_entity_score_threshold,
        },
        "dropped": {
            "record_score": counters["dropped_record_score"],
            "min_entities": counters["dropped_min_entities"],
            "entity_count": counters["dropped_entity_count"],
            "text_length": counters["dropped_text_length"],
            "low_score_share": counters["dropped_low_score_share"],
            "location_ratio": counters["dropped_location_ratio"],
            "short_span_ratio": counters["dropped_short_span_ratio"],
            "generic_entity_texts": counters["dropped_generic_entity_texts"],
            "list_like_person_dump": counters["dropped_list_like_person_dump"],
            "political_copypasta": counters["dropped_political_copypasta"],
            "required_labels": counters["dropped_required_labels"],
        },
        "selected_stats": {
            "avg_quality_score": (sum(quality_scores) / len(quality_scores)) if quality_scores else None,
            "avg_record_score": (sum(record_scores) / len(record_scores)) if record_scores else None,
            "avg_entities_per_tip": (sum(entity_counts) / len(entity_counts)) if entity_counts else None,
            "max_entities_per_tip": max(entity_counts) if entity_counts else None,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rank pseudolabel candidates for manual review.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL with predicted entities.")
    parser.add_argument("--output-csv", required=True, help="Output CSV with ranked candidates.")
    parser.add_argument("--output-jsonl", default="", help="Optional JSONL output with ranked rows.")
    parser.add_argument("--output-html", default="", help="Optional HTML review of ranked candidates.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--title", default="Ranked Pseudolabel Candidates", help="HTML title when --output-html is used.")
    parser.add_argument("--top-n", type=int, default=100, help="Maximum number of ranked candidates to export (0 = all).")
    parser.add_argument(
        "--record-score-fields",
        default=",".join(DEFAULT_RECORD_SCORE_FIELDS),
        help="Comma-separated record-level score fields to try in order.",
    )
    parser.add_argument(
        "--entity-score-fields",
        default=",".join(DEFAULT_ENTITY_SCORE_FIELDS),
        help="Comma-separated entity-level score fields to try in order.",
    )
    parser.add_argument("--label-field", default="label", help="Entity label field.")
    parser.add_argument("--min-record-score", type=float, default=0.0, help="Minimum record score to keep when present.")
    parser.add_argument("--min-entities", type=int, default=0, help="Minimum entities per tip required to keep (0 = disabled).")
    parser.add_argument("--max-entities", type=int, default=0, help="Maximum entities per tip allowed before dropping (0 = unlimited).")
    parser.add_argument("--min-text-length", type=int, default=0, help="Minimum text length in characters required to keep (0 = disabled).")
    parser.add_argument("--max-low-score-share", type=float, default=1.0, help="Maximum allowed share of low-score entities before dropping.")
    parser.add_argument("--max-location-ratio", type=float, default=1.0, help="Maximum allowed share of Location entities before dropping.")
    parser.add_argument(
        "--max-short-span-ratio",
        type=float,
        default=1.0,
        help="Maximum allowed share of short spans before dropping.",
    )
    parser.add_argument("--short-span-max-chars", type=int, default=3, help="Spans of this length or shorter count as short.")
    parser.add_argument(
        "--high-entity-score-threshold",
        type=float,
        default=0.8,
        help="Entity score threshold used to compute high_score_share.",
    )
    parser.add_argument(
        "--low-entity-score-threshold",
        type=float,
        default=0.6,
        help="Entity score threshold used to compute low_score_share.",
    )
    parser.add_argument(
        "--required-labels",
        default="",
        help="Optional comma-separated labels; candidate must contain at least one of them.",
    )
    parser.add_argument(
        "--drop-generic-entity-texts",
        action="store_true",
        help="Drop candidates containing generic normalized entity texts such as urgente, batalhao, p2, or punctuation-only spans.",
    )
    parser.add_argument(
        "--drop-list-like-person-dumps",
        action="store_true",
        help="Drop candidates that look like artificial comma/semicolon-separated person lists with weak narrative structure.",
    )
    parser.add_argument(
        "--drop-political-copypasta",
        action="store_true",
        help="Drop candidates that match political copypasta patterns such as corruption + PEC/Lula + long named lists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    ranked_rows, counters = rank_candidates(
        rows,
        record_score_fields=_parse_csv(args.record_score_fields) or list(DEFAULT_RECORD_SCORE_FIELDS),
        entity_score_fields=_parse_csv(args.entity_score_fields) or list(DEFAULT_ENTITY_SCORE_FIELDS),
        label_field=args.label_field,
        min_record_score=args.min_record_score,
        min_entities=args.min_entities,
        max_entities=args.max_entities,
        min_text_length=args.min_text_length,
        max_low_score_share=args.max_low_score_share,
        max_location_ratio=args.max_location_ratio,
        max_short_span_ratio=args.max_short_span_ratio,
        drop_generic_entity_texts=args.drop_generic_entity_texts,
        drop_list_like_person_dumps=args.drop_list_like_person_dumps,
        drop_political_copypasta=args.drop_political_copypasta,
        short_span_max_chars=args.short_span_max_chars,
        high_entity_score_threshold=args.high_entity_score_threshold,
        low_entity_score_threshold=args.low_entity_score_threshold,
        required_labels=set(_parse_csv(args.required_labels)),
    )
    if args.top_n > 0:
        ranked_rows = ranked_rows[: args.top_n]
    summary = build_summary(ranked_rows, counters, args)

    write_csv(args.output_csv, ranked_rows)
    print(f"Saved CSV: {args.output_csv}")

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, ranked_rows)
        print(f"Saved JSONL: {args.output_jsonl}")

    if args.output_html:
        render_ranked_html(ranked_rows, args.output_html, args.title)
        print(f"Saved HTML: {args.output_html}")

    if args.summary_json:
        target = Path(args.summary_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved summary JSON: {args.summary_json}")

    print(f"Candidates kept: {summary['rows_kept']}/{summary['rows_total']}")
    selected_stats = summary["selected_stats"]
    if selected_stats["avg_quality_score"] is not None:
        print(f"Average quality score: {selected_stats['avg_quality_score']:.4f}")
    if selected_stats["avg_record_score"] is not None:
        print(f"Average record score: {selected_stats['avg_record_score']:.4f}")


if __name__ == "__main__":
    main()
