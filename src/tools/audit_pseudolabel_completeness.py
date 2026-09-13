#!/usr/bin/env python3
"""Audit entity predictions omitted from a selected pseudolabel dataset.

The selected dataset may retain any combination of labels. Each selected record
is matched to its full prediction record, then every full predicted span is
classified by exact span retention or omission. This exposes incomplete
supervision before the selected rows are used for refitting.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import get_spans, get_text, read_json_or_jsonl, write_jsonl


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(normalized.lower().split()).strip(" \t\r\n.,;:!?()[]{}\"'")


def _get_identifier(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _valid_entities(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = get_text(row)
    entities: list[dict[str, Any]] = []
    for raw in get_spans(row):
        if not isinstance(raw, dict):
            continue
        try:
            start = int(raw["start"])
            end = int(raw["end"])
        except (KeyError, TypeError, ValueError):
            continue
        label = str(raw.get("label", "")).strip()
        if not label or start < 0 or end <= start or end > len(text):
            continue
        entity = dict(raw)
        entity.update({"start": start, "end": end, "label": label, "text": text[start:end]})
        entities.append(entity)
    return entities


def _entity_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    return int(entity["start"]), int(entity["end"]), str(entity["label"])


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["start"]) < int(right["end"]) and int(right["start"]) < int(left["end"])


def _entity_score(entity: dict[str, Any], score_fields: list[str]) -> float | None:
    for field in score_fields:
        score = _safe_float(entity.get(field))
        if score is not None:
            return score
    return None


def _index_predictions(
    rows: list[dict[str, Any]], id_fields: list[str]
) -> tuple[dict[str, list[tuple[int, dict[str, Any]]]], dict[str, list[tuple[int, dict[str, Any]]]]]:
    by_id: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    by_text: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        identifier = _get_identifier(row, id_fields)
        if identifier:
            by_id[identifier].append((index, row))
        text_key = normalize_text(get_text(row))
        if text_key:
            by_text[text_key].append((index, row))
    return by_id, by_text


def _prediction_signature(row: dict[str, Any], score_fields: list[str]) -> tuple[tuple[Any, ...], ...]:
    """Return the audit-relevant prediction content for safe duplicate matching."""
    signature = []
    for entity in _valid_entities(row):
        signature.append(
            (
                entity["start"],
                entity["end"],
                entity["label"],
                *(repr(_entity_score(entity, [field])) for field in score_fields),
            )
        )
    return tuple(sorted(signature))


def _match_prediction(
    row: dict[str, Any],
    *,
    by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    by_text: dict[str, list[tuple[int, dict[str, Any]]]],
    id_fields: list[str],
    score_fields: list[str],
) -> tuple[str, int | None, dict[str, Any] | None]:
    identifier = _get_identifier(row, id_fields)
    if identifier:
        matches = by_id.get(identifier, [])
        if len(matches) == 1:
            return "source_id", matches[0][0], matches[0][1]
        if len(matches) > 1:
            return "ambiguous_source_id", None, None

    matches = by_text.get(normalize_text(get_text(row)), [])
    if len(matches) == 1:
        return "normalized_text", matches[0][0], matches[0][1]
    if len(matches) > 1:
        signatures = {_prediction_signature(match_row, score_fields) for _index, match_row in matches}
        if len(signatures) == 1:
            return "normalized_text_equivalent_duplicate", matches[0][0], matches[0][1]
        return "ambiguous_text", None, None
    return "unmatched", None, None


def audit_completeness(
    selected_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    score_fields: list[str],
    credible_score_threshold: float,
    id_fields: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_id, by_text = _index_predictions(prediction_rows, id_fields)
    entity_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    counters: Counter = Counter()
    status_counts: Counter = Counter()
    label_status_counts: Counter = Counter()

    for selected_index, selected in enumerate(selected_rows, start=1):
        counters["selected_rows"] += 1
        selected_entities = _valid_entities(selected)
        selected_keys = {_entity_key(entity) for entity in selected_entities}
        match_status, prediction_index, prediction = _match_prediction(
            selected,
            by_id=by_id,
            by_text=by_text,
            id_fields=id_fields,
            score_fields=score_fields,
        )
        counters[f"match_{match_status}"] += 1
        base = {
            "selected_index_1based": selected_index,
            "selected_source_id": _get_identifier(selected, id_fields),
            "match_status": match_status,
            "prediction_index_1based": prediction_index,
            "selected_entity_count": len(selected_entities),
            "selected_label_counts": json.dumps(Counter(entity["label"] for entity in selected_entities), ensure_ascii=False, sort_keys=True),
            "text_preview": get_text(selected).replace("\n", " ")[:260],
        }

        if prediction is None:
            record_rows.append(
                {
                    **base,
                    "full_entity_count": None,
                    "retained_exact_count": None,
                    "omitted_count": None,
                    "credible_omitted_count": None,
                    "max_credible_omitted_score": None,
                    "risk_level": "unmatched",
                }
            )
            continue

        full_entities = _valid_entities(prediction)
        full_keys = {_entity_key(entity) for entity in full_entities}
        selected_not_in_full = selected_keys - full_keys
        retained = 0
        omitted = 0
        credible_omitted: list[dict[str, Any]] = []

        for entity in full_entities:
            key = _entity_key(entity)
            score = _entity_score(entity, score_fields)
            overlaps = [candidate for candidate in selected_entities if _overlaps(entity, candidate)]
            if key in selected_keys:
                status = "retained_exact"
                retained += 1
            elif overlaps:
                status = "overlap_not_retained_same_label" if any(
                    candidate["label"] == entity["label"] for candidate in overlaps
                ) else "overlap_not_retained_label_conflict"
                omitted += 1
            else:
                status = "omitted_exact"
                omitted += 1

            credible = status != "retained_exact" and score is not None and score >= credible_score_threshold
            if credible:
                credible_omitted.append(entity)
            status_counts[status] += 1
            label_status_counts[(entity["label"], status)] += 1
            entity_rows.append(
                {
                    **base,
                    "full_start": entity["start"],
                    "full_end": entity["end"],
                    "full_label": entity["label"],
                    "full_mention": entity["text"],
                    "score": score,
                    "status": status,
                    "credible_omission": credible,
                    "overlap_selected_labels": " | ".join(sorted({item["label"] for item in overlaps})),
                    "overlap_selected_mentions": " | ".join(item["text"] for item in overlaps),
                }
            )

        for start, end, label in sorted(selected_not_in_full):
            status_counts["selected_not_in_full_prediction"] += 1
            label_status_counts[(label, "selected_not_in_full_prediction")] += 1
            entity_rows.append(
                {
                    **base,
                    "full_start": start,
                    "full_end": end,
                    "full_label": label,
                    "full_mention": get_text(selected)[start:end],
                    "score": None,
                    "status": "selected_not_in_full_prediction",
                    "credible_omission": False,
                    "overlap_selected_labels": label,
                    "overlap_selected_mentions": get_text(selected)[start:end],
                }
            )

        risk_level = "credible_omission" if credible_omitted else ("low_score_omission" if omitted else "complete")
        counters[f"risk_{risk_level}"] += 1
        record = {
            **base,
            "full_entity_count": len(full_entities),
            "full_label_counts": json.dumps(Counter(entity["label"] for entity in full_entities), ensure_ascii=False, sort_keys=True),
            "retained_exact_count": retained,
            "omitted_count": omitted,
            "credible_omitted_count": len(credible_omitted),
            "credible_omitted_labels": " | ".join(sorted({entity["label"] for entity in credible_omitted})),
            "max_credible_omitted_score": max(
                (_entity_score(entity, score_fields) for entity in credible_omitted), default=None
            ),
            "risk_level": risk_level,
        }
        record_rows.append(record)
        if risk_level != "complete":
            review_rows.append(
                {
                    **record,
                    "selected_entities": selected_entities,
                    "credible_omitted_entities": [
                        {
                            "text": entity["text"],
                            "label": entity["label"],
                            "start": entity["start"],
                            "end": entity["end"],
                            "score": _entity_score(entity, score_fields),
                        }
                        for entity in credible_omitted
                    ],
                }
            )

    matched = counters["selected_rows"] - counters["match_unmatched"] - counters["match_ambiguous_text"] - counters["match_ambiguous_source_id"]
    summary = {
        "selected_rows": len(selected_rows),
        "prediction_rows": len(prediction_rows),
        "matched_rows": matched,
        "match_counts": {key.removeprefix("match_"): value for key, value in sorted(counters.items()) if key.startswith("match_")},
        "credible_score_threshold": credible_score_threshold,
        "record_risk_counts": {key.removeprefix("risk_"): value for key, value in sorted(counters.items()) if key.startswith("risk_")},
        "credible_omission_rate_among_matched": counters["risk_credible_omission"] / matched if matched else None,
        "entity_status_counts": dict(sorted(status_counts.items())),
        "entity_status_counts_by_label": {
            label: {status: count for (row_label, status), count in sorted(label_status_counts.items()) if row_label == label}
            for label in sorted({label for label, _status in label_status_counts})
        },
    }
    return entity_rows, record_rows, review_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_review_html(path: Path, rows: list[dict[str, Any]], title: str, max_rows: int) -> None:
    displayed = rows[:max_rows] if max_rows > 0 else rows
    items = []
    for row in displayed:
        selected = ", ".join(f"{entity['label']}: {entity['text']}" for entity in row["selected_entities"])
        omitted = ", ".join(
            f"{entity['label']}: {entity['text']} ({entity['score']:.4f})"
            for entity in row["credible_omitted_entities"]
        ) or "None above threshold"
        items.append(
            "<article><h2>Row {index} - {risk}</h2><p><strong>Selected:</strong> {selected}</p>"
            "<p><strong>Credible omissions:</strong> {omitted}</p><pre>{text}</pre></article>".format(
                index=row["selected_index_1based"],
                risk=html.escape(str(row["risk_level"])),
                selected=html.escape(selected),
                omitted=html.escape(omitted),
                text=html.escape(row["text_preview"]),
            )
        )
    content = "\n".join(items) or "<p>No incomplete matched rows.</p>"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title>"
        "<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}}article{{border-top:1px solid #ccc;padding:12px 0}}"
        "pre{{white-space:pre-wrap;background:#f5f5f5;padding:10px}}</style></head><body><h1>{title}</h1>{content}</body></html>".format(
            title=html.escape(title), content=content
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit omitted entities in selected pseudolabel records.")
    parser.add_argument("--selected-jsonl", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-fields", default="score_calibrated,score")
    parser.add_argument("--credible-score-threshold", type=float, default=0.8)
    parser.add_argument("--id-fields", default="source_id,sample_id,id")
    parser.add_argument("--max-review-rows", type=int, default=200)
    parser.add_argument("--title", default="Pseudolabel supervision completeness audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_rows = read_json_or_jsonl(args.selected_jsonl)
    prediction_rows = read_json_or_jsonl(args.predictions_jsonl)
    entity_rows, record_rows, review_rows, summary = audit_completeness(
        selected_rows,
        prediction_rows,
        score_fields=_parse_csv(args.score_fields),
        credible_score_threshold=args.credible_score_threshold,
        id_fields=_parse_csv(args.id_fields),
    )
    record_rows.sort(key=lambda row: (row["risk_level"] != "credible_omission", -(row["max_credible_omitted_score"] or -1.0)))
    review_rows.sort(key=lambda row: -(row["max_credible_omitted_score"] or -1.0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "entity_completeness_audit.csv", entity_rows)
    _write_csv(output_dir / "record_completeness_audit.csv", record_rows)
    write_jsonl(output_dir / "completeness_review.jsonl", review_rows)
    _write_review_html(output_dir / "completeness_review.html", review_rows, args.title, args.max_review_rows)
    payload = {
        "inputs": {
            "selected_jsonl": str(Path(args.selected_jsonl).resolve()),
            "predictions_jsonl": str(Path(args.predictions_jsonl).resolve()),
        },
        "config": vars(args),
        "summary": summary,
        "artifacts": {
            "entity_audit_csv": str((output_dir / "entity_completeness_audit.csv").resolve()),
            "record_audit_csv": str((output_dir / "record_completeness_audit.csv").resolve()),
            "review_jsonl": str((output_dir / "completeness_review.jsonl").resolve()),
            "review_html": str((output_dir / "completeness_review.html").resolve()),
        },
    }
    (output_dir / "completeness_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved entity audit: {output_dir / 'entity_completeness_audit.csv'}")
    print(f"Saved record audit: {output_dir / 'record_completeness_audit.csv'}")
    print(f"Saved review HTML: {output_dir / 'completeness_review.html'}")
    print(
        "Matched rows={matched} | credible omission rows={credible} ({rate:.1%})".format(
            matched=summary["matched_rows"],
            credible=summary["record_risk_counts"].get("credible_omission", 0),
            rate=summary["credible_omission_rate_among_matched"] or 0.0,
        )
    )


if __name__ == "__main__":
    main()
