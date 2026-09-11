#!/usr/bin/env python3
"""Audit omitted Person/Organization predictions in Location-only pseudolabel records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import get_spans, get_text, read_json_or_jsonl, write_jsonl
from tools.render_ner_html import render_html


def _parse_csv(raw_value: str) -> list[str]:
    return [piece.strip() for piece in str(raw_value or "").split(",") if piece.strip()]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().split()).strip(" \t\r\n.,;:!?()[]{}\"'")


def _get_id(row: dict[str, Any], id_fields: list[str]) -> str:
    for field in id_fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _valid_entities(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = get_text(row)
    entities = []
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
        entity["start"] = start
        entity["end"] = end
        entity["label"] = label
        entity["text"] = text[start:end]
        entities.append(entity)
    return entities


def _entity_score(entity: dict[str, Any], score_fields: list[str]) -> float | None:
    for field in score_fields:
        score = _safe_float(entity.get(field))
        if score is not None:
            return score
    return None


def _record_score(row: dict[str, Any]) -> float | None:
    for container in (row, row.get("_pseudolabel"), row.get("_pseudolabel_selection")):
        if not isinstance(container, dict):
            continue
        for field in ("record_score_location", "record_score", "score"):
            score = _safe_float(container.get(field))
            if score is not None:
                return score
    return None


def _max_entity_score(row: dict[str, Any], score_fields: list[str]) -> float:
    return max(
        (score for entity in _valid_entities(row) if (score := _entity_score(entity, score_fields)) is not None),
        default=-1.0,
    )


def _index_predictions(
    rows: list[dict[str, Any]], *, id_fields: list[str]
) -> tuple[dict[str, list[tuple[int, dict[str, Any]]]], dict[str, list[tuple[int, dict[str, Any]]]]]:
    by_id: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    by_text: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        row_id = _get_id(row, id_fields)
        if row_id:
            by_id[row_id].append((index, row))
        text_key = normalize_text(get_text(row))
        if text_key:
            by_text[text_key].append((index, row))
    return by_id, by_text


def _match_prediction(
    candidate: dict[str, Any],
    *,
    by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    by_text: dict[str, list[tuple[int, dict[str, Any]]]],
    id_fields: list[str],
) -> tuple[str, int | None, dict[str, Any] | None]:
    row_id = _get_id(candidate, id_fields)
    if row_id:
        matches = by_id.get(row_id, [])
        if len(matches) == 1:
            index, row = matches[0]
            return "source_id", index, row
        if len(matches) > 1:
            return "ambiguous_source_id", None, None

    text_key = normalize_text(get_text(candidate))
    matches = by_text.get(text_key, []) if text_key else []
    if len(matches) == 1:
        index, row = matches[0]
        return "normalized_text", index, row
    if len(matches) > 1:
        return "ambiguous_text", None, None
    return "unmatched", None, None


def audit_candidates(
    candidate_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    target_label: str,
    omitted_labels: set[str],
    score_fields: list[str],
    credible_score_threshold: float,
    id_fields: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_id, by_text = _index_predictions(prediction_rows, id_fields=id_fields)
    audit_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    counters: Counter = Counter()
    selected_location_counts: Counter = Counter()
    full_prediction_label_counts: Counter = Counter()
    non_target_combo_counts: Counter = Counter()

    for candidate_index, candidate in enumerate(candidate_rows, start=1):
        counters["candidate_rows"] += 1
        selected_entities = _valid_entities(candidate)
        selected_locations = [entity for entity in selected_entities if entity["label"] == target_label]
        selected_location_counts.update(entity["text"] for entity in selected_locations)
        match_status, prediction_index, prediction = _match_prediction(
            candidate,
            by_id=by_id,
            by_text=by_text,
            id_fields=id_fields,
        )
        counters[f"match_{match_status}"] += 1

        row: dict[str, Any] = {
            "candidate_index_1based": candidate_index,
            "candidate_source_id": _get_id(candidate, id_fields),
            "match_status": match_status,
            "prediction_index_1based": prediction_index,
            "record_score": _record_score(candidate),
            "selected_location_count": len(selected_locations),
            "selected_locations": " | ".join(entity["text"] for entity in selected_locations),
            "selected_location_score_max": max(
                (score for entity in selected_locations if (score := _entity_score(entity, score_fields)) is not None),
                default=None,
            ),
            "text_preview": get_text(candidate).replace("\n", " ")[:260],
        }

        if prediction is None:
            row.update(
                {
                    "full_location_count": None,
                    "full_person_count": None,
                    "full_organization_count": None,
                    "omitted_non_target_count": None,
                    "credible_omitted_count": None,
                    "credible_omitted_labels": "",
                    "max_omitted_score": None,
                    "risk_level": "unmatched",
                }
            )
            audit_rows.append(row)
            continue

        full_entities = _valid_entities(prediction)
        full_prediction_label_counts.update(entity["label"] for entity in full_entities)
        omitted = [entity for entity in full_entities if entity["label"] in omitted_labels]
        credible = [
            entity
            for entity in omitted
            if (score := _entity_score(entity, score_fields)) is not None and score >= credible_score_threshold
        ]
        credible_labels = sorted({entity["label"] for entity in credible})
        all_omitted_labels = sorted({entity["label"] for entity in omitted})
        combo = "+".join(all_omitted_labels) if all_omitted_labels else "none"
        non_target_combo_counts[combo] += 1

        if credible:
            risk_level = "credible_omitted_non_target"
            counters["rows_with_credible_omitted_non_target"] += 1
        elif omitted:
            risk_level = "only_low_score_non_target"
            counters["rows_with_low_score_non_target"] += 1
        else:
            risk_level = "no_predicted_non_target"
            counters["rows_without_predicted_non_target"] += 1

        row.update(
            {
                "full_location_count": sum(entity["label"] == target_label for entity in full_entities),
                "full_person_count": sum(entity["label"] == "Person" for entity in full_entities),
                "full_organization_count": sum(entity["label"] == "Organization" for entity in full_entities),
                "omitted_non_target_count": len(omitted),
                "credible_omitted_count": len(credible),
                "credible_omitted_labels": " | ".join(credible_labels),
                "max_omitted_score": max(
                    (score for entity in omitted if (score := _entity_score(entity, score_fields)) is not None),
                    default=None,
                ),
                "risk_level": risk_level,
            }
        )
        audit_rows.append(row)

        review = dict(prediction)
        review["decision"] = risk_level
        review["_location_only_audit"] = {
            "candidate_index_1based": candidate_index,
            "selected_locations": [entity["text"] for entity in selected_locations],
            "credible_omitted_labels": credible_labels,
            "credible_omitted_count": len(credible),
            "credible_score_threshold": credible_score_threshold,
        }
        review_rows.append(review)

    def _ratio(counter_key: str) -> float | None:
        matched = counters["candidate_rows"] - counters["match_unmatched"] - counters["match_ambiguous_text"] - counters["match_ambiguous_source_id"]
        return float(counters[counter_key] / matched) if matched else None

    summary = {
        "candidate_rows": len(candidate_rows),
        "prediction_rows": len(prediction_rows),
        "match_counts": {key.removeprefix("match_"): value for key, value in sorted(counters.items()) if key.startswith("match_")},
        "matched_candidate_rows": len(review_rows),
        "target_label": target_label,
        "omitted_labels": sorted(omitted_labels),
        "credible_score_threshold": credible_score_threshold,
        "rows_without_predicted_non_target": counters["rows_without_predicted_non_target"],
        "rows_with_low_score_non_target": counters["rows_with_low_score_non_target"],
        "rows_with_credible_omitted_non_target": counters["rows_with_credible_omitted_non_target"],
        "credible_omitted_non_target_rate_among_matched": _ratio("rows_with_credible_omitted_non_target"),
        "full_prediction_label_counts_in_matched_rows": dict(sorted(full_prediction_label_counts.items())),
        "non_target_prediction_combinations": dict(sorted(non_target_combo_counts.items())),
        "selected_location_surface_counts": dict(selected_location_counts.most_common()),
    }
    return audit_rows, review_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the risk of omitted Person/Organization labels in Location-only pseudolabel records."
    )
    parser.add_argument("--selected-jsonl", required=True, help="Location-only pseudolabel records selected for possible refitting.")
    parser.add_argument("--predictions-jsonl", required=True, help="Full, unfiltered predictions from the same inference run.")
    parser.add_argument("--output-dir", required=True, help="Directory for audit artifacts.")
    parser.add_argument("--target-label", default="Location")
    parser.add_argument("--omitted-labels", default="Person,Organization")
    parser.add_argument("--score-fields", default="score_calibrated,score", help="Comma-separated entity score fields, in priority order.")
    parser.add_argument("--credible-score-threshold", type=float, default=0.6)
    parser.add_argument("--id-fields", default="source_id,sample_id,id", help="Comma-separated row identifier fields used before text matching.")
    parser.add_argument("--max-review-rows", type=int, default=200, help="Maximum matched rows rendered to HTML; 0 renders all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_rows = read_json_or_jsonl(args.selected_jsonl)
    prediction_rows = read_json_or_jsonl(args.predictions_jsonl)
    audit_rows, review_rows, summary = audit_candidates(
        candidate_rows,
        prediction_rows,
        target_label=args.target_label,
        omitted_labels=set(_parse_csv(args.omitted_labels)),
        score_fields=_parse_csv(args.score_fields),
        credible_score_threshold=args.credible_score_threshold,
        id_fields=_parse_csv(args.id_fields),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows.sort(
        key=lambda row: (
            row["risk_level"] != "credible_omitted_non_target",
            -(row["max_omitted_score"] or -1.0),
            row["candidate_index_1based"],
        )
    )
    review_rows.sort(
        key=lambda row: (
            str(row.get("decision")) != "credible_omitted_non_target",
            -_max_entity_score(row, _parse_csv(args.score_fields)),
        )
    )
    _write_csv(output_dir / "location_only_supervision_audit.csv", audit_rows)
    write_jsonl(output_dir / "location_only_supervision_review.jsonl", review_rows)
    payload = {
        "inputs": {
            "selected_jsonl": str(Path(args.selected_jsonl).resolve()),
            "predictions_jsonl": str(Path(args.predictions_jsonl).resolve()),
        },
        "config": {
            "target_label": args.target_label,
            "omitted_labels": _parse_csv(args.omitted_labels),
            "score_fields": _parse_csv(args.score_fields),
            "credible_score_threshold": args.credible_score_threshold,
            "id_fields": _parse_csv(args.id_fields),
        },
        "summary": summary,
        "artifacts": {
            "audit_csv": str((output_dir / "location_only_supervision_audit.csv").resolve()),
            "review_jsonl": str((output_dir / "location_only_supervision_review.jsonl").resolve()),
            "review_html": str((output_dir / "location_only_supervision_review.html").resolve()),
        },
    }
    (output_dir / "location_only_supervision_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    render_html(
        review_rows,
        output_path=output_dir / "location_only_supervision_review.html",
        title="Location-only pseudolabel supervision audit",
        max_reports=args.max_review_rows,
        score_fields=_parse_csv(args.score_fields),
    )

    print(f"Saved audit CSV: {output_dir / 'location_only_supervision_audit.csv'}")
    print(f"Saved review HTML: {output_dir / 'location_only_supervision_review.html'}")
    print(
        "Matched rows={matched} | credible omitted Person/Organization rows={credible} ({rate:.1%})".format(
            matched=summary["matched_candidate_rows"],
            credible=summary["rows_with_credible_omitted_non_target"],
            rate=summary["credible_omitted_non_target_rate_among_matched"] or 0.0,
        )
    )


if __name__ == "__main__":
    main()
