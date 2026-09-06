#!/usr/bin/env python3
"""Optimize a metadata-aware context boost factor from OOF predictions."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter
from copy import deepcopy
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl, save_jsonl
from base_model_training.paths import resolve_path
from pseudolabelling.compute_record_score import compute_record_score
from pseudolabelling.config import ContextBoostConfig
from pseudolabelling.context_boost import apply_context_boost_to_record, normalize_text
from pseudolabelling.evaluate_refit_pipeline import compute_span_metrics

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read OOF predictions, simulate metadata-aware context boost factors, "
            "and recommend a factor without retraining."
        )
    )
    parser.add_argument("--oof-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boost-factors", default="1.00,1.05,1.10,1.15,1.20,1.30,1.50")
    parser.add_argument("--target-label", default="Location")
    parser.add_argument("--labels", default="Person,Location,Organization")
    parser.add_argument("--base-score-field", default="score")
    parser.add_argument("--boosted-score-field", default="score_context_boosted_sim")
    parser.add_argument("--score-thresholds", default="0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--record-score-aggregation", choices=["mean", "max", "median", "p75", "mean_times_min"], default="p75")
    parser.add_argument("--record-thresholds", default="0.8,0.9,0.95")
    parser.add_argument("--precision-floor", type=float, default=0.90)
    parser.add_argument(
        "--recommendation-entity-threshold",
        type=float,
        default=None,
        help="Entity threshold used for choosing the recommended factor. Defaults to max --score-thresholds <= 0.8, or the first threshold.",
    )
    parser.add_argument("--text-field-priority", default="text,relato")
    parser.add_argument("--metadata-fields", default="logradouroLocal,bairroLocal,cidadeLocal,pontodeReferenciaLocal")
    parser.add_argument(
        "--metadata-sources",
        default="",
        help=(
            "Optional comma-separated JSON/JSONL files used to recover metadata for OOF rows "
            "by exact normalized text match."
        ),
    )
    parser.add_argument(
        "--metadata-source-text-fields",
        default="relato,text",
        help="Comma-separated text fields to try inside --metadata-sources.",
    )
    parser.add_argument(
        "--boost-scope",
        choices=["all-entities", "location-only", "matched-only", "location-matched-only"],
        default="location-matched-only",
    )
    parser.add_argument(
        "--match-policy",
        choices=["any-metadata-in-text", "entity-metadata-overlap"],
        default="any-metadata-in-text",
    )
    parser.add_argument("--location-labels", default="Location")
    parser.add_argument("--dedupe-mode", choices=["off", "label_text"], default="label_text")
    parser.add_argument("--max-review-rows", type=int, default=200)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    values = sorted({float(item) for item in _parse_csv(value)})
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def _pick_text(row: dict[str, Any], text_fields: list[str]) -> str:
    for field in text_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["label"])


def _score(span: dict[str, Any], field: str) -> float | None:
    try:
        value = float(span[field])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _filter_spans(spans: list[dict[str, Any]], *, score_field: str, threshold: float) -> list[dict[str, Any]]:
    output = []
    for span in spans:
        score = _score(span, score_field)
        if score is None or score >= threshold:
            output.append(span)
    return output


def _build_boost_record(row: dict[str, Any], *, base_score_field: str) -> dict[str, Any]:
    record = {
        "text": row["text"],
        "entities": deepcopy(row.get("pred_spans", [])),
    }
    source_fields = row.get("source_fields")
    if isinstance(source_fields, dict):
        record.update(source_fields)
    for key, value in row.items():
        if key in {"text", "pred_spans", "gold_spans", "pred_spans_eval", "source_fields"}:
            continue
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            record[key] = value
    for entity in record["entities"]:
        if base_score_field != "score" and base_score_field in entity and "score" not in entity:
            entity["score"] = entity[base_score_field]
    return record


def _metadata_payload(row: dict[str, Any], metadata_fields: list[str]) -> dict[str, str]:
    return {
        field: str(row[field]).strip()
        for field in metadata_fields
        if isinstance(row.get(field), str) and str(row[field]).strip()
    }


def _metadata_signature(payload: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(payload.items()))


def _build_metadata_lookup(
    source_paths: list[Path],
    *,
    text_fields: list[str],
    metadata_fields: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    stats = Counter()
    for source_path in source_paths:
        rows = load_jsonl(str(source_path))
        stats["source_files"] += 1
        stats["source_rows"] += len(rows)
        for row_index, row in enumerate(rows):
            text = _pick_text(row, text_fields)
            if not text:
                stats["source_rows_without_text"] += 1
                continue
            payload = _metadata_payload(row, metadata_fields)
            if not payload:
                stats["source_rows_without_metadata"] += 1
                continue
            key = normalize_text(text)
            if not key:
                stats["source_rows_without_text"] += 1
                continue
            grouped.setdefault(key, []).append(
                {
                    "source_path": str(source_path),
                    "source_row_index_0based": row_index,
                    "source_row_index_1based": row_index + 1,
                    "metadata": payload,
                }
            )

    lookup = {}
    for key, matches in grouped.items():
        signatures = {_metadata_signature(match["metadata"]) for match in matches}
        if len(signatures) == 1:
            lookup[key] = matches[0]
            stats["unique_text_keys"] += 1
        else:
            stats["ambiguous_text_keys"] += 1
    return lookup, dict(stats)


def _row_has_metadata(row: dict[str, Any], metadata_fields: list[str]) -> bool:
    source_fields = row.get("source_fields")
    for field in metadata_fields:
        if isinstance(row.get(field), str) and row[field].strip():
            return True
        if isinstance(source_fields, dict) and isinstance(source_fields.get(field), str) and source_fields[field].strip():
            return True
    return False


def enrich_rows_from_metadata_sources(
    rows: list[dict[str, Any]],
    *,
    source_paths: list[Path],
    source_text_fields: list[str],
    metadata_fields: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not source_paths:
        return rows, {"enabled": 0}

    lookup, lookup_stats = _build_metadata_lookup(
        source_paths,
        text_fields=source_text_fields,
        metadata_fields=metadata_fields,
    )
    enriched_rows = []
    stats = Counter(lookup_stats)
    stats["enabled"] = 1
    stats["oof_rows"] = len(rows)
    for row in rows:
        enriched = deepcopy(row)
        if _row_has_metadata(enriched, metadata_fields):
            stats["oof_rows_with_existing_metadata"] += 1
            enriched_rows.append(enriched)
            continue

        key = normalize_text(str(enriched.get("text", "")))
        match = lookup.get(key)
        if not match:
            stats["oof_rows_metadata_unmatched"] += 1
            enriched_rows.append(enriched)
            continue

        source_fields = dict(enriched.get("source_fields") or {})
        source_fields.update(match["metadata"])
        enriched["source_fields"] = source_fields
        for field, value in match["metadata"].items():
            enriched[field] = value
        enriched["metadata_match"] = {
            "status": "matched_unique_optimizer",
            "source_path": match["source_path"],
            "source_row_index_0based": match["source_row_index_0based"],
            "source_row_index_1based": match["source_row_index_1based"],
        }
        stats["oof_rows_metadata_matched"] += 1
        enriched_rows.append(enriched)
    return enriched_rows, dict(stats)


def _context_config(
    *,
    boost_factor: float,
    args: argparse.Namespace,
) -> ContextBoostConfig:
    return ContextBoostConfig(
        text_field_priority=_parse_csv(args.text_field_priority),
        metadata_fields=_parse_csv(args.metadata_fields),
        label_field="label",
        base_score_field=args.base_score_field,
        fallback_score_fields=[],
        output_score_field=args.boosted_score_field,
        output_record_score_field="_unused_record_score_context_boosted",
        boost_factor=boost_factor,
        boost_scope=args.boost_scope,
        match_policy=args.match_policy,
        location_labels=_parse_csv(args.location_labels) or [args.target_label],
        write_trace_fields=True,
        write_legacy_fields=False,
        clamp_scores=True,
    )


def _promoted_entities(
    *,
    row: dict[str, Any],
    entities: list[dict[str, Any]],
    entity_threshold: float,
    base_score_field: str,
    boosted_score_field: str,
    target_label: str,
) -> list[dict[str, Any]]:
    gold_set = {_span_key(span) for span in row.get("gold_spans", [])}
    promoted = []
    for entity in entities:
        if str(entity.get("label")) != target_label:
            continue
        before = _score(entity, base_score_field)
        after = _score(entity, boosted_score_field)
        if before is None or after is None:
            continue
        if before < entity_threshold <= after:
            promoted.append(
                {
                    "row_index_0based": row.get("row_index_0based"),
                    "row_index_1based": row.get("row_index_1based"),
                    "sample_id": row.get("sample_id"),
                    "fold": row.get("fold"),
                    "label": entity.get("label"),
                    "text": entity.get("text", ""),
                    "start": entity.get("start"),
                    "end": entity.get("end"),
                    "score_before": before,
                    "score_after": after,
                    "exact": _span_key(entity) in gold_set,
                    "context_boost_applied": bool(entity.get("_context_boost_applied")),
                    "context_boost_reason": entity.get("_context_boost_reason"),
                    "text_preview": str(row.get("text", ""))[:240],
                }
            )
    return promoted


def _record_score(record: dict[str, Any], *, score_field: str, args: argparse.Namespace) -> float | None:
    score, *_rest = compute_record_score(
        record,
        score_field=score_field,
        entity_key="entities",
        aggregation=args.record_score_aggregation,
        empty_entities_policy="zero",
        dedupe_mode=args.dedupe_mode,
        include_labels=[args.target_label],
    )
    return score


def _safe_div(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _metric_row(
    *,
    boost_factor: float,
    entity_threshold: float,
    record_threshold: float,
    labels: list[str],
    target_label: str,
    gold_by_row: list[list[dict[str, Any]]],
    pred_by_row: list[list[dict[str, Any]]],
    promoted: list[dict[str, Any]],
    accepted_count: int,
    promoted_record_count: int,
    promoted_record_exact_count: int,
) -> dict[str, Any]:
    metrics = compute_span_metrics(gold_by_row, pred_by_row, labels)
    target_metrics = metrics["per_label"].get(target_label, {})
    promoted_count = len(promoted)
    promoted_exact = sum(1 for row in promoted if row["exact"])
    return {
        "boost_factor": boost_factor,
        "entity_threshold": entity_threshold,
        "record_threshold": record_threshold,
        "micro_f1": metrics["micro"]["f1"],
        "macro_f1": metrics["macro_f1"],
        "target_label": target_label,
        "target_precision": target_metrics.get("precision"),
        "target_recall": target_metrics.get("recall"),
        "target_f1": target_metrics.get("f1"),
        "target_support": target_metrics.get("support"),
        "target_fp": metrics["per_label_errors"].get(target_label, {}).get("fp"),
        "target_fn": metrics["per_label_errors"].get(target_label, {}).get("fn"),
        "promoted_entity_count": promoted_count,
        "promoted_entity_exact_count": promoted_exact,
        "promoted_entity_precision": _safe_div(promoted_exact, promoted_count),
        "accepted_record_count": accepted_count,
        "promoted_record_count": promoted_record_count,
        "promoted_record_exact_count": promoted_record_exact_count,
        "promoted_record_precision": _safe_div(promoted_record_exact_count, promoted_record_count),
    }


def evaluate_boost_factors(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    labels: list[str],
    boost_factors: list[float],
    entity_thresholds: list[float],
    record_thresholds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    promoted_entity_rows = []
    promoted_record_rows = []

    original_records = [_build_boost_record(row, base_score_field=args.base_score_field) for row in rows]
    original_record_scores = [
        _record_score(record, score_field=args.base_score_field, args=args)
        for record in original_records
    ]

    for boost_factor in boost_factors:
        config = _context_config(boost_factor=boost_factor, args=args)
        boosted_records = []
        for row in rows:
            record = _build_boost_record(row, base_score_field=args.base_score_field)
            boosted, _stats = apply_context_boost_to_record(record, config)
            boosted_records.append(boosted)

        boosted_record_scores = [
            _record_score(record, score_field=args.boosted_score_field, args=args)
            for record in boosted_records
        ]

        for entity_threshold in entity_thresholds:
            gold_by_row = []
            pred_by_row = []
            promoted_for_threshold = []
            row_promoted_exact_flags: dict[int, bool] = {}
            for index, (row, boosted) in enumerate(zip(rows, boosted_records)):
                entities = boosted.get("entities", [])
                pred_spans = _filter_spans(
                    entities,
                    score_field=args.boosted_score_field,
                    threshold=entity_threshold,
                )
                gold_by_row.append(row.get("gold_spans", []) or [])
                pred_by_row.append(pred_spans)
                promoted = _promoted_entities(
                    row=row,
                    entities=entities,
                    entity_threshold=entity_threshold,
                    base_score_field=args.base_score_field,
                    boosted_score_field=args.boosted_score_field,
                    target_label=args.target_label,
                )
                for item in promoted:
                    enriched = dict(item)
                    enriched["boost_factor"] = boost_factor
                    enriched["entity_threshold"] = entity_threshold
                    promoted_for_threshold.append(enriched)
                row_promoted_exact_flags[index] = any(item["exact"] for item in promoted)

            promoted_entity_rows.extend(promoted_for_threshold)

            for record_threshold in record_thresholds:
                accepted_count = 0
                promoted_record_count = 0
                promoted_record_exact_count = 0
                for index, (row, before_score, after_score) in enumerate(
                    zip(rows, original_record_scores, boosted_record_scores)
                ):
                    before = before_score or 0.0
                    after = after_score or 0.0
                    if after >= record_threshold:
                        accepted_count += 1
                    if before < record_threshold <= after:
                        promoted_record_count += 1
                        has_exact = bool(row_promoted_exact_flags.get(index))
                        promoted_record_exact_count += int(has_exact)
                        promoted_record_rows.append(
                            {
                                "boost_factor": boost_factor,
                                "entity_threshold": entity_threshold,
                                "record_threshold": record_threshold,
                                "row_index_0based": row.get("row_index_0based"),
                                "row_index_1based": row.get("row_index_1based"),
                                "sample_id": row.get("sample_id"),
                                "fold": row.get("fold"),
                                "record_score_before": before,
                                "record_score_after": after,
                                "has_exact_promoted_entity": has_exact,
                                "text_preview": str(row.get("text", ""))[:240],
                            }
                        )
                metric_rows.append(
                    _metric_row(
                        boost_factor=boost_factor,
                        entity_threshold=entity_threshold,
                        record_threshold=record_threshold,
                        labels=labels,
                        target_label=args.target_label,
                        gold_by_row=gold_by_row,
                        pred_by_row=pred_by_row,
                        promoted=promoted_for_threshold,
                        accepted_count=accepted_count,
                        promoted_record_count=promoted_record_count,
                        promoted_record_exact_count=promoted_record_exact_count,
                    )
                )

    return metric_rows, promoted_entity_rows, promoted_record_rows


def choose_recommendation(
    metric_rows: list[dict[str, Any]],
    *,
    boost_factors: list[float],
    entity_threshold: float,
    record_threshold: float,
    precision_floor: float,
) -> dict[str, Any]:
    scoped = [
        row
        for row in metric_rows
        if float(row["entity_threshold"]) == entity_threshold
        and float(row["record_threshold"]) == record_threshold
        and float(row["boost_factor"]) > 1.0
    ]
    passing = [
        row
        for row in scoped
        if row["promoted_entity_precision"] is not None
        and row["promoted_entity_precision"] >= precision_floor
        and row["promoted_entity_count"] > 0
    ]
    if passing:
        best = sorted(
            passing,
            key=lambda row: (
                -int(row["promoted_entity_exact_count"]),
                float(row["boost_factor"]),
            ),
        )[0]
        status = "recommended"
    else:
        best = next(
            (
                row
                for row in metric_rows
                if float(row["boost_factor"]) == 1.0
                and float(row["entity_threshold"]) == entity_threshold
                and float(row["record_threshold"]) == record_threshold
            ),
            None,
        )
        if best is None:
            best = {
                "boost_factor": 1.0,
                "entity_threshold": entity_threshold,
                "record_threshold": record_threshold,
            }
        status = "no_boost_justified"
    return {
        "status": status,
        "recommended_boost_factor": float(best["boost_factor"]),
        "selection_rule": (
            "Choose the non-1.0 factor with the largest promoted_entity_exact_count "
            f"while promoted_entity_precision >= {precision_floor}; tie-break by smaller boost_factor. "
            "If none pass, recommend 1.0."
        ),
        "recommendation_entity_threshold": entity_threshold,
        "recommendation_record_threshold": record_threshold,
        "precision_floor": precision_floor,
        "selected_metrics": best,
        "boost_factors_tested": boost_factors,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_review_html(path: Path, promoted_entities: list[dict[str, Any]], *, max_rows: int) -> None:
    rows = promoted_entities[:max_rows]
    html_rows = []
    for row in rows:
        score_before = float(row.get("score_before") or 0.0)
        score_after = float(row.get("score_after") or 0.0)
        html_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('boost_factor', '')))}</td>"
            f"<td>{escape(str(row.get('entity_threshold', '')))}</td>"
            f"<td>{escape(str(row.get('row_index_1based', '')))}</td>"
            f"<td>{escape(str(row.get('label', '')))}</td>"
            f"<td>{escape(str(row.get('text', '')))}</td>"
            f"<td>{score_before:.4f}</td>"
            f"<td>{score_after:.4f}</td>"
            f"<td>{escape(str(row.get('exact', '')))}</td>"
            f"<td>{escape(str(row.get('text_preview', '')))}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Context Boost Factor Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    td {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Context Boost Factor Review</h1>
  <p>Showing {len(rows)} promoted entity rows.</p>
  <table>
    <thead>
      <tr>
        <th>Boost</th><th>Entity threshold</th><th>Row</th><th>Label</th>
        <th>Mention</th><th>Before</th><th>After</th><th>Exact</th><th>Text preview</th>
      </tr>
    </thead>
    <tbody>
      {''.join(html_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _default_recommendation_entity_threshold(thresholds: list[float]) -> float:
    candidates = [value for value in thresholds if value <= 0.8]
    return max(candidates) if candidates else thresholds[0]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    timer = perf_counter()
    script_dir = Path(__file__).resolve().parent
    oof_path = resolve_path(script_dir, args.oof_predictions)
    output_dir = resolve_path(script_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(str(oof_path))
    labels = _parse_csv(args.labels)
    boost_factors = _parse_floats(args.boost_factors)
    entity_thresholds = _parse_floats(args.score_thresholds)
    record_thresholds = _parse_floats(args.record_thresholds)
    metadata_fields = _parse_csv(args.metadata_fields)
    metadata_source_paths = [
        resolve_path(script_dir, source)
        for source in _parse_csv(args.metadata_sources)
    ]
    missing_metadata_sources = [str(path) for path in metadata_source_paths if not path.exists()]
    if missing_metadata_sources:
        raise FileNotFoundError(f"Metadata source file(s) not found: {missing_metadata_sources}")
    rows, metadata_enrichment_stats = enrich_rows_from_metadata_sources(
        rows,
        source_paths=metadata_source_paths,
        source_text_fields=_parse_csv(args.metadata_source_text_fields),
        metadata_fields=metadata_fields,
    )
    recommendation_entity_threshold = (
        args.recommendation_entity_threshold
        if args.recommendation_entity_threshold is not None
        else _default_recommendation_entity_threshold(entity_thresholds)
    )
    recommendation_record_threshold = record_thresholds[0]

    LOGGER.info("Loaded %s OOF rows from %s", len(rows), oof_path)
    metric_rows, promoted_entities, promoted_records = evaluate_boost_factors(
        rows,
        args=args,
        labels=labels,
        boost_factors=boost_factors,
        entity_thresholds=entity_thresholds,
        record_thresholds=record_thresholds,
    )
    recommendation = choose_recommendation(
        metric_rows,
        boost_factors=boost_factors,
        entity_threshold=recommendation_entity_threshold,
        record_threshold=recommendation_record_threshold,
        precision_floor=args.precision_floor,
    )

    _write_csv(output_dir / "boost_factor_metrics.csv", metric_rows)
    save_jsonl(str(output_dir / "promoted_entities.jsonl"), promoted_entities)
    save_jsonl(str(output_dir / "promoted_records.jsonl"), promoted_records)
    _write_json(output_dir / "boost_factor_recommendation.json", recommendation)
    summary = {
        "oof_predictions": str(oof_path),
        "rows_total": len(rows),
        "config": {
            **vars(args),
            "labels": labels,
            "boost_factors": boost_factors,
            "score_thresholds": entity_thresholds,
            "record_thresholds": record_thresholds,
        },
        "metadata_enrichment": metadata_enrichment_stats,
        "recommendation": recommendation,
        "output_files": {
            "metrics_csv": str((output_dir / "boost_factor_metrics.csv").resolve()),
            "promoted_entities_jsonl": str((output_dir / "promoted_entities.jsonl").resolve()),
            "promoted_records_jsonl": str((output_dir / "promoted_records.jsonl").resolve()),
            "recommendation_json": str((output_dir / "boost_factor_recommendation.json").resolve()),
            "review_html": str((output_dir / "boost_factor_review.html").resolve()),
        },
        "runtime_seconds": perf_counter() - timer,
    }
    _write_json(output_dir / "boost_factor_summary.json", summary)
    promoted_for_review = sorted(
        promoted_entities,
        key=lambda row: (
            float(row.get("boost_factor", 0)),
            float(row.get("entity_threshold", 0)),
            int(row.get("row_index_1based") or 0),
        ),
    )
    _write_review_html(output_dir / "boost_factor_review.html", promoted_for_review, max_rows=args.max_review_rows)
    LOGGER.info(
        "Recommendation: boost_factor=%s status=%s",
        recommendation["recommended_boost_factor"],
        recommendation["status"],
    )
    LOGGER.info("Saved boost-factor optimization artifacts to: %s", output_dir)


if __name__ == "__main__":
    main()
