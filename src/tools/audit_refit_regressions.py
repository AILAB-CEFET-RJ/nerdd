#!/usr/bin/env python3
"""Audit baseline-vs-candidate regressions on the same gold evaluation set."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.io_utils import load_jsonl, save_jsonl
from base_model_training.paths import resolve_path
from pseudolabelling.evaluate_refit_pipeline import compute_span_metrics, load_gt_jsonl_strict

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit regressions between two prediction runs on the same gold set.")
    parser.add_argument("--gold", required=True, help="Gold dataset path (JSON/JSONL with text+spans).")
    parser.add_argument("--baseline-pred", required=True, help="Baseline predictions.jsonl path.")
    parser.add_argument("--candidate-pred", required=True, help="Candidate predictions.jsonl path.")
    parser.add_argument("--output-dir", required=True, help="Directory for audit artifacts.")
    parser.add_argument("--title", default="Refit Regression Audit", help="Title used in markdown summary.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _normalize_span(span, text):
    start = int(span["start"])
    end = int(span["end"])
    label = str(span["label"])
    if end <= start or start < 0 or end > len(text):
        return None
    entity_text = str(span.get("text", text[start:end]))
    return {
        "start": start,
        "end": end,
        "label": label,
        "text": entity_text,
    }


def _normalize_gold_rows(rows):
    normalized = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        spans = []
        for span in row.get("spans", []) or []:
            normalized_span = _normalize_span(span, text)
            if normalized_span is not None:
                spans.append(normalized_span)
        normalized.append({"text": text, "spans": spans})
    return normalized


def _normalize_prediction_rows(rows):
    normalized = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        entities = row.get("entities")
        if entities is None:
            entities = row.get("spans", [])
        pred_spans = []
        for entity in entities or []:
            if not isinstance(entity, dict):
                continue
            normalized_span = _normalize_span(entity, text)
            if normalized_span is not None:
                pred_spans.append(normalized_span)
        normalized.append({"text": text, "entities": pred_spans})
    return normalized


def _exact_key(span):
    return (int(span["start"]), int(span["end"]), str(span["label"]))


def _span_overlap(left, right):
    return int(left["start"]) < int(right["end"]) and int(right["start"]) < int(left["end"])


def _row_f1(gold_spans, pred_spans):
    gold_set = {_exact_key(span) for span in gold_spans}
    pred_set = {_exact_key(span) for span in pred_spans}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _classify_missed_gold(gold_span, candidate_pred_spans):
    overlapping = [span for span in candidate_pred_spans if _span_overlap(span, gold_span)]
    if not overlapping:
        return "missing_entity"
    if any(span["label"] == gold_span["label"] for span in overlapping):
        return "boundary_or_partial"
    return "wrong_label"


def _classify_new_fp(pred_span, gold_spans):
    overlapping = [span for span in gold_spans if _span_overlap(span, pred_span)]
    if not overlapping:
        return "spurious_entity"
    if any(span["label"] == pred_span["label"] for span in overlapping):
        return "boundary_or_partial"
    return "wrong_label"


def _build_tp_fp_sets(gold_spans, pred_spans):
    gold_set = {_exact_key(span) for span in gold_spans}
    pred_set = {_exact_key(span) for span in pred_spans}
    return {
        "tp": gold_set & pred_set,
        "fp": pred_set - gold_set,
    }


def _index_rows_by_text(rows, key_name):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["text"]].append(row)
    duplicates = [text for text, grouped_rows in grouped.items() if len(grouped_rows) > 1]
    if duplicates:
        preview = duplicates[:3]
        raise ValueError(f"Found duplicated text values while indexing {key_name}: {preview}")
    return {text: grouped_rows[0] for text, grouped_rows in grouped.items()}


def _align_prediction_rows(gold_rows, pred_rows, pred_name):
    if len(pred_rows) == len(gold_rows) and all(g["text"] == p["text"] for g, p in zip(gold_rows, pred_rows)):
        return pred_rows

    pred_by_text = _index_rows_by_text(pred_rows, pred_name)
    aligned = []
    missing = []
    for row in gold_rows:
        pred_row = pred_by_text.get(row["text"])
        if pred_row is None:
            missing.append(row["text"][:80])
            aligned.append({"text": row["text"], "entities": []})
        else:
            aligned.append(pred_row)
    if missing:
        preview = missing[:3]
        LOGGER.warning("Missing %s rows while aligning %s. Examples: %s", len(missing), pred_name, preview)
    return aligned


def _audit_rows(gold_rows, baseline_rows, candidate_rows):
    regressions = []
    wins = []
    ties = []
    outcomes = Counter()
    loss_reason_counts = Counter()
    win_reason_counts = Counter()
    labels_affected = Counter()
    row_deltas = []

    baseline_pred_spans = []
    candidate_pred_spans = []
    gold_spans_all = []

    for index, (gold_row, base_row, cand_row) in enumerate(zip(gold_rows, baseline_rows, candidate_rows), start=1):
        gold_spans = gold_row["spans"]
        base_spans = base_row["entities"]
        cand_spans = cand_row["entities"]

        gold_spans_all.append([{k: span[k] for k in ("start", "end", "label")} for span in gold_spans])
        baseline_pred_spans.append([{k: span[k] for k in ("start", "end", "label")} for span in base_spans])
        candidate_pred_spans.append([{k: span[k] for k in ("start", "end", "label")} for span in cand_spans])

        base_f1 = _row_f1(gold_spans, base_spans)
        cand_f1 = _row_f1(gold_spans, cand_spans)
        delta_f1 = cand_f1 - base_f1
        row_deltas.append(delta_f1)

        base_sets = _build_tp_fp_sets(gold_spans, base_spans)
        cand_sets = _build_tp_fp_sets(gold_spans, cand_spans)

        lost_exact = base_sets["tp"] - cand_sets["tp"]
        gained_exact = cand_sets["tp"] - base_sets["tp"]
        new_fp = cand_sets["fp"] - base_sets["fp"]
        removed_fp = base_sets["fp"] - cand_sets["fp"]

        gold_by_key = {_exact_key(span): span for span in gold_spans}
        cand_by_key = {_exact_key(span): span for span in cand_spans}
        base_by_key = {_exact_key(span): span for span in base_spans}

        loss_reasons = Counter()
        for key in sorted(lost_exact):
            gold_span = gold_by_key[key]
            reason = _classify_missed_gold(gold_span, cand_spans)
            loss_reasons[reason] += 1
            labels_affected[gold_span["label"]] += 1
        lost_gold_spans = [gold_by_key[key] for key in lost_exact]
        for key in sorted(new_fp):
            pred_span = cand_by_key[key]
            if any(_span_overlap(pred_span, gold_span) for gold_span in lost_gold_spans):
                continue
            reason = _classify_new_fp(pred_span, gold_spans)
            loss_reasons[reason] += 1
            labels_affected[pred_span["label"]] += 1

        win_reasons = Counter()
        for key in sorted(gained_exact):
            gold_span = gold_by_key[key]
            win_reasons["recovered_exact_match"] += 1
            labels_affected[gold_span["label"]] += 1
        for key in sorted(removed_fp):
            pred_span = base_by_key[key]
            win_reasons["removed_false_positive"] += 1
            labels_affected[pred_span["label"]] += 1

        record = {
            "text": gold_row["text"],
            "spans": gold_spans,
            "baseline_entities": base_spans,
            "candidate_entities": cand_spans,
            "_audit": {
                "row_index_1based": index,
                "baseline_row_f1": base_f1,
                "candidate_row_f1": cand_f1,
                "delta_row_f1": delta_f1,
                "lost_exact_count": len(lost_exact),
                "gained_exact_count": len(gained_exact),
                "new_fp_count": len(new_fp),
                "removed_fp_count": len(removed_fp),
                "loss_reasons": dict(loss_reasons),
                "win_reasons": dict(win_reasons),
                "labels_affected": sorted(
                    {
                        *(gold_by_key[key]["label"] for key in lost_exact | gained_exact if key in gold_by_key),
                        *(cand_by_key[key]["label"] for key in new_fp if key in cand_by_key),
                        *(base_by_key[key]["label"] for key in removed_fp if key in base_by_key),
                    }
                ),
            },
        }

        if delta_f1 < 0:
            outcomes["loss"] += 1
            loss_reason_counts.update(loss_reasons)
            regressions.append(record)
        elif delta_f1 > 0:
            outcomes["win"] += 1
            win_reason_counts.update(win_reasons)
            wins.append(record)
        else:
            outcomes["tie"] += 1
            ties.append(record)

    regressions.sort(
        key=lambda row: (
            row["_audit"]["delta_row_f1"],
            -row["_audit"]["lost_exact_count"],
            -row["_audit"]["new_fp_count"],
        )
    )
    wins.sort(
        key=lambda row: (
            -row["_audit"]["delta_row_f1"],
            -row["_audit"]["gained_exact_count"],
            -row["_audit"]["removed_fp_count"],
        )
    )
    ties.sort(key=lambda row: row["_audit"]["row_index_1based"])

    baseline_metrics = compute_span_metrics(gold_spans_all, baseline_pred_spans, ["Person", "Location", "Organization"])
    candidate_metrics = compute_span_metrics(gold_spans_all, candidate_pred_spans, ["Person", "Location", "Organization"])

    summary = {
        "records": len(gold_rows),
        "outcomes": dict(outcomes),
        "mean_delta_row_f1": (sum(row_deltas) / len(row_deltas)) if row_deltas else 0.0,
        "baseline_micro_f1": baseline_metrics["micro"]["f1"],
        "candidate_micro_f1": candidate_metrics["micro"]["f1"],
        "micro_f1_delta": candidate_metrics["micro"]["f1"] - baseline_metrics["micro"]["f1"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "candidate_macro_f1": candidate_metrics["macro_f1"],
        "macro_f1_delta": candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "loss_reason_counts": dict(loss_reason_counts),
        "win_reason_counts": dict(win_reason_counts),
        "labels_affected": dict(labels_affected),
    }

    return {
        "summary": summary,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "regressions": regressions,
        "wins": wins,
        "ties": ties,
    }


def _render_markdown(title, payload):
    summary = payload["summary"]
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- records: {summary['records']}",
        f"- outcomes: loss={summary['outcomes'].get('loss', 0)} win={summary['outcomes'].get('win', 0)} tie={summary['outcomes'].get('tie', 0)}",
        f"- micro_f1: {summary['baseline_micro_f1']:.4f} -> {summary['candidate_micro_f1']:.4f} ({summary['micro_f1_delta']:+.4f})",
        f"- macro_f1: {summary['baseline_macro_f1']:.4f} -> {summary['candidate_macro_f1']:.4f} ({summary['macro_f1_delta']:+.4f})",
        f"- mean_delta_row_f1: {summary['mean_delta_row_f1']:+.4f}",
        "",
        "## Loss Reasons",
        "",
    ]
    if summary["loss_reason_counts"]:
        for reason, count in sorted(summary["loss_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Top Regressions", ""])
    for row in payload["regressions"][:20]:
        audit = row["_audit"]
        lines.append(
            f"- row {audit['row_index_1based']}: delta_f1={audit['delta_row_f1']:+.4f} "
            f"lost_exact={audit['lost_exact_count']} new_fp={audit['new_fp_count']} "
            f"labels={','.join(audit['labels_affected']) or 'none'} "
            f"loss_reasons={audit['loss_reasons']}"
        )
        lines.append(f"  text: {row['text'][:220].replace(chr(10), ' ')}")
    return "\n".join(lines) + "\n"


def run_audit(args):
    gold_path = resolve_path(Path(__file__).resolve().parent, args.gold)
    baseline_path = resolve_path(Path(__file__).resolve().parent, args.baseline_pred)
    candidate_path = resolve_path(Path(__file__).resolve().parent, args.candidate_pred)
    output_dir = resolve_path(Path(__file__).resolve().parent, args.output_dir)
    for path, label in [
        (gold_path, "gold"),
        (baseline_path, "baseline predictions"),
        (candidate_path, "candidate predictions"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = _normalize_gold_rows(load_gt_jsonl_strict(str(gold_path)))
    baseline_rows = _align_prediction_rows(gold_rows, _normalize_prediction_rows(load_jsonl(str(baseline_path))), "baseline")
    candidate_rows = _align_prediction_rows(gold_rows, _normalize_prediction_rows(load_jsonl(str(candidate_path))), "candidate")

    payload = _audit_rows(gold_rows, baseline_rows, candidate_rows)

    save_jsonl(str(output_dir / "regressions.jsonl"), payload["regressions"])
    save_jsonl(str(output_dir / "wins.jsonl"), payload["wins"])
    save_jsonl(str(output_dir / "ties.jsonl"), payload["ties"])
    (output_dir / "summary.json").write_text(json.dumps(payload["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(payload["baseline_metrics"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "candidate_metrics.json").write_text(
        json.dumps(payload["candidate_metrics"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "top_regressions.md").write_text(_render_markdown(args.title, payload), encoding="utf-8")

    LOGGER.info("Saved audit summary: %s", output_dir / "summary.json")
    LOGGER.info("Saved regressions: %s", output_dir / "regressions.jsonl")
    LOGGER.info("Saved wins: %s", output_dir / "wins.jsonl")
    LOGGER.info("Saved ties: %s", output_dir / "ties.jsonl")
    LOGGER.info("Saved markdown report: %s", output_dir / "top_regressions.md")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    run_audit(args)


if __name__ == "__main__":
    main()
