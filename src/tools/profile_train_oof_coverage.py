#!/usr/bin/env python3
"""Profile labeled-train coverage and strict OOF NER errors by entity form."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl


DESIGNATOR_ALIASES = {
    "av": "avenida",
    "aven": "avenida",
    "avenida": "avenida",
    "bairro": "bairro",
    "bar": "bar",
    "cidade": "cidade",
    "comunidade": "comunidade",
    "complexo": "complexo",
    "cond": "condominio",
    "condominio": "condominio",
    "estrada": "estrada",
    "estacao": "estacao",
    "estado": "estado",
    "favela": "favela",
    "igreja": "igreja",
    "morro": "morro",
    "municipio": "municipio",
    "ponte": "ponte",
    "posto": "posto",
    "praca": "praca",
    "quadra": "quadra",
    "rodovia": "rodovia",
    "rua": "rua",
    "r": "rua",
    "trav": "travessa",
    "travessa": "travessa",
    "trv": "travessa",
    "vila": "vila",
}

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().split()).strip(" \t\r\n.,;:!?()[]{}\"'")


def _span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["label"])


def _valid_span(text: str, span: dict[str, Any]) -> bool:
    try:
        start = int(span["start"])
        end = int(span["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= start < end <= len(text) and bool(str(span.get("label", "")).strip())


def _span_text(text: str, span: dict[str, Any]) -> str:
    return text[int(span["start"]) : int(span["end"])]


def _word_context(text: str, start: int, end: int) -> tuple[str, str]:
    tokens = [(match.start(), match.end(), normalize_text(match.group(0))) for match in WORD_PATTERN.finditer(text)]
    left = "<BOS>"
    right = "<EOS>"
    for token_start, token_end, token in tokens:
        if token_end <= start:
            left = token
        elif token_start >= end:
            right = token
            break
    return left or "<BOS>", right or "<EOS>"


def _designation(mention: str) -> str:
    first = next(iter(WORD_PATTERN.findall(normalize_text(mention))), "")
    return DESIGNATOR_ALIASES.get(first, "bare_or_other")


def _token_length_bucket(mention: str) -> str:
    count = len(WORD_PATTERN.findall(mention))
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    return "4+"


def _frequency_bucket(count: int) -> str:
    if count <= 0:
        return "unseen_in_train"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3-5"
    return "6+"


def target_features(text: str, span: dict[str, Any], mention_frequency: Counter) -> dict[str, str]:
    mention = _span_text(text, span)
    normalized_mention = normalize_text(mention)
    left, right = _word_context(text, int(span["start"]), int(span["end"]))
    return {
        "all": "ALL",
        "designator": _designation(mention),
        "span_token_length": _token_length_bucket(mention),
        "train_mention_frequency": _frequency_bucket(mention_frequency[normalized_mention]),
        "left_context": left,
        "right_context": right,
        "context_template": f"{left} __ENTITY__ {right}",
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return float(2.0 * precision * recall / (precision + recall))


def _counter_record() -> Counter:
    return Counter({"gold_support": 0, "pred_support": 0, "exact_tp": 0, "false_negative": 0, "false_positive": 0})


def _update_bucket(
    buckets: dict[tuple[str, str], Counter],
    features: dict[str, str],
    field: str,
) -> None:
    for bucket_type, bucket in features.items():
        buckets[(bucket_type, bucket)][field] += 1


def build_profile(
    train_rows: list[dict[str, Any]],
    oof_rows: list[dict[str, Any]],
    *,
    target_label: str,
    pred_field: str,
    min_bucket_support: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mention_frequency: Counter = Counter()
    train_label_counts: Counter = Counter()
    report_label_combinations: Counter = Counter()
    target_reports = 0

    for row in train_rows:
        text = str(row.get("text", ""))
        labels = set()
        for span in row.get("spans", []) or []:
            if not isinstance(span, dict) or not _valid_span(text, span):
                continue
            label = str(span["label"])
            labels.add(label)
            train_label_counts[label] += 1
            if label == target_label:
                mention_frequency[normalize_text(_span_text(text, span))] += 1
        if target_label in labels:
            target_reports += 1
        report_label_combinations[" + ".join(sorted(labels)) if labels else "none"] += 1

    buckets: dict[tuple[str, str], Counter] = defaultdict(_counter_record)
    oof_rows_with_target_gold = 0
    oof_rows_with_target_pred = 0
    target_gold_total = 0
    target_pred_total = 0

    for row in oof_rows:
        text = str(row.get("text", ""))
        gold = [
            span
            for span in row.get("gold_spans", []) or []
            if isinstance(span, dict) and _valid_span(text, span) and str(span["label"]) == target_label
        ]
        pred = [
            span
            for span in row.get(pred_field, []) or []
            if isinstance(span, dict) and _valid_span(text, span) and str(span["label"]) == target_label
        ]
        gold_keys = {_span_key(span) for span in gold}
        pred_keys = {_span_key(span) for span in pred}
        if gold:
            oof_rows_with_target_gold += 1
        if pred:
            oof_rows_with_target_pred += 1

        for span in gold:
            features = target_features(text, span, mention_frequency)
            _update_bucket(buckets, features, "gold_support")
            target_gold_total += 1
            if _span_key(span) in pred_keys:
                _update_bucket(buckets, features, "exact_tp")
            else:
                _update_bucket(buckets, features, "false_negative")

        for span in pred:
            features = target_features(text, span, mention_frequency)
            _update_bucket(buckets, features, "pred_support")
            target_pred_total += 1
            if _span_key(span) not in gold_keys:
                _update_bucket(buckets, features, "false_positive")

    bucket_rows = []
    for (bucket_type, bucket), counts in buckets.items():
        support = max(counts["gold_support"], counts["pred_support"])
        if bucket_type != "all" and support < min_bucket_support:
            continue
        precision = _safe_ratio(counts["exact_tp"], counts["pred_support"])
        recall = _safe_ratio(counts["exact_tp"], counts["gold_support"])
        bucket_rows.append(
            {
                "bucket_type": bucket_type,
                "bucket": bucket,
                "gold_support": counts["gold_support"],
                "pred_support": counts["pred_support"],
                "exact_tp": counts["exact_tp"],
                "false_negative": counts["false_negative"],
                "false_positive": counts["false_positive"],
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "false_negative_rate": _safe_ratio(counts["false_negative"], counts["gold_support"]),
            }
        )
    bucket_rows.sort(
        key=lambda row: (
            row["bucket_type"],
            -row["false_negative"],
            -row["gold_support"],
            row["bucket"],
        )
    )

    combination_rows = [
        {
            "label_combination": combination,
            "reports": count,
            "share_of_train_reports": float(count / len(train_rows)) if train_rows else None,
            "contains_target_label": target_label in combination.split(" + "),
        }
        for combination, count in report_label_combinations.most_common()
    ]

    target_unique_mentions = len(mention_frequency)
    target_singletons = sum(count == 1 for count in mention_frequency.values())
    all_bucket = next(
        (row for row in bucket_rows if row["bucket_type"] == "all" and row["bucket"] == "ALL"),
        None,
    )
    if all_bucket is None:
        raise ValueError(
            f"No valid {target_label!r} gold or predicted spans were found in the OOF input. "
            "Check --target-label and --pred-field."
        )
    summary = {
        "target_label": target_label,
        "train": {
            "rows": len(train_rows),
            "label_span_counts": dict(sorted(train_label_counts.items())),
            "target_reports": target_reports,
            "target_unique_mentions": target_unique_mentions,
            "target_singleton_mentions": target_singletons,
            "target_singleton_mention_share": _safe_ratio(target_singletons, target_unique_mentions),
        },
        "oof": {
            "rows": len(oof_rows),
            "prediction_field": pred_field,
            "rows_with_target_gold": oof_rows_with_target_gold,
            "rows_with_target_predictions": oof_rows_with_target_pred,
            "target_gold_spans": target_gold_total,
            "target_predicted_spans": target_pred_total,
            "target_metrics": all_bucket,
        },
        "config": {"min_bucket_support": min_bucket_support},
    }
    return bucket_rows, combination_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _write_html(path: Path, bucket_rows: list[dict[str, Any]], combination_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(bucket_rows, key=lambda row: (-row["false_negative"], -row["gold_support"]))[:80]
    metrics_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['bucket_type']))}</td>"
        f"<td>{html.escape(str(row['bucket']))}</td>"
        f"<td>{row['gold_support']}</td><td>{row['pred_support']}</td>"
        f"<td>{row['exact_tp']}</td><td>{row['false_negative']}</td><td>{row['false_positive']}</td>"
        f"<td>{_format_metric(row['precision'])}</td><td>{_format_metric(row['recall'])}</td><td>{_format_metric(row['f1'])}</td>"
        "</tr>"
        for row in rows
    )
    combinations_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['label_combination']))}</td><td>{row['reports']}</td>"
        f"<td>{_format_metric(row['share_of_train_reports'])}</td>"
        "</tr>"
        for row in combination_rows
    )
    target = summary["oof"]["target_metrics"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Train and OOF Coverage Profile</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#1f2937}}table{{border-collapse:collapse;margin:16px 0;width:100%}}th,td{{border:1px solid #d1d5db;padding:6px;text-align:left}}th{{background:#f3f4f6}}.meta{{color:#4b5563}}</style>
</head><body>
<h1>Train and OOF Coverage Profile: {html.escape(summary['target_label'])}</h1>
<p class="meta">Train rows: {summary['train']['rows']} | OOF rows: {summary['oof']['rows']} | Strict OOF F1: {_format_metric(target['f1'])}</p>
<h2>Highest false-negative buckets</h2>
<table><tr><th>Type</th><th>Bucket</th><th>Gold</th><th>Pred</th><th>TP</th><th>FN</th><th>FP</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{metrics_html}</table>
<h2>Label combinations in train reports</h2>
<table><tr><th>Labels</th><th>Reports</th><th>Share</th></tr>{combinations_html}</table>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile train coverage and strict OOF errors by NER entity form.")
    parser.add_argument("--train", required=True, help="Labeled train JSON or JSONL.")
    parser.add_argument("--oof-predictions", required=True, help="OOF predictions JSONL from mine_train_oof_errors.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for coverage artifacts.")
    parser.add_argument("--target-label", default="Location", help="Label to profile.")
    parser.add_argument("--pred-field", default="pred_spans_eval", help="OOF prediction field used for strict metrics.")
    parser.add_argument("--min-bucket-support", type=int, default=10, help="Minimum max(gold, pred) support for non-global buckets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_json_or_jsonl(args.train)
    oof_rows = read_json_or_jsonl(args.oof_predictions)
    bucket_rows, combination_rows, summary = build_profile(
        train_rows,
        oof_rows,
        target_label=args.target_label,
        pred_field=args.pred_field,
        min_bucket_support=args.min_bucket_support,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "target_bucket_metrics.csv", bucket_rows)
    _write_csv(output_dir / "train_label_combinations.csv", combination_rows)
    payload = {
        "inputs": {"train": str(Path(args.train).resolve()), "oof_predictions": str(Path(args.oof_predictions).resolve())},
        "summary": summary,
        "artifacts": {
            "target_bucket_metrics_csv": str((output_dir / "target_bucket_metrics.csv").resolve()),
            "train_label_combinations_csv": str((output_dir / "train_label_combinations.csv").resolve()),
            "review_html": str((output_dir / "coverage_review.html").resolve()),
        },
    }
    (output_dir / "coverage_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_html(output_dir / "coverage_review.html", bucket_rows, combination_rows, summary)

    overall = summary["oof"]["target_metrics"]
    print(f"Saved coverage profile to: {output_dir}")
    print(f"{args.target_label} strict OOF: P={_format_metric(overall['precision'])} R={_format_metric(overall['recall'])} F1={_format_metric(overall['f1'])}")


if __name__ == "__main__":
    main()
