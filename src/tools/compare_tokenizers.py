#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import get_text, read_json_or_jsonl


def _parse_csv(raw_value):
    if not raw_value:
        return []
    return [piece.strip() for piece in str(raw_value).split(",") if piece.strip()]


def _load_rows(args):
    rows = []
    if args.input:
        rows.extend(read_json_or_jsonl(args.input))
    for text in args.text:
        rows.append({"text": text})
    if args.max_records > 0:
        rows = rows[: args.max_records]
    return rows


def _tokenize(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=tokenizer.is_fast,
    )
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offsets = encoded.get("offset_mapping")
    unk_token = getattr(tokenizer, "unk_token", None)
    unk_count = sum(1 for token in tokens if unk_token is not None and token == unk_token)
    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "offsets": offsets if offsets is not None else [],
        "unk_count": unk_count,
        "token_count": len(tokens),
    }


def build_comparisons(rows, fast_tokenizer, slow_tokenizer):
    comparisons = []
    for idx, row in enumerate(rows, start=1):
        text = get_text(row) or str(row.get("text", ""))
        fast = _tokenize(fast_tokenizer, text)
        slow = _tokenize(slow_tokenizer, text)
        comparisons.append(
            {
                "row_index_1based": idx,
                "text": text,
                "fast": fast,
                "slow": slow,
                "delta": {
                    "token_count": fast["token_count"] - slow["token_count"],
                    "unk_count": fast["unk_count"] - slow["unk_count"],
                    "tokens_equal": fast["tokens"] == slow["tokens"],
                },
            }
        )
    return comparisons


def build_summary(comparisons):
    token_count_deltas = [item["delta"]["token_count"] for item in comparisons]
    unk_count_deltas = [item["delta"]["unk_count"] for item in comparisons]
    changed = sum(1 for item in comparisons if not item["delta"]["tokens_equal"])
    fast_unk_rows = sum(1 for item in comparisons if item["fast"]["unk_count"] > 0)
    slow_unk_rows = sum(1 for item in comparisons if item["slow"]["unk_count"] > 0)
    return {
        "records": len(comparisons),
        "rows_with_token_differences": changed,
        "rows_with_fast_unk": fast_unk_rows,
        "rows_with_slow_unk": slow_unk_rows,
        "avg_token_count_delta_fast_minus_slow": (sum(token_count_deltas) / len(token_count_deltas)) if token_count_deltas else 0.0,
        "avg_unk_delta_fast_minus_slow": (sum(unk_count_deltas) / len(unk_count_deltas)) if unk_count_deltas else 0.0,
    }


def render_html(title, comparisons, output_path):
    rows_html = []
    for item in comparisons:
        fast_tokens = " ".join(escape(token) for token in item["fast"]["tokens"])
        slow_tokens = " ".join(escape(token) for token in item["slow"]["tokens"])
        rows_html.append(
            "<section class='report'>"
            f"<h3>Record #{item['row_index_1based']}</h3>"
            f"<div class='meta'>fast_tokens={item['fast']['token_count']} | slow_tokens={item['slow']['token_count']} | fast_unk={item['fast']['unk_count']} | slow_unk={item['slow']['unk_count']} | tokens_equal={item['delta']['tokens_equal']}</div>"
            f"<div class='meta'><b>Text</b></div><div class='text'>{escape(item['text'])}</div>"
            f"<div class='meta' style='margin-top:8px;'><b>Fast tokens</b></div><div class='text'>{fast_tokens}</div>"
            f"<div class='meta' style='margin-top:8px;'><b>Slow tokens</b></div><div class='text'>{slow_tokens}</div>"
            "</section>"
        )
    summary = build_summary(comparisons)
    summary_html = (
        "<table class='summary'>"
        f"<tr><td><b>Records</b></td><td>{summary['records']}</td></tr>"
        f"<tr><td><b>Rows with token differences</b></td><td>{summary['rows_with_token_differences']}</td></tr>"
        f"<tr><td><b>Rows with fast UNK</b></td><td>{summary['rows_with_fast_unk']}</td></tr>"
        f"<tr><td><b>Rows with slow UNK</b></td><td>{summary['rows_with_slow_unk']}</td></tr>"
        "</table>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #1f2937; }}
    .summary td {{ border: 1px solid #d1d5db; padding: 6px 10px; }}
    .summary {{ border-collapse: collapse; margin-bottom: 18px; }}
    .report {{ border: 1px solid #d1d5db; border-radius: 8px; margin: 0 0 12px 0; padding: 10px 12px; }}
    .report h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #111827; }}
    .meta {{ font-size: 12px; color: #6b7280; margin-bottom: 8px; }}
    .text {{ line-height: 1.7; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {summary_html}
  {''.join(rows_html)}
</body>
</html>"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare fast vs slow tokenizer outputs for selected texts.")
    parser.add_argument("--model-name", required=True, help="HF model/tokenizer name, e.g. urchade/gliner_multi-v2.1 or microsoft/mdeberta-v3-base.")
    parser.add_argument("--input", default="", help="Optional JSON/JSONL input; text is read from standard text fields.")
    parser.add_argument("--text", action="append", default=[], help="Optional raw text to compare. Can be repeated.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional limit of compared records.")
    parser.add_argument("--output-json", required=True, help="Output JSON file with full comparison.")
    parser.add_argument("--output-html", default="", help="Optional HTML report.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--title", default="Fast vs slow tokenizer comparison", help="HTML title.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("transformers is required to compare tokenizers.") from exc

    rows = _load_rows(args)
    if not rows:
        raise ValueError("No input rows provided. Use --input and/or --text.")

    fast_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    slow_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    comparisons = build_comparisons(rows, fast_tokenizer, slow_tokenizer)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON: {args.output_json}")

    summary = build_summary(comparisons)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved summary JSON: {args.summary_json}")

    if args.output_html:
        render_html(args.title, comparisons, args.output_html)
        print(f"Saved HTML: {args.output_html}")

    print(f"Records compared: {summary['records']}")
    print(f"Rows with token differences: {summary['rows_with_token_differences']}")
    print(f"Rows with fast UNK: {summary['rows_with_fast_unk']}")
    print(f"Rows with slow UNK: {summary['rows_with_slow_unk']}")


if __name__ == "__main__":
    main()
