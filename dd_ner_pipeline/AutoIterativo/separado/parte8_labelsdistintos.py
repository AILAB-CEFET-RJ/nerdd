#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 10 — Listar rótulos (span labels) distintos — apenas printa
-----------------------------------------------------------------
Lê um arquivo JSON/JSONL e **só imprime** no stdout os rótulos distintos
encontrados nas entidades/spans, com contagem (CSV: label,count).

Detecta automaticamente, em cada registro:
- "entities": [{"start":..., "end":..., "label": ...}, ...]
- "ner":      [{"start":..., "end":..., "label": ...}, ...]
- "spans":    [{"start":..., "end":..., "label": ...}, ...]

Uso:
  python parte10_list_labels_print.py --in ./arquivo.jsonl
  # campo do rótulo diferente de 'label':
  python parte10_list_labels_print.py --in ./arquivo.jsonl --field rotulo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from collections import Counter

# ---------------- IO ----------------

def _parse_jsonl(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(text.splitlines(), start=1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise ValueError(f"Entrada parece JSONL, mas a linha {i} não é JSON válido: {e}")
    return out


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _parse_jsonl(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Formato JSON inválido: esperado objeto, lista ou JSONL.")

# -------------- labels --------------

def _iter_spans(rec: Dict[str, Any]):
    for key in ("entities", "ner", "spans"):
        spans = rec.get(key)
        if isinstance(spans, list):
            for e in spans:
                yield e


def collect_labels(items: List[Dict[str, Any]], field: str = "label") -> Counter:
    cnt = Counter()
    for rec in items:
        for e in _iter_spans(rec):
            lab = e.get(field)
            if isinstance(lab, str) and lab.strip():
                cnt[lab] += 1
    return cnt

# --------------- CLI ---------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Lista rótulos distintos de um arquivo (printa no stdout)")
    ap.add_argument("--in", dest="inp", required=True, help="Caminho do arquivo JSON/JSONL")
    ap.add_argument("--field", default="label", help="Nome do campo com o rótulo dentro de cada span (padrão: label)")
    args = ap.parse_args()

    items = read_json_or_jsonl(args.inp)
    counts = collect_labels(items, field=args.field)

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    if not ordered:
        print("(nenhum rótulo encontrado)")
        return
    print("label,count")
    for lab, n in ordered:
        print(f"{lab},{n}")


if __name__ == "__main__":
    main()
