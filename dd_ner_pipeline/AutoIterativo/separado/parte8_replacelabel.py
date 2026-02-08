#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script rápido — troca label "Comunidade" por "Location" em JSON/JSONL
---------------------------------------------------------------------
- Lê um arquivo JSONL (um JSON por linha) **ou** JSON (lista/objeto).
- Para cada registro, percorre `entities` (ou `ner`) e substitui
  `{"label": "Comunidade"}` por `{"label": "Location"}`.
- Escreve a saída em JSONL preservando os demais campos.

Uso:
  python parte8_replace_label.py \
    --in ./entrada.jsonl \
    --out ./saida.jsonl

Opções:
  --from_label    (padrão: Comunidade)
  --to_label      (padrão: Location)
  --ci            (case-insensitive para comparação do rótulo)
  --inplace       (sobrescreve o arquivo de entrada; ignora --out)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ---------------------------------------------
# IO helpers

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
    # Tenta JSON comum primeiro
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Trata como JSONL
        return _parse_jsonl(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Formato JSON inválido.")


def write_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------
# Core

def replace_labels(items: List[Dict[str, Any]], from_label: str = "Comunidade", to_label: str = "Location", ci: bool = False) -> int:
    count = 0
    for rec in items:
        ents = rec.get("entities") if rec.get("entities") is not None else rec.get("ner")
        if not isinstance(ents, list):
            continue
        for e in ents:
            lab = e.get("label")
            if not isinstance(lab, str):
                continue
            if (lab.lower() == from_label.lower()) if ci else (lab == from_label):
                e["label"] = to_label
                count += 1
    return count


# ---------------------------------------------
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Troca label 'Comunidade' por 'Location' em JSON/JSONL")
    p.add_argument("--in", dest="inp", required=True, help="Arquivo de entrada (JSON/JSONL)")
    p.add_argument("--out", dest="out", default=None, help="Arquivo de saída JSONL (se ausente e --inplace não for usado, gera <input>.fixed.jsonl)")
    p.add_argument("--from_label", default="PontoDeReferencia", help="Rótulo de origem (padrão: Comunidade)")
    p.add_argument("--to_label", default="Location", help="Rótulo de destino (padrão: Location)")
    p.add_argument("--ci", action="store_true", help="Comparação case-insensitive")
    p.add_argument("--inplace", action="store_true", help="Sobrescreve o arquivo de entrada (formato JSONL)")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    items = read_json_or_jsonl(args.inp)
    n = replace_labels(items, from_label=args.from_label, to_label=args.to_label, ci=args.ci)

    if args.inplace:
        out_path = args.inp
    else:
        out_path = args.out or (str(Path(args.inp)) + ".fixed.jsonl")

    write_jsonl(items, out_path)
    print(f"[ok] Substituições: {n} | Saída: {out_path}")


if __name__ == "__main__":
    main()
