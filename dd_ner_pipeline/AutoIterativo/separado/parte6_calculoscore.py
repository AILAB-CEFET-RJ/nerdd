#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 6 — Cálculo simples do score_relato
----------------------------------------
Lê um arquivo JSON ou JSONL com registros no formato do exemplo abaixo e
calcula o campo **score_relato** como a **média** dos scores das entidades.

Exemplo de registro de entrada:
{
  "assunto": "Armas",
  "relato": "Garoto botando um 38 pro alto e mostrando pros outro em Jabaquara",
  "logradouroLocal": "avenida Jabaquara",
  "bairroLocal": "Jabaquara",
  "cidadeLocal": "Paraty",
  "pontodeReferenciaLocal": "Campo de futebol",
  "text": "Garoto botando um 38 pro alto e mostrando pros outro em Jabaquara",
  "entities": [
    {"start": 0, "end": 6,  "label": "Person",   "score": 1.0},
    {"start": 56, "end": 65, "label": "Location", "score": 1.0}
  ]
}

Uso:
  python parte6_score_relato.py \
    --in  ./dados.jsonl \
    --out ./dados_com_score_relato.jsonl \
    [--score_field score]

Observações:
- Entrada pode ser JSONL (um JSON por linha), uma **lista JSON**, ou um único JSON.
- Por padrão usa o campo de score das entidades "score". Se seu arquivo tiver
  outro (ex.: "score_ts"), informe com --score_field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
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
    raise ValueError("Formato JSON inválido: esperado objeto, lista ou JSONL.")


def write_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def compute_score_relato(rec: Dict[str, Any], score_field: str = "score") -> float:
    """Média simples dos scores das entidades (ou 0.0 se não houver)."""
    ents = rec.get("entities") or rec.get("ner") or []
    vals: List[float] = []
    for e in ents:
        v = _safe_float(e.get(score_field))
        if v is not None:
            vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def process(inp: str | Path, out: str | Path, score_field: str = "score") -> None:
    items = read_json_or_jsonl(inp)
    out_items: List[Dict[str, Any]] = []
    for rec in items:
        rr = dict(rec)
        rr["score_relato"] = compute_score_relato(rr, score_field=score_field)
        out_items.append(rr)
    write_jsonl(out_items, out)
    print(f"[ok] score_relato calculado e salvo em: {out}  (n={len(out_items)})")


# ---------------------------------------------------------------------------
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parte 6 — Cálculo simples do score_relato (média das entidades)")
    p.add_argument("--in", dest="inp", required=True, help="Caminho do arquivo JSON/JSONL de entrada")
    p.add_argument("--out", dest="out", required=True, help="Caminho do JSONL de saída")
    p.add_argument("--score_field", default="score", help="Campo de score nas entidades (padrão: score)")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    process(inp=args.inp, out=args.out, score_field=args.score_field)


if __name__ == "__main__":
    main()
