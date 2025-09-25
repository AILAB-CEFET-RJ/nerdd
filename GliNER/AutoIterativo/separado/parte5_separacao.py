#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 5 — Separação por score do relato
--------------------------------------
Lê um JSONL (saída das Partes 2/4), escolhe um campo de score no nível do
registro ("score_relato_confianca" ou "score_relato"), aplica um limiar e
separa os registros em **mantidos** e **descartados**.

Definição padrão:
  - Mantidos: score >= limiar
  - Descartados: score < limiar

Saídas (em --out_dir):
  - mantidos.jsonl
  - descartados.jsonl
  - resumo.json (contagens e estatísticas simples)

Uso (exemplos):
  python parte5_separacao.py \
    --in ./saida_predicoes/preds_confianca.jsonl \
    --out_dir ./separacao \
    --score_field score_relato_confianca --thresh 0.8

  # usando score_relato
  python parte5_separacao.py \
    --in ./saida_predicoes/preds_calibradas.jsonl \
    --out_dir ./separacao_score_relato \
    --score_field score_relato --thresh 0.7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# IO helpers

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def write_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def get_record_score(rec: Dict[str, Any], field: str, fallback: Optional[str] = None,
                     missing_as_zero: bool = True) -> Optional[float]:
    if field in rec:
        return safe_float(rec.get(field), 0.0)
    if fallback and fallback in rec:
        return safe_float(rec.get(fallback), 0.0)
    return 0.0 if missing_as_zero else None


def separate(items: List[Dict[str, Any]], score_field: str, thresh: float,
             fallback_field: Optional[str] = None,
             op: str = "ge", missing_as_zero: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    disc: List[Dict[str, Any]] = []

    def _cmp(v: float) -> bool:
        if op == "ge":
            return v >= thresh
        if op == "gt":
            return v > thresh
        if op == "le":
            return v <= thresh
        if op == "lt":
            return v < thresh
        # default
        return v >= thresh

    for rec in items:
        sc = get_record_score(rec, field=score_field, fallback=fallback_field, missing_as_zero=missing_as_zero)
        if sc is None:
            # se None, trata como descartado
            rr = dict(rec)
            rr.setdefault("_separacao_info", {})
            rr["_separacao_info"].update({
                "score_field": score_field,
                "fallback": fallback_field,
                "score_usado": None,
                "thresh": thresh,
                "op": op,
                "status": "descartado",
            })
            disc.append(rr)
            continue
        rr = dict(rec)
        status = "mantido" if _cmp(sc) else "descartado"
        rr.setdefault("_separacao_info", {})
        rr["_separacao_info"].update({
            "score_field": score_field,
            "fallback": fallback_field,
            "score_usado": sc,
            "thresh": thresh,
            "op": op,
            "status": status,
        })
        (kept if status == "mantido" else disc).append(rr)
    return kept, disc


def summarize(kept: List[Dict[str, Any]], disc: List[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np
    def _scores(arr: List[Dict[str, Any]]):
        vals: List[float] = []
        for r in arr:
            info = r.get("_separacao_info", {})
            v = info.get("score_usado")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return {"n": len(arr), "mean": None, "min": None, "max": None}
        v = np.asarray(vals, dtype=float)
        return {"n": len(arr), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}

    return {"mantidos": _scores(kept), "descartados": _scores(disc)}


# ---------------------------------------------------------------------------
# Pipeline

def run(inp: str | Path, out_dir: str | Path, score_field: str, thresh: float,
        op: str = "ge", fallback_field: Optional[str] = None, missing_as_zero: bool = True,
        write_combined: bool = False) -> None:
    items = read_jsonl(inp)
    kept, disc = separate(items, score_field=score_field, thresh=thresh,
                          fallback_field=fallback_field, op=op, missing_as_zero=missing_as_zero)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    write_jsonl(kept, outp / "mantidos.jsonl")
    write_jsonl(disc, outp / "descartados.jsonl")

    if write_combined:
        combined = kept + disc
        write_jsonl(combined, outp / "combinado_rotulado.jsonl")

    resumo = summarize(kept, disc)
    resumo.update({"score_field": score_field, "fallback_field": fallback_field, "thresh": thresh, "op": op})
    with open(outp / "resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print(f"[ok] Mantidos: {len(kept)} | Descartados: {len(disc)} | Pasta: {outp}")


# ---------------------------------------------------------------------------
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parte 5 — Separação por score do relato (mantidos vs descartados)")
    p.add_argument("--in", dest="inp", required=True, help="JSONL de entrada")
    p.add_argument("--out_dir", required=True, help="Pasta de saída")
    p.add_argument("--score_field", choices=["score_relato_confianca", "score_relato"], default="score_relato_confianca",
                   help="Campo de score a considerar no nível do registro")
    p.add_argument("--thresh", type=float, required=True, help="Limiar de decisão (ex.: 0.8)")
    p.add_argument("--op", choices=["ge", "gt", "le", "lt"], default="ge", help="Operador de comparação (padrão: ge = >=)")
    p.add_argument("--fallback_field", choices=["score_relato", "score_relato_confianca"], default=None,
                   help="Campo alternativo se o principal não existir")
    p.add_argument("--missing_as_zero", action="store_true", help="Trata score ausente como 0.0 (padrão: False)")
    p.add_argument("--write_combined", action="store_true", help="Escreve arquivo combinado com rótulo mantido/descartado")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run(
        inp=args.inp,
        out_dir=args.out_dir,
        score_field=args.score_field,
        thresh=args.thresh,
        op=args.op,
        fallback_field=args.fallback_field,
        missing_as_zero=args.missing_as_zero,
        write_combined=args.write_combined,
    )


if __name__ == "__main__":
    main()
