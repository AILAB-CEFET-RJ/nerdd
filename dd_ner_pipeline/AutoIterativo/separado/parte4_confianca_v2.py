#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 4 (v2) — Boost por contexto e score_relato_confianca (sem depender de TS/ISO)
-----------------------------------------------------------------------------------
Regra:
  - Para cada *relato*, verifique se `logradouroLocal`, `bairroLocal`,
    `cidadeLocal`, `pontodeReferenciaLocal` aparecem no texto do relato
    (campo `relato`; se ausente, usa `text`).
  - Se **qualquer** um aparecer, multiplica o **score** de cada entidade por
    `factor` (padrão: 1.2) com clamp em [0,1].
  - Em seguida calcula `score_relato_confianca` como **média** dos scores
    ajustados das entidades.

Diferencial: por padrão usa **apenas `score` (RAW)** como base das entidades —
portanto **não depende de `score_ts`/`score_iso`**. Se quiser, você pode passar
`--fallback_keys score_ts,score_iso` para usar esses campos somente quando
`score` não existir.

Uso:
  python parte4_confianca_v2.py \
    --in ./saida_predicoes/preds.jsonl \
    --out ./saida_predicoes/preds_confianca.jsonl \
    --score_key score \
    --factor 1.2

Opcional:
  --fallback_keys score_ts,score_iso   # usa ts/iso apenas como fallback
  --per_match                          # fator aplicado por cada campo que casar (1.2^k)
  --no_clamp                           # não restringe a [0,1]
"""
from __future__ import annotations

import argparse
import json
import unicodedata
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
# Texto / matching helpers

def _norm(s: str) -> str:
    s = (s or "").casefold()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


def _get_relato_text(rec: Dict[str, Any]) -> str:
    if isinstance(rec.get("relato"), str) and rec["relato"].strip():
        return rec["relato"].strip()
    if isinstance(rec.get("text"), str) and rec["text"].strip():
        return rec["text"].strip()
    return ""


_LOC_KEYS = [
    "logradouroLocal",
    "bairroLocal",
    "cidadeLocal",
    "pontodeReferenciaLocal",
]


def count_location_matches(rec: Dict[str, Any]) -> Tuple[int, bool, List[str]]:
    txt_n = _norm(_get_relato_text(rec))
    n = 0
    matched: List[str] = []
    for k in _LOC_KEYS:
        v = rec.get(k)
        if not isinstance(v, str) or not v.strip():
            continue
        vv = _norm(v)
        if vv and vv in txt_n:
            n += 1
            matched.append(k)
    return n, (n > 0), matched


# ---------------------------------------------------------------------------
# Lógica de ajuste e score do relato

def _get_base_entity_score(ent: Dict[str, Any], pref_key: str = "score",
                           fallback_keys: Optional[List[str]] = None) -> Tuple[Optional[float], str]:
    keys_try: List[str] = []
    if pref_key:
        keys_try.append(pref_key)
    if fallback_keys:
        for k in fallback_keys:
            k = (k or "").strip()
            if k and k not in keys_try:
                keys_try.append(k)
    for k in keys_try:
        if k in ent:
            try:
                return float(ent.get(k)), k
            except Exception:
                continue
    return None, ""


def adjust_entity_scores(rec: Dict[str, Any], score_key: str, factor: float = 1.2, per_match: bool = False,
                         fallback_keys: Optional[List[str]] = None, clamp01: bool = True) -> Dict[str, Any]:
    rr = dict(rec)
    ents = rr.get("entities") if rr.get("entities") is not None else rr.get("ner")
    if not isinstance(ents, list):
        rr["score_relato_confianca"] = 0.0
        return rr

    n_matches, any_match, matched_keys = count_location_matches(rr)
    mult = (factor ** n_matches) if (per_match and any_match) else (factor if any_match else 1.0)

    new_ents: List[Dict[str, Any]] = []
    scores_for_relato: List[float] = []
    for e in ents:
        ee = dict(e)
        base, used_from = _get_base_entity_score(ee, pref_key=score_key, fallback_keys=fallback_keys)
        if base is None:
            new_ents.append(ee)
            continue
        new_score = float(base) * float(mult)
        if clamp01:
            if new_score < 0.0:
                new_score = 0.0
            elif new_score > 1.0:
                new_score = 1.0
        ee["score_confianca"] = new_score
        ee["score_confianca_from"] = used_from
        ee["score_confianca_factor"] = mult
        new_ents.append(ee)
        scores_for_relato.append(new_score)

    if rr.get("entities") is not None:
        rr["entities"] = new_ents
    else:
        rr["ner"] = new_ents

    rr["_loc_matches_count"] = n_matches
    rr["_loc_matched_keys"] = matched_keys
    rr["score_relato_confianca"] = float(sum(scores_for_relato) / len(scores_for_relato)) if scores_for_relato else 0.0
    return rr


# ---------------------------------------------------------------------------
# Pipeline

def process(in_path: str | Path, out_path: str | Path, score_key: str = "score",
           factor: float = 1.2, per_match: bool = False, fallback_keys: Optional[List[str]] = None,
           clamp01: bool = True) -> None:
    items = read_jsonl(in_path)
    out: List[Dict[str, Any]] = []
    for rec in items:
        rr = adjust_entity_scores(rec, score_key=score_key, factor=factor, per_match=per_match,
                                  fallback_keys=fallback_keys, clamp01=clamp01)
        out.append(rr)
    write_jsonl(out, out_path)
    print(f"[ok] Ajuste aplicado (factor={factor}{' per-match' if per_match else ''}) e salvo em: {out_path}  (n={len(out)})")


# ---------------------------------------------------------------------------
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parte 4 (v2) — Boost por contexto (usa 'score' por padrão)")
    p.add_argument("--in", dest="inp", required=True, help="JSONL de entrada (predições)")
    p.add_argument("--out", dest="out", required=True, help="JSONL de saída")
    p.add_argument("--score_key", default="score", help="Campo de score preferido nas entidades (padrão: score)")
    p.add_argument("--fallback_keys", default="", help="Fallbacks opcionais separados por vírgula (ex.: score_ts,score_iso)")
    p.add_argument("--factor", type=float, default=1.2, help="Fator multiplicativo quando há match (padrão: 1.2)")
    p.add_argument("--per_match", action="store_true", help="Multiplica por 'factor' para CADA campo que casar (1.2^k)")
    p.add_argument("--no_clamp", action="store_true", help="Não restringe o score final ao intervalo [0,1]")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    fallback_keys = [s.strip() for s in (args.fallback_keys or "").split(',') if s.strip()]
    process(
        in_path=args.inp,
        out_path=args.out,
        score_key=args.score_key,
        factor=args.factor,
        per_match=args.per_match,
        fallback_keys=fallback_keys or None,
        clamp01=(not args.no_clamp),
    )


if __name__ == "__main__":
    main()
