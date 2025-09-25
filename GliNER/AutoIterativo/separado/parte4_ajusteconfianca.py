#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 4 — Boost por contexto e score_relato_confianca
----------------------------------------------------
Regra solicitada:
  - Para cada *relato* (registro), verifique se os campos
    `logradouroLocal`, `bairroLocal`, `cidadeLocal`, `pontodeReferenciaLocal`
    estão contidos no texto do próprio relato (campo `relato`; se ausente,
    usa-se `text` como fallback).
  - Se **qualquer** desses valores aparecer no texto, o **score de cada
    entidade** deve ser **multiplicado por 1.2** (com clamp em [0,1]).
  - Em seguida, calcular um novo `score_relato_confianca` como a **média**
    dos scores ajustados das entidades do relato.
  - Salvar tudo em um **JSONL** de saída.

Observações:
  - Por padrão, o campo de score por entidade a ser ajustado é `score_ts`.
    Se a entidade não tiver `score_ts`, tentamos `score_iso` e depois `score`.
  - A busca é *casefold* e sem acentos (normalização básica) para comparar
    os campos de localização com o texto do relato.
  - Opcional: `--per_match` multiplica por 1.2 **por campo que casar** (ex.: 2 matches ⇒ 1.2^2).

Uso (exemplo):
  python parte4_confianca.py \
    --in ./saida_predicoes/preds_calibradas.jsonl \
    --out ./saida_predicoes/preds_confianca.jsonl \
    --score_key score_ts

  # Para compor o fator por número de matches encontrados:
  python parte4_confianca.py \
    --in ./saida_predicoes/preds_calibradas.jsonl \
    --out ./saida_predicoes/preds_confianca_permatch.jsonl \
    --score_key score_ts --per_match
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
    # lower/casefold + remove acentos
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


def count_location_matches(rec: Dict[str, Any], text_field: Optional[str] = None) -> Tuple[int, bool, List[str]]:
    """Conta quantos dos campos de localização aparecem no texto do relato.
    Retorna (n_matches, any_match, matched_keys).
    """
    txt = rec.get(text_field) if text_field else _get_relato_text(rec)
    txt_n = _norm(str(txt))
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

def _get_base_entity_score(ent: Dict[str, Any], pref_key: str = "score_ts") -> Tuple[Optional[float], str]:
    # Tenta pref_key, depois score_iso e score
    keys_try = [pref_key]
    if pref_key != "score_iso":
        keys_try.append("score_iso")
    if pref_key != "score":
        keys_try.append("score")
    for k in keys_try:
        if k in ent:
            try:
                return float(ent.get(k)), k
            except Exception:
                pass
    return None, ""


def adjust_entity_scores(rec: Dict[str, Any], score_key: str, factor: float = 1.2, per_match: bool = False,
                         clamp01: bool = True) -> Dict[str, Any]:
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
        base, used_from = _get_base_entity_score(ee, pref_key=score_key)
        if base is None:
            # não há score compatível: mantém
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

def process(in_path: str | Path, out_path: str | Path, score_key: str = "score_ts",
           factor: float = 1.2, per_match: bool = False, clamp01: bool = True) -> None:
    items = read_jsonl(in_path)
    out: List[Dict[str, Any]] = []
    for rec in items:
        rr = adjust_entity_scores(rec, score_key=score_key, factor=factor, per_match=per_match, clamp01=clamp01)
        out.append(rr)
    write_jsonl(out, out_path)
    print(f"[ok] Ajuste aplicado (factor={factor}{' per-match' if per_match else ''}) e salvo em: {out_path}  (n={len(out)})")


# ---------------------------------------------------------------------------
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parte 4 — Boost por contexto e score_relato_confianca")
    p.add_argument("--in", dest="inp", required=True, help="JSONL de entrada (predições, idealmente calibradas)")
    p.add_argument("--out", dest="out", required=True, help="JSONL de saída com score_confianca e score_relato_confianca")
    p.add_argument("--score_key", default="score_ts", help="Campo de score preferido por entidade (padrão: score_ts)")
    p.add_argument("--factor", type=float, default=1.2, help="Fator multiplicativo quando há match no texto (padrão: 1.2)")
    p.add_argument("--per_match", action="store_true", help="Multiplica por 'factor' para CADA campo que casar (1.2^k)")
    p.add_argument("--no_clamp", action="store_true", help="Não restringe o score final a [0,1]")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    process(
        in_path=args.inp,
        out_path=args.out,
        score_key=args.score_key,
        factor=args.factor,
        per_match=args.per_match,
        clamp01=(not args.no_clamp),
    )


if __name__ == "__main__":
    main()
