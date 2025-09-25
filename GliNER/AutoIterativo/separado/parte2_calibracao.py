#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 2 — Calibração (Isotonic Regression e Temperature Scaling)
----------------------------------------------------------------
Lê um arquivo de predições (JSONL) gerado na Parte 1, ajusta um
calibrador com base em um CSV de validação contendo Score (probabilidade)
e Validacao (0/1), aplica as calibrações às entidades e, ao final,
calcula o "score_relato".

Entrada esperada do CSV de calibração (exemplo: comparacao_calibracao.csv):
    - Coluna 'Score'     : probabilidade predita (0..1)
    - Coluna 'Validacao' : rótulo verdadeiro (0/1)

Saída:
    - JSONL com os registros originais + campos adicionais por entidade:
        * score_ts  (se --method inclui ts)
        * score_iso (se --method inclui iso)
      E também no nível do registro:
        * score_relato_ts
        * score_relato_iso
        * score_relato (opcional, quando --preferred define qual usar)

Uso:
    python parte2_calibracao.py \
        --preds_in ./saida_predicoes/preds.jsonl \
        --preds_out ./saida_predicoes/preds_calibradas.jsonl \
        --calib_csv /mnt/data/comparacao_calibracao.csv \
        --method both --preferred ts

Observações:
- Para Isotonic Regression usa-se scikit-learn (IsotonicRegression).
- Para Temperature Scaling é feita uma busca em grade (logspace) do
  parâmetro T>0 minimizando NLL em validação.
- O campo de entrada de score nas entidades é 'score' por padrão
  (ajustável em --score_in).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Dependências opcionais
try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_SK = True
except Exception:
    IsotonicRegression = None  # type: ignore
    _HAS_SK = False

# Utilidades do projeto (Parte 1) — para cálculo de score_relato
try:
    from predicao import add_score_relato_por_entidade, compute_score_relato_mean
except Exception:
    # Implementa versões locais mínimas caso predicao.py não esteja no PYTHONPATH
    def compute_score_relato_mean(rec: Dict[str, Any], score_key: str = "score_ts") -> float:
        ents = rec.get("entities") or rec.get("ner") or []
        vals: List[float] = []
        for e in ents:
            v = e.get(score_key)
            if v is None:
                v = e.get("score")
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        return float(sum(vals) / len(vals)) if vals else 0.0

    def add_score_relato_por_entidade(rec: Dict[str, Any], score_key: str = "score_ts") -> Dict[str, Any]:
        sr = compute_score_relato_mean(rec, score_key=score_key)
        ents = rec.get("entities") or rec.get("ner") or []
        new_ents = []
        for e in ents:
            ee = dict(e)
            ee["score_relato"] = float(sr)
            new_ents.append(ee)
        if rec.get("entities") is not None:
            rec["entities"] = new_ents
        else:
            rec["ner"] = new_ents
        rec["score_relato"] = float(sr)
        return rec

EPS = 1e-12

# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Calibração — Temperature Scaling (binário)

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(p: np.ndarray, y: np.ndarray,
                   grid: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, float]]:
    """Ajusta T>0 via busca em grade (minimizando NLL). Retorna (T, métricas)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    logits = _logit(p)
    if grid is None:
        # Busca log-uniforme ampla
        grid = np.geomspace(0.05, 20.0, num=80)
    best_T = 1.0
    best_loss = float("inf")
    for T in grid:
        p_cal = _sigmoid(logits / T)
        loss = _nll(y, p_cal)
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    # Métricas antes/depois (opcional)
    metrics = {
        "nll_before": _nll(y, p),
        "nll_after": _nll(y, _sigmoid(logits / best_T)),
        "T": best_T,
    }
    return best_T, metrics


def apply_ts_to_record(rec: Dict[str, Any], T: float,
                       score_in: str = "score", score_out: str = "score_ts") -> Dict[str, Any]:
    ents = rec.get("entities") or rec.get("ner") or []
    if not ents:
        return rec
    raw = np.array([float(e.get(score_in, 0.0)) for e in ents], dtype=float)
    p_cal = _sigmoid(_logit(raw) / float(T))
    new_ents = []
    for e, v in zip(ents, p_cal):
        ee = dict(e)
        ee[score_out] = float(v)
        new_ents.append(ee)
    if rec.get("entities") is not None:
        rec["entities"] = new_ents
    else:
        rec["ner"] = new_ents
    return rec


# ----------------------------------------------------------------------------
# Calibração — Isotonic Regression

def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Any:
    if not _HAS_SK:
        raise RuntimeError("scikit-learn não disponível: instale 'scikit-learn' para IsotonicRegression.")
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return ir


def apply_iso_to_record(rec: Dict[str, Any], iso_model: Any,
                        score_in: str = "score", score_out: str = "score_iso") -> Dict[str, Any]:
    ents = rec.get("entities") or rec.get("ner") or []
    if not ents:
        return rec
    raw = np.array([float(e.get(score_in, 0.0)) for e in ents], dtype=float)
    p_cal = iso_model.predict(raw)
    p_cal = np.clip(p_cal, 0.0, 1.0)
    new_ents = []
    for e, v in zip(ents, p_cal):
        ee = dict(e)
        ee[score_out] = float(v)
        new_ents.append(ee)
    if rec.get("entities") is not None:
        rec["entities"] = new_ents
    else:
        rec["ner"] = new_ents
    return rec


# ----------------------------------------------------------------------------
# Fluxo principal

def run_calibration(preds_in: str | Path, preds_out: str | Path,
                    calib_csv: str | Path, method: str = "both",
                    preferred: Optional[str] = None,
                    score_in: str = "score") -> None:
    # 1) Carrega dados de calibração e valida
    df = pd.read_csv(calib_csv)
    if "Score" not in df.columns or "Validacao" not in df.columns:
        raise ValueError("CSV de calibração deve conter colunas 'Score' e 'Validacao'.")
    p = df["Score"].astype(float).to_numpy()
    y = df["Validacao"].astype(float).to_numpy()
    # 
    # 2) Ajusta calibradores
    T: Optional[float] = None
    iso = None
    if method in {"ts", "both"}:
        T, ts_metrics = fit_temperature(p, y)
        print(f"[TS] T={T:.4f} | NLL before={ts_metrics['nll_before']:.5f} after={ts_metrics['nll_after']:.5f}")
    if method in {"iso", "both"}:
        iso = fit_isotonic(p, y)
        # NLL simples para referência
        from math import isfinite
        p_iso = np.clip(iso.predict(p), 0.0, 1.0)
        print(f"[ISO] NLL before={_nll(y, p):.5f} after={_nll(y, p_iso):.5f}")

    # 3) Aplica aos registros
    items = read_jsonl(preds_in)
    out: List[Dict[str, Any]] = []
    for rec in items:
        rr = dict(rec)
        if method in {"ts", "both"} and T is not None:
            rr = apply_ts_to_record(rr, T, score_in=score_in, score_out="score_ts")
            # score_relato_ts
            sr_ts = compute_score_relato_mean(rr, score_key="score_ts")
            rr["score_relato_ts"] = float(sr_ts)
        if method in {"iso", "both"} and iso is not None:
            rr = apply_iso_to_record(rr, iso, score_in=score_in, score_out="score_iso")
            # score_relato_iso
            sr_iso = compute_score_relato_mean(rr, score_key="score_iso")
            rr["score_relato_iso"] = float(sr_iso)
        # score_relato principal (opcional)
        if preferred in {"ts", "iso"}:
            key = "score_relato_ts" if preferred == "ts" else "score_relato_iso"
            rr["score_relato"] = float(rr.get(key, 0.0))
            # replica campo por entidade, caso queira consistência
            rr = add_score_relato_por_entidade(rr, score_key=("score_ts" if preferred == "ts" else "score_iso"))
        out.append(rr)

    # 4) Salva
    write_jsonl(out, preds_out)
    print(f"[ok] Calibração '{method}' aplicada e salva em: {preds_out}  (n={len(out)})")


# ----------------------------------------------------------------------------
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parte 2 — Calibração (TS/ISO) e score_relato")
    p.add_argument("--preds_in", required=True, help="JSONL de entrada (predições da Parte 1)")
    p.add_argument("--preds_out", required=True, help="JSONL de saída (com scores calibrados)")
    p.add_argument("--calib_csv", required=True, help="CSV com colunas Score e Validacao")
    p.add_argument("--method", choices=["ts", "iso", "both"], default="both", help="Método(s) de calibração a aplicar")
    p.add_argument("--preferred", choices=["ts", "iso"], default=None, help="Qual método define o campo 'score_relato'")
    p.add_argument("--score_in", default="score", help="Nome do campo de score bruto nas entidades")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run_calibration(
        preds_in=args.preds_in,
        preds_out=args.preds_out,
        calib_csv=args.calib_csv,
        method=args.method,
        preferred=args.preferred,
        score_in=args.score_in,
    )


if __name__ == "__main__":
    main()
