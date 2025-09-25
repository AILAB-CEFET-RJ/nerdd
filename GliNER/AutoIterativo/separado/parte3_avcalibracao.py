#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 3 — Avaliação de Calibração
---------------------------------
Compara métodos de calibração (RAW, Temperature Scaling, Isotonic Regression)
utilizando um CSV com colunas 'Score' (probabilidade) e 'Validacao' (0/1).

Saídas principais (em --out_dir):
  - metrics_summary.csv           : métricas agregadas por método (média/DP nos folds)
  - reliability_raw.csv           : curva de confiabilidade do RAW (por bins)
  - reliability_ts.csv            : curva de confiabilidade do TS (por bins)
  - reliability_iso.csv           : curva de confiabilidade do ISO (por bins)
  - reliability_curves.png        : gráfico com as curvas de confiabilidade

Modo de avaliação:
  - Por padrão usa validação cruzada (k-fold) com k=5 para estimar generalização.
  - Alternativamente, use --cv 1 para avaliação simples (in-sample, sem validação).

Uso (exemplos):
  python parte3_avaliacao.py \
    --calib_csv /mnt/data/comparacao_calibracao.csv \
    --out_dir ./avaliacao_calibracao \
    --bins 10 --cv 5

Observações:
  - Requer numpy, pandas e matplotlib. Para ISO, requer scikit-learn.
  - Para TS é realizada busca em T>0 (logspace) minimizando NLL no fold de treino.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn é opcional: para Isotonic, KFold e AUC
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    _HAS_SK = True
except Exception:
    IsotonicRegression = None  # type: ignore
    roc_auc_score = None  # type: ignore
    KFold = None  # type: ignore
    _HAS_SK = False

EPS = 1e-12

# ---------------------------------------------------------------------------
# Helpers matemáticos

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Temperature Scaling

def fit_temperature(p: np.ndarray, y: np.ndarray, grid: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, float]]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    logits = _logit(p)
    if grid is None:
        grid = np.geomspace(0.05, 20.0, num=80)
    best_T = 1.0
    best_loss = float("inf")
    for T in grid:
        p_cal = _sigmoid(logits / T)
        loss = _nll(y, p_cal)
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    metrics = {
        "nll_before": _nll(y, p),
        "nll_after": _nll(y, _sigmoid(logits / best_T)),
        "T": best_T,
    }
    return best_T, metrics


def apply_ts(p: np.ndarray, T: float) -> np.ndarray:
    return _sigmoid(_logit(np.asarray(p, dtype=float)) / float(T))


# ---------------------------------------------------------------------------
# Isotonic Regression

def fit_isotonic(p: np.ndarray, y: np.ndarray):
    if IsotonicRegression is None:
        raise RuntimeError("scikit-learn não disponível para IsotonicRegression.")
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return ir


def apply_iso(p: np.ndarray, ir) -> np.ndarray:
    return np.clip(ir.predict(np.asarray(p, dtype=float)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Métricas de calibração

def reliability_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    # Bins fechados à direita [0,1]
    edges = np.linspace(0.0, 1.0, bins + 1)
    inds = np.digitize(p, edges[1:-1], right=True)  # 0..bins-1
    rows = []
    for b in range(bins):
        mask = inds == b
        cnt = int(mask.sum())
        lo, hi = float(edges[b]), float(edges[b + 1])
        if cnt == 0:
            rows.append({
                "bin": b,
                "bin_lower": lo,
                "bin_upper": hi,
                "count": 0,
                "conf_mean": np.nan,
                "acc": np.nan,
                "gap": np.nan,
            })
            continue
        conf_mean = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        gap = abs(acc - conf_mean)
        rows.append({
            "bin": b,
            "bin_lower": lo,
            "bin_upper": hi,
            "count": cnt,
            "conf_mean": conf_mean,
            "acc": acc,
            "gap": float(gap),
        })
    return pd.DataFrame(rows)


def ece_from_bins(df_bins: pd.DataFrame) -> Tuple[float, float]:
    df = df_bins.dropna(subset=["count", "gap"]).copy()
    n = df["count"].sum()
    if n <= 0:
        return float("nan"), float("nan")
    weights = df["count"] / n
    ece = float((weights * df["gap"]).sum())
    mce = float(df["gap"].max())
    return ece, mce


# ---------------------------------------------------------------------------
# Avaliação (um fold)

def evaluate_fold(p_train: np.ndarray, y_train: np.ndarray,
                  p_val: np.ndarray, y_val: np.ndarray,
                  bins: int = 10,
                  enable_iso: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # RAW
    raw_bins = reliability_bins(p_val, y_val, bins=bins)
    raw_ece, raw_mce = ece_from_bins(raw_bins)
    results["raw"] = {
        "p": p_val,
        "bins": raw_bins,
        "ece": raw_ece,
        "mce": raw_mce,
        "brier": _brier(y_val, p_val),
        "nll": _nll(y_val, p_val),
    }
    if roc_auc_score is not None:
        try:
            results["raw"]["auc"] = float(roc_auc_score(y_val, p_val))
        except Exception:
            results["raw"]["auc"] = np.nan
    else:
        results["raw"]["auc"] = np.nan

    # TS
    T, _ = fit_temperature(p_train, y_train)
    p_ts = apply_ts(p_val, T)
    ts_bins = reliability_bins(p_ts, y_val, bins=bins)
    ts_ece, ts_mce = ece_from_bins(ts_bins)
    results["ts"] = {
        "T": T,
        "p": p_ts,
        "bins": ts_bins,
        "ece": ts_ece,
        "mce": ts_mce,
        "brier": _brier(y_val, p_ts),
        "nll": _nll(y_val, p_ts),
    }
    if roc_auc_score is not None:
        try:
            results["ts"]["auc"] = float(roc_auc_score(y_val, p_ts))
        except Exception:
            results["ts"]["auc"] = np.nan
    else:
        results["ts"]["auc"] = np.nan

    # ISO
    if enable_iso and IsotonicRegression is not None:
        ir = fit_isotonic(p_train, y_train)
        p_iso = apply_iso(p_val, ir)
        iso_bins = reliability_bins(p_iso, y_val, bins=bins)
        iso_ece, iso_mce = ece_from_bins(iso_bins)
        results["iso"] = {
            "p": p_iso,
            "bins": iso_bins,
            "ece": iso_ece,
            "mce": iso_mce,
            "brier": _brier(y_val, p_iso),
            "nll": _nll(y_val, p_iso),
        }
        if roc_auc_score is not None:
            try:
                results["iso"]["auc"] = float(roc_auc_score(y_val, p_iso))
            except Exception:
                results["iso"]["auc"] = np.nan
        else:
            results["iso"]["auc"] = np.nan
    else:
        results["iso"] = None

    return results


# ---------------------------------------------------------------------------
# Agregação e salvamento

def aggregate_metrics(fold_results: List[Dict[str, Any]], out_dir: Path, bins: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["raw", "ts", "iso"]
    rows = []
    # Média e DP por método
    for m in methods:
        vals: Dict[str, List[float]] = {"ece": [], "mce": [], "brier": [], "nll": [], "auc": []}
        for fr in fold_results:
            r = fr.get(m)
            if not r:
                continue
            for k in vals:
                v = r.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals[k].append(float(v))
        for k, lst in vals.items():
            if not lst:
                mean_v, std_v = np.nan, np.nan
            else:
                mean_v, std_v = float(np.mean(lst)), float(np.std(lst, ddof=1) if len(lst) > 1 else 0.0)
            rows.append({"method": m, "metric": k, "mean": mean_v, "std": std_v, "n_folds": len(lst)})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_summary.csv", index=False)

    # Curvas de confiabilidade (média por bin)
    def _save_bins(method: str, key: str):
        # Empilha bins de todos os folds e tira médias por índice de bin
        stacks: List[pd.DataFrame] = []
        for fr in fold_results:
            r = fr.get(method)
            if not r:
                continue
            dfb = r["bins"][ ["bin", "bin_lower", "bin_upper", "count", "conf_mean", "acc"] ].copy()
            stacks.append(dfb)
        if not stacks:
            return None
        big = pd.concat(stacks, ignore_index=True)
        grp = big.groupby("bin", as_index=False).agg({
            "bin_lower": "first",
            "bin_upper": "first",
            "count": "sum",
            "conf_mean": "mean",
            "acc": "mean",
        })
        path = out_dir / f"reliability_{key}.csv"
        grp.to_csv(path, index=False)
        return grp

    bins_raw = _save_bins("raw", "raw")
    bins_ts  = _save_bins("ts",  "ts")
    bins_iso = _save_bins("iso", "iso")

    # Gráfico
    plt.figure(figsize=(6, 6), dpi=140)
    xx = np.linspace(0, 1, 201)
    plt.plot(xx, xx, linestyle='--', linewidth=1.0, label='Perfeita')
    if bins_raw is not None:
        plt.plot(bins_raw["conf_mean"], bins_raw["acc"], marker='o', label='RAW')
    if bins_ts is not None:
        plt.plot(bins_ts["conf_mean"], bins_ts["acc"], marker='o', label='TS')
    if bins_iso is not None:
        plt.plot(bins_iso["conf_mean"], bins_iso["acc"], marker='o', label='ISO')
    plt.xlabel('Confiança média por bin')
    plt.ylabel('Acurácia empírica por bin')
    plt.title('Curvas de Confiabilidade')
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_curves.png")
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline principal

def run(calib_csv: str | Path, out_dir: str | Path, bins: int = 10, cv: int = 5, seed: int = 42,
        enable_iso: bool = True) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(calib_csv)
    if "Score" not in df.columns or "Validacao" not in df.columns:
        raise ValueError("CSV deve conter colunas 'Score' e 'Validacao'.")
    p_all = df["Score"].astype(float).to_numpy()
    y_all = df["Validacao"].astype(float).to_numpy()

    fold_results: List[Dict[str, Any]] = []

    if cv <= 1:
        # Avaliação in-sample (sem validação): treina e avalia nas mesmas amostras
        res = evaluate_fold(p_all, y_all, p_all, y_all, bins=bins, enable_iso=enable_iso)
        fold_results.append(res)
    else:
        if KFold is None:
            raise RuntimeError("Validação cruzada requer scikit-learn (KFold). Instale scikit-learn ou use --cv 1.")
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        for i, (tr_idx, va_idx) in enumerate(kf.split(p_all), start=1):
            p_tr, y_tr = p_all[tr_idx], y_all[tr_idx]
            p_va, y_va = p_all[va_idx], y_all[va_idx]
            res = evaluate_fold(p_tr, y_tr, p_va, y_va, bins=bins, enable_iso=enable_iso)
            fold_results.append(res)
            print(f"[fold {i}/{cv}] RAW ECE={res['raw']['ece']:.4f} | TS ECE={res['ts']['ece']:.4f} | ISO ECE={(res['iso']['ece'] if res['iso'] else np.nan):.4f}")

    aggregate_metrics(fold_results, outp, bins=bins)
    print(f"[ok] Relatórios salvos em: {outp}")


# ---------------------------------------------------------------------------
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parte 3 — Avaliação de calibração (ECE/BRIER/NLL/AUC e curvas)")
    p.add_argument("--calib_csv", required=True, help="CSV com colunas Score e Validacao")
    p.add_argument("--out_dir", required=True, help="Pasta de saída para relatórios")
    p.add_argument("--bins", type=int, default=10, help="Número de bins para ECE (padrão: 10)")
    p.add_argument("--cv", type=int, default=5, help="Número de folds de validação cruzada (padrão: 5). Use 1 para in-sample.")
    p.add_argument("--seed", type=int, default=42, help="Semente de aleatoriedade para CV")
    p.add_argument("--no_iso", action="store_true", help="Desabilita o Isotonic Regression mesmo se scikit-learn estiver disponível")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run(
        calib_csv=args.calib_csv,
        out_dir=args.out_dir,
        bins=args.bins,
        cv=args.cv,
        seed=args.seed,
        enable_iso=(not args.no_iso),
    )


if __name__ == "__main__":
    main()
