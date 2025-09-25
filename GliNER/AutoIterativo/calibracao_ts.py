# calibracao_ts.py
# ------------------------------------------------------------
# Temperature Scaling (TS) para probabilidades [0,1] vindas de um
# classificador binário por entidade. O ajuste do T utiliza
# minimização 1D (Golden-Section Search) do objetivo escolhido
# (NLL por padrão). Inclui ECE/BRIER e plots de confiabilidade.
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import math
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ------------------------- Utilidades numéricas -------------------------

_EPS = 1e-12
_PHI = (1 + 5 ** 0.5) / 2.0  # razão áurea

def _clip01(p: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Clipa para [eps, 1-eps] e trata NaNs/Infs."""
    return np.clip(np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0), eps, 1.0 - eps)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip01(p)
    return np.log(p) - np.log1p(-p)  # log(p/(1-p))

def _ensure_1d(a: Union[np.ndarray, Iterable[float]]) -> np.ndarray:
    a = np.asarray(list(a) if not isinstance(a, np.ndarray) else a, dtype=float)
    return a.reshape(-1)


# ---------------------------- Métricas ----------------------------------

def _nll(y_true: np.ndarray, p: np.ndarray) -> float:
    y = _ensure_1d(y_true).astype(float)
    p = _clip01(_ensure_1d(p))
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    y = _ensure_1d(y_true).astype(float)
    p = _ensure_1d(p)
    return float(np.mean((p - y) ** 2))

def _ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (binning com intervalos iguais).
    """
    y = _ensure_1d(y_true).astype(float)
    p = _clip01(_ensure_1d(p))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    N = len(p)
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (nb / N) * abs(acc - conf)
    return float(ece)


# --------------------- Reliability plot e salvamento --------------------

def _save_reliability_plot(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int,
    out_path: Union[str, Path],
    title: str = "Reliability"
) -> None:
    """
    Salva o diagrama de confiabilidade (previsão média vs acurácia por bin).
    Se matplotlib não estiver disponível, cria apenas um CSV auxiliar.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y = _ensure_1d(y_true).astype(float)
    p = _clip01(_ensure_1d(p))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1

    rows = []
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            rows.append({"bin": b, "count": 0, "conf": np.nan, "acc": np.nan})
        else:
            rows.append({
                "bin": b,
                "count": nb,
                "conf": float(p[mask].mean()),
                "acc": float(y[mask].mean()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8")

    if not _HAS_MPL:
        # Sem matplotlib, apenas sai com o CSV.
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=130)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Ideal")

    # pontos por bin
    valid = df.dropna(subset=["conf", "acc"])
    ax.scatter(valid["conf"], valid["acc"], s=30, alpha=0.9, label="Bins")

    # barras verticais (opcional)
    for _, r in valid.iterrows():
        ax.plot([r["conf"], r["conf"]], [r["conf"], r["acc"]], linewidth=1, alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confiabilidade prevista (conf)")
    ax.set_ylabel("Acurácia observada (acc)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------- Núcleo do TS -------------------------------

class TemperatureScaler:
    """
    Ajusta e aplica Temperature Scaling:
        p' = sigmoid( logit(p) / T )

    objective: "nll" (padrão) | "ece" | "brier"
    bounds: intervalo de busca de T (ex.: (0.25, 8.0))
    """

    def __init__(self, objective: str = "nll", bounds: Tuple[float, float] = (0.25, 50.0), n_bins: int = 15):
        self.objective = objective.lower().strip()
        self.bounds = (float(bounds[0]), float(bounds[1]))
        if not (self.bounds[0] > 0 and self.bounds[1] > self.bounds[0]):
            raise ValueError("bounds inválido: use (a, b) com 0 < a < b.")
        self.n_bins = int(n_bins)
        self.T_: Optional[float] = None

    # --- funções objetivo ---
    def _objective_value(self, y: np.ndarray, p_raw: np.ndarray, T: float) -> float:
        z = _logit(p_raw)
        pT = _sigmoid(z / max(T, _EPS))
        if self.objective == "nll":
            return _nll(y, pT)
        elif self.objective == "ece":
            return _ece(y, pT, n_bins=self.n_bins)
        elif self.objective == "brier":
            return _brier(y, pT)
        else:
            raise ValueError(f"objective inválido: {self.objective}")

    # --- busca 1D por Golden-Section ---
    def _golden_section_search(self, y: np.ndarray, p_raw: np.ndarray, a: float, b: float, iters: int = 80) -> float:
        c = b - (b - a) / _PHI
        d = a + (b - a) / _PHI
        fc = self._objective_value(y, p_raw, c)
        fd = self._objective_value(y, p_raw, d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - (b - a) / _PHI
                fc = self._objective_value(y, p_raw, c)
            else:
                a, c, fc = c, d, fd
                d = a + (b - a) / _PHI
                fd = self._objective_value(y, p_raw, d)
            if abs(b - a) <= 1e-6:
                break
        return float((a + b) / 2.0)

    # --- API pública ---
    def fit(self, p_raw: Union[np.ndarray, Iterable[float]], y_true: Union[np.ndarray, Iterable[float]]) -> float:
        p = _clip01(_ensure_1d(p_raw))
        y = (_ensure_1d(y_true) > 0).astype(float)
        self.T_ = self._golden_section_search(y, p, self.bounds[0], self.bounds[1], iters=80)
        return float(self.T_)

    def transform(self, p_raw: Union[np.ndarray, Iterable[float]], T: Optional[float] = None) -> np.ndarray:
        if T is None:
            if self.T_ is None:
                raise RuntimeError("TemperatureScaler ainda não foi ajustado. Chame fit() ou passe T.")
            T = self.T_
        p = _clip01(_ensure_1d(p_raw))
        z = _logit(p)
        return _sigmoid(z / max(float(T), _EPS))

    def fit_transform(self, p_raw: Union[np.ndarray, Iterable[float]], y_true: Union[np.ndarray, Iterable[float]]) -> Tuple[float, np.ndarray]:
        T = self.fit(p_raw, y_true)
        return T, self.transform(p_raw, T)


# ----------------------- Funções compatíveis (API) ----------------------

def apply_temperature_scaling(conf: np.ndarray, T: float) -> np.ndarray:
    """
    Aplica TS nas probabilidades `conf` (0..1):
        p' = sigmoid( logit(p) / T )
    """
    conf = _ensure_1d(conf)
    z = _logit(conf)
    return _sigmoid(z / max(float(T), _EPS))


def add_score_ts_to_record(
    record: Dict[str, Any],
    T: Union[float, Dict[str, float]],
    score_in: str = "score",
    score_out: str = "score_ts",
    label_key: str = "label",
) -> Dict[str, Any]:
    """
    Aplica TS em-place nas entidades de um registro.
    - Se T for float → usa o mesmo T para todas as entidades.
    - Se T for dict[label->float] → usa T específico por rótulo (fallback para T global se houver chave "_default").
    """
    ents = record.get("entities", [])
    if not ents:
        return record

    per_label = isinstance(T, dict)
    T_default = None
    if per_label:
        T_default = T.get("_default", None)

    for e in ents:
        p = float(e.get(score_in, 0.0))
        if per_label:
            lab = str(e.get(label_key, ""))
            t_use = T.get(lab, T_default)
            if t_use is None:
                # se não tem T para o rótulo, mantemos score (identidade)
                e[score_out] = float(np.clip(p, 0.0, 1.0))
                continue
        else:
            t_use = T
        e[score_out] = float(apply_temperature_scaling(np.array([p], dtype=float), float(t_use))[0])
    return record


# ---------------------- Fit a partir de CSV (conveniência) ----------------------

def _fit_ts_global(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    objective: str,
    bounds: Tuple[float, float],
    n_bins: int,
    save_plots_to: Optional[Union[str, Path]] = None,
    plots_prefix: Optional[str] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Ajusta um único T global. Retorna (T, metrics_before, metrics_after).
    """
    p_raw = _clip01(df[p_col].to_numpy(dtype=float))
    y = (df[y_col].to_numpy(dtype=float) > 0).astype(float)

    scaler = TemperatureScaler(objective=objective, bounds=bounds, n_bins=n_bins)
    T = scaler.fit(p_raw, y)
    p_after = scaler.transform(p_raw, T)

    metrics_before = {
        "ece": _ece(y, p_raw, n_bins),
        "brier": _brier(y, p_raw),
        "nll": _nll(y, p_raw),
    }
    metrics_after = {
        "ece": _ece(y, p_after, n_bins),
        "brier": _brier(y, p_after),
        "nll": _nll(y, p_after),
    }

    if save_plots_to is not None:
        out = Path(save_plots_to)
        out.mkdir(parents=True, exist_ok=True)
        prefix = plots_prefix or "ts"
        _save_reliability_plot(y, p_raw, n_bins, out / f"{prefix}_reliability_before.png", title="Reliability (raw)")
        _save_reliability_plot(y, p_after, n_bins, out / f"{prefix}_reliability_after.png", title=f"Reliability (TS, T={T:.3f})")
        pd.DataFrame(
            {"metric": ["ece","brier","nll"],
             "before": [metrics_before["ece"], metrics_before["brier"], metrics_before["nll"]],
             "after":  [metrics_after["ece"],  metrics_after["brier"],  metrics_after["nll"]]}
        ).to_csv(out / f"{prefix}_metrics.csv", index=False, encoding="utf-8")

    return float(T), metrics_before, metrics_after


def _fit_ts_per_label(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    label_col: str,
    objective: str,
    bounds: Tuple[float, float],
    n_bins: int,
    min_count: int = 50,
    save_plots_to: Optional[Union[str, Path]] = None,
    plots_prefix: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Ajusta T por rótulo (label). Rotulos com menos que `min_count` exemplos positivos+negativos
    manterão identidade (T=1.0). Retorna (T_por_label, metrics_before_por_label, metrics_after_por_label).
    """
    Ts: Dict[str, float] = {}
    m_before: Dict[str, Dict[str, float]] = {}
    m_after: Dict[str, Dict[str, float]] = {}

    labels = sorted([str(x) for x in df[label_col].dropna().unique().tolist()])
    for lab in labels:
        sub = df[df[label_col] == lab]
        if len(sub) < min_count:
            Ts[lab] = 1.0
            continue

        T, mb, ma = _fit_ts_global(
            sub, y_col=y_col, p_col=p_col, objective=objective,
            bounds=bounds, n_bins=n_bins,
            save_plots_to=(None if save_plots_to is None else Path(save_plots_to) / f"label_{lab}"),
            plots_prefix=(None if plots_prefix is None else f"{plots_prefix}_{lab}")
        )
        Ts[lab] = float(T)
        m_before[lab] = mb
        m_after[lab]  = ma

    return Ts, m_before, m_after


def fit_temperature_from_csv(
    csv_path: Union[str, Path],
    y_col: str = "Validacao",
    p_col: str = "Score",
    label_col: Optional[str] = None,
    objective: str = "nll",                 # "nll" | "ece" | "brier"
    bounds: Tuple[float, float] = (0.25, 50.0),
    n_bins: int = 15,
    return_table: bool = False,
    save_plots_to: Optional[Union[str, Path]] = None,
    plots_prefix: Optional[str] = None,
) -> Union[float, Dict[str, float], Tuple[Union[float, Dict[str, float]], pd.DataFrame]]:
    """
    Ajusta Temperature Scaling a partir de um CSV com colunas:
      - p_col: probabilidade prevista (0..1)
      - y_col: rótulo binário (1/0)
    Opcionalmente:
      - label_col: rótulo da entidade para ter um T por rótulo.

    Retorna:
      - Sem label_col: float T
      - Com label_col: dict[label -> T]
    Se return_table=True, retorna também um DataFrame de métricas agregadas.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    for col in [p_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"CSV precisa ter a coluna '{col}'.")

    if label_col is None:
        T, mb, ma = _fit_ts_global(
            df, y_col=y_col, p_col=p_col, objective=objective,
            bounds=bounds, n_bins=n_bins,
            save_plots_to=save_plots_to, plots_prefix=plots_prefix
        )
        if return_table:
            table = pd.DataFrame({
                "metric": ["ece","brier","nll"],
                "before": [mb["ece"], mb["brier"], mb["nll"]],
                "after":  [ma["ece"], ma["brier"], ma["nll"]],
            })
            return T, table
        return T
    else:
        if label_col not in df.columns:
            raise ValueError(f"CSV precisa ter a coluna de rótulo '{label_col}'.")
        Ts, mb, ma = _fit_ts_per_label(
            df, y_col=y_col, p_col=p_col, label_col=label_col,
            objective=objective, bounds=bounds, n_bins=n_bins,
            save_plots_to=save_plots_to, plots_prefix=plots_prefix
        )
        if return_table:
            rows = []
            for lab, T in Ts.items():
                mb_l = mb.get(lab, {"ece": np.nan, "brier": np.nan, "nll": np.nan})
                ma_l = ma.get(lab, {"ece": np.nan, "brier": np.nan, "nll": np.nan})
                rows.append({
                    "label": lab, "T": T,
                    "ece_before": mb_l["ece"], "ece_after": ma_l["ece"],
                    "brier_before": mb_l["brier"], "brier_after": ma_l["brier"],
                    "nll_before": mb_l["nll"], "nll_after": ma_l["nll"],
                })
            table = pd.DataFrame(rows).sort_values("label")
            return Ts, table
        return Ts
