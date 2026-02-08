# calibracao_iso.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Union, Iterable
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

_EPS = 1e-12

def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

def _ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = (np.asarray(y_true) > 0).astype(float)
    p = _clip01(np.asarray(p, dtype=float))
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    N, ece = len(p), 0.0
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        ece += (nb / N) * abs(y[m].mean() - p[m].mean())
    return float(ece)

def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    y = (np.asarray(y_true) > 0).astype(float)
    p = _clip01(np.asarray(p, dtype=float))
    return float(np.mean((p - y) ** 2))

def _save_reliability_plot(y_true, p, n_bins, out_path, title="Reliability"):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    y = (np.asarray(y_true) > 0).astype(float)
    p = _clip01(np.asarray(p, dtype=float))
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    rows = []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            rows.append({"bin": b, "count": 0, "conf": np.nan, "acc": np.nan})
        else:
            rows.append({"bin": b, "count": nb, "conf": float(p[m].mean()), "acc": float(y[m].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8")
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=130)
    ax.plot([0, 1], [0, 1], "--", lw=1.2, label="Ideal")
    v = df.dropna(subset=["conf", "acc"])
    ax.scatter(v["conf"], v["acc"], s=30, alpha=0.9, label="Bins")
    for _, r in v.iterrows():
        ax.plot([r["conf"], r["conf"]], [r["conf"], r["acc"]], lw=1, alpha=0.7)
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="conf (previsto)", ylabel="acc (observado)", title=title)
    ax.grid(alpha=0.25); ax.legend(loc="lower right", frameon=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

# --------- Núcleo ISO ---------
from sklearn.isotonic import IsotonicRegression

def fit_isotonic_from_csv(
    csv_path: Union[str, Path],
    y_col: str = "Validacao",
    p_col: str = "Score",
    n_bins: int = 15,
    save_plots_to: Union[str, Path, None] = None,
    plots_prefix: str | None = None,
) -> IsotonicRegression:
    df = pd.read_csv(csv_path)
    if p_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"CSV deve conter '{p_col}' e '{y_col}'")
    p = _clip01(df[p_col].to_numpy(dtype=float))
    y = (df[y_col].to_numpy(dtype=float) > 0).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    if save_plots_to is not None:
        out = Path(save_plots_to); out.mkdir(parents=True, exist_ok=True)
        prefix = plots_prefix or "iso"
        _save_reliability_plot(y, p, n_bins, out / f"{prefix}_reliability_before.png", title="Reliability (raw)")
        p_iso = iso.predict(p)
        _save_reliability_plot(y, p_iso, n_bins, out / f"{prefix}_reliability_after.png", title="Reliability (ISO)")
        pd.DataFrame({
            "metric": ["ece","ece","brier","brier"],
            "phase":  ["before","after","before","after"],
            "value":  [_ece(y,p,n_bins), _ece(y,p_iso,n_bins), _brier(y,p), _brier(y,p_iso)],
        }).to_csv(out / f"{prefix}_metrics.csv", index=False)
    return iso

def add_score_iso_to_record(
    record: Dict[str, Any],
    iso_model: Optional[IsotonicRegression] = None,
    score_in: str = "score",
    score_out: str = "score_ts",
) -> Dict[str, Any]:
    ents = record.get("entities") or record.get("ner") or []
    if not ents:
        return record
    raw = _clip01(np.array([float(e.get(score_in, 0.0)) for e in ents], dtype=float))
    p_cal = raw if iso_model is None else _clip01(iso_model.predict(raw))
    for e, v in zip(ents, p_cal):
        e[score_out] = float(v)
    return record
