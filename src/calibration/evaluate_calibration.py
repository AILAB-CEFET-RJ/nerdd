#!/usr/bin/env python3
"""
Calibration evaluation for confidence scores.

Compares RAW scores, Temperature Scaling, and Isotonic Regression using a
calibration CSV with score and binary target columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from methods.isotonic import apply_isotonic, fit_isotonic
from methods.temperature import apply_temperature, fit_temperature

EPS = 1e-12


def _nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_bins(scores: np.ndarray, targets: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(scores, edges[1:-1], right=True)
    rows = []
    for idx in range(bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        lower, upper = float(edges[idx]), float(edges[idx + 1])
        if count == 0:
            rows.append(
                {
                    "bin": idx,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "conf_mean": np.nan,
                    "acc": np.nan,
                    "gap": np.nan,
                }
            )
            continue
        conf_mean = float(np.mean(scores[mask]))
        acc = float(np.mean(targets[mask]))
        rows.append(
            {
                "bin": idx,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "conf_mean": conf_mean,
                "acc": acc,
                "gap": abs(acc - conf_mean),
            }
        )
    return pd.DataFrame(rows)


def ece_mce_from_bins(bin_df: pd.DataFrame) -> tuple[float, float]:
    valid = bin_df.dropna(subset=["count", "gap"]).copy()
    total = valid["count"].sum()
    if total <= 0:
        return float("nan"), float("nan")
    weights = valid["count"] / total
    ece = float((weights * valid["gap"]).sum())
    mce = float(valid["gap"].max())
    return ece, mce


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def evaluate_fold(
    train_scores: np.ndarray,
    train_targets: np.ndarray,
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    bins: int,
    disable_isotonic: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    raw_bins = reliability_bins(val_scores, val_targets, bins=bins)
    raw_ece, raw_mce = ece_mce_from_bins(raw_bins)
    result["raw"] = {
        "scores": val_scores,
        "bins": raw_bins,
        "ece": raw_ece,
        "mce": raw_mce,
        "brier": _brier(val_targets, val_scores),
        "nll": _nll(val_targets, val_scores),
        "auc": _safe_auc(val_targets, val_scores),
    }

    temperature, fit_loss = fit_temperature(train_scores, train_targets)
    ts_scores = apply_temperature(val_scores, temperature)
    ts_bins = reliability_bins(ts_scores, val_targets, bins=bins)
    ts_ece, ts_mce = ece_mce_from_bins(ts_bins)
    result["temperature"] = {
        "temperature": temperature,
        "fit_loss": fit_loss,
        "scores": ts_scores,
        "bins": ts_bins,
        "ece": ts_ece,
        "mce": ts_mce,
        "brier": _brier(val_targets, ts_scores),
        "nll": _nll(val_targets, ts_scores),
        "auc": _safe_auc(val_targets, ts_scores),
    }

    if disable_isotonic:
        result["isotonic"] = None
    else:
        isotonic_model = fit_isotonic(train_scores, train_targets)
        iso_scores = np.clip(apply_isotonic(isotonic_model, val_scores), 0.0, 1.0)
        iso_bins = reliability_bins(iso_scores, val_targets, bins=bins)
        iso_ece, iso_mce = ece_mce_from_bins(iso_bins)
        result["isotonic"] = {
            "scores": iso_scores,
            "bins": iso_bins,
            "ece": iso_ece,
            "mce": iso_mce,
            "brier": _brier(val_targets, iso_scores),
            "nll": _nll(val_targets, iso_scores),
            "auc": _safe_auc(val_targets, iso_scores),
        }

    return result


def _save_reliability_csvs(fold_results: list[dict[str, Any]], output_dir: Path) -> dict[str, pd.DataFrame]:
    method_map = {"raw": "raw", "temperature": "ts", "isotonic": "iso"}
    saved: dict[str, pd.DataFrame] = {}
    for method_name, suffix in method_map.items():
        stacks = []
        for fold_result in fold_results:
            data = fold_result.get(method_name)
            if not data:
                continue
            stacks.append(data["bins"][["bin", "bin_lower", "bin_upper", "count", "conf_mean", "acc"]].copy())
        if not stacks:
            continue
        merged = pd.concat(stacks, ignore_index=True)
        grouped = merged.groupby("bin", as_index=False).agg(
            {"bin_lower": "first", "bin_upper": "first", "count": "sum", "conf_mean": "mean", "acc": "mean"}
        )
        grouped.to_csv(output_dir / f"reliability_{suffix}.csv", index=False)
        saved[method_name] = grouped
    return saved


def _save_summary_csv(fold_results: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    methods = ["raw", "temperature", "isotonic"]
    metrics = ["ece", "mce", "brier", "nll", "auc"]
    for method_name in methods:
        for metric_name in metrics:
            values = []
            for fold_result in fold_results:
                method_data = fold_result.get(method_name)
                if not method_data:
                    continue
                value = method_data.get(metric_name)
                if value is None:
                    continue
                if isinstance(value, float) and np.isnan(value):
                    continue
                values.append(float(value))
            rows.append(
                {
                    "method": method_name,
                    "metric": metric_name,
                    "mean": float(np.mean(values)) if values else np.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else np.nan),
                    "n_folds": len(values),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)


def _save_reliability_plot(output_dir: Path, reliability_data: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(6, 6), dpi=140)
    diagonal = np.linspace(0, 1, 201)
    plt.plot(diagonal, diagonal, linestyle="--", linewidth=1.0, label="Perfect calibration")

    label_map = {"raw": "RAW", "temperature": "TS", "isotonic": "ISO"}
    for method_name in ["raw", "temperature", "isotonic"]:
        if method_name not in reliability_data:
            continue
        df = reliability_data[method_name]
        plt.plot(df["conf_mean"], df["acc"], marker="o", label=label_map[method_name])

    plt.xlabel("Mean confidence per bin")
    plt.ylabel("Empirical accuracy per bin")
    plt.title("Reliability Curves")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "reliability_curves.png")
    plt.close()


def run_calibration_evaluation(
    calibration_csv: str,
    output_dir: str,
    score_col: str = "Score",
    label_col: str = "Validacao",
    bins: int = 10,
    cv: int = 5,
    seed: int = 42,
    disable_isotonic: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    calibration_df = pd.read_csv(calibration_csv)
    if score_col not in calibration_df.columns:
        raise ValueError(f"CSV is missing score column '{score_col}'.")
    if label_col not in calibration_df.columns:
        raise ValueError(f"CSV is missing label column '{label_col}'.")

    scores = calibration_df[score_col].astype(float).to_numpy()
    targets = calibration_df[label_col].astype(float).to_numpy()

    fold_results = []
    if cv <= 1:
        fold_results.append(
            evaluate_fold(
                train_scores=scores,
                train_targets=targets,
                val_scores=scores,
                val_targets=targets,
                bins=bins,
                disable_isotonic=disable_isotonic,
            )
        )
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(scores):
            fold_results.append(
                evaluate_fold(
                    train_scores=scores[train_idx],
                    train_targets=targets[train_idx],
                    val_scores=scores[val_idx],
                    val_targets=targets[val_idx],
                    bins=bins,
                    disable_isotonic=disable_isotonic,
                )
            )

    _save_summary_csv(fold_results, output_path)
    reliability_data = _save_reliability_csvs(fold_results, output_path)
    _save_reliability_plot(output_path, reliability_data)

    print(f"Calibration evaluation artifacts written to: {output_path.resolve()}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate confidence calibration (RAW vs TS vs ISO).")
    parser.add_argument("--calibration-csv", required=True, help="CSV with score and binary target columns.")
    parser.add_argument("--out-dir", required=True, help="Directory for evaluation artifacts.")
    parser.add_argument("--score-col", default="Score", help="Score column name in calibration CSV.")
    parser.add_argument("--label-col", default="Validacao", help="Binary target column name in calibration CSV.")
    parser.add_argument("--bins", type=int, default=10, help="Number of reliability bins.")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds; use 1 for in-sample evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used in CV splitting.")
    parser.add_argument("--no-isotonic", action="store_true", help="Disable isotonic evaluation.")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    run_calibration_evaluation(
        calibration_csv=args.calibration_csv,
        output_dir=args.out_dir,
        score_col=args.score_col,
        label_col=args.label_col,
        bins=args.bins,
        cv=args.cv,
        seed=args.seed,
        disable_isotonic=args.no_isotonic,
    )


if __name__ == "__main__":
    main()
