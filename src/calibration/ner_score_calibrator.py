"""Fit and apply NER score calibrators from exact-match prediction outcomes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


EPSILON = 1e-6


@dataclass(frozen=True)
class CalibrationExample:
    label: str
    score: float
    target: int


def clip_score(score: Any) -> float:
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite score: {score}")
    return float(min(max(value, EPSILON), 1.0 - EPSILON))


def span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["label"])


def build_examples_from_oof_rows(
    rows: list[dict[str, Any]],
    *,
    labels: set[str],
    score_field: str = "score",
    pred_field: str = "pred_spans",
    gold_field: str = "gold_spans",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    gold_support = Counter()
    for row_index, row in enumerate(rows):
        text = str(row.get("text", ""))
        gold_spans = [
            span
            for span in (row.get(gold_field) or [])
            if str(span.get("label")) in labels
        ]
        gold_set = {span_key(span) for span in gold_spans}
        for span in gold_spans:
            gold_support[str(span["label"])] += 1

        for pred in row.get(pred_field) or []:
            label = str(pred.get("label", ""))
            if label not in labels or score_field not in pred:
                continue
            try:
                score = clip_score(pred[score_field])
            except (TypeError, ValueError):
                continue
            start = int(pred["start"])
            end = int(pred["end"])
            examples.append(
                {
                    "row_index_0based": row.get("row_index_0based", row_index),
                    "row_index_1based": row.get("row_index_1based", row_index + 1),
                    "sample_id": row.get("sample_id"),
                    "fold": row.get("fold"),
                    "label": label,
                    "mention": pred.get("text", text[start:end]),
                    "start": start,
                    "end": end,
                    "score_raw": score,
                    "target": 1 if (start, end, label) in gold_set else 0,
                    "text_preview": text[:180],
                }
            )
    return examples, dict(gold_support)


def reliability_rows(scores: np.ndarray, targets: np.ndarray, *, bins: int, prefix: str = "") -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(scores, edges[1:-1], right=True)
    rows = []
    for index in range(bins):
        mask = bin_ids == index
        count = int(np.sum(mask))
        score_mean = float(np.mean(scores[mask])) if count else None
        empirical_precision = float(np.mean(targets[mask])) if count else None
        gap = (
            float(abs(score_mean - empirical_precision))
            if score_mean is not None and empirical_precision is not None
            else None
        )
        rows.append(
            {
                "bin_index": index,
                f"{prefix}score_low": float(edges[index]),
                f"{prefix}score_high": float(edges[index + 1]),
                "count": count,
                "correct": int(np.sum(targets[mask])) if count else 0,
                f"{prefix}score_mean": score_mean,
                "empirical_precision": empirical_precision,
                f"{prefix}calibration_gap": gap,
            }
        )
    return rows


def ece_mce(rows: list[dict[str, Any]], *, gap_key: str) -> tuple[float | None, float | None]:
    valid = [row for row in rows if int(row["count"]) > 0 and row.get(gap_key) is not None]
    total = sum(int(row["count"]) for row in valid)
    if not total:
        return None, None
    ece = sum((int(row["count"]) / total) * float(row[gap_key]) for row in valid)
    mce = max(float(row[gap_key]) for row in valid)
    return float(ece), float(mce)


def apply_model_to_array(scores: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    method = model["method"]
    if method == "identity":
        return np.asarray(scores, dtype=np.float64)
    if method == "isotonic":
        x = np.asarray(model["x_thresholds"], dtype=np.float64)
        y = np.asarray(model["y_thresholds"], dtype=np.float64)
        return np.interp(scores, x, y)
    if method == "platt":
        coef = float(model["coef"])
        intercept = float(model["intercept"])
        logits = coef * scores + intercept
        return 1.0 / (1.0 + np.exp(-logits))
    raise ValueError(f"Unsupported NER calibrator method: {method}")


def apply_calibrator_to_score(score: Any, label: str, calibrator: dict[str, Any]) -> float:
    models = calibrator.get("label_models") or {}
    model = models.get(label) or calibrator.get("fallback_model") or {"method": "identity"}
    raw = np.asarray([clip_score(score)], dtype=np.float64)
    calibrated = apply_model_to_array(raw, model)[0]
    return float(min(max(calibrated, 0.0), 1.0))


def _fit_identity(scores: np.ndarray, targets: np.ndarray, *, reason: str) -> tuple[np.ndarray, dict[str, Any]]:
    return scores.copy(), {
        "method": "identity",
        "source": reason,
        "support": int(len(scores)),
        "positives": int(np.sum(targets)),
        "negatives": int(len(targets) - np.sum(targets)),
    }


def fit_label_model(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    method: str,
    min_positive: int,
    min_negative: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    positives = int(np.sum(targets))
    negatives = int(len(targets) - positives)
    if positives < min_positive or negatives < min_negative:
        return _fit_identity(scores, targets, reason="insufficient-positive-or-negative-examples")

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrated = model.fit_transform(scores, targets)
        payload = {
            "method": "isotonic",
            "source": "per-label",
            "support": int(len(scores)),
            "positives": positives,
            "negatives": negatives,
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        }
        return calibrated, payload

    if method == "platt":
        model = LogisticRegression(solver="lbfgs")
        model.fit(scores.reshape(-1, 1), targets)
        calibrated = model.predict_proba(scores.reshape(-1, 1))[:, 1]
        payload = {
            "method": "platt",
            "source": "per-label",
            "support": int(len(scores)),
            "positives": positives,
            "negatives": negatives,
            "coef": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        }
        return calibrated, payload

    raise ValueError(f"Unsupported calibration method: {method}")


def fit_ner_score_calibrator(
    examples: list[dict[str, Any]],
    *,
    labels: list[str],
    method: str,
    min_positive: int = 20,
    min_negative: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_models: dict[str, dict[str, Any]] = {}
    calibrated_rows: list[dict[str, Any]] = []
    for label in labels:
        label_examples = [row for row in examples if row["label"] == label]
        if not label_examples:
            label_models[label] = {
                "method": "identity",
                "source": "no-examples",
                "support": 0,
                "positives": 0,
                "negatives": 0,
            }
            continue
        scores = np.asarray([row["score_raw"] for row in label_examples], dtype=np.float64)
        targets = np.asarray([row["target"] for row in label_examples], dtype=np.int64)
        calibrated, model_payload = fit_label_model(
            scores,
            targets,
            method=method,
            min_positive=min_positive,
            min_negative=min_negative,
        )
        label_models[label] = model_payload
        for row, calibrated_score in zip(label_examples, calibrated):
            enriched = dict(row)
            enriched["score_calibrated"] = float(min(max(calibrated_score, 0.0), 1.0))
            calibrated_rows.append(enriched)

    calibrator = {
        "kind": "ner_score_calibrator_oof",
        "version": 1,
        "method": method,
        "labels": labels,
        "min_positive": min_positive,
        "min_negative": min_negative,
        "label_models": label_models,
        "fallback_model": {"method": "identity", "source": "unknown-label"},
    }
    calibrated_rows.sort(key=lambda row: (str(row["label"]), int(row["row_index_1based"]), int(row["start"]), int(row["end"])))
    return calibrator, calibrated_rows


def metric_summary(
    rows: list[dict[str, Any]],
    *,
    labels: list[str],
    bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: dict[str, Any] = {}
    raw_reliability: list[dict[str, Any]] = []
    calibrated_reliability: list[dict[str, Any]] = []
    scopes = [("ALL", rows)] + [(label, [row for row in rows if row["label"] == label]) for label in labels]
    for label, scoped_rows in scopes:
        if not scoped_rows:
            summary[label] = {"count": 0}
            continue
        raw_scores = np.asarray([row["score_raw"] for row in scoped_rows], dtype=np.float64)
        calibrated_scores = np.asarray([row["score_calibrated"] for row in scoped_rows], dtype=np.float64)
        targets = np.asarray([row["target"] for row in scoped_rows], dtype=np.int64)
        raw_rows = reliability_rows(raw_scores, targets, bins=bins, prefix="raw_")
        calibrated_rows = reliability_rows(calibrated_scores, targets, bins=bins, prefix="calibrated_")
        for row in raw_rows:
            raw_reliability.append({"label": label, **row})
        for row in calibrated_rows:
            calibrated_reliability.append({"label": label, **row})
        raw_ece, raw_mce = ece_mce(raw_rows, gap_key="raw_calibration_gap")
        calibrated_ece, calibrated_mce = ece_mce(calibrated_rows, gap_key="calibrated_calibration_gap")
        summary[label] = {
            "count": int(len(scoped_rows)),
            "positives": int(np.sum(targets)),
            "negatives": int(len(targets) - np.sum(targets)),
            "empirical_precision": float(np.mean(targets)),
            "raw_score_mean": float(np.mean(raw_scores)),
            "calibrated_score_mean": float(np.mean(calibrated_scores)),
            "raw_brier": float(brier_score_loss(targets, raw_scores)),
            "calibrated_brier": float(brier_score_loss(targets, calibrated_scores)),
            "raw_ece": raw_ece,
            "calibrated_ece": calibrated_ece,
            "raw_mce": raw_mce,
            "calibrated_mce": calibrated_mce,
        }
    return summary, raw_reliability, calibrated_reliability
