import json
import logging
import csv
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import brier_score_loss

try:
    from .methods.isotonic import apply_isotonic, fit_isotonic
    from .methods.temperature import apply_temperature, fit_temperature
except ImportError:  # pragma: no cover
    from methods.isotonic import apply_isotonic, fit_isotonic
    from methods.temperature import apply_temperature, fit_temperature

LOGGER = logging.getLogger(__name__)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def save_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def resolve_path(base_dir, configured_path):
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _extract_entities(rows, score_field, label_field, allowed_labels):
    entries = []
    for row_idx, row in enumerate(rows):
        for ent_idx, entity in enumerate(row.get("entities", [])):
            if score_field not in entity:
                continue
            label = str(entity.get(label_field, ""))
            if allowed_labels and label not in allowed_labels:
                continue
            try:
                score = float(entity[score_field])
            except (TypeError, ValueError):
                continue
            score = min(max(score, 1e-6), 1 - 1e-6)
            entries.append(
                {
                    "row_idx": row_idx,
                    "ent_idx": ent_idx,
                    "label": label,
                    "score": score,
                }
            )
    return entries


def _build_pseudo_labels(scores, mode, positive_threshold, lower_quantile, upper_quantile):
    scores = np.asarray(scores, dtype=np.float64)
    if mode == "score-threshold":
        labels = (scores >= positive_threshold).astype(np.int64)
        mask = np.ones_like(labels, dtype=bool)
        return labels, mask

    if mode == "quantile-bands":
        low = np.percentile(scores, lower_quantile)
        high = np.percentile(scores, upper_quantile)
        mask = (scores <= low) | (scores >= high)
        labels = (scores >= high).astype(np.int64)
        return labels, mask

    raise ValueError(f"Unsupported label source: {mode}")


def _load_calibration_csv(csv_path, score_col, label_col, class_col, allowed_labels):
    scores = []
    labels = []
    classes = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if score_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing score column '{score_col}'.")
        if label_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing label column '{label_col}'.")
        use_class = bool(class_col)
        if use_class and class_col not in (reader.fieldnames or []):
            raise ValueError(f"Calibration CSV missing class column '{class_col}'.")

        for row in reader:
            try:
                score = float(row[score_col])
                label = int(float(row[label_col]))
            except (TypeError, ValueError):
                continue
            score = min(max(score, 1e-6), 1 - 1e-6)
            if label not in (0, 1):
                continue

            if use_class:
                class_name = str(row[class_col]).strip()
                if allowed_labels and class_name not in allowed_labels:
                    continue
                classes.append(class_name)
            else:
                classes.append("")

            scores.append(score)
            labels.append(label)

    if not scores:
        raise ValueError("Calibration CSV yielded zero valid rows.")

    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64), classes


def _assign_calibrated_scores(rows, entries, calibrated_scores, output_score_field, original_score_field, score_field):
    for entry, calibrated in zip(entries, calibrated_scores):
        entity = rows[entry["row_idx"]]["entities"][entry["ent_idx"]]
        if original_score_field and original_score_field not in entity:
            entity[original_score_field] = entity.get(score_field)
        entity[output_score_field] = float(calibrated)


def _fit_global_temperature(entries, pseudo_labels, fit_mask, config):
    scores = np.asarray([entry["score"] for entry in entries], dtype=np.float64)
    temperature, fit_loss = fit_temperature(
        scores=scores[fit_mask],
        labels=pseudo_labels[fit_mask],
        t_min=config.temperature_min,
        t_max=config.temperature_max,
        grid_size=config.temperature_grid_size,
    )
    calibrated = apply_temperature(scores, temperature)
    params = {"temperature": temperature, "fit_loss": fit_loss}
    return calibrated, params


def _fit_global_temperature_from_arrays(target_scores, fit_scores, fit_labels, config):
    temperature, fit_loss = fit_temperature(
        scores=fit_scores,
        labels=fit_labels,
        t_min=config.temperature_min,
        t_max=config.temperature_max,
        grid_size=config.temperature_grid_size,
    )
    calibrated = apply_temperature(target_scores, temperature)
    params = {"temperature": temperature, "fit_loss": fit_loss}
    return calibrated, params


def _fit_per_class_temperature(entries, pseudo_labels, fit_mask, config):
    scores = np.asarray([entry["score"] for entry in entries], dtype=np.float64)
    calibrated = np.copy(scores)
    class_to_indices = defaultdict(list)
    for idx, entry in enumerate(entries):
        class_to_indices[entry["label"]].append(idx)

    params = {"per_class_temperature": {}}
    for label, indices in class_to_indices.items():
        idx_array = np.asarray(indices, dtype=np.int64)
        fit_idx = idx_array[fit_mask[idx_array]]
        if len(fit_idx) == 0 or len(set(pseudo_labels[fit_idx].tolist())) < 2:
            params["per_class_temperature"][label] = None
            continue
        temperature, fit_loss = fit_temperature(
            scores=scores[fit_idx],
            labels=pseudo_labels[fit_idx],
            t_min=config.temperature_min,
            t_max=config.temperature_max,
            grid_size=config.temperature_grid_size,
        )
        calibrated[idx_array] = apply_temperature(scores[idx_array], temperature)
        params["per_class_temperature"][label] = {
            "temperature": float(temperature),
            "fit_loss": float(fit_loss),
            "support": int(len(fit_idx)),
        }

    return calibrated, params


def _fit_per_class_temperature_from_arrays(entries, target_scores, fit_scores, fit_labels, fit_classes, config):
    global_temperature = None
    if len(set(fit_labels.tolist())) >= 2:
        global_temperature, _ = fit_temperature(
            scores=fit_scores,
            labels=fit_labels,
            t_min=config.temperature_min,
            t_max=config.temperature_max,
            grid_size=config.temperature_grid_size,
        )

    calibrated = np.copy(target_scores)
    target_class_to_indices = defaultdict(list)
    for idx, entry in enumerate(entries):
        target_class_to_indices[entry["label"]].append(idx)

    fit_class_to_indices = defaultdict(list)
    for idx, class_name in enumerate(fit_classes):
        fit_class_to_indices[class_name].append(idx)

    params = {"per_class_temperature": {}, "global_fallback_temperature": global_temperature}
    for class_name, indices in target_class_to_indices.items():
        target_idx = np.asarray(indices, dtype=np.int64)
        fit_idx = np.asarray(fit_class_to_indices.get(class_name, []), dtype=np.int64)

        if len(fit_idx) > 0 and len(set(fit_labels[fit_idx].tolist())) >= 2:
            temperature, fit_loss = fit_temperature(
                scores=fit_scores[fit_idx],
                labels=fit_labels[fit_idx],
                t_min=config.temperature_min,
                t_max=config.temperature_max,
                grid_size=config.temperature_grid_size,
            )
            calibrated[target_idx] = apply_temperature(target_scores[target_idx], temperature)
            params["per_class_temperature"][class_name] = {
                "temperature": float(temperature),
                "fit_loss": float(fit_loss),
                "support": int(len(fit_idx)),
                "source": "per-class",
            }
            continue

        if global_temperature is not None:
            calibrated[target_idx] = apply_temperature(target_scores[target_idx], global_temperature)
            params["per_class_temperature"][class_name] = {
                "temperature": float(global_temperature),
                "fit_loss": None,
                "support": int(len(fit_idx)),
                "source": "global-fallback",
            }
        else:
            params["per_class_temperature"][class_name] = {
                "temperature": None,
                "fit_loss": None,
                "support": int(len(fit_idx)),
                "source": "identity",
            }

    return calibrated, params


def _fit_global_isotonic(entries, pseudo_labels, fit_mask):
    scores = np.asarray([entry["score"] for entry in entries], dtype=np.float64)
    model = fit_isotonic(scores[fit_mask], pseudo_labels[fit_mask])
    calibrated = apply_isotonic(model, scores)
    params = {
        "isotonic_x_thresholds": [float(x) for x in model.X_thresholds_],
        "isotonic_y_thresholds": [float(y) for y in model.y_thresholds_],
    }
    return calibrated, params


def run_calibration(config, script_path):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()

    script_dir = Path(script_path).resolve().parent
    input_jsonl = resolve_path(script_dir, config.input_jsonl)
    output_jsonl = resolve_path(script_dir, config.output_jsonl)
    stats_json = resolve_path(script_dir, config.stats_json)
    calibration_csv = resolve_path(script_dir, config.calibration_csv)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    rows = load_jsonl(str(input_jsonl))
    entries = _extract_entities(
        rows=rows,
        score_field=config.score_field,
        label_field=config.label_field,
        allowed_labels=set(config.labels),
    )
    if not entries:
        raise ValueError("No valid entities with scores were found for calibration.")

    scores = np.asarray([entry["score"] for entry in entries], dtype=np.float64)
    if config.label_source == "calibration-csv":
        if not calibration_csv.exists():
            raise FileNotFoundError(f"Calibration CSV not found: {calibration_csv}")
        fit_scores, fit_labels, fit_classes = _load_calibration_csv(
            csv_path=calibration_csv,
            score_col=config.csv_score_col,
            label_col=config.csv_label_col,
            class_col=config.csv_class_col,
            allowed_labels=set(config.labels),
        )
        fit_mask = np.ones_like(fit_labels, dtype=bool)
        fit_targets = fit_labels
    else:
        fit_targets, fit_mask = _build_pseudo_labels(
            scores=scores,
            mode=config.label_source,
            positive_threshold=config.positive_threshold,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
        )
        fit_scores = scores
        fit_labels = fit_targets
        fit_classes = [entry["label"] for entry in entries]

    if config.method == "temperature":
        if config.label_source == "calibration-csv":
            calibrated_scores, params = _fit_global_temperature_from_arrays(
                target_scores=scores,
                fit_scores=fit_scores,
                fit_labels=fit_labels,
                config=config,
            )
        else:
            calibrated_scores, params = _fit_global_temperature(entries, fit_targets, fit_mask, config)
    elif config.method == "temperature-per-class":
        if config.label_source == "calibration-csv":
            calibrated_scores, params = _fit_per_class_temperature_from_arrays(
                entries=entries,
                target_scores=scores,
                fit_scores=fit_scores,
                fit_labels=fit_labels,
                fit_classes=fit_classes,
                config=config,
            )
        else:
            calibrated_scores, params = _fit_per_class_temperature(entries, fit_targets, fit_mask, config)
    elif config.method == "isotonic":
        if config.label_source == "calibration-csv":
            model = fit_isotonic(fit_scores, fit_labels)
            calibrated_scores = apply_isotonic(model, scores)
            params = {
                "isotonic_x_thresholds": [float(x) for x in model.X_thresholds_],
                "isotonic_y_thresholds": [float(y) for y in model.y_thresholds_],
            }
        else:
            calibrated_scores, params = _fit_global_isotonic(entries, fit_targets, fit_mask)
    else:
        raise ValueError(f"Unsupported method: {config.method}")

    _assign_calibrated_scores(
        rows=rows,
        entries=entries,
        calibrated_scores=calibrated_scores,
        output_score_field=config.output_score_field,
        original_score_field=config.preserve_original_score_field,
        score_field=config.score_field,
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(str(output_jsonl), rows)

    label_counts = Counter(entry["label"] for entry in entries)
    brier_before = brier_score_loss(fit_labels[fit_mask], fit_scores[fit_mask])
    if config.method == "temperature":
        fit_calibrated = apply_temperature(fit_scores[fit_mask], params["temperature"])
    elif config.method == "temperature-per-class":
        fit_calibrated = np.copy(fit_scores[fit_mask])
        fit_labels_classes = np.asarray(fit_classes, dtype=object)[fit_mask]
        for idx, class_name in enumerate(fit_labels_classes):
            class_params = params["per_class_temperature"].get(class_name)
            if not class_params or class_params["temperature"] is None:
                continue
            fit_calibrated[idx] = apply_temperature(
                np.asarray([fit_scores[fit_mask][idx]], dtype=np.float64),
                class_params["temperature"],
            )[0]
    else:
        if config.label_source == "calibration-csv":
            iso_x = np.asarray(params["isotonic_x_thresholds"], dtype=np.float64)
            iso_y = np.asarray(params["isotonic_y_thresholds"], dtype=np.float64)
            fit_calibrated = np.interp(fit_scores[fit_mask], iso_x, iso_y)
        else:
            fit_calibrated = apply_isotonic(fit_isotonic(fit_scores[fit_mask], fit_labels[fit_mask]), fit_scores[fit_mask])
    brier_after = brier_score_loss(fit_labels[fit_mask], fit_calibrated)

    runtime_seconds = perf_counter() - timer
    finished_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": asdict(config),
        "summary": {
            "total_rows": len(rows),
            "total_entities": len(entries),
            "fit_entities": int(np.sum(fit_mask)),
            "entities_by_label": dict(label_counts),
            "fit_positive_rate": float(np.mean(fit_labels[fit_mask])),
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
        },
        "method_parameters": params,
    }
    with open(stats_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Calibration method: %s", config.method)
    LOGGER.info("Entities calibrated: %s (fit subset: %s)", len(entries), int(np.sum(fit_mask)))
    LOGGER.info("Brier score before: %.6f | after: %.6f", brier_before, brier_after)
    LOGGER.info("Saved calibrated JSONL to: %s", output_jsonl)
    LOGGER.info("Saved calibration stats to: %s", stats_json)
